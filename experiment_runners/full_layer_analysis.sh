#!/bin/bash
#SBATCH --job-name=layerwise_probing
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_debug
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

conda activate qtype-eval
#env variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "GPU information:"
nvidia-smi

echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

LANGUAGES=("ar" "en")
LAYERS=(2 6 11 12)  
MAIN_TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "n_tokens")         # "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density"
CONTROL_INDICES=(1)

# Base output directory
OUTPUT_BASE_DIR="$VSC_SCRATCH/layerwise_output"
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/results_tracker.csv"
mkdir -p $OUTPUT_BASE_DIR

# Initialize results tracker
echo "experiment_type,language,layer,task,submetric,control_index,metric,value" > $RESULTS_TRACKER

# Metrics extraction script
cat > ${OUTPUT_BASE_DIR}/extract_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os

def extract_metrics(result_file, tracker_file, exp_type, language, layer, task, submetric, control_index):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract test metrics
        test_metrics = data.get('test_metrics', {})
        
        # Validate metrics
        if not test_metrics:
            print(f"Warning: No test metrics found in {result_file}")
            return False
            
        if task == "question_type" and "accuracy" not in test_metrics:
            print(f"Warning: No accuracy metric found for classification task in {result_file}")
        
        if task in ["complexity", "single_submetric"] and "r2" not in test_metrics:
            print(f"Warning: No r2 metric found for regression task in {result_file}")
        
        # Append to tracker file
        with open(tracker_file, 'a') as f:
            writer = csv.writer(f)
            for metric, value in test_metrics.items():
                if value is not None:
                    writer.writerow([
                        exp_type, language, layer, task, 
                        submetric if submetric else 'None', 
                        control_index if control_index != 'None' else 'None',
                        metric, value
                    ])
        return True
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: extract_metrics.py <result_file> <tracker_file> <exp_type> <language> <layer> <task> <submetric> <control_index>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    tracker_file = sys.argv[2]
    exp_type = sys.argv[3]
    language = sys.argv[4]
    layer = sys.argv[5]
    task = sys.argv[6]
    submetric = sys.argv[7]
    control_index = sys.argv[8]
    
    if extract_metrics(result_file, tracker_file, exp_type, language, layer, task, submetric, control_index):
        print(f"Successfully extracted metrics from {result_file}")
    else:
        print(f"Failed to extract metrics from {result_file}")
EOF

chmod +x ${OUTPUT_BASE_DIR}/extract_metrics.py

# Standardized experiment runner
run_standard_experiment() {
    local LANGUAGE=$1
    local LAYER=$2
    local TASK=$3
    local TASK_TYPE=$4
    local SUBMETRIC=$5
    
    local EXPERIMENT_TYPE="question_type"
    local TASK_SPEC="question_type"
    local EXPERIMENT_NAME="layer_${LAYER}_question_type_${LANGUAGE}"
    local OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/question_type"
    
    if [ "$TASK" == "complexity" ]; then
        EXPERIMENT_TYPE="complexity"
        TASK_SPEC="complexity"
        EXPERIMENT_NAME="layer_${LAYER}_complexity_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/complexity"
    
    elif [ -n "$SUBMETRIC" ]; then
        EXPERIMENT_TYPE="submetrics"
        TASK_SPEC="single_submetric"
        EXPERIMENT_NAME="layer_${LAYER}_${SUBMETRIC}_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/${SUBMETRIC}"
    fi
    
    mkdir -p "$OUTPUT_SUBDIR"
    
    echo "Running $TASK_SPEC experiment for language $LANGUAGE, layer $LAYER"
    
    local COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${EXPERIMENT_TYPE}\" \
        \"experiment.tasks=${TASK_SPEC}\" \
        \"model=lm_probe\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.layer_wise=true\" \
        \"model.layer_index=${LAYER}\" \
        \"model.freeze_model=true\" \
        \"+model.probe_hidden_size=96\" \
        \"data.languages=[${LANGUAGE}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=15\" \
        \"training.lr=1e-4\" \
        \"training.batch_size=16\" \
        \"+training.gradient_accumulation_steps=2\" \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${OUTPUT_SUBDIR}\" \
        \"wandb.mode=offline\""
    
    if [ -n "$SUBMETRIC" ]; then
        COMMAND+=" \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    eval $COMMAND
    
    local RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Standard experiment completed successfully: $EXPERIMENT_NAME"
        
        # Extract metrics
        RESULTS_FILE="${OUTPUT_SUBDIR}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "standard" "$LANGUAGE" "$LAYER" \
                "$TASK_SPEC" "${SUBMETRIC:-None}" "None"
        fi
        
        return 0
    else
        echo "Error in standard experiment: $EXPERIMENT_NAME"
        return 1
    fi
}

# Control experiment runner
run_control_experiment() {
    local LANGUAGE=$1
    local LAYER=$2
    local TASK=$3
    local TASK_TYPE=$4
    local CONTROL_IDX=$5
    local SUBMETRIC=$6
    
    local EXPERIMENT_TYPE="question_type"
    local TASK_SPEC="question_type"
    local EXPERIMENT_NAME="layer_${LAYER}_question_type_control${CONTROL_IDX}_${LANGUAGE}"
    local OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/question_type/control${CONTROL_IDX}"
    
    if [ "$TASK" == "complexity" ]; then
        EXPERIMENT_TYPE="complexity"
        TASK_SPEC="complexity"
        EXPERIMENT_NAME="layer_${LAYER}_complexity_control${CONTROL_IDX}_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/complexity/control${CONTROL_IDX}"
    
    elif [ -n "$SUBMETRIC" ]; then
        EXPERIMENT_TYPE="submetrics"
        TASK_SPEC="single_submetric"
        EXPERIMENT_NAME="layer_${LAYER}_${SUBMETRIC}_control${CONTROL_IDX}_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/${SUBMETRIC}/control${CONTROL_IDX}"
    fi
    
    mkdir -p "$OUTPUT_SUBDIR"
    
    echo "Running $TASK_SPEC control experiment for language $LANGUAGE, layer $LAYER, control $CONTROL_IDX"
    
    local COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${EXPERIMENT_TYPE}\" \
        \"experiment.tasks=${TASK_SPEC}\" \
        \"model=lm_probe\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.layer_wise=true\" \
        \"model.layer_index=${LAYER}\" \
        \"model.freeze_model=true\" \
        \"+model.probe_hidden_size=96\" \
        \"experiment.use_controls=true\" \
        \"experiment.control_index=${CONTROL_IDX}\" \
        \"data.languages=[${LANGUAGE}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=15\" \
        \"training.lr=1e-4\" \
        \"training.batch_size=16\" \
        \"+training.gradient_accumulation_steps=2\" \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${OUTPUT_SUBDIR}\" \
        \"wandb.mode=offline\""
    
    if [ -n "$SUBMETRIC" ]; then
        COMMAND+=" \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    eval $COMMAND
    
    local RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Control experiment completed successfully: $EXPERIMENT_NAME"
        
        # Extract metrics
        RESULTS_FILE="${OUTPUT_SUBDIR}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "control" "$LANGUAGE" "$LAYER" \
                "$TASK_SPEC" "${SUBMETRIC:-None}" "$CONTROL_IDX"
        fi
        
        return 0
    else
        echo "Error in control experiment: $EXPERIMENT_NAME"
        return 1
    fi
}

# Experiment execution
for LANGUAGE in "${LANGUAGES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for TASK in "${MAIN_TASKS[@]}"; do
            TASK_TYPE="classification"
            if [ "$TASK" == "complexity" ]; then
                TASK_TYPE="regression"
            fi

            # Standard experiments
            run_standard_experiment "$LANGUAGE" "$LAYER" "$TASK" "$TASK_TYPE" ""

            # Submetric experiments for complexity
            if [ "$TASK" == "complexity" ]; then
                for SUBMETRIC in "${SUBMETRICS[@]}"; do
                    run_standard_experiment "$LANGUAGE" "$LAYER" "$TASK" "$TASK_TYPE" "$SUBMETRIC"
                done
            fi

            # Control experiments
            for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
                run_control_experiment "$LANGUAGE" "$LAYER" "$TASK" "$TASK_TYPE" "$CONTROL_IDX" ""

                # Control submetric experiments for complexity
                if [ "$TASK" == "complexity" ]; then
                    for SUBMETRIC in "${SUBMETRICS[@]}"; do
                        run_control_experiment "$LANGUAGE" "$LAYER" "$TASK" "$TASK_TYPE" "$CONTROL_IDX" "$SUBMETRIC"
                    done
                fi
            done
        done
    done
done

# Generate basic summary
python3 -c "
import pandas as pd
import numpy as np

df = pd.read_csv('$RESULTS_TRACKER')

# Summary statistics
summary = df.groupby(['task', 'layer', 'metric']).agg({
    'value': ['mean', 'std', 'count']
}).reset_index()

summary.columns = ['task', 'layer', 'metric', 'mean', 'std', 'count']
summary.to_csv('${OUTPUT_BASE_DIR}/layer_summary.csv', index=False)

# Experiment counts
exp_counts = df.groupby(['task', 'layer', 'experiment_type']).size().reset_index(name='count')
exp_counts.to_csv('${OUTPUT_BASE_DIR}/experiment_counts.csv', index=False)

print('Layer analysis summary generated.')
"

echo "Layer-wise probing experiments completed successfully!"