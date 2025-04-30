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

# === Environment setup ===
export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

conda activate qtype-eval
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

# === Directory setup ===
OUTPUT_BASE_DIR="$VSC_SCRATCH/probing_output"
mkdir -p $OUTPUT_BASE_DIR
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/results_tracker.csv"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
mkdir -p $LOG_DIR

# Initialize results tracker
echo "experiment_type,language,layer,task,submetric,control_index,metric,value" > $RESULTS_TRACKER

# Also track failed experiments
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch $FAILED_LOG

# Create metrics extraction script
cat > "${OUTPUT_BASE_DIR}/extract_metrics.py" << 'EOF'
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
        
        # Append to tracker file
        with open(tracker_file, 'a') as f:
            writer = csv.writer(f)
            for metric, value in test_metrics.items():
                if value is not None:
                    writer.writerow([
                        exp_type, language, layer, task, 
                        submetric if submetric != "None" else "None", 
                        control_index if control_index != "None" else "None",
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

# Create results analysis script
cat > "${OUTPUT_BASE_DIR}/generate_summaries.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import os

def generate_summaries(tracker_file, output_dir):
    print(f"Reading tracker file: {tracker_file}")
    df = pd.read_csv(tracker_file)
    
    # Replace empty strings with NaN for proper handling
    df = df.replace('', np.nan)
    df = df.replace('None', np.nan)
    
    # Make layer values numeric for proper sorting
    if 'layer' in df.columns:
        df['layer'] = pd.to_numeric(df['layer'], errors='coerce')
    
    print(f"Generating summaries in: {output_dir}")
    
    # 1. Layer Summary - average metrics by layer and task
    layer_summary = df.pivot_table(
        index=['layer', 'task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    layer_summary.to_csv(os.path.join(output_dir, 'layer_summary.csv'))
    
    # 2. Language Summary - average metrics by language and task
    language_summary = df.pivot_table(
        index=['language', 'task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    language_summary.to_csv(os.path.join(output_dir, 'language_summary.csv'))
    
    # 3. Submetric Summary - only for submetric tasks
    submetric_df = df[df['submetric'].notna()]
    if not submetric_df.empty:
        submetric_summary = submetric_df.pivot_table(
            index=['submetric', 'layer'], 
            columns='metric', 
            values='value',
            aggfunc='mean'
        )
        submetric_summary.to_csv(os.path.join(output_dir, 'submetric_summary.csv'))
    
    # 4. Control Summary - compare control vs. non-control
    control_summary = df.pivot_table(
        index=['task', 'control_index'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    control_summary.to_csv(os.path.join(output_dir, 'control_summary.csv'))
    
    # 5. Complete matrix for all experiments
    # Reshape the data to have one row per unique experiment
    experiment_matrix = df.pivot_table(
        index=['experiment_type', 'language', 'layer', 'task', 'submetric', 'control_index'],
        columns='metric',
        values='value'
    )
    experiment_matrix.to_csv(os.path.join(output_dir, 'experiment_matrix.csv'))
    
    # 6. Task-specific summaries
    for task in df['task'].dropna().unique():
        task_df = df[df['task'] == task]
        task_summary = task_df.pivot_table(
            index=['language', 'layer'], 
            columns='metric', 
            values='value',
            aggfunc='mean'
        )
        task_summary.to_csv(os.path.join(output_dir, f'{task}_summary.csv'))
    
    print("All summary files generated successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_summaries.py <tracker_file> <output_dir>")
        sys.exit(1)
    
    tracker_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    generate_summaries(tracker_file, output_dir)
EOF

chmod +x "${OUTPUT_BASE_DIR}/extract_metrics.py"
chmod +x "${OUTPUT_BASE_DIR}/generate_summaries.py"

# === Configuration ===
LANGUAGES=("ru")
# For full run uncommment this: 
# LANGUAGES=("en" "ar" "fi" "id" "ja" "ko" "ru")

# Use a small set of layers for testing
LAYERS=(7)
# For full run uncommment this:
# LAYERS=(1 2 3 4 5 6 7 8 9 10 11 12)

MAIN_TASKS=("question_type" "complexity")
SUBMETRICS=("avg_verb_edges")
# For full run uncommment this:
# SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")

CONTROL_INDICES=(1)
# For full run uncommment this:
# CONTROL_INDICES=(1 2 3)



# === Experiment functions ===
run_standard_experiment() {
    local LANGUAGE=$1
    local LAYER=$2
    local TASK=$3
    local TASK_TYPE=$4
    local SUBMETRIC=$5
    local MAX_RETRIES=2
    
    local EXPERIMENT_TYPE="standard"
    local TASK_SPEC="question_type"
    local EXPERIMENT_NAME="layer_${LAYER}_question_type_${LANGUAGE}"
    local OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/question_type"
    
    if [ "$TASK" == "complexity" ]; then
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
    
    if [ -f "${OUTPUT_SUBDIR}/results.json" ]; then
        echo "Experiment ${EXPERIMENT_NAME} already completed successfully. Extracting metrics..."
        python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
            "${OUTPUT_SUBDIR}/results.json" "$RESULTS_TRACKER" \
            "$EXPERIMENT_TYPE" "$LANGUAGE" "$LAYER" \
            "$TASK_SPEC" "${SUBMETRIC:-None}" "None"
        return 0
    fi
    
    # Skip if already failed too many times
    local FAIL_COUNT=$(grep -c "^${EXPERIMENT_NAME}$" $FAILED_LOG)
    if [ $FAIL_COUNT -ge $MAX_RETRIES ]; then
        echo "Experiment ${EXPERIMENT_NAME} has failed $FAIL_COUNT times already. Skipping."
        return 1
    fi

    echo "==============================================================="
    echo "Running $TASK_SPEC experiment for language $LANGUAGE, layer $LAYER"
    if [ -n "$SUBMETRIC" ]; then
        echo "Submetric: $SUBMETRIC"
    fi
    echo "==============================================================="
    
    # Build command with all necessary parameters
    local COMMAND=""
    
    if [ -n "$SUBMETRIC" ]; then
        # For submetric tasks, use the submetrics experiment config
        COMMAND="python -m src.experiments.run_experiment \
            \"hydra.job.chdir=False\" \
            \"hydra.run.dir=.\" \
            \"experiment=single_submetric\" \
            \"experiment.tasks=single_submetric\" \
            \"experiment.submetric=${SUBMETRIC}\" \
            \"model=lm_probe\" \
            \"model.model_type=lm_probe\" \
            \"model.lm_name=cis-lmu/glot500-base\" \
            \"model.layer_wise=true\" \
            \"model.layer_index=${LAYER}\" \
            \"model.freeze_model=true\" \
            \"model.probe_hidden_size=192\" \
            \"data.languages=[${LANGUAGE}]\" \
            \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
            \"training.task_type=${TASK_TYPE}\" \
            \"training.num_epochs=15\" \
            \"training.lr=3e-5\" \
            \"training.batch_size=16\" \
            \"+training.gradient_accumulation_steps=2\" \
            \"experiment_name=${EXPERIMENT_NAME}\" \
            \"output_dir=${OUTPUT_SUBDIR}\" \
            \"wandb.mode=offline\""
    else
        # For main tasks, use the task-specific experiment config
        COMMAND="python -m src.experiments.run_experiment \
            \"hydra.job.chdir=False\" \
            \"hydra.run.dir=.\" \
            \"experiment=${TASK}\" \
            \"experiment.tasks=${TASK}\" \
            \"model=lm_probe\" \
            \"model.model_type=lm_probe\" \
            \"model.lm_name=cis-lmu/glot500-base\" \
            \"model.layer_wise=true\" \
            \"model.layer_index=${LAYER}\" \
            \"model.freeze_model=true\" \
            \"model.probe_hidden_size=192\" \
            \"data.languages=[${LANGUAGE}]\" \
            \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
            \"training.task_type=${TASK_TYPE}\" \
            \"training.num_epochs=15\" \
            \"training.lr=3e-5\" \
            \"training.batch_size=16\" \
            \"+training.gradient_accumulation_steps=2\" \
            \"experiment_name=${EXPERIMENT_NAME}\" \
            \"output_dir=${OUTPUT_SUBDIR}\" \
            \"wandb.mode=offline\""
    fi
    
    echo "Command: $COMMAND"
    
    eval $COMMAND 

    local RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "==============================================================="
        echo "Standard experiment completed successfully: $EXPERIMENT_NAME"
        echo "==============================================================="
        
        # Extract metrics
        RESULTS_FILE="${OUTPUT_SUBDIR}/${LANGUAGE}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" \
                "$EXPERIMENT_TYPE" "$LANGUAGE" "$LAYER" \
                "$TASK_SPEC" "${SUBMETRIC:-None}" "None"
            
            return 0
        else
            echo "WARNING: Results file not found: $RESULTS_FILE" 
            echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
            return 1
        fi
    else
        echo "ERROR in standard experiment: $EXPERIMENT_NAME" 
        echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 1
    fi
}

run_control_experiment() {
    local LANGUAGE=$1
    local LAYER=$2
    local TASK=$3
    local TASK_TYPE=$4
    local CONTROL_IDX=$5
    local SUBMETRIC=$6
    local MAX_RETRIES=2
    
    local EXPERIMENT_TYPE="control"
    local TASK_SPEC="question_type"
    local EXPERIMENT_NAME="layer_${LAYER}_question_type_control${CONTROL_IDX}_${LANGUAGE}"
    local OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/question_type/control${CONTROL_IDX}/${LANGUAGE}"
    
    if [ "$TASK" == "complexity" ]; then
        TASK_SPEC="complexity"
        EXPERIMENT_NAME="layer_${LAYER}_complexity_control${CONTROL_IDX}_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/complexity/control${CONTROL_IDX}/${LANGUAGE}"
    
    elif [ -n "$SUBMETRIC" ]; then
        EXPERIMENT_TYPE="submetrics_control"
        TASK_SPEC="single_submetric"
        EXPERIMENT_NAME="layer_${LAYER}_${SUBMETRIC}_control${CONTROL_IDX}_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/${SUBMETRIC}/control${CONTROL_IDX}/${LANGUAGE}"
    fi
    
    mkdir -p "$OUTPUT_SUBDIR"
    
    # Skip if already successful
    if [ -f "${OUTPUT_SUBDIR}/results.json" ]; then
        echo "Control experiment ${EXPERIMENT_NAME} already completed successfully. Extracting metrics..."
        python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
            "${OUTPUT_SUBDIR}/results.json" "$RESULTS_TRACKER" \
            "$EXPERIMENT_TYPE" "$LANGUAGE" "$LAYER" \
            "$TASK_SPEC" "${SUBMETRIC:-None}" "$CONTROL_IDX"
        return 0
    fi
    
    # Skip if already failed too many times
    local FAIL_COUNT=$(grep -c "^${EXPERIMENT_NAME}$" $FAILED_LOG)
    if [ $FAIL_COUNT -ge $MAX_RETRIES ]; then
        echo "Control experiment ${EXPERIMENT_NAME} has failed $FAIL_COUNT times already. Skipping."
        return 1
    fi
    
    echo "==============================================================="
    echo "Running $TASK_SPEC control experiment for language $LANGUAGE, layer $LAYER, control $CONTROL_IDX"
    if [ -n "$SUBMETRIC" ]; then
        echo "Submetric: $SUBMETRIC"
    fi
    echo "==============================================================="
    
    # Build command
    local COMMAND=""
    
    if [ -n "$SUBMETRIC" ]; then
        # For submetric tasks with control
        COMMAND="python -m src.experiments.run_experiment \
            \"hydra.job.chdir=False\" \
            \"hydra.run.dir=.\" \
            \"experiment=single_submetric\" \
            \"experiment.tasks=single_submetric\" \
            \"experiment.submetric=${SUBMETRIC}\" \
            \"experiment.use_controls=true\" \
            \"experiment.control_index=${CONTROL_IDX}\" \
            \"model=lm_probe\" \
            \"model.model_type=lm_probe\" \
            \"model.lm_name=cis-lmu/glot500-base\" \
            \"model.layer_wise=true\" \
            \"model.layer_index=${LAYER}\" \
            \"model.freeze_model=true\" \
            \"model.probe_hidden_size=192\" \
            \"data.languages=[${LANGUAGE}]\" \
            \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
            \"training.task_type=${TASK_TYPE}\" \
            \"training.num_epochs=15\" \
            \"training.lr=3e-5\" \
            \"training.batch_size=16\" \
            \"+training.gradient_accumulation_steps=2\" \
            \"experiment_name=${EXPERIMENT_NAME}\" \
            \"output_dir=${OUTPUT_SUBDIR}\" \
            \"wandb.mode=offline\""
    else
        # For main tasks with control
        COMMAND="python -m src.experiments.run_experiment \
            \"hydra.job.chdir=False\" \
            \"hydra.run.dir=.\" \
            \"experiment=${TASK}\" \
            \"experiment.tasks=${TASK}\" \
            \"experiment.use_controls=true\" \
            \"experiment.control_index=${CONTROL_IDX}\" \
            \"model=lm_probe\" \
            \"model.model_type=lm_probe\" \
            \"model.lm_name=cis-lmu/glot500-base\" \
            \"model.layer_wise=true\" \
            \"model.layer_index=${LAYER}\" \
            \"model.freeze_model=true\" \
            \"model.probe_hidden_size=192\" \
            \"data.languages=[${LANGUAGE}]\" \
            \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
            \"training.task_type=${TASK_TYPE}\" \
            \"training.num_epochs=15\" \
            \"training.lr=3e-5\" \
            \"training.batch_size=16\" \
            \"+training.gradient_accumulation_steps=2\" \
            \"experiment_name=${EXPERIMENT_NAME}\" \
            \"output_dir=${OUTPUT_SUBDIR}\" \
            \"wandb.mode=offline\""
    fi
    
    echo "Command: $COMMAND"
    
    eval $COMMAND 
    local RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "==============================================================="
        echo "Control experiment completed successfully: $EXPERIMENT_NAME"
        echo "==============================================================="
        
        # Extract metrics
        RESULTS_FILE="${OUTPUT_SUBDIR}/${LANGUAGE}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" \
                "$EXPERIMENT_TYPE" "$LANGUAGE" "$LAYER" \
                "$TASK_SPEC" "${SUBMETRIC:-None}" "$CONTROL_IDX"
            
            return 0
        else
            echo "WARNING: Results file not found: $RESULTS_FILE" 
            echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
            return 1
        fi
    else
        echo "ERROR in control experiment: $EXPERIMENT_NAME" 
        echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 1
    fi
}

# 1. Main tasks (question_type and complexity) for all languages and layers
echo "========= Running main tasks experiments =========="
for LANGUAGE in "${LANGUAGES[@]}"; do
    echo "Processing language: ${LANGUAGE}"
    
    for LAYER in "${LAYERS[@]}"; do
        
        echo "Processing layer: ${LAYER}"
        
        for TASK in "${MAIN_TASKS[@]}"; do
            TASK_TYPE="classification"
            if [ "$TASK" == "complexity" ]; then
                TASK_TYPE="regression"
            fi
            
            # Run standard experiment for this task
            run_standard_experiment "$LANGUAGE" "$LAYER" "$TASK" "$TASK_TYPE" ""
            
            # Wait a bit between experiments
            sleep 5
        done
    done
    
    # Clear GPU memory after each language
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 15
done

# 2. Submetric tasks for all languages and layers
echo "========= Running submetric tasks experiments =========="
for LANGUAGE in "${LANGUAGES[@]}"; do
    echo "Processing language: ${LANGUAGE}"
    
    for LAYER in "${LAYERS[@]}"; do
        echo "Processing layer: ${LAYER}"
        
        for SUBMETRIC in "${SUBMETRICS[@]}"; do
            
            # Run standard experiment for this submetric
            run_standard_experiment "$LANGUAGE" "$LAYER" "" "regression" "$SUBMETRIC"
            
            # Wait a bit between experiments
            sleep 5
        done
    done
    
    # Clear GPU memory after each language
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 15
done

# 3. Control experiments for main tasks
echo "========= Running control experiments for main tasks =========="
for LANGUAGE in "${LANGUAGES[@]}"; do
    echo "Processing language: ${LANGUAGE}"
    
    for LAYER in "${LAYERS[@]}"; do
        echo "Processing layer: ${LAYER}"
        
        for TASK in "${MAIN_TASKS[@]}"; do
            TASK_TYPE="classification"
            if [ "$TASK" == "complexity" ]; then
                TASK_TYPE="regression"
            fi
            
            for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
                
                # Run control experiment
                run_control_experiment "$LANGUAGE" "$LAYER" "$TASK" "$TASK_TYPE" "$CONTROL_IDX" ""
                
                # Wait a bit between experiments
                sleep 5
            done
        done
    done
    
    # Clear GPU memory after each language
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 15
done

# 4. Control experiments for submetrics
echo "========= Running control experiments for submetrics =========="
for LANGUAGE in "${LANGUAGES[@]}"; do
    echo "Processing language: ${LANGUAGE}"
    
    for LAYER in "${LAYERS[@]}"; do
        echo "Processing layer: ${LAYER}"
        
        for SUBMETRIC in "${SUBMETRICS[@]}"; do
            for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
                # Run control experiment for this submetric
                run_control_experiment "$LANGUAGE" "$LAYER" "" "regression" "$CONTROL_IDX" "$SUBMETRIC"
                
                # Wait a bit between experiments
                sleep 5
            done
        done
    done
    
    # Clear GPU memory after each language
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 15
done

# === Generate summary reports ===
echo "============= Generating summary reports ==============="
python3 ${OUTPUT_BASE_DIR}/generate_summaries.py "${RESULTS_TRACKER}" "${OUTPUT_BASE_DIR}"

echo "Layer-wise probing experiments completed. Results are saved in ${OUTPUT_BASE_DIR}"
echo "Summary files are available in ${OUTPUT_BASE_DIR}"
echo "Check ${OUTPUT_BASE_DIR}/figures for visualizations"

# Print some statistics
TOTAL_EXPERIMENTS=$(wc -l < "${RESULTS_TRACKER}")
FAILED_EXPERIMENTS=$(wc -l < "${FAILED_LOG}")
echo "Total experiments recorded: $((TOTAL_EXPERIMENTS-1)) (excluding header)"
echo "Failed experiments: ${FAILED_EXPERIMENTS}"
echo "Success rate: $(( (TOTAL_EXPERIMENTS-1-FAILED_EXPERIMENTS)*100/(TOTAL_EXPERIMENTS-1) ))%"