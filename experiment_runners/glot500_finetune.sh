#!/bin/bash
#SBATCH --job-name=finetune_with_probe_parameters
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16     
#SBATCH --gpus-per-node=1     
#SBATCH --mem-per-cpu=11700M  
#SBATCH --time=12:00:00


export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

conda activate qtype-eval
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=/data/leuven/371/vsc37132/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
TASKS=("complexity" "question_type")    
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
CONTROL_INDICES=(1 2 3)

OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_output"
mkdir -p $OUTPUT_BASE_DIR

RESULTS_TRACKER="${OUTPUT_BASE_DIR}/finetune_results.csv"
echo "experiment_type,language,task,submetric,control_index,metric,value" > $RESULTS_TRACKER

# Failed experiments tracker
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch $FAILED_LOG

# creates a python script in the results folder that gets called after each run and logs results into a csv file
#outputs get printed in slurm*.out
cat > ${OUTPUT_BASE_DIR}/extract_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os

def extract_metrics(result_file, tracker_file, exp_type, language, task, submetric, control_index):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        test_metrics = data.get('test_metrics', {})
        
        if not test_metrics:
            print(f"Warning: No test metrics found in {result_file}")
            return False
            
        if task == "question_type" and "accuracy" not in test_metrics:
            print(f"Warning: No accuracy metric found for classification task in {result_file}")
        
        if task in ["complexity", "single_submetric"] and "r2" not in test_metrics:
            print(f"Warning: No r2 metric found for regression task in {result_file}")
        
        with open(tracker_file, 'a') as f:
            writer = csv.writer(f)
            for metric, value in test_metrics.items():
                if value is not None:
                    writer.writerow([
                        exp_type, language, task, 
                        submetric if submetric else 'None', 
                        control_index if control_index != 'None' else 'None',
                        metric, value
                    ])
        return True
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: extract_metrics.py <result_file> <tracker_file> <exp_type> <language> <task> <submetric> <control_index>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    tracker_file = sys.argv[2]
    exp_type = sys.argv[3]
    language = sys.argv[4]
    task = sys.argv[5]
    submetric = sys.argv[6]
    control_index = sys.argv[7]
    
    if extract_metrics(result_file, tracker_file, exp_type, language, task, submetric, control_index):
        print(f"Successfully extracted metrics from {result_file}")
    else:
        print(f"Failed to extract metrics from {result_file}")
EOF

chmod +x ${OUTPUT_BASE_DIR}/extract_metrics.py

run_finetune_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    local OUTPUT_SUBDIR=$6
    local MAX_RETRIES=2
    
    local EXPERIMENT_NAME=""
    local COMMAND=""
    local EXPERIMENT_TYPE="finetune"
    
    if [ -n "$SUBMETRIC" ]; then
        TASK="single_submetric"
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_${LANG}"
        else
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_control${CONTROL_IDX}_${LANG}"
        fi
    else
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${TASK}_${LANG}"
        else
            EXPERIMENT_NAME="finetune_${TASK}_control${CONTROL_IDX}_${LANG}"
        fi
    fi
    
    if [ -f "${OUTPUT_SUBDIR}/${LANG}/results.json" ]; then
        echo "Experiment ${EXPERIMENT_NAME} already completed.  metrics..."
        CONTROL_PARAM=${CONTROL_IDX:-None}
        python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
            "${OUTPUT_SUBDIR}/${LANG}/results.json" "$RESULTS_TRACKER" "finetune" "$LANG" "$TASK" \
            "${SUBMETRIC:-}" "$CONTROL_PARAM"
        return 0
    fi
    
    local FAIL_COUNT=$(grep -c "^${EXPERIMENT_NAME}$" $FAILED_LOG)
    if [ $FAIL_COUNT -ge $MAX_RETRIES ]; then
        echo "tried running ${EXPERIMENT_NAME}, failed $FAIL_COUNT times . skip."
        return 1
    fi
    
    local DEBUG_PARAM=""
    if [ "$LANG" == "en" ] && [ -z "$CONTROL_IDX" ]; then
        DEBUG_PARAM="+training.debug_mode=true"
    fi
    
    local FINETUNE_CONFIG=""
    
    if [ "$TASK_TYPE" == "classification" ]; then               # manual adjustments of configs
        FINETUNE_CONFIG="\
            \"model.head_hidden_size=786\" \
            \"model.head_layers=2\" \
            \"model.dropout=0.05\""
            
        TRAINING_CONFIG="\
            \"training.lr=1e-3\" \
            \"training.patience=3\" \
            \"training.scheduler_factor=0.5\" \
            \"training.scheduler_patience=2\" \
            \"+training.gradient_accumulation_steps=4\""
    else
        FINETUNE_CONFIG="\
            \"model.head_hidden_size=786\" \
            \"model.head_layers=3\" \
            \"model.dropout=0.2\""
            
        TRAINING_CONFIG="\
            \"training.lr=1e-4\" \
            \"training.patience=4\" \
            \"training.scheduler_factor=0.5\" \
            \"training.scheduler_patience=3\" \
            \"+training.gradient_accumulation_steps=4\""
    fi
    
    COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${TASK}\" \
        \"experiment.tasks=${TASK}\" \
        \"experiment.type=lm_finetune\" \
        \"model=glot500_finetune\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.freeze_model=false\" \
        \"model.finetune=true\" \
        ${FINETUNE_CONFIG} \
        \"data.languages=[${LANG}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=15\" \
        \"training.batch_size=16\" \
        ${TRAINING_CONFIG} \
        ${DEBUG_PARAM} \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${OUTPUT_SUBDIR}\" \
        \"wandb.mode=offline\""
    
    if [ -n "$CONTROL_IDX" ]; then
        COMMAND="$COMMAND \
            \"experiment.use_controls=true\" \
            \"experiment.control_index=${CONTROL_IDX}\""
    fi
    
    if [ -n "$SUBMETRIC" ]; then
        COMMAND="$COMMAND \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    echo "RUNNING  EXPERIMENT: ${EXPERIMENT_NAME}"
    echo "COMMAND: $COMMAND"
    
    eval $COMMAND
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully"
        
        RESULTS_FILE="${OUTPUT_SUBDIR}/${LANG}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            CONTROL_PARAM=${CONTROL_IDX:-None}
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "finetune" "$LANG" "$TASK" \
                "${SUBMETRIC:-}" "$CONTROL_PARAM"
                
            echo "Experiment: $EXPERIMENT_NAME" > "${OUTPUT_SUBDIR}/${LANG}/experiment_summary.txt"
            if [ -n "$CONTROL_IDX" ]; then
                echo "CONTROL EXPERIMENT (random labels)" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            fi
            echo "Language: $LANG, Task: $TASK" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Results:" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            python -c "import json; f=open('${RESULTS_FILE}'); data=json.load(f); print(json.dumps(data.get('test_metrics', {}), indent=2))" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
        else
            echo "Warning: Results file not found: $RESULTS_FILE"
            echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        fi
        
        return 0
    else
        echo "ERROR in experiment ${EXPERIMENT_NAME}"
        echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true      #   manual gpu cleaning util
        
        return 1
    fi
}


echo "RUNNING MAIN FINETUNING (non-control)..."

for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        TASK_DIR="${OUTPUT_BASE_DIR}/${TASK}"
        mkdir -p "$TASK_DIR"
        
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" "" "$TASK_DIR"          # non control run
    done
done

echo "RUNNING CONTROL FINETUNING"

for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            CONTROL_DIR="${OUTPUT_BASE_DIR}/${TASK}/control${CONTROL_IDX}"
            mkdir -p "$CONTROL_DIR"
            
            run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "$CONTROL_IDX" "" "$CONTROL_DIR"       #control run
        done
    done
done

echo "RUNNING SUBMETRICS"

for LANG in "${LANGUAGES[@]}"; do
    for SUBMETRIC in "${SUBMETRICS[@]}"; do
        SUBMETRIC_DIR="${OUTPUT_BASE_DIR}/submetrics/${SUBMETRIC}"
        mkdir -p "$SUBMETRIC_DIR"
        
        run_finetune_experiment "regression" "$LANG" "single_submetric" "" "$SUBMETRIC" "$SUBMETRIC_DIR"
    done
done

echo "RUNNING SUBMETRIC CONTROLS"

for LANG in "${LANGUAGES[@]}"; do
    for SUBMETRIC in "${SUBMETRICS[@]}"; do
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            CONTROL_DIR="${OUTPUT_BASE_DIR}/submetrics/${SUBMETRIC}/control${CONTROL_IDX}"
            mkdir -p "$CONTROL_DIR"
            
            run_finetune_experiment "regression" "$LANG" "single_submetric" "$CONTROL_IDX" "$SUBMETRIC" "$CONTROL_DIR"
        done
    done
done


if [ -f "$FAILED_LOG" ]; then
    FAILED_COUNT=$(wc -l < "$FAILED_LOG")
    if [ $FAILED_COUNT -gt 0 ]; then
        echo "Some failed. See $FAILED_LOG for details."
        echo "Failed ($FAILED_COUNT):"
        cat "$FAILED_LOG"
    else
        echo "All experiments OK"
    fi
fi

TOTAL_EXPERIMENTS=$((${#LANGUAGES[@]} * (${#TASKS[@]} + ${#SUBMETRICS[@]}) * (1 + ${#CONTROL_INDICES[@]})))
COMPLETED_EXPERIMENTS=$(grep -v "experiment_type" "$RESULTS_TRACKER" | wc -l)

echo "=============================================="
echo "Finetuning completed"
