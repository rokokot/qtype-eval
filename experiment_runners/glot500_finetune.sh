#!/bin/bash
#SBATCH --job-name=finetune_experiments
#SBATCH --time=48:00:00  # Increase time allocation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100  # A100 is more powerful than P100
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
OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_output"
mkdir -p $OUTPUT_BASE_DIR
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/finetune_results.csv"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
mkdir -p $LOG_DIR

# Initialize results tracker if it doesn't exist
if [ ! -f "$RESULTS_TRACKER" ]; then
    echo "experiment_type,language,task,submetric,control_index,metric,value" > $RESULTS_TRACKER
fi

# Also track failed experiments
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch $FAILED_LOG

# === Configuration ===
LANGUAGES=("en" "ar" "fi" "id" "ja" "ko" "ru")
TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
CONTROL_INDICES=(1 2 3)

# === Create metrics extractor script ===
# (Keep existing extract_metrics.py script)

# === Improved experiment function with retry logic ===
run_finetune_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    local OUTPUT_SUBDIR=$6
    local MAX_RETRIES=2  # Allow up to 2 retries
    
    local EXPERIMENT_NAME=""
    local COMMAND=""
    local EXPERIMENT_TYPE="finetune"
    
    # Set experiment name based on parameters
    if [ -n "$SUBMETRIC" ]; then
        # This is a submetric experiment
        TASK="single_submetric"
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_${LANG}"
        else
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_control${CONTROL_IDX}_${LANG}"
        fi
    else
        # Regular task experiment
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${TASK}_${LANG}"
        else
            EXPERIMENT_NAME="finetune_${TASK}_control${CONTROL_IDX}_${LANG}"
        fi
    fi
    
    # Create log files
    LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"
    ERROR_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.err"
    
    # Skip if already successful
    if [ -f "${OUTPUT_SUBDIR}/results.json" ]; then
        echo "Experiment ${EXPERIMENT_NAME} already completed successfully. Skipping."
        return 0
    fi
    
    # Skip if already failed too many times
    local FAIL_COUNT=$(grep -c "^${EXPERIMENT_NAME}$" $FAILED_LOG)
    if [ $FAIL_COUNT -ge $MAX_RETRIES ]; then
        echo "Experiment ${EXPERIMENT_NAME} has failed $FAIL_COUNT times already. Skipping."
        return 1
    fi
    
    echo "Running experiment: ${EXPERIMENT_NAME}"
    
    # Build command with all necessary parameters
    # Note the explicit probe_hidden_size parameter
    COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${TASK}\" \
        \"experiment.tasks=${TASK}\" \
        \"model=glot500_finetune\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.dropout=0.1\" \
        \"model.freeze_model=false\" \
        \"model.finetune=true\" \
        \"model.probe_hidden_size=96\" \
        \"data.languages=[${LANG}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=10\" \
        \"training.batch_size=16\" \
        \"training.lr=2e-5\" \
        \"+training.gradient_accumulation_steps=2\" \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${OUTPUT_SUBDIR}\" \
        \"wandb.mode=offline\""
    
    # Add control parameters if needed
    if [ -n "$CONTROL_IDX" ]; then
        COMMAND="$COMMAND \
            \"experiment.use_controls=true\" \
            \"experiment.control_index=${CONTROL_IDX}\""
    fi
    
    # Add submetric parameter if needed
    if [ -n "$SUBMETRIC" ]; then
        COMMAND="$COMMAND \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    # Log the command
    echo "Command: $COMMAND" | tee -a $LOG_FILE
    
    # Execute the experiment with error handling
    eval $COMMAND >> $LOG_FILE 2> $ERROR_FILE
    local RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully"
        
        # Extract metrics if results file exists
        RESULTS_FILE="${OUTPUT_SUBDIR}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "finetune" "$LANG" "$TASK" \
                "${SUBMETRIC:-}" "${CONTROL_IDX:-None}"
            
            # Create a summary file
            echo "Experiment: $EXPERIMENT_NAME" > "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Language: $LANG, Task: $TASK" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            cat "$RESULTS_FILE" | grep -E "test_metrics|train_metrics" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
        else
            echo "Warning: Results file not found: $RESULTS_FILE" >> $ERROR_FILE
            echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
            return 1
        fi
        
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}" | tee -a $ERROR_FILE
        echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 1
    fi
}

# === Prioritize experiments ===
# Run a smaller subset first to ensure everything works
echo "Running priority experiments..."
PRIORITY_LANGUAGES=("en")
PRIORITY_TASKS=("question_type")

for LANG in "${PRIORITY_LANGUAGES[@]}"; do
    for TASK in "${PRIORITY_TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        # Create output directory
        TASK_DIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}"
        mkdir -p "$TASK_DIR"
        
        # Run standard (non-control) experiment
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" "" "$TASK_DIR"
        
        # Wait a bit to let GPU memory clear
        sleep 10
    done
done

# === Run main experiments ===
echo "Running main finetuning experiments (non-control)..."

for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        # Skip priority experiments that have already been run
        if [[ " ${PRIORITY_LANGUAGES[@]} " =~ " ${LANG} " ]] && [[ " ${PRIORITY_TASKS[@]} " =~ " ${TASK} " ]]; then
            continue
        fi
        
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        # Create output directory
        TASK_DIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}"
        mkdir -p "$TASK_DIR"
        
        # Run standard (non-control) experiment
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" "" "$TASK_DIR"
        
        # Wait a bit to let GPU memory clear
        sleep 10
    done
done

# ... (continue with control and submetric experiments, adding sleep commands between runs)