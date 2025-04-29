#!/bin/bash
#SBATCH --job-name=layerwise_probing
#SBATCH --time=24:00:00  # Increase time allocation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100  # A100 is better for this task
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
OUTPUT_BASE_DIR="$VSC_SCRATCH/layerwise_output"
mkdir -p $OUTPUT_BASE_DIR
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/results_tracker.csv"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
mkdir -p $LOG_DIR

# Initialize results tracker
echo "experiment_type,language,layer,task,submetric,control_index,metric,value" > $RESULTS_TRACKER

# Also track failed experiments
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch $FAILED_LOG

# === Configuration ===
LANGUAGES=("en" "ar" "fi" "id" "ja" "ko" "ru")
LAYERS=(2 6 11 12)
MAIN_TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
CONTROL_INDICES=(1 2 3)

# === Create metrics extractor script ===
# (Keep existing extract_metrics.py script)

# === Improved experiment functions ===
run_standard_experiment() {
    local LANGUAGE=$1
    local LAYER=$2
    local TASK=$3
    local TASK_TYPE=$4
    local SUBMETRIC=$5
    local MAX_RETRIES=2
    
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
    
    echo "Running $TASK_SPEC experiment for language $LANGUAGE, layer $LAYER"
    
    # Build command with all necessary parameters
    # Note the explicit probe_hidden_size parameter
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
        \"model.probe_hidden_size=96\" \
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
    
    # Log the command
    echo "Command: $COMMAND" | tee -a $LOG_FILE
    
    # Execute the experiment with error handling
    eval $COMMAND >> $LOG_FILE 2> $ERROR_FILE
    local RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Standard experiment completed successfully: $EXPERIMENT_NAME"
        
        # Extract metrics
        RESULTS_FILE="${OUTPUT_SUBDIR}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "standard" "$LANGUAGE" "$LAYER" \
                "$TASK_SPEC" "${SUBMETRIC:-None}" "None"
            
            return 0
        else
            echo "Warning: Results file not found: $RESULTS_FILE" >> $ERROR_FILE
            echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
            return 1
        fi
    else
        echo "Error in standard experiment: $EXPERIMENT_NAME" | tee -a $ERROR_FILE
        echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 1
    fi
}

run_control_experiment() {
    # ... (similar to run_standard_experiment with control parameters)
    # Be sure to add explicit probe_hidden_size parameter
}

# === Prioritize experiments ===
# Run a smaller subset first to ensure everything works
echo "Running priority experiments..."
PRIORITY_LANGUAGES=("en")
PRIORITY_LAYERS=(12)  # Start with the last layer

for LANGUAGE in "${PRIORITY_LANGUAGES[@]}"; do
    for LAYER in "${PRIORITY_LAYERS[@]}"; do
        for TASK in "${MAIN_TASKS[@]}"; do
            TASK_TYPE="classification"
            if [ "$TASK" == "complexity" ]; then
                TASK_TYPE="regression"
            fi

            # Standard experiments
            run_standard_experiment "$LANGUAGE" "$LAYER" "$TASK" "$TASK_TYPE" ""
            
            # Wait a bit to let GPU memory clear
            sleep 10
            
            # Run one control experiment as a test
            run_control_experiment "$LANGUAGE" "$LAYER" "$TASK" "$TASK_TYPE" "1" ""
            
            # Wait a bit to let GPU memory clear
            sleep 10
        done
    done
done

# === Run main experiments ===
# ... (continue with main experiment loops, adding sleep commands between runs)