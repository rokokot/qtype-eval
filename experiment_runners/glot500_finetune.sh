#!/bin/bash
#SBATCH --job-name=finetune_experiments
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p100
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37132

export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the environment
echo "Activating conda environment..."
conda activate qtype-eval

# Install required packages if needed
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install hydra-core hydra-submitit-launcher
pip install "transformers>=4.30.0,<4.36.0" torch datasets wandb

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

# Print environment information
echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "GPU information:"
nvidia-smi

echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Define configuration
LANGUAGES=("en" "ar" "fi" "id" "ja" "ko" "ru")
TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
CONTROL_INDICES=(1 2 3)

# Use maximum head sizes for all tasks
# For fine-tuning, use the same large head size for all task types
HEAD_SIZE=768  # Full model dimension for maximum expressivity

# Base output directory
OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_output"
mkdir -p $OUTPUT_BASE_DIR

# Set up experiment tracking and logging
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/experiment_results.csv"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
mkdir -p $LOG_DIR

# Initialize results tracker if it doesn't exist
if [ ! -f "$RESULTS_TRACKER" ]; then
    echo "language,task,control_index,submetric,status,runtime_seconds,gpu_memory_mb,date_completed" > $RESULTS_TRACKER
fi

# Track failed experiments
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch $FAILED_LOG

# Define priority experiments to run first (validation set)
PRIORITY_LANGUAGES=("en")
PRIORITY_TASKS=("question_type")

# Verify lm_finetune config exists
LM_FINETUNE_CONFIG="configs/model/lm_finetune.yaml"
if [ ! -f "$LM_FINETUNE_CONFIG" ]; then
    echo "Creating missing lm_finetune.yaml config file..."
    mkdir -p $(dirname "$LM_FINETUNE_CONFIG")
    cat > "$LM_FINETUNE_CONFIG" << 'EOF'
# configs/model/lm_finetune.yaml
model_type: "lm_finetune"
lm_name: "cis-lmu/glot500-base"
dropout: 0.1
layer_wise: false
layer_index: -1
num_outputs: 1
head_hidden_size: 768
head_layers: 2
EOF
    echo "Created $LM_FINETUNE_CONFIG"
fi

# Function to run a finetuning experiment
run_finetune_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    local MAX_RETRIES=2
    
    local EXPERIMENT_TYPE="finetune"
    local TASK_SPEC=$TASK
    local EXPERIMENT_NAME=""
    local OUTPUT_SUBDIR=""
    
    # Set experiment name and output directory based on parameters
    if [ -n "$SUBMETRIC" ]; then
        # This is a submetric experiment
        TASK_SPEC="single_submetric"
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_${LANG}"
            OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/single_submetric/${LANG}/${SUBMETRIC}"
        else
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_control${CONTROL_IDX}_${LANG}"
            OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/single_submetric/${LANG}/${SUBMETRIC}/control${CONTROL_IDX}"
        fi
    else:
        # Regular task experiment
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${TASK}_${LANG}"
            OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}"
        else
            EXPERIMENT_NAME="finetune_${TASK}_control${CONTROL_IDX}_${LANG}"
            OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}/control${CONTROL_IDX}"
        fi
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
    
    echo "Running experiment: ${EXPERIMENT_NAME}"
    echo "Output directory: ${OUTPUT_SUBDIR}"
    
    # Record start time
    local START_TIME=$(date +%s)
    
    # Build command - using lm_finetune model type with maximum head size
    # IMPORTANT: Output goes directly to SLURM, not to separate log files
    local COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${TASK_SPEC}\" \
        \"experiment.tasks=${TASK_SPEC}\" \
        \"model=lm_finetune\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.dropout=0.1\" \
        \"model.head_hidden_size=${HEAD_SIZE}\" \
        \"model.head_layers=2\" \
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
    echo "Command: $COMMAND"
    
    # Execute the experiment with error handling - OUTPUT DIRECTLY TO SLURM
    eval $COMMAND
    local RESULT=$?
    
    # Calculate runtime
    local END_TIME=$(date +%s)
    local RUNTIME=$((END_TIME - START_TIME))
    
    # Get GPU memory usage if possible
    local GPU_MEM="NA"
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    fi
    
    # Handle result
    if [ $RESULT -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully in ${RUNTIME} seconds"
        
        # Still save command/outputs to log files for reference
        echo "Command: $COMMAND" > $LOG_FILE
        echo "Status: Success, Runtime: ${RUNTIME}s, Completed: $(date)" >> $LOG_FILE
        
        # Record success in tracker
        echo "${LANG},${TASK},${CONTROL_IDX:-None},${SUBMETRIC:-None},success,${RUNTIME},${GPU_MEM},$(date +"%Y-%m-%d %H:%M:%S")" >> $RESULTS_TRACKER
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}"
        echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        
        # Still save error info to log files for reference
        echo "Command: $COMMAND" > $ERROR_FILE
        echo "Status: Failed, Runtime: ${RUNTIME}s, Completed: $(date)" >> $ERROR_FILE
        
        # Record failure in tracker
        echo "${LANG},${TASK},${CONTROL_IDX:-None},${SUBMETRIC:-None},failed,${RUNTIME},${GPU_MEM},$(date +"%Y-%m-%d %H:%M:%S")" >> $RESULTS_TRACKER
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 1
    fi
}

# Verify that finetuning is set up correctly
echo "Verifying finetuning model configuration..."
python -c "
import sys
import os
sys.path.append(os.getcwd())
try:
    from src.models.model_factory import create_model
    
    # Create a fine-tuning model
    model = create_model('lm_finetune', 'classification', 
                         lm_name='cis-lmu/glot500-base',
                         head_hidden_size=768)
    
    # Check if model is actually unfrozen
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Print status
    print(f'Fine-tuning model check:')
    print(f'- Trainable parameters: {trainable_params:,}')
    print(f'- Total parameters: {total_params:,}')
    print(f'- Percentage trainable: {trainable_params/total_params*100:.2f}%')
    
    # Check if encoder is trainable (should be for fine-tuning)
    encoder_trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    encoder_total = sum(p.numel() for p in model.model.parameters())
    
    print(f'- Encoder trainable: {encoder_trainable:,} / {encoder_total:,} ({encoder_trainable/encoder_total*100:.2f}%)')
    
    # Check head size
    if hasattr(model, 'head') and isinstance(model.head, torch.nn.Sequential):
        for module in model.head:
            if isinstance(module, torch.nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                print(f'- Head layer: {in_features} → {out_features} features')
    
    if encoder_trainable == 0:
        print('WARNING: Encoder is completely frozen! This is not fine-tuning!')
        sys.exit(1)
    else:
        print('SUCCESS: Model is properly set up for fine-tuning with maximum head size.')
except Exception as e:
    print(f'Error checking model: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# If model check failed, exit
if [ $? -ne 0 ]; then
    echo "ERROR: Fine-tuning model configuration is incorrect. Fix the model implementation before continuing."
    exit 1
fi

# Run priority experiments first
echo "===== Running priority experiments ====="
for LANG in "${PRIORITY_LANGUAGES[@]}"; do
    for TASK in "${PRIORITY_TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        echo "Running priority experiment: $LANG, $TASK"
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" ""
        
        # Wait a bit to let GPU memory clear
        sleep 10
        
        # Test one control experiment as a validation
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "1" ""
        
        # Wait a bit to let GPU memory clear
        sleep 10
        
        # Test one submetric experiment as a validation
        run_finetune_experiment "regression" "$LANG" "single_submetric" "" "avg_links_len"
        
        # Wait a bit to let GPU memory clear
        sleep 10
    done
done

# Standard experiments (non-control)
echo "===== Running main finetuning experiments ====="
for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        # Skip priority experiments that have already been run
        if [[ " ${PRIORITY_LANGUAGES[@]} " =~ " ${LANG} " ]] && [[ " ${PRIORITY_TASKS[@]} " =~ " ${TASK} " ]]; then
            echo "Skipping already run priority experiment: $LANG, $TASK"
            continue
        fi
        
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        echo "Running standard experiment: $LANG, $TASK"
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" ""
        
        # Wait between experiments
        sleep 30
    done
    
    # Clear GPU memory after each language
    echo "Clearing GPU memory after language $LANG"
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 30  # Longer pause between languages
done

# Control experiments
echo "===== Running control experiments ====="
for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            # Skip control 1 for priority experiments as it's already run
            if [[ " ${PRIORITY_LANGUAGES[@]} " =~ " ${LANG} " ]] && [[ " ${PRIORITY_TASKS[@]} " =~ " ${TASK} " ]] && [ "$CONTROL_IDX" == "1" ]; then
                echo "Skipping already run priority control experiment: $LANG, $TASK, control $CONTROL_IDX"
                continue
            fi
            
            echo "Running control experiment: $LANG, $TASK, control $CONTROL_IDX"
            run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "$CONTROL_IDX" ""
            
            # Wait between experiments
            sleep 30
        done
    done
    
    # Clear GPU memory after each language
    echo "Clearing GPU memory after language $LANG controls"
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 30  # Longer pause between languages
done

# Submetric experiments
echo "===== Running submetric experiments ====="
for LANG in "${LANGUAGES[@]}"; do
    for SUBMETRIC in "${SUBMETRICS[@]}"; do
        # Skip priority submetric that was already run
        if [[ " ${PRIORITY_LANGUAGES[@]} " =~ " ${LANG} " ]] && [ "$SUBMETRIC" == "avg_links_len" ]; then
            echo "Skipping already run priority submetric experiment: $LANG, $SUBMETRIC"
            continue
        fi
        
        echo "Running submetric experiment: $LANG, $SUBMETRIC"
        run_finetune_experiment "regression" "$LANG" "single_submetric" "" "$SUBMETRIC"
        
        # Wait between experiments
        sleep 30
        
        # Run control experiments for submetrics
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            echo "Running submetric control experiment: $LANG, $SUBMETRIC, control $CONTROL_IDX"
            run_finetune_experiment "regression" "$LANG" "single_submetric" "$CONTROL_IDX" "$SUBMETRIC"
            
            # Wait between experiments
            sleep 30
        done
    done
    
    # Clear GPU memory after each language
    echo "Clearing GPU memory after language $LANG submetrics"
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 30  # Longer pause between languages
done

# Generate summary of experiments
echo "===== Generating experiment summary ====="

echo "All finetuning experiments completed"
echo "Results can be found in ${OUTPUT_BASE_DIR}"