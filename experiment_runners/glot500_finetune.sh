#!/bin/bash
#SBATCH --job-name=finetune_experiments
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
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
echo "HYDRA_JOB_CHDIR=${HYDRA_JOB_CHDIR}"
echo "GPU information:"
nvidia-smi

echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Define configuration
LANGUAGES=("en" "ar" "fi" "id" "ja" "ko" "ru")
TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
CONTROL_INDICES=(1 2 3)

# Base output directory
OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_output"
mkdir -p $OUTPUT_BASE_DIR

# Function to run a finetuning experiment
run_finetune_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    
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
    else
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
    
    echo "Running experiment: ${EXPERIMENT_NAME}"
    
    # Build command
    local COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${TASK_SPEC}\" \
        \"experiment.tasks=${TASK_SPEC}\" \
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
    
    # Execute the experiment
    eval $COMMAND
    
    if [ $? -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully"
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}"
        return 1
    fi
}

# Run priority experiments first
echo "===== Running priority experiments ====="
PRIORITY_LANGUAGES=("en")
PRIORITY_TASKS=("question_type")

for LANG in "${PRIORITY_LANGUAGES[@]}"; do
    for TASK in "${PRIORITY_TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        echo "Running priority experiment: $LANG, $TASK"
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" ""
        
        # Wait between experiments
        sleep 30
    done
done

# Run main experiments
echo "===== Running main finetuning experiments ====="

# Standard experiments (non-control)
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
            echo "Running control experiment: $LANG, $TASK, control $CONTROL_IDX"
            run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "$CONTROL_IDX" ""
            
            # Wait between experiments
            sleep 30
        done
    done
done

# Submetric experiments
echo "===== Running submetric experiments ====="

for LANG in "${LANGUAGES[@]}"; do
    for SUBMETRIC in "${SUBMETRICS[@]}"; do
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
done

echo "All finetuning experiments completed"