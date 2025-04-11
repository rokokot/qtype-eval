#!/bin/bash
#SBATCH --job-name=layerwise_analysis
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

# Use your personal Miniconda installation
export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the environment
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

# Define parameters
LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
LAYERS=(2 6 11)  # Early, middle, and second-to-last layer
MAIN_TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
CONTROL_INDICES=(1 2 3)

# Base output directory
OUTPUT_BASE_DIR="$VSC_SCRATCH/layerwise_output"
mkdir -p $OUTPUT_BASE_DIR

# Function to run a standard experiment (without controls)
run_standard_experiment() {
    local LANGUAGE=$1
    local LAYER=$2
    local TASK=$3
    local TASK_TYPE=$4
    local SUBMETRIC=$5
    
    # Determine the experiment type and task specification
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
    
    # Create output directory
    mkdir -p "$OUTPUT_SUBDIR"
    
    echo "Running $TASK_SPEC experiment for language $LANGUAGE, layer $LAYER"
    
    # Build the command with conditional submetric parameter
    local COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${EXPERIMENT_TYPE}\" \
        \"experiment.tasks=${TASK_SPEC}\" \
        \"model=lm_probe\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.layer_wise=true\" \
        \"model.layer_index=${LAYER}\" \
        \"data.languages=[${LANGUAGE}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=10\" \
        \"training.batch_size=16\" \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${OUTPUT_SUBDIR}\" \
        \"wandb.mode=offline\""
    
    # Add submetric parameter if provided
    if [ -n "$SUBMETRIC" ]; then
        COMMAND+=" \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    # Execute the command
    eval $COMMAND
    
    if [ $? -eq 0 ]; then
        echo "Standard experiment completed successfully: $EXPERIMENT_NAME"
        return 0
    else
        echo "Error in standard experiment: $EXPERIMENT_NAME"
        return 1
    fi
}

# Function to run a control experiment
run_control_experiment() {
    local LANGUAGE=$1
    local LAYER=$2
    local TASK=$3
    local TASK_TYPE=$4
    local CONTROL_IDX=$5
    local SUBMETRIC=$6
    
    # Determine the experiment type and task specification
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
    
    # Create output directory
    mkdir -p "$OUTPUT_SUBDIR"
    
    echo "Running $TASK_SPEC control experiment for language $LANGUAGE, layer $LAYER, control $CONTROL_IDX"
    
    # Build the command with conditional submetric parameter
    local COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${EXPERIMENT_TYPE}\" \
        \"experiment.tasks=${TASK_SPEC}\" \
        \"model=lm_probe\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.layer_wise=true\" \
        \"model.layer_index=${LAYER}\" \
        \"experiment.use_controls=true\" \
        \"experiment.control_index=${CONTROL_IDX}\" \
        \"data.languages=[${LANGUAGE}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=10\" \
        \"training.batch_size=16\" \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${OUTPUT_SUBDIR}\" \
        \"wandb.mode=offline\""
    
    # Add submetric parameter if provided
    if [ -n "$SUBMETRIC" ]; then
        COMMAND+=" \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    # Execute the command
    eval $COMMAND
    
    if [ $? -eq 0 ]; then
        echo "Control experiment completed successfully: $EXPERIMENT_NAME"
        return 0
    else
        echo "Error in control experiment: $EXPERIMENT_NAME"
        return 1
    fi
}

# Main execution loop - Standard Experiments

# 1. Main tasks (question_type and complexity)
for LANGUAGE in "${LANGUAGES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        # Create layer directory
        mkdir -p "${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}"
        
        # Run question_type experiments (classification)
        run_standard_experiment "$LANGUAGE" "$LAYER" "question_type" "classification"
        
        # Run complexity experiments (regression)
        run_standard_experiment "$LANGUAGE" "$LAYER" "complexity" "regression"
    done
done

# 2. Submetric tasks
for LANGUAGE in "${LANGUAGES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for SUBMETRIC in "${SUBMETRICS[@]}"; do
            run_standard_experiment "$LANGUAGE" "$LAYER" "" "regression" "$SUBMETRIC"
        done
    done
done

# Control Experiments

# 1. Main tasks with controls
for LANGUAGE in "${LANGUAGES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            # Run question_type control experiments
            run_control_experiment "$LANGUAGE" "$LAYER" "question_type" "classification" "$CONTROL_IDX"
            
            # Run complexity control experiments
            run_control_experiment "$LANGUAGE" "$LAYER" "complexity" "regression" "$CONTROL_IDX"
        done
    done
done

# 2. Submetric tasks with controls
for LANGUAGE in "${LANGUAGES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for SUBMETRIC in "${SUBMETRICS[@]}"; do
            for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
                run_control_experiment "$LANGUAGE" "$LAYER" "" "regression" "$CONTROL_IDX" "$SUBMETRIC"
            done
        done
    done
done

echo "Layer-wise analysis completed"
