#!/bin/bash
#SBATCH --job-name=test_layerwise
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_debug
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
export WANDB_DIR="$VSC_DATA/wandb"
mkdir -p "$VSC_DATA/wandb"

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
LANGUAGE="en"
LAYERS=(2 6 11)  # Early, middle, and second-to-last layer
TASKS=("question_type")  # Only test question_type for test script

# Base output directory
OUTPUT_BASE_DIR="$VSC_DATA/layerwise_test_output"
mkdir -p $OUTPUT_BASE_DIR

# Run layer-wise experiments for question_type
for LAYER in "${LAYERS[@]}"; do
    echo "==============================================="
    echo "Running layer $LAYER experiment for $LANGUAGE"
    echo "==============================================="
    
    LAYER_DIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}"
    mkdir -p "${LAYER_DIR}"
    
    # Run the experiment with layer_wise=true and the specific layer index
    python -m src.experiments.run_experiment \
        "hydra.job.chdir=False" \
        "hydra.run.dir=." \
        "experiment=question_type" \
        "experiment.tasks=question_type" \
        "model=lm_probe" \
        "model.lm_name=cis-lmu/glot500-base" \
        "model.layer_wise=true" \
        "model.layer_index=${LAYER}" \
        "data.languages=[${LANGUAGE}]" \
        "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
        "training.task_type=classification" \
        "training.num_epochs=3" \
        "training.batch_size=16" \
        "experiment_name=layer_${LAYER}_question_type_${LANGUAGE}" \
        "output_dir=${LAYER_DIR}" \
        "wandb.mode=offline"
    
    if [ $? -eq 0 ]; then
        echo "Layer $LAYER experiment for $LANGUAGE completed successfully"
    else
        echo "Error in layer $LAYER experiment for $LANGUAGE"
    fi
done

# Check the results
echo "Results summary:"
for LAYER in "${LAYERS[@]}"; do
    RESULT_FILE="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/results.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "Layer $LAYER results:"
        cat "$RESULT_FILE" | grep -E "accuracy|f1|test_metrics"
        echo ""
    else
        echo "No results file found for layer $LAYER"
    fi
done

echo "Test script completed"