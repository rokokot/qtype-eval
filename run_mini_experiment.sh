#!/bin/bash
#SBATCH --job-name=mini_test
#SBATCH --time=01:00:00
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
conda activate qtype_env

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Disable wandb for test run
export WANDB_MODE="disabled"

# Create output directory
mkdir -p mini_test_output

# Print environment information
echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run a mini experiment on just one language with a small model
echo "Running mini experiment..."
python -m src.experiments.run_experiment \
    experiment=question_type \
    model=dummy \
    data.languages="[en]" \
    experiment_name="mini_test" \
    output_dir="./mini_test_output" \
    training.num_epochs=1

# Check the output
ls -la mini_test_output
cat mini_test_output/results*.json

echo "Mini experiment completed!"
