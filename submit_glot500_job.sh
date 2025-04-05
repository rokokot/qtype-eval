#!/bin/bash
#SBATCH --job-name=glot500_exp
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18     
#SBATCH --mem=123G             
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=robin.edu.hr@gmail.com

# Use your personal Miniconda installation
export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Create the environment if it doesn't exist
conda create -n qtype-eval python=3.9 -y || echo "Environment already exists"

# Activate the environment
conda activate qtype-eval

echo "Installing PyTorch with CUDA support..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing other dependencies..."
pip install hydra-core hydra-submitit-launcher
pip install -r requirements.txt
pip install --no-cache-dir wandb

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export WANDB_API_KEY="282936b31f3ab3415a24a3dba88151d5f7e5bf10"
export WANDB_ENTITY="rokii-ku-leuven"
export WANDB_PROJECT="multilingual-question-probing"

# Log in to wandb
wandb login $WANDB_API_KEY

# Print environment information for debugging
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Active conda env: $CONDA_DEFAULT_ENV"
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"



# Run experiments
if [ -z "$1" ]; then
    bash run_all_experiments.sh
else
    case "$1" in
        "basic")
            bash run_basic_experiments.sh
            ;;
        "control")
            bash run_control_experiments.sh
            ;;
        "submetric")
            bash run_all_submetrics.sh
            ;;
        "cross")
            bash run_all_cross_lingual.sh
            ;;
        *)
            echo "Unknown experiment type: $1"
            echo "Valid options: basic, control, submetric, cross"
            exit 1
            ;;
    esac
fi
