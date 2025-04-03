#!/bin/bash
# Test script for debug partition on VSC

# Slurm configurations (uncomment and modify for sbatch submission)
#SBATCH --time=00:30:00
#SBATCH --account=vsc37132  # Replace with your actual project
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1

# Check if running on a compute node
if [[ $(hostname) == *"r"* || $(hostname) == *"c"* || $(hostname) == *"s"* || $(hostname) == *"m"* ]]; then
    echo "Running on a compute node: $(hostname)"
    
    echo "Available resources:"
    echo "- CPU cores: $(nproc)"
    if [ -x "$(command -v nvidia-smi)" ]; then
        echo "- GPUs: $(nvidia-smi --list-gpus | wc -l)"
        nvidia-smi
    else
        echo "- No GPUs available"
    fi
    
    # Set up environment
    echo "Setting up environment..."
    module purge  # Clear modules
    module load Python/3.9
    
    # If using Miniconda
    if [ -d "$VSC_DATA/miniconda3" ]; then
        echo "Activating Miniconda environment..."
        export PATH="$VSC_DATA/miniconda3/bin:$PATH"
        source $VSC_DATA/miniconda3/bin/activate
        
        # If you have a specific conda environment
        # conda activate qtype-eval
    fi
    
    # Add project to PYTHONPATH
    export PYTHONPATH=$PYTHONPATH:$PWD
    
    # Create necessary directories
    mkdir -p data/cache data/features
    mkdir -p outputs/debug_test
    
    echo "Testing storage quotas..."
    myquota
    
    echo "Testing data loading..."
    python -c "from src.data.datasets import load_sklearn_data; print('Attempting to load data...'); try: (X_train, y_train), _, _ = load_sklearn_data(['en'], 'question_type', vectors_dir='./data/features'); print(f'Loaded {X_train.shape[0] if hasattr(X_train, \"shape\") else \"?\"} examples'); except Exception as e: print(f'Error loading data: {e}')"
    
    echo "Running a small experiment..."
    python -m src.experiments.run_experiment \
        experiment=question_type \
        model=dummy \
        data.languages="[en]" \
        experiment_name="debug_test" \
        output_dir="./debug_outputs" \
        wandb.mode="disabled"
    
    echo "Debug test completed."
else
    echo "Not running on a compute node. Submit this script to a debug partition:"
    echo ""
    echo "For Genius debug partition:"
    echo "sbatch --partition=gpu_p100_debug --time=00:30:00 --account=YOUR_PROJECT test_debug_partition.sh"
    echo ""
    echo "For wICE debug partition:"
    echo "sbatch --partition=gpu_a100_debug --time=00:30:00 --account=YOUR_PROJECT test_debug_partition.sh"
fi