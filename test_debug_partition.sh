#!/bin/bash
# Test script for debug partition

# Check if running on a compute node
if [[ $(hostname) == *"r"* || $(hostname) == *"c"* ]]; then
    echo "Running on a compute node: $(hostname)"
    
    # Print available resources
    echo "Available resources:"
    echo "- CPU cores: $(nproc)"
    if [ -x "$(command -v nvidia-smi)" ]; then
        echo "- GPUs: $(nvidia-smi --list-gpus | wc -l)"
        nvidia-smi
    else
        echo "- No GPUs available"
    fi
    
    # Check if poetry is active
    if [ -z "$POETRY_ACTIVE" ]; then
        echo "Poetry environment not active. Activating..."
        if [ -d ".venv" ]; then
            source .venv/bin/activate
        else
            echo "Warning: No .venv directory found. Make sure to run in your poetry environment."
        fi
    fi
    
    # Run a simple test
    echo "Testing data loading..."
    python -c "from src.data.datasets import load_sklearn_data; (X_train, y_train), _, _ = load_sklearn_data(['en'], 'question_type'); print(f'Loaded {X_train.shape[0]} examples with {X_train.shape[1]} features')"
    
    # Run a small experiment
    echo "Running a small experiment..."
    python -m src.experiments.run_experiment \
        experiment=question_type \
        model=dummy \
        data.languages="[en]" \
        experiment_name="debug_test" \
        output_dir="./debug_outputs"
    
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