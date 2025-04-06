#!/bin/bash
#SBATCH --job-name=mini_classification_debug
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

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DEBUG=1  # Enable debug mode (reduces workers for dataloaders)

# Print environment information
echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"

# Clear previous output directory
rm -rf mini_classification_output
mkdir -p mini_classification_output

# Print environment information
echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run a mini experiment with more verbose logging
echo "Running mini classification experiment..."

python -m src.experiments.run_experiment \
    "experiment=[complexity]" \
    "model=lm_probe" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.languages=[ar]" \
    "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
    "training.num_epochs=3" \
    "training.batch_size=8" \
    "training.task_type=regression" \
    "experiment_name=mini_test_classification_glot500_ar" \
    "output_dir=./mini_classification_output"

# Check output files
echo "Output directory contents:"
ls -la mini_classification_output
cat mini_classification_output/all_results.json || echo "No results file found"
echo "Error files if any:"
ls -la mini_classification_output/error_*.json 2>/dev/null || echo "No error files found"

# If error files exist, show their contents
for error_file in mini_classification_output/error_*.json; do
    if [ -f "$error_file" ]; then
        echo "Contents of $error_file:"
        cat "$error_file"
    fi
done

echo "Mini classification test completed"

# Show GPU usage
echo "GPU memory usage:"
nvidia-smi --query-gpu=memory.used --format=csv
