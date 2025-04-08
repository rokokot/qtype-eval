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

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Python packages
pip install hydra-core hydra-submitit-launcher
pip install "transformers>=4.30.0,<4.36.0" torch datasets wandb
# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DEBUG=1  # Enable debug mode (reduces workers for dataloaders)

# IMPORTANT: Disable Hydra's working directory changes
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1

# Print environment information
echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
echo "HYDRA_JOB_CHDIR=${HYDRA_JOB_CHDIR}"

# Clear previous output directory
MINI_OUTPUT_DIR="${PWD}/mini_classification_output"
rm -rf $MINI_OUTPUT_DIR
mkdir -p $MINI_OUTPUT_DIR
echo "Output directory: ${MINI_OUTPUT_DIR} (absolute path)"

# Create language subdirectory explicitly
mkdir -p "${MINI_OUTPUT_DIR}/ar"
echo "Created language directory: ${MINI_OUTPUT_DIR}/ar"

# Print environment information
echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run a mini experiment with more verbose logging
echo "Running mini regression experiment for complexity..."

# Use hydra command-line overrides to disable directory changes
python -m src.experiments.run_experiment \
    "hydra.job.chdir=False" \
    "hydra.run.dir=." \
    "experiment=submetrics" \
    "experiment.tasks=single_submetric" \
    "experiment.submetric=n_tokens" \
    "model=lm_probe" \
    "model.lm_name=cis-lmu/glot500-base" \
    "experiment.use_controls=true" \
    "experiment.control_index=1" \
    "data.languages=[ja]" \
    "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
    "training.num_epochs=10" \
    "training.batch_size=16" \
    "training.task_type=regression" \
    "experiment_name=mini_complexity_regression_glot500_ja" \
    "output_dir=${MINI_OUTPUT_DIR}" \
    "wandb.mode=offline"

# Function to find and check results
check_results() {
    echo "Output directory contents:"
    ls -la $MINI_OUTPUT_DIR
    
    # Try to find results in multiple possible locations
    all_results=()
    all_results+=("${MINI_OUTPUT_DIR}/all_results.json")
    all_results+=("${MINI_OUTPUT_DIR}/ar/results.json")
    all_results+=("${MINI_OUTPUT_DIR}/results.json")
    all_results+=("./outputs/mini_complexity_regression_glot500_ar/*/all_results.json")
    all_results+=("./outputs/mini_complexity_regression_glot500_ar/*/*/all_results.json")
    
    results_found=false
    
    for result_path in "${all_results[@]}"; do
        # Use compgen for wildcard expansion
        for file in $(compgen -G "$result_path" 2>/dev/null || echo ""); do
            if [ -f "$file" ]; then
                echo "Found results file: $file"
                cat "$file" | grep -E "task|task_type|language|metrics" || echo "No matching content found in file"
                results_found=true
                
                # Copy the result to the standard location if it's not already there
                if [ "$file" != "${MINI_OUTPUT_DIR}/all_results.json" ] && [ "$file" != "${MINI_OUTPUT_DIR}/ar/results.json" ]; then
                    echo "Copying result to standard location..."
                    if [[ "$file" == *"/ar/results.json" ]]; then
                        mkdir -p "${MINI_OUTPUT_DIR}/ar"
                        cp "$file" "${MINI_OUTPUT_DIR}/ar/results.json"
                    else
                        cp "$file" "${MINI_OUTPUT_DIR}/all_results.json"
                    fi
                fi
                
                break
            fi
        done
    done
    
    if [ "$results_found" = false ]; then
        echo "No results file found"
        # Search recursively for any JSON files
        echo "Searching recursively for any JSON files..."
        find . -name "*.json" -mtime -1 -not -path "*/\.git/*" | sort
    fi
}

# Check output files
check_results

# Check for error files
echo "Error files if any:"
ERROR_FILES=$(find $MINI_OUTPUT_DIR -name "error_*.json" 2>/dev/null)
if [ -n "$ERROR_FILES" ]; then
    for error_file in $ERROR_FILES; do
        echo "Contents of $error_file:"
        cat "$error_file"
    done
else 
    echo "No error files found"
fi

# Check for wandb logs
echo "Checking for wandb logs:"
WANDB_DIRS=()
WANDB_DIRS+=("${MINI_OUTPUT_DIR}/wandb")
WANDB_DIRS+=("./wandb")
WANDB_DIRS+=("./outputs/mini_complexity_regression_glot500_ar/*/wandb")

for wandb_dir in "${WANDB_DIRS[@]}"; do
    # Use compgen for wildcard expansion
    for dir in $(compgen -G "$wandb_dir" 2>/dev/null || echo ""); do
        if [ -d "$dir" ]; then
            echo "Found wandb directory: $dir"
            ls -la "$dir"
            
            # Try to get the last run ID
            OFFLINE_RUN=$(find "$dir" -name "offline-run-*" -type d | sort | tail -n 1)
            if [ -n "$OFFLINE_RUN" ]; then
                echo "Latest wandb run: $OFFLINE_RUN"
                if [ -f "${OFFLINE_RUN}/files/wandb-summary.json" ]; then
                    echo "WandB summary:"
                    cat "${OFFLINE_RUN}/files/wandb-summary.json" | grep -E "final_|best_"
                fi
            fi
            
            # Copy wandb logs to standard location if they're not already there
            if [ "$dir" != "${MINI_OUTPUT_DIR}/wandb" ]; then
                echo "Copying wandb logs to standard location..."
                mkdir -p "${MINI_OUTPUT_DIR}/wandb"
                cp -r "$dir"/* "${MINI_OUTPUT_DIR}/wandb/"
            fi
        fi
    done
done

echo "Mini regression test completed"

# Show GPU usage
echo "GPU memory usage:"
nvidia-smi --query-gpu=memory.used --format=csv

# Provide instructions for syncing WandB data
echo ""
echo "To sync WandB data to the cloud, run:"
echo "wandb sync ${MINI_OUTPUT_DIR}/wandb/offline-run-*"
