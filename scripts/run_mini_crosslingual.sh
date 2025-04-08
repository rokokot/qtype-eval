#!/bin/bash
#SBATCH --job-name=mini_crosslingual_debug
#SBATCH --time=00:20:00
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

# Ensure necessary packages are installed
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install required Python packages
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
MINI_OUTPUT_DIR="${PWD}/mini_crosslingual_output"
rm -rf $MINI_OUTPUT_DIR
mkdir -p $MINI_OUTPUT_DIR
echo "Output directory: ${MINI_OUTPUT_DIR} (absolute path)"

# Print environment information
echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Define source and target languages for testing
SRC_LANG="en"
TGT_LANG="ja"

# Run mini cross-lingual experiment for question type classification
echo "Running mini cross-lingual experiment for question type classification..."
echo "Source language: ${SRC_LANG}, Target language: ${TGT_LANG}"

# Create subdirectory
CLASS_DIR="${MINI_OUTPUT_DIR}/question_type_${SRC_LANG}_to_${TGT_LANG}"
mkdir -p "${CLASS_DIR}"

# Run question type classification experiment
# Simply add +experiment.use_controls=false to set the missing parameter
python -m src.experiments.run_experiment \
    "hydra.job.chdir=False" \
    "hydra.run.dir=." \
    "experiment=cross_lingual" \
    "experiment.tasks=question_type" \
    "experiment.cross_lingual=true" \
    "+experiment.use_controls=false" \
    "model=lm_probe" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.train_language=${SRC_LANG}" \
    "data.eval_language=${TGT_LANG}" \
    "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
    "training.task_type=classification" \
    "training.num_epochs=5" \
    "training.batch_size=16" \
    "experiment_name=mini_crosslingual_question_type_${SRC_LANG}_to_${TGT_LANG}" \
    "output_dir=${CLASS_DIR}" \
    "wandb.mode=offline"

echo "Question type cross-lingual experiment completed with status: $?"

# Run mini cross-lingual experiment for complexity regression
echo "Running mini cross-lingual experiment for complexity regression..."
echo "Source language: ${SRC_LANG}, Target language: ${TGT_LANG}"

# Create subdirectory
REG_DIR="${MINI_OUTPUT_DIR}/complexity_${SRC_LANG}_to_${TGT_LANG}"
mkdir -p "${REG_DIR}"

# Run complexity regression experiment
# Simply add +experiment.use_controls=false to set the missing parameter
python -m src.experiments.run_experiment \
    "hydra.job.chdir=False" \
    "hydra.run.dir=." \
    "experiment=cross_lingual" \
    "experiment.tasks=complexity" \
    "experiment.cross_lingual=true" \
    "+experiment.use_controls=false" \
    "model=lm_probe" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.train_language=${SRC_LANG}" \
    "data.eval_language=${TGT_LANG}" \
    "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
    "training.task_type=regression" \
    "training.num_epochs=5" \
    "training.batch_size=16" \
    "experiment_name=mini_crosslingual_complexity_${SRC_LANG}_to_${TGT_LANG}" \
    "output_dir=${REG_DIR}" \
    "wandb.mode=offline"

echo "Complexity regression cross-lingual experiment completed with status: $?"

# Function to find and check results
check_results() {
    local DIR=$1
    local TASK=$2
    echo "Checking results for $TASK cross-lingual experiment..."
    
    echo "Output directory contents for $TASK:"
    ls -la $DIR
    
    # Look for results file
    local RESULTS_FILE="${DIR}/cross_lingual_results.json"
    
    if [ -f "$RESULTS_FILE" ]; then
        echo "Found results file: $RESULTS_FILE"
        echo "Results content:"
        cat "$RESULTS_FILE" | grep -E "task|task_type|train_language|eval_language|metrics" || echo "No matching content found in file"
    else
        echo "No results file found at expected location: $RESULTS_FILE"
        
        # Search recursively for any JSON files
        echo "Searching recursively for any JSON files..."
        find $DIR -name "*.json" -mtime -1 -not -path "*/\.git/*" | sort
        
        # If found, display their content
        JSON_FILES=$(find $DIR -name "*.json" -mtime -1 -not -path "*/\.git/*")
        if [ -n "$JSON_FILES" ]; then
            for file in $JSON_FILES; do
                echo "Content of $file:"
                cat "$file" | grep -E "task|task_type|train_language|eval_language|metrics" || echo "No matching content found in $file"
            done
        fi
    fi
    
    # Check for error files
    local ERROR_FILES=$(find $DIR -name "error_*.json" 2>/dev/null)
    if [ -n "$ERROR_FILES" ]; then
        echo "Error files found for $TASK:"
        for error_file in $ERROR_FILES; do
            echo "Contents of $error_file:"
            cat "$error_file"
        done
    else 
        echo "No error files found for $TASK"
    fi
    
    # Check for wandb logs
    echo "Checking for wandb logs for $TASK:"
    local WANDB_DIR="${DIR}/wandb"
    if [ -d "$WANDB_DIR" ]; then
        echo "Found wandb directory: $WANDB_DIR"
        ls -la "$WANDB_DIR"
        
        # Try to get the last run ID
        local OFFLINE_RUN=$(find "$WANDB_DIR" -name "offline-run-*" -type d | sort | tail -n 1)
        if [ -n "$OFFLINE_RUN" ]; then
            echo "Latest wandb run: $OFFLINE_RUN"
            if [ -f "${OFFLINE_RUN}/files/wandb-summary.json" ]; then
                echo "WandB summary:"
                cat "${OFFLINE_RUN}/files/wandb-summary.json" | grep -E "final_|best_"
            fi
        fi
    else
        echo "No wandb directory found at: $WANDB_DIR"
        
        # Look for wandb directory in subdirectories
        WANDB_DIRS=$(find $DIR -name "wandb" -type d)
        if [ -n "$WANDB_DIRS" ]; then
            for wandb_dir in $WANDB_DIRS; do
                echo "Found wandb directory: $wandb_dir"
                ls -la "$wandb_dir"
            done
        else
            echo "No wandb directories found under $DIR"
        fi
    fi
}

# Check results for both experiments
check_results "$CLASS_DIR" "question_type"
check_results "$REG_DIR" "complexity"

# Show GPU usage
echo "GPU memory usage at the end of experiments:"
nvidia-smi --query-gpu=memory.used --format=csv

# Collect all results into a single file for easier reference
echo "Collecting all results into a single summary file..."

cat > ${MINI_OUTPUT_DIR}/summary.md <<EOL
# Cross-Lingual Experiment Summary

## Configuration
- Source Language: ${SRC_LANG}
- Target Language: ${TGT_LANG}
- Model: cis-lmu/glot500-base
- Epochs: 5
- Batch Size: 16

## Results

### Question Type Classification (${SRC_LANG} → ${TGT_LANG})
EOL

if [ -f "${CLASS_DIR}/cross_lingual_results.json" ]; then
    echo "\`\`\`" >> ${MINI_OUTPUT_DIR}/summary.md
    cat "${CLASS_DIR}/cross_lingual_results.json" >> ${MINI_OUTPUT_DIR}/summary.md
    echo "\`\`\`" >> ${MINI_OUTPUT_DIR}/summary.md
else
    echo "No results file found" >> ${MINI_OUTPUT_DIR}/summary.md
fi

cat >> ${MINI_OUTPUT_DIR}/summary.md <<EOL

### Complexity Regression (${SRC_LANG} → ${TGT_LANG})
EOL

if [ -f "${REG_DIR}/cross_lingual_results.json" ]; then
    echo "\`\`\`" >> ${MINI_OUTPUT_DIR}/summary.md
    cat "${REG_DIR}/cross_lingual_results.json" >> ${MINI_OUTPUT_DIR}/summary.md
    echo "\`\`\`" >> ${MINI_OUTPUT_DIR}/summary.md
else
    echo "No results file found" >> ${MINI_OUTPUT_DIR}/summary.md
fi

echo "Summary file created: ${MINI_OUTPUT_DIR}/summary.md"
echo "Mini cross-lingual experiments completed."

# Provide instructions for syncing WandB data
echo ""
echo "To sync WandB data to the cloud, run:"
echo "wandb sync ${MINI_OUTPUT_DIR}/*/wandb/offline-run-*"