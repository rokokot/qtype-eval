#!/bin/bash
#SBATCH --job-name=glot500_exp
#SBATCH --time=12:00:00        
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

# IMPORTANT: Disable Hydra's working directory changes
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1

export WANDB_API_KEY="282936b31f3ab3415a24a3dba88151d5f7e5bf10"
export WANDB_ENTITY="rokii-ku-leuven"
export WANDB_PROJECT="multilingual-question-probing"
export WANDB_MODE="offline"

# Log in to wandb
wandb login $WANDB_API_KEY

# Print environment information for debugging
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Active conda env: $CONDA_DEFAULT_ENV"
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "HYDRA_JOB_CHDIR: $HYDRA_JOB_CHDIR"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Create base output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR="${PWD}/outputs/glot500_experiments_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT_DIR
echo "Main output directory: $BASE_OUTPUT_DIR"

# Set up log file
LOG_FILE="${BASE_OUTPUT_DIR}/main_experiment.log"
echo "Starting glot500 experiments at $(date)" > $LOG_FILE
echo "Setting up environment..." | tee -a $LOG_FILE

# Modify individual experiment scripts to use absolute paths
modify_script() {
    local SCRIPT_PATH=$1
    local OUTPUT_DIR=$2
    local SCRIPT_BACKUP="${SCRIPT_PATH}.bak"
    
    # Create backup
    cp $SCRIPT_PATH $SCRIPT_BACKUP
    
    # Update script to use absolute paths and disable Hydra path manipulation
    sed -i "s|export PYTHONPATH=\$PYTHONPATH:\$PWD|export PYTHONPATH=\$PYTHONPATH:${PWD}|g" $SCRIPT_PATH
    sed -i "s|OUTPUT_DIR=\"outputs/.*|OUTPUT_DIR=\"${OUTPUT_DIR}\"|g" $SCRIPT_PATH
    
    # Add Hydra configuration
    if ! grep -q "HYDRA_JOB_CHDIR" $SCRIPT_PATH; then
        sed -i "/^export TRANSFORMERS_OFFLINE=/a export HYDRA_JOB_CHDIR=False\nexport HYDRA_FULL_ERROR=1" $SCRIPT_PATH
    fi
    
    # Add explicit hydra command line overrides
    sed -i "s|python -m src.experiments.run_experiment|python -m src.experiments.run_experiment \"hydra.job.chdir=False\" \"hydra.run.dir=.\"|g" $SCRIPT_PATH
    
    echo "Modified script: $SCRIPT_PATH"
    echo "  - Set output directory: $OUTPUT_DIR"
    echo "  - Disabled Hydra path manipulation"
}

# Function to run experiment with logging
run_experiment_set() {
    local EXP_NAME=$1
    local SCRIPT_PATH=$2
    local EXP_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXP_NAME}"
    mkdir -p $EXP_OUTPUT_DIR
    
    echo "===========================================" | tee -a $LOG_FILE
    echo "Running experiment set: ${EXP_NAME}" | tee -a $LOG_FILE
    echo "Using script: ${SCRIPT_PATH}" | tee -a $LOG_FILE
    echo "Output directory: ${EXP_OUTPUT_DIR}" | tee -a $LOG_FILE
    echo "Start time: $(date)" | tee -a $LOG_FILE
    echo "===========================================" | tee -a $LOG_FILE
    
    # Modify script to use absolute paths
    modify_script $SCRIPT_PATH $EXP_OUTPUT_DIR
    
    # Run the experiment
    bash $SCRIPT_PATH 2>&1 | tee -a "${EXP_OUTPUT_DIR}/run.log"
    
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "Experiment ${EXP_NAME} completed successfully at $(date)" | tee -a $LOG_FILE
    else
        echo "ERROR: Experiment ${EXP_NAME} failed with code $RESULT at $(date)" | tee -a $LOG_FILE
    fi
    
    echo "------------------------------------------" | tee -a $LOG_FILE
    
    # Find and consolidate results files
    echo "Consolidating results files for ${EXP_NAME}..." | tee -a $LOG_FILE
    
    # Look for all results files
    find "${EXP_OUTPUT_DIR}" -name "*results*.json" -o -name "results.json" > "${EXP_OUTPUT_DIR}/results_files.txt"
    
    if [ -s "${EXP_OUTPUT_DIR}/results_files.txt" ]; then
        echo "Found $(wc -l < ${EXP_OUTPUT_DIR}/results_files.txt) results files" | tee -a $LOG_FILE
        
        # Create a summary of results
        echo "{" > "${EXP_OUTPUT_DIR}/consolidated_results.json"
        echo "  \"experiment\": \"${EXP_NAME}\"," >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
        echo "  \"timestamp\": \"$(date +%Y-%m-%d\ %H:%M:%S)\"," >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
        echo "  \"results\": [" >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
        
        first=true
        while read -r result_file; do
            if [ "$first" = true ]; then
                first=false
            else
                echo "    ," >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
            fi
            
            # Extract language and task info
            lang=$(grep -o "\"language\":[^,]*" "$result_file" | head -1 || echo "\"language\":\"unknown\"")
            task=$(grep -o "\"task\":[^,]*" "$result_file" | head -1 || echo "\"task\":\"unknown\"")
            
            echo "    {" >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
            echo "      \"file\": \"${result_file#${BASE_OUTPUT_DIR}/}\"," >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
            echo "      ${lang}," >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
            echo "      ${task}" >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
            echo "    }" >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
        done < "${EXP_OUTPUT_DIR}/results_files.txt"
        
        echo "  ]" >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
        echo "}" >> "${EXP_OUTPUT_DIR}/consolidated_results.json"
    else
        echo "WARNING: No results files found for ${EXP_NAME}" | tee -a $LOG_FILE
    fi
    
    # Sync wandb data after each experiment set
    echo "Syncing wandb data for ${EXP_NAME}..." | tee -a $LOG_FILE
    wandb sync --include-offline
    
    # Restore original script
    cp "${SCRIPT_PATH}.bak" $SCRIPT_PATH
    
    return $RESULT
}

# Run experiments based on arguments
if [ -z "$1" ]; then
    # Run all experiments in sequence
    echo "Running all experiment sets in sequence" | tee -a $LOG_FILE
    
    run_experiment_set "basic" "run_basic_experiments.sh"
    run_experiment_set "control" "run_control_experiments.sh"
    run_experiment_set "submetric" "run_all_submetrics.sh"
    run_experiment_set "cross" "run_all_cross_lingual.sh"
    
    echo "All experiment sets completed at $(date)" | tee -a $LOG_FILE
    
else
    # Run specific experiment
    case "$1" in
        "basic")
            run_experiment_set "basic" "run_basic_experiments.sh"
            ;;
        "control")
            run_experiment_set "control" "run_control_experiments.sh"
            ;;
        "submetric")
            run_experiment_set "submetric" "run_all_submetrics.sh"
            ;;
        "cross")
            run_experiment_set "cross" "run_all_cross_lingual.sh"
            ;;
        *)
            echo "Unknown experiment type: $1" | tee -a $LOG_FILE
            echo "Valid options: basic, control, submetric, cross" | tee -a $LOG_FILE
            exit 1
            ;;
    esac
fi

# Collect and summarize all results
echo "Collecting and summarizing results..." | tee -a $LOG_FILE
python scripts/analysis/analyze_results.py --results-dir $BASE_OUTPUT_DIR --output-dir "${BASE_OUTPUT_DIR}/analysis" --log-to-wandb

echo "Experiment suite completed at $(date)" | tee -a $LOG_FILE

# Provide instructions for finding results
echo "" | tee -a $LOG_FILE
echo "Results are available in: ${BASE_OUTPUT_DIR}" | tee -a $LOG_FILE
echo "Analysis is available in: ${BASE_OUTPUT_DIR}/analysis" | tee -a $LOG_FILE
echo "To sync wandb data to the cloud, run: wandb sync ${BASE_OUTPUT_DIR}/*/wandb/offline-run-*" | tee -a $LOG_FILE
