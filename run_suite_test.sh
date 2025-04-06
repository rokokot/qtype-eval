#!/bin/bash
#SBATCH --job-name=sequential_debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_debug
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

conda activate qtype-eval

export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DEBUG=1  # Enable debug mode (reduces workers for dataloaders)

# Important: Disable Hydra path manipulation
export HYDRA_FULL_ERROR=1
export HYDRA_JOB_CHDIR=False

# Print environment information
echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
echo "HYDRA_JOB_CHDIR=${HYDRA_JOB_CHDIR}"

# Create base output directory for tests with absolute path
TESTS_OUTPUT="${PWD}/sequential_debug_output"
rm -rf $TESTS_OUTPUT
mkdir -p $TESTS_OUTPUT
echo "Test output directory: ${TESTS_OUTPUT} (absolute path)"

# Print environment information
echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Function to run a test and log results
run_test() {
    local TEST_NAME=$1
    shift
    local TEST_DIR="${TESTS_OUTPUT}/${TEST_NAME}"
    
    echo "========================================="
    echo "Running test: ${TEST_NAME}"
    echo "========================================="
    
    mkdir -p "${TEST_DIR}"
    mkdir -p "${TEST_DIR}/en"  # Create language subdirectory explicitly
    
    echo "Created test directory: ${TEST_DIR}"
    echo "Created language directory: ${TEST_DIR}/en"
    
    # Set environment variables to control output paths
    export TEST_OUTPUT_DIR="${TEST_DIR}"
    
    # Run the experiment with absolute output directory path and hydra overrides
    python -m src.experiments.run_experiment \
        "hydra.job.chdir=False" \
        "hydra.run.dir=." \
        "output_dir=${TEST_DIR}" \
        "experiment_name=${TEST_NAME}" \
        "data.cache_dir=${HF_HOME}" \
        "wandb.mode=offline" \
        "$@" \
        2>&1 | tee "${TEST_DIR}/run.log"
    
    # Check results
    echo "Results for ${TEST_NAME}:"
    
    # Try to find results in all possible locations
    all_results=()
    all_results+=("${TEST_DIR}/all_results.json")
    all_results+=("${TEST_DIR}/en/results.json")
    all_results+=("${TEST_DIR}/results.json")
    all_results+=("./outputs/${TEST_NAME}/*/all_results.json")
    all_results+=("./outputs/${TEST_NAME}/*/*/all_results.json")
    all_results+=("./outputs/${TEST_NAME}/*/*/*/all_results.json")
    
    results_found=false
    
    for result_path in "${all_results[@]}"; do
        # Use compgen for wildcard expansion
        for file in $(compgen -G "$result_path" 2>/dev/null || echo ""); do
            if [ -f "$file" ]; then
                echo "Found results file: $file"
                cat "$file" | grep -E "task|task_type|language|metrics" || echo "No matching content found in file"
                results_found=true
                
                # Copy the result to the standard location if it's not already there
                if [ "$file" != "${TEST_DIR}/all_results.json" ] && [ "$file" != "${TEST_DIR}/en/results.json" ]; then
                    echo "Copying result to standard location..."
                    if [[ "$file" == *"/en/results.json" ]]; then
                        mkdir -p "${TEST_DIR}/en"
                        cp "$file" "${TEST_DIR}/en/results.json"
                    else
                        cp "$file" "${TEST_DIR}/all_results.json"
                    fi
                fi
                
                break
            fi
        done
    done
    
    if [ "$results_found" = false ]; then
        echo "No results file found."
        
        # Search recursively for any results files
        echo "Searching recursively for any results files..."
        find . -name "*results*.json" -mtime -1 | sort
    fi
    
    # Check for errors
    ERROR_FILES=$(find ${TEST_DIR} -name "error_*.json" 2>/dev/null)
    if [ -n "$ERROR_FILES" ]; then
        echo "Error files found:"
        for error_file in $ERROR_FILES; do
            echo "Contents of $error_file:"
            cat "$error_file"
        done
    else
        echo "No error files found."
    fi
    
    # Check for wandb logs
    echo "Checking for wandb logs:"
    WANDB_DIRS=()
    WANDB_DIRS+=("${TEST_DIR}/wandb")
    WANDB_DIRS+=("./wandb")
    WANDB_DIRS+=("./outputs/${TEST_NAME}/*/*/wandb")
    
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
                if [ "$dir" != "${TEST_DIR}/wandb" ]; then
                    echo "Copying wandb logs to standard location..."
                    mkdir -p "${TEST_DIR}/wandb"
                    cp -r "$dir"/* "${TEST_DIR}/wandb/"
                fi
            fi
        done
    done
    
    echo "Test ${TEST_NAME} completed."
    echo "----------------------------------------"
}

# STEP 1: Classification Test
echo "STEP 1: RUNNING CLASSIFICATION TEST"
run_test "1_classification_en" \
    "experiment=question_type" \
    "experiment.tasks=question_type" \
    "model=lm_probe" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.languages=[en]" \
    "training.num_epochs=2" \
    "training.batch_size=8"

# Pause to check results
echo "Classification test complete. Press ENTER to continue to regression test..."
read

# STEP 2: Regression Test
echo "STEP 2: RUNNING REGRESSION TEST"
run_test "2_complexity_en" \
    "experiment=complexity" \
    "experiment.tasks=complexity" \
    "model=lm_probe" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.languages=[en]" \
    "training.task_type=regression" \
    "training.num_epochs=2" \
    "training.batch_size=8"

# Pause to check results
echo "Regression test complete. Press ENTER to continue to submetrics test..."
read

# STEP 3: Submetric Test
echo "STEP 3: RUNNING SUBMETRICS TEST"
run_test "3_submetric_en" \
    "experiment=submetrics" \
    "experiment.tasks=single_submetric" \
    "experiment.submetric=avg_links_len" \
    "model=lm_probe" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.languages=[en]" \
    "training.task_type=regression" \
    "training.num_epochs=2" \
    "training.batch_size=8"

# Pause to check results
echo "Submetrics test complete. Press ENTER to continue to control test..."
read

# STEP 4: Control Test
echo "STEP 4: RUNNING CONTROL TEST"
run_test "4_control_en" \
    "experiment=question_type" \
    "experiment.tasks=question_type" \
    "model=lm_probe" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.languages=[en]" \
    "experiment.use_controls=true" \
    "experiment.control_index=1" \
    "training.num_epochs=2" \
    "training.batch_size=8"

# Show overall summary
echo "========================================="
echo "TEST SUMMARY"
echo "========================================="
for test_dir in ${TESTS_OUTPUT}/*; do
    test_name=$(basename $test_dir)
    if [ -f "${test_dir}/all_results.json" ]; then
        echo "${test_name}: SUCCESS (found all_results.json)"
    elif [ -d "${test_dir}/en" ] && [ -f "${test_dir}/en/results.json" ]; then
        echo "${test_name}: SUCCESS (found language-specific results)"
    else
        # Check for error files
        if ls ${test_dir}/error_*.json 1> /dev/null 2>&1; then
            echo "${test_name}: FAILED (errors found)"
        else
            echo "${test_name}: FAILED (no results file, no errors)"
        fi
    fi
    
    # Check for wandb
    if [ -d "${test_dir}/wandb" ]; then
        echo "${test_name}: WandB metrics available"
        
        # Summarize metrics from WandB if available
        OFFLINE_RUN=$(find "${test_dir}/wandb" -name "offline-run-*" -type d | sort | tail -n 1)
        if [ -n "$OFFLINE_RUN" ] && [ -f "${OFFLINE_RUN}/files/wandb-summary.json" ]; then
            echo "  Test metrics summary:"
            grep "final_test" "${OFFLINE_RUN}/files/wandb-summary.json" | sed 's/\"//g' | sed 's/,//g'
        fi
    else
        echo "${test_name}: WandB metrics NOT available"
    fi
done

# Show GPU usage
echo "GPU memory usage:"
nvidia-smi --query-gpu=memory.used --format=csv

echo "All tests completed."

# Provide instructions for syncing WandB data
echo ""
echo "To sync WandB data to the cloud, run:"
echo "wandb sync ${TESTS_OUTPUT}/*/wandb/offline-run-*"