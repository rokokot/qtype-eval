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


# Print environment information
echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"

# Create base output directory for tests
TESTS_OUTPUT="sequential_debug_output"
rm -rf $TESTS_OUTPUT
mkdir -p $TESTS_OUTPUT

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
    
    # Run the experiment
    python -m src.experiments.run_experiment \
        "output_dir=${TEST_DIR}" \
        "experiment_name=${TEST_NAME}" \
        "data.cache_dir=${HF_HOME}" \
        "wandb.mode=disabled" \
        "$@" \
        2>&1 | tee "${TEST_DIR}/run.log"
    
    # Check results
    echo "Results for ${TEST_NAME}:"
    if [ -f "${TEST_DIR}/all_results.json" ]; then
        echo "Success! Found results file."
        cat "${TEST_DIR}/all_results.json" | grep -E "task|task_type|language"
    elif [ -d "${TEST_DIR}/en" ] && [ -f "${TEST_DIR}/en/results.json" ]; then
        echo "Success! Found language-specific results file."
        cat "${TEST_DIR}/en/results.json" | grep -E "task|task_type|language"
    else
        echo "No results file found."
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
done

# Show GPU usage
echo "GPU memory usage:"
nvidia-smi --query-gpu=memory.used --format=csv

echo "All tests completed."
