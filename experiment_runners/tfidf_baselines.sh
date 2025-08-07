#!/bin/bash
#SBATCH --job-name=tfidf_baselines
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --account=intro_vsc37132

export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

conda activate qtype-eval
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1

LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
MODELS=("dummy" "logistic" "ridge" "xgboost")
TASKS=("question_type" "complexity")

OUTPUT_BASE_DIR="$VSC_SCRATCH/tfidf_baselines_output"
mkdir -p $OUTPUT_BASE_DIR

RESULTS_TRACKER="${OUTPUT_BASE_DIR}/tfidf_results.csv"
echo "experiment_type,language,task,model,metric,value" > $RESULTS_TRACKER

# Failed experiments tracker
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch $FAILED_LOG

# TF-IDF features directory
FEATURES_DIR="${OUTPUT_BASE_DIR}/tfidf_features"

# Create metrics extractor
cat > ${OUTPUT_BASE_DIR}/extract_tfidf_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os

def extract_metrics(result_file, tracker_file, exp_type, language, task, model):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract test metrics
        test_metrics = data.get('test_metrics', {})
        
        if not test_metrics:
            print(f"Warning: No test metrics found in {result_file}")
            return False
        
        # Append to tracker file
        with open(tracker_file, 'a') as f:
            writer = csv.writer(f)
            for metric, value in test_metrics.items():
                if value is not None:
                    writer.writerow([exp_type, language, task, model, metric, value])
        return True
        
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: extract_tfidf_metrics.py <result_file> <tracker_file> <exp_type> <language> <task> <model>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    tracker_file = sys.argv[2] 
    exp_type = sys.argv[3]
    language = sys.argv[4]
    task = sys.argv[5]
    model = sys.argv[6]
    
    if extract_metrics(result_file, tracker_file, exp_type, language, task, model):
        print(f"Successfully extracted metrics from {result_file}")
    else:
        print(f"Failed to extract metrics from {result_file}")
EOF

chmod +x ${OUTPUT_BASE_DIR}/extract_tfidf_metrics.py

# Function to run TF-IDF experiment
run_tfidf_experiment() {
    local LANG=$1
    local TASK=$2
    local MODEL=$3
    local MAX_RETRIES=2
    
    local EXPERIMENT_NAME="tfidf_${MODEL}_${TASK}_${LANG}"
    local TASK_TYPE="classification"
    if [ "$TASK" == "complexity" ]; then
        TASK_TYPE="regression"
    fi
    
    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TASK}/${MODEL}/${LANG}"
    mkdir -p "$OUTPUT_DIR"
    
    # Skip if already completed successfully
    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo "Experiment ${EXPERIMENT_NAME} already completed. Extracting metrics..."
        python3 ${OUTPUT_BASE_DIR}/extract_tfidf_metrics.py \
            "${OUTPUT_DIR}/results.json" "$RESULTS_TRACKER" "tfidf" "$LANG" "$TASK" "$MODEL"
        return 0
    fi
    
    # Skip if failed too many times
    local FAIL_COUNT=$(grep -c "^${EXPERIMENT_NAME}$" $FAILED_LOG)
    if [ $FAIL_COUNT -ge $MAX_RETRIES ]; then
        echo "Experiment ${EXPERIMENT_NAME} has failed $FAIL_COUNT times. Skipping."
        return 1
    fi
    
    echo "Running experiment: ${EXPERIMENT_NAME}"
    
    # Build command
    COMMAND="python scripts/run_tfidf_experiments.py \
        experiment=tfidf_baselines \
        data.languages=[${LANG}] \
        model.model_type=${MODEL} \
        experiment.tasks=${TASK} \
        training.task_type=${TASK_TYPE} \
        output_dir=${OUTPUT_DIR} \
        tfidf.features_dir=${FEATURES_DIR} \
        experiment_name=${EXPERIMENT_NAME}"
    
    echo "Command: $COMMAND"
    
    # Execute experiment
    eval $COMMAND
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully"
        
        # Extract metrics
        RESULTS_FILE="${OUTPUT_DIR}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 ${OUTPUT_BASE_DIR}/extract_tfidf_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "tfidf" "$LANG" "$TASK" "$MODEL"
        else
            echo "Warning: Results file not found: $RESULTS_FILE"
            echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        fi
        
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}"
        echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        return 1
    fi
}

# Step 1: Generate TF-IDF features if not exist
echo "Checking TF-IDF features..."
if [ ! -f "${FEATURES_DIR}/metadata.json" ]; then
    echo "Generating TF-IDF features..."
    python scripts/generate_tfidf_glot500.py \
        --output-dir "${FEATURES_DIR}" \
        --max-features 128000 \
        --cache-dir $HF_HOME
    
    if [ $? -ne 0 ]; then
        echo "Failed to generate TF-IDF features. Exiting."
        exit 1
    fi
else
    echo "TF-IDF features already exist."
fi

# Step 2: Run TF-IDF baseline experiments
echo "Starting TF-IDF baseline experiments..."

TOTAL_EXPERIMENTS=0
COMPLETED_EXPERIMENTS=0

for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 1))
            
            echo "========================="
            echo "EXPERIMENT $TOTAL_EXPERIMENTS"
            echo "Language: $LANG, Task: $TASK, Model: $MODEL"
            echo "========================="
            
            if run_tfidf_experiment "$LANG" "$TASK" "$MODEL"; then
                COMPLETED_EXPERIMENTS=$((COMPLETED_EXPERIMENTS + 1))
            fi
        done
    done
done

# List any failed experiments
if [ -f "$FAILED_LOG" ]; then
    FAILED_COUNT=$(wc -l < "$FAILED_LOG")
    if [ $FAILED_COUNT -gt 0 ]; then
        echo "Some experiments failed. See $FAILED_LOG for details."
        echo "Failed experiments ($FAILED_COUNT):"
        cat "$FAILED_LOG"
    else
        echo "All experiments completed successfully!"
    fi
fi

# Print summary statistics
echo "=============================================="
echo "TF-IDF baseline experiments completed!"
echo "Total planned experiments: $TOTAL_EXPERIMENTS"
echo "Successfully completed: $COMPLETED_EXPERIMENTS"
echo "Results saved to: $RESULTS_TRACKER"
echo "=============================================="