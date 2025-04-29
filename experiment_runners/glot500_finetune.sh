#!/bin/bash
#SBATCH --job-name=finetune_experiments
#SBATCH --time=00:30:00  # 48 hours allocation for comprehensive experiments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_debug  # A100 is more powerful for this workload
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# ===== Environment setup =====
export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate conda environment
echo "Activating conda environment..."
conda activate qtype-eval || { echo "Failed to activate conda environment"; exit 1; }

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

# Print system information
echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
echo "GPU Information:"
nvidia-smi

# ===== Directory and file setup =====
OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_output"
mkdir -p "$OUTPUT_BASE_DIR"
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/finetune_results.csv"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
mkdir -p "$LOG_DIR"

# Initialize results tracker if it doesn't exist
if [ ! -f "$RESULTS_TRACKER" ]; then
    echo "experiment_type,language,task,submetric,control_index,metric,value" > "$RESULTS_TRACKER"
    echo "Created new results tracker at $RESULTS_TRACKER"
else
    echo "Using existing results tracker at $RESULTS_TRACKER"
fi

# Track failed experiments for retry logic
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch "$FAILED_LOG"

# Create metrics extractor script for processing results
cat > "${OUTPUT_BASE_DIR}/extract_metrics.py" << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os

def extract_metrics(result_file, tracker_file, exp_type, language, task, submetric=None, control_index="None"):
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
                    writer.writerow([
                        exp_type, language, task, 
                        submetric or "None",
                        control_index,
                        metric, value
                    ])
        return True
    except Exception as e:
        print(f"Error extracting metrics from {result_file}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: extract_metrics.py <result_file> <tracker_file> <exp_type> <language> <task> [<submetric>] [<control_index>]")
        sys.exit(1)
    
    result_file = sys.argv[1]
    tracker_file = sys.argv[2]
    exp_type = sys.argv[3]
    language = sys.argv[4]
    task = sys.argv[5]
    submetric = sys.argv[6] if len(sys.argv) > 6 else None
    control_index = sys.argv[7] if len(sys.argv) > 7 else "None"
    
    if extract_metrics(result_file, tracker_file, exp_type, language, task, submetric, control_index):
        print(f"Successfully extracted metrics from {result_file}")
    else:
        print(f"Failed to extract metrics from {result_file}")
EOF

chmod +x "${OUTPUT_BASE_DIR}/extract_metrics.py"

# ===== Configuration =====
LANGUAGES=("en" "ar" "fi" "id" "ja" "ko" "ru")
TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
CONTROL_INDICES=(1 2 3)
MAX_RETRIES=2

# ===== Function to run finetuning experiment =====
run_finetune_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    local OUTPUT_SUBDIR=$6
    
    local EXPERIMENT_NAME=""
    local COMMAND=""
    local EXPERIMENT_TYPE="finetune"
    
    # Set experiment name based on parameters
    if [ -n "$SUBMETRIC" ]; then
        # This is a submetric experiment
        TASK="single_submetric"
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_${LANG}"
        else
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_control${CONTROL_IDX}_${LANG}"
        fi
    else
        # Regular task experiment
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${TASK}_${LANG}"
        else
            EXPERIMENT_NAME="finetune_${TASK}_control${CONTROL_IDX}_${LANG}"
        fi
    fi
    
    # Create log files
    LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"
    ERROR_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.err"
    
    # Skip if already successful
    if [ -f "${OUTPUT_SUBDIR}/results.json" ]; then
        echo "✓ Experiment ${EXPERIMENT_NAME} already completed successfully. Skipping."
        
        # Extract metrics if not already done
        if [ -f "$RESULTS_TRACKER" ]; then
            if ! grep -q "${EXPERIMENT_NAME}" "$RESULTS_TRACKER"; then
                echo "Extracting metrics for previously completed experiment..."
                python3 "${OUTPUT_BASE_DIR}/extract_metrics.py" \
                    "${OUTPUT_SUBDIR}/results.json" "$RESULTS_TRACKER" "finetune" "$LANG" "$TASK" \
                    "${SUBMETRIC:-}" "${CONTROL_IDX:-None}"
            fi
        fi
        
        return 0
    fi
    
    # Skip if already failed too many times
    local FAIL_COUNT=$(grep -c "^${EXPERIMENT_NAME}$" "$FAILED_LOG")
    if [ $FAIL_COUNT -ge $MAX_RETRIES ]; then
        echo "✗ Experiment ${EXPERIMENT_NAME} has failed $FAIL_COUNT times already. Skipping."
        return 1
    fi
    
    echo "Running experiment: ${EXPERIMENT_NAME}"
    
    # Build command with all necessary parameters
    COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${TASK}\" \
        \"experiment.tasks=${TASK}\" \
        \"model=glot500_finetune\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.dropout=0.1\" \
        \"model.freeze_model=false\" \
        \"model.finetune=true\" \
        \"model.probe_hidden_size=96\" \
        \"data.languages=[${LANG}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=10\" \
        \"training.batch_size=16\" \
        \"training.lr=2e-5\" \
        \"+training.gradient_accumulation_steps=2\" \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${OUTPUT_SUBDIR}\" \
        \"wandb.mode=offline\""
    
    # Add control parameters if needed
    if [ -n "$CONTROL_IDX" ]; then
        COMMAND="$COMMAND \
            \"experiment.use_controls=true\" \
            \"experiment.control_index=${CONTROL_IDX}\""
    fi
    
    # Add submetric parameter if needed
    if [ -n "$SUBMETRIC" ]; then
        COMMAND="$COMMAND \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    # Log the command
    echo "Command: $COMMAND" | tee -a "$LOG_FILE"
    
    # Execute the experiment with error handling
    START_TIME=$(date +%s)
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    
    # Execute the experiment with error handling
    eval $COMMAND >> "$LOG_FILE" 2> "$ERROR_FILE"
    RESULT=$?
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Finished at: $(date), duration: $DURATION seconds" | tee -a "$LOG_FILE"
    
    if [ $RESULT -eq 0 ]; then
        echo "✓ Experiment ${EXPERIMENT_NAME} completed successfully in $DURATION seconds"
        
        # Extract metrics if results file exists
        RESULTS_FILE="${OUTPUT_SUBDIR}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 "${OUTPUT_BASE_DIR}/extract_metrics.py" \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "finetune" "$LANG" "$TASK" \
                "${SUBMETRIC:-}" "${CONTROL_IDX:-None}"
            
            # Create a summary file
            echo "Experiment: $EXPERIMENT_NAME" > "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Language: $LANG, Task: $TASK" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Duration: $DURATION seconds" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Results:" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            cat "$RESULTS_FILE" | grep -E "test_metrics|train_metrics" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
        else
            echo "Warning: Results file not found: $RESULTS_FILE" | tee -a "$ERROR_FILE"
            echo "${EXPERIMENT_NAME}" >> "$FAILED_LOG"
            return 1
        fi
        
        return 0
    else
        echo "✗ Error in experiment ${EXPERIMENT_NAME}" | tee -a "$ERROR_FILE"
        echo "${EXPERIMENT_NAME}" >> "$FAILED_LOG"
        
        # Record specific errors
        ERROR_DESC=$(grep -A 5 "Error" "$ERROR_FILE" | head -n 6)
        echo "Error details: ${ERROR_DESC}" | tee -a "$ERROR_FILE"
        
        # Clean GPU memory explicitly
        echo "Clearing GPU memory after failed experiment..." | tee -a "$LOG_FILE"
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 1
    fi
}

# ===== Run priority experiments =====
echo "===== Running priority experiments ====="
PRIORITY_LANGUAGES=("en")
PRIORITY_TASKS=("question_type")

for LANG in "${PRIORITY_LANGUAGES[@]}"; do
    for TASK in "${PRIORITY_TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        # Create output directory
        TASK_DIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}"
        mkdir -p "$TASK_DIR"
        
        echo "===== Running priority experiment: $LANG, $TASK ====="
        # Run standard (non-control) experiment
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" "" "$TASK_DIR"
        
        # Wait to let GPU memory clear and cool down
        echo "Waiting 30 seconds before next experiment..."
        sleep 30
    done
done

# ===== Run main experiments =====
echo "===== Running main finetuning experiments (non-control) ====="

for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        # Skip priority experiments that have already been run
        if [[ " ${PRIORITY_LANGUAGES[@]} " =~ " ${LANG} " ]] && [[ " ${PRIORITY_TASKS[@]} " =~ " ${TASK} " ]]; then
            echo "Skipping already run priority experiment: $LANG, $TASK"
            continue
        fi
        
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        # Create output directory
        TASK_DIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}"
        mkdir -p "$TASK_DIR"
        
        echo "===== Running standard experiment: $LANG, $TASK ====="
        # Run standard (non-control) experiment
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" "" "$TASK_DIR"
        
        # Wait to let GPU memory clear and cool down
        echo "Waiting 30 seconds before next experiment..."
        sleep 30
    done
done

# ===== Run control experiments =====
echo "===== Running control experiments ====="

for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            # Create output directory
            CONTROL_DIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}/control${CONTROL_IDX}"
            mkdir -p "$CONTROL_DIR"
            
            echo "===== Running control experiment: $LANG, $TASK, control $CONTROL_IDX ====="
            # Run control experiment
            run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "$CONTROL_IDX" "" "$CONTROL_DIR"
            
            # Wait to let GPU memory clear and cool down
            echo "Waiting 30 seconds before next experiment..."
            sleep 30
        done
    done
done

# ===== Run submetric experiments =====
echo "===== Running submetric experiments ====="

for LANG in "${LANGUAGES[@]}"; do
    for SUBMETRIC in "${SUBMETRICS[@]}"; do
        # Create output directory
        SUBMETRIC_DIR="${OUTPUT_BASE_DIR}/single_submetric/${LANG}/${SUBMETRIC}"
        mkdir -p "$SUBMETRIC_DIR"
        
        echo "===== Running submetric experiment: $LANG, $SUBMETRIC ====="
        # Run submetric experiment
        run_finetune_experiment "regression" "$LANG" "single_submetric" "" "$SUBMETRIC" "$SUBMETRIC_DIR"
        
        # Wait to let GPU memory clear and cool down
        echo "Waiting 30 seconds before next experiment..."
        sleep 30
        
        # Run control experiments for submetrics
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            # Create output directory
            CONTROL_DIR="${OUTPUT_BASE_DIR}/single_submetric/${LANG}/${SUBMETRIC}/control${CONTROL_IDX}"
            mkdir -p "$CONTROL_DIR"
            
            echo "===== Running submetric control experiment: $LANG, $SUBMETRIC, control $CONTROL_IDX ====="
            # Run submetric control experiment
            run_finetune_experiment "regression" "$LANG" "single_submetric" "$CONTROL_IDX" "$SUBMETRIC" "$CONTROL_DIR"
            
            # Wait to let GPU memory clear and cool down
            echo "Waiting 30 seconds before next experiment..."
            sleep 30
        done
    done
done

# ===== Generate summary reports =====
echo "===== Generating summary reports ====="

# Create summary generator script
cat > "${OUTPUT_BASE_DIR}/generate_summary.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def generate_summaries(tracker_file, output_dir):
    print(f"Reading data from {tracker_file}")
    
    # Load data
    df = pd.read_csv(tracker_file)
    
    # Create summaries directory
    summaries_dir = os.path.join(output_dir, 'summaries')
    os.makedirs(summaries_dir, exist_ok=True)
    
    # Basic stats
    total_experiments = df['language'].count()
    languages = df['language'].unique()
    tasks = df['task'].unique()
    metrics = df['metric'].unique()
    
    # Generate overview report
    with open(os.path.join(summaries_dir, 'overview.txt'), 'w') as f:
        f.write(f"Finetuning Experiments Summary\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total data points: {total_experiments}\n")
        f.write(f"Languages: {', '.join(languages)}\n")
        f.write(f"Tasks: {', '.join(tasks)}\n")
        f.write(f"Metrics: {', '.join(metrics)}\n\n")
        
        # Summary by language
        f.write("Performance by language:\n")
        for lang in languages:
            lang_df = df[df['language'] == lang]
            f.write(f"  {lang}: {len(lang_df)} data points\n")
            
            # For classification task
            class_df = lang_df[lang_df['task'] == 'question_type']
            if not class_df.empty:
                acc = class_df[class_df['metric'] == 'accuracy']['value'].mean()
                f1 = class_df[class_df['metric'] == 'f1']['value'].mean()
                f.write(f"    Question type: avg accuracy={acc:.4f}, avg f1={f1:.4f}\n")
            
            # For regression task
            reg_df = lang_df[lang_df['task'] == 'complexity']
            if not reg_df.empty:
                mse = reg_df[reg_df['metric'] == 'mse']['value'].mean()
                r2 = reg_df[reg_df['metric'] == 'r2']['value'].mean()
                f.write(f"    Complexity: avg mse={mse:.4f}, avg r2={r2:.4f}\n")
    
    # Create plots
    print("Generating visualizations...")
    
    # Plot accuracy by language for question_type task
    plt.figure(figsize=(10, 6))
    acc_df = df[(df['task'] == 'question_type') & (df['metric'] == 'accuracy')]
    
    if not acc_df.empty:
        sns.barplot(x='language', y='value', data=acc_df)
        plt.title('Question Type Classification Accuracy by Language')
        plt.ylabel('Accuracy')
        plt.xlabel('Language')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(summaries_dir, 'question_type_accuracy.png'))
    
    # Plot R2 by language for complexity task
    plt.figure(figsize=(10, 6))
    r2_df = df[(df['task'] == 'complexity') & (df['metric'] == 'r2')]
    
    if not r2_df.empty:
        sns.barplot(x='language', y='value', data=r2_df)
        plt.title('Complexity Regression R² by Language')
        plt.ylabel('R²')
        plt.xlabel('Language')
        plt.savefig(os.path.join(summaries_dir, 'complexity_r2.png'))
    
    # Compare control vs. non-control experiments
    plt.figure(figsize=(12, 6))
    control_df = df[df['control_index'] != 'None'].copy()
    
    if not control_df.empty:
        control_df['is_control'] = 'Control'
        noncontrol_df = df[df['control_index'] == 'None'].copy()
        noncontrol_df['is_control'] = 'Standard'
        
        combined_df = pd.concat([control_df, noncontrol_df])
        
        acc_combined = combined_df[(combined_df['task'] == 'question_type') & 
                                  (combined_df['metric'] == 'accuracy')]
        
        if not acc_combined.empty:
            sns.boxplot(x='language', y='value', hue='is_control', data=acc_combined)
            plt.title('Control vs. Standard Experiments (Question Type Accuracy)')
            plt.ylabel('Accuracy')
            plt.xlabel('Language')
            plt.legend(title='Experiment Type')
            plt.savefig(os.path.join(summaries_dir, 'control_vs_standard.png'))
    
    print(f"Summaries and visualizations saved to {summaries_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_summary.py <tracker_file> <output_dir>")
        sys.exit(1)
    
    tracker_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(tracker_file):
        print(f"Error: Tracker file {tracker_file} not found.")
        sys.exit(1)
    
    generate_summaries(tracker_file, output_dir)
EOF

chmod +x "${OUTPUT_BASE_DIR}/generate_summary.py"

# Run summary generator
python3 "${OUTPUT_BASE_DIR}/generate_summary.py" "$RESULTS_TRACKER" "$OUTPUT_BASE_DIR"

echo "===== All experiments completed ====="
echo "Results saved to $OUTPUT_BASE_DIR"
echo "Results summary available at ${OUTPUT_BASE_DIR}/summaries/"

# Final cleanup
echo "Cleaning up GPU memory..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Calculate total statistics
TOTAL_EXPERIMENTS=$(wc -l < "$RESULTS_TRACKER")
TOTAL_FAILED=$(wc -l < "$FAILED_LOG")
SUCCESS_RATE=$(( (TOTAL_EXPERIMENTS - TOTAL_FAILED) * 100 / TOTAL_EXPERIMENTS ))

echo "===== Final Statistics ====="
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Failed experiments: $TOTAL_FAILED"
echo "Success rate: $SUCCESS_RATE%"

exit 0