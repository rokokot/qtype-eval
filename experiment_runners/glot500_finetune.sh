#!/bin/bash
#SBATCH --job-name=finetuning_glot500
#SBATCH --time=00:30:00  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_debug
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the environment
echo "Activating conda environment..."
conda activate qtype-eval

# Install required packages if needed
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install hydra-core hydra-submitit-launcher
pip install "transformers>=4.30.0,<4.36.0" torch datasets wandb

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

# Print environment information
echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "GPU information:"
nvidia-smi

echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Define configuration
LANGUAGES=("ar")
TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "n_tokens")
CONTROL_INDICES=(1)

# Use smaller head for classification, larger for regression
CLASS_HEAD_SIZE=256
REGR_HEAD_SIZE=512

# Base output directory
OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_output"
mkdir -p $OUTPUT_BASE_DIR

# Set up experiment tracking and logging
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/experiment_results.csv"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
mkdir -p $LOG_DIR

# Initialize results tracker if it doesn't exist
if [ ! -f "$RESULTS_TRACKER" ]; then
    echo "language,task,control_index,submetric,status,runtime_seconds,gpu_memory_mb,date_completed" > $RESULTS_TRACKER
fi

# Track failed experiments
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch $FAILED_LOG

# Function to run a finetuning experiment
run_finetune_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    local MAX_RETRIES=2
    
    local EXPERIMENT_TYPE="finetune"
    local TASK_SPEC=$TASK
    local EXPERIMENT_NAME=""
    local OUTPUT_SUBDIR=""
    
    # Set appropriate head size based on task type
    local HEAD_SIZE=$CLASS_HEAD_SIZE
    if [ "$TASK_TYPE" == "regression" ]; then
        HEAD_SIZE=$REGR_HEAD_SIZE
    fi
    
    # Set experiment name and output directory based on parameters
    if [ -n "$SUBMETRIC" ]; then
        # This is a submetric experiment
        TASK_SPEC="single_submetric"
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_${LANG}"
            OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/single_submetric/${LANG}/${SUBMETRIC}"
        else
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_control${CONTROL_IDX}_${LANG}"
            OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/single_submetric/${LANG}/${SUBMETRIC}/control${CONTROL_IDX}"
        fi
    else
        # Regular task experiment
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="finetune_${TASK}_${LANG}"
            OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}"
        else
            EXPERIMENT_NAME="finetune_${TASK}_control${CONTROL_IDX}_${LANG}"
            OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}/control${CONTROL_IDX}"
        fi
    fi
    
    mkdir -p "$OUTPUT_SUBDIR"
    
    # Create log files
    LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"
    ERROR_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.err"
    
    # Skip if already successful
    if [ -f "${OUTPUT_SUBDIR}/results.json" ]; then
        echo "Experiment ${EXPERIMENT_NAME} already completed successfully. Skipping."
        return 0
    fi
    
    # Skip if already failed too many times
    local FAIL_COUNT=$(grep -c "^${EXPERIMENT_NAME}$" $FAILED_LOG)
    if [ $FAIL_COUNT -ge $MAX_RETRIES ]; then
        echo "Experiment ${EXPERIMENT_NAME} has failed $FAIL_COUNT times already. Skipping."
        return 1
    fi
    
    echo "Running experiment: ${EXPERIMENT_NAME}"
    echo "Output directory: ${OUTPUT_SUBDIR}"
    
    # Record start time
    local START_TIME=$(date +%s)
    
    # Build command - now using lm_finetune model type with appropriate head size
    local COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${TASK_SPEC}\" \
        \"experiment.tasks=${TASK_SPEC}\" \
        \"model=lm_finetune\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.dropout=0.1\" \
        \"model.head_hidden_size=${HEAD_SIZE}\" \
        \"model.head_layers=2\" \
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
    echo "Command: $COMMAND" | tee -a $LOG_FILE
    
    # Execute the experiment with error handling
    eval $COMMAND >> $LOG_FILE 2> $ERROR_FILE
    local RESULT=$?
    
    # Calculate runtime
    local END_TIME=$(date +%s)
    local RUNTIME=$((END_TIME - START_TIME))
    
    # Get GPU memory usage if possible
    local GPU_MEM="NA"
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
    fi
    
    # Handle result
    if [ $RESULT -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully in ${RUNTIME} seconds"
        
        # Record success in tracker
        echo "${LANG},${TASK},${CONTROL_IDX:-None},${SUBMETRIC:-None},success,${RUNTIME},${GPU_MEM},$(date +"%Y-%m-%d %H:%M:%S")" >> $RESULTS_TRACKER
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}" | tee -a $ERROR_FILE
        echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        
        # Record failure in tracker
        echo "${LANG},${TASK},${CONTROL_IDX:-None},${SUBMETRIC:-None},failed,${RUNTIME},${GPU_MEM},$(date +"%Y-%m-%d %H:%M:%S")" >> $RESULTS_TRACKER
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 1
    fi
}

# Standard experiments (non-control)
echo "===== Running main finetuning experiments ====="
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
        
        echo "Running standard experiment: $LANG, $TASK"
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" ""
        
        # Wait between experiments
        sleep 30
    done
    
    # Clear GPU memory after each language
    echo "Clearing GPU memory after language $LANG"
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 30  # Longer pause between languages
done

# Control experiments
echo "===== Running control experiments ====="
for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            # Skip control 1 for priority experiments as it's already run
            if [[ " ${PRIORITY_LANGUAGES[@]} " =~ " ${LANG} " ]] && [[ " ${PRIORITY_TASKS[@]} " =~ " ${TASK} " ]] && [ "$CONTROL_IDX" == "1" ]; then
                echo "Skipping already run priority control experiment: $LANG, $TASK, control $CONTROL_IDX"
                continue
            fi
            
            echo "Running control experiment: $LANG, $TASK, control $CONTROL_IDX"
            run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "$CONTROL_IDX" ""
            
            # Wait between experiments
            sleep 30
        done
    done
    
    # Clear GPU memory after each language
    echo "Clearing GPU memory after language $LANG controls"
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 30  # Longer pause between languages
done

# Submetric experiments
echo "===== Running submetric experiments ====="
for LANG in "${LANGUAGES[@]}"; do
    for SUBMETRIC in "${SUBMETRICS[@]}"; do
        # Skip priority submetric that was already run
        if [[ " ${PRIORITY_LANGUAGES[@]} " =~ " ${LANG} " ]] && [ "$SUBMETRIC" == "avg_links_len" ]; then
            echo "Skipping already run priority submetric experiment: $LANG, $SUBMETRIC"
            continue
        fi
        
        echo "Running submetric experiment: $LANG, $SUBMETRIC"
        run_finetune_experiment "regression" "$LANG" "single_submetric" "" "$SUBMETRIC"
        
        # Wait between experiments
        sleep 30
        
        # Run control experiments for submetrics
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            echo "Running submetric control experiment: $LANG, $SUBMETRIC, control $CONTROL_IDX"
            run_finetune_experiment "regression" "$LANG" "single_submetric" "$CONTROL_IDX" "$SUBMETRIC"
            
            # Wait between experiments
            sleep 30
        done
    done
    
    # Clear GPU memory after each language
    echo "Clearing GPU memory after language $LANG submetrics"
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 30  # Longer pause between languages
done

# Generate summary of experiments
echo "===== Generating experiment summary ====="

# Create a Python script to analyze the results
cat > ${OUTPUT_BASE_DIR}/analyze_results.py << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import os
import json
import glob
import sys
from collections import defaultdict

def analyze_results(base_dir, results_csv):
    # Load the experiment tracker
    df = pd.read_csv(results_csv)
    print(f"Total experiments tracked: {len(df)}")
    print(f"Successful experiments: {len(df[df['status'] == 'success'])}")
    print(f"Failed experiments: {len(df[df['status'] == 'failed'])}")
    
    # Analyze runtime
    print("\nRuntime Statistics (successful experiments):")
    runtime_stats = df[df['status'] == 'success']['runtime_seconds'].describe()
    print(f"Mean runtime: {runtime_stats['mean']:.1f} seconds ({runtime_stats['mean']/60:.1f} minutes)")
    print(f"Median runtime: {runtime_stats['50%']:.1f} seconds ({runtime_stats['50%']/60:.1f} minutes)")
    print(f"Max runtime: {runtime_stats['max']:.1f} seconds ({runtime_stats['max']/60:.1f} minutes)")
    
    # Collect performance metrics from result files
    results = defaultdict(list)
    
    # Process regular tasks
    for task in ['question_type', 'complexity']:
        task_dir = os.path.join(base_dir, task)
        if not os.path.exists(task_dir):
            continue
            
        for lang_dir in os.listdir(task_dir):
            lang_path = os.path.join(task_dir, lang_dir)
            if not os.path.isdir(lang_path):
                continue
                
            # Check standard experiment
            result_file = os.path.join(lang_path, 'results.json')
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract metrics based on task
                    if task == 'question_type':
                        metric_name = 'accuracy'
                        metric_value = data.get('test_metrics', {}).get('accuracy')
                    else:  # complexity
                        metric_name = 'r2'
                        metric_value = data.get('test_metrics', {}).get('r2')
                    
                    if metric_value is not None:
                        results[f"{task}_{metric_name}"].append({
                            'language': lang_dir,
                            'control': 'None',
                            'value': metric_value
                        })
                except Exception as e:
                    print(f"Error processing {result_file}: {e}")
            
            # Check control experiments
            for control_dir in glob.glob(os.path.join(lang_path, 'control*')):
                if not os.path.isdir(control_dir):
                    continue
                    
                control_name = os.path.basename(control_dir)
                control_result = os.path.join(control_dir, 'results.json')
                
                if os.path.exists(control_result):
                    try:
                        with open(control_result, 'r') as f:
                            data = json.load(f)
                        
                        # Extract metrics based on task
                        if task == 'question_type':
                            metric_name = 'accuracy'
                            metric_value = data.get('test_metrics', {}).get('accuracy')
                        else:  # complexity
                            metric_name = 'r2'
                            metric_value = data.get('test_metrics', {}).get('r2')
                        
                        if metric_value is not None:
                            results[f"{task}_{metric_name}"].append({
                                'language': lang_dir,
                                'control': control_name,
                                'value': metric_value
                            })
                    except Exception as e:
                        print(f"Error processing {control_result}: {e}")
    
    # Process submetrics
    submetric_dir = os.path.join(base_dir, 'single_submetric')
    if os.path.exists(submetric_dir):
        for lang_dir in os.listdir(submetric_dir):
            lang_path = os.path.join(submetric_dir, lang_dir)
            if not os.path.isdir(lang_path):
                continue
                
            for submetric_dir in os.listdir(lang_path):
                submetric_path = os.path.join(lang_path, submetric_dir)
                if not os.path.isdir(submetric_path):
                    continue
                    
                # Check standard experiment
                result_file = os.path.join(submetric_path, 'results.json')
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                        
                        # Extract metrics
                        metric_name = 'r2'
                        metric_value = data.get('test_metrics', {}).get('r2')
                        
                        if metric_value is not None:
                            results[f"{submetric_dir}_{metric_name}"].append({
                                'language': lang_dir,
                                'control': 'None',
                                'value': metric_value
                            })
                    except Exception as e:
                        print(f"Error processing {result_file}: {e}")
                
                # Check control experiments
                for control_dir in glob.glob(os.path.join(submetric_path, 'control*')):
                    if not os.path.isdir(control_dir):
                        continue
                        
                    control_name = os.path.basename(control_dir)
                    control_result = os.path.join(control_dir, 'results.json')
                    
                    if os.path.exists(control_result):
                        try:
                            with open(control_result, 'r') as f:
                                data = json.load(f)
                            
                            # Extract metrics
                            metric_name = 'r2'
                            metric_value = data.get('test_metrics', {}).get('r2')
                            
                            if metric_value is not None:
                                results[f"{submetric_dir}_{metric_name}"].append({
                                    'language': lang_dir,
                                    'control': control_name,
                                    'value': metric_value
                                })
                        except Exception as e:
                            print(f"Error processing {control_result}: {e}")
    
    # Calculate statistics and generate summary
    print("\nPerformance Metrics Summary:")
    summary_rows = []
    
    for metric_key, values in results.items():
        if not values:
            continue
            
        # Convert to DataFrame for easier analysis
        metric_df = pd.DataFrame(values)
        
        # Calculate statistics for non-control experiments
        std_df = metric_df[metric_df['control'] == 'None']
        if not std_df.empty:
            avg_value = std_df['value'].mean()
            std_dev = std_df['value'].std()
            
            task_name = metric_key.split('_')[0]
            metric_name = '_'.join(metric_key.split('_')[1:])
            
            # Get control statistics
            control_values = {}
            for control in ['control1', 'control2', 'control3']:
                control_df = metric_df[metric_df['control'] == control]
                if not control_df.empty:
                    control_values[control] = control_df['value'].mean()
            
            # Print statistics
            print(f"\n{task_name} - {metric_name}:")
            print(f"  Standard: {avg_value:.4f} ± {std_dev:.4f}")
            
            for control, value in control_values.items():
                print(f"  {control}: {value:.4f}")
            
            # Store for summary CSV
            summary_rows.append({
                'task': task_name,
                'metric': metric_name,
                'standard_avg': avg_value,
                'standard_std': std_dev,
                'control1_avg': control_values.get('control1'),
                'control2_avg': control_values.get('control2'),
                'control3_avg': control_values.get('control3'),
                'num_languages': len(std_df['language'].unique()),
                'num_samples': len(std_df)
            })
    
    # Create summary CSV
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = os.path.join(base_dir, 'performance_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_results.py <base_dir> <results_csv>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    results_csv = sys.argv[2]
    
    analyze_results(base_dir, results_csv)
EOF

chmod +x ${OUTPUT_BASE_DIR}/analyze_results.py

# Run analysis
python ${OUTPUT_BASE_DIR}/analyze_results.py ${OUTPUT_BASE_DIR} ${RESULTS_TRACKER}

echo "All finetuning experiments completed"
echo "Results can be found in ${OUTPUT_BASE_DIR}"
echo "Summary reports are in ${OUTPUT_BASE_DIR}/performance_summary.csv"