#!/bin/bash
#SBATCH --job-name=finetune_experiments
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

conda activate qtype-eval
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

# Run a test experiment first
OUTPUT_TEST_DIR="$VSC_SCRATCH/finetune_test_output"
mkdir -p $OUTPUT_TEST_DIR

python -m src.experiments.run_experiment \
    "hydra.job.chdir=False" \
    "hydra.run.dir=." \
    "experiment=finetune" \
    "experiment.tasks=question_type" \
    "model=glot500_finetune" \
    "model.lm_name=cis-lmu/glot500-base" \
    "model.dropout=0.1" \
    "model.freeze_model=false" \
    "model.finetune=true" \
    "data.languages=[en]" \
    "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
    "training.task_type=classification" \
    "training.num_epochs=2" \
    "training.batch_size=16" \
    "training.lr=2e-5" \
    "+training.debug_mode=true" \
    "experiment_name=test_finetune" \
    "output_dir=${OUTPUT_TEST_DIR}" \
    "wandb.mode=disabled"

if [ $? -ne 0 ]; then
    echo "Test experiment failed. Please check the logs for issues."
    exit 1
fi

echo "Test experiment completed successfully. Proceeding with full experiments."

LANGUAGES=("ar" "en" "fi")          # note
TASKS=("question_type" "complexity")    #note
CONTROL_INDICES=(1 2)                   #note

OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_output"
mkdir -p $OUTPUT_BASE_DIR

RESULTS_TRACKER="${OUTPUT_BASE_DIR}/finetune_results.csv"
echo "experiment_type,language,task,submetric,control_index,metric,value" > $RESULTS_TRACKER

# Create metrics extractor
cat > ${OUTPUT_BASE_DIR}/extract_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os

def extract_metrics(result_file, tracker_file, exp_type, language, task, submetric, control_index):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract test metrics
        test_metrics = data.get('test_metrics', {})
        
        # Validate metrics
        if not test_metrics:
            print(f"Warning: No test metrics found in {result_file}")
            return False
            
        if task == "question_type" and "accuracy" not in test_metrics:
            print(f"Warning: No accuracy metric found for classification task in {result_file}")
        
        if task in ["complexity", "single_submetric"] and "r2" not in test_metrics:
            print(f"Warning: No r2 metric found for regression task in {result_file}")
        
        # Append to tracker file
        with open(tracker_file, 'a') as f:
            writer = csv.writer(f)
            for metric, value in test_metrics.items():
                if value is not None:
                    writer.writerow([
                        exp_type, language, task, 
                        submetric if submetric else '', 
                        control_index if control_index != 'None' else 'None',
                        metric, value
                    ])
        return True
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: extract_metrics.py <result_file> <tracker_file> <exp_type> <language> <task> <submetric> <control_index>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    tracker_file = sys.argv[2]
    exp_type = sys.argv[3]
    language = sys.argv[4]
    task = sys.argv[5]
    submetric = sys.argv[6]
    control_index = sys.argv[7]
    
    if extract_metrics(result_file, tracker_file, exp_type, language, task, submetric, control_index):
        print(f"Successfully extracted metrics from {result_file}")
    else:
        print(f"Failed to extract metrics from {result_file}")
EOF

chmod +x ${OUTPUT_BASE_DIR}/extract_metrics.py

# Function to run finetuning experiment
run_finetune_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    local OUTPUT_SUBDIR=$6
    
    local EXPERIMENT_NAME=""
    local COMMAND=""
    
    # Add debug mode for first experiments
    local DEBUG_PARAM=""
    if [ "$LANG" == "en" ] && [ -z "$CONTROL_IDX" ]; then
        DEBUG_PARAM="+training.debug_mode=true"
    fi
    
    # Set experiment name and command
    if [ -z "$CONTROL_IDX" ]; then
        # Regular experiment
        EXPERIMENT_NAME="finetune_${TASK}_${LANG}"
        COMMAND="python -m src.experiments.run_experiment \
            \"hydra.job.chdir=False\" \
            \"hydra.run.dir=.\" \
            \"experiment=finetune\" \
            \"experiment.tasks=${TASK}\" \
            \"model=glot500_finetune\" \
            \"model.lm_name=cis-lmu/glot500-base\" \
            \"model.dropout=0.1\" \
            \"model.freeze_model=false\" \
            \"model.finetune=true\" \
            \"data.languages=[${LANG}]\" \
            \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
            \"training.task_type=${TASK_TYPE}\" \
            \"training.num_epochs=10\" \
            \"training.batch_size=16\" \
            \"training.lr=2e-5\" \
            \"training.gradient_accumulation_steps=2\" \
            ${DEBUG_PARAM} \
            \"experiment_name=${EXPERIMENT_NAME}\" \
            \"output_dir=${OUTPUT_SUBDIR}\" \
            \"wandb.mode=offline\""
    else
        # Control experiment
        EXPERIMENT_NAME="finetune_${TASK}_control${CONTROL_IDX}_${LANG}"
        COMMAND="python -m src.experiments.run_experiment \
            \"hydra.job.chdir=False\" \
            \"hydra.run.dir=.\" \
            \"experiment=finetune\" \
            \"experiment.tasks=${TASK}\" \
            \"experiment.use_controls=true\" \
            \"experiment.control_index=${CONTROL_IDX}\" \
            \"model=glot500_finetune\" \
            \"model.lm_name=cis-lmu/glot500-base\" \
            \"model.dropout=0.1\" \
            \"model.freeze_model=false\" \
            \"model.finetune=true\" \
            \"data.languages=[${LANG}]\" \
            \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
            \"training.task_type=${TASK_TYPE}\" \
            \"training.num_epochs=10\" \
            \"training.batch_size=16\" \
            \"training.lr=2e-5\" \
            \"training.gradient_accumulation_steps=2\" \
            ${DEBUG_PARAM} \
            \"experiment_name=${EXPERIMENT_NAME}\" \
            \"output_dir=${OUTPUT_SUBDIR}\" \
            \"wandb.mode=offline\""
    fi
    
    # Print the command for debugging
    echo "Running experiment: ${EXPERIMENT_NAME}"
    echo "Command: $COMMAND"
    
    # Execute the experiment
    eval $COMMAND
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully"
        
        # Extract metrics
        RESULTS_FILE="${OUTPUT_SUBDIR}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            CONTROL_PARAM=${CONTROL_IDX:-None}
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "finetune" "$LANG" "$TASK" \
                "${SUBMETRIC:-}" "$CONTROL_PARAM"
                
            # Create a summary
            echo "Experiment: $EXPERIMENT_NAME" > "${OUTPUT_SUBDIR}/experiment_summary.txt"
            if [ -n "$CONTROL_IDX" ]; then
                echo "CONTROL EXPERIMENT (random labels)" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            fi
            echo "Language: $LANG, Task: $TASK" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Results:" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            python -c "import json; f=open('${RESULTS_FILE}'); data=json.load(f); print(json.dumps(data.get('test_metrics', {}), indent=2))" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
        else
            echo "Warning: Results file not found: $RESULTS_FILE"
        fi
        
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}"
        echo "$EXPERIMENT_NAME" >> "${OUTPUT_BASE_DIR}/failed_experiments.log"
        return 1
    fi
}

# Run Main Experiments First

echo "Running main finetuning experiments (non-control)..."

for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" == "complexity" ]; then
            TASK_TYPE="regression"
        fi
        
        # Create output directory
        TASK_DIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}"
        mkdir -p "$TASK_DIR"
        
        # Run standard (non-control) experiment
        run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "" "" "$TASK_DIR"
    done
done

echo "Running control finetuning experiments..."

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
            
            # Run control experiment
            run_finetune_experiment "$TASK_TYPE" "$LANG" "$TASK" "$CONTROL_IDX" "" "$CONTROL_DIR"
        done
    done
done

# Generate summary visualizations
cat > ${OUTPUT_BASE_DIR}/generate_finetune_summaries.py << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def generate_summaries(tracker_file, output_dir):
    print(f"Reading tracker file: {tracker_file}")
    df = pd.read_csv(tracker_file)
    
    # Replace empty strings with NaN for proper handling
    df = df.replace('', np.nan)
    
    print(f"Generating summaries in: {output_dir}")
    
    # Create visualization directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Task Summary
    task_summary = df.pivot_table(
        index=['task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    task_summary.to_csv(os.path.join(output_dir, 'finetune_task_summary.csv'))
    print("Generated finetune_task_summary.csv")
    
    # Language Summary
    language_summary = df.pivot_table(
        index=['language', 'task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    language_summary.to_csv(os.path.join(output_dir, 'finetune_language_summary.csv'))
    print("Generated finetune_language_summary.csv")
    
    # Control vs Non-control Comparison
    # Create a new column indicating control status
    df['is_control'] = df['control_index'].notna() & (df['control_index'] != 'None')
    
    control_comparison = df.pivot_table(
        index=['task', 'is_control'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    control_comparison.to_csv(os.path.join(output_dir, 'finetune_control_comparison.csv'))
    print("Generated finetune_control_comparison.csv")
    
    # Complete experiment matrix
    experiment_matrix = df.pivot_table(
        index=['experiment_type', 'language', 'task', 'control_index'],
        columns='metric',
        values='value'
    )
    experiment_matrix.to_csv(os.path.join(output_dir, 'finetune_experiment_matrix.csv'))
    print("Generated finetune_experiment_matrix.csv")
    
    # Visualization: Task performance by language
    for task in df['task'].unique():
        if pd.isna(task):
            continue
            
        # Get metrics for this task
        task_df = df[df['task'] == task]
        metrics = [col for col in task_df['metric'].unique() if col in ['accuracy', 'f1', 'r2', 'mse']]
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Filter data: non-control entries with this metric
            plot_data = task_df[
                (task_df['metric'] == metric) & 
                (~task_df['is_control'])
            ]
            
            if plot_data.empty:
                plt.close()
                continue
                
            # Create bar chart of performance by language
            lang_perf = plot_data.pivot_table(
                index='language',
                values='value',
                aggfunc='mean'
            )
            
            lang_perf.sort_values('value', ascending=False).plot(
                kind='bar', 
                color='skyblue',
                ylim=(0, 1) if metric in ['accuracy', 'f1', 'r2'] else None
            )
            
            plt.title(f'{task} - {metric} by Language (Finetuned)')
            plt.xlabel('Language')
            plt.ylabel(metric)
            plt.grid(True, linestyle='--', alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(os.path.join(viz_dir, f'finetune_{task}_{metric}_by_language.png'))
            plt.close()
            print(f"Generated finetune_{task}_{metric}_by_language.png")
    
    # Visualization: Control vs non-control performance
    for task in df['task'].unique():
        if pd.isna(task):
            continue
            
        metrics = [col for col in df[df['task'] == task]['metric'].unique() if col in ['accuracy', 'f1', 'r2', 'mse']]
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Filter data
            task_metric_df = df[(df['task'] == task) & (df['metric'] == metric)]
            
            if task_metric_df.empty:
                plt.close()
                continue
                
            # Group by control status and calculate mean
            control_group = task_metric_df.groupby('is_control')['value'].mean()
            
            if len(control_group) < 2:
                plt.close()
                continue
                
            # Plot bar chart
            control_group.plot(
                kind='bar', 
                color=['green', 'red'],
                ylim=(0, 1) if metric in ['accuracy', 'f1', 'r2'] else None
            )
            
            plt.title(f'{task} - {metric}: Control vs Non-Control (Finetuned)')
            plt.xlabel('Is Control')
            plt.ylabel(metric)
            plt.xticks(rotation=0)
            plt.grid(True, linestyle='--', alpha=0.3, axis='y')
            plt.tight_layout()
            
            plt.savefig(os.path.join(viz_dir, f'finetune_{task}_{metric}_control_comparison.png'))
            plt.close()
            print(f"Generated finetune_{task}_{metric}_control_comparison.png")
    
    print("All summary files generated successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_finetune_summaries.py <tracker_file> <output_dir>")
        sys.exit(1)
    
    tracker_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    generate_summaries(tracker_file, output_dir)
EOF

chmod +x ${OUTPUT_BASE_DIR}/generate_finetune_summaries.py

# Generate summary files
python3 ${OUTPUT_BASE_DIR}/generate_finetune_summaries.py "$RESULTS_TRACKER" "$OUTPUT_BASE_DIR"

# Create metadata file
cat > "${OUTPUT_BASE_DIR}/finetune_metadata.json" << EOF
{
  "experiment_type": "fine-tuning",
  "description": "Full fine-tuning of language model for classification and regression tasks",
  "model": "cis-lmu/glot500-base",
  "model_config": "glot500_finetune",
  "languages": $(python -c "import json; print(json.dumps(${LANGUAGES[@]}))"),
  "tasks": $(python -c "import json; print(json.dumps(${TASKS[@]}))"),
  "freeze_model": false,
  "finetune": true,
  "layer_wise": false,
  "control_indices": $(python -c "import json; print(json.dumps(${CONTROL_INDICES[@]}))"),
  "batch_size": 16,
  "gradient_accumulation_steps": 2,
  "effective_batch_size": 32,
  "learning_rate": "2e-5",
  "date_completed": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "total_experiments": $(cat "$RESULTS_TRACKER" | wc -l)
}
EOF

# List any failed experiments
if [ -f "${OUTPUT_BASE_DIR}/failed_experiments.log" ]; then
    echo "Some experiments failed. See ${OUTPUT_BASE_DIR}/failed_experiments.log for details."
    echo "Failed experiments ($(cat "${OUTPUT_BASE_DIR}/failed_experiments.log" | wc -l)):"
    cat "${OUTPUT_BASE_DIR}/failed_experiments.log"
fi

echo "Fine-tuning experiments completed. Results available in: ${OUTPUT_BASE_DIR}"