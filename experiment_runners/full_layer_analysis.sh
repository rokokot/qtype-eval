#!/bin/bash
#SBATCH --job-name=layerwise_probing
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_a100_debug
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

conda activate qtype-eval
#env variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "GPU information:"
nvidia-smi

echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

LANGUAGES=("ar")  
LAYERS=(1 2 6 11 12)  
MAIN_TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "n_tokens")  # Just 2 key submetrics
CONTROL_INDICES=(1)  # Only  control runs

# Base output directory
OUTPUT_BASE_DIR="$VSC_SCRATCH/layerwise_output"
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/results_tracker.csv"
mkdir -p $OUTPUT_BASE_DIR

# Initialize results tracker with header
echo "experiment_type,language,layer,task,submetric,control_index,metric,value" > $RESULTS_TRACKER

# Create Python script for extracting and saving metrics
cat > ${OUTPUT_BASE_DIR}/extract_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os

def extract_metrics(result_file, tracker_file, exp_type, language, layer, task, submetric, control_index):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract test metrics
        test_metrics = data.get('test_metrics', {})
        
        # Add some basic validation
        if not test_metrics:
            print(f"Warning: No test metrics found in {result_file}")
            return False
            
        # Check for common metrics based on task type
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
                        exp_type, language, layer, task, 
                        submetric if submetric else 'None', 
                        control_index if control_index != 'None' else 'None',
                        metric, value
                    ])
        return True
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: extract_metrics.py <result_file> <tracker_file> <exp_type> <language> <layer> <task> <submetric> <control_index>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    tracker_file = sys.argv[2]
    exp_type = sys.argv[3]
    language = sys.argv[4]
    layer = sys.argv[5]
    task = sys.argv[6]
    submetric = sys.argv[7]
    control_index = sys.argv[8]
    
    if extract_metrics(result_file, tracker_file, exp_type, language, layer, task, submetric, control_index):
        print(f"Successfully extracted metrics from {result_file}")
    else:
        print(f"Failed to extract metrics from {result_file}")
EOF

chmod +x ${OUTPUT_BASE_DIR}/extract_metrics.py

# Function to run a standard experiment
run_standard_experiment() {
    local LANGUAGE=$1
    local LAYER=$2
    local TASK=$3
    local TASK_TYPE=$4
    local SUBMETRIC=$5
    
    local EXPERIMENT_TYPE="question_type"
    local TASK_SPEC="question_type"
    local EXPERIMENT_NAME="layer_${LAYER}_question_type_${LANGUAGE}"
    local OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/question_type"
    
    if [ "$TASK" == "complexity" ]; then
        EXPERIMENT_TYPE="complexity"
        TASK_SPEC="complexity"
        EXPERIMENT_NAME="layer_${LAYER}_complexity_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/complexity"
    
    elif [ -n "$SUBMETRIC" ]; then
        EXPERIMENT_TYPE="submetrics"
        TASK_SPEC="single_submetric"
        EXPERIMENT_NAME="layer_${LAYER}_${SUBMETRIC}_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/${SUBMETRIC}"
    fi
    
    mkdir -p "$OUTPUT_SUBDIR"
    
    echo "Running $TASK_SPEC experiment for language $LANGUAGE, layer $LAYER"
    
    # Add debug mode for the first experiment of each type
    local DEBUG_PARAM=""
    if [ "$LANGUAGE" == "en" ] && [ "$LAYER" == "6" ]; then
        DEBUG_PARAM="+training.debug_mode=true"
    fi
    
    local COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${EXPERIMENT_TYPE}\" \
        \"experiment.tasks=${TASK_SPEC}\" \
        \"model=lm_probe\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.layer_wise=true\" \
        \"model.layer_index=${LAYER}\" \
        \"model.freeze_model=true\" \
        \"+model.probe_hidden_size=96\" \
        \"data.languages=[${LANGUAGE}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=15\" \
        \"training.batch_size=16\" \
        \"training.lr=2e-5\" \
        ${DEBUG_PARAM} \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${OUTPUT_SUBDIR}\" \
        \"wandb.mode=offline\""
    
    if [ -n "$SUBMETRIC" ]; then
        COMMAND+=" \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    # Execute the command
    eval $COMMAND
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Standard experiment completed successfully: $EXPERIMENT_NAME"
        
        # Extract metrics and add to tracker
        RESULTS_FILE="${OUTPUT_SUBDIR}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "standard" "$LANGUAGE" "$LAYER" \
                "$TASK_SPEC" "$SUBMETRIC" "None"
                
            # Create a summary of the experiment
            echo "Experiment: $EXPERIMENT_NAME" > "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Language: $LANGUAGE, Layer: $LAYER, Task: $TASK_SPEC" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Results:" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            python -c "import json; f=open('${RESULTS_FILE}'); data=json.load(f); print(json.dumps(data.get('test_metrics', {}), indent=2))" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
        else
            echo "Warning: Results file not found: $RESULTS_FILE"
        fi
        
        return 0
    else
        echo "Error in standard experiment: $EXPERIMENT_NAME"
        echo "$EXPERIMENT_NAME" >> "${OUTPUT_BASE_DIR}/failed_experiments.log"
        return 1
    fi
}

# Function to run a control experiment
run_control_experiment() {
    local LANGUAGE=$1
    local LAYER=$2
    local TASK=$3
    local TASK_TYPE=$4
    local CONTROL_IDX=$5
    local SUBMETRIC=$6
    
    local EXPERIMENT_TYPE="question_type"
    local TASK_SPEC="question_type"
    local EXPERIMENT_NAME="layer_${LAYER}_question_type_control${CONTROL_IDX}_${LANGUAGE}"
    local OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/question_type/control${CONTROL_IDX}"
    
    if [ "$TASK" == "complexity" ]; then
        EXPERIMENT_TYPE="complexity"
        TASK_SPEC="complexity"
        EXPERIMENT_NAME="layer_${LAYER}_complexity_control${CONTROL_IDX}_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/complexity/control${CONTROL_IDX}"
    
    elif [ -n "$SUBMETRIC" ]; then
        EXPERIMENT_TYPE="submetrics"
        TASK_SPEC="single_submetric"
        EXPERIMENT_NAME="layer_${LAYER}_${SUBMETRIC}_control${CONTROL_IDX}_${LANGUAGE}"
        OUTPUT_SUBDIR="${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}/${SUBMETRIC}/control${CONTROL_IDX}"
    fi
    
    mkdir -p "$OUTPUT_SUBDIR"
    
    echo "Running $TASK_SPEC control experiment for language $LANGUAGE, layer $LAYER, control $CONTROL_IDX"
    
    # Add debug mode for the first control experiment
    local DEBUG_PARAM=""
    if [ "$LANGUAGE" == "en" ] && [ "$LAYER" == "6" ] && [ "$CONTROL_IDX" == "1" ]; then
        DEBUG_PARAM="+training.debug_mode=true"
    fi
    
    local COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${EXPERIMENT_TYPE}\" \
        \"experiment.tasks=${TASK_SPEC}\" \
        \"model=lm_probe\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.layer_wise=true\" \
        \"+model.probe_hidden_size=96\" \
        \"model.layer_index=${LAYER}\" \
        \"model.freeze_model=true\" \
        \"experiment.use_controls=true\" \
        \"experiment.control_index=${CONTROL_IDX}\" \
        \"data.languages=[${LANGUAGE}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=15\" \
        \"training.batch_size=16\" \
        \"training.lr=2e-5\" \
        ${DEBUG_PARAM} \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${OUTPUT_SUBDIR}\" \
        \"wandb.mode=offline\""
    
    if [ -n "$SUBMETRIC" ]; then
        COMMAND+=" \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    # Execute the command
    eval $COMMAND
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Control experiment completed successfully: $EXPERIMENT_NAME"
        
        # Extract metrics and add to tracker
        RESULTS_FILE="${OUTPUT_SUBDIR}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "control" "$LANGUAGE" "$LAYER" \
                "$TASK_SPEC" "$SUBMETRIC" "$CONTROL_IDX"
                
            # Create a summary of the experiment
            echo "Experiment: $EXPERIMENT_NAME (CONTROL)" > "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Language: $LANGUAGE, Layer: $LAYER, Task: $TASK_SPEC, Control: $CONTROL_IDX" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            echo "Results:" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
            python -c "import json; f=open('${RESULTS_FILE}'); data=json.load(f); print(json.dumps(data.get('test_metrics', {}), indent=2))" >> "${OUTPUT_SUBDIR}/experiment_summary.txt"
        else
            echo "Warning: Results file not found: $RESULTS_FILE"
        fi
        
        return 0
    else
        echo "Error in control experiment: $EXPERIMENT_NAME"
        echo "$EXPERIMENT_NAME (CONTROL)" >> "${OUTPUT_BASE_DIR}/failed_experiments.log"
        return 1
    fi
}

# Use a phased approach - first run a few key experiments to validate
echo "Phase 1: Running key validation experiments"

# Run on English first to validate the approach
run_standard_experiment "en" "6" "question_type" "classification"
if [ $? -ne 0 ]; then
    echo "Critical error: Failed to run basic question type experiment on English"
    exit 1
fi

run_standard_experiment "en" "6" "complexity" "regression"
if [ $? -ne 0 ]; then
    echo "Critical error: Failed to run basic complexity experiment on English"
    exit 1
fi

run_control_experiment "en" "6" "question_type" "classification" "1"
if [ $? -ne 0 ]; then
    echo "Critical error: Failed to run control question type experiment on English"
    exit 1
fi

# Create an analysis of the initial results
echo "Initial validation results:"
cat "${OUTPUT_BASE_DIR}/results_tracker.csv"

# Ask for confirmation before continuing with full sweep
echo ""
echo "Initial experiments completed. Please check the results above."
echo "Press Enter to continue with full experiment suite, or Ctrl+C to abort."
read -r

# Phase 2: Main experiments
echo "Phase 2: Running main experiments"

# 1. Main tasks (question_type and complexity)
for LANGUAGE in "${LANGUAGES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        mkdir -p "${OUTPUT_BASE_DIR}/${LANGUAGE}/layer_${LAYER}"
        
        # Run question_type experiments (classification)
        run_standard_experiment "$LANGUAGE" "$LAYER" "question_type" "classification"
        
        # Run complexity experiments (regression)
        run_standard_experiment "$LANGUAGE" "$LAYER" "complexity" "regression"
    done
done

# 2. Submetric tasks
for LANGUAGE in "${LANGUAGES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for SUBMETRIC in "${SUBMETRICS[@]}"; do
            run_standard_experiment "$LANGUAGE" "$LAYER" "" "regression" "$SUBMETRIC"
        done
    done
done

# Control Experiments
echo "Phase 3: Running control experiments"

# 1. Main tasks with controls
for LANGUAGE in "${LANGUAGES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            # Run question_type control experiments
            run_control_experiment "$LANGUAGE" "$LAYER" "question_type" "classification" "$CONTROL_IDX"
            
            # Run complexity control experiments
            run_control_experiment "$LANGUAGE" "$LAYER" "complexity" "regression" "$CONTROL_IDX"
        done
    done
done

# 2. Submetric tasks with controls
for LANGUAGE in "${LANGUAGES[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for SUBMETRIC in "${SUBMETRICS[@]}"; do
            for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
                run_control_experiment "$LANGUAGE" "$LAYER" "" "regression" "$CONTROL_IDX" "$SUBMETRIC"
            done
        done
    done
done

# Generate summary files for easier visualization
echo "Generating summary files..."

# Create Python script for generating summary files
cat > ${OUTPUT_BASE_DIR}/generate_summaries.py << 'EOF'
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
    
    # 1. Layer Summary - average metrics by layer and task
    layer_summary = df.pivot_table(
        index=['layer', 'task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    layer_summary.to_csv(os.path.join(output_dir, 'layer_summary.csv'))
    print("Generated layer_summary.csv")
    
    # 2. Language Summary - average metrics by language and task
    language_summary = df.pivot_table(
        index=['language', 'task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    language_summary.to_csv(os.path.join(output_dir, 'language_summary.csv'))
    print("Generated language_summary.csv")
    
    # 3. Submetric Summary - only for submetric tasks
    submetric_df = df[df['submetric'].notna() & (df['submetric'] != 'None')]
    if not submetric_df.empty:
        submetric_summary = submetric_df.pivot_table(
            index=['submetric', 'layer'], 
            columns='metric', 
            values='value',
            aggfunc='mean'
        )
        submetric_summary.to_csv(os.path.join(output_dir, 'submetric_summary.csv'))
        print("Generated submetric_summary.csv")
    
    # 4. Control vs Non-control Comparison
    # Filter for tasks that have both control and non-control experiments
    tasks_with_controls = df[df['control_index'] != 'None']['task'].unique()
    
    if len(tasks_with_controls) > 0:
        # Create a new column indicating if it's a control experiment
        df['is_control'] = df['control_index'] != 'None'
        
        control_comparison = df[df['task'].isin(tasks_with_controls)].pivot_table(
            index=['task', 'is_control'], 
            columns='metric', 
            values='value',
            aggfunc='mean'
        )
        control_comparison.to_csv(os.path.join(output_dir, 'control_comparison.csv'))
        print("Generated control_comparison.csv")
    
    # 5. Complete matrix for all experiments
    # Reshape the data to have one row per unique experiment
    experiment_matrix = df.pivot_table(
        index=['experiment_type', 'language', 'layer', 'task', 'submetric', 'control_index'],
        columns='metric',
        values='value'
    )
    experiment_matrix.to_csv(os.path.join(output_dir, 'experiment_matrix.csv'))
    print("Generated experiment_matrix.csv")
    
    # 6. Task-specific summaries
    for task in df['task'].dropna().unique():
        if pd.notna(task) and task != 'None':
            task_df = df[df['task'] == task]
            task_summary = task_df.pivot_table(
                index=['language', 'layer'], 
                columns='metric', 
                values='value',
                aggfunc='mean'
            )
            task_summary.to_csv(os.path.join(output_dir, f'{task}_summary.csv'))
            print(f"Generated {task}_summary.csv")
    
    # 7. Generate plots
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Layer performance plot for each task
    for task in df['task'].dropna().unique():
        if pd.notna(task) and task != 'None':
            task_df = df[(df['task'] == task) & (df['control_index'] == 'None')]
            
            if task_df.empty:
                continue
                
            metrics = [col for col in task_df['metric'].unique() if col in ['accuracy', 'f1', 'r2', 'mse']]
            
            for metric in metrics:
                plt.figure(figsize=(12, 8))
                
                metric_df = task_df[task_df['metric'] == metric]
                
                for language in metric_df['language'].unique():
                    lang_data = metric_df[metric_df['language'] == language]
                    lang_data = lang_data.sort_values('layer')
                    
                    plt.plot(lang_data['layer'], lang_data['value'], 'o-', label=language)
                
                plt.title(f'{task} - {metric} by Layer')
                plt.xlabel('Layer')
                plt.ylabel(metric)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{task}_{metric}_by_layer.png'))
                plt.close()
                
                print(f"Generated {task}_{metric}_by_layer.png")
    
    # Control vs non-control performance
    if len(tasks_with_controls) > 0:
        for task in tasks_with_controls:
            metrics = [col for col in df[df['task'] == task]['metric'].unique() if col in ['accuracy', 'f1', 'r2', 'mse']]
            
            for metric in metrics:
                plt.figure(figsize=(10, 6))
                
                # Filter data
                task_metric_df = df[(df['task'] == task) & (df['metric'] == metric)]
                
                # Group by whether it's a control experiment and calculate the mean
                control_group = task_metric_df.groupby('is_control')['value'].mean()
                
                # Plot as a bar chart
                control_group.plot(kind='bar', color=['green', 'red'])
                plt.title(f'{task} - {metric}: Control vs Non-Control')
                plt.xlabel('Is Control')
                plt.ylabel(metric)
                plt.xticks(rotation=0)
                plt.grid(True, linestyle='--', alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{task}_{metric}_control_comparison.png'))
                plt.close()
                
                print(f"Generated {task}_{metric}_control_comparison.png")
    
    print("All summary files generated successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_summaries.py <tracker_file> <output_dir>")
        sys.exit(1)
    
    tracker_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    generate_summaries(tracker_file, output_dir)
EOF

chmod +x ${OUTPUT_BASE_DIR}/generate_summaries.py

# Generate all summary files
python3 ${OUTPUT_BASE_DIR}/generate_summaries.py "$RESULTS_TRACKER" "$OUTPUT_BASE_DIR"

# Create metadata file with experiment details
cat > "${OUTPUT_BASE_DIR}/probing_metadata.json" << EOF
{
  "experiment_type": "layerwise_probing",
  "description": "Layer-wise probing analysis with frozen model representations",
  "model": "cis-lmu/glot500-base",
  "languages": $(python -c "import json; print(json.dumps(${LANGUAGES[@]}))"),
  "layers": $(python -c "import json; print(json.dumps(${LAYERS[@]}))"),
  "tasks": $(python -c "import json; print(json.dumps(${MAIN_TASKS[@]}))"),
  "submetrics": $(python -c "import json; print(json.dumps(${SUBMETRICS[@]}))"),
  "control_indices": $(python -c "import json; print(json.dumps(${CONTROL_INDICES[@]}))"),
  "freeze_model": true,
  "layer_wise": true,
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

echo "Layer-wise analysis completed"
echo "Results are available in ${OUTPUT_BASE_DIR}"
echo "Summary files have been generated for easier visualization"