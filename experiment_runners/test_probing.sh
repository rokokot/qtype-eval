#!/bin/bash
#SBATCH --job-name=layer_probe_experiments
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

LANGUAGES=("ja")
TASKS=("question_type")
SUBMETRICS=()
CONTROL_INDICES=()

LAYER_INDICES=(3 4 5 6 7 8 9 10 11 12)

# Base directory for outputs
OUTPUT_BASE_DIR="$VSC_SCRATCH/layer_probe_output"
mkdir -p $OUTPUT_BASE_DIR

# CSV file to track results across all layers
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/layer_probe_results.csv"
echo "experiment_type,language,task,submetric,control_index,layer_index,metric,value" > $RESULTS_TRACKER

# Failed experiments tracker
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch $FAILED_LOG

# Create metrics extractor with layer information
cat > ${OUTPUT_BASE_DIR}/extract_layer_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os

def extract_metrics(result_file, tracker_file, exp_type, language, task, submetric, control_index, layer_index):
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
        
        # Append to tracker file with layer information
        with open(tracker_file, 'a') as f:
            writer = csv.writer(f)
            for metric, value in test_metrics.items():
                if value is not None:
                    writer.writerow([
                        exp_type, language, task, 
                        submetric if submetric else 'None', 
                        control_index if control_index != 'None' else 'None',
                        layer_index,
                        metric, value
                    ])
        return True
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: extract_layer_metrics.py <result_file> <tracker_file> <exp_type> <language> <task> <submetric> <control_index> <layer_index>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    tracker_file = sys.argv[2]
    exp_type = sys.argv[3]
    language = sys.argv[4]
    task = sys.argv[5]
    submetric = sys.argv[6]
    control_index = sys.argv[7]
    layer_index = sys.argv[8]
    
    if extract_metrics(result_file, tracker_file, exp_type, language, task, submetric, control_index, layer_index):
        print(f"Successfully extracted metrics from {result_file} for layer {layer_index}")
    else:
        print(f"Failed to extract metrics from {result_file} for layer {layer_index}")
EOF

chmod +x ${OUTPUT_BASE_DIR}/extract_layer_metrics.py

# Create a script to create layer analysis visualizations
cat > ${OUTPUT_BASE_DIR}/analyze_layers.py << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
import numpy as np
import seaborn as sns

def analyze_layer_performance(csv_file, output_dir):
    """Analyze layer-wise performance for each task and language"""
    print(f"Reading data from {csv_file}")
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print("No data found in CSV file")
            return
            
        print(f"Loaded {len(df)} rows of data")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create output directory for plots
        plots_dir = os.path.join(output_dir, "layer_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create a combined results dictionary
        combined_results = {}
        
        # Group by task type
        tasks = df['task'].unique()
        for task in tasks:
            task_df = df[df['task'] == task]
            primary_metric = 'accuracy' if task == 'question_type' else 'r2'
            
            # Create a plot directory for this task
            task_dir = os.path.join(plots_dir, task)
            os.makedirs(task_dir, exist_ok=True)
            
            # For each language, plot layer performance
            languages = task_df['language'].unique()
            
            task_results = {}
            
            for language in languages:
                lang_df = task_df[task_df['language'] == language]
                
                # Create a figure for this language/task
                plt.figure(figsize=(12, 8))
                
                # For regular vs control conditions
                control_values = lang_df['control_index'].unique()
                language_results = {}
                
                for control_value in control_values:
                    if pd.isna(control_value):
                        control_label = "regular"
                        control_condition = lang_df['control_index'].isna()
                    else:
                        control_label = f"control-{control_value}"
                        control_condition = lang_df['control_index'] == control_value
                    
                    control_df = lang_df[control_condition]
                    
                    if control_df.empty:
                        continue
                        
                    # Extract layers and metric
                    layers = control_df['layer_index'].astype(int).values
                    metric_values = control_df[control_df['metric'] == primary_metric]['value'].values
                    
                    if len(layers) == 0 or len(metric_values) == 0:
                        continue
                        
                    # Plot
                    plt.plot(layers, metric_values, 'o-', label=f"{control_label}")
                    
                    # Store for combined results
                    language_results[control_label] = {
                        'layers': layers.tolist(),
                        'values': metric_values.tolist(),
                        'best_layer': int(layers[np.argmax(metric_values)]),
                        'best_value': float(np.max(metric_values))
                    }
                
                plt.title(f"{task.capitalize()} - {language} - {primary_metric.upper()} by Layer")
                plt.xlabel("Layer Index")
                plt.ylabel(primary_metric.capitalize())
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Add a horizontal line for 0 if this is a regression task
                if primary_metric == 'r2':
                    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(task_dir, f"{language}_{task}_{primary_metric}_by_layer.png"))
                plt.close()
                
                task_results[language] = language_results
            
            combined_results[task] = task_results
        
        # Save the combined results as JSON
        with open(os.path.join(output_dir, 'layer_performance_summary.json'), 'w') as f:
            json.dump(combined_results, f, indent=2)

        # Generate a heatmap showing performance across all layers and languages
        for task in tasks:
            primary_metric = 'accuracy' if task == 'question_type' else 'r2'
            
            # Filter data for this task and metric
            task_metric_df = df[(df['task'] == task) & (df['metric'] == primary_metric) & df['control_index'].isna()]
            
            if task_metric_df.empty:
                continue
                
            # Create a pivot table for the heatmap
            heatmap_data = task_metric_df.pivot_table(
                index='language', 
                columns='layer_index', 
                values='value',
                aggfunc='first'  # Take the first value
            )
            
            # Plot heatmap
            plt.figure(figsize=(14, 8))
            ax = sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".3f")
            plt.title(f"{task.capitalize()} - {primary_metric.upper()} by Layer and Language")
            plt.xlabel("Layer Index")
            plt.ylabel("Language")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{task}_{primary_metric}_layer_language_heatmap.png"))
            plt.close()
            
        print(f"Analysis complete. Results saved to {output_dir}")
        return combined_results
            
    except Exception as e:
        print(f"Error analyzing layer performance: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: analyze_layers.py <results_csv> <output_dir>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    analyze_layer_performance(csv_file, output_dir)
EOF

chmod +x ${OUTPUT_BASE_DIR}/analyze_layers.py

# Function to run probe experiment for a specific layer
run_layer_probe_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    local OUTPUT_SUBDIR=$6
    local LAYER_INDEX=$7
    local MAX_RETRIES=2
    
    local EXPERIMENT_NAME=""
    local COMMAND=""
    local EXPERIMENT_TYPE="layer_probe"
    
    # Set experiment name based on parameters
    if [ -n "$SUBMETRIC" ]; then
        # This is a submetric experiment
        TASK="single_submetric"
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="layer${LAYER_INDEX}_probe_${SUBMETRIC}_${LANG}"
        else
            EXPERIMENT_NAME="layer${LAYER_INDEX}_probe_${SUBMETRIC}_control${CONTROL_IDX}_${LANG}"
        fi
    else
        # Regular task experiment
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="layer${LAYER_INDEX}_probe_${TASK}_${LANG}"
        else
            EXPERIMENT_NAME="layer${LAYER_INDEX}_probe_${TASK}_control${CONTROL_IDX}_${LANG}"
        fi
    fi
    
    # Create layer-specific output directory
    local LAYER_OUTPUT_DIR="${OUTPUT_SUBDIR}/layer${LAYER_INDEX}"
    mkdir -p "$LAYER_OUTPUT_DIR"
    
    # Skip if already completed successfully
    if [ -f "${LAYER_OUTPUT_DIR}/${LANG}/results.json" ]; then
        echo "Experiment ${EXPERIMENT_NAME} already completed successfully. Extracting metrics..."
        CONTROL_PARAM=${CONTROL_IDX:-None}
        python3 ${OUTPUT_BASE_DIR}/extract_layer_metrics.py \
            "${LAYER_OUTPUT_DIR}/${LANG}/results.json" "$RESULTS_TRACKER" "layer_probe" "$LANG" "$TASK" \
            "${SUBMETRIC:-}" "$CONTROL_PARAM" "$LAYER_INDEX"
        return 0
    fi
    
    # Skip if already failed too many times
    local FAIL_COUNT=$(grep -c "^${EXPERIMENT_NAME}$" $FAILED_LOG)
    if [ $FAIL_COUNT -ge $MAX_RETRIES ]; then
        echo "Experiment ${EXPERIMENT_NAME} has failed $FAIL_COUNT times already. Skipping."
        return 1
    fi
    
    # Set task-specific configuration
    local PROBE_CONFIG=""
    
    if [ "$TASK_TYPE" == "classification" ]; then
        # Classification probe configuration
        PROBE_CONFIG="\"model.probe_hidden_size=768\" \"model.probe_depth=2\" \"model.dropout=0.2\" \"model.activation=gelu\" \"model.normalization=layer\""
            
        TRAINING_CONFIG="\"training.lr=1e-4\" \"training.patience=3\" \"training.scheduler_factor=0.5\" \"training.scheduler_patience=2\" \"+training.gradient_accumulation_steps=4\""
    else
        # Regression probe configuration
        PROBE_CONFIG="\"model.probe_hidden_size=512\" \"model.probe_depth=3\" \"model.dropout=0.01\" \"model.activation=silu\" \"model.normalization=layer\" \"model.output_standardization=true\""
            
        TRAINING_CONFIG="\"training.lr=2e-5\" \"training.patience=4\" \"training.scheduler_factor=0.5\" \"training.scheduler_patience=2\" \"+training.gradient_accumulation_steps=2\""
    fi
    
    # Build command with explicit layer_index
    COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${TASK}\" \
        \"experiment.tasks=${TASK}\" \
        \"experiment.type=lm_probe\" \
        \"model=lm_probe\" \
        \"model.model_type=lm_probe\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.freeze_model=true\" \
        \"model.layer_wise=true\" \
        \"model.layer_index=${LAYER_INDEX}\" \
        ${PROBE_CONFIG} \
        \"data.languages=[${LANG}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=15\" \
        \"training.batch_size=16\" \
        ${TRAINING_CONFIG} \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${LAYER_OUTPUT_DIR}\" \
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
    
    # Execute the experiment
    echo "Running experiment: ${EXPERIMENT_NAME}"
    eval $COMMAND
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully"
        
        # Extract metrics with layer information
        RESULTS_FILE="${LAYER_OUTPUT_DIR}/${LANG}/results.json"
        if [ -f "$RESULTS_FILE" ]; then
            CONTROL_PARAM=${CONTROL_IDX:-None}
            python3 ${OUTPUT_BASE_DIR}/extract_layer_metrics.py \
                "$RESULTS_FILE" "$RESULTS_TRACKER" "layer_probe" "$LANG" "$TASK" \
                "${SUBMETRIC:-}" "$CONTROL_PARAM" "$LAYER_INDEX"
                
            # Create a summary
            echo "Experiment: $EXPERIMENT_NAME" > "${LAYER_OUTPUT_DIR}/${LANG}/experiment_summary.txt"
            echo "Layer: $LAYER_INDEX" >> "${LAYER_OUTPUT_DIR}/${LANG}/experiment_summary.txt"
            if [ -n "$CONTROL_IDX" ]; then
                echo "CONTROL EXPERIMENT (random labels)" >> "${LAYER_OUTPUT_DIR}/${LANG}/experiment_summary.txt"
            fi
            echo "Language: $LANG, Task: $TASK" >> "${LAYER_OUTPUT_DIR}/${LANG}/experiment_summary.txt"
            echo "Results:" >> "${LAYER_OUTPUT_DIR}/${LANG}/experiment_summary.txt"
            python -c "import json; f=open('${RESULTS_FILE}'); data=json.load(f); print(json.dumps(data.get('test_metrics', {}), indent=2))" >> "${LAYER_OUTPUT_DIR}/${LANG}/experiment_summary.txt"
        else
            echo "Warning: Results file not found: $RESULTS_FILE"
            echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        fi
        
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}"
        echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 1
    fi
}

# Print info about the layers
echo "Running probing experiments for the following layers:"
for LAYER_IDX in "${LAYER_INDICES[@]}"; do
    if [ "$LAYER_IDX" -eq 0 ]; then
        echo "  Layer $LAYER_IDX (Embedding layer / lexical features)"
    elif [ "$LAYER_IDX" -eq "${LAYER_INDICES[-1]}" ]; then
        echo "  Layer $LAYER_IDX (Final layer / task-specific features)"
    else
        echo "  Layer $LAYER_IDX (Intermediate representation)"
    fi
done
echo "Total of ${#LAYER_INDICES[@]} layers to probe"

# Run Main Experiments for each layer
for LAYER_IDX in "${LAYER_INDICES[@]}"; do
    echo "======================="
    echo "Processing layer ${LAYER_IDX}"
    echo "======================="

    for LANG in "${LANGUAGES[@]}"; do
        for TASK in "${TASKS[@]}"; do
            TASK_TYPE="classification"
            if [ "$TASK" == "complexity" ]; then
                TASK_TYPE="regression"
            fi
            
            # Create output directory
            TASK_DIR="${OUTPUT_BASE_DIR}/${TASK}"
            mkdir -p "$TASK_DIR"
            
            # Run standard (non-control) experiment for this layer
            run_layer_probe_experiment "$TASK_TYPE" "$LANG" "$TASK" "" "" "$TASK_DIR" "$LAYER_IDX"
            
            # Run control experiments for this layer
            for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
                # Create output directory
                CONTROL_DIR="${OUTPUT_BASE_DIR}/${TASK}/control${CONTROL_IDX}"
                mkdir -p "$CONTROL_DIR"
                
                # Run control experiment
                run_layer_probe_experiment "$TASK_TYPE" "$LANG" "$TASK" "$CONTROL_IDX" "" "$CONTROL_DIR" "$LAYER_IDX"
            done
        done
        
        # Run submetric experiments for this layer
        for SUBMETRIC in "${SUBMETRICS[@]}"; do
            # Create output directory
            SUBMETRIC_DIR="${OUTPUT_BASE_DIR}/submetrics/${SUBMETRIC}"
            mkdir -p "$SUBMETRIC_DIR"
            
            # Run standard (non-control) submetric experiment
            run_layer_probe_experiment "regression" "$LANG" "single_submetric" "" "$SUBMETRIC" "$SUBMETRIC_DIR" "$LAYER_IDX"
            
            # Run control submetric experiments for this layer
            for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
                # Create output directory
                CONTROL_DIR="${OUTPUT_BASE_DIR}/submetrics/${SUBMETRIC}/control${CONTROL_IDX}"
                mkdir -p "$CONTROL_DIR"
                
                # Run control submetric experiment
                run_layer_probe_experiment "regression" "$LANG" "single_submetric" "$CONTROL_IDX" "$SUBMETRIC" "$CONTROL_DIR" "$LAYER_IDX"
            done
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

# Run layer analysis
echo "Analyzing layer results..."
python ${OUTPUT_BASE_DIR}/analyze_layers.py "${RESULTS_TRACKER}" "${OUTPUT_BASE_DIR}"

# Print summary statistics
TOTAL_EXPERIMENTS=$((${#LANGUAGES[@]} * (${#TASKS[@]} + ${#SUBMETRICS[@]}) * (1 + ${#CONTROL_INDICES[@]}) * ${#LAYER_INDICES[@]}))
COMPLETED_EXPERIMENTS=$(grep -v "experiment_type" "$RESULTS_TRACKER" | wc -l)

echo "=============================================="
echo "Layer probing experiments completed!"
echo "=============================================="
echo "Total planned experiments: $TOTAL_EXPERIMENTS"
echo "Successfully completed: $COMPLETED_EXPERIMENTS"
echo "Failed experiments: ${FAILED_COUNT:-0}"
echo "Success rate: $(( COMPLETED_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS ))%"
echo "Results available in: ${OUTPUT_BASE_DIR}"
echo "Layer summary: ${OUTPUT_BASE_DIR}/layer_performance_summary.json"
echo "=============================================="