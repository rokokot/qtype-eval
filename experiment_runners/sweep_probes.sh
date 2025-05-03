#!/bin/bash
#SBATCH --job-name=sweep_probes_combined
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16     
#SBATCH --gpus-per-node=1     
#SBATCH --mem-per-cpu=11700M  
#SBATCH --time=03:00:00

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

# Define parameter sweep values
LANGUAGES=("ar" "ru")
TASKS=("complexity")
SUBMETRICS=("avg_links_len" "n_tokens")
LAYERS=(2 6 11)
HIDDEN_SIZES=(128 384)
DROPOUT_RATES=(0.01 0.2)
LEARNING_RATES=(1e-3 1e-4)

# Base directory for outputs
OUTPUT_BASE_DIR="$VSC_SCRATCH/param_sweep_output"
mkdir -p $OUTPUT_BASE_DIR

# Results tracking file
RESULTS_TRACKER="${OUTPUT_BASE_DIR}/param_sweep_results.csv"
echo "experiment_id,language,task,layer_index,hidden_size,dropout,learning_rate,metric,value" > $RESULTS_TRACKER

# Failed experiments tracker
FAILED_LOG="${OUTPUT_BASE_DIR}/failed_experiments.log"
touch $FAILED_LOG

# Create metrics extractor script
cat > ${OUTPUT_BASE_DIR}/extract_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os

def extract_metrics(result_file, tracker_file, experiment_id, language, task, layer_index, hidden_size, dropout, learning_rate):
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
                        experiment_id,
                        language,
                        task,
                        layer_index,
                        hidden_size,
                        dropout,
                        learning_rate,
                        metric,
                        value
                    ])
        return True
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 10:
        print("Usage: extract_metrics.py <result_file> <tracker_file> <experiment_id> <language> <task> <layer_index> <hidden_size> <dropout> <learning_rate>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    tracker_file = sys.argv[2]
    experiment_id = sys.argv[3]
    language = sys.argv[4]
    task = sys.argv[5]
    layer_index = sys.argv[6]
    hidden_size = sys.argv[7]
    dropout = sys.argv[8]
    learning_rate = sys.argv[9]
    
    if extract_metrics(result_file, tracker_file, experiment_id, language, task, layer_index, hidden_size, dropout, learning_rate):
        print(f"Successfully extracted metrics from {result_file}")
    else:
        print(f"Failed to extract metrics from {result_file}")
EOF

chmod +x ${OUTPUT_BASE_DIR}/extract_metrics.py

# Create an experiment ID counter
experiment_id=1

# Run the parameter sweep
for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for LAYER in "${LAYERS[@]}"; do
            for HIDDEN_SIZE in "${HIDDEN_SIZES[@]}"; do
                for DROPOUT in "${DROPOUT_RATES[@]}"; do
                    for LR in "${LEARNING_RATES[@]}"; do
                        # Set task type based on task
                        if [ "$TASK" == "question_type" ]; then
                            TASK_TYPE="classification"
                        else
                            TASK_TYPE="regression"
                        fi
                        
                        # Create unique experiment name and output directory
                        EXPERIMENT_NAME="probe_${LANG}_${TASK}_layer${LAYER}_h${HIDDEN_SIZE}_d${DROPOUT}_lr${LR}"
                        EXPERIMENT_DIR="${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}"
                        mkdir -p "$EXPERIMENT_DIR"
                        
                        echo "=============================================="
                        echo "Running experiment ID: $experiment_id"
                        echo "Language: $LANG, Task: $TASK, Layer: $LAYER"
                        echo "Hidden Size: $HIDDEN_SIZE, Dropout: $DROPOUT, LR: $LR"
                        echo "=============================================="
                        
                        # Set specific model configuration based on task type
                        if [ "$TASK_TYPE" == "classification" ]; then
                            # Classification probe configuration 
                            PROBE_CONFIG="\"model.probe_hidden_size=${HIDDEN_SIZE}\" \"model.probe_depth=2\" \"model.dropout=${DROPOUT}\" \"model.activation=gelu\" \"model.normalization=layer\" \"model.use_mean_pooling=true\""
                            
                            TRAINING_CONFIG="\"training.lr=${LR}\" \"training.patience=3\" \"training.scheduler_factor=0.5\" \"training.scheduler_patience=2\" \"+training.gradient_accumulation_steps=2\""
                        else
                            # Regression probe configuration
                            PROBE_CONFIG="\"model.probe_hidden_size=${HIDDEN_SIZE}\" \"model.probe_depth=2\" \"model.dropout=${DROPOUT}\" \"model.activation=silu\" \"model.normalization=layer\" \"model.output_standardization=true\" \"model.use_mean_pooling=true\""
                            
                            TRAINING_CONFIG="\"training.lr=${LR}\" \"training.patience=4\" \"training.scheduler_factor=0.5\" \"training.scheduler_patience=2\" \"+training.gradient_accumulation_steps=2\""
                        fi
                        
                        # Construct the command
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
                            \"model.layer_index=${LAYER}\" \
                            ${PROBE_CONFIG} \
                            \"data.languages=[${LANG}]\" \
                            \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
                            \"training.task_type=${TASK_TYPE}\" \
                            \"training.num_epochs=15\" \
                            \"training.batch_size=16\" \
                            ${TRAINING_CONFIG} \
                            \"experiment_name=${EXPERIMENT_NAME}\" \
                            \"output_dir=${EXPERIMENT_DIR}\" \
                            \"wandb.mode=offline\""
                        
                        # Execute the experiment
                        eval $COMMAND
                        
                        # Check if experiment succeeded
                        if [ $? -eq 0 ]; then
                            echo "Experiment $experiment_id completed successfully"
                            
                            # Extract metrics
                            RESULTS_FILE="${EXPERIMENT_DIR}/${LANG}/results.json"
                            if [ -f "$RESULTS_FILE" ]; then
                                python3 ${OUTPUT_BASE_DIR}/extract_metrics.py \
                                    "$RESULTS_FILE" "$RESULTS_TRACKER" "$experiment_id" "$LANG" "$TASK" \
                                    "$LAYER" "$HIDDEN_SIZE" "$DROPOUT" "$LR"
                            else
                                echo "Warning: Results file not found: $RESULTS_FILE"
                                echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
                            fi
                        else
                            echo "Error in experiment $experiment_id"
                            echo "${EXPERIMENT_NAME}" >> $FAILED_LOG
                        fi
                        
                        # Clean GPU memory
                        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
                        
                        # Increment experiment ID
                        experiment_id=$((experiment_id+1))
                        
                    done
                done
            done
        done
    done
done

# Count total number of experiments
TOTAL_EXPERIMENTS=$((${#LANGUAGES[@]} * ${#TASKS[@]} * ${#LAYERS[@]} * ${#HIDDEN_SIZES[@]} * ${#DROPOUT_RATES[@]} * ${#LEARNING_RATES[@]}))
COMPLETED_EXPERIMENTS=$(grep -v "experiment_id" "$RESULTS_TRACKER" | wc -l)
FAILED_COUNT=$(wc -l < "$FAILED_LOG")

echo "=============================================="
echo "Parameter sweep completed!"
echo "=============================================="
echo "Total planned experiments: $TOTAL_EXPERIMENTS"
echo "Successfully completed: $COMPLETED_EXPERIMENTS"
echo "Failed experiments: ${FAILED_COUNT:-0}"
echo "Success rate: $(( COMPLETED_EXPERIMENTS * 100 / TOTAL_EXPERIMENTS ))%"
echo "Results available in: ${OUTPUT_BASE_DIR}"
echo "=============================================="
