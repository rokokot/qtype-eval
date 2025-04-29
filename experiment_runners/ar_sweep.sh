#!/bin/bash
#SBATCH --job-name=arabic_sweep
#SBATCH --time=06:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=36G  
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p100
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37132

# Set up environment
export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

conda activate qtype-eval
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install hydra-core hydra-submitit-launcher
pip install "transformers>=4.30.0,<4.36.0" torch datasets wandb

export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

# Sweep configuration arrays
LAYERS=(2 5 6 9 11 12)
DROPOUT_RATES=(0.1 0.2 0.3)
LEARNING_RATES=("1e-5" "2e-5" "5e-5")
PROBE_HIDDEN_SIZES=(64 96 128)
FREEZE_OPTIONS=(true false)
BATCH_SIZES=(8 16 32 64)  # Batch size variations
TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
CONTROL_INDICES=(1 2 3)

OUTPUT_BASE_DIR="$VSC_SCRATCH/arabic_sweep"
mkdir -p "$OUTPUT_BASE_DIR"

RESULTS_TRACKER="${OUTPUT_BASE_DIR}/sweep_results.csv"
echo "experiment_type,language,task,submetric,layer,dropout,learning_rate,probe_hidden_size,freeze_model,control_index,batch_size,metric,value" > "$RESULTS_TRACKER"

# Function to run an experiment with tracking
run_experiment() {
    local TASK_TYPE=$1
    local TASK=$2
    local LAYER=$3
    local DROPOUT=$4
    local LR=$5
    local PROBE_HIDDEN_SIZE=$6
    local FREEZE_MODEL=$7
    local CONTROL_IDX=$8
    local BS=$9
    local SUBMETRIC="${10:-none}"
    
    local EXPERIMENT_NAME=""
    local COMMAND=""
    
    # Construct experiment name and command
    if [ "$TASK" == "single_submetric" ]; then
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="sweep_${SUBMETRIC}_layer${LAYER}_dropout${DROPOUT}_lr${LR}_probe${PROBE_HIDDEN_SIZE}_freeze${FREEZE_MODEL}_bs${BS}"
            COMMAND="python -m src.experiments.run_experiment \
                \"hydra.job.chdir=False\" \
                \"hydra.run.dir=.\" \
                \"experiment=submetrics\" \
                \"experiment.submetric=${SUBMETRIC}\" \
                \"model=lm_probe\" \
                \"model.lm_name=cis-lmu/glot500-base\" \
                \"model.dropout=${DROPOUT}\" \
                \"model.layer_wise=true\" \
                \"model.layer_index=${LAYER}\" \
                \"+model.probe_hidden_size=${PROBE_HIDDEN_SIZE}\" \
                \"model.freeze_model=${FREEZE_MODEL}\" \
                \"data.languages=[ar]\" \
                \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
                \"training.task_type=${TASK_TYPE}\" \
                \"training.lr=${LR}\" \
                \"training.batch_size=${BS}\" \
                \"training.num_epochs=10\" \
                \"experiment_name=${EXPERIMENT_NAME}\" \
                \"output_dir=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}\" \
                \"wandb.mode=offline\""
        else
            EXPERIMENT_NAME="sweep_${SUBMETRIC}_layer${LAYER}_dropout${DROPOUT}_lr${LR}_probe${PROBE_HIDDEN_SIZE}_freeze${FREEZE_MODEL}_control${CONTROL_IDX}_bs${BS}"
            COMMAND="python -m src.experiments.run_experiment \
                \"hydra.job.chdir=False\" \
                \"hydra.run.dir=.\" \
                \"experiment=submetrics\" \
                \"experiment.submetric=${SUBMETRIC}\" \
                \"experiment.use_controls=true\" \
                \"experiment.control_index=${CONTROL_IDX}\" \
                \"model=lm_probe\" \
                \"model.lm_name=cis-lmu/glot500-base\" \
                \"model.dropout=${DROPOUT}\" \
                \"model.layer_wise=true\" \
                \"model.layer_index=${LAYER}\" \
                \"+model.probe_hidden_size=${PROBE_HIDDEN_SIZE}\" \
                \"model.freeze_model=${FREEZE_MODEL}\" \
                \"data.languages=[ar]\" \
                \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
                \"training.task_type=${TASK_TYPE}\" \
                \"training.lr=${LR}\" \
                \"training.batch_size=${BS}\" \
                \"training.num_epochs=10\" \
                \"experiment_name=${EXPERIMENT_NAME}\" \
                \"output_dir=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}\" \
                \"wandb.mode=offline\""
        fi
    else
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="sweep_${TASK}_layer${LAYER}_dropout${DROPOUT}_lr${LR}_probe${PROBE_HIDDEN_SIZE}_freeze${FREEZE_MODEL}_bs${BS}"
            COMMAND="python -m src.experiments.run_experiment \
                \"hydra.job.chdir=False\" \
                \"hydra.run.dir=.\" \
                \"experiment=${TASK}\" \
                \"experiment.tasks=${TASK}\" \
                \"model=lm_probe\" \
                \"model.lm_name=cis-lmu/glot500-base\" \
                \"model.dropout=${DROPOUT}\" \
                \"model.layer_wise=true\" \
                \"model.layer_index=${LAYER}\" \
                \"+model.probe_hidden_size=${PROBE_HIDDEN_SIZE}\" \
                \"model.freeze_model=${FREEZE_MODEL}\" \
                \"data.languages=[ar]\" \
                \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
                \"training.task_type=${TASK_TYPE}\" \
                \"training.lr=${LR}\" \
                \"training.batch_size=${BS}\" \
                \"training.num_epochs=10\" \
                \"experiment_name=${EXPERIMENT_NAME}\" \
                \"output_dir=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}\" \
                \"wandb.mode=offline\""
        else
            EXPERIMENT_NAME="sweep_${TASK}_layer${LAYER}_dropout${DROPOUT}_lr${LR}_probe${PROBE_HIDDEN_SIZE}_freeze${FREEZE_MODEL}_control${CONTROL_IDX}_bs${BS}"
            COMMAND="python -m src.experiments.run_experiment \
                \"hydra.job.chdir=False\" \
                \"hydra.run.dir=.\" \
                \"experiment=${TASK}\" \
                \"experiment.tasks=${TASK}\" \
                \"experiment.use_controls=true\" \
                \"experiment.control_index=${CONTROL_IDX}\" \
                \"model=lm_probe\" \
                \"model.lm_name=cis-lmu/glot500-base\" \
                \"model.dropout=${DROPOUT}\" \
                \"model.layer_wise=true\" \
                \"model.layer_index=${LAYER}\" \
                \"+model.probe_hidden_size=${PROBE_HIDDEN_SIZE}\" \
                \"model.freeze_model=${FREEZE_MODEL}\" \
                \"data.languages=[ar]\" \
                \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
                \"training.task_type=${TASK_TYPE}\" \
                \"training.lr=${LR}\" \
                \"training.batch_size=${BS}\" \
                \"training.num_epochs=10\" \
                \"experiment_name=${EXPERIMENT_NAME}\" \
                \"output_dir=${OUTPUT_BASE_DIR}/${EXPERIMENT_NAME}\" \
                \"wandb.mode=offline\""
        fi
    fi
    
    # Execute the experiment
    echo "Running experiment: ${EXPERIMENT_NAME}"
    eval $COMMAND
    
    if [ $? -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully"
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}"
        return 1
    fi
}

# Extract metrics function
function extract_metrics() {
    local result_file="$1"
    local experiment_name="$2"
    local task="$3"
    local submetric="${4:-none}"
    local layer="$5"
    local dropout="$6"
    local lr="$7"
    local probe_hidden_size="$8"
    local freeze_model="$9"
    local control_index="${10}"
    local batch_size="${11}"

    python3 - << EOF
import json
import csv

def process_metrics(data):
    test_metrics = data.get('test_metrics', {})
    if test_metrics:
        with open("$RESULTS_TRACKER", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for metric, value in test_metrics.items():
                writer.writerow([
                    "probe", "ar", "$task", "$submetric", $layer, $dropout, "$lr", 
                    $probe_hidden_size, $freeze_model, "${control_index:-0}", $batch_size, 
                    metric, value
                ])

try:
    with open("$result_file", 'r') as f:
        data = json.load(f)
    process_metrics(data)
except Exception as e:
    print(f"Error processing {result_file}: {e}")
EOF
}

# Nested loops for the complete parameter sweep
for TASK in "${TASKS[@]}"; do
    TASK_TYPE="classification"
    if [ "$TASK" == "complexity" ]; then
        TASK_TYPE="regression"
    fi

    for LAYER in "${LAYERS[@]}"; do
        for DROPOUT in "${DROPOUT_RATES[@]}"; do
            for LR in "${LEARNING_RATES[@]}"; do
                for PROBE_HIDDEN_SIZE in "${PROBE_HIDDEN_SIZES[@]}"; do
                    for FREEZE_MODEL in "${FREEZE_OPTIONS[@]}"; do
                        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
                            for BS in "${BATCH_SIZES[@]}"; do
                                run_experiment "$TASK_TYPE" "$TASK" "$LAYER" "$DROPOUT" "$LR" "$PROBE_HIDDEN_SIZE" "$FREEZE_MODEL" "$CONTROL_IDX" "$BS"
                            done
                        done
                    done
                done
            done
        done
    done
done

# Submetric experiments
for SUBMETRIC in "${SUBMETRICS[@]}"; do
    for LAYER in "${LAYERS[@]}"; do
        for DROPOUT in "${DROPOUT_RATES[@]}"; do
            for LR in "${LEARNING_RATES[@]}"; do
                for PROBE_HIDDEN_SIZE in "${PROBE_HIDDEN_SIZES[@]}"; do
                    for FREEZE_MODEL in "${FREEZE_OPTIONS[@]}"; do
                        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
                            for BS in "${BATCH_SIZES[@]}"; do
                                run_experiment "regression" "single_submetric" "$LAYER" "$DROPOUT" "$LR" "$PROBE_HIDDEN_SIZE" "$FREEZE_MODEL" "$CONTROL_IDX" "$BS" "$SUBMETRIC"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Sweep completed. Results saved to $RESULTS_TRACKER"