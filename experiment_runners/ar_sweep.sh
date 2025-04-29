#!/bin/bash
#SBATCH --job-name=arabic_sweep
#SBATCH --time=06:00:00  # Extended time for more comprehensive sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G  # Increased memory
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p100
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37132

# Set up environment
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

RESULTS_FILE="$OUTPUT_BASE_DIR/sweep_results.csv"
echo "task,submetric,layer,dropout,learning_rate,probe_hidden_size,freeze_model,control_index,batch_size,accuracy,f1,mse,rmse,r2" > "$RESULTS_FILE"

run_experiment() {
    local task="$1"
    local layer="$2"
    local dropout="$3"
    local lr="$4"
    local probe_hidden_size="$5"
    local freeze_model="$6"
    local control_index="$7"
    local BS="$8"
    local submetric="${9:-none}"

    local experiment_dir="$OUTPUT_BASE_DIR/${task}_layer${layer}_dropout${dropout}_lr${lr}_probe${probe_hidden_size}_freeze${freeze_model}_control${control_index}_bs${BS}"
    mkdir -p "$experiment_dir"

    local task_type="classification"
    if [[ "$task" == "complexity" ]] || [[ "$submetric" != "none" ]]; then
        task_type="regression"
    fi

    local experiment_command=(
        python -m src.experiments.run_experiment
        "hydra.job.chdir=False"
        "hydra.run.dir=."
        "experiment=${task}"
        "experiment.tasks=${task}"
        "model=lm_probe"
        "model.lm_name=cis-lmu/glot500-base"
        "model.dropout=${dropout}"
        "model.layer_wise=true"
        "model.layer_index=${layer}"
        "+model.probe_hidden_size=${probe_hidden_size}"
        "model.freeze_model=${freeze_model}"
        "data.languages=[ar]"
        "data.cache_dir=$VSC_DATA/qtype-eval/data/cache"
        "training.task_type=${task_type}"
        "training.lr=${lr}"
        "training.batch_size=${BS}"
        "training.num_epochs=10"
        "experiment_name=arabic_sweep_${task}_layer${layer}"
        "output_dir=${experiment_dir}"
        "wandb.mode=offline"
    )

    if [[ "$submetric" != "none" ]]; then
        experiment_command+=("experiment.submetric=${submetric}")
    fi

    if [[ "$control_index" -gt 0 ]]; then
        experiment_command+=(
            "experiment.use_controls=true"
            "experiment.control_index=${control_index}"
        )
    fi

    "${experiment_command[@]}"

    if [[ -f "${experiment_dir}/results.json" ]]; then
        python3 - << EOF
import json
import csv

with open("${experiment_dir}/results.json", 'r') as f:
    results = json.load(f)

task = "${task}"
submetric = "${submetric}" if "${submetric}" != "none" else ""
layer = ${layer}
dropout = ${dropout}
lr = "${lr}"
probe_hidden_size = ${probe_hidden_size}
freeze_model = ${freeze_model}
control_index = ${control_index}
batch_size = ${BS}

# Determine which metrics to extract based on task type
metrics = results.get('test_metrics', {})

accuracy = metrics.get('accuracy', 'N/A')
f1 = metrics.get('f1', 'N/A')
mse = metrics.get('mse', 'N/A')
rmse = metrics.get('rmse', 'N/A')
r2 = metrics.get('r2', 'N/A')

with open("$RESULTS_FILE", 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        task, submetric, layer, dropout, lr, 
        probe_hidden_size, freeze_model, control_index, batch_size,
        accuracy, f1, mse, rmse, r2
    ])
EOF
    fi
}

for task in "${TASKS[@]}"; do
    for layer in "${LAYERS[@]}"; do
        for dropout in "${DROPOUT_RATES[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for probe_hidden_size in "${PROBE_HIDDEN_SIZES[@]}"; do
                    for freeze_model in "${FREEZE_OPTIONS[@]}"; do
                        for control_index in "${CONTROL_INDICES[@]}"; do
                            for BS in "${BATCH_SIZES[@]}"; do
                                run_experiment "$task" "$layer" "$dropout" "$lr" "$probe_hidden_size" "$freeze_model" "$control_index" "$BS"
                            done
                        done
                    done
                done
            done
        done
    done
done

for submetric in "${SUBMETRICS[@]}"; do
    for layer in "${LAYERS[@]}"; do
        for dropout in "${DROPOUT_RATES[@]}"; do
            for lr in "${LEARNING_RATES[@]}"; do
                for probe_hidden_size in "${PROBE_HIDDEN_SIZES[@]}"; do
                    for freeze_model in "${FREEZE_OPTIONS[@]}"; do
                        for control_index in "${CONTROL_INDICES[@]}"; do
                            for BS in "${BATCH_SIZES[@]}"; do
                                run_experiment "submetrics" "$layer" "$dropout" "$lr" "$probe_hidden_size" "$freeze_model" "$control_index" "$BS" "$submetric"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "Sweep completed. Results saved to $RESULTS_FILE"