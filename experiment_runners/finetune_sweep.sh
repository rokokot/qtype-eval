# experiment_runners/finetune_sweep.sh
#!/bin/bash
#SBATCH --job-name=finetune_sweep
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
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

LANGUAGES=("en")  # optimize on English
MAIN_TASKS=("question_type" "complexity")
LRS=("1e-5" "2e-5" "5e-5")
BATCH_SIZES=("4" "8" "16")
OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_sweep"

for LANG in "${LANGUAGES[@]}"; do
    for TASK in "${MAIN_TASKS[@]}"; do
        TASK_TYPE="classification"
        if [ "$TASK" != "question_type" ]; then
            TASK_TYPE="regression"
        fi
        
        for LR in "${LRS[@]}"; do
            for BS in "${BATCH_SIZES[@]}"; do
                RUN_NAME="finetune_${TASK}_${LANG}_lr${LR}_bs${BS}"
                OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TASK}/${LANG}/lr${LR}_bs${BS}"
                
                mkdir -p "$OUTPUT_DIR"
                
                echo "Running sweep: $RUN_NAME"
                
                python -m src.experiments.run_experiment \
                    "hydra.job.chdir=False" \
                    "hydra.run.dir=." \
                    "experiment=finetune" \
                    "experiment.tasks=${TASK}" \
                    "model=lm_finetune" \
                    "model.lm_name=cis-lmu/glot500-base" \
                    "data.languages=[${LANG}]" \
                    "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
                    "training.task_type=${TASK_TYPE}" \
                    "training.lr=${LR}" \
                    "training.batch_size=${BS}" \
                    "training.num_epochs=3" \
                    "experiment_name=${RUN_NAME}" \
                    "output_dir=${OUTPUT_DIR}" \
                    "wandb.mode=online"
            done
        done
    done
done

echo "Parameter sweep completed."