#!/bin/bash
#SBATCH --job-name=xling_experiments
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_debug
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

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

echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
echo "HYDRA_JOB_CHDIR=${HYDRA_JOB_CHDIR}"
echo "GPU information:"
nvidia-smi

echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"


SOURCE_LANGUAGES=("en") 
TARGET_LANGUAGES=("ru")
TASKS=("question_type" "complexity")

OUTPUT_BASE_DIR="$VSC_SCRATCH/cross_lingual_output"
mkdir -p $OUTPUT_BASE_DIR

for SRC_LANG in "${SOURCE_LANGUAGES[@]}"; do
    for TGT_LANG in "${TARGET_LANGUAGES[@]}"; do
        if [ "$SRC_LANG" = "$TGT_LANG" ]; then
            continue
        fi
        
        for TASK in "${TASKS[@]}"; do
            if [ "$TASK" = "question_type" ]; then
                TASK_TYPE="classification"
            else
                TASK_TYPE="regression"
            fi
            
            PAIR_DIR="${OUTPUT_BASE_DIR}/${SRC_LANG}_to_${TGT_LANG}/${TASK}"
            mkdir -p "${PAIR_DIR}"
            echo "Running cross-lingual ${TASK} from ${SRC_LANG} to ${TGT_LANG}"
            python -m src.experiments.run_experiment \
                "hydra.job.chdir=False" \
                "hydra.run.dir=." \
                "experiment=cross_lingual" \
                "experiment.tasks=${TASK}" \
                "experiment.cross_lingual=true" \
                "+experiment.use_controls=false" \
                "model=lm_probe" \
                "model.lm_name=cis-lmu/glot500-base" \
                "data.train_language=${SRC_LANG}" \
                "data.eval_language=${TGT_LANG}" \
                "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
                "training.task_type=${TASK_TYPE}" \
                "training.num_epochs=10" \
                "training.batch_size=16" \
                "experiment_name=cross_lingual_${TASK}_${SRC_LANG}_to_${TGT_LANG}" \
                "output_dir=${PAIR_DIR}" \
                "wandb.mode=offline"
                
            if [ $? -eq 0 ]; then
                echo "Cross-lingual experiment for ${TASK} (${SRC_LANG} → ${TGT_LANG}) completed successfully"
            else
                echo "Error in cross-lingual experiment for ${TASK} (${SRC_LANG} → ${TGT_LANG})"
            fi
        done
    done
done

echo "Cross-lingual experiments completed"
