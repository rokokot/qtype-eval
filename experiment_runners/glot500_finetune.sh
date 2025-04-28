#!/bin/bash
#SBATCH --job-name=finetune_experiments
#SBATCH --time=12:00:00
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
echo "GPU information:"
nvidia-smi

echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
MAIN_TASKS=("question_type" "complexity")
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
CONTORL_INDICES=(1 2 3)

OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_output"
mkdir -p $OUTPUT_BASE_DIR

# Question type fine-tuning experiments
for LANG in "${LANGUAGES[@]}"; do
    mkdir -p "${OUTPUT_BASE_DIR}/question_type/${LANG}"
    echo "Running question type fine-tuning for ${LANG}"
    python -m src.experiments.run_experiment \
        "hydra.job.chdir=False" \
        "hydra.run.dir=." \
        "experiment=finetune" \
        "experiment.tasks=question_type" \
        "experiment.use_controls=false" \
        "model=lm_finetune" \
        "model.lm_name=cis-lmu/glot500-base" \
        "data.languages=[${LANG}]" \
        "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
        "training.task_type=classification" \
        "training.num_epochs=10" \
        "training.batch_size=16" \
        "experiment_name=finetune_question_type_${LANG}" \
        "output_dir=${OUTPUT_BASE_DIR}/question_type/${LANG}" \
        "wandb.mode=offline"
        
    if [ $? -eq 0 ]; then
        echo "Fine-tuning experiment for question_type ${LANG} completed successfully"
    else
        echo "Error in fine-tuning experiment for question_type ${LANG}"
    fi
    for CONTROL in "${CONTROL_INDICES[@]}"; do
        mkdir -p "${OUTPUT_BASE_DIR}/question_type/${LANG}/control${CONTROL}"
        echo "Running control ${CONTROL} question type fine-tuning for ${LANG}"
        python -m src.experiments.run_experiment \
            "hydra.job.chdir=False" \
            "hydra.run.dir=." \
            "experiment=finetune" \
            "experiment.tasks=question_type" \
            "experiment.use_controls=true" \
            "experiment.control_index=${CONTROL}" \
            "model=lm_finetune" \
            "model.lm_name=cis-lmu/glot500-base" \
            "data.languages=[${LANG}]" \
            "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
            "training.task_type=classification" \
            "training.num_epochs=10" \
            "training.batch_size=16" \
            "experiment_name=finetune_question_type_control${CONTROL}_${LANG}" \
            "output_dir=${OUTPUT_BASE_DIR}/question_type/${LANG}/control${CONTROL}" \
            "wandb.mode=offline"
            
        if [ $? -eq 0 ]; then
            echo "Control ${CONTROL} fine-tuning experiment for question_type ${LANG} completed successfully"
        else
            echo "Error in control ${CONTROL} fine-tuning experiment for question_type ${LANG}"
        fi
    done
done

# Complexity regression fine-tuning experiments
for LANG in "${LANGUAGES[@]}"; do
    mkdir -p "${OUTPUT_BASE_DIR}/complexity/${LANG}"
    echo "Running complexity fine-tuning for ${LANG}"
    python -m src.experiments.run_experiment \
        "hydra.job.chdir=False" \
        "hydra.run.dir=." \
        "experiment=finetune" \
        "experiment.tasks=complexity" \
        "experiment.use_controls=false"  \
        "model=lm_finetune" \
        "model.lm_name=cis-lmu/glot500-base" \
        "data.languages=[${LANG}]" \
        "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
        "training.task_type=regression" \
        "training.num_epochs=10" \
        "training.batch_size=16" \
        "experiment_name=finetune_complexity_${LANG}" \
        "output_dir=${OUTPUT_BASE_DIR}/complexity/${LANG}" \
        "wandb.mode=offline"
        
    if [ $? -eq 0 ]; then
        echo "Finetuning experiment for complexity ${LANG} completed successfully"
    else
        echo "Error in finetuning experiment for complexity ${LANG}"
    fi
    for CONTROL in "${CONTROL_INDICES[@]}"; do
        mkdir -p "${OUTPUT_BASE_DIR}/complexity/${LANG}/control${CONTROL}"
        echo "Running control ${CONTROL} complexity fine-tuning for ${LANG}"
        python -m src.experiments.run_experiment \
            "hydra.job.chdir=False" \
            "hydra.run.dir=." \
            "experiment=finetune" \
            "experiment.tasks=complexity" \
            "experiment.use_controls=true" \
            "experiment.control_index=${CONTROL}" \
            "model=lm_finetune" \
            "model.lm_name=cis-lmu/glot500-base" \
            "data.languages=[${LANG}]" \
            "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
            "training.task_type=regression" \
            "training.num_epochs=10" \
            "training.batch_size=16" \
            "experiment_name=finetune_complexity_control${CONTROL}_${LANG}" \
            "output_dir=${OUTPUT_BASE_DIR}/complexity/${LANG}/control${CONTROL}" \
            "wandb.mode=offline"
            
        if [ $? -eq 0 ]; then
            echo "Control ${CONTROL} finetuning experiment for complexity ${LANG} completed successfully"
        else
            echo "Error in control ${CONTROL} finetuning experiment for complexity ${LANG}"
        fi
    done
done

# Submetric regression fine-tuning experiments
for LANG in "${LANGUAGES[@]}"; do
    for SUBMETRIC in "${SUBMETRICS[@]}"; do
        mkdir -p "${OUTPUT_BASE_DIR}/submetrics/${LANG}/${SUBMETRIC}"
        echo "Running submetric ${SUBMETRIC} fine-tuning for ${LANG}"
        python -m src.experiments.run_experiment \
            "hydra.job.chdir=False" \
            "hydra.run.dir=." \
            "experiment=finetune_submetric" \
            "experiment.submetric=${SUBMETRIC}" \
            "model=lm_finetune" \
            "model.lm_name=cis-lmu/glot500-base" \
            "data.languages=[${LANG}]" \
            "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
            "training.task_type=regression" \
            "training.num_epochs=10" \
            "training.batch_size=16" \
            "experiment_name=finetune_${SUBMETRIC}_${LANG}" \
            "output_dir=${OUTPUT_BASE_DIR}/submetrics/${LANG}/${SUBMETRIC}" \
            "wandb.mode=offline"
            
        if [ $? -eq 0 ]; then
            echo "Fine-tuning experiment for ${SUBMETRIC} ${LANG} completed successfully"
        else
            echo "Error in fine-tuning experiment for ${SUBMETRIC} ${LANG}"
        fi
        for CONTROL in "${CONTROL_INDICES[@]}"; do
            mkdir -p "${OUTPUT_BASE_DIR}/submetrics/${LANG}/${SUBMETRIC}/control${CONTROL}"
            echo "Running control ${CONTROL} submetric ${SUBMETRIC} fine-tuning for ${LANG}"
            python -m src.experiments.run_experiment \
                "hydra.job.chdir=False" \
                "hydra.run.dir=." \
                "experiment=finetune_submetric" \
                "experiment.submetric=${SUBMETRIC}" \
                "experiment.use_controls=true" \
                "experiment.control_index=${CONTROL}" \
                "model=lm_finetune" \
                "model.lm_name=cis-lmu/glot500-base" \
                "data.languages=[${LANG}]" \
                "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
                "training.task_type=regression" \
                "training.num_epochs=10" \
                "training.batch_size=16" \
                "experiment_name=finetune_${SUBMETRIC}_control${CONTROL}_${LANG}" \
                "output_dir=${OUTPUT_BASE_DIR}/submetrics/${LANG}/${SUBMETRIC}/control${CONTROL}" \
                "wandb.mode=offline"
                
            if [ $? -eq 0 ]; then
                echo "Control ${CONTROL} fine-tuning experiment for ${SUBMETRIC} ${LANG} completed successfully"
            else
                echo "Error in control ${CONTROL} fine-tuning experiment for ${SUBMETRIC} ${LANG}"
            fi
        done
    done
done

echo "Finetuning completed"