#!/bin/bash
#SBATCH --job-name=finetune_test
#SBATCH --time=00:30:00
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
echo "GPU information:"
nvidia-smi

echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

TEST_DIR="$VSC_SCRATCH/finetune_test"
mkdir -p $TEST_DIR

echo "Testing question type fine-tuning on English..."
python -m src.experiments.run_experiment \
    "hydra.job.chdir=False" \
    "hydra.run.dir=." \
    "experiment=finetune" \
    "experiment.tasks=question_type" \
    "model=glot500_finetune" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.languages=[en]" \
    "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
    "training.task_type=classification" \
    "training.num_epochs=1" \
    "training.batch_size=4" \
    "training.lr=2e-5" \
    "experiment_name=test_finetune_qtype_en" \
    "output_dir=${TEST_DIR}/question_type/en" \
    "wandb.mode=offline"

if [ $? -eq 0 ]; then
    echo "Question type fine-tuning test completed successfully!"
else
    echo "Question type fine-tuning test failed!"
    exit 1
fi

echo "Testing complexity fine-tuning on English..."
python -m src.experiments.run_experiment \
    "hydra.job.chdir=False" \
    "hydra.run.dir=." \
    "experiment=finetune" \
    "experiment.tasks=complexity" \
    "model=glot500_finetune" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.languages=[en]" \
    "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
    "training.task_type=regression" \
    "training.num_epochs=1" \
    "training.batch_size=4" \
    "training.lr=2e-5" \
    "experiment_name=test_finetune_complexity_en" \
    "output_dir=${TEST_DIR}/complexity/en" \
    "wandb.mode=offline"

if [ $? -eq 0 ]; then
    echo "Complexity fine-tuning test completed successfully!"
else
    echo "Complexity fine-tuning test failed!"
    exit 1
fi

echo "Testing submetric (avg_links_len) fine-tuning on English..."
python -m src.experiments.run_experiment \
    "hydra.job.chdir=False" \
    "hydra.run.dir=." \
    "experiment=finetune" \
    "experiment.tasks=single_submetric" \
    "experiment.submetric=avg_links_len" \
    "model=glot500_finetune" \
    "model.lm_name=cis-lmu/glot500-base" \
    "data.languages=[en]" \
    "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
    "training.task_type=regression" \
    "training.num_epochs=1" \
    "training.batch_size=4" \
    "training.lr=2e-5" \
    "experiment_name=test_finetune_avg_links_len_en" \
    "output_dir=${TEST_DIR}/submetrics/en/avg_links_len" \
    "wandb.mode=offline"

if [ $? -eq 0 ]; then
    echo "Submetric fine-tuning test completed successfully!"
else
    echo "Submetric fine-tuning test failed!"
    exit 1
fi

echo "All fine-tuning tests completed successfully!"
echo "Results can be found in: $TEST_DIR"