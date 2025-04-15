#!/bin/bash
#SBATCH --job-name=qtype_experiments
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132


# Use your personal Miniconda installation
export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the environment
conda activate qtype-eval

# Ensure necessary packages are installed
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install required Python packages
pip install hydra-core hydra-submitit-launcher
pip install "transformers>=4.30.0,<4.36.0" torch datasets wandb

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# IMPORTANT: Disable Hydra's working directory changes
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1

# Print environment information
echo "Environment variables:"
echo "PYTHONPATH=${PYTHONPATH}"
echo "HF_HOME=${HF_HOME}"
echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
echo "HYDRA_JOB_CHDIR=${HYDRA_JOB_CHDIR}"
echo "GPU information:"
nvidia-smi

# Print environment information
echo "Python executable: $(which python)"
echo "PyTorch CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"


LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
CONTROL_INDICES=(1 2 3)

# Base output directory
OUTPUT_BASE_DIR="$VSC_SCRATCH/question_type_output"
mkdir -p $OUTPUT_BASE_DIR

# Run standard experiments for each language
for LANG in "${LANGUAGES[@]}"; do
    # Create language subdirectory
    mkdir -p "${OUTPUT_BASE_DIR}/${LANG}"
    
    # Run standard experiment
    echo "Running question type classification for ${LANG}"
    python -m src.experiments.run_experiment \
        "hydra.job.chdir=False" \
        "hydra.run.dir=." \
        "experiment=question_type" \
        "experiment.tasks=question_type" \
        "model=lm_probe" \
        "model.lm_name=cis-lmu/glot500-base" \
        "data.languages=[${LANG}]" \
        "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
        "training.task_type=classification" \
        "training.num_epochs=10" \
        "training.batch_size=16" \
        "experiment_name=question_type_${LANG}" \
        "output_dir=${OUTPUT_BASE_DIR}/${LANG}" \
        "wandb.mode=offline"
        
    # Check for successful completion
    if [ $? -eq 0 ]; then
        echo "Standard experiment for ${LANG} completed successfully"
    else
        echo "Error in standard experiment for ${LANG}"
    fi
done

# Run control experiments for each language and control index
for LANG in "${LANGUAGES[@]}"; do
    for CONTROL in "${CONTROL_INDICES[@]}"; do
        # Create control subdirectory
        CONTROL_DIR="${OUTPUT_BASE_DIR}/${LANG}/control${CONTROL}"
        mkdir -p "${CONTROL_DIR}"
        
        echo "Running question type control=${CONTROL} for ${LANG}"
        python -m src.experiments.run_experiment \
            "hydra.job.chdir=False" \
            "hydra.run.dir=." \
            "experiment=question_type" \
            "experiment.tasks=question_type" \
            "model=lm_probe" \
            "model.lm_name=cis-lmu/glot500-base" \
            "experiment.use_controls=true" \
            "experiment.control_index=${CONTROL}" \
            "data.languages=[${LANG}]" \
            "data.cache_dir=$VSC_DATA/qtype-eval/data/cache" \
            "training.num_epochs=10" \
            "training.batch_size=16" \
            "experiment_name=question_type_control${CONTROL}_${LANG}" \
            "output_dir=${CONTROL_DIR}" \
            "wandb.mode=offline"
            
        if [ $? -eq 0 ]; then
            echo "Control experiment for ${LANG} (control=${CONTROL}) completed successfully"
        else
            echo "Error in control experiment for ${LANG} (control=${CONTROL})"
        fi
    done
done


echo "Question type classification experiments completed"
