#!/bin/bash
#SBATCH --job-name=glot500_control_experiments
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=126G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=robin.edu.hr@gmail.com

set -e  

LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
TASKS=("question_type" "complexity")
MODEL_TYPE="lm_probe"
MODEL_NAME="cis-lmu/glot500-base"
CONTROL_INDICES=(1 2 3)

export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/hf_cache
export HF_DATASETS_OFFLINE=1  
export TRANSFORMERS_OFFLINE=1  

OUTPUT_DIR="outputs/control_glot500_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

LOG_FILE="${OUTPUT_DIR}/run_log.txt"
echo "Starting control experiments at $(date)" > $LOG_FILE

for lang in "${LANGUAGES[@]}"; do
    for task in "${TASKS[@]}"; do
        for control in "${CONTROL_INDICES[@]}"; do
            echo "Running $task experiment for $lang with control $control" | tee -a $LOG_FILE
            
            task_dir="${OUTPUT_DIR}/${task}"
            mkdir -p "${task_dir}/${lang}"
            
            if [ "$task" == "question_type" ]; then
                task_type="classification"
            else
                task_type="regression"
            fi
            
            python -m src.experiments.run_experiment \
                experiment=${task} \
                model=lm_probe \
                model.lm_name=${MODEL_NAME} \
                data.languages="[${lang}]" \
                training.task_type=${task_type} \
                experiment.use_controls=true \
                experiment.control_index=${control} \
                experiment_name="${MODEL_TYPE}_${task}_control${control}_${lang}" \
                output_dir="${task_dir}/${lang}/control${control}" \
                2>&1 | tee -a "${task_dir}/${lang}_control${control}.log"
            
            echo "Completed $task experiment for $lang with control $control" | tee -a $LOG_FILE
            echo "----------------------------------------" | tee -a $LOG_FILE
        done
    done
done

echo "All control experiments completed at $(date)" | tee -a $LOG_FILE
