#!/bin/bash
set -e 

LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
TASKS=("question_type" "complexity")
MODEL_TYPE="lm_probe"
MODEL_NAME="cis-lmu/glot500-base"

export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/hf_cache
export HF_DATASETS_OFFLINE=1  
export TRANSFORMERS_OFFLINE=1 

OUTPUT_DIR="outputs/basic_glot500_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

LOG_FILE="${OUTPUT_DIR}/run_log.txt"
echo "Starting basic experiments at $(date)" > $LOG_FILE

for lang in "${LANGUAGES[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "Running $task experiment for $lang" | tee -a $LOG_FILE
        
        task_dir="${OUTPUT_DIR}/${task}"
        mkdir -p $task_dir
        
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
            experiment_name="${MODEL_TYPE}_${task}_${lang}" \
            output_dir="${task_dir}/${lang}" \
            2>&1 | tee -a "${task_dir}/${lang}.log"
        
        echo "Completed $task experiment for $lang" | tee -a $LOG_FILE
        echo "----------------------------------------" | tee -a $LOG_FILE
    done
done

echo "All basic experiments completed at $(date)" | tee -a $LOG_FILE