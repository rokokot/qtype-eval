#!/bin/bash
#SBATCH --job-name=glot500_cross_lingual_experiments
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem-per-cpu=126G
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

export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTPUT_DIR="outputs/cross_lingual_glot500_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

LOG_FILE="${OUTPUT_DIR}/run_log.txt"
echo "Starting cross-lingual experiments at $(date)" > $LOG_FILE

for task in "${TASKS[@]}"; do
    task_dir="${OUTPUT_DIR}/${task}"
    mkdir -p $task_dir
 
    if [ "$task" == "question_type" ]; then
        task_type="classification"
    else
        task_type="regression"
    fi
    
    for src_lang in "${LANGUAGES[@]}"; do
        for tgt_lang in "${LANGUAGES[@]}"; do
         
            if [[ "$src_lang" == "$tgt_lang" ]]; then
                continue
            fi
            
            echo "Running $task cross-lingual experiment: $src_lang -> $tgt_lang" | tee -a $LOG_FILE
            
            pair_dir="${task_dir}/${src_lang}_to_${tgt_lang}"
            mkdir -p $pair_dir
            
     
            python -m src.experiments.run_experiment \
                experiment=cross_lingual \
                experiment.type=lm_probe_cross_lingual \
                model=lm_probe \
                model.lm_name=${MODEL_NAME} \
                data.train_language=${src_lang} \
                data.eval_language=${tgt_lang} \
                experiment.tasks="[${task}]" \
                training.task_type=${task_type} \
                experiment_name="cross_lingual_${task}_${src_lang}_to_${tgt_lang}" \
                output_dir=$pair_dir \
                2>&1 | tee -a "${pair_dir}/log.txt"
            
            echo "Completed $task cross-lingual experiment: $src_lang -> $tgt_lang" | tee -a $LOG_FILE
            echo "----------------------------------------" | tee -a $LOG_FILE
        done
    done
done

echo "All cross-lingual experiments completed at $(date)" | tee -a $LOG_FILE
