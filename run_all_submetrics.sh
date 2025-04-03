#!/bin/bash
#SBATCH --job-name=glot500_submetric_experiments
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
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
MODEL_TYPE="lm_probe"
MODEL_NAME="cis-lmu/glot500-base"

export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/hf_cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUTPUT_DIR="outputs/submetrics_glot500_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

LOG_FILE="${OUTPUT_DIR}/run_log.txt"
echo "Starting submetric experiments at $(date)" > $LOG_FILE

for lang in "${LANGUAGES[@]}"; do
    for submetric in "${SUBMETRICS[@]}"; do
        echo "Running experiment for $submetric on $lang" | tee -a $LOG_FILE
        
        submetric_dir="${OUTPUT_DIR}/${submetric}"
        mkdir -p $submetric_dir
        
        python -m src.experiments.run_experiment \
            experiment=submetrics \
            experiment.submetric=${submetric} \
            model=lm_probe \
            model.lm_name=${MODEL_NAME} \
            data.languages="[${lang}]" \
            training.task_type="regression" \
            experiment_name="${MODEL_TYPE}_${submetric}_${lang}" \
            output_dir="${submetric_dir}/${lang}" \
            2>&1 | tee -a "${submetric_dir}/${lang}.log"
        
        echo "Completed experiment for $submetric on $lang" | tee -a $LOG_FILE
        echo "----------------------------------------" | tee -a $LOG_FILE
    done
done

echo "All submetric experiments completed at $(date)" | tee -a $LOG_FILE
