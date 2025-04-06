#!/bin/bash
#SBATCH --job-name=glot500_basic_experiments
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=123
#SBATCH --mem=123G
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

export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

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
            "hydra.job.chdir=False" \
            "hydra.run.dir=." \
            experiment=${task} \
            experiment.tasks="[${task}]" \  # Added this line
            model=lm_probe \
            model.lm_name=${MODEL_NAME} \
            data.languages="[${lang}]" \
            data.cache_dir=$HF_HOME \  # Added this line
            training.task_type=${task_type} \
            training.num_epochs=10 \  # Added this line
            training.batch_size=16 \  # Added this line
            experiment_name="${MODEL_TYPE}_${task}_${lang}" \
            output_dir="${task_dir}/${lang}" \
            wandb.mode=offline \
            2>&1 | tee -a "${task_dir}/${lang}.log"
            
            results_file="${task_dir}/${lang}/results.json"
            if [ -f "$results_file" ]; then
                echo "Results saved successfully at: $results_file" | tee -a $LOG_FILE
            else
                echo "WARNING: No results file found at expected location: $results_file" | tee -a $LOG_FILE
                
                # Try to find results in other locations
                other_results=$(find "${task_dir}/${lang}" -name "*results*.json" -o -name "results.json")
                if [ -n "$other_results" ]; then
                    echo "Found results in alternative location(s):" | tee -a $LOG_FILE
                    echo "$other_results" | tee -a $LOG_FILE
                    
                    # Copy to expected location
                    first_result=$(echo "$other_results" | head -n 1)
                    cp "$first_result" "$results_file"
                    echo "Copied $first_result to $results_file" | tee -a $LOG_FILE
                else
                    # Try to find in Hydra's output directories
                    hydra_results=$(find "./outputs/${MODEL_TYPE}_${task}_${lang}" -name "*results*.json" -o -name "results.json")
                    if [ -n "$hydra_results" ]; then
                        echo "Found results in Hydra output directory:" | tee -a $LOG_FILE
                        echo "$hydra_results" | tee -a $LOG_FILE
                        
                        # Copy to expected location
                        first_result=$(echo "$hydra_results" | head -n 1)
                        cp "$first_result" "$results_file"
                        echo "Copied $first_result to $results_file" | tee -a $LOG_FILE
                    fi
                fi
            fi
            
        echo "Completed $task experiment for $lang" | tee -a $LOG_FILE
        echo "----------------------------------------" | tee -a $LOG_FILE
    done
done


echo "Generating results summary..." | tee -a $LOG_FILE

# Create summary JSON
summary_file="${OUTPUT_DIR}/summary.json"
echo "{" > $summary_file
echo "  \"timestamp\": \"$(date +%Y-%m-%d\ %H:%M:%S)\"," >> $summary_file
echo "  \"experiment_type\": \"basic\"," >> $summary_file
echo "  \"results\": {" >> $summary_file

first_task=true
for task in "${TASKS[@]}"; do
    if [ "$first_task" = true ]; then
        first_task=false
    else
        echo "    ," >> $summary_file
    fi
    
    echo "    \"${task}\": {" >> $summary_file
    
    first_lang=true
    for lang in "${LANGUAGES[@]}"; do
        if [ "$first_lang" = true ]; then
            first_lang=false
        else
            echo "      ," >> $summary_file
        fi
        
        results_file="${OUTPUT_DIR}/${task}/${lang}/results.json"
        if [ -f "$results_file" ]; then
            echo "      \"${lang}\": $(cat $results_file)" >> $summary_file
        else
            echo "      \"${lang}\": { \"error\": \"No results file found\" }" >> $summary_file
        fi
    done
    
    echo "    }" >> $summary_file
done

echo "  }" >> $summary_file
echo "}" >> $summary_file

echo "Summary saved to $summary_file" | tee -a $LOG_FILE
echo "All basic experiments completed at $(date)" | tee -a $LOG_FILE
