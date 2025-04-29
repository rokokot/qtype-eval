#!/bin/bash
#SBATCH --job-name=finetune_experiments
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p100
#SBATCH --clusters=genius
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
CONTROL_INDICES=(1 2 3)

OUTPUT_BASE_DIR="$VSC_SCRATCH/finetune_output"
mkdir -p $OUTPUT_BASE_DIR

RESULTS_TRACKER="${OUTPUT_BASE_DIR}/finetune_results_tracker.csv"
echo "experiment_type,language,task,submetric,control_index,metric,value" > $RESULTS_TRACKER

cat > ${OUTPUT_BASE_DIR}/extract_finetune_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os
import glob

def extract_metrics(result_file, tracker_file, exp_type, language, task, submetric, control_index):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract test metrics
        test_metrics = data.get('test_metrics', {})
        
        # Append to tracker file
        with open(tracker_file, 'a') as f:
            writer = csv.writer(f)
            for metric, value in test_metrics.items():
                if value is not None:
                    writer.writerow([
                        exp_type, language, task, 
                        submetric if submetric else '', 
                        control_index if control_index != 'None' else 'None',
                        metric, value
                    ])
        return True
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return False

def find_and_extract_all_metrics(base_dir, tracker_file):
    """Find all results.json files and extract metrics from them."""
    results_count = 0
    
    # Process question_type results
    for lang_dir in glob.glob(f"{base_dir}/question_type/*/"):
        lang = os.path.basename(os.path.dirname(lang_dir))
        
        # Main experiment
        main_result_file = os.path.join(lang_dir, "results.json")
        if os.path.exists(main_result_file):
            if extract_metrics(main_result_file, tracker_file, "finetune", lang, "question_type", "", "None"):
                results_count += 1
        
        # Control experiments
        for control_dir in glob.glob(f"{lang_dir}/control*/"):
            control_idx = os.path.basename(control_dir).replace("control", "")
            control_result_file = os.path.join(control_dir, "results.json")
            if os.path.exists(control_result_file):
                if extract_metrics(control_result_file, tracker_file, "finetune_control", lang, "question_type", "", control_idx):
                    results_count += 1
    
    # Process complexity results
    for lang_dir in glob.glob(f"{base_dir}/complexity/*/"):
        lang = os.path.basename(os.path.dirname(lang_dir))
        
        # Main experiment
        main_result_file = os.path.join(lang_dir, "results.json")
        if os.path.exists(main_result_file):
            if extract_metrics(main_result_file, tracker_file, "finetune", lang, "complexity", "", "None"):
                results_count += 1
        
        # Control experiments
        for control_dir in glob.glob(f"{lang_dir}/control*/"):
            control_idx = os.path.basename(control_dir).replace("control", "")
            control_result_file = os.path.join(control_dir, "results.json")
            if os.path.exists(control_result_file):
                if extract_metrics(control_result_file, tracker_file, "finetune_control", lang, "complexity", "", control_idx):
                    results_count += 1
    
    # Process submetric results
    for lang_dir in glob.glob(f"{base_dir}/submetrics/*/"):
        lang = os.path.basename(os.path.dirname(lang_dir))
        
        for submetric_dir in glob.glob(f"{lang_dir}/*/"):
            submetric = os.path.basename(os.path.dirname(submetric_dir))
            
            # Main experiment
            main_result_file = os.path.join(submetric_dir, "results.json")
            if os.path.exists(main_result_file):
                if extract_metrics(main_result_file, tracker_file, "finetune", lang, "single_submetric", submetric, "None"):
                    results_count += 1
            
            # Control experiments
            for control_dir in glob.glob(f"{submetric_dir}/control*/"):
                control_idx = os.path.basename(control_dir).replace("control", "")
                control_result_file = os.path.join(control_dir, "results.json")
                if os.path.exists(control_result_file):
                    if extract_metrics(control_result_file, tracker_file, "finetune_control", lang, "single_submetric", submetric, control_idx):
                        results_count += 1
    
    return results_count

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: extract_finetune_metrics.py <base_output_dir> <tracker_file>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    tracker_file = sys.argv[2]
    
    processed_count = find_and_extract_all_metrics(base_dir, tracker_file)
    print(f"Successfully processed {processed_count} result files")
EOF

chmod +x ${OUTPUT_BASE_DIR}/extract_finetune_metrics.py

# Create script for generating summary files
cat > ${OUTPUT_BASE_DIR}/generate_finetune_summaries.py << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import os

def generate_summaries(tracker_file, output_dir):
    print(f"Reading tracker file: {tracker_file}")
    df = pd.read_csv(tracker_file)
    
    # Replace empty strings with NaN for proper handling
    df = df.replace('', np.nan)
    
    print(f"Generating summaries in: {output_dir}")
    
    # 1. Task Summary - average metrics by task
    task_summary = df.pivot_table(
        index=['task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    task_summary.to_csv(os.path.join(output_dir, 'finetune_task_summary.csv'))
    print("Generated finetune_task_summary.csv")
    
    # 2. Language Summary - average metrics by language and task
    language_summary = df.pivot_table(
        index=['language', 'task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    language_summary.to_csv(os.path.join(output_dir, 'finetune_language_summary.csv'))
    print("Generated finetune_language_summary.csv")
    
    # 3. Submetric Summary - only for submetric tasks
    submetric_df = df[df['submetric'].notna()]
    if not submetric_df.empty:
        submetric_summary = submetric_df.pivot_table(
            index=['submetric'], 
            columns='metric', 
            values='value',
            aggfunc='mean'
        )
        submetric_summary.to_csv(os.path.join(output_dir, 'finetune_submetric_summary.csv'))
        print("Generated finetune_submetric_summary.csv")
    
    # 4. Control Summary - compare control vs. non-control
    control_summary = df.pivot_table(
        index=['task', 'control_index'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    control_summary.to_csv(os.path.join(output_dir, 'finetune_control_summary.csv'))
    print("Generated finetune_control_summary.csv")
    
    # 5. Complete matrix for all experiments
    # Reshape the data to have one row per unique experiment
    experiment_matrix = df.pivot_table(
        index=['experiment_type', 'language', 'task', 'submetric', 'control_index'],
        columns='metric',
        values='value'
    )
    experiment_matrix.to_csv(os.path.join(output_dir, 'finetune_experiment_matrix.csv'))
    print("Generated finetune_experiment_matrix.csv")
    
    # 6. Task-specific summaries
    for task in df['task'].unique():
        if pd.notna(task):
            task_df = df[df['task'] == task]
            task_summary = task_df.pivot_table(
                index=['language'], 
                columns='metric', 
                values='value',
                aggfunc='mean'
            )
            task_summary.to_csv(os.path.join(output_dir, f'finetune_{task}_summary.csv'))
            print(f"Generated finetune_{task}_summary.csv")
    
    print("All summary files generated successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_finetune_summaries.py <tracker_file> <output_dir>")
        sys.exit(1)
    
    tracker_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    generate_summaries(tracker_file, output_dir)
EOF

chmod +x ${OUTPUT_BASE_DIR}/generate_finetune_summaries.py

# Function to run fine-tuning experiment and track results
run_finetune_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    local OUTPUT_SUBDIR=$6
    
    local EXPERIMENT_NAME=""
    local COMMAND=""
    
    # Set experiment name and command based on task and control index
    if [ "$TASK" == "single_submetric" ]; then
        # Submetric experiment
        if [ -z "$CONTROL_IDX" ]; then
            # Regular submetric experiment
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_${LANG}"
            COMMAND="python -m src.experiments.run_experiment \
                \"hydra.job.chdir=False\" \
                \"hydra.run.dir=.\" \
                \"experiment=finetune_submetric\" \
                \"experiment.submetric=${SUBMETRIC}\" \
                \"model=glot500_finetune\" \
                \"model.lm_name=cis-lmu/glot500-base\" \
                \"data.languages=[${LANG}]\" \
                \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
                \"training.task_type=${TASK_TYPE}\" \
                \"training.num_epochs=10\" \
                \"training.batch_size=16\" \
                \"+training.gradient_accumulation_steps=4\" \
                \"experiment_name=${EXPERIMENT_NAME}\" \
                \"output_dir=${OUTPUT_SUBDIR}\" \
                \"wandb.mode=offline\""
        else
            # Control submetric experiment
            EXPERIMENT_NAME="finetune_${SUBMETRIC}_control${CONTROL_IDX}_${LANG}"
            COMMAND="python -m src.experiments.run_experiment \
                \"hydra.job.chdir=False\" \
                \"hydra.run.dir=.\" \
                \"experiment=finetune_submetric\" \
                \"experiment.submetric=${SUBMETRIC}\" \
                \"experiment.use_controls=true\" \
                \"experiment.control_index=${CONTROL_IDX}\" \
                \"model=glot500_finetune\" \
                \"model.lm_name=cis-lmu/glot500-base\" \
                \"data.languages=[${LANG}]\" \
                \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
                \"training.task_type=${TASK_TYPE}\" \
                \"training.num_epochs=10\" \
                \"training.batch_size=16\" \
                \"+training.gradient_accumulation_steps=4\" \
                \"experiment_name=${EXPERIMENT_NAME}\" \
                \"output_dir=${OUTPUT_SUBDIR}\" \
                \"wandb.mode=offline\""
        fi
    else
        # Question type or complexity task
        if [ -z "$CONTROL_IDX" ]; then
            # Regular experiment
            EXPERIMENT_NAME="finetune_${TASK}_${LANG}"
            COMMAND="python -m src.experiments.run_experiment \
                \"hydra.job.chdir=False\" \
                \"hydra.run.dir=.\" \
                \"experiment=finetune\" \
                \"experiment.tasks=${TASK}\" \
                \"model=glot500_finetune\" \
                \"model.lm_name=cis-lmu/glot500-base\" \
                \"data.languages=[${LANG}]\" \
                \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
                \"training.task_type=${TASK_TYPE}\" \
                \"training.num_epochs=10\" \
                \"training.batch_size=16\" \
                \"+training.gradient_accumulation_steps=4\" \
                \"experiment_name=${EXPERIMENT_NAME}\" \
                \"output_dir=${OUTPUT_SUBDIR}\" \
                \"wandb.mode=offline\""
        else
            # Control experiment
            EXPERIMENT_NAME="finetune_${TASK}_control${CONTROL_IDX}_${LANG}"
            COMMAND="python -m src.experiments.run_experiment \
                \"hydra.job.chdir=False\" \
                \"hydra.run.dir=.\" \
                \"experiment=finetune\" \
                \"experiment.tasks=${TASK}\" \
                \"experiment.use_controls=true\" \
                \"experiment.control_index=${CONTROL_IDX}\" \
                \"model=glot500_finetune\" \
                \"model.lm_name=cis-lmu/glot500-base\" \
                \"data.languages=[${LANG}]\" \
                \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
                \"training.task_type=${TASK_TYPE}\" \
                \"training.num_epochs=10\" \
                \"training.batch_size=16\" \
                \"+training.gradient_accumulation_steps=4\" \
                \"experiment_name=${EXPERIMENT_NAME}\" \
                \"output_dir=${OUTPUT_SUBDIR}\" \
                \"wandb.mode=offline\""
        fi
    fi
    
    # Execute the experiment
    echo "Running experiment: ${EXPERIMENT_NAME}"
    eval $COMMAND
    
    if [ $? -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully"
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}"
        return 1
    fi
}

# Question type fine-tuning experiments
for LANG in "${LANGUAGES[@]}"; do
    # Regular experiment
    TASK_OUTPUT_DIR="${OUTPUT_BASE_DIR}/question_type/${LANG}"
    mkdir -p "$TASK_OUTPUT_DIR"
    run_finetune_experiment "classification" "$LANG" "question_type" "" "" "$TASK_OUTPUT_DIR"
    
    # Control experiments
    for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
        CONTROL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/question_type/${LANG}/control${CONTROL_IDX}"
        mkdir -p "$CONTROL_OUTPUT_DIR"
        run_finetune_experiment "classification" "$LANG" "question_type" "$CONTROL_IDX" "" "$CONTROL_OUTPUT_DIR"
    done
done

# Complexity regression fine-tuning experiments
for LANG in "${LANGUAGES[@]}"; do
    # Regular experiment
    TASK_OUTPUT_DIR="${OUTPUT_BASE_DIR}/complexity/${LANG}"
    mkdir -p "$TASK_OUTPUT_DIR"
    run_finetune_experiment "regression" "$LANG" "complexity" "" "" "$TASK_OUTPUT_DIR"
    
    # Control experiments
    for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
        CONTROL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/complexity/${LANG}/control${CONTROL_IDX}"
        mkdir -p "$CONTROL_OUTPUT_DIR"
        run_finetune_experiment "regression" "$LANG" "complexity" "$CONTROL_IDX" "" "$CONTROL_OUTPUT_DIR"
    done
done

# Submetric regression fine-tuning experiments
for LANG in "${LANGUAGES[@]}"; do
    for SUBMETRIC in "${SUBMETRICS[@]}"; do
        # Regular experiment
        SUBMETRIC_OUTPUT_DIR="${OUTPUT_BASE_DIR}/submetrics/${LANG}/${SUBMETRIC}"
        mkdir -p "$SUBMETRIC_OUTPUT_DIR"
        run_finetune_experiment "regression" "$LANG" "single_submetric" "" "$SUBMETRIC" "$SUBMETRIC_OUTPUT_DIR"
        
        # Control experiments
        for CONTROL_IDX in "${CONTROL_INDICES[@]}"; do
            CONTROL_OUTPUT_DIR="${OUTPUT_BASE_DIR}/submetrics/${LANG}/${SUBMETRIC}/control${CONTROL_IDX}"
            mkdir -p "$CONTROL_OUTPUT_DIR"
            run_finetune_experiment "regression" "$LANG" "single_submetric" "$CONTROL_IDX" "$SUBMETRIC" "$CONTROL_OUTPUT_DIR"
        done
    done
done

# Collect all results
echo "Collecting all experiment results..."
python3 ${OUTPUT_BASE_DIR}/extract_finetune_metrics.py "$OUTPUT_BASE_DIR" "$RESULTS_TRACKER"

# Generate summary files
echo "Generating summary files..."
python3 ${OUTPUT_BASE_DIR}/generate_finetune_summaries.py "$RESULTS_TRACKER" "$OUTPUT_BASE_DIR"

# Create metadata file
cat > "${OUTPUT_BASE_DIR}/finetune_metadata.json" << EOF
{
  "experiment_type": "fine-tuning",
  "description": "Full fine-tuning of language model for classification and regression tasks",
  "model": "cis-lmu/glot500-base",
  "model_config": "glot500_finetune",
  "languages": ["ar", "en", "fi", "id", "ja", "ko", "ru"],
  "tasks": ["question_type", "complexity"],
  "submetrics": ["avg_links_len", "avg_max_depth", "avg_subordinate_chain_len", "avg_verb_edges", "lexical_density", "n_tokens"],
  "freeze_model": false,
  "finetune": true,
  "layer_wise": false,
  "control_indices": [1, 2, 3],
  "batch_size": 16,
  "gradient_accumulation_steps": 4,
  "effective_batch_size": 64,
  "date_completed": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo "Fine-tuning completed. Results are available in ${OUTPUT_BASE_DIR}"
echo "Summary files have been generated for easier visualization"