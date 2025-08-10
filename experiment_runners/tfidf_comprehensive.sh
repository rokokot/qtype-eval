#!/bin/bash
#SBATCH --job-name=tfidf_comprehensive
#SBATCH --clusters=wice
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=batch
#SBATCH --account=intro_vsc37132

echo "========================================================================="
echo "SLURM Job Information:"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_USER: $SLURM_JOB_USER"
echo "SLURM_JOB_ACCOUNT: $SLURM_JOB_ACCOUNT"
echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_CLUSTER_NAME: $SLURM_CLUSTER_NAME"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_CPUS_PER_NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "Date: $(date)"
echo "Walltime: $SLURM_TIMELIMIT"
echo "========================================================================="

# Activate environment
echo "🔄 Activating clean qtype-eval environment..."

module --force purge
module load cluster/wice/oldhierarchy

if module load Miniconda3/py310_22.11.1-1 2>/dev/null; then
    echo "✅ Miniconda module loaded successfully"
    if [ -f "$EBROOTMINICONDA3/etc/profile.d/conda.sh" ]; then
        source $EBROOTMINICONDA3/etc/profile.d/conda.sh
        conda activate qtype-eval
    else
        echo "❌ Conda initialization script not found at expected path"
        exit 1
    fi
else
    echo "⚠️ Miniconda module failed to load, using fallback..."
    eval "$(conda shell.bash hook)"
    conda activate qtype-eval
fi

# Set up environment
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME="$VSC_DATA/hf_cache"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1

echo "🐍 Environment verification:"
echo "  Python path: $(which python)"
echo "  Python version: $(python --version)"
echo "  Conda environment: $CONDA_DEFAULT_ENV"
echo "  Conda prefix: $CONDA_PREFIX"

# Quick package check
echo "📦 Checking required packages..."
python3 -c "
required = ['numpy', 'pandas', 'sklearn', 'xgboost', 'transformers', 'datasets', 'hydra_core', 'sentencepiece']
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        print(f'❌ {pkg}')
        missing.append(pkg)

if missing:
    print(f'Installing missing packages: {missing}')
    import subprocess
    import sys
    for pkg in missing:
        subprocess.run([sys.executable, '-m', 'pip', 'install', pkg])
"

# Experiment configuration
LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
MODELS=("dummy" "logistic" "ridge" "xgboost")

# Task definitions
declare -A MAIN_TASKS
MAIN_TASKS[question_type]="classification"
MAIN_TASKS[lang_norm_complexity_score]="regression"

# Individual linguistic metrics (all regression)
declare -A LINGUISTIC_TASKS
LINGUISTIC_TASKS[avg_links_len]="regression"
LINGUISTIC_TASKS[avg_max_depth]="regression"  
LINGUISTIC_TASKS[avg_subordinate_chain_len]="regression"
LINGUISTIC_TASKS[avg_verb_edges]="regression"
LINGUISTIC_TASKS[lexical_density]="regression"
LINGUISTIC_TASKS[n_tokens]="regression"

OUTPUT_BASE_DIR="$VSC_SCRATCH/tfidf_comprehensive_output"
mkdir -p "$OUTPUT_BASE_DIR"

FEATURES_DIR="./data/tfidf_features_128k_optimized"

# Generate high-quality TF-IDF features
echo "🔍 Checking TF-IDF features..."
if [ ! -f "${FEATURES_DIR}/metadata.json" ]; then
    echo "⚙️ Generating high-quality TF-IDF features..."
    python3 scripts/generate_tfidf_glot500.py \
        --output-dir "${FEATURES_DIR}" \
        --model-name "cis-lmu/glot500-base" \
        --max-features 128000 \
        --min-df 1 \
        --max-df 0.99 \
        --verify
    
    if [ $? -ne 0 ]; then
        echo "❌ TF-IDF generation failed"
        exit 1
    fi
else
    echo "✅ TF-IDF features found"
fi

# Function to run a single experiment
run_experiment() {
    local LANG=$1
    local TASK=$2
    local MODEL=$3
    local CONFIG=$4
    local EXP_TYPE=$5  # "main" or "control"
    
    local EXP_NAME="${MODEL}_${TASK}_${LANG}_${CONFIG}"
    local TASK_TYPE=${MAIN_TASKS[$TASK]:-${LINGUISTIC_TASKS[$TASK]}}
    
    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXP_TYPE}/${EXP_NAME}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "🔬 Running: $EXP_NAME ($EXP_TYPE)"
    
    # Check if already completed
    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo "  ✅ Already completed"
        return 0
    fi
    
    # Run the experiment
    python3 -c "
import sys
sys.path.append('.')

from src.data.datasets import load_sklearn_data_with_config
from src.models.tfidf_baselines import create_tfidf_baseline_model
from src.training.sklearn_trainer import SklearnTrainer
import json
import traceback

try:
    # Load data with specific config
    print('  📊 Loading data...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data_with_config(
        task='$TASK',
        languages=['$LANG'],
        dataset_config='$CONFIG',
        tfidf_features_dir='$FEATURES_DIR',
        use_tfidf_loader=True
    )
    print(f'    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
    
    # Create model
    print('  🤖 Creating model...')
    model = create_tfidf_baseline_model(
        model_type='$MODEL',
        task_type='$TASK_TYPE',
        tfidf_features_dir='$FEATURES_DIR',
        target_languages=['$LANG']
    )
    
    # Train
    print('  🏋️ Training...')
    trainer = SklearnTrainer(
        model=model.model,
        task_type='$TASK_TYPE',
        output_dir='$OUTPUT_DIR'
    )
    
    results = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        test_data=(X_test, y_test)
    )
    
    # Save results with metadata
    results['experiment_name'] = '$EXP_NAME'
    results['language'] = '$LANG'
    results['task'] = '$TASK'
    results['model'] = '$MODEL'
    results['config'] = '$CONFIG'
    results['experiment_type'] = '$EXP_TYPE'
    results['task_type'] = '$TASK_TYPE'
    
    with open('$OUTPUT_DIR/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print('  ✅ Completed successfully')

except Exception as e:
    print(f'  ❌ Failed: {e}')
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee -a "${OUTPUT_DIR}/log.txt"
    
    return ${PIPESTATUS[0]}
}

# Run main experiments (base config)
echo "🚀 Starting main TF-IDF experiments..."
main_total=0
main_completed=0

for lang in "${LANGUAGES[@]}"; do
    # Main tasks
    for task in "${!MAIN_TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            # Skip incompatible combinations
            if [[ "$model" == "logistic" && "${MAIN_TASKS[$task]}" == "regression" ]]; then
                echo "⏭️ Skipping $model + $task (incompatible)"
                continue
            fi
            
            ((main_total++))
            echo ""
            echo "📋 Main Experiment $main_total: $model + $task + $lang"
            echo "----------------------------------------"
            
            if run_experiment "$lang" "$task" "$model" "base" "main"; then
                ((main_completed++))
                echo "✅ Success ($main_completed/$main_total)"
            else
                echo "❌ Failed ($main_completed/$main_total)"
            fi
        done
    done
    
    # Individual linguistic metrics
    for task in "${!LINGUISTIC_TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            # Skip logistic for regression tasks
            if [[ "$model" == "logistic" ]]; then
                continue
            fi
            
            ((main_total++))
            echo ""
            echo "📋 Main Experiment $main_total: $model + $task + $lang"
            echo "----------------------------------------"
            
            if run_experiment "$lang" "$task" "$model" "base" "main"; then
                ((main_completed++))
                echo "✅ Success ($main_completed/$main_total)"
            else
                echo "❌ Failed ($main_completed/$main_total)"
            fi
        done
    done
done

# Run control experiments
echo ""
echo "🎯 Starting control TF-IDF experiments..."
control_total=0
control_completed=0

for lang in "${LANGUAGES[@]}"; do
    # Question type controls
    for seed in 1 2 3; do
        for model in "${MODELS[@]}"; do
            if [[ "$model" == "logistic" && "${MAIN_TASKS[question_type]}" == "regression" ]]; then
                continue
            fi
            
            ((control_total++))
            echo ""
            echo "📋 Control Experiment $control_total: $model + question_type + $lang + seed$seed"
            echo "----------------------------------------"
            
            if run_experiment "$lang" "question_type" "$model" "control_question_type_seed$seed" "control"; then
                ((control_completed++))
                echo "✅ Success ($control_completed/$control_total)"
            else
                echo "❌ Failed ($control_completed/$control_total)"
            fi
        done
    done
    
    # Complexity controls
    for seed in 1 2 3; do
        for model in "${MODELS[@]}"; do
            if [[ "$model" == "logistic" ]]; then
                continue
            fi
            
            ((control_total++))
            echo ""
            echo "📋 Control Experiment $control_total: $model + lang_norm_complexity_score + $lang + seed$seed"
            echo "----------------------------------------"
            
            if run_experiment "$lang" "lang_norm_complexity_score" "$model" "control_complexity_seed$seed" "control"; then
                ((control_completed++))
                echo "✅ Success ($control_completed/$control_total)"
            else
                echo "❌ Failed ($control_completed/$control_total)"
            fi
        done
    done
    
    # Individual metric controls (sample - just avg_links_len for now)
    for seed in 1 2 3; do
        for model in "${MODELS[@]}"; do
            if [[ "$model" == "logistic" ]]; then
                continue
            fi
            
            ((control_total++))
            echo ""
            echo "📋 Control Experiment $control_total: $model + avg_links_len + $lang + seed$seed"
            echo "----------------------------------------"
            
            if run_experiment "$lang" "avg_links_len" "$model" "control_avg_links_len_seed$seed" "control"; then
                ((control_completed++))
                echo "✅ Success ($control_completed/$control_total)"
            else
                echo "❌ Failed ($control_completed/$control_total)"
            fi
        done
    done
done

echo ""
echo "========================================================================="
echo "🏁 TF-IDF Comprehensive Experiments Completed!"
echo "========================================================================="
echo "Main experiments: $main_completed/$main_total"
echo "Control experiments: $control_completed/$control_total"
echo "Total: $((main_completed + control_completed))/$((main_total + control_total))"
echo "Results directory: $OUTPUT_BASE_DIR"
echo "========================================================================="

# Create comprehensive summary with main vs control comparison
echo "📊 Creating comprehensive summary..."
python3 -c "
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

output_dir = Path('$OUTPUT_BASE_DIR')
results = []

# Load all results
for result_file in output_dir.rglob('results.json'):
    try:
        with open(result_file) as f:
            data = json.load(f)
        
        exp_type = data.get('experiment_type', 'unknown')
        config = data.get('config', 'unknown')
        test_metrics = data.get('test_metrics', {})
        
        for metric, value in test_metrics.items():
            results.append({
                'experiment_type': exp_type,
                'experiment': data.get('experiment_name', 'unknown'),
                'language': data.get('language', 'unknown'),
                'task': data.get('task', 'unknown'),
                'model': data.get('model', 'unknown'),
                'config': config,
                'metric': metric,
                'value': value,
                'task_type': data.get('task_type', 'unknown')
            })
    except Exception as e:
        print(f'Error processing {result_file}: {e}')

if results:
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'comprehensive_results.csv', index=False)
    print(f'Comprehensive results saved: {len(results)} entries')
    
    # Separate main and control results
    main_df = df[df['experiment_type'] == 'main'].copy()
    control_df = df[df['experiment_type'] == 'control'].copy()
    
    # Convert values to numeric
    main_df['value'] = pd.to_numeric(main_df['value'], errors='coerce')
    control_df['value'] = pd.to_numeric(control_df['value'], errors='coerce')
    
    # Average control results across seeds
    control_avg = control_df.groupby(['task', 'model', 'language', 'metric'])['value'].mean().reset_index()
    control_avg['experiment_type'] = 'control_avg'
    
    # Combine for comparison
    comparison_data = []
    
    # Main results
    for _, row in main_df.iterrows():
        comparison_data.append({
            'task': row['task'],
            'model': row['model'], 
            'language': row['language'],
            'metric': row['metric'],
            'main_score': row['value'],
            'control_score': None
        })
    
    # Add control averages
    for _, row in control_avg.iterrows():
        # Find matching main result
        match_idx = None
        for i, comp_row in enumerate(comparison_data):
            if (comp_row['task'] == row['task'] and 
                comp_row['model'] == row['model'] and
                comp_row['language'] == row['language'] and
                comp_row['metric'] == row['metric']):
                match_idx = i
                break
        
        if match_idx is not None:
            comparison_data[match_idx]['control_score'] = row['value']
        else:
            comparison_data.append({
                'task': row['task'],
                'model': row['model'],
                'language': row['language'], 
                'metric': row['metric'],
                'main_score': None,
                'control_score': row['value']
            })
    
    # Save comparison results
    comp_df = pd.DataFrame(comparison_data)
    comp_df.to_csv(output_dir / 'main_vs_control_comparison.csv', index=False)
    
    # Show top results
    print('\\n=== MAIN VS CONTROL SUMMARY ===')
    
    # Accuracy results (classification tasks)
    acc_comp = comp_df[comp_df['metric'] == 'accuracy'].copy()
    if len(acc_comp) > 0:
        acc_comp = acc_comp.dropna(subset=['main_score', 'control_score'])
        if len(acc_comp) > 0:
            acc_comp['difference'] = acc_comp['main_score'] - acc_comp['control_score']
            print('\\nTop Accuracy Differences (Main - Control):')
            top_acc = acc_comp.nlargest(5, 'difference')
            for _, row in top_acc.iterrows():
                print(f'  {row[\"task\"]} ({row[\"model\"]}, {row[\"language\"]}): {row[\"main_score\"]:.3f} vs {row[\"control_score\"]:.3f} (diff: +{row[\"difference\"]:.3f})')
    
    # MSE results (regression tasks) 
    mse_comp = comp_df[comp_df['metric'] == 'mse'].copy()
    if len(mse_comp) > 0:
        mse_comp = mse_comp.dropna(subset=['main_score', 'control_score'])
        if len(mse_comp) > 0:
            mse_comp['difference'] = mse_comp['control_score'] - mse_comp['main_score']  # Lower MSE is better
            print('\\nTop MSE Improvements (Control MSE - Main MSE, positive = main better):')
            top_mse = mse_comp.nlargest(5, 'difference')
            for _, row in top_mse.iterrows():
                print(f'  {row[\"task\"]} ({row[\"model\"]}, {row[\"language\"]}): {row[\"main_score\"]:.4f} vs {row[\"control_score\"]:.4f} (improvement: +{row[\"difference\"]:.4f})')

else:
    print('No results found')
"

echo "✅ Comprehensive analysis complete!"