#!/bin/bash
#SBATCH --job-name=tfidf_experiments
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
echo "Activating clean qtype-eval environment..."

module --force purge
module load cluster/wice/oldhierarchy

if module load Miniconda3/py310_22.11.1-1 2>/dev/null; then
    echo "Miniconda module loaded successfully"
    if [ -f "$EBROOTMINICONDA3/etc/profile.d/conda.sh" ]; then
        source $EBROOTMINICONDA3/etc/profile.d/conda.sh
        conda activate qtype-eval
    else
        echo "Conda initialization script not found at expected path"
        exit 1
    fi
else
    echo "Miniconda module failed to load, using fallback..."
    eval "$(conda shell.bash hook)"
    conda activate qtype-eval
fi

# Set up environment
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME="$VSC_DATA/hf_cache"
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1

# Explicitly disable all HF offline modes
unset HF_HUB_OFFLINE
unset HUGGINGFACE_HUB_OFFLINE
export HF_HUB_OFFLINE=0

echo "Environment verification:"
echo "  Python path: $(which python)"
echo "  Python version: $(python --version)"
echo "  Conda environment: $CONDA_DEFAULT_ENV"
echo "  Conda prefix: $CONDA_PREFIX"

# Quick package check
echo "Checking required packages..."
python3 -c "
required = [
    ('numpy', 'numpy'), 
    ('pandas', 'pandas'), 
    ('sklearn', 'scikit-learn'), 
    ('xgboost', 'xgboost'), 
    ('transformers', 'transformers'), 
    ('datasets', 'datasets'), 
    ('hydra', 'hydra-core'), 
    ('sentencepiece', 'sentencepiece'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn')
]
missing = []
for import_name, pip_name in required:
    try:
        __import__(import_name)
        print(f'{pip_name}')
    except ImportError:
        print(f'{pip_name}')
        missing.append(pip_name)

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

# Enhanced experiment tracking
EXPERIMENT_BATCH_NAME="tfidf_results_$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE_DIR="$VSC_SCRATCH/tfidf_experiments"
EXPERIMENT_DIR="$OUTPUT_BASE_DIR/$EXPERIMENT_BATCH_NAME"

# Force fresh run by removing existing results
echo "Setting up experiment directory..."
mkdir -p "$EXPERIMENT_DIR"

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

FEATURES_DIR="$PWD/data/tfidf_features"

# Generate high-quality TF-IDF features
echo "Checking TF-IDF features..."
echo "Features directory: ${FEATURES_DIR}"
echo "Working directory: $(pwd)"
echo "Features exist: $(ls -la "${FEATURES_DIR}" 2>/dev/null | head -5 || echo 'Directory not found')"

if [ ! -f "${FEATURES_DIR}/metadata.json" ]; then
    echo "Generating hybrid XLM-RoBERTa + text2text TF-IDF features..."
    python3 scripts/generate_tfidf_features.py \
        --output-dir "${FEATURES_DIR}" \
        --model-name "xlm-roberta-base" \
        --max-features 32000 \
        --min-df 3 \
        --max-df 0.95 \
        --verify
    
    if [ $? -ne 0 ]; then
        echo "Hybrid TF-IDF generation failed"
        exit 1
    fi
else
    echo "Hybrid XLM-RoBERTa + text2text TF-IDF features found"
    echo "  Features: $(cat ${FEATURES_DIR}/metadata.json | grep -o '\"vocab_size\":[^,]*' | cut -d':' -f2)"
    echo "  Tokenizer: XLM-RoBERTa-base with text2text methodology"
fi

# Initialize enhanced experiment logger
echo "Initializing enhanced experiment tracking..."
python3 -c "
import sys
sys.path.append('.')
from src.utils.experiment_logger import ExperimentLogger

# Initialize experiment logger
logger = ExperimentLogger(
    base_output_dir='$OUTPUT_BASE_DIR',
    experiment_name='$EXPERIMENT_BATCH_NAME'
)

print(f'Experiment logger initialized: {logger.experiment_name}')
print(f'Results directory: {logger.results_dir}')
print(f'Visualizations directory: {logger.viz_dir}')
"

if [ $? -ne 0 ]; then
    echo "Failed to initialize experiment logger"
    exit 1
fi

# Function to run a single experiment with enhanced logging
run_enhanced_experiment() {
    local LANG=$1
    local TASK=$2
    local MODEL=$3
    local CONFIG=$4
    local EXP_TYPE=$5  # "main" or "control"
    
    local EXP_NAME="${MODEL}_${TASK}_${LANG}_${CONFIG}"
    local TASK_TYPE=${MAIN_TASKS[$TASK]:-${LINGUISTIC_TASKS[$TASK]}}
    
    echo "Running: $EXP_NAME ($EXP_TYPE)"
    
    # Run the experiment with enhanced tracking - pass variables directly to avoid JSON parsing issues
    python3 -c "
import sys
sys.path.append('.')

import json
import traceback
from src.utils.experiment_logger import ExperimentLogger
from src.data.datasets import load_sklearn_data_with_config
from src.models.tfidf_baselines import create_tfidf_baseline_model
from src.training.sklearn_trainer import SklearnTrainer

# Initialize experiment logger
exp_logger = ExperimentLogger(
    base_output_dir='$OUTPUT_BASE_DIR',
    experiment_name='$EXPERIMENT_BATCH_NAME'
)

# Create experiment configuration directly
experiment_config = {
    'experiment_name': '$EXP_NAME',
    'language': '$LANG',
    'task': '$TASK',
    'model_type': '$MODEL',
    'task_type': '$TASK_TYPE',
    'config': '$CONFIG',
    'experiment_type': '$EXP_TYPE',
    'features_dir': '$FEATURES_DIR',
    'batch_name': '$EXPERIMENT_BATCH_NAME'
}

# Log experiment start
exp_id = exp_logger.log_experiment_start(experiment_config)
print(f'Started tracking experiment: {exp_id}')

try:
    # Load data with specific config
    print('  Loading data...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data_with_config(
        task='$TASK',
        languages=['$LANG'],
        dataset_config='$CONFIG',
        tfidf_features_dir='$FEATURES_DIR',
        use_tfidf_loader=True
    )
    print(f'    Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
    
    # Create model
    print('  Creating model...')
    model = create_tfidf_baseline_model(
        model_type='$MODEL',
        task_type='$TASK_TYPE',
        tfidf_features_dir='$FEATURES_DIR',
        target_languages=['$LANG']
    )
    
    # Train with enhanced tracking
    print('  Training...')
    trainer = SklearnTrainer(
        model=model.model,
        task_type='$TASK_TYPE',
        output_dir=None  # We'll handle saving separately
    )
    
    results = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        test_data=(X_test, y_test)
    )
    
    # Add additional metadata to results
    results['data_shapes'] = {
        'train': list(X_train.shape),
        'val': list(X_val.shape),
        'test': list(X_test.shape)
    }
    
    # Log successful completion
    exp_logger.log_experiment_result(experiment_config, results)
    
    # Print main test results
    test_metrics = results.get('test_metrics', {})
    if '$TASK_TYPE' == 'classification':
        acc = test_metrics.get('accuracy', 'N/A')
        f1 = test_metrics.get('f1', 'N/A')
        print(f'    Test Accuracy: {acc:.4f}' if isinstance(acc, (int, float)) else f'    Test Accuracy: {acc}')
        print(f'    Test F1: {f1:.4f}' if isinstance(f1, (int, float)) else f'    Test F1: {f1}')
    else:
        mse = test_metrics.get('mse', 'N/A') 
        r2 = test_metrics.get('r2', 'N/A')
        print(f'    Test MSE: {mse:.4f}' if isinstance(mse, (int, float)) else f'    Test MSE: {mse}')
        print(f'    Test R²: {r2:.4f}' if isinstance(r2, (int, float)) else f'    Test R²: {r2}')
    
    print('  Completed successfully')
    print(f'  Results logged with ID: {exp_id}')

except Exception as e:
    print(f'  Failed: {e}')
    traceback.print_exc()
    
    # Log failure
    exp_logger.log_experiment_result(experiment_config, {}, error=str(e))
    sys.exit(1)
"
    
    return $?
}

# Run main experiments (base config)
echo ""
echo "Starting main TF-IDF experiments with enhanced tracking..."
main_total=0
main_completed=0

for lang in "${LANGUAGES[@]}"; do
    # Main tasks
    for task in "${!MAIN_TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            # Skip incompatible combinations
            if [[ "$model" == "logistic" && "${MAIN_TASKS[$task]}" == "regression" ]]; then
                echo "Skipping $model + $task (incompatible)"
                continue
            fi
            if [[ "$model" == "ridge" && "${MAIN_TASKS[$task]}" == "classification" ]]; then
                echo "Skipping $model + $task (incompatible)"
                continue
            fi
            
            ((main_total++))
            echo ""
            echo "Main Experiment $main_total: $model + $task + $lang"
            echo "----------------------------------------"
            
            if run_enhanced_experiment "$lang" "$task" "$model" "base" "main"; then
                ((main_completed++))
                echo "Success ($main_completed/$main_total)"
            else
                echo "Failed ($main_completed/$main_total)"
            fi
            
            # Print progress every 10 experiments
            if (( main_total % 10 == 0 )); then
                echo ""
                echo "Progress Update: $main_completed/$main_total completed"
                # Get current experiment status
                python3 -c "
import sys
sys.path.append('.')
from src.utils.experiment_logger import ExperimentLogger

exp_logger = ExperimentLogger('$OUTPUT_BASE_DIR', '$EXPERIMENT_BATCH_NAME')
status = exp_logger.get_experiment_status()
print(f'Current Status: {status[\"completed\"]} completed, {status[\"failed\"]} failed, {status[\"success_rate\"]*100:.1f}% success rate')
"
                echo ""
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
            echo "Main Experiment $main_total: $model + $task + $lang"
            echo "----------------------------------------"
            
            if run_enhanced_experiment "$lang" "$task" "$model" "base" "main"; then
                ((main_completed++))
                echo "Success ($main_completed/$main_total)"
            else
                echo "Failed ($main_completed/$main_total)"
            fi
        done
    done
done

# Run control experiments
echo ""
echo "Starting control TF-IDF experiments with enhanced tracking..."
control_total=0
control_completed=0

for lang in "${LANGUAGES[@]}"; do
    # Question type controls
    for seed in 1 2 3; do
        for model in "${MODELS[@]}"; do
            # Skip incompatible combinations
            if [[ "$model" == "logistic" && "${MAIN_TASKS[question_type]}" == "regression" ]]; then
                continue
            fi
            if [[ "$model" == "ridge" && "${MAIN_TASKS[question_type]}" == "classification" ]]; then
                continue
            fi
            
            ((control_total++))
            echo ""
            echo "Control Experiment $control_total: $model + question_type + $lang + seed$seed"
            echo "----------------------------------------"
            
            if run_enhanced_experiment "$lang" "question_type" "$model" "control_question_type_seed$seed" "control"; then
                ((control_completed++))
                echo "Success ($control_completed/$control_total)"
            else
                echo "Failed ($control_completed/$control_total)"
            fi
        done
    done
    
    # Complexity controls
    for seed in 1 2 3; do
        for model in "${MODELS[@]}"; do
            # Skip logistic for regression tasks
            if [[ "$model" == "logistic" ]]; then
                continue
            fi
            
            ((control_total++))
            echo ""
            echo "Control Experiment $control_total: $model + lang_norm_complexity_score + $lang + seed$seed"
            echo "----------------------------------------"
            
            if run_enhanced_experiment "$lang" "lang_norm_complexity_score" "$model" "control_complexity_seed$seed" "control"; then
                ((control_completed++))
                echo "Success ($control_completed/$control_total)"
            else
                echo "Failed ($control_completed/$control_total)"
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
            echo "Control Experiment $control_total: $model + avg_links_len + $lang + seed$seed"
            echo "----------------------------------------"
            
            if run_enhanced_experiment "$lang" "avg_links_len" "$model" "control_avg_links_len_seed$seed" "control"; then
                ((control_completed++))
                echo "Success ($control_completed/$control_total)"
            else
                echo "Failed ($control_completed/$control_total)"
            fi
        done
    done
done

echo ""
echo "========================================================================="
echo "Enhanced TF-IDF Experiments Completed!"
echo "========================================================================="
echo "Main experiments: $main_completed/$main_total"
echo "Control experiments: $control_completed/$control_total"
echo "Total: $((main_completed + control_completed))/$((main_total + control_total))"
echo ""
echo "Feature Information:"
if [ -f "${FEATURES_DIR}/metadata.json" ]; then
    VOCAB_SIZE=$(cat ${FEATURES_DIR}/metadata.json | grep -o '\"vocab_size\":[^,]*' | cut -d':' -f2)
    TOKENIZER=$(cat ${FEATURES_DIR}/metadata.json | grep -o '\"model_name\":\"[^\"]*' | cut -d':' -f2 | tr -d '"')
    echo "  Features generated: $VOCAB_SIZE (hybrid approach)"
    echo "  Tokenizer: $TOKENIZER with text2text methodology"
    echo "  Approach: XLM-RoBERTa tokenization + sklearn TF-IDF"
else
    echo "  Features: Hybrid XLM-RoBERTa + text2text approach"
fi
echo ""
echo "Results directory: $EXPERIMENT_DIR"
echo ""

# Finalize experiment and create comprehensive analysis
echo "Creating comprehensive analysis and visualizations..."
python3 -c "
import sys
sys.path.append('.')
from src.utils.experiment_logger import ExperimentLogger

# Finalize experiment logging
exp_logger = ExperimentLogger('$OUTPUT_BASE_DIR', '$EXPERIMENT_BATCH_NAME')
summaries = exp_logger.finalize_experiment()

print('Experiment logging finalized')
print('Summary tables created:')
for name, df in summaries.items():
    if hasattr(df, '__len__'):
        print(f'  - {name}: {len(df)} entries')
"

# Generate comprehensive visualizations
echo "Generating visualizations..."
python3 scripts/visualize_experiment_results.py "$EXPERIMENT_DIR" --dpi 300

if [ $? -eq 0 ]; then
    echo "Visualizations generated successfully"
else
    echo "Visualization generation encountered issues (check logs)"
fi

echo ""
echo "========================================================================="
echo "FINAL EXPERIMENT STATUS"
echo "========================================================================="

# Get final status
python3 -c "
import sys
sys.path.append('.')
from src.utils.experiment_logger import ExperimentLogger

exp_logger = ExperimentLogger('$OUTPUT_BASE_DIR', '$EXPERIMENT_BATCH_NAME')
status = exp_logger.get_experiment_status()

print(f'Experiment Batch: {status[\"experiment_name\"]}')
print(f'Total Experiments: {status[\"total_experiments\"]}')
print(f'Successful: {status[\"completed\"]} ({status[\"success_rate\"]*100:.1f}%)')
print(f'Failed: {status[\"failed\"]}')
print(f'Results Directory: {status[\"experiment_dir\"]}')
print('')
print('Generated Files:')
print('  - results/: Individual experiment results (JSON)')
print('  - summaries/: Aggregated CSV summaries')
print('  - visualizations/: Charts and plots')
print('  - logs/: Experiment execution logs')
print('')
print('Key Visualizations:')
print('  - main_vs_control_comparison.png')
print('  - performance_by_language_task.png') 
print('  - model_performance_comparison.png')
print('  - experiment_summary_report.md')
"

echo "========================================================================="