#!/bin/bash
#SBATCH --job-name=clean_tfidf
#SBATCH --clusters=wice
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

# Activate the clean environment
echo "🔄 Activating clean qtype-eval environment..."

# Load Miniconda and activate environment
module purge
module load cluster/wice/oldhierarchy
module load Miniconda3/py310_22.11.1-1

# Initialize conda with fallback
if [ -f "$EBROOTMINICONDA3/etc/profile.d/conda.sh" ]; then
    source $EBROOTMINICONDA3/etc/profile.d/conda.sh
    conda activate qtype-eval
else
    echo "⚠️ Conda initialization failed, checking if conda is already available..."
    if command -v conda >/dev/null 2>&1; then
        echo "✅ Conda found, activating qtype-eval environment..."
        conda activate qtype-eval
    else
        echo "❌ Conda not available, exiting..."
        exit 1
    fi
fi

# Set up environment (keeping HF settings)
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
required = ['numpy', 'pandas', 'sklearn', 'xgboost', 'transformers', 'datasets', 'hydra_core']
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

# Initialize experiment setup
LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
MODELS=("dummy" "logistic" "ridge" "xgboost")  
TASKS=("question_type" "complexity")

OUTPUT_BASE_DIR="$VSC_SCRATCH/clean_tfidf_output"
mkdir -p "$OUTPUT_BASE_DIR"

RESULTS_TRACKER="${OUTPUT_BASE_DIR}/results.csv"
echo "experiment,language,task,model,metric,value,status" > "$RESULTS_TRACKER"

FEATURES_DIR="./data/tfidf_features"

# Generate TF-IDF features if needed
echo "🔍 Checking TF-IDF features..."
if [ ! -f "${FEATURES_DIR}/metadata.json" ]; then
    echo "⚙️ Generating TF-IDF features..."
    python3 scripts/generate_tfidf_glot500.py \
        --output-dir "${FEATURES_DIR}" \
        --model-name "cis-lmu/glot500-base" \
        --max-features 50000 \
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
    
    local EXP_NAME="${MODEL}_${TASK}_${LANG}"
    local TASK_TYPE="classification"
    [ "$TASK" == "complexity" ] && TASK_TYPE="regression"
    
    local OUTPUT_DIR="${OUTPUT_BASE_DIR}/${EXP_NAME}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "🔬 Running: $EXP_NAME"
    
    # Check if already completed
    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo "  ✅ Already completed"
        return 0
    fi
    
    # Run the experiment using Python script
    python3 -c "
import sys
sys.path.append('.')

from src.data.datasets import load_sklearn_data
from src.models.tfidf_baselines import create_tfidf_baseline_model
from src.training.sklearn_trainer import SklearnTrainer
import json
import traceback

try:
    # Load data
    print('  📊 Loading data...')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
        task='$TASK',
        languages=['$LANG'],
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
    
    # Save results
    results['experiment_name'] = '$EXP_NAME'
    results['language'] = '$LANG'
    results['task'] = '$TASK'
    results['model'] = '$MODEL'
    
    with open('$OUTPUT_DIR/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log to tracker
    test_metrics = results.get('test_metrics', {})
    for metric, value in test_metrics.items():
        print(f'$EXP_NAME,$LANG,$TASK,$MODEL,{metric},{value},success')
    
    print('  ✅ Completed successfully')

except Exception as e:
    print(f'  ❌ Failed: {e}')
    print(f'$EXP_NAME,$LANG,$TASK,$MODEL,error,{str(e)},failed')
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee -a "${OUTPUT_DIR}/log.txt"
    
    return ${PIPESTATUS[0]}
}

# Run all experiments
echo "🚀 Starting TF-IDF experiments..."
total=0
completed=0

for lang in "${LANGUAGES[@]}"; do
    for task in "${TASKS[@]}"; do
        for model in "${MODELS[@]}"; do
            # Skip incompatible combinations
            if [[ "$model" == "logistic" && "$task" == "complexity" ]]; then
                echo "⏭️ Skipping $model + $task (incompatible)"
                continue
            fi
            
            ((total++))
            
            echo ""
            echo "📋 Experiment $total: $model + $task + $lang"
            echo "----------------------------------------"
            
            if run_experiment "$lang" "$task" "$model"; then
                ((completed++))
                echo "✅ Success ($completed/$total)"
            else
                echo "❌ Failed ($completed/$total)"
            fi
        done
    done
done

echo ""
echo "========================================================================="
echo "🏁 TF-IDF Experiments Completed!"
echo "========================================================================="
echo "Total experiments: $total"
echo "Completed successfully: $completed"
echo "Failed: $((total - completed))"
echo "Results directory: $OUTPUT_BASE_DIR"
echo "========================================================================="

# Create summary
echo "📊 Creating summary..."
python3 -c "
import json
import pandas as pd
from pathlib import Path

output_dir = Path('$OUTPUT_BASE_DIR')
results = []

for result_file in output_dir.rglob('results.json'):
    try:
        with open(result_file) as f:
            data = json.load(f)
        
        exp_name = data.get('experiment_name', 'unknown')
        test_metrics = data.get('test_metrics', {})
        
        for metric, value in test_metrics.items():
            results.append({
                'experiment': exp_name,
                'language': data.get('language', 'unknown'),
                'task': data.get('task', 'unknown'),
                'model': data.get('model', 'unknown'),
                'metric': metric,
                'value': value
            })
    except Exception as e:
        print(f'Error processing {result_file}: {e}')

if results:
    df = pd.DataFrame(results)
    df.to_csv('$OUTPUT_BASE_DIR/summary.csv', index=False)
    print(f'Summary saved: {len(results)} result entries')
    
    # Print best results
    if 'accuracy' in df['metric'].values:
        best_acc = df[df['metric'] == 'accuracy'].nlargest(5, 'value')
        print('\\nTop 5 Accuracy Results:')
        for _, row in best_acc.iterrows():
            print(f'  {row[\"experiment\"]}: {row[\"value\"]:.3f}')
else:
    print('No results found')
"

echo "✅ All done!"
