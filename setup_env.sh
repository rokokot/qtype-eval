#!/bin/bash
# Environment setup script for qtype-eval project
# Uses requirements-hpc.txt for package management

set -e  # Exit on any error

echo "🚀 Setting up qtype-eval environment..."

# Load Python module
echo "📦 Loading Python module..."
module purge
module load Python/3.9.5-GCCcore-10.3.0

# Set up user package directory
export PYTHONUSERBASE=$VSC_DATA/python_packages
mkdir -p $PYTHONUSERBASE
export PATH=$PYTHONUSERBASE/bin:$PATH

echo "🐍 Python setup:"
echo "  Python path: $(which python)"
echo "  Python version: $(python --version)"
echo "  Package directory: $PYTHONUSERBASE"

# Check if this is the first run
MARKER_FILE="$PYTHONUSERBASE/.qtype_eval_installed"

if [ -f "$MARKER_FILE" ]; then
    echo "📋 Packages already installed (marker file exists)"
    echo "   To reinstall, delete: $MARKER_FILE"
else
    echo "📋 Installing packages from requirements-hpc.txt..."
    
    if [ ! -f "requirements-hpc.txt" ]; then
        echo "❌ requirements-hpc.txt not found in $(pwd)"
        echo "   Make sure you're in the project root directory"
        exit 1
    fi
    
    # Install packages
    pip install --user --upgrade pip
    pip install --user -r requirements-hpc.txt
    
    # Create marker file
    echo "$(date): qtype-eval packages installed" > "$MARKER_FILE"
    echo "✅ Package installation completed"
fi

# Set project environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=${HF_HOME:-/data/leuven/371/vsc37132/hf_cache}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1

# Create cache directories
mkdir -p $HF_HOME
mkdir -p ./data/cache

echo "🔧 Environment variables set:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  HF_HOME: $HF_HOME"

# Verify installation
echo "🧪 Verifying package installation..."
python -c "
import sys
packages = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('datasets', 'Datasets'),
    ('sklearn', 'Scikit-learn'),
    ('xgboost', 'XGBoost'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('hydra', 'Hydra'),
    ('omegaconf', 'OmegaConf'),
    ('scipy', 'SciPy'),
    ('wandb', 'Weights & Biases')
]

failed = []
for pkg, name in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {name}: {version}')
    except ImportError as e:
        print(f'❌ {name}: {e}')
        failed.append(name)

if failed:
    print(f'\\n❌ Failed to import: {failed}')
    print('💡 Try running: pip install --user --force-reinstall -r requirements-hpc.txt')
    sys.exit(1)
else:
    print('\\n🎉 All packages verified successfully!')
"

if [ $? -eq 0 ]; then
    echo "✅ Environment setup completed successfully!"
    echo ""
    echo "🏃 You can now run:"
    echo "  python scripts/generate_tfidf_glot500.py --output-dir ./data/tfidf_features"
    echo "  sbatch experiment_runners/tfidf_baselines.sh"
else
    echo "❌ Environment setup failed!"
    exit 1
fi
