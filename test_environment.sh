#!/bin/bash
#SBATCH --job-name=test_env
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_debug
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

# Use your personal Miniconda installation
export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Create and activate environment
conda create -n qtype-eval python=3.9 -y || echo "Environment already exists"
conda activate qtype-eval

conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install hydra-core hydra-submitit-launcher
pip install -r requirements.txt
pip install --no-cache-dir wandb

# Install minimal dependencies for testing
pip install transformers datasets

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/hf_cache

# Print environment information
echo "========== ENVIRONMENT INFO =========="
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "Active conda env: $CONDA_DEFAULT_ENV"
echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
echo "========== DEPENDENCY CHECK =========="

echo "Checking for pytorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || echo "ERROR: PyTorch import failed!"

echo "Checking for hydra..."
python -c "import hydra; print(f'Hydra version: {hydra.__version__}')" || echo "ERROR: Hydra import failed!"

echo "Checking for transformers..."
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')" || echo "ERROR: Transformers import failed!"

echo "Checking for datasets..."
python -c "import datasets; print(f'Datasets version: {datasets.__version__}')" || echo "ERROR: Datasets import failed!"

echo "Checking for wandb..."
python -c "import wandb; print(f'Wandb version: {wandb.__version__}')" || echo "ERROR: Wandb import failed!"


# Test GPU availability
echo "========== GPU CHECK =========="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU available')"

# Test a minimal hydra script
echo "========== HYDRA TEST =========="
cat > test_hydra.py << 'EOF'
import hydra
from omegaconf import DictConfig

@hydra.main(config_path=None)
def main(cfg: DictConfig) -> None:
    print("Hydra is working!")
    print(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")

if __name__ == "__main__":
    main()
EOF

python test_hydra.py


echo "========== TRANSFORMERS TEST =========="
python -c "
from transformers import AutoTokenizer, AutoModel
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
print('Tokenizer loaded successfully')
print('Loading model...')
model = AutoModel.from_pretrained('distilbert-base-uncased')
print('Model loaded successfully')
print('Model on GPU:', next(model.parameters()).device)
"


echo "All tests completed!"
