#!/bin/bash
#SBATCH --job-name=check_configs
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=batch
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

# Use your personal Miniconda installation
export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the environment
conda activate qtype-eval

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD

# Check configuration directory structure
echo "Checking configuration directories..."
find configs -type d | sort

# Check main config files
echo "Checking main config files..."
cat configs/config.yaml

# Check a few key config files
echo "Checking experiment configs..."
cat configs/experiment/question_type.yaml

echo "Checking model configs..."
cat configs/model/lm_probe.yaml

echo "Checking training configs..."
cat configs/training/default.yaml

# Verify Hydra can parse configs
echo "Verifying Hydra can parse configs..."
python -c "
import hydra
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

@hydra.main(config_path='configs', config_name='config', version_base='1.3')
def check_config(cfg):
    print('Configuration verified!')
    print(OmegaConf.to_yaml(cfg))
    
if __name__ == '__main__':
    check_config()
"

echo "Config check completed!"
