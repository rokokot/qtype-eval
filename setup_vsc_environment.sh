#!/bin/bash
set -e 

echo "Setting up environment for experiments"

mkdir -p data/cache data/features
mkdir -p outputs logs

module purge
module load Python/3.9

if [ ! -d "$VSC_DATA/miniconda3" ]; then
    echo "Setting up Miniconda in VSC_DATA..."
    cd $VSC_DATA
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $VSC_DATA/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    
    echo 'export PATH="$VSC_DATA/miniconda3/bin:$PATH"' >> ~/.bashrc
    export PATH="$VSC_DATA/miniconda3/bin:$PATH"
fi

# Add project to PYTHONPATH
echo 'export PYTHONPATH=$PYTHONPATH:$PWD' >> ~/.bashrc
export PYTHONPATH=$PYTHONPATH:$PWD

# Configure cache directories to use scratch
echo 'export HF_HOME=$VSC_DATA/hf_cache' >> ~/.bashrc
export HF_HOME=$VSC_DATA/hf_cache

# Activate Miniconda
export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source $VSC_DATA/miniconda3/bin/activate

# Install required packages
pip install -r requirements.txt
pip install --no-cache-dir wandb

wandb login

# Check if TF-IDF vectors exist
echo "Checking for TF-IDF vectors..."
if [ ! -f "data/features/tfidf_vectors_train.pkl" ] || \
   [ ! -f "data/features/tfidf_vectors_dev.pkl" ] || \
   [ ! -f "data/features/tfidf_vectors_test.pkl" ]; then
    echo "WARNING: TF-IDF vectors not found in 'data/features'"
    echo "Please add your pre-extracted TF-IDF vectors before running sklearn experiments."
fi

echo "Caching datasets and models for offline use..."
python scripts/cache_resources.py --cache-dir $HF_HOME

echo -e "\nSetup complete! Ready to run experiments."
