#!/bin/bash
# Setup script for multilingual question probing experiments on VSC

set -e  # Exit on error

echo "Setting up environment for multilingual question probing experiments..."

# Create necessary directories
mkdir -p data/cache data/features
mkdir -p outputs logs

# Load Python module
module purge
module load Python/3.9

# Set up Miniconda if needed
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
echo 'export HF_HOME=$VSC_SCRATCH/hf_cache' >> ~/.bashrc
export HF_HOME=$VSC_SCRATCH/hf_cache

# Check if TF-IDF vectors exist
echo "Checking for TF-IDF vectors..."
if [ ! -f "data/features/tfidf_vectors_train.pkl" ] || \
   [ ! -f "data/features/tfidf_vectors_dev.pkl" ] || \
   [ ! -f "data/features/tfidf_vectors_test.pkl" ]; then
    echo "WARNING: TF-IDF vectors not found in 'data/features'"
    echo "Please add your pre-extracted TF-IDF vectors before running experiments."
fi

echo -e "\nSetup complete! To run an experiment, use:"
echo "python -m src.experiments.run_experiment experiment=question_type model=dummy data.languages=\"[en]\""

echo 'export HF_DATASETS_OFFLINE=1' >> ~/.bashrc
export HF_DATASETS_OFFLINE=1
echo 'export TRANSFORMERS_OFFLINE=1' >> ~/.bashrc
export TRANSFORMERS_OFFLINE=1