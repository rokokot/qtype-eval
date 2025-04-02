#!/bin/bash
# Setup script for multilingual question probing experiments

set -e  # Exit on error

echo "Setting up environment for multilingual question probing experiments..."

# Create necessary directories
mkdir -p data/cache data/features
mkdir -p outputs logs
mkdir -p configs/{data,model,training,experiment,hydra/launcher}

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install poetry first."
    echo "See https://python-poetry.org/docs/#installation for instructions."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
poetry install

# Check if TF-IDF vectors exist
echo "Checking for TF-IDF vectors..."
if [ ! -f "data/features/tfidf_vectors_train.pkl" ] || \
   [ ! -f "data/features/tfidf_vectors_dev.pkl" ] || \
   [ ! -f "data/features/tfidf_vectors_test.pkl" ]; then
    echo "WARNING: TF-IDF vectors not found in 'data/features'"
    echo "Please add your pre-extracted TF-IDF vectors before running experiments."
fi

echo -e "\nSetup complete! To run an experiment, use:"
echo "./submit_jobs.sh --experiment sklearn_baseline --model logistic --task question_type"
echo -e "\nOr for a submetric experiment:"
echo "./submit_jobs.sh --experiment sklearn_baseline --model ridge --task single_submetric --submetric avg_links_len"
