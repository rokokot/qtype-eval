#!/bin/bash
# scripts/setup_tfidf.sh
# Setup script for TF-IDF features integration

set -e

echo "Setting up TF-IDF features for multilingual question probing..."

# Check if we're in the right directory
if [ ! -f "configs/config.yaml" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/tfidf_features
mkdir -p outputs/tfidf_experiments

# Check if HuggingFace cache is available
if [ -z "$HF_HOME" ]; then
    export HF_HOME="./data/cache"
    echo "Setting HF_HOME to $HF_HOME"
fi

# Check Python environment
echo "Checking Python environment..."
python -c "import transformers, sklearn, datasets, scipy; print('✓ All required packages available')" || {
    echo "Error: Missing required packages. Please install:"
    echo "pip install transformers scikit-learn datasets scipy xgboost"
    exit 1
}

# Generate TF-IDF features
echo "Generating TF-IDF features with Glot500 tokenizer..."
python scripts/generate_tfidf_glot500.py \
    --output-dir ./data/tfidf_features \
    --model-name cis-lmu/glot500-base \
    --max-features 50000 \
    --verify

if [ $? -eq 0 ]; then
    echo "✓ TF-IDF features generated successfully"
else
    echo "✗ TF-IDF feature generation failed"
    exit 1
fi

# Test TF-IDF feature loading
echo "Testing TF-IDF feature loading..."
python -c "
from src.data.tfidf_features import TfidfFeatureLoader
loader = TfidfFeatureLoader('./data/tfidf_features')
if loader.verify_features():
    print('✓ TF-IDF features verified successfully')
    print(f'  Vocabulary size: {loader.get_vocab_size()}')
else:
    print('✗ TF-IDF feature verification failed')
    exit(1)
"

# Test integration with existing data loading
echo "Testing integration with existing data pipeline..."
python -c "
from src.data.datasets import load_sklearn_data
try:
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
        languages=['en'], 
        task='question_type',
        vectors_dir='./data/tfidf_features',
        use_tfidf_loader=True
    )
    print(f'✓ Data loading successful')
    print(f'  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}')
    print(f'  Labels: {len(y_train)} train, {len(y_val)} val, {len(y_test)} test')
except Exception as e:
    print(f'✗ Data loading failed: {e}')
    exit(1)
"

# Test TF-IDF model creation
echo "Testing TF-IDF model creation..."
python -c "
from src.models.tfidf_baselines import create_tfidf_baseline_model
try:
    model = create_tfidf_baseline_model(
        model_type='dummy',
        task_type='classification',
        tfidf_features_dir='./data/tfidf_features',
        target_languages=['en']
    )
    print('✓ TF-IDF model creation successful')
    print(f'  Model type: {model.model_type}')
    print(f'  Task type: {model.task_type}')
except Exception as e:
    print(f'✗ TF-IDF model creation failed: {e}')
    exit(1)
"

# Run a minimal test experiment
echo "Running minimal test experiment..."
python -c "
import sys
import os
sys.path.append('.')

from src.data.datasets import load_sklearn_data
from src.models.tfidf_baselines import create_tfidf_baseline_model
from src.training.sklearn_trainer import SklearnTrainer

# Load minimal data
(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
    languages=['en'], 
    task='question_type',
    vectors_dir='./data/tfidf_features'
)

# Create and train dummy model
model = create_tfidf_baseline_model(
    model_type='dummy',
    task_type='classification', 
    tfidf_features_dir='./data/tfidf_features',
    target_languages=['en']
)

# Create trainer
trainer = SklearnTrainer(
    model=model.model,
    task_type='classification',
    output_dir='./outputs/tfidf_experiments/test'
)

# Train and evaluate
results = trainer.train(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    test_data=(X_test, y_test)
)

print('✓ Test experiment completed successfully')
print(f'  Test accuracy: {results[\"test_metrics\"][\"accuracy\"]:.3f}')
"

echo ""
echo "🎉 TF-IDF setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Run TF-IDF experiments:"
echo "   python scripts/run_tfidf_experiments.py experiment=tfidf_baselines"
echo ""
echo "2. Or run specific experiments:"
echo "   python scripts/run_tfidf_experiments.py experiment=tfidf_baselines models=[dummy,logistic] languages=[en,ru]"
echo ""
echo "3. Check results in: ./outputs/tfidf_experiments/"