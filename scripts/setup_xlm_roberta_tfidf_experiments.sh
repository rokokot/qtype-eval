#!/bin/bash
# scripts/setup_xlm_roberta_tfidf_experiments.sh
# Complete setup for XLM-RoBERTa-consistent TF-IDF experiments

set -e

echo "======================================================================"
echo "Setting up XLM-RoBERTa-Consistent TF-IDF Experiments"
echo "======================================================================"
echo ""
echo "This script ensures TF-IDF experiments use the same XLM-RoBERTa tokenizer"
echo "as your neural experiments for perfect consistency."
echo ""

# Check if we're in the right directory
if [ ! -f "configs/config.yaml" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/xlm_roberta_tfidf_features
mkdir -p outputs/xlm_roberta_tfidf_experiments

# Check Python environment
echo "Checking Python environment..."
python -c "import sys; print(f'Python {sys.version}')"

# Install required dependencies
echo "Installing required dependencies..."
pip install transformers scikit-learn datasets scipy xgboost pandas

# Verify installations
echo "Verifying installations..."
python -c "
try:
    from transformers import AutoTokenizer
    print('✓ transformers library available')
except ImportError:
    print('✗ transformers installation failed')
    exit(1)
"

python -c "
try:
    import xgboost as xgb
    print('✓ xgboost library available')
except ImportError:
    print('✗ xgboost installation failed')
    exit(1)
"

# Check if HuggingFace cache is available
if [ -z "$HF_HOME" ]; then
    export HF_HOME="./data/cache"
    echo "Setting HF_HOME to $HF_HOME"
fi

# Verify XLM-RoBERTa tokenizer accessibility
echo "Verifying XLM-RoBERTa tokenizer..."
python -c "
from transformers import AutoTokenizer
import os

try:
    # Test loading XLM-RoBERTa tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'xlm-roberta-base',
        cache_dir=os.environ.get('HF_HOME', './data/cache')
    )
    
    # Verify configuration matches requirements
    expected_tokens = {
        'bos_token': '<s>',
        'eos_token': '</s>',
        'cls_token': '<s>',
        'sep_token': '</s>',
        'pad_token': '<pad>',
        'unk_token': '<unk>'
    }
    
    print('✓ XLM-RoBERTa tokenizer loaded successfully')
    print(f'  Vocab size: {len(tokenizer)}')
    print(f'  Model max length: {tokenizer.model_max_length}')
    print(f'  Tokenizer class: {tokenizer.__class__.__name__}')
    
    # Check special tokens
    all_match = True
    for token_name, expected_token in expected_tokens.items():
        actual_token = getattr(tokenizer, token_name, None)
        if actual_token == expected_token:
            print(f'  ✓ {token_name}: {actual_token}')
        else:
            print(f'  ✗ {token_name}: {actual_token} (expected {expected_token})')
            all_match = False
    
    if all_match:
        print('✓ All special tokens match requirements')
    else:
        print('⚠️ Some special tokens do not match - this may affect consistency')
        
except Exception as e:
    print(f'✗ XLM-RoBERTa tokenizer test failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "✗ XLM-RoBERTa tokenizer verification failed"
    exit 1
fi

# Generate XLM-RoBERTa-consistent TF-IDF features
echo ""
echo "Generating XLM-RoBERTa-consistent TF-IDF features..."
python scripts/generate_xlm_roberta_tfidf.py \
    --output-dir ./data/xlm_roberta_tfidf_features \
    --model-name xlm-roberta-base \
    --max-features 50000 \
    --min-df 2 \
    --max-df 0.95 \
    --random-state 42 \
    --check-tokenizer \
    --verify

if [ $? -eq 0 ]; then
    echo "✓ XLM-RoBERTa TF-IDF features generated successfully"
else
    echo "✗ XLM-RoBERTa TF-IDF feature generation failed"
    exit 1
fi

# Verify feature consistency
echo ""
echo "Verifying feature consistency..."
python -c "
import json
from pathlib import Path

# Load metadata to verify tokenizer consistency
metadata_file = Path('./data/xlm_roberta_tfidf_features/metadata.json')
if metadata_file.exists():
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    tokenizer_info = metadata.get('tokenizer_info', {})
    print('Feature generation details:')
    print(f'  Tokenizer: {tokenizer_info.get(\"model_name\", \"unknown\")}')
    print(f'  Vocab size: {metadata.get(\"vocab_size\", \"unknown\")}')
    print(f'  Feature shapes: {metadata.get(\"feature_shape\", {})}')
    
    # Check if using XLM-RoBERTa
    if 'xlm-roberta' in tokenizer_info.get('model_name', '').lower():
        print('✓ Features use XLM-RoBERTa tokenizer')
    else:
        print('✗ Features do not use XLM-RoBERTa tokenizer')
        exit(1)
else:
    print('✗ No metadata file found')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "✗ Feature consistency verification failed"
    exit 1
fi

# Test data loading integration
echo ""
echo "Testing data loading integration..."
python -c "
import sys
sys.path.append('.')

from src.data.datasets import load_sklearn_data_with_config

try:
    # Test loading with XLM-RoBERTa features
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data_with_config(
        task='question_type',
        languages=['en'], 
        dataset_config='base',
        tfidf_features_dir='./data/xlm_roberta_tfidf_features',
        use_tfidf_loader=True
    )
    
    print('✓ Data loading integration successful')
    print(f'  Train: X={X_train.shape}, y={len(y_train)}')
    print(f'  Val: X={X_val.shape}, y={len(y_val)}')
    print(f'  Test: X={X_test.shape}, y={len(y_test)}')
    
    # Verify reasonable shapes
    if X_train.shape[0] > 0 and X_train.shape[1] > 0:
        print('✓ Feature dimensions look reasonable')
    else:
        print('✗ Unexpected feature dimensions')
        exit(1)
        
except Exception as e:
    print(f'✗ Data loading integration failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "✗ Data loading integration test failed"
    exit 1
fi

# Run XLM-RoBERTa-consistent TF-IDF experiments
echo ""
echo "Running XLM-RoBERTa-consistent TF-IDF experiments..."
python scripts/run_xlm_roberta_tfidf_experiments.py \
    --tfidf-features-dir ./data/xlm_roberta_tfidf_features \
    --output-dir ./outputs/xlm_roberta_tfidf_experiments \
    --cache-dir ./data/cache \
    --random-state 42

if [ $? -eq 0 ]; then
    echo "✓ XLM-RoBERTa-consistent experiments completed successfully"
else
    echo "✗ XLM-RoBERTa-consistent experiments failed"
    exit 1
fi

# Display results summary
echo ""
echo "======================================================================"
echo "XLM-RoBERTa-Consistent TF-IDF Experiments Complete"
echo "======================================================================"
echo ""
echo "✅ SUCCESS: All experiments completed with tokenizer consistency!"
echo ""
echo "Key Benefits:"
echo "  • TF-IDF and neural experiments now use identical XLM-RoBERTa tokenization"
echo "  • Fair comparison between baseline and neural approaches"
echo "  • Consistent vocabulary and token handling across all methods"
echo ""
echo "Results available in:"
echo "  - ./outputs/xlm_roberta_tfidf_experiments/all_results.json"
echo "  - ./outputs/xlm_roberta_tfidf_experiments/results_summary.csv"
echo "  - ./outputs/xlm_roberta_tfidf_experiments/analysis_report.md"
echo ""
echo "Features saved in:"
echo "  - ./data/xlm_roberta_tfidf_features/"
echo ""

# Show quick summary if results exist
if [ -f "./outputs/xlm_roberta_tfidf_experiments/results_summary.csv" ]; then
    echo "Quick Results Summary:"
    echo "----------------------"
    python -c "
import pandas as pd
try:
    df = pd.read_csv('./outputs/xlm_roberta_tfidf_experiments/results_summary.csv')
    print(f'Total experiments: {len(df)}')
    print(f'Tokenizer consistent: {df[\"tokenizer_consistent\"].sum()}/{len(df)}')
    
    # Show performance by task and model
    if 'test_accuracy' in df.columns:
        acc_summary = df.groupby(['task', 'model_type'])['test_accuracy'].agg(['count', 'mean', 'std']).round(4)
        print('\\nClassification Performance (Accuracy):')
        print(acc_summary)
    
    if 'test_r2' in df.columns:
        r2_summary = df.groupby(['task', 'model_type'])['test_r2'].agg(['count', 'mean', 'std']).round(4)
        print('\\nRegression Performance (R²):')
        print(r2_summary)
        
except Exception as e:
    print(f'Could not generate summary: {e}')
"
echo ""
fi

echo "Next Steps:"
echo "1. Compare TF-IDF results with your neural model results"
echo "2. Verify that tokenizer consistency improves baseline comparisons"
echo "3. Use these results to validate neural model improvements"
echo ""
echo "For detailed analysis, see: ./outputs/xlm_roberta_tfidf_experiments/analysis_report.md"