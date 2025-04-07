#!/bin/bash

echo "Testing GPU access..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU available')"

# Testing model loading
echo "Testing model loading..."
python -c "from transformers import AutoModel, AutoTokenizer; model_name='cis-lmu/glot500-base'; print(f'Loading tokenizer for {model_name}...'); tokenizer = AutoTokenizer.from_pretrained(model_name); print('Tokenizer loaded successfully'); print(f'Loading model for {model_name}...'); model = AutoModel.from_pretrained(model_name); print('Model loaded successfully')"

# Testing dataset loading
echo "Testing dataset loading..."
python -c "from datasets import load_dataset; dataset_name='rokokot/question-type-and-complexity'; print(f'Loading dataset {dataset_name}...'); dataset = load_dataset(dataset_name, name='base', split='train'); print(f'Dataset loaded successfully with {len(dataset)} examples')"

echo "Debug test completed."