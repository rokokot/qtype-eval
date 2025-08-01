# Makefile - Enhanced with TF-IDF baseline support
# Preserves all existing functionality + adds TF-IDF commands

.PHONY: help setup install test clean
.PHONY: generate-tfidf test-tfidf run-tfidf-experiments
.PHONY: run-experiments run-baselines cache-data

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup commands:"
	@echo "  setup                 - Complete project setup"
	@echo "  install               - Install Python dependencies"
	@echo "  cache-data            - Cache HuggingFace datasets and models"
	@echo ""
	@echo "TF-IDF baseline commands (recommended for testing):"
	@echo "  generate-tfidf-tiny   - Generate minimal TF-IDF features (100 features) for testing"
	@echo "  test-tfidf-setup      - Test TF-IDF setup step-by-step (recommended first step)"
	@echo "  generate-tfidf-small  - Generate small TF-IDF features (1000 features)"
	@echo "  generate-tfidf        - Generate full TF-IDF features (50000 features)"
	@echo "  run-tfidf-minimal     - Run minimal TF-IDF test (dummy model, English only)"
	@echo "  run-tfidf-test        - Run small TF-IDF test (dummy+logistic, English only)"
	@echo "  run-tfidf-experiments - Run all TF-IDF baseline experiments"
	@echo ""
	@echo "Neural model commands:"
	@echo "  run-experiments       - Run neural model experiments (existing)"
	@echo "  run-baselines         - Run traditional sklearn baselines (existing)"
	@echo ""
	@echo "Utility commands:"
	@echo "  test                  - Run test suite"
	@echo "  clean                 - Clean temporary files"

# Setup commands
setup: install cache-data
	@echo "✓ Project setup completed"

install:
	pip install -r requirements.txt
	pip install xgboost  # Ensure XGBoost is available for TF-IDF baselines

cache-data:
	python scripts/cache_resources.py --cache-dir ${HF_HOME:-./data/cache}

# TF-IDF commands
generate-tfidf:
	@echo "Generating TF-IDF features with Glot500 tokenizer..."
	mkdir -p data/tfidf_features
	python scripts/generate_tfidf_glot500.py \
		--output-dir ./data/tfidf_features \
		--model-name cis-lmu/glot500-base \
		--max-features 50000 \
		--cache-dir ./data/cache \
		--verify

# Small-scale test version for development/testing
generate-tfidf-small:
	@echo "Generating small TF-IDF features for testing..."
	mkdir -p data/tfidf_features_test
	python scripts/generate_tfidf_glot500.py \
		--output-dir ./data/tfidf_features_test \
		--model-name cis-lmu/glot500-base \
		--max-features 1000 \
		--min-df 1 \
		--cache-dir ./data/cache \
		--verify

# Tiny test version (minimal features for quick testing)
generate-tfidf-tiny:
	@echo "Generating tiny TF-IDF features for quick testing..."
	mkdir -p data/tfidf_features_tiny
	python scripts/generate_tfidf_glot500.py \
		--output-dir ./data/tfidf_features_tiny \
		--model-name cis-lmu/glot500-base \
		--max-features 100 \
		--min-df 1 \
		--cache-dir ./data/cache \
		--verify

# Simple diagnostic test
test-tfidf-setup:
	@echo "Running comprehensive TF-IDF setup test..."
	python scripts/simple_tfidf_test.py --features-dir ./data/tfidf_features_tiny

test-tfidf-setup-small:
	@echo "Running TF-IDF setup test with small features..."
	python scripts/simple_tfidf_test.py --features-dir ./data/tfidf_features_test

setup-tfidf-small: generate-tfidf-small test-tfidf
	@echo "✓ Small TF-IDF setup completed successfully"
	@echo ""
	@echo "You can now run small TF-IDF experiments with:"
	@echo "  make run-tfidf-test"

setup-tfidf-tiny: generate-tfidf-tiny 
	@echo "✓ Tiny TF-IDF setup completed successfully"
	@echo ""
	@echo "You can now test TF-IDF with minimal features"

# Quick TF-IDF test with tiny features
run-tfidf-test:
	@echo "Running TF-IDF test with small features..."
	python scripts/run_tfidf_experiments.py \
		experiment=tfidf_baselines \
		models=[dummy,logistic] \
		languages=[en] \
		tasks=[question_type] \
		controls.enabled=false \
		tfidf.features_dir=./data/tfidf_features_test \
		output_dir=./outputs/tfidf_test

# Minimal TF-IDF test with tiny features  
run-tfidf-minimal:
	@echo "Running minimal TF-IDF test..."
	python scripts/run_tfidf_experiments.py \
		experiment=tfidf_baselines \
		models=[dummy] \
		languages=[en] \
		tasks=[question_type] \
		controls.enabled=false \
		tfidf.features_dir=./data/tfidf_features_tiny \
		output_dir=./outputs/tfidf_minimal

run-tfidf-experiments:
	@echo "Running TF-IDF baseline experiments..."
	mkdir -p outputs/tfidf_experiments
	python scripts/run_tfidf_experiments.py \
		experiment=tfidf_baselines \
		output_dir=./outputs/tfidf_experiments

# Quick TF-IDF test with minimal models/languages
test-tfidf-quick:
	@echo "Running quick TF-IDF test..."
	python scripts/run_tfidf_experiments.py \
		experiment=tfidf_baselines \
		models=[dummy,logistic] \
		languages=[en] \
		tasks=[question_type] \
		controls.enabled=false \
		output_dir=./outputs/tfidf_test

# Run TF-IDF for specific configuration
run-tfidf-custom:
	@echo "Running custom TF-IDF experiments..."
	@echo "Usage: make run-tfidf-custom MODELS='[dummy,ridge]' LANGUAGES='[en,ru]' TASKS='[question_type]'"
	python scripts/run_tfidf_experiments.py \
		experiment=tfidf_baselines \
		models=${MODELS:-[dummy,logistic]} \
		languages=${LANGUAGES:-[en]} \
		tasks=${TASKS:-[question_type]} \
		output_dir=./outputs/tfidf_custom

# Existing commands (preserved for backward compatibility)
run-experiments:
	@echo "Running neural model experiments..."
	python -m src.experiments.run_experiment \
		experiment=question_type \
		model=lm_probe \
		data.languages=[en]

run-baselines:
	@echo "Running traditional sklearn baselines..."
	python -m src.experiments.run_experiment \
		experiment=question_type \
		model=dummy \
		data.languages=[en]

# Testing
test:
	pytest tests/

test-all: test test-tfidf
	@echo "✓ All tests completed"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf outputs/tfidf_test/
	@echo "✓ Cleaned temporary files"

clean-tfidf:
	rm -rf data/tfidf_features/
	rm -rf outputs/tfidf_experiments/
	@echo "✓ Cleaned TF-IDF files"

# Verify everything is working
verify-setup: test-tfidf
	@echo "✓ Setup verification completed"

# Development helpers
check-tfidf-features:
	@echo "Checking TF-IDF features status..."
	@if [ -d "data/tfidf_features" ]; then \
		echo "✓ TF-IDF features directory exists"; \
		python -c "from src.data.tfidf_features import TfidfFeatureLoader; loader = TfidfFeatureLoader('./data/tfidf_features'); print(f'Vocab size: {loader.get_vocab_size()}') if loader.verify_features() else print('Features invalid')"; \
	else \
		echo "✗ TF-IDF features not found. Run 'make generate-tfidf'"; \
	fi

# Show current status
status:
	@echo "Project Status:"
	@echo "==============="
	@echo -n "HuggingFace cache: "
	@if [ -d "${HF_HOME:-./data/cache}" ]; then echo "✓"; else echo "✗"; fi
	@echo -n "TF-IDF features: "
	@if [ -d "data/tfidf_features" ]; then echo "✓"; else echo "✗"; fi
	@echo -n "Requirements: "
	@python -c "import transformers, sklearn, datasets, scipy; print('✓')" 2>/dev/null || echo "✗"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make setup          # Initial setup"
	@echo "  2. make setup-tfidf    # Setup TF-IDF baselines"
	@echo "  3. make run-tfidf-experiments  # Run experiments"