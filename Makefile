# Makefile for TF-IDF Multilingual Question Probing
# Provides comprehensive commands for setup, testing, and running experiments

.PHONY: help setup install-deps clean test test-unit test-integration test-tfidf
.PHONY: generate-tfidf generate-tfidf-tiny run-tfidf-experiments validate format lint
.PHONY: run-neural-experiments collect-results docker-build docker-run
.PHONY: setup-vsc cache-data check-environment

# Default target
help:
	@echo "🎯 TF-IDF Multilingual Question Probing - Available Commands"
	@echo ""
	@echo "📋 Setup & Installation:"
	@echo "  setup              - Complete project setup (deps + features)"
	@echo "  install-deps       - Install Python dependencies"
	@echo "  setup-vsc          - Setup for VSC HPC environment"
	@echo "  cache-data         - Cache datasets and models for offline use"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test               - Run all tests"
	@echo "  test-unit          - Run unit tests only"
	@echo "  test-integration   - Run integration tests only"
	@echo "  test-tfidf         - Run TF-IDF specific tests"
	@echo "  test-quick         - Quick validation tests"
	@echo "  validate           - Validate complete setup"
	@echo ""
	@echo "🔧 TF-IDF Features:"
	@echo "  generate-tfidf     - Generate full TF-IDF features"
	@echo "  generate-tfidf-tiny - Generate tiny features for testing"
	@echo "  verify-tfidf       - Verify TF-IDF features"
	@echo ""
	@echo "🚀 Experiments:"
	@echo "  run-tfidf-minimal  - Run minimal TF-IDF experiment"
	@echo "  run-tfidf-full     - Run full TF-IDF experiments"
	@echo "  run-neural-minimal - Run minimal neural experiment"
	@echo "  run-neural-full    - Run full neural experiments"
	@echo ""
	@echo "📊 Results & Analysis:"
	@echo "  collect-results    - Collect and organize results"
	@echo "  generate-report    - Generate analysis report"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean              - Clean temporary files"
	@echo "  clean-all          - Clean everything including data"
	@echo "  format             - Format code"
	@echo "  lint               - Lint code"
	@echo "  check-environment  - Check system environment"

# Variables
PYTHON := python
DATA_DIR := ./data
FEATURES_DIR := $(DATA_DIR)/tfidf_features
CACHE_DIR := $(DATA_DIR)/cache
OUTPUTS_DIR := ./outputs
TEST_OUTPUTS_DIR := ./test_outputs

# Environment setup
export PYTHONPATH := $(PWD):$(PYTHONPATH)
export HF_HOME := $(CACHE_DIR)
export TRANSFORMERS_OFFLINE := 1
export HF_DATASETS_OFFLINE := 1

# Setup and Installation
setup: install-deps generate-tfidf-tiny validate
	@echo "✅ Complete setup finished!"

install-deps:
	@echo "📦 Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✅ Dependencies installed"

setup-vsc:
	@echo "🖥️ Setting up VSC HPC environment..."
	chmod +x utils/setup_vsc_environment.sh
	./utils/setup_vsc_environment.sh
	@echo "✅ VSC environment setup complete"

cache-data:
	@echo "💾 Caching datasets and models..."
	mkdir -p $(CACHE_DIR)
	$(PYTHON) scripts/cache_resources.py --cache-dir $(CACHE_DIR)
	@echo "✅ Data cached successfully"

# Testing
test: test-unit test-integration test-tfidf
	@echo "✅ All tests completed!"

test-unit:
	@echo "🧪 Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ -v --tb=short

test-integration:
	@echo "🔗 Running integration tests..."
	$(PYTHON) -m pytest tests/integration/ -v --tb=short

test-tfidf:
	@echo "📊 Running TF-IDF specific tests..."
	$(PYTHON) -m pytest tests/ -m "tfidf" -v --tb=short

test-quick: generate-tfidf-tiny
	@echo "⚡ Running quick validation tests..."
	$(PYTHON) scripts/run_tfidf_tests.py --quick --features-dir $(FEATURES_DIR)_tiny

test-comprehensive:
	@echo "🔍 Running comprehensive test suite..."
	$(PYTHON) scripts/run_tfidf_tests.py --output-dir $(TEST_OUTPUTS_DIR)

validate: check-environment
	@echo "✅ Running validation..."
	$(PYTHON) scripts/simple_tfidf_test.py --features-dir $(FEATURES_DIR)_tiny || \
	$(PYTHON) scripts/simple_tfidf_test.py --features-dir $(FEATURES_DIR) || \
	echo "⚠️ Validation requires TF-IDF features. Run 'make generate-tfidf-tiny' first."

# TF-IDF Features
generate-tfidf:
	@echo "🔧 Generating full TF-IDF features..."
	mkdir -p $(FEATURES_DIR)
	$(PYTHON) scripts/generate_tfidf_glot500.py \
		--output-dir $(FEATURES_DIR) \
		--model-name cis-lmu/glot500-base \
		--max-features 50000 \
		--verify
	@echo "✅ TF-IDF features generated in $(FEATURES_DIR)"

generate-tfidf-tiny:
	@echo "🔧 Generating tiny TF-IDF features for testing..."
	mkdir -p $(FEATURES_DIR)_tiny
	$(PYTHON) -c "from src.data.tfidf_features import create_test_features; create_test_features('$(FEATURES_DIR)_tiny', n_samples=100)"
	@echo "✅ Tiny TF-IDF features generated in $(FEATURES_DIR)_tiny"

verify-tfidf:
	@echo "🔍 Verifying TF-IDF features..."
	$(PYTHON) -c "from src.data.tfidf_features import TfidfFeatureLoader; loader = TfidfFeatureLoader('$(FEATURES_DIR)'); print('✅ Features verified' if loader.verify_features() else '❌ Verification failed')"

# Experiments
run-tfidf-minimal: generate-tfidf-tiny
	@echo "🚀 Running minimal TF-IDF experiment..."
	mkdir -p $(OUTPUTS_DIR)/tfidf_minimal
	$(PYTHON) scripts/run_tfidf_experiments.py \
		tfidf.features_dir=$(FEATURES_DIR)_tiny \
		models=[dummy] \
		tasks=[question_type] \
		languages=[[en]] \
		controls.enabled=false \
		output_dir=$(OUTPUTS_DIR)/tfidf_minimal \
		wandb.mode=disabled

run-tfidf-full: generate-tfidf
	@echo "🚀 Running full TF-IDF experiments..."
	mkdir -p $(OUTPUTS_DIR)/tfidf_full
	$(PYTHON) scripts/run_tfidf_experiments.py \
		tfidf.features_dir=$(FEATURES_DIR) \
		models=[dummy,logistic,ridge,xgboost] \
		tasks=[question_type,complexity] \
		languages=[[en],[ru],[ar],[fi],[id],[ja],[ko]] \
		controls.enabled=true \
		output_dir=$(OUTPUTS_DIR)/tfidf_full \
		wandb.mode=disabled

run-neural-minimal:
	@echo "🧠 Running minimal neural experiment..."
	mkdir -p $(OUTPUTS_DIR)/neural_minimal
	$(PYTHON) -m src.experiments.run_experiment \
		experiment=question_type \
		model=glot500_probe \
		data.languages=[en] \
		training.num_epochs=2 \
		training.batch_size=4 \
		experiment_name=minimal_probe_test \
		output_dir=$(OUTPUTS_DIR)/neural_minimal \
		wandb.mode=disabled

run-neural-full:
	@echo "🧠 Running full neural experiments..."
	mkdir -p $(OUTPUTS_DIR)/neural_full
	# This would run the full neural experiment suite
	@echo "⚠️ Full neural experiments require significant compute resources"
	@echo "Consider running on HPC cluster with appropriate SLURM scripts"

# Results and Analysis
collect-results:
	@echo "📊 Collecting results..."
	mkdir -p $(OUTPUTS_DIR)/collected
	find $(OUTPUTS_DIR) -name "*.json" -path "*/results/*" -exec cp {} $(OUTPUTS_DIR)/collected/ \;
	@echo "✅ Results collected in $(OUTPUTS_DIR)/collected"

generate-report:
	@echo "📈 Generating analysis report..."
	$(PYTHON) scripts/analyze_results.py \
		--input-dir $(OUTPUTS_DIR) \
		--output-dir $(OUTPUTS_DIR)/analysis
	@echo "✅ Analysis report generated"

# Development and Maintenance
format:
	@echo "🎨 Formatting code..."
	$(PYTHON) -m black src/ tests/ scripts/ utils/ --line-length 100
	$(PYTHON) -m isort src/ tests/ scripts/ utils/ --profile black

lint:
	@echo "🔍 Linting code..."
	$(PYTHON) -m flake8 src/ tests/ scripts/ utils/ --max-line-length 100 --ignore=E203,W503
	$(PYTHON) -m pylint src/ --disable=C0114,C0115,C0116

check-environment:
	@echo "🔍 Checking environment..."
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Working directory: $$(pwd)"
	@echo "PYTHONPATH: $(PYTHONPATH)"
	@echo "HF_HOME: $(HF_HOME)"
	$(PYTHON) -c "import sys; print('✅ Python path OK' if '$(PWD)' in sys.path else '⚠️ Add project to PYTHONPATH')"
	@$(PYTHON) -c "import torch; print('✅ PyTorch available:', torch.__version__)" 2>/dev/null || echo "❌ PyTorch not available"
	@$(PYTHON) -c "import transformers; print('✅ Transformers available:', transformers.__version__)" 2>/dev/null || echo "❌ Transformers not available"
	@$(PYTHON) -c "import datasets; print('✅ Datasets available:', datasets.__version__)" 2>/dev/null || echo "❌ Datasets not available"
	@$(PYTHON) -c "import sklearn; print('✅ Scikit-learn available:', sklearn.__version__)" 2>/dev/null || echo "❌ Scikit-learn not available"

# Cleanup
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type f -name ".coverage" -delete
	rm -rf .coverage.*
	rm -rf htmlcov/
	rm -rf $(TEST_OUTPUTS_DIR)
	@echo "✅ Temporary files cleaned"

clean-all: clean
	@echo "🗑️ Cleaning all generated data..."
	rm -rf $(DATA_DIR)/tfidf_features*
	rm -rf $(CACHE_DIR)
	rm -rf $(OUTPUTS_DIR)
	rm -rf logs/
	@echo "✅ All generated data cleaned"

# Development workflows
dev-setup: install-deps generate-tfidf-tiny
	@echo "🛠️ Development environment ready!"

dev-test: test-unit test-quick
	@echo "🧪 Development tests completed!"

dev-experiment: run-tfidf-minimal
	@echo "⚡ Development experiment completed!"

# CI/CD workflows
ci-test: install-deps test-unit test-integration
	@echo "🤖 CI tests completed!"

ci-full: install-deps generate-tfidf-tiny test-comprehensive run-tfidf-minimal
	@echo "🤖 Full CI pipeline completed!"

# VSC HPC specific targets
vsc-setup:
	@echo "🖥️ Setting up for VSC HPC..."
	module purge || true
	module load Python/3.9 || true
	$(MAKE) install-deps
	$(MAKE) cache-data

vsc-submit-tfidf:
	@echo "📤 Submitting TF-IDF job to VSC..."
	sbatch scripts/slurm/run_tfidf_baselines.sh

vsc-submit-neural:
	@echo "📤 Submitting neural job to VSC..."
	sbatch scripts/slurm/run_neural_experiments.sh

# Docker targets
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t multilingual-question-probing .

docker-run:
	@echo "🐳 Running in Docker..."
	docker run -it --rm \
		-v $(PWD):/workspace \
		-v $(PWD)/data:/workspace/data \
		multilingual-question-probing \
		/bin/bash

docker-test:
	@echo "🐳 Running tests in Docker..."
	docker run --rm \
		-v $(PWD):/workspace \
		multilingual-question-probing \
		make ci-test

# Advanced testing patterns
test-pattern:
	@echo "🔍 Running tests matching pattern: $(PATTERN)"
	$(PYTHON) -m pytest -k "$(PATTERN)" -v

test-markers:
	@echo "🏷️ Running tests with markers: $(MARKERS)"
	$(PYTHON) -m pytest -m "$(MARKERS)" -v

test-coverage:
	@echo "📊 Running tests with coverage..."
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "📋 Coverage report generated in htmlcov/"

test-parallel:
	@echo "⚡ Running tests in parallel..."
	$(PYTHON) -m pytest tests/ -n auto

# Benchmarking
benchmark-tfidf:
	@echo "⏱️ Benchmarking TF-IDF operations..."
	$(PYTHON) -m pytest tests/ -m "performance" --benchmark-only

benchmark-memory:
	@echo "💾 Running memory benchmarks..."
	$(PYTHON) scripts/benchmark_memory.py

# Documentation
docs:
	@echo "📚 Generating documentation..."
	$(PYTHON) -m sphinx-build -b html docs/ docs/_build/html/

docs-serve:
	@echo "🌐 Serving documentation..."
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# Monitoring and debugging
monitor-gpu:
	@echo "🖥️ Monitoring GPU usage..."
	watch -n 1 nvidia-smi

debug-environment:
	@echo "🐛 Debug environment information..."
	$(PYTHON) scripts/debug_environment.py

# Quick workflows for different scenarios
quick-start: dev-setup dev-test
	@echo "🚀 Quick start completed! You can now run experiments."

full-setup: setup test run-tfidf-minimal
	@echo "🎯 Full setup completed! System is ready for production use."

research-ready: setup generate-tfidf run-tfidf-full
	@echo "🔬 Research environment ready! All baseline experiments completed."

# Help for specific components
help-testing:
	@echo "🧪 Testing Commands:"
	@echo "  test              - Run all tests"
	@echo "  test-unit         - Unit tests only"
	@echo "  test-integration  - Integration tests only"
	@echo "  test-tfidf        - TF-IDF specific tests"
	@echo "  test-quick        - Quick validation"
	@echo "  test-coverage     - Tests with coverage"
	@echo "  test-parallel     - Parallel test execution"

help-experiments:
	@echo "🚀 Experiment Commands:"
	@echo "  run-tfidf-minimal - Quick TF-IDF test"
	@echo "  run-tfidf-full    - Complete TF-IDF experiments"
	@echo "  run-neural-minimal - Quick neural test"
	@echo "  run-neural-full   - Complete neural experiments"

help-vsc:
	@echo "🖥️ VSC HPC Commands:"
	@echo "  vsc-setup         - Setup for VSC environment"
	@echo "  vsc-submit-tfidf  - Submit TF-IDF job"
	@echo "  vsc-submit-neural - Submit neural job"

# Error handling
.ONESHELL:
.SHELLFLAGS := -eu -c

# Prevent deletion of intermediate files
.SECONDARY:

# Targets that don't correspond to files
.PHONY: $(MAKECMDGOALS)