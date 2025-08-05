# Alternative Makefile that skips editable install
.PHONY: help setup dev-setup-simple clean test
.DEFAULT_GOAL := help

PYTHON := python3
DATA_DIR := data
MODELS_DIR := models

$(DATA_DIR) $(MODELS_DIR):
	mkdir -p $@

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: $(DATA_DIR) $(MODELS_DIR) ## Basic setup without editable install
	$(PIP) install -r requirements.txt
	@echo "Basic setup complete"

dev-setup-simple: setup ## Simple dev setup (skip editable install)
	@echo "Adding current directory to PYTHONPATH"
	@echo "export PYTHONPATH=\$$PYTHONPATH:$(PWD)" >> ~/.bashrc
	@echo "Development setup complete (without editable install)"
	@echo "Run: source ~/.bashrc  OR  export PYTHONPATH=\$$PYTHONPATH:$(PWD)"

run-tfidf-direct: $(DATA_DIR) ## Run TF-IDF test directly
	PYTHONPATH=$(PWD) $(PYTHON) scripts/simple_tfidf_test.py --data-dir $(DATA_DIR)

generate-tfidf-tiny: $(DATA_DIR) ## Generate tiny TF-IDF features
	PYTHONPATH=$(PWD) $(PYTHON) scripts/generate_tfidf.py --output-dir $(DATA_DIR) --vocab-size 100 --max-samples 50

clean: ## Clean build artifacts
	find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .eggs/
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true