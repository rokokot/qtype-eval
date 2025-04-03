#!/bin/bash
set -e  

echo "Starting all glot500 experiments..."

echo "Running basic experiments..."
bash run_basic_experiments.sh

echo "Running control experiments..."
bash run_control_experiments.sh

echo "Running submetric experiments..."
bash run_all_submetrics.sh

echo "Running cross-lingual experiments..."
bash run_all_cross_lingual.sh

echo "Analyzing results..."
python scripts/analysis/analyze_results.py --results-dir outputs/ --output-dir analysis_results

echo "All experiments completed!"