#!/bin/bash
#SBATCH --job-name=test_collector
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=gpu_a100_debug
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

# Activate the environment
conda activate qtype-eval

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1

# Define a test directory
TEST_DIR="$VSC_SCRATCH/collector_test"
mkdir -p $TEST_DIR

echo "Testing results collection framework..."

# Create mock directories and result files
mkdir -p $TEST_DIR/en/layer_2/question_type
mkdir -p $TEST_DIR/en/layer_2/complexity
mkdir -p $TEST_DIR/en/layer_2/avg_links_len
mkdir -p $TEST_DIR/en/layer_2/question_type/control1
mkdir -p $TEST_DIR/en/layer_6/question_type

# Create mock result files
cat > $TEST_DIR/en/layer_2/question_type/results.json << EOF
{
  "train_time": 123.4,
  "train_metrics": {
    "loss": 0.245,
    "accuracy": 0.89,
    "f1": 0.87
  },
  "val_metrics": {
    "loss": 0.278,
    "accuracy": 0.85,
    "f1": 0.84
  },
  "test_metrics": {
    "loss": 0.283,
    "accuracy": 0.84,
    "f1": 0.83
  },
  "language": "en",
  "task": "question_type",
  "task_type": "classification",
  "model_type": "lm_probe",
  "is_control": false,
  "control_index": null,
  "layer": 2
}
EOF

cat > $TEST_DIR/en/layer_2/complexity/results.json << EOF
{
  "train_time": 145.6,
  "train_metrics": {
    "loss": 0.0026,
    "mse": 0.0026,
    "rmse": 0.051,
    "r2": 0.91
  },
  "val_metrics": {
    "loss": 0.0028,
    "mse": 0.0028,
    "rmse": 0.053,
    "r2": 0.90
  },
  "test_metrics": {
    "loss": 0.0029,
    "mse": 0.0029,
    "rmse": 0.054,
    "r2": 0.89
  },
  "language": "en",
  "task": "complexity",
  "task_type": "regression",
  "model_type": "lm_probe",
  "is_control": false,
  "control_index": null,
  "layer": 2
}
EOF

cat > $TEST_DIR/en/layer_2/question_type/control1/results.json << EOF
{
  "train_time": 124.8,
  "train_metrics": {
    "loss": 0.312,
    "accuracy": 0.76,
    "f1": 0.75
  },
  "val_metrics": {
    "loss": 0.325,
    "accuracy": 0.74,
    "f1": 0.73
  },
  "test_metrics": {
    "loss": 0.327,
    "accuracy": 0.73,
    "f1": 0.72
  },
  "language": "en",
  "task": "question_type",
  "task_type": "classification",
  "model_type": "lm_probe",
  "is_control": true,
  "control_index": 1,
  "layer": 2
}
EOF

cat > $TEST_DIR/en/layer_2/avg_links_len/results.json << EOF
{
  "train_time": 135.2,
  "train_metrics": {
    "loss": 0.0023,
    "mse": 0.0023,
    "rmse": 0.048,
    "r2": 0.92
  },
  "val_metrics": {
    "loss": 0.0025,
    "mse": 0.0025,
    "rmse": 0.050,
    "r2": 0.91
  },
  "test_metrics": {
    "loss": 0.0026,
    "mse": 0.0026,
    "rmse": 0.051,
    "r2": 0.90
  },
  "language": "en",
  "task": "single_submetric",
  "task_type": "regression",
  "model_type": "lm_probe",
  "is_control": false,
  "control_index": null,
  "layer": 2,
  "submetric": "avg_links_len"
}
EOF

cat > $TEST_DIR/en/layer_6/question_type/results.json << EOF
{
  "train_time": 127.9,
  "train_metrics": {
    "loss": 0.215,
    "accuracy": 0.92,
    "f1": 0.91
  },
  "val_metrics": {
    "loss": 0.237,
    "accuracy": 0.89,
    "f1": 0.88
  },
  "test_metrics": {
    "loss": 0.242,
    "accuracy": 0.88,
    "f1": 0.87
  },
  "language": "en",
  "task": "question_type",
  "task_type": "classification",
  "model_type": "lm_probe",
  "is_control": false,
  "control_index": null,
  "layer": 6
}
EOF

# Create extract_metrics.py
cat > $TEST_DIR/extract_metrics.py << 'EOF'
#!/usr/bin/env python3
import json
import csv
import sys
import os

def extract_metrics(result_file, tracker_file, exp_type, language, layer, task, submetric, control_index):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract test metrics
        test_metrics = data.get('test_metrics', {})
        
        # Append to tracker file
        with open(tracker_file, 'a') as f:
            writer = csv.writer(f)
            for metric, value in test_metrics.items():
                if value is not None:
                    writer.writerow([
                        exp_type, language, layer, task, 
                        submetric if submetric else 'None', 
                        control_index if control_index != 'None' else 'None',
                        metric, value
                    ])
        return True
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: extract_metrics.py <result_file> <tracker_file> <exp_type> <language> <layer> <task> <submetric> <control_index>")
        sys.exit(1)
    
    result_file = sys.argv[1]
    tracker_file = sys.argv[2]
    exp_type = sys.argv[3]
    language = sys.argv[4]
    layer = sys.argv[5]
    task = sys.argv[6]
    submetric = sys.argv[7]
    control_index = sys.argv[8]
    
    if extract_metrics(result_file, tracker_file, exp_type, language, layer, task, submetric, control_index):
        print(f"Successfully extracted metrics from {result_file}")
    else:
        print(f"Failed to extract metrics from {result_file}")
EOF

chmod +x $TEST_DIR/extract_metrics.py

# Create generate_summaries.py
cat > $TEST_DIR/generate_summaries.py << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import os

def generate_summaries(tracker_file, output_dir):
    print(f"Reading tracker file: {tracker_file}")
    df = pd.read_csv(tracker_file)
    
    # Replace empty strings with NaN for proper handling
    df = df.replace('', np.nan)
    
    print(f"Generating summaries in: {output_dir}")
    
    # 1. Layer Summary - average metrics by layer and task
    layer_summary = df.pivot_table(
        index=['layer', 'task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    layer_summary.to_csv(os.path.join(output_dir, 'layer_summary.csv'))
    print("Generated layer_summary.csv")
    
    # 2. Language Summary - average metrics by language and task
    language_summary = df.pivot_table(
        index=['language', 'task'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    language_summary.to_csv(os.path.join(output_dir, 'language_summary.csv'))
    print("Generated language_summary.csv")
    
    # 3. Submetric Summary - only for submetric tasks
    submetric_df = df[df['submetric'].notna()]
    if not submetric_df.empty:
        submetric_summary = submetric_df.pivot_table(
            index=['submetric', 'layer'], 
            columns='metric', 
            values='value',
            aggfunc='mean'
        )
        submetric_summary.to_csv(os.path.join(output_dir, 'submetric_summary.csv'))
        print("Generated submetric_summary.csv")
    
    # 4. Control Summary - compare control vs. non-control
    control_summary = df.pivot_table(
        index=['task', 'control_index'], 
        columns='metric', 
        values='value',
        aggfunc='mean'
    )
    control_summary.to_csv(os.path.join(output_dir, 'control_summary.csv'))
    print("Generated control_summary.csv")
    
    # 5. Complete matrix for all experiments
    # Reshape the data to have one row per unique experiment
    experiment_matrix = df.pivot_table(
        index=['experiment_type', 'language', 'layer', 'task', 'submetric', 'control_index'],
        columns='metric',
        values='value'
    )
    experiment_matrix.to_csv(os.path.join(output_dir, 'experiment_matrix.csv'))
    print("Generated experiment_matrix.csv")
    
    # 6. Task-specific summaries
    for task in df['task'].dropna().unique():
        task_df = df[df['task'] == task]
        task_summary = task_df.pivot_table(
            index=['language', 'layer'], 
            columns='metric', 
            values='value',
            aggfunc='mean'
        )
        task_summary.to_csv(os.path.join(output_dir, f'{task}_summary.csv'))
        print(f"Generated {task}_summary.csv")
    
    print("All summary files generated successfully!")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_summaries.py <tracker_file> <output_dir>")
        sys.exit(1)
    
    tracker_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    generate_summaries(tracker_file, output_dir)
EOF

chmod +x $TEST_DIR/generate_summaries.py

# Initialize results tracker
RESULTS_TRACKER="$TEST_DIR/results_tracker.csv"
echo "experiment_type,language,layer,task,submetric,control_index,metric,value" > $RESULTS_TRACKER

echo "Testing extract_metrics.py..."
# Test extraction of metrics
python3 $TEST_DIR/extract_metrics.py "$TEST_DIR/en/layer_2/question_type/results.json" "$RESULTS_TRACKER" \
    "standard" "en" "2" "question_type" "" "None"

python3 $TEST_DIR/extract_metrics.py "$TEST_DIR/en/layer_2/complexity/results.json" "$RESULTS_TRACKER" \
    "standard" "en" "2" "complexity" "" "None"

python3 $TEST_DIR/extract_metrics.py "$TEST_DIR/en/layer_2/question_type/control1/results.json" "$RESULTS_TRACKER" \
    "control" "en" "2" "question_type" "" "1"

python3 $TEST_DIR/extract_metrics.py "$TEST_DIR/en/layer_2/avg_links_len/results.json" "$RESULTS_TRACKER" \
    "standard" "en" "2" "single_submetric" "avg_links_len" "None"

python3 $TEST_DIR/extract_metrics.py "$TEST_DIR/en/layer_6/question_type/results.json" "$RESULTS_TRACKER" \
    "standard" "en" "6" "question_type" "" "None"

# Verify the contents of the tracker file
echo "Contents of results_tracker.csv:"
cat $RESULTS_TRACKER

echo "Testing generate_summaries.py..."
# Test generation of summary files
python3 $TEST_DIR/generate_summaries.py "$RESULTS_TRACKER" "$TEST_DIR"

echo "Verifying summary files..."
for file in layer_summary.csv language_summary.csv control_summary.csv experiment_matrix.csv question_type_summary.csv complexity_summary.csv single_submetric_summary.csv; do
    if [ -f "$TEST_DIR/$file" ]; then
        echo "✓ $file created successfully"
        echo "Contents of $file:"
        head -n 5 "$TEST_DIR/$file"
        echo "..."
    else
        echo "✗ $file was not created"
    fi
done

# Test visualization 
echo "Creating visualization test script..."
cat > $TEST_DIR/test_visualization.py << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def test_visualization(results_dir):
    # Load the experiment matrix
    matrix_file = os.path.join(results_dir, 'experiment_matrix.csv')
    if not os.path.exists(matrix_file):
        print(f"Error: {matrix_file} not found")
        return False
        
    df = pd.read_csv(matrix_file)
    print(f"Loaded data with shape: {df.shape}")
    
    # Create a simple plot to verify data is usable
    plt.figure(figsize=(10, 6))
    
    # Handle potential MultiIndex from pivot_table
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    
    # Find available metrics for classification and regression
    classification_metrics = [col for col in df.columns if col in ['accuracy', 'f1']]
    regression_metrics = [col for col in df.columns if col in ['mse', 'rmse', 'r2']]
    
    plot_success = False
    
    # Try to plot classification metrics
    if classification_metrics and 'layer' in df.columns and 'task' in df.columns:
        task_data = df[df['task'] == 'question_type']
        if not task_data.empty and 'layer' in task_data.columns:
            for metric in classification_metrics:
                if metric in task_data.columns:
                    plt.plot(task_data['layer'], task_data[metric], 'o-', label=f'{metric}')
                    plot_success = True
    
    # Try to plot regression metrics
    if not plot_success and regression_metrics and 'layer' in df.columns and 'task' in df.columns:
        task_data = df[df['task'] == 'complexity']
        if not task_data.empty and 'layer' in task_data.columns:
            for metric in regression_metrics:
                if metric in task_data.columns:
                    plt.plot(task_data['layer'], task_data[metric], 'o-', label=f'{metric}')
                    plot_success = True
    
    if plot_success:
        plt.title('Test Visualization')
        plt.xlabel('Layer')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'test_visualization.png'))
        print(f"Visualization saved to {os.path.join(results_dir, 'test_visualization.png')}")
        return True
    else:
        print("Could not create visualization - check data format")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: test_visualization.py <results_dir>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    success = test_visualization(results_dir)
    if success:
        print("Visualization test succeeded!")
    else:
        print("Visualization test failed.")
        sys.exit(1)
EOF

chmod +x $TEST_DIR/test_visualization.py

echo "Testing visualization..."
python3 $TEST_DIR/test_visualization.py "$TEST_DIR"

echo "---------------------"
echo "Test Summary:"
echo "✓ Created mock results"
echo "✓ Tested extract_metrics.py"
echo "✓ Tested generate_summaries.py"
echo "✓ Verified creation of summary files"
echo "✓ Tested visualization capability"
echo "---------------------"
echo "All tests completed!"