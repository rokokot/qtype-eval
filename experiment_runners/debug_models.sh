#!/bin/bash
#SBATCH --job-name=debug_layer_probe
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_debug
#SBATCH --clusters=wice
#SBATCH --account=intro_vsc37132

export PATH="$VSC_DATA/miniconda3/bin:$PATH"
source "$VSC_DATA/miniconda3/etc/profile.d/conda.sh"

conda activate qtype-eval
export PYTHONPATH=$PYTHONPATH:$PWD
export HF_HOME=$VSC_DATA/qtype-eval/data/cache
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1
export WANDB_DIR="$VSC_SCRATCH/wandb"
mkdir -p "$VSC_SCRATCH/wandb"

# DIAGNOSTIC MODE - Enable for debugging
DIAGNOSTIC_MODE=true

# Target languages and tasks - start with just one of each to debug
LANGUAGES=("ru")  # just use one language for debugging
TASKS=("question_type")  # start with the classification task
SUBMETRICS=()  # Skip submetrics for initial debugging
CONTROL_INDICES=()  # Skip control experiments for initial debugging

# Define a few key layers to probe first
LAYER_INDICES=(0 6 12)  # Try embedding, middle, and final layer

# Base directory for outputs
OUTPUT_BASE_DIR="$VSC_SCRATCH/debug_layer_probe_output"
mkdir -p $OUTPUT_BASE_DIR

# Create a test script to check model loading and representation extraction
cat > ${OUTPUT_BASE_DIR}/test_model_load.py << 'EOF'
#!/usr/bin/env python3
import os
import torch
from transformers import AutoModel, AutoTokenizer, logging
import numpy as np

# Set logging
logging.set_verbosity_info()

def test_model_loading(model_name):
    """Test if model loads correctly and produces reasonable representations"""
    print(f"Testing model loading for: {model_name}")
    
    try:
        # Load tokenizer with local_files_only=True since we're in offline mode
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME", "./data/cache")
        )
        print("✓ Tokenizer loaded successfully")
        
        # Load model with local_files_only=True
        model = AutoModel.from_pretrained(
            model_name, 
            output_hidden_states=True,
            local_files_only=True,
            cache_dir=os.environ.get("HF_HOME", "./data/cache")
        )
        print("✓ Model loaded successfully")
        print(f"Model type: {model.__class__.__name__}")
        
        # Print model configuration
        print(f"Model config: {model.config}")
        
        # Test tokenization
        sample_text = "What is the capital of France?"
        inputs = tokenizer(sample_text, return_tensors="pt")
        print(f"✓ Tokenization successful: {inputs['input_ids'].shape}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Check outputs
        print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
        if hasattr(outputs, 'hidden_states'):
            print(f"Number of hidden states: {len(outputs.hidden_states)}")
            for i, hidden_state in enumerate(outputs.hidden_states):
                print(f"  Layer {i} shape: {hidden_state.shape}")
                
                # Check for NaN or inf values
                if torch.isnan(hidden_state).any():
                    print(f"  ⚠️ WARNING: NaN values in layer {i}")
                if torch.isinf(hidden_state).any():
                    print(f"  ⚠️ WARNING: Inf values in layer {i}")
                    
                # Check stats
                layer_mean = torch.mean(hidden_state).item()
                layer_std = torch.std(hidden_state).item()
                layer_min = torch.min(hidden_state).item()
                layer_max = torch.max(hidden_state).item()
                
                print(f"  Stats: mean={layer_mean:.4f}, std={layer_std:.4f}, min={layer_min:.4f}, max={layer_max:.4f}")
        
        print("✓ Model forward pass successful")
        return True
    
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        model_name = "cis-lmu/glot500-base"
    else:
        model_name = sys.argv[1]
    
    test_model_loading(model_name)
EOF

chmod +x ${OUTPUT_BASE_DIR}/test_model_load.py

echo "=== Testing model loading and representation extraction ==="
python ${OUTPUT_BASE_DIR}/test_model_load.py "cis-lmu/glot500-base"

# Function to run probe experiment for a specific layer
run_layer_probe_experiment() {
    local TASK_TYPE=$1
    local LANG=$2
    local TASK=$3
    local CONTROL_IDX=$4
    local SUBMETRIC=$5
    local OUTPUT_SUBDIR=$6
    local LAYER_INDEX=$7
    local MAX_RETRIES=2
    
    local EXPERIMENT_NAME=""
    local COMMAND=""
    local EXPERIMENT_TYPE="debug_layer_probe"
    
    # Set experiment name based on parameters
    if [ -n "$SUBMETRIC" ]; then
        # This is a submetric experiment
        TASK="single_submetric"
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="debug_layer${LAYER_INDEX}_probe_${SUBMETRIC}_${LANG}"
        else
            EXPERIMENT_NAME="debug_layer${LAYER_INDEX}_probe_${SUBMETRIC}_control${CONTROL_IDX}_${LANG}"
        fi
    else
        # Regular task experiment
        if [ -z "$CONTROL_IDX" ]; then
            EXPERIMENT_NAME="debug_layer${LAYER_INDEX}_probe_${TASK}_${LANG}"
        else
            EXPERIMENT_NAME="debug_layer${LAYER_INDEX}_probe_${TASK}_control${CONTROL_IDX}_${LANG}"
        fi
    fi
    
    # Create layer-specific output directory
    local LAYER_OUTPUT_DIR="${OUTPUT_SUBDIR}/layer${LAYER_INDEX}"
    mkdir -p "$LAYER_OUTPUT_DIR"
    
    # Set task-specific configuration
    if [ "$TASK_TYPE" == "classification" ]; then
        # Classification probe configuration - SIMPLIFIED for debugging
        PROBE_CONFIG="\"model.probe_hidden_size=64\" \"model.probe_depth=1\" \"model.dropout=0.0\" \"model.activation=gelu\" \"model.normalization=none\" \"model.weight_init=xavier\""
            
        TRAINING_CONFIG="\"training.lr=5e-3\" \"training.patience=15\" \"training.scheduler_factor=0.7\" \"training.scheduler_patience=5\" \"+training.gradient_accumulation_steps=1\" \"training.weight_decay=0.0\""
    else
        # Regression probe configuration - SIMPLIFIED for debugging
        PROBE_CONFIG="\"model.probe_hidden_size=64\" \"model.probe_depth=1\" \"model.dropout=0.0\" \"model.activation=silu\" \"model.normalization=none\" \"model.weight_init=xavier\" \"model.output_standardization=true\""
            
        TRAINING_CONFIG="\"training.lr=5e-3\" \"training.patience=15\" \"training.scheduler_factor=0.7\" \"training.scheduler_patience=5\" \"+training.gradient_accumulation_steps=1\" \"training.weight_decay=0.0\""
    fi
    
    # Add debug mode flag
    DEBUG_CONFIG="+training.debug_mode=true"
    
    # Build command with explicit layer_index
    COMMAND="python -m src.experiments.run_experiment \
        \"hydra.job.chdir=False\" \
        \"hydra.run.dir=.\" \
        \"experiment=${TASK}\" \
        \"experiment.tasks=${TASK}\" \
        \"experiment.type=lm_probe\" \
        \"model=lm_probe\" \
        \"model.model_type=lm_probe\" \
        \"model.lm_name=cis-lmu/glot500-base\" \
        \"model.freeze_model=true\" \
        \"model.layer_wise=true\" \
        \"model.layer_index=${LAYER_INDEX}\" \
        \"model.use_mean_pooling=true\" \
        ${PROBE_CONFIG} \
        \"data.languages=[${LANG}]\" \
        \"data.cache_dir=$VSC_DATA/qtype-eval/data/cache\" \
        \"training.task_type=${TASK_TYPE}\" \
        \"training.num_epochs=50\" \
        \"training.batch_size=16\" \
        ${TRAINING_CONFIG} \
        ${DEBUG_CONFIG} \
        \"experiment_name=${EXPERIMENT_NAME}\" \
        \"output_dir=${LAYER_OUTPUT_DIR}\" \
        \"wandb.mode=offline\""
    
    # Add control parameters if needed
    if [ -n "$CONTROL_IDX" ]; then
        COMMAND="$COMMAND \
            \"experiment.use_controls=true\" \
            \"experiment.control_index=${CONTROL_IDX}\""
    fi
    
    # Add submetric parameter if needed
    if [ -n "$SUBMETRIC" ]; then
        COMMAND="$COMMAND \"experiment.submetric=${SUBMETRIC}\""
    fi
    
    # Execute the experiment
    echo "Running experiment: ${EXPERIMENT_NAME}"
    echo "Command: $COMMAND"
    eval $COMMAND
    RESULT=$?
    
    if [ $RESULT -eq 0 ]; then
        echo "Experiment ${EXPERIMENT_NAME} completed successfully"
        return 0
    else
        echo "Error in experiment ${EXPERIMENT_NAME}"
        
        # Clean GPU memory explicitly
        python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
        
        return 1
    fi
}

# Try different probing approaches
echo "=== Trying different probing approaches ==="

for LAYER_IDX in "${LAYER_INDICES[@]}"; do
    echo "======================="
    echo "PROBING LAYER ${LAYER_IDX}"
    echo "======================="

    for LANG in "${LANGUAGES[@]}"; do
        for TASK in "${TASKS[@]}"; do
            TASK_TYPE="classification"
            if [ "$TASK" == "complexity" ]; then
                TASK_TYPE="regression"
            fi
            
            # Create output directory
            TASK_DIR="${OUTPUT_BASE_DIR}/${TASK}"
            mkdir -p "$TASK_DIR"
            
            # Run standard (non-control) experiment for this layer
            run_layer_probe_experiment "$TASK_TYPE" "$LANG" "$TASK" "" "" "$TASK_DIR" "$LAYER_IDX"
        done
    done
done

echo "===== STAGE 2: Testing with alternative probing approaches ====="

# Now try with a linear probe instead for comparison
cat > ${OUTPUT_BASE_DIR}/linear_probe_test.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_features(model, tokenizer, texts, layer_idx, use_cls=True, use_mean=False):
    """Extract features from a specific layer"""
    model.eval()
    features = []
    
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states
            batch_features = hidden_states[layer_idx]
            
            if use_cls:
                # Use [CLS] token representation
                batch_features = batch_features[:, 0, :].cpu().numpy()
            elif use_mean:
                # Use mean pooling
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                batch_features = torch.sum(batch_features * attention_mask, 1) / torch.sum(attention_mask, 1)
                batch_features = batch_features.cpu().numpy()
            
            features.append(batch_features)
    
    return np.vstack(features)

def run_linear_probe_test(language, layer_idx, use_cls=True, use_mean=False):
    """Run a scikit-learn based linear probe test"""
    try:
        # Load model and tokenizer
        model_name = "cis-lmu/glot500-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Load dataset
        dataset_name = "rokokot/question-type-and-complexity"
        logging.info(f"Loading dataset for language {language}")
        
        try:
            train_dataset = load_dataset(dataset_name, name="base", split="train", verification_mode="no_checks")
            train_dataset = train_dataset.filter(lambda x: x["language"] == language)
            
            val_dataset = load_dataset(dataset_name, name="base", split="validation", verification_mode="no_checks")
            val_dataset = val_dataset.filter(lambda x: x["language"] == language)
            
            test_dataset = load_dataset(dataset_name, name="base", split="test", verification_mode="no_checks")
            test_dataset = test_dataset.filter(lambda x: x["language"] == language)
            
            logging.info(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test examples")
            
            # Extract texts and labels
            train_texts = train_dataset["text"]
            train_labels = train_dataset["question_type"]
            
            val_texts = val_dataset["text"]
            val_labels = val_dataset["question_type"]
            
            test_texts = test_dataset["text"]
            test_labels = test_dataset["question_type"]
            
            # Extract features
            logging.info(f"Extracting features from layer {layer_idx}")
            train_features = extract_features(model, tokenizer, train_texts, layer_idx, use_cls, use_mean)
            val_features = extract_features(model, tokenizer, val_texts, layer_idx, use_cls, use_mean)
            test_features = extract_features(model, tokenizer, test_texts, layer_idx, use_cls, use_mean)
            
            logging.info(f"Train features shape: {train_features.shape}")
            
            # Train logistic regression
            logging.info("Training logistic regression")
            classifier = LogisticRegression(max_iter=1000, C=1.0)
            classifier.fit(train_features, train_labels)
            
            # Evaluate
            train_preds = classifier.predict(train_features)
            val_preds = classifier.predict(val_features)
            test_preds = classifier.predict(test_features)
            
            # Calculate metrics
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average="macro")
            
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average="macro")
            
            test_acc = accuracy_score(test_labels, test_preds)
            test_f1 = f1_score(test_labels, test_preds, average="macro")
            
            logging.info(f"Train accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
            logging.info(f"Val accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
            logging.info(f"Test accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
            
            # More detailed report
            logging.info("\nDetailed classification report (test set):")
            logging.info(classification_report(test_labels, test_preds))
            
            return {
                "layer": layer_idx,
                "use_cls": use_cls,
                "use_mean": use_mean,
                "train_acc": train_acc,
                "train_f1": train_f1,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "test_acc": test_acc,
                "test_f1": test_f1
            }
            
        except Exception as e:
            logging.error(f"Error processing dataset: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        logging.error(f"Error in linear probe test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logging.error("Usage: linear_probe_test.py <language> <layer_idx> [use_cls=1/0] [use_mean=1/0]")
        sys.exit(1)
    
    language = sys.argv[1]
    layer_idx = int(sys.argv[2])
    
    use_cls = True
    use_mean = False
    
    if len(sys.argv) > 3:
        use_cls = sys.argv[3] == "1"
    if len(sys.argv) > 4:
        use_mean = sys.argv[4] == "1"
    
    logging.info(f"Running linear probe test for language {language}, layer {layer_idx}")
    logging.info(f"Using CLS token: {use_cls}, Using mean pooling: {use_mean}")
    
    results = run_linear_probe_test(language, layer_idx, use_cls, use_mean)
    
    if results:
        logging.info("Results:")
        for k, v in results.items():
            logging.info(f"  {k}: {v}")
EOF

chmod +x ${OUTPUT_BASE_DIR}/linear_probe_test.py

# Run the scikit-learn based linear probe as a comparison
echo "=== Running scikit-learn based linear probe tests ==="
for LAYER_IDX in "${LAYER_INDICES[@]}"; do
    for LANG in "${LANGUAGES[@]}"; do
        echo "Testing layer ${LAYER_IDX} with CLS token, language ${LANG}"
        python ${OUTPUT_BASE_DIR}/linear_probe_test.py ${LANG} ${LAYER_IDX} 1 0
        
        echo "Testing layer ${LAYER_IDX} with mean pooling, language ${LANG}"
        python ${OUTPUT_BASE_DIR}/linear_probe_test.py ${LANG} ${LAYER_IDX} 0 1
    done
done

echo "========================================"
echo "DEBUG PROBING EXPERIMENTS COMPLETED"
echo "========================================"
echo "Results available in: ${OUTPUT_BASE_DIR}"
echo "========================================"