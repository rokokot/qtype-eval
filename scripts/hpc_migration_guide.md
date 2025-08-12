!# HPC Migration Guide for Enhanced Hybrid TF-IDF Features

## Current Local Setup
- **Working Directory**: `/home/robin/Research/qtype-eval`
- **Enhanced Script**: `scripts/generate_xlm_roberta_text2text_tfidf.py`
- **Target Features Dir**: `data/xlm_roberta_text2text_tfidf_enhanced`
- **SLURM Script**: `experiment_runners/tfidf_comprehensive.sh`

## Step 1: Transfer Files to HPC

### 1.1 Create Transfer Package
```bash
# From local machine (/home/robin/Research/qtype-eval)
cd /home/robin/Research/qtype-eval

# Create a clean transfer package
tar -czf qtype-eval-enhanced.tar.gz \
  --exclude='data/xlm_roberta_text2text_tfidf_features' \
  --exclude='data/xlm_roberta_text2text_tfidf_enhanced' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='venv' \
  --exclude='.git' \
  scripts/ src/ experiment_runners/ configs/ requirements.txt
```

### 1.2 Transfer to HPC
```bash
# Transfer the package to HPC
scp qtype-eval-enhanced.tar.gz your_username@login.hpc.kuleuven.be:~/
```

## Step 2: Setup on HPC

### 2.1 Login and Extract
```bash
# SSH to HPC
ssh your_username@login.hpc.kuleuven.be

# Navigate to your work directory
cd $VSC_DATA
mkdir -p qtype-eval-enhanced
cd qtype-eval-enhanced

# Extract files
tar -xzf ~/qtype-eval-enhanced.tar.gz

# Verify structure
ls -la
# Should see: scripts/ src/ experiment_runners/ configs/ requirements.txt
```

### 2.2 Setup Environment
```bash
# Load required modules
module --force purge
module load cluster/wice/oldhierarchy
module load Miniconda3/py310_22.11.1-1

# Create conda environment
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda create -n qtype-eval-enhanced python=3.10 -y
conda activate qtype-eval-enhanced

# Install packages
pip install torch transformers datasets scikit-learn xgboost pandas numpy scipy tqdm hydra-core sentencepiece
```

## Step 3: Update File Paths for HPC

### 3.1 Key HPC Paths
- **Working Directory**: `$VSC_DATA/qtype-eval-enhanced`
- **Features Output**: `$VSC_DATA/qtype-eval-enhanced/data/xlm_roberta_text2text_tfidf_enhanced`
- **Cache Directory**: `$VSC_DATA/hf_cache`
- **Results Output**: `$VSC_SCRATCH/tfidf_comprehensive_output`

### 3.2 Update SLURM Script
The script `experiment_runners/tfidf_comprehensive.sh` needs path updates:

```bash
# Line 122: Update features directory path
FEATURES_DIR="$VSC_DATA/qtype-eval-enhanced/data/xlm_roberta_text2text_tfidf_enhanced"

# Line 119: Update output base directory (already correct)
OUTPUT_BASE_DIR="$VSC_SCRATCH/tfidf_comprehensive_output"
```

## Step 4: Run Enhanced Feature Generation

### 4.1 Interactive Test (Recommended First)
```bash
# Login to compute node for testing
qsub -I -l walltime=2:00:00 -l nodes=1:ppn=4 -A intro_vsc37132

# Activate environment
cd $VSC_DATA/qtype-eval-enhanced
module --force purge
module load cluster/wice/oldhierarchy
module load Miniconda3/py310_22.11.1-1
source $EBROOTMINICONDA3/etc/profile.d/conda.sh
conda activate qtype-eval-enhanced

# Set environment variables
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME="$VSC_DATA/hf_cache"
export HF_HUB_OFFLINE=0

# Test feature generation
mkdir -p data/xlm_roberta_text2text_tfidf_enhanced
python3 scripts/generate_xlm_roberta_text2text_tfidf.py \
  --output-dir "data/xlm_roberta_text2text_tfidf_enhanced" \
  --model-name "xlm-roberta-base" \
  --max-features 128000 \
  --min-df 1 \
  --max-df 0.99 \
  --verify
```

### 4.2 Submit SLURM Job
```bash
# After successful interactive test, submit full job
cd $VSC_DATA/qtype-eval-enhanced
sbatch experiment_runners/tfidf_comprehensive.sh
```

## Step 5: Monitor and Verify

### 5.1 Check Job Status
```bash
# Check job queue
squeue -u $USER

# Check job output (replace JOBID)
tail -f slurm-JOBID.out
```

### 5.2 Verify Generated Features
```bash
# Check features directory
ls -la $VSC_DATA/qtype-eval-enhanced/data/xlm_roberta_text2text_tfidf_enhanced/

# Expected files:
# - X_train_sparse.npz, X_val_sparse.npz, X_test_sparse.npz
# - X_train.npy, X_val.npy, X_test.npy
# - metadata.json, feature_names.json
# - multilingual_analysis.json
# - token_to_index_mapping.pkl

# Check feature count
cat $VSC_DATA/qtype-eval-enhanced/data/xlm_roberta_text2text_tfidf_enhanced/metadata.json | grep vocab_size
# Should show: "vocab_size": 128000 (or close to it)
```

## Step 6: File Location Tracking

### 6.1 Key File Locations on HPC

**Enhanced Feature Generation Script:**
```
$VSC_DATA/qtype-eval-enhanced/scripts/generate_xlm_roberta_text2text_tfidf.py
```

**Generated Features Directory:**
```
$VSC_DATA/qtype-eval-enhanced/data/xlm_roberta_text2text_tfidf_enhanced/
├── X_train_sparse.npz          # Sparse training matrix
├── X_val_sparse.npz            # Sparse validation matrix  
├── X_test_sparse.npz           # Sparse test matrix
├── metadata.json               # Feature generation metadata
├── feature_names.json          # List of feature names
├── multilingual_analysis.json  # Language-specific analysis
└── token_to_index_mapping.pkl  # Vocabulary mapping
```

**Experiment Results:**
```
$VSC_SCRATCH/tfidf_comprehensive_output/
├── main/                       # Main experiments
└── control/                    # Control experiments
```

**SLURM Script:**
```
$VSC_DATA/qtype-eval-enhanced/experiment_runners/tfidf_comprehensive.sh
```

## Step 7: Enhanced Features Verification

The enhanced script now includes comprehensive monitoring:

1. **Vocabulary Analysis**: Token length distribution, multilingual patterns
2. **N-gram Analysis**: Distribution of unigrams, bigrams, trigrams
3. **TF-IDF Matrix Analysis**: Sparsity, feature frequency, sample documents
4. **Language-specific Analysis**: Top features per language
5. **Reference-style Statistics**: Similar to the reference implementation

Expected output:
- ~128,000 features (matching reference)
- Detailed multilingual token analysis
- Comprehensive feature quality statistics
- Language-specific feature patterns

## Troubleshooting

### Common Issues:
1. **Module loading fails**: Ensure `cluster/wice/oldhierarchy` is loaded first
2. **Conda activation fails**: Check `$EBROOTMINICONDA3` path exists
3. **Out of memory**: Increase `--mem` in SLURM script if needed
4. **Network timeouts**: Set `HF_HUB_OFFLINE=0` for dataset downloads

### Success Indicators:
- Feature generation completes without errors
- Metadata shows ~128k features generated
- Multilingual analysis shows language-specific patterns
- Sparsity around 99.9% (similar to reference)