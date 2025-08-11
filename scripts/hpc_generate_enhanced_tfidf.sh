#!/bin/bash
#SBATCH --job-name=enhanced_tfidf_generation
#SBATCH --clusters=wice
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=batch
#SBATCH --account=intro_vsc37132

echo "========================================================================="
echo "Enhanced Hybrid TF-IDF Feature Generation on HPC"
echo "========================================================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_USER: $SLURM_JOB_USER"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "========================================================================="

# Environment setup
echo "🔄 Setting up environment..."

module --force purge
module load cluster/wice/oldhierarchy

if module load Miniconda3/py310_22.11.1-1 2>/dev/null; then
    echo "✅ Miniconda module loaded successfully"
    if [ -f "$EBROOTMINICONDA3/etc/profile.d/conda.sh" ]; then
        source $EBROOTMINICONDA3/etc/profile.d/conda.sh
        conda activate qtype-eval-enhanced
    else
        echo "❌ Conda initialization script not found"
        exit 1
    fi
else
    echo "❌ Miniconda module failed to load"
    exit 1
fi

# Set up environment variables for HPC
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME="$VSC_DATA/hf_cache"
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HYDRA_JOB_CHDIR=False
export HYDRA_FULL_ERROR=1

# Verify environment
echo "🐍 Environment verification:"
echo "  Python path: $(which python)"
echo "  Python version: $(python --version)"
echo "  Conda environment: $CONDA_DEFAULT_ENV"
echo "  Working directory: $(pwd)"
echo "  HF cache: $HF_HOME"

# Define paths (HPC-specific)
FEATURES_DIR="$VSC_DATA/qtype-eval-enhanced/data/xlm_roberta_text2text_tfidf_enhanced"
SCRIPT_PATH="$VSC_DATA/qtype-eval-enhanced/scripts/generate_xlm_roberta_text2text_tfidf.py"

echo ""
echo "📁 HPC File Paths:"
echo "  Script: $SCRIPT_PATH"
echo "  Output: $FEATURES_DIR"
echo "  Cache: $HF_HOME"

# Create output directory
mkdir -p "$FEATURES_DIR"

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ Script not found at: $SCRIPT_PATH"
    echo "Please ensure files are properly transferred to HPC"
    exit 1
fi

echo ""
echo "🚀 Starting Enhanced Hybrid TF-IDF Feature Generation..."
echo "This combines XLM-RoBERTa tokenization with text2text methodology"
echo "Target: ~128,000 features with comprehensive monitoring"
echo ""

# Run enhanced feature generation with monitoring
python3 "$SCRIPT_PATH" \
    --output-dir "$FEATURES_DIR" \
    --model-name "xlm-roberta-base" \
    --max-features 128000 \
    --min-df 1 \
    --max-df 0.99 \
    --verify \
    2>&1 | tee "${FEATURES_DIR}/generation.log"

# Check if generation succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✅ Enhanced hybrid TF-IDF feature generation completed successfully!"
    echo ""
    
    # Display summary information
    if [ -f "$FEATURES_DIR/metadata.json" ]; then
        echo "📊 Generation Summary:"
        
        # Extract key statistics
        VOCAB_SIZE=$(grep -o '"vocab_size":[^,]*' "$FEATURES_DIR/metadata.json" | cut -d':' -f2)
        echo "  Features generated: $VOCAB_SIZE"
        
        # Check file sizes
        echo ""
        echo "📁 Generated Files:"
        ls -lh "$FEATURES_DIR" | grep -E '\.(npz|json|npy|pkl)$'
        
        # Show available analysis files
        echo ""
        echo "📈 Analysis Files Available:"
        [ -f "$FEATURES_DIR/metadata.json" ] && echo "  ✅ metadata.json - Generation metadata"
        [ -f "$FEATURES_DIR/multilingual_analysis.json" ] && echo "  ✅ multilingual_analysis.json - Language-specific analysis"  
        [ -f "$FEATURES_DIR/feature_names.json" ] && echo "  ✅ feature_names.json - Complete feature vocabulary"
        
        echo ""
        echo "🎯 Next Steps:"
        echo "1. Update SLURM experiment script to use: $FEATURES_DIR"
        echo "2. Run comprehensive experiments: sbatch experiment_runners/tfidf_comprehensive.sh"
        echo "3. Compare results with previous 18k feature approach"
        
    else
        echo "⚠️ Metadata file not found, but generation may have succeeded"
    fi
    
    echo ""
    echo "📍 Features Location: $FEATURES_DIR"
    echo "📍 Generation Log: ${FEATURES_DIR}/generation.log"
    
else
    echo "❌ Enhanced hybrid TF-IDF feature generation failed"
    echo "Check the error messages above and generation log at:"
    echo "  ${FEATURES_DIR}/generation.log"
    exit 1
fi

echo ""
echo "========================================================================="
echo "Enhanced Hybrid TF-IDF Feature Generation Complete"
echo "========================================================================="