#!/bin/bash
#
# Generate hybrid XLM-RoBERTa + text2text TF-IDF features
#

echo "🚀 Generating hybrid XLM-RoBERTa + text2text TF-IDF features..."
echo "This combines XLM-RoBERTa tokenization with text2text TF-IDF methodology"
echo "Expected to generate ~128k features like the reference implementation"
echo ""

# Set up environment
export PYTHONPATH="$PWD:$PYTHONPATH"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Output directory
OUTPUT_DIR="./data/xlm_roberta_text2text_tfidf_features"
echo "📁 Output directory: $OUTPUT_DIR"

# Generate features
python3 scripts/generate_xlm_roberta_text2text_tfidf.py \
    --output-dir "$OUTPUT_DIR" \
    --model-name "xlm-roberta-base" \
    --max-features 128000 \
    --min-df 2 \
    --max-df 0.95 \
    --verify

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Hybrid TF-IDF feature generation completed successfully!"
    echo "📁 Features saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Update SLURM script to use new features directory"
    echo "2. Run comprehensive experiments with hybrid features"
    echo "3. Compare results with current XLM-RoBERTa approach"
else
    echo "❌ Hybrid TF-IDF feature generation failed"
    exit 1
fi