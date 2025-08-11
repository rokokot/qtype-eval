#!/usr/bin/env python3
"""
Test script for hybrid XLM-RoBERTa + text2text TF-IDF features.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.generate_xlm_roberta_text2text_tfidf import generate_xlm_roberta_text2text_features
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test the hybrid feature generation."""
    
    # Generate features
    output_dir = "./data/xlm_roberta_text2text_tfidf_features"
    
    logger.info("🚀 Testing hybrid XLM-RoBERTa + text2text TF-IDF feature generation...")
    
    try:
        metadata = generate_xlm_roberta_text2text_features(
            output_dir=output_dir,
            model_name="xlm-roberta-base",
            max_features=128000,  # Match reference implementation
            min_df=2,
            max_df=0.95,
            verify=True
        )
        
        logger.info("✅ Hybrid feature generation completed successfully!")
        
        # Print key statistics
        print("\n" + "="*60)
        print("HYBRID TFIDF FEATURES SUMMARY")
        print("="*60)
        
        print(f"📊 Features generated: {metadata['vocab_size']:,}")
        print(f"📊 XLM-RoBERTa vocab size: {metadata['tokenizer_info']['vocab_size']:,}")
        
        print(f"\n📏 Feature matrix shapes:")
        for split, shape in metadata['feature_shape'].items():
            sparsity = metadata['generation_info']['sparsity'][split]
            avg_features = metadata['generation_info']['avg_features_per_doc'][split]
            print(f"  {split:>5}: {shape[0]:>5} × {shape[1]:>6} (sparsity: {sparsity:.3f}, avg features/doc: {avg_features:.1f})")
        
        print(f"\n🌍 Language-specific top features:")
        multilingual = metadata['multilingual_analysis']['language_specific_features']
        for lang, features in multilingual.items():
            top_3 = features[:3]
            feature_strs = [f"'{f}' ({s:.4f})" for f, s in top_3]
            print(f"  {lang}: {', '.join(feature_strs)}")
        
        print(f"\n📈 Vocabulary statistics:")
        vocab_stats = metadata['vocab_analysis']
        print(f"  Total features: {vocab_stats['total_features']:,}")
        print(f"  Avg token length: {vocab_stats['avg_token_length']:.1f} chars")
        
        print(f"\n🔤 Token length distribution:")
        for length, count in sorted(vocab_stats['token_lengths'].items())[:10]:
            print(f"  Length {length}: {count:,} tokens")
        
        print(f"\n🎯 Comparison with current approach:")
        print(f"  Current (XLM-RoBERTa direct): ~18k features")  
        print(f"  Reference (text2text): ~128k features")
        print(f"  Hybrid (this approach): {metadata['vocab_size']:,} features")
        
        print(f"\n📁 Features saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Hybrid feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)