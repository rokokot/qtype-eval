# scripts/generate_tfidf.py
"""
TF-IDF feature generation using Glot500 tokenizer.
Completely isolated - does not affect existing codebase.
"""


import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Glot500TfidfGenerator:    
    def __init__(
        self,
        model_name: str = 'cis-lmu/glot500-base',
        max_features: int = 50000,
        min_df: int = 2,
        max_df: float = 0.95,
        cache_dir: Optional[str] = None):

        self.model_name = model_name
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        logger.info(f"Glot500 tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir)
        self.vectorizer = None
        self.feature_names = None
        
    def glot500_tokenize(self, text: str):
        
        try:
            tokens = self.tokenizer.tokenize(text)
            tokens = [t for t in tokens if t and not t.startswith('[')]
            return tokens
        except Exception as e:
            logger.warning(f"tok failed for text: {text[:50]} : {e}")
            return []
    
    def create_vectorizer(self) -> TfidfVectorizer:

        return TfidfVectorizer(
            tokenizer=self.glot500_tokenize,
            lowercase=False, 
            token_pattern=None,  
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True)
    
    def generate_features(self,train_texts: List[str],
        val_texts: List[str],
        test_texts: List[str],
        output_dir: str) -> Dict[str, np.ndarray]:
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(" tfidf vectorizer with Glot500 tokenizer")
        self.vectorizer = self.create_vectorizer()
        
        logger.info(f"Fitting vectorizer on {len(train_texts)} training data")
        X_train = self.vectorizer.fit_transform(train_texts)
        
        logger.info(f"Transforming validation set ({len(val_texts)} texts)...")
        X_val = self.vectorizer.transform(val_texts)
        
        logger.info(f"Transforming test set ({len(test_texts)} texts)...")
        X_test = self.vectorizer.transform(test_texts)
        
        logger.info("Saving TF-IDF features...")
        features = {
            'train': X_train,
            'val': X_val, 
            'test': X_test}
        
        for split, X in features.items():
            np.save(output_path / f"X_{split}.npy", X.toarray())
            
        with open(output_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
            
        self.feature_names = self.vectorizer.get_feature_names_out()
        metadata = {
            'model_name': self.model_name,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'vocab_size': len(self.feature_names),
            'feature_shape': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape}
        }
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        with open(output_path / "feature_names.json", "w") as f:
            json.dump(self.feature_names.tolist(), f)
        
        logger.info(f"features saved to {output_path}")
        logger.info(f"Vocabulary size: {len(self.feature_names)}")
        logger.info(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return features

def load_dataset_splits(dataset_name: str = "rokokot/question-type-and-complexity"):

    logger.info(f"Loading dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name, "base")
    
    train_df = dataset['train'].to_pandas()
    val_df = dataset['validation'].to_pandas() 
    test_df = dataset['test'].to_pandas()
    
    train_texts = train_df['text'].tolist()
    val_texts = val_df['text'].tolist()
    test_texts = test_df['text'].tolist()
    
    logger.info(f"Loaded splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    return train_texts, val_texts, test_texts, train_df, val_df, test_df

def main():

    import argparse
    
    parser = argparse.ArgumentParser(description="Generate TF-IDF features with Glot500 tokenizer")
    parser.add_argument("--output-dir", default="./data/tfidf_features", help="Output directory")
    parser.add_argument("--model-name", default="cis-lmu/glot500-base", help="Glot500 model name")
    parser.add_argument("--max-features", type=int, default=50000, help="Maximum features")
    parser.add_argument("--cache-dir", default="./data/cache", help="HuggingFace cache directory")
    parser.add_argument("--dataset-name", default="rokokot/question-type-and-complexity", help="Dataset name")
    
    args = parser.parse_args()
    
    generator = Glot500TfidfGenerator(
        model_name=args.model_name,
        max_features=args.max_features,
        cache_dir=args.cache_dir)
    
    train_texts, val_texts, test_texts, train_df, val_df, test_df = load_dataset_splits(args.dataset_name)
    
    features = generator.generate_features(train_texts, val_texts, test_texts, args.output_dir)
    
    language_info = {'train': train_df['language'].tolist(),
        'val': val_df['language'].tolist(),
        'test': test_df['language'].tolist()}
    
    with open(Path(args.output_dir) / "language_info.json", "w") as f:
        json.dump(language_info, f)
    
    logger.info("feature generation completed successfully!")

if __name__ == "__main__":
    main()