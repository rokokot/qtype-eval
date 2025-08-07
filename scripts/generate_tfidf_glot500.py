"""Generate TF-IDF features using Glot500 tokenizer with seeded reproducibility."""

import os
import json
import pickle
import numpy as np
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from scipy.sparse import save_npz, load_npz
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Glot500TfidfGenerator:
    """Generate TF-IDF features using Glot500 tokenizer for consistency with neural experiments."""
    
    def __init__(
        self,
        model_name: str = 'cis-lmu/glot500-base',
        max_features: int = 128000,
        min_df: int = 2,
        max_df: float = 0.95,
        cache_dir: Optional[str] = None,
        use_subword_tokens: bool = True,
        random_state: int = 42
    ):
        self.model_name = model_name
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.use_subword_tokens = use_subword_tokens
        self.random_state = random_state
        
        # Set numpy random seed for reproducibility
        np.random.seed(random_state)
        
        logger.info(f"Loading Glot500 tokenizer: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1",
		use_fast=False
            )
            logger.info(f"Tokenizer loaded successfully. Vocab size: {len(self.tokenizer)}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        self.vectorizer = None
        self.feature_names = None
    
    def _compute_data_hash(self, train_texts: List[str], val_texts: List[str], 
                          test_texts: List[str]) -> str:
        """Compute hash of input data for consistency checking."""
        # Create a deterministic hash of the input data
        all_texts = train_texts + val_texts + test_texts
        text_sample = '|'.join(sorted(all_texts[:100]))  # Sample for efficiency
        config_str = f"{self.model_name}|{self.max_features}|{self.min_df}|{self.max_df}|{self.use_subword_tokens}|{self.random_state}"
        combined = f"{text_sample}|{config_str}|{len(train_texts)}|{len(val_texts)}|{len(test_texts)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _features_exist_and_valid(self, output_dir: str, expected_hash: str) -> bool:
        """Check if features already exist and are valid."""
        output_path = Path(output_dir)
        
        # Check if metadata file exists
        metadata_file = output_path / "metadata.json"
        if not metadata_file.exists():
            return False
        
        # Check if all required files exist
        required_files = [
            "X_train.npy", "X_val.npy", "X_test.npy",
            "vectorizer.pkl", "feature_names.json", "metadata.json"
        ]
        
        for filename in required_files:
            if not (output_path / filename).exists():
                logger.warning(f"Missing required file: {filename}")
                return False
        
        # Check metadata hash
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            stored_hash = metadata.get('data_hash', '')
            if stored_hash != expected_hash:
                logger.warning(f"Data hash mismatch. Stored: {stored_hash[:8]}, Expected: {expected_hash[:8]}")
                return False
            
            logger.info("✓ Existing features found and validated")
            return True
            
        except Exception as e:
            logger.warning(f"Error validating existing features: {e}")
            return False
    
    def glot500_tokenize(self, text: str) -> List[str]:
        """Tokenize text using Glot500 tokenizer with proper handling."""
        try:
            if self.use_subword_tokens:
                # Use subword tokenization (like BERT tokens)
                tokens = self.tokenizer.tokenize(text)
                # Filter out special tokens and empty tokens
                tokens = [t for t in tokens if t and not t.startswith('[') and not t.startswith('<')]
            else:
                # Use word-level tokenization (split on tokenized boundaries)
                tokens = self.tokenizer.tokenize(text)
                # Join subword tokens back into words (remove ## prefixes)
                words = []
                current_word = ""
                for token in tokens:
                    if token.startswith('##'):
                        current_word += token[2:]
                    else:
                        if current_word:
                            words.append(current_word)
                        current_word = token
                if current_word:
                    words.append(current_word)
                tokens = words
            
            return tokens
        except Exception as e:
            logger.warning(f"Tokenization failed for text: {text[:50]}... Error: {e}")
            # Fallback to simple split
            return text.lower().split()
    
    def create_vectorizer(self) -> TfidfVectorizer:
        """Create TF-IDF vectorizer with Glot500 tokenization."""
        return TfidfVectorizer(
            tokenizer=self.glot500_tokenize,
            lowercase=False,  # Glot500 handles casing internally
            token_pattern=None,  # We provide our own tokenizer
            ngram_range=(1, 2),  # Subword unigrams + bigrams
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            stop_words=None  # No stop words for multilingual setting
        )
    
    def generate_features(
        self,
        train_texts: List[str],
        val_texts: List[str], 
        test_texts: List[str],
        output_dir: str,
        languages_info: Optional[Dict[str, List[str]]] = None,
        force_regenerate: bool = False
    ) -> Dict[str, np.ndarray]:
        """Generate TF-IDF features for all splits with checkpointing."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute data hash for consistency checking
        data_hash = self._compute_data_hash(train_texts, val_texts, test_texts)
        logger.info(f"Data hash: {data_hash[:8]}")
        
        # Check if features already exist and are valid
        if not force_regenerate and self._features_exist_and_valid(output_dir, data_hash):
            logger.info("Loading existing features from cache...")
            features = {}
            for split in ['train', 'val', 'test']:
                features[split] = np.load(output_path / f"X_{split}.npy")
                logger.info(f"Loaded {split} features: {features[split].shape}")
            return features
        
        logger.info("Creating TF-IDF vectorizer with Glot500 tokenizer")
        self.vectorizer = self.create_vectorizer()
        
        logger.info(f"Fitting vectorizer on {len(train_texts)} training texts")
        X_train = self.vectorizer.fit_transform(train_texts)
        
        logger.info(f"Transforming validation set ({len(val_texts)} texts)")
        X_val = self.vectorizer.transform(val_texts)
        
        logger.info(f"Transforming test set ({len(test_texts)} texts)")
        X_test = self.vectorizer.transform(test_texts)
        
        # Get feature names
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info("Saving TF-IDF features...")
        
        # Save features in multiple formats for compatibility
        features = {
            'train': X_train,
            'val': X_val,
            'test': X_test
        }
        
        # Save as numpy arrays (dense format for easier loading)
        for split, X in features.items():
            np.save(output_path / f"X_{split}.npy", X.toarray())
            logger.info(f"Saved {split} features: {X.shape}")
        
        # Save as scipy sparse matrices (for memory efficiency)
        for split, X in features.items():
            save_npz(output_path / f"X_{split}_sparse.npz", X)
        
        # Save as pickle for backward compatibility with existing code
        for split, X in features.items():
            pkl_split = 'dev' if split == 'val' else split
            with open(output_path / f"tfidf_vectors_{pkl_split}.pkl", "wb") as f:
                pickle.dump(X, f)
        
        # Save vectorizer
        with open(output_path / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        
        # Save feature names
        with open(output_path / "feature_names.json", "w") as f:
            json.dump(self.feature_names.tolist(), f)
        
        # Save language information if provided
        if languages_info:
            with open(output_path / "language_info.json", "w") as f:
                json.dump(languages_info, f)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'use_subword_tokens': self.use_subword_tokens,
            'random_state': self.random_state,
            'data_hash': data_hash,
            'vocab_size': len(self.feature_names),
            'feature_shape': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            },
            'generation_info': {
                'tokenizer_vocab_size': len(self.tokenizer),
                'actual_features': len(self.feature_names),
                'sparsity': {
                    'train': 1 - (X_train.nnz / np.prod(X_train.shape)),
                    'val': 1 - (X_val.nnz / np.prod(X_val.shape)),
                    'test': 1 - (X_test.nnz / np.prod(X_test.shape))
                }
            }
        }
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"TF-IDF features saved to {output_path}")
        logger.info(f"Vocabulary size: {len(self.feature_names)}")
        logger.info(f"Feature shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Log sparsity information
        for split, X in features.items():
            sparsity = 1 - (X.nnz / np.prod(X.shape))
            logger.info(f"{split.capitalize()} sparsity: {sparsity:.2%}")
        
        return features

def load_dataset_splits(
    dataset_name: str = "rokokot/question-type-and-complexity",
    cache_dir: Optional[str] = None
) -> Tuple[List[str], List[str], List[str], Dict[str, List[str]]]:
    """Load dataset and extract texts with language information."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        # Load all splits
        dataset = load_dataset(
            dataset_name, 
            "base",
            cache_dir=cache_dir,
            verification_mode='no_checks'
        )
        
        # Convert to DataFrames for easier manipulation
        train_df = dataset['train'].to_pandas()
        val_df = dataset['validation'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        # Extract texts
        train_texts = train_df['text'].tolist()
        val_texts = val_df['text'].tolist() 
        test_texts = test_df['text'].tolist()
        
        # Extract language information
        languages_info = {
            'train': train_df['language'].tolist(),
            'val': val_df['language'].tolist(),
            'test': test_df['language'].tolist()
        }
        
        # Log statistics
        for split, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            lang_counts = df['language'].value_counts()
            logger.info(f"{split.capitalize()} split: {len(df)} examples")
            logger.info(f"  Languages: {dict(lang_counts)}")
        
        return train_texts, val_texts, test_texts, languages_info
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate TF-IDF features with Glot500 tokenizer")
    parser.add_argument("--output-dir", default="./data/tfidf_features", 
                       help="Output directory for TF-IDF features")
    parser.add_argument("--model-name", default="cis-lmu/glot500-base", 
                       help="Glot500 model name for tokenization")
    parser.add_argument("--max-features", type=int, default=128000, 
                       help="Maximum number of TF-IDF features")
    parser.add_argument("--min-df", type=int, default=2,
                       help="Minimum document frequency")
    parser.add_argument("--max-df", type=float, default=0.95,
                       help="Maximum document frequency")
    parser.add_argument("--cache-dir", default=None,
                       help="HuggingFace cache directory")
    parser.add_argument("--dataset-name", default="rokokot/question-type-and-complexity",
                       help="Dataset name on HuggingFace Hub")
    parser.add_argument("--use-subword-tokens", action="store_true", default=True,
                       help="Use subword tokens (default) vs word-level tokens")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducible feature generation")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration even if valid features exist")
    parser.add_argument("--verify", action="store_true",
                       help="Verify generated features after creation")
    
    args = parser.parse_args()
    
    # Use environment variables if not specified
    if args.cache_dir is None:
        args.cache_dir = os.environ.get("HF_HOME", "./data/cache")
    
    # Create generator
    generator = Glot500TfidfGenerator(
        model_name=args.model_name,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        cache_dir=args.cache_dir,
        use_subword_tokens=args.use_subword_tokens,
        random_state=args.random_state
    )
    
    # Load dataset
    train_texts, val_texts, test_texts, languages_info = load_dataset_splits(
        args.dataset_name, args.cache_dir
    )
    
    # Generate features
    features = generator.generate_features(
        train_texts, val_texts, test_texts, 
        args.output_dir, languages_info,
        force_regenerate=args.force_regenerate
    )
    
    # Verify features if requested
    if args.verify:
        logger.info("Verifying generated features...")
        from src.data.tfidf_features import TfidfFeatureLoader
        loader = TfidfFeatureLoader(args.output_dir)
        if loader.verify_features():
            logger.info("✓ Feature verification successful!")
        else:
            logger.error("✗ Feature verification failed!")
            return 1
    
    logger.info("TF-IDF feature generation completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
