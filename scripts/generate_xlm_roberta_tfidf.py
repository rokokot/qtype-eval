#!/usr/bin/env python3
"""
Generate TF-IDF features using XLM-RoBERTa tokenizer for consistency with neural experiments.
This ensures the same tokenization is used across all experimental conditions.
"""

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
from scipy.sparse import save_npz, load_npz, csr_matrix
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class XLMRobertaTfidfGenerator:
    """Generate TF-IDF features using XLM-RoBERTa tokenizer for consistency."""
    
    def __init__(
        self,
        model_name: str = 'xlm-roberta-base',
        max_features: int = 50000,  # Reasonable size for TF-IDF
        min_df: int = 2,
        max_df: float = 0.95,
        cache_dir: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize with XLM-RoBERTa tokenizer matching your configuration.
        
        Args:
            model_name: XLM-RoBERTa model name
            max_features: Maximum number of TF-IDF features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            cache_dir: HuggingFace cache directory
            random_state: Random seed for reproducibility
        """
        self.model_name = model_name
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.random_state = random_state
        
        # Set numpy random seed for reproducibility
        np.random.seed(random_state)
        
        logger.info(f"Loading XLM-RoBERTa tokenizer: {model_name}")
        try:
            # Load tokenizer with exact configuration from your experiments
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1",
                use_fast=True  # Your config shows use_fast=true
            )
            
            # Verify tokenizer configuration matches your requirements
            expected_special_tokens = {
                'bos_token': '<s>',
                'cls_token': '<s>', 
                'eos_token': '</s>',
                'pad_token': '<pad>',
                'sep_token': '</s>',
                'unk_token': '<unk>'
            }
            
            for token_type, expected_token in expected_special_tokens.items():
                actual_token = getattr(self.tokenizer, token_type, None)
                if actual_token != expected_token:
                    logger.warning(f"Token mismatch: {token_type} = {actual_token}, expected {expected_token}")
            
            logger.info(f"✓ Tokenizer loaded successfully. Vocab size: {len(self.tokenizer)}")
            logger.info(f"✓ Model max length: {self.tokenizer.model_max_length}")
            logger.info(f"✓ Tokenizer class: {self.tokenizer.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        self.vectorizer = None
        self.feature_names = None
    
    def _compute_data_hash(self, train_texts: List[str], val_texts: List[str], 
                          test_texts: List[str]) -> str:
        """Compute hash of input data for consistency checking."""
        all_texts = train_texts + val_texts + test_texts
        text_sample = '|'.join(sorted(all_texts[:100]))  # Sample for efficiency
        config_str = f"{self.model_name}|{self.max_features}|{self.min_df}|{self.max_df}|{self.random_state}"
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
            
            # Verify tokenizer consistency
            stored_tokenizer = metadata.get('tokenizer_info', {})
            if stored_tokenizer.get('model_name') != self.model_name:
                logger.warning(f"Tokenizer model mismatch")
                return False
            
            logger.info("✓ Existing features found and validated")
            return True
            
        except Exception as e:
            logger.warning(f"Error validating existing features: {e}")
            return False
    
    def xlm_roberta_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using XLM-RoBERTa tokenizer exactly as in neural experiments.
        
        This ensures identical tokenization across TF-IDF and neural model experiments.
        """
        try:
            # Use the same tokenization as your neural experiments
            tokens = self.tokenizer.tokenize(text)
            
            # Filter out special tokens for TF-IDF (keeping only content tokens)
            special_tokens = {
                self.tokenizer.bos_token, self.tokenizer.eos_token,
                self.tokenizer.cls_token, self.tokenizer.sep_token,
                self.tokenizer.pad_token, self.tokenizer.unk_token
            }
            
            # Remove special tokens and empty tokens
            content_tokens = [
                token for token in tokens 
                if token and token not in special_tokens
            ]
            
            # Handle subword tokens (▁ prefix in XLM-RoBERTa)
            # Option 1: Keep subwords as-is for exact neural model consistency
            processed_tokens = content_tokens
            
            # Option 2: Merge subwords back to words (uncomment if preferred)
            # processed_tokens = []
            # current_word = ""
            # for token in content_tokens:
            #     if token.startswith('▁'):  # Start of new word
            #         if current_word:
            #             processed_tokens.append(current_word)
            #         current_word = token[1:]  # Remove ▁ prefix
            #     else:
            #         current_word += token
            # if current_word:
            #     processed_tokens.append(current_word)
            
            return processed_tokens
            
        except Exception as e:
            logger.warning(f"Tokenization failed for text: {text[:50]}... Error: {e}")
            # Fallback to simple split
            return text.lower().split()
    
    def create_vectorizer(self) -> TfidfVectorizer:
        """Create TF-IDF vectorizer with XLM-RoBERTa tokenization."""
        return TfidfVectorizer(
            tokenizer=self.xlm_roberta_tokenize,
            lowercase=False,  # XLM-RoBERTa handles casing internally
            token_pattern=None,  # We provide our own tokenizer
            ngram_range=(1, 2),  # Unigrams + bigrams
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
        """Generate TF-IDF features for all splits with XLM-RoBERTa tokenization."""
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
        
        logger.info("Creating TF-IDF vectorizer with XLM-RoBERTa tokenizer")
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
        
        # Save comprehensive metadata
        metadata = {
            'tokenizer_info': {
                'model_name': self.model_name,
                'tokenizer_class': self.tokenizer.__class__.__name__,
                'vocab_size': len(self.tokenizer),
                'model_max_length': self.tokenizer.model_max_length,
                'special_tokens': {
                    'bos_token': self.tokenizer.bos_token,
                    'eos_token': self.tokenizer.eos_token,
                    'cls_token': self.tokenizer.cls_token,
                    'sep_token': self.tokenizer.sep_token,
                    'pad_token': self.tokenizer.pad_token,
                    'unk_token': self.tokenizer.unk_token,
                },
                'use_fast': getattr(self.tokenizer, 'is_fast', False)
            },
            'tfidf_params': {
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': (1, 2),
                'norm': 'l2',
                'use_idf': True,
                'smooth_idf': True,
                'sublinear_tf': True
            },
            'random_state': self.random_state,
            'data_hash': data_hash,
            'vocab_size': len(self.feature_names),
            'feature_shape': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            },
            'generation_info': {
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
        
        # Verify tokenizer consistency
        logger.info("Tokenizer verification:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Vocab size: {len(self.tokenizer)}")
        logger.info(f"  Max length: {self.tokenizer.model_max_length}")
        logger.info(f"  Special tokens: {metadata['tokenizer_info']['special_tokens']}")
        
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


def verify_tokenizer_consistency(
    tokenizer_config_path: Optional[str] = None,
    model_name: str = 'xlm-roberta-base'
) -> bool:
    """Verify that the tokenizer matches the configuration from neural experiments."""
    
    logger.info("Verifying tokenizer consistency...")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return False
    
    # Expected configuration from your neural experiments
    expected_config = {
        "model_max_length": 512,
        "name_or_path": "xlm-roberta-base",
        "tokenizer_class": "XLMRobertaTokenizer",
        "use_fast": True,
        "special_tokens": {
            "bos_token": "<s>",
            "cls_token": "<s>", 
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "sep_token": "</s>",
            "unk_token": "<unk>"
        }
    }
    
    # Check configuration
    checks_passed = 0
    total_checks = 0
    
    # Check model name
    total_checks += 1
    if hasattr(tokenizer, 'name_or_path') and expected_config["name_or_path"] in tokenizer.name_or_path:
        checks_passed += 1
        logger.info(f"✓ Model name: {tokenizer.name_or_path}")
    else:
        logger.warning(f"✗ Model name mismatch: {getattr(tokenizer, 'name_or_path', 'unknown')}")
    
    # Check tokenizer class
    total_checks += 1
    if tokenizer.__class__.__name__ == expected_config["tokenizer_class"]:
        checks_passed += 1
        logger.info(f"✓ Tokenizer class: {tokenizer.__class__.__name__}")
    else:
        logger.warning(f"✗ Tokenizer class: {tokenizer.__class__.__name__} (expected {expected_config['tokenizer_class']})")
    
    # Check special tokens
    for token_name, expected_token in expected_config["special_tokens"].items():
        total_checks += 1
        actual_token = getattr(tokenizer, token_name, None)
        if actual_token == expected_token:
            checks_passed += 1
            logger.info(f"✓ {token_name}: {actual_token}")
        else:
            logger.warning(f"✗ {token_name}: {actual_token} (expected {expected_token})")
    
    # Check model max length
    total_checks += 1
    if tokenizer.model_max_length == expected_config["model_max_length"]:
        checks_passed += 1
        logger.info(f"✓ Model max length: {tokenizer.model_max_length}")
    else:
        logger.warning(f"✗ Model max length: {tokenizer.model_max_length} (expected {expected_config['model_max_length']})")
    
    consistency_score = checks_passed / total_checks
    logger.info(f"Tokenizer consistency: {checks_passed}/{total_checks} ({consistency_score:.1%})")
    
    return consistency_score >= 0.8  # 80% consistency threshold


def verify_features(output_dir: str) -> bool:
    """Verify that generated features are valid and loadable."""
    logger.info("Verifying generated features...")
    output_path = Path(output_dir)
    
    try:
        # Check required files exist
        required_files = [
            "X_train.npy", "X_val.npy", "X_test.npy",
            "metadata.json", "vectorizer.pkl", "feature_names.json"
        ]
        
        missing_files = []
        for filename in required_files:
            if not (output_path / filename).exists():
                missing_files.append(filename)
        
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        # Try to load each feature matrix
        for split in ['train', 'val', 'test']:
            try:
                X = np.load(output_path / f"X_{split}.npy")
                logger.info(f"✓ {split} features: {X.shape}")
                
                # Basic sanity checks
                if X.shape[0] == 0:
                    logger.error(f"✗ {split} features have 0 samples")
                    return False
                
                if X.shape[1] == 0:
                    logger.error(f"✗ {split} features have 0 features")
                    return False
                
                # Check for reasonable values
                if not np.all(np.isfinite(X)):
                    logger.error(f"✗ {split} features contain non-finite values")
                    return False
                
            except Exception as e:
                logger.error(f"✗ Failed to load {split} features: {e}")
                return False
        
        # Load and verify metadata
        try:
            with open(output_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Check tokenizer info
            tokenizer_info = metadata.get('tokenizer_info', {})
            if tokenizer_info.get('model_name') == 'xlm-roberta-base':
                logger.info("✓ Uses XLM-RoBERTa tokenizer")
            else:
                logger.warning(f"✗ Unexpected tokenizer: {tokenizer_info.get('model_name')}")
            
            logger.info(f"✓ Metadata: vocab_size={metadata.get('vocab_size')}")
            
        except Exception as e:
            logger.error(f"✗ Failed to load metadata: {e}")
            return False
        
        # Load and verify vectorizer
        try:
            with open(output_path / "vectorizer.pkl", 'rb') as f:
                vectorizer = pickle.load(f)
            logger.info("✓ Vectorizer loaded successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to load vectorizer: {e}")
            return False
        
        logger.info("✓ All feature verification checks passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Feature verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate TF-IDF features with XLM-RoBERTa tokenizer")
    parser.add_argument("--output-dir", default="./data/xlm_roberta_tfidf_features", 
                       help="Output directory for TF-IDF features")
    parser.add_argument("--model-name", default="xlm-roberta-base", 
                       help="XLM-RoBERTa model name for tokenization")
    parser.add_argument("--max-features", type=int, default=50000, 
                       help="Maximum number of TF-IDF features")
    parser.add_argument("--min-df", type=int, default=2,
                       help="Minimum document frequency")
    parser.add_argument("--max-df", type=float, default=0.95,
                       help="Maximum document frequency")
    parser.add_argument("--cache-dir", default=None,
                       help="HuggingFace cache directory")
    parser.add_argument("--dataset-name", default="rokokot/question-type-and-complexity",
                       help="Dataset name on HuggingFace Hub")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducible feature generation")
    parser.add_argument("--force-regenerate", action="store_true",
                       help="Force regeneration even if valid features exist")
    parser.add_argument("--verify", action="store_true",
                       help="Verify generated features after creation")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing features without generating")
    parser.add_argument("--check-tokenizer", action="store_true",
                       help="Check tokenizer consistency with neural experiments")
    
    args = parser.parse_args()
    
    # Use environment variables if not specified
    if args.cache_dir is None:
        args.cache_dir = os.environ.get("HF_HOME", "./data/cache")
    
    # Check tokenizer consistency first
    if args.check_tokenizer:
        if verify_tokenizer_consistency(model_name=args.model_name):
            logger.info("✓ Tokenizer consistency check passed!")
        else:
            logger.error("✗ Tokenizer consistency check failed!")
            return 1
    
    # If only verification requested
    if args.verify_only:
        if verify_features(args.output_dir):
            logger.info("✓ Feature verification successful!")
            return 0
        else:
            logger.error("✗ Feature verification failed!")
            return 1
    
    try:
        # Create generator
        generator = XLMRobertaTfidfGenerator(
            model_name=args.model_name,
            max_features=args.max_features,
            min_df=args.min_df,
            max_df=args.max_df,
            cache_dir=args.cache_dir,
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
            if verify_features(args.output_dir):
                logger.info("✓ Feature verification successful!")
            else:
                logger.error("✗ Feature verification failed!")
                return 1
        
        logger.info("✓ XLM-RoBERTa TF-IDF feature generation completed successfully!")
        logger.info(f"Features saved to: {args.output_dir}")
        logger.info("Features are now consistent with neural experiment tokenization!")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ XLM-RoBERTa TF-IDF feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())