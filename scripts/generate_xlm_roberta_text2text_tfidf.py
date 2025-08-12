#!/usr/bin/env python3
"""
Hybrid TF-IDF feature extractor combining XLM-RoBERTa tokenization with text2text methodology.
This approach uses XLM-RoBERTa's multilingual subword tokenization but applies text2text's
superior TF-IDF feature extraction approach for better multilingual handling.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import scipy.sparse as sparse
from tqdm.auto import tqdm
import logging

from datasets import load_dataset
from transformers import XLMRobertaTokenizerFast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XLMRobertaText2TextTfidfExtractor:
    """
    Hybrid TF-IDF extractor using XLM-RoBERTa tokenization with text2text methodology.
    """
    
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        max_features: int = 32000,   # Reduced from 128000 to focus on most informative features
        min_df: int = 3,         # Increased from 1 to remove very rare terms (noise reduction)
        max_df: float = 0.95,    # Reduced from 0.99 to filter more common terms  
        ngram_range: Tuple[int, int] = (1, 2),  # Keep trigrams as they capture important phrase patterns
        use_full_vocab: bool = False,
        random_state: int = 42
    ):
        """
        Initialize the hybrid TF-IDF extractor.
        
        Args:
            model_name: XLM-RoBERTa model name
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            ngram_range: N-gram range for features
            use_full_vocab: Whether to use full XLM-RoBERTa vocab (250k) or dataset vocab
            random_state: Random seed
        """
        self.model_name = model_name
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_full_vocab = use_full_vocab
        self.random_state = random_state
        
        # Initialize tokenizer
        logger.info(f"Loading XLM-RoBERTa tokenizer: {model_name}")
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            model_name,
            local_files_only=False
        )
        
        # Initialize vectorizer (will be fitted later)
        self.vectorizer = None
        self.is_fitted = False
        
        # Store vocab mapping
        self.vocab_to_index = {}
        self.index_to_vocab = {}
        
    def _preprocess_texts(self, texts: List[str], languages: Optional[List[str]] = None) -> List[str]:
        """
        Preprocess texts using XLM-RoBERTa tokenization.
        
        Args:
            texts: List of input texts
            languages: Optional language codes (for logging/analysis)
            
        Returns:
            List of preprocessed texts with subword tokens
        """
        preprocessed = []
        
        for i, text in enumerate(tqdm(texts, desc="Tokenizing with XLM-RoBERTa")):
            # Tokenize using XLM-RoBERTa
            tokens = self.tokenizer.tokenize(text)
            
            # Join tokens with spaces for sklearn compatibility
            tokenized_text = " ".join(tokens)
            preprocessed.append(tokenized_text)
            
            # Log example for first few texts
            if i < 3:
                lang = languages[i] if languages else "unknown"
                logger.info(f"Example {i+1} ({lang}): {text[:50]}...")
                logger.info(f"  Tokens: {tokens[:10]}...")
        
        return preprocessed
    
    def _create_custom_vectorizer(self, preprocessed_texts: List[str]) -> TfidfVectorizer:
        """
        Create TfidfVectorizer with XLM-RoBERTa token preprocessing and proper n-gram support.
        """
        vectorizer = TfidfVectorizer(
            analyzer='word',    # Use word analyzer to enable n-grams
            preprocessor=None,  # We already preprocessed
            tokenizer=str.split,  # Simple split on spaces since we pre-tokenized
            token_pattern=None,   # Don't use regex token pattern
            lowercase=False,    # XLM-RoBERTa tokens are already processed
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            norm='l2',          # L2 normalization like text2text
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True   # Use log scaling
        )
        
        return vectorizer
    
    def fit(self, texts: List[str], languages: Optional[List[str]] = None) -> 'XLMRobertaText2TextTfidfExtractor':
        """
        Fit the TF-IDF vectorizer on the training texts.
        
        Args:
            texts: Training texts
            languages: Optional language codes
            
        Returns:
            Self for chaining
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts")
        
        # Preprocess texts using XLM-RoBERTa
        preprocessed_texts = self._preprocess_texts(texts, languages)
        
        # Create and fit vectorizer
        self.vectorizer = self._create_custom_vectorizer(preprocessed_texts)
        self.vectorizer.fit(preprocessed_texts)
        
        # Store vocabulary mapping
        self.vocab_to_index = self.vectorizer.vocabulary_
        self.index_to_vocab = {idx: token for token, idx in self.vocab_to_index.items()}
        
        self.is_fitted = True
        
        logger.info(f"Fitted vectorizer with {len(self.vocab_to_index)} features")
        logger.info(f"Feature range: {min(self.vocab_to_index.values())} to {max(self.vocab_to_index.values())}")
        
        # Detailed vocabulary analysis like reference implementation
        self._analyze_fitted_vocabulary()
        
        # Log some example features
        sample_features = list(self.vocab_to_index.items())[:10]
        logger.info(f"Sample features: {sample_features}")
        
        return self
    
    def _analyze_fitted_vocabulary(self):
        """Analyze fitted vocabulary with detailed statistics like reference implementation."""
        if not self.vocab_to_index:
            return
        
        logger.info("\n" + "="*50)
        logger.info("VOCABULARY ANALYSIS (Reference Implementation Style)")
        logger.info("="*50)
        
        feature_names = list(self.vocab_to_index.keys())
        
        # Basic statistics
        token_lengths = [len(token) for token in feature_names]
        avg_len = sum(token_lengths) / len(token_lengths)
        
        logger.info(f"Total features: {len(feature_names):,}")
        logger.info(f"Average token length: {avg_len:.2f} characters")
        logger.info(f"Min token length: {min(token_lengths)} characters")
        logger.info(f"Max token length: {max(token_lengths)} characters")
        
        # Count by first character (multilingual analysis)
        from collections import Counter
        first_chars = Counter([token[0] if token else '' for token in feature_names])
        logger.info("\nMost common first characters:")
        for char, count in first_chars.most_common(10):
            percentage = (count / len(feature_names)) * 100
            logger.info(f"  '{char}': {count:,} features ({percentage:.1f}%)")
        
        # Token length distribution
        length_counts = Counter(token_lengths)
        logger.info("\nFeature count by token length:")
        for length in sorted(length_counts.keys())[:15]:  # Show first 15 lengths
            count = length_counts[length]
            percentage = (count / len(feature_names)) * 100
            logger.info(f"  Length {length}: {count:,} features ({percentage:.1f}%)")
        
        # Sample tokens by length (like reference)
        logger.info("\nSample features by length:")
        for length in range(1, min(11, max(token_lengths) + 1)):
            tokens_of_length = [t for t in feature_names if len(t) == length]
            if tokens_of_length:
                sample = tokens_of_length[:8]  # Show up to 8 examples
                logger.info(f"  Length {length}: {', '.join(repr(t) for t in sample)}")
        
        # Multilingual token analysis
        self._analyze_multilingual_tokens(feature_names)
        
        # N-gram analysis
        self._analyze_ngrams(feature_names)
        
        logger.info("="*50)
    
    def _analyze_multilingual_tokens(self, feature_names):
        """Analyze multilingual characteristics of tokens."""
        logger.info("\nMultilingual Token Analysis:")
        
        # Count tokens starting with multilingual prefixes
        multilingual_patterns = {
            '▁': 'Subword prefix tokens',
            'ا': 'Arabic tokens',
            'в': 'Russian tokens', 
            'は': 'Japanese tokens',
            'の': 'Japanese particles',
            '은': 'Korean particles',
            '는': 'Korean particles',
            'ä': 'Finnish/German tokens',
            'ö': 'Finnish/German tokens'
        }
        
        for pattern, description in multilingual_patterns.items():
            count = sum(1 for token in feature_names if token.startswith(pattern))
            if count > 0:
                percentage = (count / len(feature_names)) * 100
                logger.info(f"  {description}: {count:,} ({percentage:.2f}%)")
        
        # Question markers analysis (key for our task)
        question_markers = ['?', '؟', '▁هل', '▁Apakah', '▁apakah', '▁Onko', '▁ли']
        logger.info("\nQuestion Marker Features:")
        for marker in question_markers:
            if marker in feature_names:
                idx = self.vocab_to_index[marker]
                logger.info(f"  '{marker}' -> feature index {idx}")
    
    def _analyze_ngrams(self, feature_names):
        """Analyze n-gram distribution."""
        logger.info("\nN-gram Analysis:")
        
        unigrams = [f for f in feature_names if ' ' not in f]
        bigrams = [f for f in feature_names if f.count(' ') == 1]
        trigrams = [f for f in feature_names if f.count(' ') == 2]
        higher_grams = [f for f in feature_names if f.count(' ') >= 3]
        
        total = len(feature_names)
        logger.info(f"  Unigrams (1-gram): {len(unigrams):,} ({len(unigrams)/total*100:.1f}%)")
        logger.info(f"  Bigrams (2-gram): {len(bigrams):,} ({len(bigrams)/total*100:.1f}%)")
        logger.info(f"  Trigrams (3-gram): {len(trigrams):,} ({len(trigrams)/total*100:.1f}%)")
        if higher_grams:
            logger.info(f"  Higher n-grams: {len(higher_grams):,} ({len(higher_grams)/total*100:.1f}%)")
        
        # Show sample n-grams
        if bigrams:
            logger.info(f"  Sample bigrams: {bigrams[:5]}")
        if trigrams:
            logger.info(f"  Sample trigrams: {trigrams[:5]}")
    
    def transform(self, texts: List[str], languages: Optional[List[str]] = None) -> sparse.csr_matrix:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: Input texts
            languages: Optional language codes
            
        Returns:
            TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        logger.info(f"Transforming {len(texts)} texts to TF-IDF features")
        
        # Preprocess texts
        preprocessed_texts = self._preprocess_texts(texts, languages)
        
        # Transform to TF-IDF
        tfidf_matrix = self.vectorizer.transform(preprocessed_texts)
        
        logger.info(f"Generated TF-IDF matrix: {tfidf_matrix.shape}")
        sparsity = 1 - (tfidf_matrix.nnz / np.prod(tfidf_matrix.shape))
        logger.info(f"Sparsity: {sparsity:.4f}")
        
        # Detailed TF-IDF analysis like reference implementation
        self._analyze_tfidf_matrix(tfidf_matrix, languages)
        
        return tfidf_matrix
    
    def _analyze_tfidf_matrix(self, tfidf_matrix, languages=None):
        """Analyze TF-IDF matrix with detailed statistics like reference."""
        logger.info("\n" + "="*50)
        logger.info("TF-IDF MATRIX ANALYSIS (Reference Implementation Style)")
        logger.info("="*50)
        
        # Basic statistics
        n_samples, n_features = tfidf_matrix.shape
        logger.info(f"Matrix dimensions: {n_samples:,} samples × {n_features:,} features")
        
        # Non-zero statistics
        non_zero_per_doc = [tfidf_matrix[i].nnz for i in range(n_samples)]
        logger.info(f"Average non-zero features per document: {np.mean(non_zero_per_doc):.2f}")
        logger.info(f"Min non-zero features: {min(non_zero_per_doc)}")
        logger.info(f"Max non-zero features: {max(non_zero_per_doc)}")
        logger.info(f"Median non-zero features: {np.median(non_zero_per_doc):.1f}")
        
        # Sparsity breakdown
        sparsity = 1 - (tfidf_matrix.nnz / np.prod(tfidf_matrix.shape))
        density_percentage = (1 - sparsity) * 100
        logger.info(f"Matrix sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
        logger.info(f"Matrix density: {density_percentage:.6f}%")
        logger.info(f"Total non-zero elements: {tfidf_matrix.nnz:,}")
        
        # Feature frequency analysis
        feature_counts = np.zeros(n_features)
        for i in range(n_samples):
            feature_counts[tfidf_matrix[i].nonzero()[1]] += 1
        
        # Top features by document frequency
        top_feature_indices = np.argsort(-feature_counts)[:20]
        logger.info("\nTop 20 features by document frequency:")
        feature_names = self.get_feature_names()
        for i, idx in enumerate(top_feature_indices, 1):
            token = feature_names[idx] if idx < len(feature_names) else f"UNKNOWN_{idx}"
            freq = feature_counts[idx]
            percentage = (freq / n_samples) * 100
            logger.info(f"  {i:2d}. '{token}': {freq:.0f} docs ({percentage:.1f}%)")
        
        # Language-specific analysis if available
        if languages:
            self._analyze_language_specific_tfidf(tfidf_matrix, languages, feature_names)
        
        # Sample document analysis (like reference)
        self._analyze_sample_documents(tfidf_matrix, feature_names)
        
        logger.info("="*50)
    
    def _analyze_language_specific_tfidf(self, tfidf_matrix, languages, feature_names):
        """Analyze TF-IDF patterns by language."""
        logger.info("\nLanguage-specific TF-IDF Analysis:")
        
        unique_languages = list(set(languages))
        for lang in unique_languages[:7]:  # Limit to first 7 languages
            lang_indices = [i for i, l in enumerate(languages) if l == lang]
            if not lang_indices:
                continue
                
            lang_matrix = tfidf_matrix[lang_indices]
            
            # Calculate average TF-IDF scores for this language
            lang_avg_scores = np.array(lang_matrix.mean(axis=0)).flatten()
            
            # Get top features for this language
            top_indices = np.argsort(lang_avg_scores)[-10:][::-1]
            
            logger.info(f"\n  {lang} (n={len(lang_indices)}): Top features")
            for i, idx in enumerate(top_indices, 1):
                if lang_avg_scores[idx] > 0:
                    token = feature_names[idx] if idx < len(feature_names) else f"UNKNOWN_{idx}"
                    score = lang_avg_scores[idx]
                    logger.info(f"    {i:2d}. '{token}': {score:.6f}")
    
    def _analyze_sample_documents(self, tfidf_matrix, feature_names, num_samples=3):
        """Analyze sample documents' TF-IDF values like reference."""
        logger.info(f"\nSample Document Analysis (first {num_samples} documents):")
        
        for doc_idx in range(min(num_samples, tfidf_matrix.shape[0])):
            feature_vector = tfidf_matrix[doc_idx]
            non_zero_indices = feature_vector.nonzero()[1]
            feature_values = feature_vector.data
            
            if len(non_zero_indices) == 0:
                logger.info(f"  Document {doc_idx}: No features (empty)")
                continue
            
            # Sort by TF-IDF value
            sorted_indices = sorted(zip(non_zero_indices, feature_values),
                                   key=lambda x: x[1], reverse=True)
            
            logger.info(f"  Document {doc_idx}: {len(non_zero_indices)} features")
            logger.info("    Top 10 features by TF-IDF value:")
            for i, (idx, value) in enumerate(sorted_indices[:10], 1):
                token = feature_names[idx] if idx < len(feature_names) else f"UNKNOWN_{idx}"
                logger.info(f"      {i:2d}. '{token}': {value:.6f}")
    
    def fit_transform(self, texts: List[str], languages: Optional[List[str]] = None) -> sparse.csr_matrix:
        """
        Fit vectorizer and transform texts in one step.
        """
        return self.fit(texts, languages).transform(texts, languages)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (tokens) in order."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        return [self.index_to_vocab[i] for i in range(len(self.index_to_vocab))]
    
    def get_vocab_info(self) -> Dict[str, Any]:
        """Get vocabulary information and statistics."""
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        feature_names = self.get_feature_names()
        
        # Analyze token types
        token_stats = {
            'total_features': len(feature_names),
            'avg_token_length': np.mean([len(token) for token in feature_names]),
            'token_lengths': Counter([len(token) for token in feature_names]),
            'first_chars': Counter([token[0] if token else '' for token in feature_names]),
            'xlm_roberta_vocab_size': self.tokenizer.vocab_size,
            'model_name': self.model_name
        }
        
        return token_stats
    
    def analyze_multilingual_features(self, texts: List[str], languages: List[str]) -> Dict[str, Any]:
        """
        Analyze multilingual features and their distribution.
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first")
        
        # Transform texts
        tfidf_matrix = self.transform(texts, languages)
        feature_names = self.get_feature_names()
        
        # Calculate feature importance by language
        lang_feature_importance = {}
        unique_languages = list(set(languages))
        
        for lang in unique_languages:
            lang_indices = [i for i, l in enumerate(languages) if l == lang]
            lang_matrix = tfidf_matrix[lang_indices]
            
            # Calculate average TF-IDF scores for this language
            lang_avg_scores = np.array(lang_matrix.mean(axis=0)).flatten()
            
            # Get top features for this language
            top_indices = np.argsort(lang_avg_scores)[-20:][::-1]
            top_features = [(feature_names[idx], lang_avg_scores[idx]) for idx in top_indices if lang_avg_scores[idx] > 0]
            
            lang_feature_importance[lang] = top_features
        
        return {
            'language_specific_features': lang_feature_importance,
            'total_features': len(feature_names),
            'average_features_per_doc': np.mean([tfidf_matrix[i].nnz for i in range(tfidf_matrix.shape[0])]),
            'sparsity': 1 - (tfidf_matrix.nnz / np.prod(tfidf_matrix.shape))
        }

def generate_xlm_roberta_text2text_features(
    output_dir: str,
    model_name: str = "xlm-roberta-base",
    max_features: int = 32000,   # Reduced from 128000 for better signal-to-noise ratio
    min_df: int = 3,         # Increased from 1 to remove very rare terms
    max_df: float = 0.95,    # Reduced from 0.99 to filter more common terms  
    dataset_name: str = "rokokot/question-type-and-complexity",
    dataset_config: str = "base",
    verify: bool = True
) -> Dict[str, Any]:
    """
    Generate TF-IDF features using XLM-RoBERTa tokenization with text2text methodology.
    
    Args:
        output_dir: Directory to save features
        model_name: XLM-RoBERTa model name  
        max_features: Maximum number of features
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        verify: Whether to verify generated features
        
    Returns:
        Dictionary with generation metadata
    """
    logger.info(f"Generating XLM-RoBERTa + text2text TF-IDF features")
    logger.info(f"Model: {model_name}, Max features: {max_features}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name} ({dataset_config})")
    dataset = load_dataset(dataset_name, dataset_config)
    
    train_df = dataset['train'].to_pandas()
    val_df = dataset['validation'].to_pandas()  
    test_df = dataset['test'].to_pandas()
    
    logger.info(f"Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Extract texts and languages
    train_texts = train_df['text'].tolist()
    val_texts = val_df['text'].tolist()
    test_texts = test_df['text'].tolist()
    
    train_langs = train_df['language'].tolist()
    val_langs = val_df['language'].tolist() 
    test_langs = test_df['language'].tolist()
    
    # Initialize extractor
    extractor = XLMRobertaText2TextTfidfExtractor(
        model_name=model_name,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 2),  # Include trigrams for more features
        random_state=42
    )
    
    # Fit on training data
    logger.info("Fitting TF-IDF vectorizer on training data...")
    extractor.fit(train_texts, train_langs)
    
    # Transform all splits
    logger.info("Transforming all splits...")
    X_train = extractor.transform(train_texts, train_langs)
    X_val = extractor.transform(val_texts, val_langs)
    X_test = extractor.transform(test_texts, test_langs)
    
    # Save features
    logger.info("Saving TF-IDF features...")
    
    # Save sparse matrices
    sparse.save_npz(output_path / "X_train_sparse.npz", X_train)
    sparse.save_npz(output_path / "X_val_sparse.npz", X_val)
    sparse.save_npz(output_path / "X_test_sparse.npz", X_test)
    
    # Save dense matrices  
    np.save(output_path / "X_train.npy", X_train.toarray())
    np.save(output_path / "X_val.npy", X_val.toarray())
    np.save(output_path / "X_test.npy", X_test.toarray())
    
    # Save legacy pickle format
    with open(output_path / "tfidf_vectors_train.pkl", 'wb') as f:
        pickle.dump(X_train, f)
    with open(output_path / "tfidf_vectors_dev.pkl", 'wb') as f:
        pickle.dump(X_val, f)
    with open(output_path / "tfidf_vectors_test.pkl", 'wb') as f:
        pickle.dump(X_test, f)
    
    # Save feature names and vocabulary
    feature_names = extractor.get_feature_names()
    with open(output_path / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save vocabulary mapping
    with open(output_path / "token_to_index_mapping.pkl", 'wb') as f:
        pickle.dump(extractor.vocab_to_index, f)
    
    # Save language info
    language_info = {
        'train': train_langs,
        'val': val_langs, 
        'test': test_langs
    }
    with open(output_path / "language_info.json", 'w') as f:
        json.dump(language_info, f, indent=2)
    
    # Analyze multilingual features
    logger.info("Analyzing multilingual features...")
    multilingual_analysis = extractor.analyze_multilingual_features(
        train_texts + val_texts + test_texts,
        train_langs + val_langs + test_langs
    )
    
    # Create metadata
    vocab_info = extractor.get_vocab_info()
    metadata = {
        'tokenizer_info': {
            'model_name': model_name,
            'tokenizer_class': extractor.tokenizer.__class__.__name__,
            'vocab_size': extractor.tokenizer.vocab_size,
            'model_max_length': extractor.tokenizer.model_max_length,
            'special_tokens': {
                'bos_token': extractor.tokenizer.bos_token,
                'eos_token': extractor.tokenizer.eos_token,
                'cls_token': extractor.tokenizer.cls_token,
                'sep_token': extractor.tokenizer.sep_token,
                'pad_token': extractor.tokenizer.pad_token,
                'unk_token': extractor.tokenizer.unk_token,
            },
            'use_fast': hasattr(extractor.tokenizer, 'is_fast') and extractor.tokenizer.is_fast
        },
        'tfidf_params': {
            'max_features': max_features,
            'min_df': min_df,
            'max_df': max_df,
            'ngram_range': list(extractor.ngram_range),
            'norm': 'l2',
            'use_idf': True,
            'smooth_idf': True,
            'sublinear_tf': True
        },
        'random_state': 42,
        'data_hash': str(hash(str(train_texts + val_texts + test_texts))),
        'vocab_size': len(feature_names),
        'feature_shape': {
            'train': list(X_train.shape),
            'val': list(X_val.shape),
            'test': list(X_test.shape)
        },
        'generation_info': {
            'actual_features': len(feature_names),
            'sparsity': {
                'train': 1 - (X_train.nnz / np.prod(X_train.shape)),
                'val': 1 - (X_val.nnz / np.prod(X_val.shape)),
                'test': 1 - (X_test.nnz / np.prod(X_test.shape))
            },
            'avg_features_per_doc': {
                'train': np.mean([X_train[i].nnz for i in range(X_train.shape[0])]),
                'val': np.mean([X_val[i].nnz for i in range(X_val.shape[0])]),
                'test': np.mean([X_test[i].nnz for i in range(X_test.shape[0])])
            }
        },
        'vocab_analysis': vocab_info,
        'multilingual_analysis': multilingual_analysis
    }
    
    # Save metadata
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save multilingual analysis separately
    with open(output_path / "multilingual_analysis.json", 'w') as f:
        json.dump(multilingual_analysis, f, indent=2)
    
    logger.info(f"Features saved to {output_dir}")
    logger.info(f"Generated {len(feature_names)} features")
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Print sample multilingual features
    logger.info("\nSample language-specific features:")
    for lang, features in multilingual_analysis['language_specific_features'].items():
        top_features = features[:5]  # Top 5 features per language
        logger.info(f"{lang}: {[(f, f'{s:.4f}') for f, s in top_features]}")
    
    if verify:
        logger.info("Verifying generated features...")
        verify_features(output_dir)
    
    return metadata

def verify_features(features_dir: str) -> bool:
    """Verify that generated features are valid."""
    try:
        # Try to import TfidfFeatureLoader
        import sys
        from pathlib import Path
        
        # Add src directory to path if not already there
        src_path = Path(__file__).parent.parent / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from data.tfidf_features import TfidfFeatureLoader
        
        loader = TfidfFeatureLoader(features_dir)
        success = loader.verify_features()
        
        if success:
            logger.info("✅ Feature verification passed")
        else:
            logger.error("❌ Feature verification failed")
            
        return success
        
    except Exception as e:
        logger.warning(f"Could not verify features: {e}")
        # Manual verification fallback
        features_path = Path(features_dir)
        required_files = ['X_train_sparse.npz', 'X_val_sparse.npz', 'X_test_sparse.npz', 'metadata.json']
        all_exist = all((features_path / f).exists() for f in required_files)
        
        if all_exist:
            logger.info("✅ Basic file verification passed")
            return True
        else:
            logger.error("❌ Missing required feature files")
            return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate hybrid XLM-RoBERTa + text2text TF-IDF features")
    parser.add_argument("--output-dir", required=True, help="Output directory for features")
    parser.add_argument("--model-name", default="xlm-roberta-base", help="XLM-RoBERTa model name")
    parser.add_argument("--max-features", type=int, default=128000, help="Maximum number of features")
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency")
    parser.add_argument("--max-df", type=float, default=0.95, help="Maximum document frequency")
    parser.add_argument("--dataset", default="rokokot/question-type-and-complexity", help="Dataset name")
    parser.add_argument("--config", default="base", help="Dataset config")
    parser.add_argument("--verify", action="store_true", help="Verify generated features")
    
    args = parser.parse_args()
    
    generate_xlm_roberta_text2text_features(
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        dataset_name=args.dataset,
        dataset_config=args.config,
        verify=args.verify
    )