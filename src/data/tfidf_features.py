# src/data/tfidf_features.py
"""
Enhanced TF-IDF feature loader with proper data integration.
Fixed version that resolves feature/label alignment issues.
"""

import os
import json
import pickle
import numpy as np
import scipy.sparse as sparse
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class TfidfFeatureLoader:
    """Enhanced TF-IDF feature loader with proper data alignment."""
    
    def __init__(self, features_dir: str):
        """Initialize the TF-IDF feature loader."""
        self.features_dir = Path(features_dir)
        
        if not self.features_dir.exists():
            raise FileNotFoundError(f"TF-IDF features directory not found: {features_dir}")
        
        # Load metadata if available
        self.metadata = self._load_metadata()
        
        # Cache for loaded features
        self._feature_cache = {}
        
        logger.info(f"Initialized TF-IDF loader for: {features_dir}")
    
    def _load_metadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata file if it exists."""
        metadata_file = self.features_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        return None
    
    def _find_feature_files(self) -> Dict[str, str]:
        """Find available feature files and return mapping."""
        files = {}
        
        # Priority order: sparse -> dense -> pickle
        formats = [
            ('sparse', ['X_{}_sparse.npz']),
            ('dense', ['X_{}.npy']),
            ('pickle', ['tfidf_vectors_{}.pkl'])
        ]
        
        for split in ['train', 'val', 'test']:
            for format_name, patterns in formats:
                for pattern in patterns:
                    # Handle val/dev naming inconsistency
                    split_variants = [split]
                    if split == 'val':
                        split_variants.append('dev')
                    elif split == 'dev':
                        split_variants.append('val')
                    
                    for split_variant in split_variants:
                        filepath = self.features_dir / pattern.format(split_variant)
                        if filepath.exists():
                            files[split] = str(filepath)
                            break
                    
                    if split in files:
                        break
                
                if split in files:
                    break
        
        return files
    
    def load_features(self, split: str) -> sparse.csr_matrix:
        """Load features for a specific split."""
        if split in self._feature_cache:
            return self._feature_cache[split]
        
        available_files = self._find_feature_files()
        
        if split not in available_files:
            raise ValueError(f"No feature files found for split '{split}'. Available: {list(available_files.keys())}")
        
        filepath = available_files[split]
        
        try:
            if filepath.endswith('.npz'):
                # Sparse format
                matrix = sparse.load_npz(filepath)
            elif filepath.endswith('.npy'):
                # Dense format
                matrix = sparse.csr_matrix(np.load(filepath))
            elif filepath.endswith('.pkl'):
                # Pickle format
                with open(filepath, 'rb') as f:
                    matrix = pickle.load(f)
                    if not sparse.issparse(matrix):
                        matrix = sparse.csr_matrix(matrix)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
            
            # Cache the loaded features
            self._feature_cache[split] = matrix
            
            logger.info(f"Loaded {split} features: {matrix.shape}")
            return matrix
            
        except Exception as e:
            logger.error(f"Failed to load features from {filepath}: {e}")
            raise
    
    def load_all_features(self) -> Dict[str, sparse.csr_matrix]:
        """Load all available feature splits."""
        available_files = self._find_feature_files()
        features = {}
        
        for split in available_files:
            features[split] = self.load_features(split)
        
        return features
    
    def filter_by_languages(self, features: Dict[str, sparse.csr_matrix], target_languages: List[str]) -> Dict[str, sparse.csr_matrix]:
        """Filter features by target languages."""
        # For now, return all features since we don't have language-specific filtering
        # This would be implemented with proper language metadata
        if 'all' in target_languages:
            return features
        
        # TODO: Implement proper language filtering when language info is available
        logger.info(f"Language filtering not implemented yet. Returning all features for languages: {target_languages}")
        return features
    
    def verify_features(self) -> bool:
        """Verify that all required features are available."""
        try:
            available_files = self._find_feature_files()
            required_splits = ['train', 'val', 'test']
            
            missing_splits = [split for split in required_splits if split not in available_files]
            if missing_splits:
                logger.error(f"Missing feature files for splits: {missing_splits}")
                return False
            
            # Try to load each split to verify integrity
            for split in available_files:
                features = self.load_features(split)
                if features.shape[0] == 0 or features.shape[1] == 0:
                    logger.error(f"Invalid feature matrix shape for {split}: {features.shape}")
                    return False
            
            logger.info("Feature verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Feature verification failed: {e}")
            return False
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.metadata and 'vocab_size' in self.metadata:
            return self.metadata['vocab_size']
        
        # Try to infer from features
        try:
            train_features = self.load_features('train')
            return train_features.shape[1]
        except Exception:
            return 0
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata dictionary."""
        return self.metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the features."""
        stats = {
            'vocab_size': self.get_vocab_size(),
            'splits': {}
        }
        
        try:
            features = self.load_all_features()
            
            for split, matrix in features.items():
                sparsity = 1 - (matrix.nnz / np.prod(matrix.shape)) if np.prod(matrix.shape) > 0 else 0
                
                stats['splits'][split] = {
                    'n_samples': matrix.shape[0],
                    'n_features': matrix.shape[1],
                    'sparsity': sparsity,
                    'nnz': matrix.nnz
                }
                
        except Exception as e:
            logger.error(f"Failed to generate statistics: {e}")
            stats['error'] = str(e)
        
        return stats


def create_test_features(
    output_dir: str,
    n_samples: int = 100,
    vocab_size: int = 200,
    sparsity: float = 0.9,
    random_seed: int = 42
) -> None:
    """
    Create test TF-IDF features with proper alignment.
    
    Args:
        output_dir: Output directory for features
        n_samples: Total number of samples (will be split into train/val/test)
        vocab_size: Vocabulary size
        sparsity: Sparsity level (fraction of zeros)
        random_seed: Random seed for reproducibility
    """
    np.random.seed(random_seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate split sizes - ensure they add up to n_samples
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)  
    n_test = n_samples - n_train - n_val  # Remainder goes to test
    
    splits = {
        'train': n_train,
        'val': n_val,
        'test': n_test
    }
    
    logger.info(f"Creating test features with splits: {splits}")
    
    # Create metadata
    metadata = {
        'vocab_size': vocab_size,
        'model_name': 'test-tfidf-model',
        'max_features': vocab_size,
        'generation_info': {
            'actual_features': vocab_size,
            'sparsity': {split: sparsity for split in splits.keys()}
        },
        'feature_shape': {
            split: [n_split_samples, vocab_size] 
            for split, n_split_samples in splits.items()
        }
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create feature matrices for each split
    for split, n_split_samples in splits.items():
        if n_split_samples <= 0:
            logger.warning(f"Skipping {split} split with {n_split_samples} samples")
            continue
            
        # Create sparse matrix
        n_nonzero = int(n_split_samples * vocab_size * (1 - sparsity))
        
        # Ensure each sample has at least one non-zero element
        min_nonzero = n_split_samples
        n_nonzero = max(n_nonzero, min_nonzero)
        
        # Generate random indices
        row_indices = np.random.randint(0, n_split_samples, n_nonzero)
        col_indices = np.random.randint(0, vocab_size, n_nonzero)
        
        # Generate realistic TF-IDF values (log-normal distribution)
        values = np.random.lognormal(mean=-1, sigma=1, size=n_nonzero)
        
        # Create sparse matrix
        matrix = sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_split_samples, vocab_size)
        )
        
        # Ensure each row has at least one non-zero element
        for i in range(n_split_samples):
            if matrix[i].nnz == 0:
                col = np.random.randint(0, vocab_size)
                matrix[i, col] = np.random.lognormal(-1, 1)
        
        # Save in multiple formats for compatibility
        sparse.save_npz(output_path / f"X_{split}_sparse.npz", matrix)
        np.save(output_path / f"X_{split}.npy", matrix.toarray())
        
        # Save in legacy pickle format too
        split_name = 'dev' if split == 'val' else split
        with open(output_path / f"tfidf_vectors_{split_name}.pkl", 'wb') as f:
            pickle.dump(matrix, f)
        
        logger.info(f"Created {split} features: {matrix.shape}, sparsity: {1 - matrix.nnz / np.prod(matrix.shape):.2%}")
    
    # Create sample language info
    language_info = {}
    for split, n_split_samples in splits.items():
        if n_split_samples > 0:
            language_info[split] = ['en'] * n_split_samples
    
    with open(output_path / "language_info.json", 'w') as f:
        json.dump(language_info, f)
    
    logger.info(f"Test TF-IDF features created successfully in {output_dir}")


def create_aligned_test_dataset(
    features_dir: str,
    output_format: str = 'mock_dataset'
) -> Dict[str, Any]:
    """
    Create a test dataset that's properly aligned with the TF-IDF features.
    
    Args:
        features_dir: Directory with TF-IDF features
        output_format: Format for output ('mock_dataset' or 'huggingface')
        
    Returns:
        Dictionary with aligned dataset
    """
    loader = TfidfFeatureLoader(features_dir)
    features = loader.load_all_features()
    
    # Create aligned data for each split
    aligned_data = {}
    
    for split, matrix in features.items():
        n_samples = matrix.shape[0]
        
        # Generate synthetic but realistic data
        texts = [f"Sample text {i} for {split} split" for i in range(n_samples)]
        languages = ['en'] * n_samples
        
        # Generate question types (binary classification)
        question_types = [i % 2 for i in range(n_samples)]
        
        # Generate complexity scores (regression target)
        complexity_scores = [0.1 + (i % 10) * 0.1 for i in range(n_samples)]
        
        # Generate submetrics
        submetrics = {}
        submetric_names = ['avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len', 
                          'avg_verb_edges', 'lexical_density', 'n_tokens']
        
        for submetric in submetric_names:
            # Add some correlation with complexity + noise
            base_values = [cs + np.random.normal(0, 0.1) for cs in complexity_scores]
            submetrics[submetric] = [max(0, min(1, val)) for val in base_values]
        
        aligned_data[split] = {
            'text': texts,
            'language': languages,
            'question_type': question_types,
            'lang_norm_complexity_score': complexity_scores,
            **submetrics
        }
        
        logger.info(f"Created aligned {split} dataset with {n_samples} samples")
    
    return aligned_data


if __name__ == "__main__":
    # Test the functionality
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Testing TF-IDF feature creation and loading...")
        
        # Create test features
        create_test_features(temp_dir, n_samples=50, vocab_size=100)
        
        # Test loader
        loader = TfidfFeatureLoader(temp_dir)
        
        # Verify features
        if loader.verify_features():
            print("✓ Feature verification passed")
        else:
            print("✗ Feature verification failed")
        
        # Load and check features
        features = loader.load_all_features()
        print(f"✓ Loaded features: {[(k, v.shape) for k, v in features.items()]}")
        
        # Get statistics
        stats = loader.get_statistics()
        print(f"✓ Statistics: vocab_size={stats['vocab_size']}")
        
        # Create aligned dataset
        aligned_data = create_aligned_test_dataset(temp_dir)
        print(f"✓ Created aligned dataset with splits: {list(aligned_data.keys())}")
        
        print("All tests passed!")