# src/data/datasets.py 
"""
Enhanced dataset loading with TF-IDF  integration.
Maintains backward compatibility  adding TF-IDF support.
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import scipy.sparse

# Import TF-IDF feature loader
from .tfidf_features import TfidfFeatureLoader

logger = logging.getLogger(__name__)

# Define constants (maintain backward compatibility)
TASK_TO_FEATURE = {
    "question_type": "question_type",
    "complexity": "complexity_score",
    "single_submetric": None,  # Will be determined by submetric parameter
    "avg_links_len": "avg_links_len",
    "avg_max_depth": "avg_max_depth", 
    "avg_subordinate_chain_len": "avg_subordinate_chain_len",
    "avg_verb_edges": "avg_verb_edges",
    "lexical_density": "lexical_density",
    "n_tokens": "n_tokens"
}

AVAILABLE_LANGUAGES = ["ar", "en", "fi", "id", "ja", "ko", "ru"]

def ensure_string_task(task) -> str:
    """Ensure task is a string (backward compatibility)."""
    if isinstance(task, list):
        return task[0] if task else "question_type"
    return str(task) if task else "question_type"

def load_sklearn_data(
    languages: List[str] = ["all"],
    task: str = "question_type", 
    submetric: Optional[str] = None,
    control_index: Optional[int] = None,
    cache_dir: str = "./data/cache",
    vectors_dir: str = "./data/features",
    use_tfidf_loader: bool = False,
    tfidf_features_dir: Optional[str] = None
) -> Tuple[Tuple[Union[np.ndarray, scipy.sparse.csr_matrix], np.ndarray], 
           Tuple[Union[np.ndarray, scipy.sparse.csr_matrix], np.ndarray], 
           Tuple[Union[np.ndarray, scipy.sparse.csr_matrix], np.ndarray]]:
    """
    Load sklearn-compatible data with optional TF-IDF feature support.
    
    Args:
        languages: List of language codes or ["all"]
        task: Task name ("question_type", "complexity", "single_submetric", or submetric name)
        submetric: Specific submetric for single_submetric task
        control_index: Control experiment index (1, 2, 3)
        cache_dir: Directory for cached datasets
        vectors_dir: Directory containing legacy TF-IDF vectors
        use_tfidf_loader: Whether to use new TF-IDF loader
        tfidf_features_dir: Directory for new TF-IDF features (if use_tfidf_loader=True)
        
    Returns:
        Tuple of (train, val, test) data, each containing (X, y)
    """
    logger.info(f"Loading sklearn data: task={task}, languages={languages}, control_index={control_index}")
    
    # Normalize task name
    task = ensure_string_task(task)
    
    # Load dataset and labels
    train_labels, val_labels, test_labels = load_labels(
        languages=languages,
        task=task,
        submetric=submetric,
        control_index=control_index,
        cache_dir=cache_dir
    )
    
    # Load features - choose between TF-IDF loader and legacy vectors
    if use_tfidf_loader:
        logger.info("Using new TF-IDF feature loader")
        features_dir = tfidf_features_dir or vectors_dir
        X_train, X_val, X_test = load_tfidf_features_new(features_dir, languages)
    else:
        logger.info("Using legacy TF-IDF vectors")
        X_train, X_val, X_test = load_tfidf_vectors(vectors_dir, languages)
    
    logger.info(f"Loaded features: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    logger.info(f"Loaded labels: train={len(train_labels)}, val={len(val_labels)}, test={len(test_labels)}")
    
    # Validate shapes
    if X_train.shape[0] != len(train_labels):
        logger.warning(f"Feature/label mismatch in train: {X_train.shape[0]} vs {len(train_labels)}")
    if X_val.shape[0] != len(val_labels):
        logger.warning(f"Feature/label mismatch in val: {X_val.shape[0]} vs {len(val_labels)}")
    if X_test.shape[0] != len(test_labels):
        logger.warning(f"Feature/label mismatch in test: {X_test.shape[0]} vs {len(test_labels)}")
    
    return (X_train, train_labels), (X_val, val_labels), (X_test, test_labels)

def load_tfidf_features_new(
    features_dir: str, 
    languages: List[str]
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
    """
    Load TF-IDF features using the new TfidfFeatureLoader.
    
    Args:
        features_dir: Directory containing TF-IDF features
        languages: List of target languages
        
    Returns:
        Tuple of (train, val, test) feature matrices
    """
    try:
        # Initialize TF-IDF loader
        loader = TfidfFeatureLoader(features_dir)
        
        # Load all features
        all_features = loader.load_all_features()
        
        # Filter by languages
        filtered_features = loader.filter_by_languages(all_features, languages)
        
        # Return in expected order
        return filtered_features['train'], filtered_features['val'], filtered_features['test']
        
    except Exception as e:
        logger.error(f"Failed to load TF-IDF features with new loader: {e}")
        raise

def load_tfidf_vectors(vectors_dir: str, languages: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load legacy TF-IDF vectors (backward compatibility).
    
    Args:
        vectors_dir: Directory containing TF-IDF vector pickle files
        languages: List of target languages (ignored for legacy vectors)
        
    Returns:
        Tuple of (train, val, test) feature arrays
    """
    try:
        train_path = os.path.join(vectors_dir, "tfidf_vectors_train.pkl")
        dev_path = os.path.join(vectors_dir, "tfidf_vectors_dev.pkl")
        test_path = os.path.join(vectors_dir, "tfidf_vectors_test.pkl")
        
        # Check if files exist
        for path, split in [(train_path, "train"), (dev_path, "dev"), (test_path, "test")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"TF-IDF vectors not found: {path}")
        
        # Load vectors
        with open(train_path, "rb") as f:
            X_train = pickle.load(f)
        with open(dev_path, "rb") as f:
            X_val = pickle.load(f)
        with open(test_path, "rb") as f:
            X_test = pickle.load(f)
        
        # Convert to arrays if sparse
        if scipy.sparse.issparse(X_train):
            X_train = X_train.toarray()
        if scipy.sparse.issparse(X_val):
            X_val = X_val.toarray()
        if scipy.sparse.issparse(X_test):
            X_test = X_test.toarray()
        
        logger.info("Loaded legacy TF-IDF vectors successfully")
        return X_train, X_val, X_test
        
    except Exception as e:
        logger.error(f"Failed to load legacy TF-IDF vectors: {e}")
        raise

def load_labels(
    languages: List[str] = ["all"],
    task: str = "question_type",
    submetric: Optional[str] = None,
    control_index: Optional[int] = None,
    cache_dir: str = "./data/cache"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load labels for the specified task and languages.
    
    Args:
        languages: List of language codes or ["all"]
        task: Task name
        submetric: Specific submetric (if task is "single_submetric")
        control_index: Control experiment index
        cache_dir: Directory for cached datasets
        
    Returns:
        Tuple of (train_labels, val_labels, test_labels)
    """
    # Determine dataset configuration
    if control_index is not None:
        if task == "single_submetric" and submetric:
            config_name = f"control_{submetric}_seed{control_index}"
        else:
            config_name = f"control_{task}_seed{control_index}"
    else:
        config_name = "base"
    
    # Determine target feature
    if task == "single_submetric":
        if submetric is None:
            raise ValueError("submetric must be specified when task is 'single_submetric'")
        target_feature = submetric
    elif task in TASK_TO_FEATURE:
        target_feature = TASK_TO_FEATURE[task]
    else:
        # Assume task is a submetric name
        target_feature = task
    
    logger.info(f"Loading labels: config={config_name}, feature={target_feature}, languages={languages}")
    
    # Load dataset
    try:
        dataset = load_dataset(
            "rokokot/question-type-and-complexity",
            name=config_name,
            cache_dir=cache_dir,
            verification_mode='no_checks'
        )
    except Exception as e:
        logger.error(f"Failed to load dataset with config '{config_name}': {e}")
        raise
    
    # Extract labels for each split
    train_labels = extract_labels_from_split(dataset['train'], target_feature, languages)
    val_labels = extract_labels_from_split(dataset['validation'], target_feature, languages)
    test_labels = extract_labels_from_split(dataset['test'], target_feature, languages)
    
    return train_labels, val_labels, test_labels

def extract_labels_from_split(split_data, target_feature: str, languages: List[str]) -> np.ndarray:
    """
    Extract labels from a dataset split with language filtering.
    
    Args:
        split_data: Dataset split
        target_feature: Name of the target feature
        languages: List of target languages
        
    Returns:
        Numpy array of labels
    """
    # Convert to pandas for easier manipulation
    df = split_data.to_pandas()
    
    # Filter by languages if specified
    if languages != ["all"]:
        df = df[df['language'].isin(languages)]
    
    # Extract target labels
    if target_feature not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Target feature '{target_feature}' not found. Available: {available_cols}")
    
    labels = df[target_feature].values
    
    # Handle missing values
    if pd.isna(labels).any():
        logger.warning(f"Found {pd.isna(labels).sum()} missing values in {target_feature}")
        # For now, replace NaN with 0 (could be made configurable)
        labels = np.nan_to_num(labels, nan=0.0)
    
    return labels

# Language Model DataLoader functions (keep existing functionality)
class QuestionDataset(Dataset):
    """Dataset class for question data with tokenization."""
    
    def __init__(self, texts: List[str], labels: List[float], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.float)
        }

def create_lm_dataloaders(
    language: str,
    task: str,
    model_name: str = "cis-lmu/glot500-base",
    batch_size: int = 16,
    max_length: int = 128,
    control_index: Optional[int] = None,
    cache_dir: str = "./data/cache",
    num_workers: int = 0,
    submetric: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for language model training/evaluation.
    
    Args:
        language: Language code
        task: Task name
        model_name: Model name for tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        control_index: Control experiment index
        cache_dir: Cache directory
        num_workers: Number of data loading workers
        submetric: Specific submetric (if task is "single_submetric")
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Normalize task
    task = ensure_string_task(task)
    
    # Determine dataset configuration
    if control_index is not None:
        if task == "single_submetric" and submetric:
            config_name = f"control_{submetric}_seed{control_index}"
        else:
            config_name = f"control_{task}_seed{control_index}"
    else:
        config_name = "base"
    
    # Load tokenizer
    try:
        local_files_only = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer {model_name}: {e}")
        raise
    
    # Load dataset
    try:
        dataset = load_dataset(
            "rokokot/question-type-and-complexity",
            name=config_name,
            cache_dir=cache_dir,
            verification_mode='no_checks'
        )
    except Exception as e:
        logger.error(f"Failed to load dataset with config '{config_name}': {e}")
        raise
    
    # Determine target feature
    if task == "single_submetric":
        if submetric is None:
            raise ValueError("submetric must be specified when task is 'single_submetric'")
        target_feature = submetric
    elif task in TASK_TO_FEATURE:
        target_feature = TASK_TO_FEATURE[task]
    else:
        # Assume task is a submetric name
        target_feature = task
    
    # Create datasets for each split
    datasets = {}
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset[split_name].to_pandas()
        
        # Filter by language
        lang_data = split_data[split_data['language'] == language]
        
        if len(lang_data) == 0:
            raise ValueError(f"No data found for language '{language}' in {split_name} split")
        
        # Extract texts and labels
        texts = lang_data['text'].tolist()
        
        if target_feature not in lang_data.columns:
            available_cols = list(lang_data.columns)
            raise ValueError(f"Target feature '{target_feature}' not found. Available: {available_cols}")
        
        labels = lang_data[target_feature].fillna(0).tolist()
        
        # Create dataset
        datasets[split_name] = QuestionDataset(texts, labels, tokenizer, max_length)
        
        logger.info(f"Created {split_name} dataset: {len(texts)} examples for language '{language}'")
    
    # Create data loaders
    train_loader = DataLoader(
        datasets['train'], 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        datasets['validation'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        datasets['test'], 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_statistics(
    languages: List[str] = ["all"],
    cache_dir: str = "./data/cache"
) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the dataset.
    
    Args:
        languages: List of language codes
        cache_dir: Cache directory
        
    Returns:
        Dictionary with dataset statistics
    """
    try:
        # Load base dataset
        dataset = load_dataset(
            "rokokot/question-type-and-complexity",
            name="base",
            cache_dir=cache_dir,
            verification_mode='no_checks'
        )
        
        stats = {
            'languages': languages,
            'splits': {},
            'tasks': list(TASK_TO_FEATURE.keys()),
            'available_languages': AVAILABLE_LANGUAGES
        }
        
        for split_name in ['train', 'validation', 'test']:
            split_data = dataset[split_name].to_pandas()
            
            # Filter by languages if specified
            if languages != ["all"]:
                split_data = split_data[split_data['language'].isin(languages)]
            
            split_stats = {
                'total_examples': len(split_data),
                'languages': split_data['language'].value_counts().to_dict(),
                'question_types': split_data['question_type'].value_counts().to_dict() if 'question_type' in split_data.columns else {},
                'text_lengths': {
                    'mean': float(split_data['text'].str.len().mean()),
                    'std': float(split_data['text'].str.len().std()),
                    'min': int(split_data['text'].str.len().min()),
                    'max': int(split_data['text'].str.len().max())
                }
            }
            
            # Add complexity statistics if available
            if 'complexity_score' in split_data.columns:
                complexity_values = split_data['complexity_score'].dropna()
                split_stats['complexity'] = {
                    'mean': float(complexity_values.mean()),
                    'std': float(complexity_values.std()),
                    'min': float(complexity_values.min()),
                    'max': float(complexity_values.max()),
                    'count': len(complexity_values)
                }
            
            stats['splits'][split_name] = split_stats
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get dataset statistics: {e}")
        return {'error': str(e)}


def validate_dataset_integrity(
    cache_dir: str = "./data/cache",
    tfidf_features_dir: Optional[str] = None
) -> Dict[str, bool]:
    """
    Validate the integrity of datasets and features.
    
    Args:
        cache_dir: Cache directory for datasets
        tfidf_features_dir: Directory for TF-IDF features
        
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Check dataset loading
    try:
        dataset = load_dataset(
            "rokokot/question-type-and-complexity",
            name="base",
            cache_dir=cache_dir,
            verification_mode='no_checks'
        )
        
        # Check required splits
        required_splits = ['train', 'validation', 'test']
        splits_ok = all(split in dataset for split in required_splits)
        results['dataset_splits'] = splits_ok
        
        # Check required columns
        sample_data = dataset['train'].to_pandas()
        required_columns = ['text', 'language', 'question_type']
        columns_ok = all(col in sample_data.columns for col in required_columns)
        results['dataset_columns'] = columns_ok
        
        # Check languages
        available_langs = set(sample_data['language'].unique())
        expected_langs = set(AVAILABLE_LANGUAGES)
        languages_ok = expected_langs.issubset(available_langs)
        results['dataset_languages'] = languages_ok
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        results['dataset_splits'] = False
        results['dataset_columns'] = False
        results['dataset_languages'] = False
    
    # Check TF-IDF features if directory provided
    if tfidf_features_dir:
        try:
            loader = TfidfFeatureLoader(tfidf_features_dir)
            results['tfidf_features'] = loader.verify_features()
        except Exception as e:
            logger.error(f"TF-IDF features validation failed: {e}")
            results['tfidf_features'] = False
    
    return results


def create_tiny_dataset(
    output_dir: str,
    n_samples_per_lang: int = 10,
    languages: List[str] = ["en", "ru"]
) -> None:
    """
    Create a tiny dataset for testing purposes.
    
    Args:
        output_dir: Directory to save the tiny dataset
        n_samples_per_lang: Number of samples per language
        languages: List of languages to include
    """
    from .tfidf_features import create_test_features
    
    logger.info(f"Creating tiny dataset in {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create test TF-IDF features
    create_test_features(
        output_dir=str(output_path / "tfidf_features"),
        n_samples=n_samples_per_lang * len(languages)
    )
    
    # Create test labels
    total_samples = n_samples_per_lang * len(languages)
    
    # Generate synthetic labels for different tasks
    np.random.seed(42)  # For reproducibility
    
    labels_data = {
        'train': {
            'question_type': np.random.randint(0, 2, total_samples),
            'complexity_score': np.random.randn(total_samples) + 5,
            'languages': np.repeat(languages, n_samples_per_lang)
        },
        'val': {
            'question_type': np.random.randint(0, 2, total_samples // 4),
            'complexity_score': np.random.randn(total_samples // 4) + 5,
            'languages': np.repeat(languages, n_samples_per_lang // 4)
        },
        'test': {
            'question_type': np.random.randint(0, 2, total_samples // 4),
            'complexity_score': np.random.randn(total_samples // 4) + 5,
            'languages': np.repeat(languages, n_samples_per_lang // 4)
        }
    }
    
    # Save labels as pickle files for testing
    for split, data in labels_data.items():
        with open(output_path / f"labels_{split}.pkl", "wb") as f:
            pickle.dump(data, f)
    
    logger.info(f"✓ Tiny dataset created successfully in {output_dir}")


# Utility functions for backward compatibility
def get_available_tasks() -> List[str]:
    """Get list of available tasks."""
    return list(TASK_TO_FEATURE.keys())

def get_available_languages() -> List[str]:
    """Get list of available languages."""
    return AVAILABLE_LANGUAGES.copy()

def get_task_type(task: str) -> str:
    """
    Determine task type (classification or regression) from task name.
    
    Args:
        task: Task name
        
    Returns:
        'classification' or 'regression'
    """
    task = ensure_string_task(task)
    
    if task == "question_type":
        return "classification"
    elif task in ["complexity", "single_submetric"] or task in [
        "avg_links_len", "avg_max_depth", "avg_subordinate_chain_len", 
        "avg_verb_edges", "lexical_density", "n_tokens"
    ]:
        return "regression"
    else:
        logger.warning(f"Unknown task '{task}', defaulting to classification")
        return "classification"


if __name__ == "__main__":
    # Test the enhanced dataset loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Test enhanced dataset loading")
    parser.add_argument("--cache-dir", default="./data/cache", help="Cache directory")
    parser.add_argument("--tfidf-dir", help="TF-IDF features directory")
    parser.add_argument("--create-tiny", action="store_true", help="Create tiny test dataset")
    parser.add_argument("--test-sklearn", action="store_true", help="Test sklearn data loading")
    parser.add_argument("--test-lm", action="store_true", help="Test language model data loading")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    if args.create_tiny:
        create_tiny_dataset("./data/tfidf_features_tiny")
        print("✅ Tiny dataset created")
    
    if args.test_sklearn:
        print("Testing sklearn data loading...")
        
        # Test with new TF-IDF loader if directory provided
        if args.tfidf_dir:
            try:
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
                    languages=['en'],
                    task='question_type',
                    use_tfidf_loader=True,
                    tfidf_features_dir=args.tfidf_dir
                )
                print(f"✅ New TF-IDF loader: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
                print(f"✅ Labels: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
            except Exception as e:
                print(f"❌ New TF-IDF loader failed: {e}")
    
    if args.test_lm:
        print("Testing language model data loading...")
        try:
            train_loader, val_loader, test_loader = create_lm_dataloaders(
                language='en',
                task='question_type',
                batch_size=4,
                cache_dir=args.cache_dir
            )
            print(f"✅ LM DataLoaders created: {len(train_loader)} train batches")
            
            # Test one batch
            batch = next(iter(train_loader))
            print(f"✅ Sample batch: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
        except Exception as e:
            print(f"❌ LM DataLoader test failed: {e}")
    
    # Test dataset validation
    print("Testing dataset validation...")
    validation_results = validate_dataset_integrity(
        cache_dir=args.cache_dir,
        tfidf_features_dir=args.tfidf_dir
    )
    
    for check, passed in validation_results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}: {'PASS' if passed else 'FAIL'}")
    
    print(" All tests completed!")