# data loading and processing utilities for multilingual question probing experiments
# uses pre-extracted TF-IDF features

import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any, Optional, Union
from datasets import load_dataset
from transformers import AutoTokenizer
from scipy.sparse import vstack
from scipy import sparse
import logging

logger = logging.getLogger(__name__)

DATASET_NAME = "rokokot/question-type-and-complexity"

TASK_TO_FEATURE = {
    "question_type": "question_type",
    "complexity": "lang_norm_complexity_score", # lang_norm_complexity_score
    "avg_links_len": "avg_links_len",
    "avg_max_depth": "avg_max_depth",
    "avg_subordinate_chain_len": "avg_subordinate_chain_len",
    "avg_verb_edges": "avg_verb_edges",
    "lexical_density": "lexical_density",
    "n_tokens": "n_tokens"}

class MultilingualQuestionDataset(Dataset):     # Dataset for sklearn models using TF-IDF features
    def __init__(
        self, 
        data: pd.DataFrame,
        features: np.ndarray,
        labels: np.ndarray,
        language_ids: Optional[np.ndarray] = None):
        self.data = data
        self.features = features
        self.labels = labels
        self.language_ids = language_ids
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            "features": self.features[idx],
            "label": self.labels[idx]}
        
        if self.language_ids is not None:
            item["language_id"] = self.language_ids[idx]
        
        return item

class LMQuestionDataset(Dataset):
    
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        task: str,
        max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length
        

        self.is_classification = task == "question_type"
        

        self.texts = data["text"].tolist()
        
        feature_name = TASK_TO_FEATURE[task]
        

        if task == "sub_metrics":
            submetrics = ["avg_links_len", "avg_max_depth", "avg_subordinate_chain_len", 
                         "avg_verb_edges", "lexical_density", "n_tokens"]
            self.labels = data[submetrics].values.astype(np.float32)
        else:
            self.labels = data[feature_name].values
            
            if self.is_classification:
                self.labels = self.labels.astype(np.int64)
            else:
                self.labels = self.labels.astype(np.float32)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
   
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")
        
  
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        if self.is_classification:
            encoding["labels"] = torch.tensor(label, dtype=torch.long)
        elif self.task == "sub_metrics":
            encoding["labels"] = torch.tensor(label, dtype=torch.float)
        else:
            encoding["labels"] = torch.tensor(label, dtype=torch.float).view(1)
        
        return encoding

def load_hf_data(language: str,
    task: str = "question_type",
    split: str = "train",
    control_index: Optional[int] = None,
    cache_dir: str = "./data/cache") -> pd.DataFrame:


    config_name = "base"
    if control_index is not None:
        if task == "question_type":
            config_name = f"control_question_type_seed{control_index}"
        elif task == "complexity":
            config_name = f"control_complexity_seed{control_index}"
        else:
            config_name = f"control_{task}_seed{control_index}"
    
    logger.info(f"Loading {config_name} dataset ({language}, {split})")
    
    try:
        dataset = load_dataset(DATASET_NAME, name=config_name, split=split, cache_dir=cache_dir)
        df = dataset.to_pandas()
        
      
        df = df[df["language"] == language].reset_index(drop=True)
        logger.info(f"Loaded {len(df)} examples for {language}, {task}, {split}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise




def prepare_sparse_matrices(features):
    """Ensure features are in CSR sparse matrix format."""
    # If already sparse, convert to CSR if needed
    if sparse.issparse(features):
        return sparse.csr_matrix(features)
    
    logger.warning(f"Features are not sparse, attempting to convert type: {type(features)}")
    
    try:
        # If it's a numpy array of sparse matrices, stack them
        if isinstance(features, np.ndarray) and features.shape[1] == 1:
            sparse_matrices = []
            for i in range(features.shape[0]):
                if sparse.issparse(features[i, 0]):
                    sparse_matrices.append(features[i, 0])
                else:
                    logger.warning(f"Non-sparse item found at index {i}")
                    # Add an empty matrix
                    sparse_matrices.append(sparse.csr_matrix((1, 128104)))
            
            return sparse.vstack(sparse_matrices)
        
        # Direct conversion attempt
        return sparse.csr_matrix(features)
    except Exception as e:
        logger.error(f"Error converting to sparse matrix: {str(e)}")
        # Provide a placeholder matrix
        return sparse.csr_matrix((features.shape[0] if hasattr(features, 'shape') else 100, 128104))

def load_tfidf_features(split: str, vectors_dir: str = "./data/features"):
    """Load TF-IDF features from file."""
    file_path = os.path.join(vectors_dir, f"tfidf_vectors_{split}.pkl")
    processed_path = os.path.join(vectors_dir, f"processed_{split}_vectors.pkl")
    
    logger.info(f"Loading TF-IDF features from {file_path}")
    
    # Try the processed file first, if it exists
    if os.path.exists(processed_path):
        try:
            with open(processed_path, 'rb') as f:
                vectors = pickle.load(f)
            
            # Check if it's already in the right format
            if sparse.issparse(vectors):
                return vectors
        except Exception as e:
            logger.warning(f"Error loading processed vectors: {e}")
    
    # If that fails, try the original file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TF-IDF features not found at {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            vectors = pickle.load(f)
        
        # Check if it's already a sparse matrix
        if sparse.issparse(vectors):
            return vectors
        
        # If it's an array of sparse matrices, stack them
        if isinstance(vectors, np.ndarray) and vectors.shape[1] == 1:
            logger.info("Found array of sparse matrices, converting to stacked format")
            sparse_matrices = []
            
            for matrix in vectors:
                if sparse.issparse(matrix[0]):
                    sparse_matrices.append(matrix[0])
                else:
                    logger.warning("Non-sparse matrix found, creating empty matrix")
                    # Use a default shape that matches your data
                    sparse_matrices.append(sparse.csr_matrix((1, 128104)))
            
            stacked_vectors = sparse.vstack(sparse_matrices)
            logger.info(f"Stacked {len(sparse_matrices)} matrices to shape {stacked_vectors.shape}")
            
            # Save the processed version for future use
            with open(processed_path, 'wb') as f:
                pickle.dump(stacked_vectors, f)
            
            return stacked_vectors
        
        # Fallback: try direct conversion
        return sparse.csr_matrix(vectors)
    
    except Exception as e:
        logger.error(f"Error loading TF-IDF features: {e}")
        # Create an empty matrix as last resort
        logger.warning("Creating empty matrix as fallback")
        return sparse.csr_matrix((100, 128104))

def load_sklearn_data(
    languages: List[str],
    task: str = "question_type",
    submetric: Optional[str] = None,
    control_index: Optional[int] = None,
    cache_dir: str = "./data/cache",
    vectors_dir: str = "./data/features"):
   
 
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    for language in languages:
        train_dfs.append(load_hf_data(language, task, "train", control_index, cache_dir))
        val_dfs.append(load_hf_data(language, task, "validation", control_index, cache_dir))
        test_dfs.append(load_hf_data(language, task, "test", None, cache_dir))
    
  
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    

    train_features = load_tfidf_features("train", vectors_dir)
    val_features = load_tfidf_features("dev", vectors_dir) 
    test_features = load_tfidf_features("test", vectors_dir)
    
    train_matrices = [prepare_sparse_matrices(train_features) for _ in languages]
    val_matrices = [prepare_sparse_matrices(val_features) for _ in languages]
    test_matrices = [prepare_sparse_matrices(test_features) for _ in languages]
    
    if len(languages) > 1:
        train_features = vstack(train_matrices)
        val_features = vstack(val_matrices)
        test_features = vstack(test_matrices)
    else:
        train_features = train_matrices[0]
        val_features = val_matrices[0]
        test_features = test_matrices[0]


    if task == "question_type":
        feature_name = "question_type"
    elif task == "complexity":
        feature_name = "lang_norm_complexity_score"
    elif task == "single_submetric" and submetric is not None:
        
        feature_name = submetric
    else:
  
        feature_name = TASK_TO_FEATURE.get(task, task)
    
    train_labels = train_df[feature_name].values
    val_labels = val_df[feature_name].values
    test_labels = test_df[feature_name].values
    
    if task == "question_type":
        train_labels = train_labels.astype(np.int64)
        val_labels = val_labels.astype(np.int64)
        test_labels = test_labels.astype(np.int64)
    else:
        train_labels = train_labels.astype(np.float32)
        val_labels = val_labels.astype(np.float32)
        test_labels = test_labels.astype(np.float32)
    
    return (train_features, train_labels), (val_features, val_labels), (test_features, test_labels)

def create_lm_dataloaders(
    language: str,
    task: str = "question_type",
    model_name: str = "cis-lmu/glot500-base",
    batch_size: int = 16,
    control_index: Optional[int] = None,
    cache_dir: str = "./data/cache",
    num_workers: int = 4):
   
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_df = load_hf_data(language, task, "train", control_index, cache_dir)
    val_df = load_hf_data(language, task, "validation", control_index, cache_dir)
    test_df = load_hf_data(language, task, "test", None, cache_dir)  # Always use real test data
    
    train_dataset = LMQuestionDataset(train_df, tokenizer, task)
    val_dataset = LMQuestionDataset(val_df, tokenizer, task)
    test_dataset = LMQuestionDataset(test_df, tokenizer, task)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=True)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader