# data loading and processing utilities for multilingual question probing experiments
# uses pre-extracted TF-IDF features

import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from scipy.sparse import vstack
from scipy import sparse
import logging

logger = logging.getLogger(__name__)

DATASET_NAME = "rokokot/question-type-and-complexity"
CACHE_DIR = os.environ.get("HF_HOME", "./data/cache")


TASK_TO_FEATURE = {
    "question_type": "question_type",
    "complexity": "lang_norm_complexity_score",
    "avg_links_len": "avg_links_len",
    "avg_max_depth": "avg_max_depth",
    "avg_subordinate_chain_len": "avg_subordinate_chain_len",
    "avg_verb_edges": "avg_verb_edges",
    "lexical_density": "lexical_density",
    "n_tokens": "n_tokens",
}


class MultilingualQuestionDataset(Dataset):  # Dataset for sklearn models using TF-IDF features
    def __init__(
        self, data: pd.DataFrame, features: np.ndarray, labels: np.ndarray, language_ids: Optional[np.ndarray] = None
    ):
        self.data = data
        self.features = features
        self.labels = labels
        self.language_ids = language_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {"features": self.features[idx], "label": self.labels[idx]}

        if self.language_ids is not None:
            item["language_id"] = self.language_ids[idx]

        return item


class LMQuestionDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, task: str, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length

        self.is_classification = task == "question_type"

        self.texts = data["text"].tolist()

        feature_name = TASK_TO_FEATURE[task]

        if task == "sub_metrics":
            submetrics = [
                "avg_links_len",
                "avg_max_depth",
                "avg_subordinate_chain_len",
                "avg_verb_edges",
                "lexical_density",
                "n_tokens",
            ]
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

        encoding = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        if self.is_classification:
            encoding["labels"] = torch.tensor(label, dtype=torch.long)
        elif self.task == "sub_metrics":
            encoding["labels"] = torch.tensor(label, dtype=torch.float)
        else:
            encoding["labels"] = torch.tensor(label, dtype=torch.float).view(1)

        return encoding

def load_combined_dataset(
    split: str,
    task: str = "question_type",
    control_index: Optional[int] = None,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """Load the combined dataset for all languages."""
    config_name = "base"
    if control_index is not None:
        if task == "question_type":
            config_name = f"control_question_type_seed{control_index}"
        elif task == "complexity":
            config_name = f"control_complexity_seed{control_index}"
        else:
            config_name = f"control_{task}_seed{control_index}"

    logger.info(f"Loading {config_name} dataset (all languages, {split})")

    try:
        dataset = load_dataset(DATASET_NAME, name=config_name, split=split, cache_dir=cache_dir)
        df = dataset.to_pandas()
        logger.info(f"Loaded {len(df)} examples for {split}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
    
def load_hf_data(language, task, split, control_index=None, cache_dir=CACHE_DIR):
    config_name = "base"
    if control_index is not None:
        if task == "question_type":
            config_name = f"control_question_type_seed{control_index}"
        elif task == "complexity":
            config_name = f"control_complexity_seed{control_index}"
        else:
            config_name = f"control_{task}_seed{control_index}"

    logger.info(f"Loading {config_name} dataset for {language} language ({split})")

    try:
        # Try loading with local_files_only first if offline mode is set
        local_only = os.environ.get("HF_DATASETS_OFFLINE", "0") == "1"
        
        try:
            if local_only:
                dataset = load_dataset(
                    DATASET_NAME, 
                    name=config_name, 
                    split=split, 
                    cache_dir=cache_dir,
                    local_files_only=True  # Force using cached version
                )
                logger.info("Successfully loaded dataset from local cache")
            else:
                dataset = load_dataset(
                    DATASET_NAME, 
                    name=config_name, 
                    split=split, 
                    cache_dir=cache_dir
                )
        except Exception as cache_error:
            if local_only:
                logger.warning(f"Could not load from cache with local_files_only=True: {cache_error}")
                logger.info("Trying again without local_files_only restriction...")
                dataset = load_dataset(
                    DATASET_NAME, 
                    name=config_name, 
                    split=split, 
                    cache_dir=cache_dir,
                    local_files_only=False
                )
            else:
                raise
        
        if language != "all":
            dataset = dataset.filter(lambda example: example["language"] == language)
            
        df = dataset.to_pandas()
        logger.info(f"Loaded {len(df)} examples for {language} ({split})")
        return df
    except Exception as e:
        logger.error(f"Error loading data for {language}: {e}")
        raise

def load_tfidf_features(split: str, vectors_dir: str = "./data/features"):
   
    file_path = os.path.join(vectors_dir, f"tfidf_vectors_{split}.pkl")
    
    logger.info(f"Loading TF-IDF features from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TF-IDF features not found at {file_path}")
    
    try:
        with open(file_path, "rb") as f:
            vectors = pickle.load(f)
        
        if isinstance(vectors, list) or (isinstance(vectors, np.ndarray) and vectors.ndim > 1):
            sparse_matrices = [matrix[0] for matrix in vectors]
            stacked_vectors = sparse.vstack(sparse_matrices)

            logger.info(f"Stacked {len(sparse_matrices)} matrices to shape {stacked_vectors.shape}, all ok")
            return stacked_vectors
        
        if sparse.issparse(vectors):
            return vectors
        
        return sparse.csr_matrix(vectors)
    
    except Exception as e:
        logger.error(f"Error loading TF-IDF features: {e}")
        logger.warning("Creating empty matrix as fallback")
        return sparse.csr_matrix((100, 128104))


def load_sklearn_data(languages: List[str],task: str = "question_type",submetric: Optional[str] = None,control_index: Optional[int] = None,cache_dir: str = "./data/cache",vectors_dir: str = "./data/features",):

    train_features = load_tfidf_features("train", vectors_dir)
    val_features = load_tfidf_features("dev", vectors_dir)
    test_features = load_tfidf_features("test", vectors_dir)


    logger.info(f"Loaded feature matrices with shapes: {train_features.shape}, {val_features.shape}, {test_features.shape}")

    train_df = load_combined_dataset("train", task, control_index, cache_dir)
    val_df = load_combined_dataset("validation", task, control_index, cache_dir)
    test_df = load_combined_dataset("test", None, cache_dir)  
    
    if task == "question_type":
        feature_name = "question_type"
    elif task == "complexity":
        feature_name = "lang_norm_complexity_score"
    elif task == "single_submetric" and submetric is not None:
        feature_name = submetric
    else:
        feature_name = TASK_TO_FEATURE.get(task, task)
    
    # Extract labels
    train_labels = train_df[feature_name].values
    val_labels = val_df[feature_name].values
    test_labels = test_df[feature_name].values
    
    # Convert to appropriate types
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
    num_workers: int = 4,
):


    try:
        # Use offline-compatible tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            local_files_only=True,  # Force using cache
            use_fast=True  # Use faster tokenizer implementation
        )
        logger.info(f"Loaded tokenizer: {model_name}")
    except Exception as e:
        logger.warning(f"Error loading tokenizer with local_files_only=True: {e}")
        logger.info("Trying again without local_files_only...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


    train_df = load_hf_data(language, task, "train", control_index, cache_dir)
    val_df = load_hf_data(language, task, "validation", None, cache_dir)
    test_df = load_hf_data(language, task, "test", None, cache_dir)  # Always use real test data

    train_dataset = LMQuestionDataset(train_df, tokenizer, task)
    val_dataset = LMQuestionDataset(val_df, tokenizer, task)
    test_dataset = LMQuestionDataset(test_df, tokenizer, task)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
