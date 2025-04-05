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
print(f"Using Dataset: {DATASET_NAME}")
print(f"Cache Directory: {CACHE_DIR}")

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


def ensure_string_task(task):
    """Make sure a task is a string, not a list."""
    if isinstance(task, list) and len(task) > 0:
        return task[0]
    elif isinstance(task, str):
        return task
    else:
        return "question_type"  # Default fallback

def get_feature_name_from_task(task):
    """Get the feature name from a task, handling both string and list inputs."""
    task_str = ensure_string_task(task)
    return TASK_TO_FEATURE.get(task_str)


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
        self.task = ensure_string_task
        self.max_length = max_length

        self.is_classification = task == "question_type"

        if "text" not in data.columns:
            available_columns = data.columns.tolist()
            logger.error(f"Required column 'text' not found in data. Available columns: {available_columns}")
            raise ValueError(f"Required column 'text' not found in data")

        self.texts = data["text"].tolist()

        feature_name = get_feature_name_from_task(self.task)
        if feature_name is None:
            logger.error(f"Unknown task: {self.task}, available tasks: {list(TASK_TO_FEATURE.keys())}")
            raise ValueError(f"Unknown task: {self.task}")
        
        if feature_name not in data.columns:
            available_columns = data.columns.tolist()
            logger.error(f"Feature '{feature_name}' not found in data columns: {available_columns}")
            raise ValueError(f"Feature '{feature_name}' not found in data")


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
        try:
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
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            logger.error(f"Text: {self.texts[idx] if idx < len(self.texts) else 'Index out of range'}")
            logger.error(f"Label type: {type(self.labels).__name__}")
            logger.error(f"Label shape: {self.labels.shape if hasattr(self.labels, 'shape') else 'N/A'}")
            raise

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
    
def load_hf_data(language, task, split, control_index=None, cache_dir=None):
    
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
        dataset = load_dataset(
            DATASET_NAME, 
            config_name,
            split=split,
            cache_dir=cache_dir,
            verification_mode='no_checks'
        )
        
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
    task = "question_type",
    model_name: str = "cis-lmu/glot500-base",
    batch_size: int = 16,
    control_index: Optional[int] = None,
    cache_dir: str = "./data/cache",
    num_workers: int = 4
):
    
    task_str = ensure_string_task(task)  # Convert to string
    logger.info(f"Creating dataloaders for task: {task_str} (original: {task})")
    
    model_path = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    
    try:
        # Load tokenizer with improved error handling
        try:
            snapshot_path = os.path.join(model_path, "snapshots")
            if os.path.exists(snapshot_path):
                snapshot_dirs = [
                    os.path.join(snapshot_path, d) 
                    for d in os.listdir(snapshot_path) 
                    if os.path.isdir(os.path.join(snapshot_path, d))
                ]
                
                if snapshot_dirs:
                    local_model_path = snapshot_dirs[0]
                    logger.info(f"Found model snapshot: {local_model_path}")
                    
                    tokenizer = AutoTokenizer.from_pretrained(
                        local_model_path,
                        use_fast=True,
                        local_files_only=True
                    )
                else:
                    # Fallback to original model name if no snapshots
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        use_fast=True
                    )
            else:
                # Fallback to original model name if no snapshots directory
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=True
                )
        except Exception as tokenizer_error:
            logger.error(f"Error loading tokenizer: {tokenizer_error}")
            raise
        
        logger.info(f"Successfully loaded tokenizer for {model_name}")
        
        # Load datasets with improved error handling
        try:
            train_df = load_hf_data(language, task, "train", control_index, cache_dir)
            val_df = load_hf_data(language, task, "validation", None, cache_dir)
            test_df = load_hf_data(language, task, "test", None, cache_dir)
        except Exception as load_error:
            logger.error(f"Error loading datasets: {load_error}")
            raise
        
        # Log dataset info for debugging
        logger.info(f"Train data columns: {train_df.columns.tolist()}")
        if 'text' in train_df.columns and len(train_df) > 0:
            logger.info(f"Sample text: {train_df['text'].iloc[0][:100]}...")
        
        feature_name = get_feature_name_from_task(task_str)
        if feature_name and feature_name in train_df.columns:
            logger.info(f"Sample {feature_name}: {train_df[feature_name].iloc[0] if len(train_df) > 0 else 'No samples'}")
        else:
            logger.warning(f"Feature for task '{task_str}' not found in columns: {train_df.columns.tolist()}")
        
        # Create datasets with improved error handling
        try:
            train_dataset = LMQuestionDataset(train_df, tokenizer, task_str)
            logger.info(f"Created train dataset with {len(train_dataset)} examples")
            
            val_dataset = LMQuestionDataset(val_df, tokenizer, task_str)
            logger.info(f"Created validation dataset with {len(val_dataset)} examples")
            
            test_dataset = LMQuestionDataset(test_df, tokenizer, task_str)
            logger.info(f"Created test dataset with {len(test_dataset)} examples")
            
            # Test processing a sample to validate
            sample = train_dataset[0]
            logger.info(f"Sample processed successfully with keys: {list(sample.keys())}")
            
        except Exception as dataset_error:
            logger.error(f"Error creating datasets for task {task_str}: {dataset_error}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Create dataloaders with improved error handling
        try:
            # Reduce num_workers for debugging if needed
            debug_mode = os.environ.get("DEBUG", "0") == "1"
            actual_workers = 0 if debug_mode else num_workers
            
            logger.info(f"Creating dataloaders with {actual_workers} workers")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=actual_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=actual_workers,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=actual_workers,
                pin_memory=True
            )
            
            logger.info("Successfully created all dataloaders")
            return train_loader, val_loader, test_loader
        except Exception as loader_error:
            logger.error(f"Error creating dataloaders: {loader_error}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    except Exception as e:
        logger.error(f"Error in create_lm_dataloaders for task {task_str}: {e}")
        # Additional detailed logging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Log detailed information about the task and model
        logger.error(f"Task (original format): {task}")
        logger.error(f"Task (string format): {task_str}")
        logger.error(f"Model name: {model_name}")
        logger.error(f"Cache directory: {cache_dir}")
        logger.error(f"Language: {language}")
        
        raise