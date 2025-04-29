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


# Train data columns: ['unique_id', 'text', 'language', 'avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len', 'avg_verb_edges', 'lexical_density', 'n_tokens', 'question_type']

TASK_TO_FEATURE = {
    "question_type": {
        "feature": "question_type",
        "task_type": "classification",
        "label_type": np.int64
    },
    "complexity": {
        "feature": "lang_norm_complexity_score",
        "task_type": "regression",
        "label_type": np.float32
    },
    "single_submetric": {
        "feature": None, 
        "task_type": "regression",
        "label_type": np.float32
    },
    "avg_links_len": {
        "feature": "avg_links_len",
        "task_type": "regression",
        "label_type": np.float32
    },
    "avg_max_depth": {
        "feature": "avg_max_depth",
        "task_type": "regression",
        "label_type": np.float32
    },
    "avg_subordinate_chain_len": {
        "feature": "avg_subordinate_chain_len",
        "task_type": "regression",
        "label_type": np.float32
    },
    "avg_verb_edges": {
        "feature": "avg_verb_edges",
        "task_type": "regression",
        "label_type": np.float32
    },
    "lexical_density": {
        "feature": "lexical_density",
        "task_type": "regression",
        "label_type": np.float32
    },
    "n_tokens": {
        "feature": "n_tokens",
        "task_type": "regression",
        "label_type": np.float32
    }
}


def ensure_string_task(task):
    # Handle None or empty input
    if task is None:
        return "question_type"
    
    # If task is a list, take the first non-empty item
    if isinstance(task, list):
        task = next((str(t).strip().lower() for t in task if t), "question_type")
    
    # Convert to string and lowercase
    task = str(task).strip().lower()
    
    task_mapping = {
        "question_type": "question_type",
        "complexity": "complexity",
        "complexity_score": "complexity",
        "lang_norm_complexity_score": "complexity",
        "single_submetric": "single_submetric"
    }
    
    submetrics = [
        "avg_links_len", "avg_max_depth", "avg_subordinate_chain_len", 
        "avg_verb_edges", "lexical_density", "n_tokens"
    ]
    
    if task in submetrics:
        return task
    
    return task_mapping.get(task, task)

def get_feature_name_from_task(task, submetric=None, available_columns=None):
    
    task = task.strip().lower() if isinstance(task, str) else "question_type"
    logger.info(f"Getting feature name for task: '{task}', submetric: '{submetric}'")
    
    valid_submetrics = [
        "avg_links_len", "avg_max_depth", 
        "avg_subordinate_chain_len", "avg_verb_edges", 
        "lexical_density", "n_tokens"
    ]
    
    feature_name = None
    
    if task == "single_submetric" and submetric is not None:
        if submetric in valid_submetrics:
            feature_name = submetric
        else:
            logger.warning(f"Invalid submetric: '{submetric}'. Using default 'avg_links_len'.")
            feature_name = "avg_links_len"
    
    elif task in valid_submetrics:
        feature_name = task
    
    else:
        task_mapping = {
            "question_type": "question_type",
            "complexity": "lang_norm_complexity_score",
            "complexity_score": "lang_norm_complexity_score",
            "lang_norm_complexity_score": "lang_norm_complexity_score"
        }
        
        feature_name = task_mapping.get(task)
        
        if not feature_name:
            logger.warning(f"Unrecognized task: '{task}'. Using default 'question_type'.")
            feature_name = "question_type"
    
    if available_columns is not None:
        if feature_name not in available_columns:
            logger.error(f"Selected feature '{feature_name}' not found in available columns: {available_columns}")
            
            # Try to find alternative feature for complexity
            if feature_name == "lang_norm_complexity_score" and "complexity_score" in available_columns:
                logger.info("Falling back to 'complexity_score' feature")
                feature_name = "complexity_score"
            # Try to find any of the valid submetrics
            elif any(sm in available_columns for sm in valid_submetrics):
                for sm in valid_submetrics:
                    if sm in available_columns:
                        logger.info(f"Falling back to available submetric: '{sm}'")
                        feature_name = sm
                        break
            # Default to a required column that should always be there
            elif "question_type" in available_columns:
                logger.info("Falling back to 'question_type' feature")
                feature_name = "question_type"
            else:
                logger.error(f"Cannot find a valid feature in available columns: {available_columns}")
                raise ValueError(f"No valid feature found in columns: {available_columns}")
    
    logger.info(f"Selected feature name: '{feature_name}' for task: '{task}'")
    return feature_name

class MultilingualQuestionDataset(Dataset): 
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
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer, task: str, max_length: int = 128, submetric: Optional[str] = None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate and normalize task
        self.task = task.strip().lower() if isinstance(task, str) else "question_type"
        self.submetric = submetric
        
        # Check required text column
        if "text" not in data.columns:
            available_cols = data.columns.tolist()
            raise ValueError(f"Required column 'text' not found in data. Available columns: {available_cols}")
        
        self.texts = data["text"].tolist()
        
        # Determine task type
        self.is_classification = self.task == "question_type"
        logger.info(f"Task '{self.task}' is classification: {self.is_classification}")
        
        # Get the appropriate feature name with validation
        self.feature_name = get_feature_name_from_task(
            self.task, 
            self.submetric,
            available_columns=data.columns.tolist()
        )
        
        # Validate feature column exists
        if self.feature_name not in data.columns:
            raise ValueError(f"Feature '{self.feature_name}' not found in data columns: {data.columns.tolist()}")
        
        # Extract labels
        self.labels = data[self.feature_name].values
        
        # Log basic statistics about the labels
        logger.info(f"Label statistics for {self.task} (feature: {self.feature_name}):")
        if self.is_classification:
            # Convert to appropriate type for classification
            self.labels = self.labels.astype(np.int64)
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                logger.info(f"  Label {label}: {count} examples ({count/len(self.labels)*100:.1f}%)")
        else:
            # Convert to appropriate type for regression
            self.labels = self.labels.astype(np.float32)
            logger.info(f"  Min: {np.min(self.labels):.4f}, Max: {np.max(self.labels):.4f}")
            logger.info(f"  Mean: {np.mean(self.labels):.4f}, Std: {np.std(self.labels):.4f}")
        
        # Sample validation
        if len(self.texts) > 0:
            logger.info(f"Sample text: {self.texts[0][:50]}...")
            logger.info(f"Sample label: {self.labels[0]}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        try:
            text = self.texts[idx]
            label = self.labels[idx]
    
            # Tokenize text
            encoding = self.tokenizer(
                text, 
                max_length=self.max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
    
            # Add label with appropriate type
            if self.is_classification:
                encoding["labels"] = torch.tensor(label, dtype=torch.long)
            else:
                encoding["labels"] = torch.tensor(label, dtype=torch.float).reshape(-1)
    
            return encoding
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            logger.error(f"Text: {self.texts[idx] if idx < len(self.texts) else 'Index out of range'}")
            logger.error(f"Label: {self.labels[idx] if idx < len(self.labels) else 'Index out of range'}")
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
    
def load_hf_data(language, task, split, control_index=None, cache_dir=None, submetric=None):
    
    config_name = "base"
    using_control = False

    if control_index is not None:
        using_control = True
        if task == "single_submetric" and submetric is not None:
            config_name = f"control_{submetric}_seed{control_index}"
        # For standard tasks
        elif task == "question_type":
            config_name = f"control_question_type_seed{control_index}"
        elif task in ["complexity", "complexity_score", "lang_norm_complexity_score"]:
            config_name = f"control_complexity_seed{control_index}"
        # For direct submetric tasks
        elif task in ["avg_links_len", "avg_max_depth", "avg_subordinate_chain_len", "avg_verb_edges", "lexical_density", "n_tokens"]:
            config_name = f"control_{task}_seed{control_index}"
        else:
            logger.warning(f"Unknown task '{task}' for control data. Using base config.")
    
    logger.info(f"Loading '{config_name}' dataset for {language} language ({split})")
    
    try:
        dataset = load_dataset(
            DATASET_NAME, 
            name=config_name,  
            split=split,
            cache_dir=cache_dir,
            verification_mode='no_checks'
        )
        
        if language != "all":
            original_len = len(dataset)
            dataset = dataset.filter(lambda example: example["language"] == language)
            filtered_len = len(dataset)
            
            if filtered_len == 0:
                logger.warning(f"No examples found for language '{language}' in {config_name} ({split})")
                if original_len > 0:
                    all_langs = set(dataset["language"])
                    logger.info(f"Available languages: {all_langs}")
            else:
                logger.info(f"Filtered from {original_len} to {filtered_len} examples for language '{language}'")
        
        df = dataset.to_pandas()
        
        logger.info(f"Columns in dataset: {list(df.columns)}")
        logger.info(f"Loaded {len(df)} examples for {language} ({split})")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data for {language} from config '{config_name}': {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
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


def load_sklearn_data(languages: List[str], task: str = "question_type", submetric: Optional[str] = None, control_index: Optional[int] = None, cache_dir: str = "./data/cache", vectors_dir: str = "./data/features"):

    train_features = load_tfidf_features("train", vectors_dir)
    val_features = load_tfidf_features("dev", vectors_dir)
    test_features = load_tfidf_features("test", vectors_dir)

    logger.info(f"Loaded feature matrices with shapes: {train_features.shape}, {val_features.shape}, {test_features.shape}")

    train_df = load_combined_dataset("train", task, control_index, cache_dir)
    val_df = load_combined_dataset("validation", task, control_index, cache_dir)
    test_df = load_combined_dataset("test", None, cache_dir)  
    
    # Use the helper function to get the correct feature name
    feature_name = get_feature_name_from_task(task, submetric)
    
    logger.info(f"Using feature: {feature_name} for task: {task}")
    
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
    submetric: Optional[str] = None
):
    
    logger.info(f"Creating dataloaders for language: '{language}', task: '{task}', submetric: '{submetric}'")
    
    try:
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                cache_dir=cache_dir,
                local_files_only=True
            )
            logger.info(f"Successfully loaded tokenizer for {model_name}")
        except Exception as tokenizer_error:
            logger.error(f"Error loading tokenizer: {tokenizer_error}")
            import traceback
            logger.error(f"Tokenizer traceback: {traceback.format_exc()}")
            raise
        
        # Load datasets with improved error handling
        try:
            # Use the enhanced load_hf_data function with submetric
            train_df = load_hf_data(language, task, "train", control_index, cache_dir, submetric)
            val_df = load_hf_data(language, task, "validation", None, cache_dir, submetric)
            test_df = load_hf_data(language, task, "test", None, cache_dir, submetric)
            
            logger.info(f"Loaded datasets: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} examples")
        except Exception as load_error:
            logger.error(f"Error loading datasets: {load_error}")
            import traceback
            logger.error(f"Dataset loading traceback: {traceback.format_exc()}")
            raise
        
        # Create datasets with the enhanced LMQuestionDataset class
        try:
            train_dataset = LMQuestionDataset(train_df, tokenizer, task, submetric=submetric)
            val_dataset = LMQuestionDataset(val_df, tokenizer, task, submetric=submetric)
            test_dataset = LMQuestionDataset(test_df, tokenizer, task, submetric=submetric)
            
            logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        except Exception as dataset_error:
            logger.error(f"Error creating datasets: {dataset_error}")
            import traceback
            logger.error(f"Dataset creation traceback: {traceback.format_exc()}")
            raise
        
        # Create dataloaders
        try:
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
            logger.error(f"Dataloader creation traceback: {traceback.format_exc()}")
            raise
    
    except Exception as e:
        logger.error(f"Error in create_lm_dataloaders: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
