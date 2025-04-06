#!/usr/bin/env python3

import logging
import os
import argparse
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "rokokot/question-type-and-complexity"
VALID_TASKS = ["question_type", "complexity", "single_submetric", 
               "avg_links_len", "avg_max_depth", "avg_subordinate_chain_len", 
               "avg_verb_edges", "lexical_density", "n_tokens"]

def diagnostic_load_dataset(config_name, split, cache_dir, language=None):
    """Load dataset and diagnose configuration issues."""
    logger.info(f"Loading config '{config_name}', split '{split}'")
    
    try:
        dataset = load_dataset(
            DATASET_NAME, 
            name=config_name,
            split=split,
            cache_dir=cache_dir,
            verification_mode='no_checks'
        )
        
        logger.info(f"Successfully loaded {len(dataset)} examples")
        
        # Check columns
        columns = dataset.column_names
        logger.info(f"Available columns: {columns}")
        
        # Check for key columns
        for col in ["text", "language", "question_type", "lang_norm_complexity_score"]:
            if col in columns:
                logger.info(f"Column '{col}' is present")
            else:
                logger.warning(f"Column '{col}' is MISSING")
        
        # Check languages
        if "language" in columns:
            languages = set(dataset["language"])
            logger.info(f"Available languages: {languages}")
            
            # Filter by language if specified
            if language:
                original_len = len(dataset)
                dataset = dataset.filter(lambda example: example["language"] == language)
                filtered_len = len(dataset)
                logger.info(f"Filtered from {original_len} to {filtered_len} examples for language '{language}'")
        
        # Convert to pandas for easier inspection
        df = dataset.to_pandas()
        
        # Check for NaN values
        na_cols = df.isna().sum()
        if na_cols.sum() > 0:
            logger.warning("Found NaN values in columns:")
            for col, count in na_cols[na_cols > 0].items():
                logger.warning(f"  {col}: {count} NaN values")
        
        # Show sample data
        logger.info("Sample row:")
        for col in df.columns:
            if col == "text":
                text_sample = df[col].iloc[0]
                logger.info(f"  {col}: {text_sample[:50]}...")
            else:
                logger.info(f"  {col}: {df[col].iloc[0]}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main(args):
    logger.info(f"Running dataset diagnostics:")
    logger.info(f"  Task: {args.task}")
    logger.info(f"  Language: {args.language}")
    logger.info(f"  Cache dir: {args.cache_dir}")
    
    # Check for task-specific configuration
    config_name = "base"
    if args.control_index is not None:
        if args.task == "question_type":
            config_name = f"control_question_type_seed{args.control_index}"
        elif args.task in ["complexity", "complexity_score", "lang_norm_complexity_score"]:
            config_name = f"control_complexity_seed{args.control_index}"
        elif args.submetric:
            config_name = f"control_{args.submetric}_seed{args.control_index}"
        else:
            config_name = f"control_{args.task}_seed{args.control_index}"
    
    # Check dataset configurations
    for split in ["train", "validation", "test"]:
        logger.info(f"\n=== Loading {split} split with config '{config_name}' ===")
        df = diagnostic_load_dataset(config_name, split, args.cache_dir, args.language)
        
        if df is not None:
            # Determine column to use
            if args.task == "question_type":
                feature_col = "question_type"
            elif args.task in ["complexity", "complexity_score"]:
                feature_col = "lang_norm_complexity_score" if "lang_norm_complexity_score" in df.columns else "complexity_score"
            elif args.task == "single_submetric" and args.submetric:
                feature_col = args.submetric
            else:
                feature_col = args.task
            
            # Check if column exists
            if feature_col in df.columns:
                logger.info(f"Feature column '{feature_col}' exists")
                # Show statistics
                if args.task == "question_type":
                    label_counts = df[feature_col].value_counts()
                    logger.info(f"Label distribution: {label_counts.to_dict()}")
                else:
                    logger.info(f"Numeric stats: min={df[feature_col].min()}, max={df[feature_col].max()}, mean={df[feature_col].mean()}")
            else:
                logger.error(f"Feature column '{feature_col}' DOES NOT EXIST in available columns: {df.columns.tolist()}")
                
                # Suggest alternatives
                if args.task in ["complexity", "complexity_score", "lang_norm_complexity_score"]:
                    alternatives = [col for col in df.columns if "complex" in col.lower()]
                    if alternatives:
                        logger.info(f"Possible alternatives: {alternatives}")
                        
                elif args.task == "single_submetric" or args.submetric:
                    submetrics = ["avg_links_len", "avg_max_depth", "avg_subordinate_chain_len", 
                                 "avg_verb_edges", "lexical_density", "n_tokens"]
                    available_submetrics = [col for col in df.columns if col in submetrics]
                    if available_submetrics:
                        logger.info(f"Available submetrics: {available_submetrics}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose dataset loading and column selection")
    parser.add_argument("--task", type=str, default="question_type", choices=VALID_TASKS,
                        help="Task to diagnose")
    parser.add_argument("--language", type=str, default="en",
                        help="Language to filter for")
    parser.add_argument("--control-index", type=int, default=None,
                        help="Control dataset index (1, 2, or 3)")
    parser.add_argument("--submetric", type=str, default=None,
                        help="Submetric for single_submetric task")
    parser.add_argument("--cache-dir", type=str, default="./data/cache",
                        help="Cache directory for datasets")
    
    args = parser.parse_args()
    main(args)