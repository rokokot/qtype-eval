#!/usr/bin/env python3

"""
Helper script to cache the dataset, model, and tokenizer to a local directory.

"""


import os
import argparse
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "rokokot/question-type-and-complexity"
MODEL_NAME = "cis-lmu/glot500-base"
LANGUAGES = ["ar", "en", "fi", "id", "ja", "ko", "ru"]

CONFIGS = ["base"]

CONFIGS += [f"control_question_type_seed{i}" for i in range(1, 4)]

CONFIGS += [f"control_complexity_seed{i}" for i in range(1, 4)]

SUBMETRICS = ["avg_links_len", "avg_max_depth","avg_subordinate_chain_len","avg_verb_edges","lexical_density","n_tokens"]

for submetric in SUBMETRICS:
    CONFIGS += [f"control_{submetric}_seed{i}" for i in range(1, 4)]

SPLITS = ["train", "validation", "test"]

def cache_datasets(cache_dir):
    logger.info(f"saving datasets to {cache_dir}")
    
    for config in CONFIGS:
        logger.info(f"saving dataset config: {config}")
        for split in SPLITS:
            try:
                dataset = load_dataset(DATASET_NAME, name=config, split=split, cache_dir=cache_dir)
                logger.info(f" cached {config} ({split}): {len(dataset)} examples")
            except Exception as e:
                logger.error(f"error  {config} ({split}): {e}")
                logger.warning(f"skip {config} ({split}) CHECK FILES")

def cache_model(cache_dir):
    logger.info(f" model: {MODEL_NAME}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
        logger.info(f" cached tokenizer: {MODEL_NAME}")
        
        model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
        logger.info(f" cached model: {MODEL_NAME}")
    except Exception as e:
        logger.error(f"error: {e}")

def verify_cache(cache_dir):
    """Verify that all required resources are cached and accessible offline."""
    logger.info("Verifying cached resources...")
    
    missing_resources = []
    
    # Verify datasets
    for config in CONFIGS:
        for split in SPLITS:
            try:
                dataset = load_dataset(DATASET_NAME, name=config, split=split, cache_dir=cache_dir)
                logger.debug(f"Verified {config} ({split}): {len(dataset)} examples")
            except Exception as e:
                missing_resources.append(f"Dataset {config} ({split})")
                logger.warning(f"Missing dataset {config} ({split}): {e}")
    
    # Verify model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir, local_files_only=True)
        logger.debug(f"Verified tokenizer: {MODEL_NAME}")
    except Exception as e:
        missing_resources.append(f"Tokenizer {MODEL_NAME}")
        logger.warning(f"Missing tokenizer: {e}")
    
    try:
        model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir, local_files_only=True)
        logger.debug(f"Verified model: {MODEL_NAME}")
    except Exception as e:
        missing_resources.append(f"Model {MODEL_NAME}")
        logger.warning(f"Missing model: {e}")
    
    if missing_resources:
        logger.error(f"Missing resources: {missing_resources}")
        return False
    else:
        logger.info("All resources verified and available offline!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Cache HuggingFace resources for offline use")
    parser.add_argument("--cache-dir", type=str, default="./data/cache", help="Cache directory for datasets and models")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing cache, don't download")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cache exists")
    args = parser.parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    
    if args.verify_only:
        success = verify_cache(args.cache_dir)
        exit(0 if success else 1)
    
    if not args.force and verify_cache(args.cache_dir):
        logger.info("All resources already cached. Use --force to re-download.")
        return
    
    logger.info(f"Caching {len(CONFIGS)} dataset configurations in total")
    cache_datasets(args.cache_dir)
    cache_model(args.cache_dir)
    
    logger.info("Caching complete! Verifying...")
    verify_cache(args.cache_dir)

if __name__ == "__main__":
    main()