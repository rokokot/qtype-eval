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

def main():
    parser = argparse.ArgumentParser(description="Cache HuggingFace resources for offline use")
    parser.add_argument("--cache-dir", type=str, default="./data/cache", help="Cache directory for datasets and models")
    args = parser.parse_args()
    
    os.makedirs(args.cache_dir, exist_ok=True)
    
    logger.info(f" cache {len(CONFIGS)} dataset configurations in total")
    cache_datasets(args.cache_dir)
    cache_model(args.cache_dir)
    
    logger.info("ok, complete!")

if __name__ == "__main__":
    main()