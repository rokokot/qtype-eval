# scripts/data_processing/verify_tfidf_features.py

import os
import pickle
import argparse
import logging
from scipy.sparse import issparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def verify_tfidf_files(vectors_dir):
    """Verify that the TF-IDF feature files exist and are valid."""
    required_files = [
        "tfidf_vectors_train.pkl",
        "tfidf_vectors_dev.pkl",
        "tfidf_vectors_test.pkl",
        "idf_values.pkl",
        "token_to_index_mapping.pkl"
    ]
    
    missing_files = []
    invalid_files = []
    
    for file_name in required_files:
        file_path = os.path.join(vectors_dir, file_name)
        
        # Check if file exists
        if not os.path.exists(file_path):
            missing_files.append(file_name)
            continue
        
        # Check if file is a valid pickle
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                
            # Check if vectors are sparse matrices
            if "vectors" in file_name and not issparse(data):
                invalid_files.append(f"{file_name} (not a sparse matrix)")
                
            logger.info(f"{file_name}: Valid")
            
            # Print shape for vector files
            if "vectors" in file_name:
                if issparse(data):
                    logger.info(f"  Shape: {data.shape}")
                else:
                    logger.info(f"  Type: {type(data)}")
        except Exception as e:
            invalid_files.append(f"{file_name} ({str(e)})")
    
    if missing_files:
        logger.warning("Missing files:")
        for file in missing_files:
            logger.warning(f"  - {file}")
    
    if invalid_files:
        logger.warning("Invalid files:")
        for file in invalid_files:
            logger.warning(f"  - {file}")
    
    if not missing_files and not invalid_files:
        logger.info("All TF-IDF feature files are valid!")
        return True
    else:
        return False

def main(args):

    if not os.path.exists(args.vectors_dir):
        logger.error(f"Vectors directory not found: {args.vectors_dir}")
        return False
    
    return verify_tfidf_files(args.vectors_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify TF-IDF features")
    parser.add_argument("--vectors-dir", type=str, default="./data/features", help="Directory containing TF-IDF features")
    args = parser.parse_args()
    
    success = main(args)
    if not success:
        exit(1)