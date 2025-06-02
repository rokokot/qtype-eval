import os
import pickle
import logging
import numpy as np
from scipy.sparse import issparse, vstack

#  logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def verify_tfidf_files(vectors_dir):

    required_files = ["tfidf_vectors_train.pkl","tfidf_vectors_dev.pkl","tfidf_vectors_test.pkl","idf_values.pkl","token_to_index_mapping.pkl",]

    for file_name in required_files:
        file_path = os.path.join(vectors_dir, file_name)

        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)

            logger.info(f"{file_name}: Loaded successfully")
            
            #  look for vector files
            if "vectors" in file_name:
                split = file_name.split('_')[2].split('.')[0]
                logger.info(f"Processing {split} vectors")
                
                if isinstance(data, list):
                    try:
                        sparse_matrices = [matrix[0] if isinstance(matrix, list) and len(matrix) > 0 else matrix for matrix in data]
                        
                        non_sparse = [type(m) for m in sparse_matrices if not issparse(m)]
                        if non_sparse:
                            logger.warning(f"Non-sparse matrices found: {non_sparse}")
                        
                        stacked_vectors = vstack(sparse_matrices)
                        
                        logger.info(f"{split.capitalize()} vectors:")
                        logger.info(f"  Original list length: {len(data)}")
                        logger.info(f"  Stacked matrix shape: {stacked_vectors.shape}")
                        logger.info(f"  Non-zero elements: {stacked_vectors.nnz}")
                        logger.info(f"  Sparsity: {1 - (stacked_vectors.nnz / np.prod(stacked_vectors.shape)):.2%}")
                    
                    except Exception as stack_error:
                        logger.error(f"Error stacking matrices: {stack_error}")
                
                elif issparse(data):
                    logger.info(f"{split.capitalize()} vectors:")
                    logger.info(f"  Sparse matrix shape: {data.shape}")
                    logger.info(f"  Non-zero elements: {data.nnz}")
                    logger.info(f"  Sparsity: {1 - (data.nnz / np.prod(data.shape)):.2%}")

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")

def main(args):
    if not os.path.exists(args.vectors_dir):
        logger.error(f"Vectors directory not found: {args.vectors_dir}")
        return False

    verify_tfidf_files(args.vectors_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verify TF-IDF features")
    parser.add_argument("--vectors-dir", type=str, default="./data/features", help="Directory containing TF-IDF features")
    args = parser.parse_args()

    main(args)