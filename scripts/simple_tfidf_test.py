#!/usr/bin/env python3
"""
Fixed version of the TF-IDF integration test script.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def test_dependencies():
    """Test if all required dependencies are available."""
    logger.info("Checking dependencies...")
    
    try:
        import torch
        logger.info(f"PyTorch version {torch.__version__} available.")
        
        import sklearn
        import numpy as np
        import pandas as pd
        import scipy
        import transformers
        import datasets
        
        logger.info("✓ All basic dependencies available")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        return False

def test_tfidf_features(features_dir):
    """Test if TF-IDF feature files exist."""
    logger.info(f"Checking TF-IDF features in {features_dir}...")
    
    required_files = [
        "metadata.json",
        "X_train_sparse.npz", 
        "X_val_sparse.npz",
        "X_test_sparse.npz"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(features_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"✗ Missing TF-IDF files: {missing_files}")
        return False
    
    logger.info("✓ All TF-IDF feature files found")
    return True

def test_module_imports():
    """Test if our custom modules can be imported."""
    logger.info("Testing module imports...")
    
    try:
        from src.data.datasets import load_sklearn_data
        logger.info("✓ Imported datasets module")
        
        from src.models.tfidf_baselines import create_tfidf_baseline_model
        logger.info("✓ Imported tfidf_baselines module")
        
        from src.experiments.sklearn_trainer import SklearnTrainer
        logger.info("✓ Imported sklearn_trainer module")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_tfidf_loading(features_dir):
    """Test TF-IDF feature loading."""
    logger.info("Testing TF-IDF feature loading...")
    
    try:
        from src.data.tfidf_features import TfidfFeatureLoader
        
        loader = TfidfFeatureLoader(features_dir)
        metadata = loader.metadata
        if metadata:
            logger.info(f"Loaded TF-IDF metadata: {metadata.get('vocab_size', metadata.get('n_features', 'unknown'))} features")
        else:
            logger.info("No metadata available")
        
        # Test loading each split
        train_features = loader.load_features('train')
        val_features = loader.load_features('val')
        test_features = loader.load_features('test')
        
        logger.info(f"✓ Loaded features: train={train_features.shape}, val={val_features.shape}, test={test_features.shape}")
        
        # Test loading all splits at once
        all_features = loader.load_all_features()
        splits = list(all_features.keys())
        logger.info(f"✓ Loaded all features: {splits}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ TF-IDF loading failed: {e}")
        return False

def test_data_integration(features_dir):
    """Test data loading integration with TF-IDF features."""
    logger.info("Testing data loading integration...")
    
    try:
        from src.data.datasets import load_sklearn_data
        
        # Test with TF-IDF features - FIXED parameter name
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
            task="question_type",
            languages=["en"], 
            tfidf_features_dir="./data/tfidf_features_tiny"
        )
        
        logger.info(f"✓ Data loaded successfully:")
        logger.info(f"  Train: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"  Val: X={X_val.shape}, y={y_val.shape}")
        logger.info(f"  Test: X={X_test.shape}, y={y_test.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data integration failed: {e}")
        import traceback
        logger.error("Traceback:")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test TF-IDF model creation."""
    logger.info("Testing TF-IDF model creation...")
    
    try:
        from src.models.tfidf_baselines import create_tfidf_baseline_model
        
        # Test different model types
        model_types = ["dummy", "logistic", "random_forest"]
        
        for model_type in model_types:
            model = create_tfidf_baseline_model(model_type, task_type="classification", tfidf_features_dir="./data/tfidf_features_tiny")
            logger.info(f"✓ Created {model_type} model: {type(model).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        return False

def test_end_to_end_training(features_dir):
    """Test end-to-end training with TF-IDF features."""
    logger.info("Testing end-to-end training...")
    
    try:
        from src.data.datasets import load_sklearn_data
        from src.models.tfidf_baselines import create_tfidf_baseline_model
        from src.experiments.sklearn_trainer import SklearnTrainer
        
        # Load small sample of data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
            task="question_type",
            languages=["en"],
            tfidf_features_dir=features_dir)
        
        # Take only first 100 samples for quick test
        X_train_small = X_train[:100]
        y_train_small = y_train[:100]
        X_val_small = X_val[:50] 
        y_val_small = y_val[:50]
        
        model = create_tfidf_baseline_model("dummy", task_type="classification",tfidf_features_dir=features_dir)
        
        trainer = SklearnTrainer(model)
        
        # Train the model
        trainer.train(X_train_small, y_train_small)
        
        # Evaluate
        train_metrics = trainer.evaluate(X_train_small, y_train_small, "train")
        val_metrics = trainer.evaluate(X_val_small, y_val_small, "val")
        
        logger.info(f"✓ Training completed:")
        logger.info(f"  Train accuracy: {train_metrics.get('accuracy', 'N/A')}")
        logger.info(f"  Val accuracy: {val_metrics.get('accuracy', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ End-to-end training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test TF-IDF integration")
    parser.add_argument("--features-dir", required=True, help="Directory containing TF-IDF features")
    args = parser.parse_args()
    
    features_dir = args.features_dir
    
    logger.info("Starting TF-IDF setup verification...")
    logger.info(f"Using features directory: {features_dir}")
    
    # Define tests
    tests = [
        ("Dependencies", lambda: test_dependencies()),
        ("TF-IDF Features", lambda: test_tfidf_features(features_dir)),
        ("Module Imports", lambda: test_module_imports()),
        ("TF-IDF Loading", lambda: test_tfidf_loading(features_dir)),
        ("Data Integration", lambda: test_data_integration(features_dir)),
        ("Model Creation", lambda: test_model_creation()),
        ("End-to-End Training", lambda: test_end_to_end_training(features_dir)),
    ]
    
    # Run tests
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"RUNNING: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED - stopping here")
                break
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results[test_name] = False
            break
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    if passed == total:
        logger.info(f"\n🎉 All {total} tests passed! TF-IDF integration is working correctly.")
        logger.info("\nNext steps:")
        logger.info("  make run-tfidf-minimal")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} out of {total} tests failed.")
        logger.info("\nNext steps:")
        logger.info("  Fix the failing tests and try again")
        return 1

if __name__ == "__main__":
    sys.exit(main())