# scripts/test_tfidf_integration.py
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def test_tfidf_feature_loader(features_dir="./data/tfidf_features"):
    """Test TF-IDF feature loading."""
    logger.info(f"Testing TF-IDF feature loader with directory: {features_dir}")
    
    try:
        from src.data.tfidf_features import TfidfFeatureLoader
        
        if not Path(features_dir).exists():
            logger.error(f"TF-IDF features directory not found: {features_dir}")
            logger.error("Please run: make generate-tfidf-tiny (for testing) or make generate-tfidf")
            return False
        
        loader = TfidfFeatureLoader(features_dir)
        
        # Test loading individual splits
        for split in ['train', 'val', 'test']:
            try:
                features = loader.load_features(split)
                logger.info(f"✓ Loaded {split} features: {features.shape}")
            except FileNotFoundError:
                # Try alternative split name
                alt_split = 'dev' if split == 'val' else split
                try:
                    features = loader.load_features(alt_split)
                    logger.info(f"✓ Loaded {split} features (as {alt_split}): {features.shape}")
                except FileNotFoundError:
                    logger.error(f"✗ Could not load {split} features")
                    return False
        
        # Test loading all features
        all_features = loader.load_all_features()
        logger.info(f"✓ Loaded all features: {list(all_features.keys())}")
        
        # Test language filtering
        filtered = loader.filter_by_languages(all_features, ['en'])
        logger.info(f"✓ Language filtering works")
        
        # Test verification
        if loader.verify_features():
            logger.info("✓ Feature verification passed")
        else:
            logger.warning("⚠ Feature verification failed")
        
        logger.info(f"✓ TF-IDF feature loader test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TF-IDF feature loader test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_data_integration(features_dir="./data/tfidf_features"):
    """Test integration with existing data loading."""
    logger.info("Testing data loading integration...")
    
    try:
        from src.data.datasets import load_sklearn_data
        
        # Test with different configurations
        test_configs = [
            {'languages': ['en'], 'task': 'question_type'},
            {'languages': ['all'], 'task': 'complexity'},
            {'languages': ['en'], 'task': 'single_submetric', 'submetric': 'avg_links_len'},
        ]
        
        for config in test_configs:
            logger.info(f"Testing config: {config}")
            
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
                vectors_dir=features_dir,
                use_tfidf_loader=True,
                **config
            )
            
            logger.info(f"  ✓ Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
            logger.info(f"  ✓ Labels: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
            
            # Basic sanity checks
            assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature dimensions mismatch"
            assert len(y_train) == X_train.shape[0], "Train labels/features mismatch"
            assert len(y_val) == X_val.shape[0], "Val labels/features mismatch"
            assert len(y_test) == X_test.shape[0], "Test labels/features mismatch"
        
        logger.info("✓ Data loading integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loading integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model_creation(features_dir="./data/tfidf_features"):
    """Test TF-IDF model creation."""
    logger.info("Testing TF-IDF model creation...")
    
    try:
        from src.models.tfidf_baselines import create_tfidf_baseline_model
        
        # Test different model types
        test_models = [
            {'model_type': 'dummy', 'task_type': 'classification'},
            {'model_type': 'dummy', 'task_type': 'regression'},
            {'model_type': 'logistic', 'task_type': 'classification'},
            {'model_type': 'ridge', 'task_type': 'regression'},
            {'model_type': 'xgboost', 'task_type': 'classification'},
            {'model_type': 'xgboost', 'task_type': 'regression'},
        ]
        
        for config in test_models:
            logger.info(f"Testing model: {config}")
            
            model = create_tfidf_baseline_model(
                tfidf_features_dir=features_dir,
                target_languages=['en'],
                **config
            )
            
            logger.info(f"  ✓ Created {config['model_type']} for {config['task_type']}")
            
            # Test model info
            info = model.get_model_info()
            logger.info(f"  ✓ Model info: {info['model_type']}, vocab_size: {info.get('vocab_size', 'unknown')}")
        
        logger.info("✓ Model creation test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model creation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_training_pipeline(features_dir="./data/tfidf_features"):
    """Test complete training pipeline."""
    logger.info("Testing complete training pipeline...")
    
    try:
        from src.data.datasets import load_sklearn_data
        from src.models.tfidf_baselines import create_tfidf_baseline_model
        from src.training.sklearn_trainer import SklearnTrainer
        
        # Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
            languages=['en'], 
            task='question_type',
            vectors_dir=features_dir,
            use_tfidf_loader=True
        )
        
        logger.info(f"✓ Loaded data: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        # Create model
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=features_dir,
            target_languages=['en']
        )
        
        logger.info("✓ Created model")
        
        # Create trainer
        output_dir = "./outputs/tfidf_test"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        trainer = SklearnTrainer(
            model=model.model,
            task_type='classification',
            output_dir=output_dir
        )
        
        logger.info("✓ Created trainer")
        
        # Train
        results = trainer.train(
            train_data=(X_train, y_train),
            val_data=(X_val, y_val),
            test_data=(X_test, y_test)
        )
        
        logger.info(f"✓ Training completed")
        logger.info(f"  Test accuracy: {results['test_metrics']['accuracy']:.3f}")
        logger.info(f"  Test F1: {results['test_metrics']['f1']:.3f}")
        
        # Check output files
        assert Path(output_dir, "results.json").exists(), "Results file not created"
        assert Path(output_dir, "model.joblib").exists(), "Model file not created"
        
        logger.info("✓ Training pipeline test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Training pipeline test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_hydra_integration():
    """Test Hydra configuration integration."""
    logger.info("Testing Hydra configuration integration...")
    
    try:
        from omegaconf import OmegaConf
        
        # Test loading TF-IDF config
        config_file = Path("configs/experiment/tfidf_baselines.yaml")
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_file}")
            logger.warning("Skipping Hydra integration test")
            return True
        
        config = OmegaConf.load(config_file)
        logger.info(f"✓ Loaded config: {config.experiment_type}")
        
        # Validate config structure
        required_keys = ['models', 'tasks', 'languages', 'tfidf']
        for key in required_keys:
            assert key in config, f"Missing required config key: {key}"
        
        logger.info(f"✓ Config validation passed")
        logger.info(f"  Models: {config.models}")
        logger.info(f"  Tasks: {config.tasks}")
        logger.info(f"  Languages: {config.languages}")
        
        logger.info("✓ Hydra integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Hydra integration test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests."""
    import argparse
    parser = argparse.ArgumentParser(description="Test TF-IDF integration")
    parser.add_argument("--features-dir", default="./data/tfidf_features", 
                       help="Directory containing TF-IDF features")
    args = parser.parse_args()
    
    logger.info("Starting TF-IDF integration tests...")
    logger.info(f"Using features directory: {args.features_dir}")
    
    tests = [
        ("TF-IDF Feature Loader", lambda: test_tfidf_feature_loader(args.features_dir)),
        ("Data Integration", lambda: test_data_integration(args.features_dir)), 
        ("Model Creation", lambda: test_model_creation(args.features_dir)),
        ("Training Pipeline", lambda: test_training_pipeline(args.features_dir)),
        ("Hydra Integration", test_hydra_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! TF-IDF integration is working correctly.")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())