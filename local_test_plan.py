#!/usr/bin/env python3
"""
Local Testing Plan - No Heavy Computing Required
Tests core functionality without downloading large models or datasets.
"""

import sys
import os
import yaml
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_imports():
    """Test 1: Core module imports"""
    print("🧪 TEST 1: Core Module Imports")
    print("-" * 40)
    
    tests = [
        ("src.data.datasets", "load_labels, get_available_languages"),
        ("src.models.tfidf_baselines", "create_tfidf_baseline_model"),
        ("src.training.sklearn_trainer", "SklearnTrainer"),
        ("src.data.tfidf_features", "TfidfFeatureLoader"),
    ]
    
    for module_name, items in tests:
        try:
            exec(f"from {module_name} import {items}")
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            return False
    
    print("✅ All imports successful!")
    return True

def test_configs():
    """Test 2: Configuration loading"""
    print("\n🧪 TEST 2: Configuration Files")
    print("-" * 40)
    
    config_files = [
        "configs/experiment/tfidf_baselines.yaml",
        "configs/model/lm_probe.yaml",
        "configs/model/glot500_finetune.yaml"
    ]
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ {config_file}")
            if 'tfidf' in config_file:
                print(f"   Model: {config.get('tfidf', {}).get('model_name', 'N/A')}")
        except Exception as e:
            print(f"❌ {config_file}: {e}")
            return False
    
    print("✅ All configs loaded successfully!")
    return True

def test_sklearn_functionality():
    """Test 3: Sklearn functionality without real data"""
    print("\n🧪 TEST 3: Sklearn Functionality")
    print("-" * 40)
    
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.dummy import DummyClassifier
        from src.training.sklearn_trainer import SklearnTrainer
        
        # Create fake data
        fake_texts = [
            "What is the capital of France?",
            "How does photosynthesis work?", 
            "When was the first computer built?",
            "Why do birds migrate?",
            "Where is the Sahara desert?"
        ]
        fake_labels = [0, 1, 0, 1, 0]  # Binary labels
        
        # Test TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(fake_texts)
        print(f"✅ TF-IDF vectorization: {X.shape}")
        
        # Test sklearn models
        models = [
            ("DummyClassifier", DummyClassifier()),
            ("LogisticRegression", LogisticRegression(max_iter=100))
        ]
        
        for name, model in models:
            model.fit(X, fake_labels)
            accuracy = model.score(X, fake_labels)
            print(f"✅ {name}: accuracy={accuracy:.3f}")
            
        # Test our trainer (minimal test)
        trainer = SklearnTrainer(DummyClassifier(), 'classification')
        print(f"✅ SklearnTrainer initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Sklearn functionality test failed: {e}")
        return False

def test_directory_structure():
    """Test 4: Directory structure and essential files"""
    print("\n🧪 TEST 4: Directory Structure")
    print("-" * 40)
    
    essential_dirs = ["src", "configs", "scripts", "tests", "data", "logs", "outputs", "models"]
    essential_files = ["README.md", "Makefile", "requirements.txt", "setup.py"]
    
    for directory in essential_dirs:
        if Path(directory).exists():
            print(f"✅ {directory}/")
        else:
            print(f"❌ Missing: {directory}/")
            return False
            
    for file in essential_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ Missing: {file}")
            return False
    
    return True

def test_scripts_exist():
    """Test 5: Essential scripts exist and are readable"""
    print("\n🧪 TEST 5: Essential Scripts")
    print("-" * 40)
    
    scripts = [
        "scripts/generate_tfidf_glot500.py",
        "scripts/simple_tfidf_test.py", 
        "scripts/run_tfidf_experiments.py"
    ]
    
    for script in scripts:
        if Path(script).exists():
            try:
                with open(script, 'r') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                print(f"✅ {script} ({lines} lines)")
            except Exception as e:
                print(f"❌ {script}: Cannot read - {e}")
                return False
        else:
            print(f"❌ Missing: {script}")
            return False
    
    return True

def test_mock_data_pipeline():
    """Test 6: Mock data pipeline without external dependencies"""
    print("\n🧪 TEST 6: Mock Data Pipeline")
    print("-" * 40)
    
    try:
        # Test that we can create the basic pipeline components
        from src.data.tfidf_features import TfidfFeatureLoader
        from src.models.tfidf_baselines import TfidfBaselineModel
        
        # Mock TF-IDF loader (won't actually load real data)
        print("✅ TfidfFeatureLoader class available")
        print("✅ TfidfBaselineModel class available")
        
        # Test configuration loading for models
        from src.data.datasets import get_available_languages, get_available_tasks
        
        languages = get_available_languages()
        tasks = get_available_tasks()
        
        print(f"✅ Available languages: {languages}")
        print(f"✅ Available tasks: {tasks}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock data pipeline test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("🧪 LOCAL TESTING SUITE - NO HEAVY COMPUTING")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configs,
        test_sklearn_functionality, 
        test_directory_structure,
        test_scripts_exist,
        test_mock_data_pipeline
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL {total} TESTS PASSED!")
        print("\n✅ Your repository is ready for:")
        print("  1. Local development and testing")
        print("  2. HPC deployment") 
        print("  3. Full-scale experiments")
        print("\nNext steps:")
        print("  - Install full dependencies: pip install -r requirements.txt")
        print("  - Test with small datasets: make generate-tfidf-tiny")
        print("  - Deploy to HPC for full experiments")
        return True
    else:
        print(f"❌ {total - passed} out of {total} tests failed.")
        print("Please fix failing components before proceeding.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)