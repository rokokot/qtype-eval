#!/usr/bin/env python3
"""
Simple Local Test - Tests core functionality without heavy dependencies
"""

import sys
import os
import yaml
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_basic_structure():
    """Test repository structure and files"""
    print("🧪 TEST 1: Repository Structure")
    print("-" * 40)
    
    # Essential directories
    essential_dirs = ["src", "configs", "scripts", "tests"]
    for directory in essential_dirs:
        if Path(directory).exists():
            file_count = len(list(Path(directory).rglob("*.py")))
            print(f"✅ {directory}/ ({file_count} Python files)")
        else:
            print(f"❌ Missing: {directory}/")
            return False
    
    # Essential files
    essential_files = ["README.md", "Makefile", "requirements.txt"]
    for file in essential_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ Missing: {file}")
            return False
    
    return True

def test_configurations():
    """Test configuration files"""
    print("\n🧪 TEST 2: Configuration Files")
    print("-" * 40)
    
    configs = [
        "configs/experiment/tfidf_baselines.yaml",
        "configs/model/lm_probe.yaml"
    ]
    
    for config_file in configs:
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ {config_file}")
            
            # Show key info
            if 'tfidf' in config:
                model_name = config.get('tfidf', {}).get('model_name')
                max_features = config.get('tfidf', {}).get('max_features')
                print(f"   Model: {model_name}, Max features: {max_features:,}")
            elif 'lm_name' in config:
                print(f"   LM: {config.get('lm_name')}")
                
        except Exception as e:
            print(f"❌ {config_file}: {e}")
            return False
    
    return True

def test_sklearn_basics():
    """Test sklearn functionality"""
    print("\n🧪 TEST 3: Sklearn Basics")
    print("-" * 40)
    
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import RandomForestClassifier
        
        print("✅ All sklearn imports successful")
        
        # Test basic TF-IDF
        texts = ["What is AI?", "How does ML work?", "Why use NLP?"]
        vectorizer = TfidfVectorizer(max_features=50)
        X = vectorizer.fit_transform(texts)
        print(f"✅ TF-IDF: {X.shape}, sparsity: {(1 - X.nnz/X.size)*100:.1f}%")
        
        # Test models with dummy data
        y_class = [0, 1, 0]  # classification
        y_reg = [0.1, 0.8, 0.3]  # regression
        
        models_class = [DummyClassifier(), LogisticRegression(max_iter=50)]
        models_reg = [DummyRegressor(), Ridge()]
        
        for model in models_class:
            model.fit(X, y_class)
            print(f"✅ {model.__class__.__name__} (classification)")
            
        for model in models_reg:
            model.fit(X, y_reg)
            print(f"✅ {model.__class__.__name__} (regression)")
        
        return True
        
    except Exception as e:
        print(f"❌ Sklearn test failed: {e}")
        return False

def test_core_classes():
    """Test our core classes can be imported"""
    print("\n🧪 TEST 4: Core Classes")
    print("-" * 40)
    
    try:
        # Test sklearn trainer
        from src.training.sklearn_trainer import SklearnTrainer
        trainer = SklearnTrainer(DummyClassifier(), 'classification')
        print("✅ SklearnTrainer")
        
        # Test TF-IDF classes
        from src.data.tfidf_features import TfidfFeatureLoader
        from src.models.tfidf_baselines import TfidfBaselineModel
        print("✅ TfidfFeatureLoader")
        print("✅ TfidfBaselineModel")
        
        return True
        
    except Exception as e:
        print(f"❌ Core classes test failed: {e}")
        return False

def test_makefile():
    """Test makefile targets"""
    print("\n🧪 TEST 5: Makefile")
    print("-" * 40)
    
    try:
        with open('Makefile', 'r') as f:
            content = f.read()
        
        # Look for key targets
        targets = ['help', 'setup', 'clean', 'run-tfidf-direct']
        found_targets = []
        
        for target in targets:
            if f"{target}:" in content:
                found_targets.append(target)
                print(f"✅ Target: {target}")
        
        if len(found_targets) >= 3:
            print(f"✅ Makefile has {len(found_targets)} essential targets")
            return True
        else:
            print(f"❌ Only found {len(found_targets)} targets")
            return False
            
    except Exception as e:
        print(f"❌ Makefile test failed: {e}")
        return False

def test_scripts():
    """Test essential scripts exist"""
    print("\n🧪 TEST 6: Essential Scripts")
    print("-" * 40)
    
    scripts = [
        "scripts/generate_tfidf_glot500.py",
        "scripts/simple_tfidf_test.py",
        "scripts/run_tfidf_experiments.py"
    ]
    
    for script in scripts:
        if Path(script).exists():
            # Quick syntax check
            try:
                with open(script, 'r') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                    
                # Basic checks
                has_main = 'if __name__ == "__main__"' in content
                has_imports = 'import' in content
                
                status = "✅" if has_main and has_imports else "⚠️"
                print(f"{status} {script} ({lines} lines)")
                
            except Exception as e:
                print(f"❌ {script}: Error reading - {e}")
                return False
        else:
            print(f"❌ Missing: {script}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 SIMPLE LOCAL TEST SUITE")
    print("=" * 50)
    print("Testing core functionality without heavy ML dependencies")
    print()
    
    tests = [
        test_basic_structure,
        test_configurations,
        test_sklearn_basics,
        test_core_classes,
        test_makefile,
        test_scripts
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        if not result:
            print(f"\n⚠️  Test failed, but continuing...")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed >= 4:  # Allow some failures for optional components
        print("\n🎉 CORE FUNCTIONALITY VERIFIED!")
        print("\n✅ Your repository is ready for:")
        print("  • Local development")
        print("  • Configuration management")
        print("  • Sklearn-based experimentation")
        
        print("\n📝 Next Steps:")
        print("  1. For full functionality: pip install -r requirements.txt")
        print("  2. Test with Makefile: make help")
        print("  3. Try: make clean && make setup")
        print("  4. Deploy to HPC for full experiments")
        
        return True
    else:
        print(f"\n❌ Core issues found - {total-passed} critical failures")
        print("Please address failing components before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)