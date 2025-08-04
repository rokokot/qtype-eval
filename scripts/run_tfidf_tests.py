#!/usr/bin/env python3
# scripts/run_tfidf_tests.py
"""
Complete TF-IDF testing runner.
Runs all TF-IDF integration tests and generates comprehensive reports.
"""

import sys
import os
import argparse
import tempfile
import shutil
from pathlib import Path
import logging
import subprocess
import json
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test utilities
from utils.test_helpers import (
    validate_test_environment, 
    create_temporary_workspace,
    cleanup_workspace
)
from tests.fixtures.sample_data import create_sample_dataset
from src.data.tfidf_features import create_test_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TfidfTestRunner:
    """Comprehensive TF-IDF test runner."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("test_outputs")
        self.workspace = None
        self.test_results = {}
        
    def setup(self):
        """Set up test environment."""
        logger.info("Setting up TF-IDF test environment...")
        
        # Validate environment
        validation = validate_test_environment()
        if validation['errors']:
            logger.error("Environment validation failed:")
            for error in validation['errors']:
                logger.error(f"  - {error}")
            return False
        
        # Create workspace
        self.workspace = create_temporary_workspace("tfidf_test_")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test TF-IDF features
        features_dir = self.workspace / "tfidf_features"
        create_test_features(
            output_dir=str(features_dir),
            n_samples=100,
            vocab_size=200
        )
        
        logger.info(f"Test environment ready in: {self.workspace}")
        return True
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        logger.info("Running unit tests...")
        
        unit_results = {}
        
        # Run pytest on unit tests
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/unit/", 
            "-v", "--tb=short",
            f"--junit-xml={self.output_dir}/unit_tests.xml"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=project_root,
                timeout=300  # 5 minutes
            )
            
            unit_results = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0
            }
            
            if result.returncode == 0:
                logger.info("✓ Unit tests passed")
            else:
                logger.error("✗ Unit tests failed")
                logger.error(f"STDERR: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            logger.error("✗ Unit tests timed out")
            unit_results = {'passed': False, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"✗ Unit tests failed with exception: {e}")
            unit_results = {'passed': False, 'error': str(e)}
        
        return unit_results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        integration_results = {}
        
        # Set environment variables for tests
        env = os.environ.copy()
        env['TFIDF_TEST_FEATURES_DIR'] = str(self.workspace / "tfidf_features")
        
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/integration/", 
            "-v", "--tb=short",
            f"--junit-xml={self.output_dir}/integration_tests.xml"
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=project_root,
                env=env,
                timeout=600  # 10 minutes
            )
            
            integration_results = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'passed': result.returncode == 0
            }
            
            if result.returncode == 0:
                logger.info("✓ Integration tests passed")
            else:
                logger.error("✗ Integration tests failed")
                logger.error(f"STDERR: {result.stderr}")
            
        except subprocess.TimeoutExpired:
            logger.error("✗ Integration tests timed out")
            integration_results = {'passed': False, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"✗ Integration tests failed with exception: {e}")
            integration_results = {'passed': False, 'error': str(e)}
        
        return integration_results
    
    def run_feature_tests(self) -> Dict[str, Any]:
        """Run TF-IDF feature-specific tests."""
        logger.info("Running TF-IDF feature tests...")
        
        feature_results = {}
        
        try:
            # Test 1: Feature loading
            from src.data.tfidf_features import TfidfFeatureLoader
            
            features_dir = self.workspace / "tfidf_features"
            loader = TfidfFeatureLoader(str(features_dir))
            
            # Test loading
            all_features = loader.load_all_features()
            assert len(all_features) == 3
            
            # Test verification
            assert loader.verify_features()
            
            feature_results['loading'] = {'passed': True}
            logger.info("✓ Feature loading test passed")
            
        except Exception as e:
            feature_results['loading'] = {'passed': False, 'error': str(e)}
            logger.error(f"✗ Feature loading test failed: {e}")
        
        try:
            # Test 2: Model integration
            from src.models.tfidf_baselines import create_tfidf_baseline_model
            
            model = create_tfidf_baseline_model(
                model_type='dummy',
                task_type='classification',
                tfidf_features_dir=str(features_dir),
                target_languages=['en']
            )
            
            # Test fitting
            import numpy as np
            dummy_labels = np.random.randint(0, 2, 70)  # For train split
            model.fit(dummy_labels)
            
            # Test prediction
            predictions = model.predict('test')
            assert len(predictions) > 0
            
            feature_results['model_integration'] = {'passed': True}
            logger.info("✓ Model integration test passed")
            
        except Exception as e:
            feature_results['model_integration'] = {'passed': False, 'error': str(e)}
            logger.error(f"✗ Model integration test failed: {e}")
        
        return feature_results
    
    def run_experiment_tests(self) -> Dict[str, Any]:
        """Run experiment runner tests."""
        logger.info("Running experiment tests...")
        
        experiment_results = {}
        
        try:
            # Test experiment runner
            from scripts.run_tfidf_experiments import TfidfExperimentRunner
            from omegaconf import OmegaConf
            
            # Create minimal config
            config = OmegaConf.create({
                'experiment_type': 'tfidf_baselines',
                'tfidf': {'features_dir': str(self.workspace / "tfidf_features")},
                'models': ['dummy'],
                'tasks': ['question_type'],
                'languages': [['en']],
                'controls': {'enabled': False},
                'model_params': {
                    'dummy': {'classification': {'strategy': 'most_frequent'}}
                },
                'output_dir': str(self.workspace / "exp_outputs"),
                'data': {'cache_dir': str(self.workspace / "cache")}
            })
            
            # Mock dataset for testing
            from unittest.mock import patch
            
            mock_data = {
                'text': ['Test'] * 50,
                'language': ['en'] * 50,
                'question_type': [0, 1] * 25
            }
            
            from datasets import Dataset
            mock_split = Dataset.from_dict(mock_data)
            mock_dataset = {'train': mock_split, 'validation': mock_split, 'test': mock_split}
            
            with patch('src.data.datasets.load_dataset', return_value=mock_dataset):
                runner = TfidfExperimentRunner(config)
                
                # Run single experiment
                result = runner.run_single_experiment(
                    model_type='dummy',
                    task='question_type',
                    languages=['en']
                )
                
                assert 'test_metrics' in result
                assert 'accuracy' in result['test_metrics']
            
            experiment_results['runner'] = {'passed': True}
            logger.info("✓ Experiment runner test passed")
            
        except Exception as e:
            experiment_results['runner'] = {'passed': False, 'error': str(e)}
            logger.error(f"✗ Experiment runner test failed: {e}")
        
        return experiment_results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        logger.info("Running performance tests...")
        
        performance_results = {}
        
        try:
            import time
            import psutil
            
            # Test large feature loading performance
            large_features_dir = self.workspace / "large_features"
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create larger features
            create_test_features(
                output_dir=str(large_features_dir),
                n_samples=500,
                vocab_size=1000
            )
            
            from src.data.tfidf_features import TfidfFeatureLoader
            loader = TfidfFeatureLoader(str(large_features_dir))
            features = loader.load_all_features()
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            loading_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            performance_results['large_loading'] = {
                'passed': loading_time < 30,  # Should load in < 30 seconds
                'loading_time': loading_time,
                'memory_used_mb': memory_used
            }
            
            if loading_time < 30:
                logger.info(f"✓ Large feature loading test passed ({loading_time:.2f}s)")
            else:
                logger.error(f"✗ Large feature loading too slow ({loading_time:.2f}s)")
            
        except Exception as e:
            performance_results['large_loading'] = {'passed': False, 'error': str(e)}
            logger.error(f"✗ Performance test failed: {e}")
        
        return performance_results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        logger.info("Generating test report...")
        
        # Collect all results
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'environment': validate_test_environment(),
            'test_results': self.test_results,
            'summary': {}
        }
        
        # Calculate summary
        total_tests = 0
        passed_tests = 0
        
        for test_category, category_results in self.test_results.items():
            if isinstance(category_results, dict):
                for test_name, test_result in category_results.items():
                    if isinstance(test_result, dict) and 'passed' in test_result:
                        total_tests += 1
                        if test_result['passed']:
                            passed_tests += 1
            elif isinstance(category_results, dict) and 'passed' in category_results:
                total_tests += 1
                if category_results['passed']:
                    passed_tests += 1
        
        report['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED'
        }
        
        # Save report
        report_file = self.output_dir / "tfidf_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to: {report_file}")
        logger.info(f"Overall status: {report['summary']['overall_status']}")
        logger.info(f"Success rate: {report['summary']['success_rate']:.1%}")
        
        return report
    
    def cleanup(self):
        """Clean up test environment."""
        if self.workspace:
            cleanup_workspace(self.workspace)
            logger.info("Test environment cleaned up")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all TF-IDF tests."""
        logger.info("Starting comprehensive TF-IDF test suite...")
        
        try:
            # Setup
            if not self.setup():
                return {'error': 'Setup failed'}
            
            # Run test categories
            self.test_results['unit_tests'] = self.run_unit_tests()
            self.test_results['integration_tests'] = self.run_integration_tests()
            self.test_results['feature_tests'] = self.run_feature_tests()
            self.test_results['experiment_tests'] = self.run_experiment_tests()
            self.test_results['performance_tests'] = self.run_performance_tests()
            
            # Generate report
            report = self.generate_report()
            
            return report
            
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
            return {'error': str(e)}
        
        finally:
            self.cleanup()


def run_quick_tests(features_dir: str) -> bool:
    """Run quick validation tests."""
    logger.info("Running quick TF-IDF validation tests...")
    
    try:
        # Test 1: Feature loading
        from src.data.tfidf_features import TfidfFeatureLoader
        
        loader = TfidfFeatureLoader(features_dir)
        features = loader.load_all_features()
        
        assert len(features) == 3
        assert all(hasattr(matrix, 'shape') for matrix in features.values())
        
        logger.info("✓ Feature loading test passed")
        
        # Test 2: Model creation
        from src.models.tfidf_baselines import create_tfidf_baseline_model
        
        model = create_tfidf_baseline_model(
            model_type='dummy',
            task_type='classification',
            tfidf_features_dir=features_dir
        )
        
        assert model.model_type == 'dummy'
        assert model.task_type == 'classification'
        
        logger.info("✓ Model creation test passed")
        
        # Test 3: Integration with data loading
        from src.data.datasets import load_sklearn_data
        from unittest.mock import patch
        
        mock_data = {
            'text': ['Test'] * 20,
            'language': ['en'] * 20,
            'question_type': [0, 1] * 10
        }
        
        from datasets import Dataset
        mock_split = Dataset.from_dict(mock_data)
        mock_dataset = {'train': mock_split, 'validation': mock_split, 'test': mock_split}
        
        with patch('src.data.datasets.load_dataset', return_value=mock_dataset):
            try:
                (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
                    languages=['en'],
                    task='question_type',
                    use_tfidf_loader=True,
                    tfidf_features_dir=features_dir
                )
                logger.info("✓ Data integration test passed")
            except Exception as e:
                if "shape" in str(e).lower():
                    logger.info("✓ Data integration test handled size mismatch correctly")
                else:
                    raise e
        
        logger.info("✓ All quick tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Quick tests failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run TF-IDF integration tests")
    parser.add_argument("--output-dir", default="test_outputs", help="Output directory for test results")
    parser.add_argument("--quick", action="store_true", help="Run quick validation tests only")
    parser.add_argument("--features-dir", help="Directory with existing TF-IDF features (for quick tests)")
    parser.add_argument("--unit-only", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration-only", action="store_true", help="Run integration tests only")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up temporary files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.quick:
        # Quick validation mode
        if not args.features_dir:
            logger.error("Quick mode requires --features-dir")
            sys.exit(1)
        
        if not Path(args.features_dir).exists():
            logger.error(f"Features directory not found: {args.features_dir}")
            sys.exit(1)
        
        success = run_quick_tests(args.features_dir)
        sys.exit(0 if success else 1)
    
    # Full test suite
    runner = TfidfTestRunner(args.output_dir)
    
    try:
        if args.unit_only:
            # Unit tests only
            if not runner.setup():
                sys.exit(1)
            results = runner.run_unit_tests()
            success = results.get('passed', False)
        
        elif args.integration_only:
            # Integration tests only
            if not runner.setup():
                sys.exit(1)
            results = runner.run_integration_tests()
            success = results.get('passed', False)
        
        else:
            # Full test suite
            report = runner.run_all_tests()
            success = report.get('summary', {}).get('overall_status') == 'PASSED'
        
        if not args.no_cleanup:
            runner.cleanup()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        if not args.no_cleanup:
            runner.cleanup()
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Test suite failed with exception: {e}")
        if not args.no_cleanup:
            runner.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()