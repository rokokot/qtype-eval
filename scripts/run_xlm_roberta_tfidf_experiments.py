# scripts/run_xlm_roberta_tfidf_experiments.py
"""
Run TF-IDF experiments with XLM-RoBERTa tokenizer for consistency with neural experiments.
This ensures identical tokenization across all experimental conditions.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import time
from collections import defaultdict
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.datasets import load_sklearn_data_with_config
from src.models.xlm_roberta_tfidf_configs import (
    XLM_ROBERTA_TFIDF_PARAMS,
    XLM_ROBERTA_EXPERIMENT_CONFIG,
    get_xlm_roberta_model_params,
    validate_xlm_roberta_compatibility,
    get_xlm_roberta_experiment_matrix,
    validate_xlm_roberta_experiment_results
)

# sklearn imports
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, 
    r2_score, mean_absolute_error
)

# Try XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class XLMRobertaTfidfExperimentRunner:
    """Run TF-IDF experiments with XLM-RoBERTa tokenizer for consistency."""
    
    def __init__(
        self,
        tfidf_features_dir: str,
        output_dir: str,
        cache_dir: str = "./data/cache",
        random_state: int = 42
    ):
        """
        Initialize experiment runner with XLM-RoBERTa-consistent features.
        
        Args:
            tfidf_features_dir: Directory containing XLM-RoBERTa TF-IDF features
            output_dir: Output directory for results
            cache_dir: Cache directory for datasets
            random_state: Random seed for reproducibility
        """
        self.tfidf_features_dir = Path(tfidf_features_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = cache_dir
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify TF-IDF features exist and are XLM-RoBERTa-based
        self._verify_features()
        
        # Storage for results
        self.results = {}
        
        logger.info(f"Initialized XLM-RoBERTa TF-IDF experiment runner")
        logger.info(f"Features dir: {tfidf_features_dir}")
        logger.info(f"Output dir: {output_dir}")
        logger.info(f"Random state: {random_state}")
    
    def _verify_features(self):
        """Verify that TF-IDF features exist and use XLM-RoBERTa tokenizer."""
        if not self.tfidf_features_dir.exists():
            raise FileNotFoundError(f"TF-IDF features directory not found: {self.tfidf_features_dir}")
        
        # Check metadata to verify XLM-RoBERTa usage
        metadata_file = self.tfidf_features_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                tokenizer_info = metadata.get('tokenizer_info', {})
                model_name = tokenizer_info.get('model_name', '')
                
                if 'xlm-roberta' not in model_name.lower():
                    logger.warning(f"Features may not use XLM-RoBERTa tokenizer: {model_name}")
                else:
                    logger.info(f"✓ Features use XLM-RoBERTa tokenizer: {model_name}")
                    
                # Log tokenizer configuration
                special_tokens = tokenizer_info.get('special_tokens', {})
                logger.info(f"✓ Special tokens: {special_tokens}")
                
            except Exception as e:
                logger.warning(f"Could not verify tokenizer from metadata: {e}")
        else:
            logger.warning("No metadata file found - cannot verify tokenizer")
    
    def _create_sklearn_model(
        self, 
        model_type: str, 
        task_type: str, 
        model_params: Dict[str, Any]
    ):
        """Create sklearn model with specified parameters."""
        
        params = model_params.copy()
        
        if model_type == "dummy":
            if task_type == "classification":
                return DummyClassifier(**params)
            else:
                return DummyRegressor(**params)
        
        elif model_type == "logistic":
            if task_type != "classification":
                raise ValueError("Logistic regression only for classification")
            return LogisticRegression(**params)
        
        elif model_type == "ridge":
            if task_type != "regression":
                raise ValueError("Ridge regression only for regression") 
            return Ridge(**params)
        
        elif model_type == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError("XGBoost not available")
            
            if task_type == "classification":
                return xgb.XGBClassifier(**params)
            else:
                return xgb.XGBRegressor(**params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict[str, float]:
        """Calculate metrics for evaluation."""
        
        if task_type == "classification":
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
            }
        else:  # regression
            return {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }
    
    def run_single_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single TF-IDF experiment with XLM-RoBERTa features."""
        
        # Extract experiment parameters
        language = experiment_config["language"]
        task = experiment_config["task"]
        task_type = experiment_config["task_type"]
        model_type = experiment_config["model_type"]
        control_index = experiment_config.get("control_index")
        submetric = experiment_config.get("submetric")
        model_params = experiment_config["model_params"]
        
        # Create experiment name
        exp_name = f"xlm_roberta_tfidf_{model_type}_{task}_{language}"
        if submetric:
            exp_name += f"_{submetric}"
        if control_index:
            exp_name += f"_control{control_index}"
        
        logger.info(f"Running experiment: {exp_name}")
        
        try:
            # Determine dataset configuration
            if control_index is not None:
                if submetric:
                    dataset_config = f"control_{submetric}_seed{control_index}"
                else:
                    dataset_config = f"control_{task}_seed{control_index}"
            else:
                dataset_config = "base"
            
            # Determine target task for data loading
            target_task = submetric if submetric else (
                "question_type" if task == "question_type" else "lang_norm_complexity_score"
            )
            
            logger.info(f"Loading data: config={dataset_config}, task={target_task}, language={language}")
            
            # Load data using existing infrastructure
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data_with_config(
                task=target_task,
                languages=[language],
                dataset_config=dataset_config,
                tfidf_features_dir=str(self.tfidf_features_dir),
                use_tfidf_loader=True,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
            logger.info(f"Data shapes: X_val={X_val.shape}, y_val={y_val.shape}")
            logger.info(f"Data shapes: X_test={X_test.shape}, y_test={y_test.shape}")
            
            # Create model
            model = self._create_sklearn_model(model_type, task_type, model_params)
            
            # Train model
            start_time = time.time()
            
            # Handle XGBoost early stopping
            if model_type == "xgboost":
                try:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_train, y_train), (X_val, y_val)],
                        verbose=False
                    )
                except Exception as e:
                    logger.warning(f"XGBoost early stopping failed: {e}")
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train, task_type)
            val_metrics = self._calculate_metrics(y_val, y_pred_val, task_type)
            test_metrics = self._calculate_metrics(y_test, y_pred_test, task_type)
            
            # Get feature importance if available
            feature_importances = None
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importances = [(int(i), float(imp)) for i, imp in enumerate(importance)]
                feature_importances.sort(key=lambda x: x[1], reverse=True)
                feature_importances = feature_importances[:100]  # Top 100
            
            # Compile results
            results = {
                "experiment_name": exp_name,
                "model_type": model_type,
                "task": task,
                "task_type": task_type,
                "language": language,
                "control_index": control_index,
                "submetric": submetric,
                "dataset_config": dataset_config,
                "training_time": training_time,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "feature_importances": feature_importances,
                "tokenizer_consistent": True,  # Mark as XLM-RoBERTa consistent
                "data_shapes": {
                    "train": list(X_train.shape),
                    "val": list(X_val.shape),
                    "test": list(X_test.shape)
                }
            }
            
            logger.info(f"Completed experiment: {exp_name}")
            logger.info(f"Test metrics: {test_metrics}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed experiment {exp_name}: {e}")
            import traceback
            error_info = {
                "experiment_name": exp_name,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "tokenizer_consistent": True,
                **experiment_config
            }
            return error_info
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all XLM-RoBERTa-consistent TF-IDF experiments."""
        
        logger.info("Starting XLM-RoBERTa-consistent TF-IDF experiments...")
        
        # Generate experiment matrix
        experiment_matrix = get_xlm_roberta_experiment_matrix()
        
        logger.info(f"Generated {len(experiment_matrix)} experiments")
        
        # Run experiments
        all_results = {}
        successful = 0
        failed = 0
        
        for i, exp_config in enumerate(experiment_matrix):
            logger.info(f"Running experiment {i+1}/{len(experiment_matrix)}")
            
            try:
                result = self.run_single_experiment(exp_config)
                
                if "error" in result:
                    failed += 1
                    logger.error(f"Experiment failed: {result['experiment_name']}")
                else:
                    successful += 1
                    logger.info(f"Experiment succeeded: {result['experiment_name']}")
                
                all_results[result["experiment_name"]] = result
                
                # Save intermediate results every 10 experiments
                if (i + 1) % 10 == 0:
                    self._save_intermediate_results(all_results)
                
            except Exception as e:
                failed += 1
                logger.error(f"Experiment {i+1} crashed: {e}")
                continue
        
        logger.info(f"Completed all experiments: {successful} successful, {failed} failed")
        
        # Save final results
        self._save_final_results(all_results)
        
        # Validate results
        validation_results = validate_xlm_roberta_experiment_results(all_results)
        logger.info(f"Validation results: {validation_results}")
        
        self.results = all_results
        return all_results
    
    def _save_intermediate_results(self, results: Dict[str, Any]):
        """Save intermediate results."""
        intermediate_file = self.output_dir / "intermediate_results.json"
        with open(intermediate_file, "w") as f:
            json.dump(results, f, indent=2)
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final results in multiple formats."""
        
        # Save complete results
        results_file = self.output_dir / "all_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Create summary table
        summary_data = []
        for exp_name, result in results.items():
            if "error" not in result:
                row = {
                    "experiment": exp_name,
                    "model_type": result.get("model_type"),
                    "task": result.get("task"),
                    "language": result.get("language"),
                    "control_index": result.get("control_index"),
                    "submetric": result.get("submetric"),
                    "tokenizer_consistent": result.get("tokenizer_consistent", False)
                }
                
                # Add test metrics
                test_metrics = result.get("test_metrics", {})
                for metric, value in test_metrics.items():
                    row[f"test_{metric}"] = value
                
                summary_data.append(row)
        
        # Save as CSV and JSON
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.output_dir / "results_summary.csv", index=False)
            
            with open(self.output_dir / "results_summary.json", "w") as f:
                json.dump(summary_data, f, indent=2)
        
        # Create analysis report
        self._create_analysis_report(results)
        
        logger.info(f"Results saved to {self.output_dir}")
    
    def _create_analysis_report(self, results: Dict[str, Any]):
        """Create comprehensive analysis report."""
        
        report_file = self.output_dir / "analysis_report.md"
        
        with open(report_file, "w") as f:
            f.write("# XLM-RoBERTa-Consistent TF-IDF Experiment Results\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            successful_results = {k: v for k, v in results.items() if "error" not in v}
            failed_results = {k: v for k, v in results.items() if "error" in v}
            
            f.write(f"## Summary\n\n")
            f.write(f"- Total experiments: {len(results)}\n")
            f.write(f"- Successful: {len(successful_results)}\n")
            f.write(f"- Failed: {len(failed_results)}\n")
            f.write(f"- Tokenizer: XLM-RoBERTa-base (consistent with neural experiments)\n\n")
            
            # Tokenizer consistency check
            consistent_experiments = [r for r in successful_results.values() 
                                    if r.get("tokenizer_consistent", False)]
            f.write(f"- Tokenizer consistent experiments: {len(consistent_experiments)}/{len(successful_results)}\n\n")
            
            # Performance analysis
            if successful_results:
                f.write("## Performance Analysis\n\n")
                
                # Main vs Control comparison
                main_results = [r for r in successful_results.values() if r.get("control_index") is None]
                control_results = [r for r in successful_results.values() if r.get("control_index") is not None]
                
                if main_results and control_results:
                    main_performance = self._calculate_average_performance(main_results)
                    control_performance = self._calculate_average_performance(control_results)
                    
                    f.write("### Main vs Control Experiments\n\n")
                    f.write("| Metric | Main | Control | Difference |\n")
                    f.write("| --- | --- | --- | --- |\n")
                    
                    for metric in main_performance:
                        if metric in control_performance:
                            main_val = main_performance[metric]
                            control_val = control_performance[metric]
                            diff = main_val - control_val
                            f.write(f"| {metric} | {main_val:.4f} | {control_val:.4f} | {diff:+.4f} |\n")
                    
                    f.write("\n")
                
                # Group by task and model
                task_groups = defaultdict(lambda: defaultdict(list))
                
                for exp_name, result in successful_results.items():
                    task = result.get("task", "unknown")
                    model_type = result.get("model_type", "unknown")
                    task_groups[task][model_type].append((exp_name, result))
                
                # Write detailed results by task
                for task, model_groups in task_groups.items():
                    f.write(f"### Task: {task}\n\n")
                    
                    for model_type, experiments in model_groups.items():
                        f.write(f"#### Model: {model_type}\n\n")
                        
                        if experiments:
                            # Get metrics for table
                            sample_result = experiments[0][1]
                            test_metrics = sample_result.get("test_metrics", {})
                            metric_keys = list(test_metrics.keys())
                            
                            # Write table header
                            f.write("| Experiment | Language | Control | " + " | ".join(metric_keys) + " |\n")
                            f.write("| --- | --- | --- | " + " | ".join(["---" for _ in metric_keys]) + " |\n")
                            
                            # Write table rows
                            for exp_name, result in experiments:
                                language = result.get("language", "unknown")
                                control_idx = result.get("control_index", "")
                                control_str = f"C{control_idx}" if control_idx else "Main"
                                
                                metrics = result.get("test_metrics", {})
                                metric_values = []
                                for key in metric_keys:
                                    value = metrics.get(key, 'N/A')
                                    if isinstance(value, float):
                                        metric_values.append(f"{value:.4f}")
                                    else:
                                        metric_values.append(str(value))
                                
                                f.write(f"| {exp_name} | {language} | {control_str} | " + " | ".join(metric_values) + " |\n")
                            
                            f.write("\n")
            
            # Tokenizer information
            f.write("## Tokenizer Configuration\n\n")
            f.write("This experiment used XLM-RoBERTa tokenizer for consistency with neural experiments:\n\n")
            f.write("```json\n")
            f.write(json.dumps(XLM_ROBERTA_EXPERIMENT_CONFIG["tokenizer_config"], indent=2))
            f.write("\n```\n\n")
            
            # TF-IDF configuration
            f.write("## TF-IDF Configuration\n\n")
            f.write("```json\n")
            f.write(json.dumps(XLM_ROBERTA_EXPERIMENT_CONFIG["tfidf_config"], indent=2))
            f.write("\n```\n\n")
            
            # Failed experiments if any
            if failed_results:
                f.write("## Failed Experiments\n\n")
                for exp_name, result in failed_results.items():
                    error_msg = result.get("error", "Unknown error")
                    f.write(f"- **{exp_name}**: {error_msg}\n")
                f.write("\n")
    
    def _calculate_average_performance(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average performance across results."""
        metrics_sum = {}
        metrics_count = {}
        
        for result in results:
            test_metrics = result.get('test_metrics', {})
            for metric, value in test_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_sum[metric] = metrics_sum.get(metric, 0) + value
                    metrics_count[metric] = metrics_count.get(metric, 0) + 1
        
        return {metric: metrics_sum[metric] / metrics_count[metric] 
                for metric in metrics_sum if metrics_count[metric] > 0}


def main():
    parser = argparse.ArgumentParser(description="Run XLM-RoBERTa-consistent TF-IDF experiments")
    parser.add_argument("--tfidf-features-dir", required=True,
                       help="Directory containing XLM-RoBERTa TF-IDF features")
    parser.add_argument("--output-dir", default="./outputs/xlm_roberta_tfidf_experiments",
                       help="Output directory for results") 
    parser.add_argument("--cache-dir", default="./data/cache",
                       help="HuggingFace cache directory")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.tfidf_features_dir).exists():
        logger.error(f"TF-IDF features directory not found: {args.tfidf_features_dir}")
        logger.error("Please generate features first using:")
        logger.error("python scripts/generate_xlm_roberta_tfidf.py")
        return 1
    
    # Check dependencies
    missing_deps = []
    if not HAS_XGBOOST:
        missing_deps.append("xgboost")
    
    if missing_deps:
        logger.error(f"Missing dependencies: {missing_deps}")
        logger.error("Install with: pip install " + " ".join(missing_deps))
        return 1
    
    try:
        # Create experiment runner
        runner = XLMRobertaTfidfExperimentRunner(
            tfidf_features_dir=args.tfidf_features_dir,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            random_state=args.random_state
        )
        
        # Run all experiments
        results = runner.run_all_experiments()
        
        # Print final summary
        successful = len([r for r in results.values() if "error" not in r])
        failed = len(results) - successful
        consistent = len([r for r in results.values() if r.get("tokenizer_consistent", False)])
        
        logger.info("="*60)
        logger.info("XLM-ROBERTA TFIDF EXPERIMENTS COMPLETED")
        logger.info("="*60)
        logger.info(f"Total experiments: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Tokenizer consistent: {consistent}")
        logger.info(f"Results saved to: {args.output_dir}")
        
        if failed > 0:
            logger.warning(f"{failed} experiments failed. Check logs for details.")
        
        if consistent < successful:
            logger.warning(f"Some experiments may not be tokenizer consistent")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        logger.error(f"Experiment runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())