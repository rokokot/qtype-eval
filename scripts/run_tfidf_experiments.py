# scripts/run_tfidf_experiments.py
"""
TF-IDF baseline experiment runner.
Integrates seamlessly with existing Hydra configuration system.
"""

import os
import sys
import json
import hydra
from pathlib import Path
from typing import Dict, List, Any, Optional
from omegaconf import DictConfig, OmegaConf
import logging
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.datasets import load_sklearn_data
from src.models.tfidf_baselines import create_tfidf_baseline_model
from src.training.sklearn_trainer import SklearnTrainer

logger = logging.getLogger(__name__)

class TfidfExperimentRunner:
    """Run TF-IDF baseline experiments with full configuration support."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.results = {}
        
        # Validate TF-IDF features directory
        features_dir = config.get('tfidf', {}).get('features_dir', './data/tfidf_features')
        if not Path(features_dir).exists():
            raise FileNotFoundError(
                f"TF-IDF features not found at {features_dir}. "
                f"Please run: python scripts/generate_tfidf_glot500.py --output-dir {features_dir}"
            )
        
        self.features_dir = features_dir
        logger.info(f"Using TF-IDF features from: {features_dir}")
    
    def run_single_experiment(
        self, 
        model_type: str, 
        task: str, 
        languages: List[str],
        control_index: Optional[int] = None,
        submetric: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a single TF-IDF baseline experiment."""
        
        # Determine task type
        task_type = "classification" if task == "question_type" else "regression"
        
        # Create experiment name
        exp_name = f"tfidf_{model_type}_{task}"
        if languages != ['all']:
            exp_name += f"_{'_'.join(languages)}"
        if control_index is not None:
            exp_name += f"_control{control_index}"
        if submetric:
            exp_name += f"_{submetric}"
        
        logger.info(f"Running experiment: {exp_name}")
        
        try:
            # Load data using existing infrastructure
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
                languages=languages,
                task=task,
                submetric=submetric,
                control_index=control_index,
                cache_dir=self.config.get('data',{}).get('cache_dir', os.getenv('VSC_DATA', './data') + '/qtype-eval/data/cache'),
                vectors_dir=self.features_dir,
                use_tfidf_loader=True  # Use new TF-IDF loader
            )
            
            logger.info(f"Loaded data shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
            
            # Get model parameters from config
            model_params = {}
            if hasattr(self.config, 'model_params') and model_type in self.config.model_params:
                if task_type in self.config.model_params[model_type]:
                    model_params = OmegaConf.to_container(
                        self.config.model_params[model_type][task_type], resolve=True
                    )
                else:
                    model_params = OmegaConf.to_container(
                        self.config.model_params[model_type], resolve=True
                    )
            
            # Create model using factory function
            model = create_tfidf_baseline_model(
                model_type=model_type,
                task_type=task_type,
                tfidf_features_dir=self.features_dir,
                target_languages=languages,
                model_params=model_params,
                random_state=self.config.get('seed', 42)
            )
            
            # Create output directory for this experiment
            exp_output_dir = Path(self.config.output_dir) / exp_name
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Train using existing trainer
            trainer = SklearnTrainer(
                model=model.model,  # Pass the underlying sklearn model
                task_type=task_type,
                output_dir=str(exp_output_dir),
                wandb_run=None  # Disable wandb for TF-IDF experiments
            )
            
            results = trainer.train(
                train_data=(X_train, y_train),
                val_data=(X_val, y_val),
                test_data=(X_test, y_test)
            )
            
            # Add experiment metadata
            results.update({
                'experiment_name': exp_name,
                'model_type': model_type,
                'task': task,
                'task_type': task_type,
                'languages': languages,
                'control_index': control_index,
                'submetric': submetric,
                'features_dir': self.features_dir,
                'data_shapes': {
                    'train': list(X_train.shape),
                    'val': list(X_val.shape),
                    'test': list(X_test.shape)
                },
                'model_info': model.get_model_info()
            })
            
            # Save experiment-specific results
            with open(exp_output_dir / "results.json", "w") as f:
                json.dump(OmegaConf.to_container(results, resolve=True), f, indent=2)
            
            logger.info(f"Completed experiment: {exp_name}")
            logger.info(f"Test metrics: {results.get('test_metrics', {})}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed experiment {exp_name}: {e}")
            import traceback
            error_info = {
                'experiment_name': exp_name,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'model_type': model_type,
                'task': task,
                'languages': languages,
                'control_index': control_index,
                'submetric': submetric
            }
            
            # Save error info
            error_dir = Path(self.config.output_dir) / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            with open(error_dir / f"{exp_name}_error.json", "w") as f:
                json.dump(error_info, f, indent=2)
            
            return error_info
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all configured TF-IDF experiments."""
        all_results = {}
        
        # Get experiment configuration
        models = self.config.get('models', ['dummy', 'logistic', 'ridge'])
        tasks = self.config.get('tasks', ['question_type', 'complexity'])
        languages_configs = self.config.get('languages', [['all']])
        
        # Ensure languages_configs is a list of lists
        if isinstance(languages_configs[0], str):
            languages_configs = [languages_configs]
        
        # Run main experiments
        for model_type in models:
            for task in tasks:
                for languages in languages_configs:
                    
                    # Skip logistic regression for regression tasks
                    if model_type == "logistic" and task != "question_type":
                        logger.info(f"Skipping logistic regression for {task} (not applicable)")
                        continue
                    
                    # Regular experiment
                    try:
                        result = self.run_single_experiment(
                            model_type=model_type,
                            task=task,
                            languages=languages
                        )
                        all_results[result['experiment_name']] = result
                    except Exception as e:
                        logger.error(f"Failed experiment {model_type}_{task}: {e}")
                    
                    # Control experiments
                    if self.config.get('controls', {}).get('enabled', False):
                        control_indices = self.config.controls.get('indices', [1, 2, 3])
                        for control_idx in control_indices:
                            try:
                                result = self.run_single_experiment(
                                    model_type=model_type,
                                    task=task,
                                    languages=languages,
                                    control_index=control_idx
                                )
                                all_results[result['experiment_name']] = result
                            except Exception as e:
                                logger.error(f"Failed control experiment: {e}")
        
        # Run submetric experiments if configured
        submetrics = ['avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len', 
                     'avg_verb_edges', 'lexical_density', 'n_tokens']
        
        if 'single_submetric' in tasks or any(sm in tasks for sm in submetrics):
            for model_type in models:
                if model_type == "logistic":  # Skip logistic for regression
                    continue
                    
                for submetric in submetrics:
                    for languages in languages_configs:
                        # Regular submetric experiment
                        try:
                            result = self.run_single_experiment(
                                model_type=model_type,
                                task='single_submetric',
                                languages=languages,
                                submetric=submetric
                            )
                            all_results[result['experiment_name']] = result
                        except Exception as e:
                            logger.error(f"Failed submetric experiment: {e}")
                        
                        # Control submetric experiments
                        if self.config.get('controls', {}).get('enabled', False):
                            control_indices = self.config.controls.get('indices', [1, 2, 3])
                            for control_idx in control_indices:
                                try:
                                    result = self.run_single_experiment(
                                        model_type=model_type,
                                        task='single_submetric',
                                        languages=languages,
                                        control_index=control_idx,
                                        submetric=submetric
                                    )
                                    all_results[result['experiment_name']] = result
                                except Exception as e:
                                    logger.error(f"Failed control submetric experiment: {e}")
        
        self.results = all_results
        return all_results
    
    def save_results(self):
        """Save all experiment results."""
        output_dir = Path(self.config.output_dir)
        
        # Save individual results (already saved in run_single_experiment)
        
        # Save summary
        summary_file = output_dir / "tfidf_experiments_summary.json"
        with open(summary_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Create a results table for easy analysis
        results_table = []
        for exp_name, result in self.results.items():
            if 'error' not in result:
                row = {
                    'experiment': exp_name,
                    'model_type': result.get('model_type'),
                    'task': result.get('task'),
                    'languages': '|'.join(result.get('languages', [])),
                    'control_index': result.get('control_index'),
                    'submetric': result.get('submetric')
                }
                
                # Add test metrics
                test_metrics = result.get('test_metrics', {})
                for metric, value in test_metrics.items():
                    row[f'test_{metric}'] = value
                
                results_table.append(row)
        
        # Save results table as JSON for easy loading
        table_file = output_dir / "results_table.json"
        with open(table_file, "w") as f:
            json.dump(results_table, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Summary: {len(self.results)} experiments, {len(results_table)} successful")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for TF-IDF experiments."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Create a minimal config if experiment config is not properly set
    if not hasattr(cfg, 'experiment') or not hasattr(cfg.experiment, 'experiment_type'):
        logger.info("Creating default TF-IDF experiment configuration...")
        
        # Use the provided overrides to build experiment config
        tfidf_config = OmegaConf.create({
            'experiment_type': 'tfidf_baselines',
            'name': 'tfidf_glot500_baselines',
            'tfidf': {
                'features_dir': './data/tfidf_features',
                'model_name': 'cis-lmu/glot500-base',
                'max_features': 50000
            },
            'models': ['dummy', 'logistic', 'ridge', 'xgboost'],
            'tasks': ['question_type', 'complexity'],
            'languages': [['all'], ['en'], ['ru'], ['ar'], ['fi'], ['id'], ['ja'], ['ko']],
            'controls': {
                'enabled': True,
                'indices': [1, 2, 3]
            },
            'model_params': {
                'dummy': {
                    'classification': {'strategy': 'most_frequent'},
                    'regression': {'strategy': 'mean'}
                },
                'logistic': {
                    'C': 1.0,
                    'solver': 'liblinear',
                    'max_iter': 1000
                },
                'ridge': {'alpha': 1.0},
                'xgboost': {
                    'classification': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1
                    },
                    'regression': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1
                    }
                }
            },
            'output_dir': cfg.get('output_dir', './outputs/tfidf_experiments')
        })
        
        # Apply any command line overrides
        tfidf_config = OmegaConf.merge(tfidf_config, cfg)
        
        # Create experiment runner
        runner = TfidfExperimentRunner(tfidf_config)
    else:
        # Check if we're running TF-IDF experiments
        if (hasattr(cfg.experiment, 'experiment_type') and 
            cfg.experiment.experiment_type == 'tfidf_baselines'):
            runner = TfidfExperimentRunner(cfg.experiment)
        else:
            logger.error("This script requires experiment.experiment_type='tfidf_baselines'")
            logger.error("Please run with: experiment=tfidf_baselines")
            return
    
    # Run all experiments
    logger.info("Starting TF-IDF baseline experiments...")
    results = runner.run_all_experiments()
    
    # Save results
    runner.save_results()
    
    # Print summary
    successful = len([r for r in results.values() if 'error' not in r])
    failed = len(results) - successful
    
    logger.info(f"TF-IDF experiments completed!")
    logger.info(f"Total experiments: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        logger.warning(f"Some experiments failed. Check error logs in {runner.config.output_dir}/errors/")

if __name__ == "__main__":
    main()
