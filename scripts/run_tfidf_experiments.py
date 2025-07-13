# scripts/run_tfidf_experiments.py
"""


"""
import os
import sys
import json
import hydra
from pathlib import Path
from typing import Dict, List, Any
from omegaconf import DictConfig
import logging

sys.path.append(str(Path(__file__).parent.parent / "src")) # dirtyy

from data.tfidf_features import TfidfFeatureLoader
from ..models.tfidf_baselines.py import create_tfidf_baseline_model
from datasets import load_dataset

logger = logging.getLogger(__name__)

class TfidfExperimentRunner:    
    def __init__(self, config: DictConfig):
        self.config = config
        self.tfidf_loader = None
        self.results = {}
    
    def setup_tfidf_loader(self):
        features_dir = self.config.tfidf.features_dir
        if not Path(features_dir).exists():
            raise FileNotFoundError(
                f"TF-IDF features not found at {features_dir}. "
                f"Please run: python scripts/generate_tfidf_glot500.py --output-dir {features_dir}"
            )
        
        self.tfidf_loader = TfidfFeatureLoader(features_dir)
        logger.info(f"Loaded TF-IDF features from {features_dir}")
        logger.info(f"Vocabulary size: {self.tfidf_loader.get_vocab_size()}")
    
    def load_labels(self, task: str, control_index: Optional[int] = None, submetric: Optional[str] = None):
        """Load labels for the specified task."""
        # Determine dataset config
        if control_index is None:
            config_name = "base"
        elif task == "question_type":
            config_name = f"control_question_type_seed{control_index}"
        elif task == "complexity":
            if submetric:
                config_name = f"control_{submetric}_seed{control_index}"
            else:
                config_name = f"control_complexity_seed{control_index}"
        else:
            config_name = "base"
        
        # Load dataset
        dataset = load_dataset("rokokot/question-type-and-complexity", config_name)
        
        train_df = dataset['train'].to_pandas()
        val_df = dataset['validation'].to_pandas() 
        test_df = dataset['test'].to_pandas()
        
        # Extract labels
        if task == "question_type":
            label_col = "question_type"
        elif task == "complexity":
            if submetric:
                label_col = submetric
            else:
                label_col = "lang_norm_complexity_score"
        
        y_train = train_df[label_col].values
        y_val = val_df[label_col].values
        y_test = test_df[label_col].values
        
        # Get language info for filtering
        languages_train = train_df['language'].tolist()
        languages_val = val_df['language'].tolist()
        languages_test = test_df['language'].tolist()
        
        return (y_train, y_val, y_test), (languages_train, languages_val, languages_test)
    
    def filter_labels_by_language(self, labels, languages, target_languages):
        """Filter labels to match language-filtered features."""
        if target_languages == ['all']:
            return labels
        
        y_train, y_val, y_test = labels
        lang_train, lang_val, lang_test = languages
        
        # Filter training data
        train_indices = [i for i, lang in enumerate(lang_train) if lang in target_languages]
        y_train_filtered = y_train[train_indices]
        
        # Filter validation data  
        val_indices = [i for i, lang in enumerate(lang_val) if lang in target_languages]
        y_val_filtered = y_val[val_indices]
        
        # Filter test data
        test_indices = [i for i, lang in enumerate(lang_test) if lang in target_languages]
        y_test_filtered = y_test[test_indices]
        
        return y_train_filtered, y_val_filtered, y_test_filtered
    
    def run_single_experiment(
        self, 
        model_type: str, 
        task: str, 
        target_languages: List[str] = ['all'],
        control_index: Optional[int] = None,
        submetric: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a single TF-IDF experiment."""
        
        # Determine task type
        task_type = "classification" if task == "question_type" else "regression"
        
        # Load labels
        labels, language_info = self.load_labels(task, control_index, submetric)
        
        # Filter labels by language
        y_train, y_val, y_test = self.filter_labels_by_language(
            labels, language_info, target_languages
        )
        
        # Get model parameters
        model_params = {}
        if model_type in self.config.model_params:
            if task_type in self.config.model_params[model_type]:
                model_params = self.config.model_params[model_type][task_type]
            else:
                model_params = self.config.model_params[model_type]
        
        # Create model
        model = create_tfidf_baseline_model(
            model_type=model_type,
            task_type=task_type,
            tfidf_features_dir=self.config.tfidf.features_dir,
            target_languages=target_languages,
            model_params=model_params
        )
        
        # Train model
        logger.info(f"Training {model_type} for {task} ({'|'.join(target_languages)})")
        model.fit(y_train)
        
        # Evaluate
        train_metrics = model.evaluate(y_train, 'train')
        val_metrics = model.evaluate(y_val, 'val') 
        test_metrics = model.evaluate(y_test, 'test')
        
        # Prepare results
        exp_name = f"{model_type}_{task}"
        if target_languages != ['all']:
            exp_name += f"_{'_'.join(target_languages)}"
        if control_index is not None:
            exp_name += f"_control{control_index}"
        if submetric:
            exp_name += f"_{submetric}"
        
        result = {
            'experiment_name': exp_name,
            'model_type': model_type,
            'task': task,
            'task_type': task_type,
            'languages': target_languages,
            'control_index': control_index,
            'submetric': submetric,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'data_shapes': {
                'train': len(y_train),
                'val': len(y_val), 
                'test': len(y_test)
            }
        }
        
        logger.info(f"Completed {exp_name}")
        logger.info(f"Test metrics: {test_metrics}")
        
        return result
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """Run all configured TF-IDF experiments."""
        self.setup_tfidf_loader()
        
        all_results = {}
        
        # Generate experiment combinations
        for model_type in self.config.models:
            for task in self.config.tasks:
                for languages in self.config.languages:
                    target_languages = [languages] if isinstance(languages, str) else languages
                    
                    # Regular experiment
                    try:
                        result = self.run_single_experiment(
                            model_type, task, target_languages
                        )
                        all_results[result['experiment_name']] = result
                    except Exception as e:
                        logger.error(f"Failed experiment {model_type}_{task}: {e}")
                    
                    # Control experiments
                    if self.config.controls.enabled:
                        for control_idx in self.config.controls.indices:
                            try:
                                result = self.run_single_experiment(
                                    model_type, task, target_languages, control_idx
                                )
                                all_results[result['experiment_name']] = result
                            except Exception as e:
                                logger.error(f"Failed control experiment: {e}")
        
        self.results = all_results
        return all_results
    
    def save_results(self, output_dir: str):
        """Save experiment results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        for exp_name, result in self.results.items():
            with open(output_path / f"{exp_name}.json", "w") as f:
                json.dump(result, f, indent=2)
        
        # Save summary
        with open(output_path / "tfidf_experiments_summary.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for TF-IDF experiments."""
    
    # Override config to use TF-IDF experiment settings
    if hasattr(cfg, 'experiment') and cfg.experiment._target_ == 'tfidf_baselines':
        runner = TfidfExperimentRunner(cfg)
        results = runner.run_all_experiments()
        
        output_dir = cfg.get('output_dir', './outputs/tfidf_experiments')
        runner.save_results(output_dir)
        
        logger.info(f"TF-IDF experiments completed! {len(results)} experiments run.")
    else:
        logger.error("Please use tfidf_baselines experiment config")

if __name__ == "__main__":
    main()