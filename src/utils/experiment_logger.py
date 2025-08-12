"""
Comprehensive experiment logging and result collection system.
Provides standardized format for tracking TF-IDF experiments with base vs control comparisons.
"""

import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """Centralized experiment logging with standardized format and collection."""
    
    def __init__(self, base_output_dir: str, experiment_name: str = None):
        """
        Initialize experiment logger.
        
        Args:
            base_output_dir: Base directory for all experiment results
            experiment_name: Optional experiment batch name (auto-generated if None)
        """
        self.base_output_dir = Path(base_output_dir)
        self.experiment_name = experiment_name or f"tfidf_experiment_{int(time.time())}"
        
        # Create experiment directory structure
        self.experiment_dir = self.base_output_dir / self.experiment_name
        self.results_dir = self.experiment_dir / "results"
        self.logs_dir = self.experiment_dir / "logs" 
        self.summaries_dir = self.experiment_dir / "summaries"
        self.viz_dir = self.experiment_dir / "visualizations"
        
        # Create all directories
        for dir_path in [self.experiment_dir, self.results_dir, self.logs_dir, 
                        self.summaries_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment metadata
        self.metadata = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "base_output_dir": str(self.base_output_dir),
            "experiment_dir": str(self.experiment_dir),
            "completed_experiments": [],
            "failed_experiments": [],
            "total_experiments": 0,
            "schema_version": "1.0"
        }
        
        self._save_metadata()
        logger.info(f"Initialized experiment logger: {self.experiment_name}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def log_experiment_result(self, 
                             experiment_config: Dict[str, Any],
                             results: Dict[str, Any],
                             error: Optional[str] = None) -> str:
        """
        Log a single experiment result with standardized format.
        
        Args:
            experiment_config: Configuration used for the experiment
            results: Training/evaluation results
            error: Error message if experiment failed
            
        Returns:
            Unique experiment ID
        """
        # Generate unique experiment ID
        config_str = json.dumps(experiment_config, sort_keys=True)
        exp_id = hashlib.md5(config_str.encode()).hexdigest()[:12]
        
        # Create standardized result entry
        result_entry = {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "config": experiment_config,
            "status": "failed" if error else "completed",
            "error": error,
            **results  # Merge in the actual results
        }
        
        # Save individual result file
        result_file = self.results_dir / f"{exp_id}.json"
        with open(result_file, 'w') as f:
            json.dump(result_entry, f, indent=2, default=self._json_serialize)
        
        # Update metadata
        if error:
            self.metadata["failed_experiments"].append(exp_id)
        else:
            self.metadata["completed_experiments"].append(exp_id)
        
        self.metadata["total_experiments"] += 1
        self._save_metadata()
        
        logger.info(f"Logged experiment {exp_id}: {'FAILED' if error else 'SUCCESS'}")
        return exp_id
    
    def log_experiment_start(self, experiment_config: Dict[str, Any]) -> str:
        """
        Log the start of an experiment and return tracking ID.
        
        Args:
            experiment_config: Configuration for the experiment
            
        Returns:
            Unique experiment ID for tracking
        """
        config_str = json.dumps(experiment_config, sort_keys=True)
        exp_id = hashlib.md5(config_str.encode()).hexdigest()[:12]
        
        start_entry = {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "status": "started",
            "config": experiment_config
        }
        
        # Save start log
        start_file = self.logs_dir / f"start_{exp_id}.json"
        with open(start_file, 'w') as f:
            json.dump(start_entry, f, indent=2)
        
        logger.info(f"Started experiment {exp_id}: {experiment_config.get('experiment_name', 'unnamed')}")
        return exp_id
    
    def collect_all_results(self) -> List[Dict[str, Any]]:
        """
        Collect and standardize all experiment results.
        
        Returns:
            List of all experiment results in standardized format
        """
        all_results = []
        
        # Load all result files
        for result_file in self.results_dir.glob("*.json"):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                # Standardize the result format
                standardized_result = self._standardize_result_format(result)
                all_results.append(standardized_result)
                
            except Exception as e:
                logger.error(f"Error loading result file {result_file}: {e}")
        
        # Sort by timestamp
        all_results.sort(key=lambda x: x.get('timestamp', ''))
        
        logger.info(f"Collected {len(all_results)} experiment results")
        return all_results
    
    def create_summary_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Create comprehensive summary tables for analysis.
        
        Returns:
            Dictionary of summary DataFrames
        """
        all_results = self.collect_all_results()
        
        if not all_results:
            logger.warning("No results found to summarize")
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_results)
        
        # Separate successful and failed experiments
        successful_df = df[df['status'] == 'completed'].copy()
        failed_df = df[df['status'] == 'failed'].copy()
        
        summaries = {
            'all_results': df,
            'successful_results': successful_df,
            'failed_results': failed_df
        }
        
        if len(successful_df) > 0:
            # Create main vs control comparison
            main_results = successful_df[successful_df['experiment_type'] == 'main'].copy()
            control_results = successful_df[successful_df['experiment_type'] == 'control'].copy()
            
            if len(main_results) > 0 and len(control_results) > 0:
                comparison_df = self._create_main_vs_control_comparison(main_results, control_results)
                summaries['main_vs_control'] = comparison_df
            
            # Create performance summaries by task and language
            summaries['by_task_language'] = self._create_task_language_summary(successful_df)
            summaries['by_model_type'] = self._create_model_type_summary(successful_df)
        
        # Save all summaries
        for name, summary_df in summaries.items():
            if isinstance(summary_df, pd.DataFrame) and len(summary_df) > 0:
                summary_file = self.summaries_dir / f"{name}_summary.csv"
                summary_df.to_csv(summary_file, index=False)
                logger.info(f"Saved {name} summary: {len(summary_df)} entries")
        
        return summaries
    
    def _standardize_result_format(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize result format across different experiment types."""
        
        # Extract core information
        standardized = {
            "experiment_id": result.get("experiment_id", "unknown"),
            "timestamp": result.get("timestamp", ""),
            "status": result.get("status", "unknown"),
            "error": result.get("error"),
            
            # Experiment configuration
            "language": result.get("language") or result.get("config", {}).get("language"),
            "task": result.get("task") or result.get("config", {}).get("task"),
            "model_type": result.get("model") or result.get("model_type") or result.get("config", {}).get("model_type"),
            "task_type": result.get("task_type") or result.get("config", {}).get("task_type"),
            "dataset_config": result.get("config", {}).get("config") or result.get("config", {}).get("dataset_config"),
            
            # Determine experiment type (main vs control)
            "experiment_type": self._determine_experiment_type(result),
            "control_seed": self._extract_control_seed(result),
            
            # Training information
            "train_time": result.get("train_time", 0),
            "model_class": result.get("model_type") or result.get("model_class"),
            
            # Metrics (flatten for easier analysis)
            **self._flatten_metrics(result.get("train_metrics", {}), "train"),
            **self._flatten_metrics(result.get("val_metrics", {}), "val"),
            **self._flatten_metrics(result.get("test_metrics", {}), "test"),
            
            # Data shapes
            "train_samples": self._extract_sample_count(result, "train"),
            "val_samples": self._extract_sample_count(result, "val"),
            "test_samples": self._extract_sample_count(result, "test"),
            
            # Raw result for reference
            "_raw_result": result
        }
        
        return standardized
    
    def _determine_experiment_type(self, result: Dict[str, Any]) -> str:
        """Determine if experiment is main or control based on configuration."""
        config_str = str(result.get("config", {})).lower()
        dataset_config = result.get("config", {}).get("config", "").lower()
        
        if "control" in config_str or "control" in dataset_config:
            return "control"
        else:
            return "main"
    
    def _extract_control_seed(self, result: Dict[str, Any]) -> Optional[int]:
        """Extract control seed number if applicable."""
        config_str = str(result.get("config", {}))
        dataset_config = result.get("config", {}).get("config", "")
        
        # Look for seed patterns
        import re
        seed_match = re.search(r'seed(\d+)', config_str + dataset_config, re.IGNORECASE)
        if seed_match:
            return int(seed_match.group(1))
        return None
    
    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str) -> Dict[str, float]:
        """Flatten metrics dictionary with prefix."""
        flattened = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                flattened[f"{prefix}_{key}"] = float(value)
        return flattened
    
    def _extract_sample_count(self, result: Dict[str, Any], split: str) -> Optional[int]:
        """Extract sample count for a data split."""
        # Try various places where sample counts might be stored
        data_shapes = result.get("data_shapes", {})
        if split in data_shapes:
            shape = data_shapes[split]
            if isinstance(shape, list) and len(shape) > 0:
                return int(shape[0])
        
        # Try other locations
        split_info = result.get(f"{split}_info", {})
        if "samples" in split_info:
            return int(split_info["samples"])
        
        return None
    
    def _create_main_vs_control_comparison(self, main_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.DataFrame:
        """Create main vs control comparison table."""
        
        # Average control results by task, model, language (across seeds)
        control_groupby_cols = ['task', 'model_type', 'language', 'task_type']
        control_avg = control_df.groupby(control_groupby_cols).agg({
            col: 'mean' for col in control_df.columns 
            if col.startswith(('test_', 'val_', 'train_')) and pd.api.types.is_numeric_dtype(control_df[col])
        }).reset_index()
        
        # Prepare main results for merging
        main_subset = main_df[control_groupby_cols + [col for col in main_df.columns 
                                                    if col.startswith(('test_', 'val_', 'train_'))]].copy()
        
        # Merge main and control results
        comparison_df = main_subset.merge(
            control_avg, 
            on=control_groupby_cols,
            how='inner',
            suffixes=('_main', '_control')
        )
        
        # Calculate differences for key metrics
        for metric in ['accuracy', 'f1', 'mse', 'r2']:
            test_main_col = f'test_{metric}_main'
            test_control_col = f'test_{metric}_control'
            
            if test_main_col in comparison_df.columns and test_control_col in comparison_df.columns:
                comparison_df[f'test_{metric}_diff'] = comparison_df[test_main_col] - comparison_df[test_control_col]
                
                # For MSE, lower is better, so positive diff means main is worse
                if metric == 'mse':
                    comparison_df[f'test_{metric}_improvement'] = comparison_df[test_control_col] - comparison_df[test_main_col]
                else:
                    comparison_df[f'test_{metric}_improvement'] = comparison_df[f'test_{metric}_diff']
        
        return comparison_df
    
    def _create_task_language_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary grouped by task and language."""
        
        groupby_cols = ['task', 'language', 'experiment_type']
        
        # Get numeric columns for aggregation
        numeric_cols = [col for col in df.columns 
                       if col.startswith('test_') and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols:
            return pd.DataFrame()
        
        summary = df.groupby(groupby_cols)[numeric_cols].agg(['mean', 'std', 'count']).reset_index()
        
        # Flatten column names
        summary.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                          for col in summary.columns]
        
        return summary
    
    def _create_model_type_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary grouped by model type."""
        
        groupby_cols = ['model_type', 'task_type', 'experiment_type']
        
        # Get numeric columns for aggregation
        numeric_cols = [col for col in df.columns 
                       if col.startswith('test_') and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_cols:
            return pd.DataFrame()
        
        summary = df.groupby(groupby_cols)[numeric_cols].agg(['mean', 'std', 'count']).reset_index()
        
        # Flatten column names
        summary.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] 
                          for col in summary.columns]
        
        return summary
    
    def _save_metadata(self):
        """Save experiment metadata."""
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=self._json_serialize)
    
    def _json_serialize(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return str(obj)
    
    def finalize_experiment(self):
        """Finalize experiment logging and create final summaries."""
        self.metadata["end_time"] = datetime.now().isoformat()
        self.metadata["duration_hours"] = (
            datetime.fromisoformat(self.metadata["end_time"]) - 
            datetime.fromisoformat(self.metadata["start_time"])
        ).total_seconds() / 3600
        
        # Create final summaries
        summaries = self.create_summary_tables()
        
        # Save final metadata
        self._save_metadata()
        
        logger.info(f"Experiment {self.experiment_name} finalized")
        logger.info(f"Total experiments: {self.metadata['total_experiments']}")
        logger.info(f"Successful: {len(self.metadata['completed_experiments'])}")
        logger.info(f"Failed: {len(self.metadata['failed_experiments'])}")
        
        return summaries
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get current experiment status."""
        return {
            "experiment_name": self.experiment_name,
            "total_experiments": self.metadata["total_experiments"],
            "completed": len(self.metadata["completed_experiments"]),
            "failed": len(self.metadata["failed_experiments"]),
            "success_rate": len(self.metadata["completed_experiments"]) / max(1, self.metadata["total_experiments"]),
            "experiment_dir": str(self.experiment_dir)
        }