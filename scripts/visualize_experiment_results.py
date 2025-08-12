#!/usr/bin/env python3
"""
Comprehensive visualization script for TF-IDF experiment results.
Creates charts comparing base vs control performance across languages, tasks, and models.
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.experiment_logger import ExperimentLogger

# Configure matplotlib for headless environments
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ExperimentVisualizer:
    """Create comprehensive visualizations for TF-IDF experiment results."""
    
    def __init__(self, experiment_dir: str):
        """
        Initialize visualizer with experiment directory.
        
        Args:
            experiment_dir: Directory containing experiment results
        """
        self.experiment_dir = Path(experiment_dir)
        self.viz_dir = self.experiment_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Load experiment logger
        self.logger = ExperimentLogger(
            base_output_dir=self.experiment_dir.parent,
            experiment_name=self.experiment_dir.name
        )
        
        # Collect all results
        self.all_results = self.logger.collect_all_results()
        self.summaries = self.logger.create_summary_tables()
        
        logger.info(f"Initialized visualizer for {len(self.all_results)} experiments")
    
    def create_all_visualizations(self):
        """Create all visualization charts."""
        
        if not self.all_results:
            logger.error("No experiment results found")
            return
        
        logger.info("Creating comprehensive visualizations...")
        
        # 1. Overall experiment status
        self._plot_experiment_status()
        
        # 2. Main vs Control performance comparison
        self._plot_main_vs_control_comparison()
        
        # 3. Performance by language and task
        self._plot_performance_by_language_task()
        
        # 4. Model comparison across tasks
        self._plot_model_performance_comparison()
        
        # 5. Training time analysis
        self._plot_training_time_analysis()
        
        # 6. Detailed heatmaps
        self._plot_performance_heatmaps()
        
        # 7. Statistical significance analysis
        self._plot_statistical_analysis()
        
        # 8. Error analysis for failed experiments
        self._plot_error_analysis()
        
        logger.info(f"All visualizations saved to {self.viz_dir}")
    
    def _plot_experiment_status(self):
        """Plot overall experiment execution status."""
        
        df = pd.DataFrame(self.all_results)
        
        if df.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Experiment Execution Overview', fontsize=16, fontweight='bold')
        
        # 1. Success/Failure pie chart
        status_counts = df['status'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#f39c12'][:len(status_counts)]
        ax1.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Experiment Success Rate')
        
        # 2. Experiments by task type
        if 'task' in df.columns:
            task_counts = df['task'].value_counts()
            ax2.bar(range(len(task_counts)), task_counts.values, color='skyblue')
            ax2.set_xticks(range(len(task_counts)))
            ax2.set_xticklabels(task_counts.index, rotation=45)
            ax2.set_title('Experiments by Task')
            ax2.set_ylabel('Count')
        
        # 3. Experiments by model type
        if 'model_type' in df.columns:
            model_counts = df['model_type'].value_counts()
            ax3.bar(range(len(model_counts)), model_counts.values, color='lightcoral')
            ax3.set_xticks(range(len(model_counts)))
            ax3.set_xticklabels(model_counts.index, rotation=45)
            ax3.set_title('Experiments by Model Type')
            ax3.set_ylabel('Count')
        
        # 4. Experiments by language
        if 'language' in df.columns:
            lang_counts = df['language'].value_counts()
            ax4.bar(range(len(lang_counts)), lang_counts.values, color='lightgreen')
            ax4.set_xticks(range(len(lang_counts)))
            ax4.set_xticklabels(lang_counts.index)
            ax4.set_title('Experiments by Language')
            ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'experiment_status_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_main_vs_control_comparison(self):
        """Plot main vs control performance comparison."""
        
        if 'main_vs_control' not in self.summaries:
            logger.warning("No main vs control comparison data available")
            return
        
        comparison_df = self.summaries['main_vs_control']
        
        if comparison_df.empty:
            logger.warning("Empty main vs control comparison data")
            return
        
        # Find available metrics
        improvement_cols = [col for col in comparison_df.columns if col.endswith('_improvement')]
        
        if not improvement_cols:
            logger.warning("No improvement metrics found for comparison")
            return
        
        # Create subplots for each metric
        n_metrics = len(improvement_cols)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 6 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle('Main vs Control Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, metric_col in enumerate(improvement_cols):
            ax = axes[i]
            metric_name = metric_col.replace('test_', '').replace('_improvement', '')
            
            # Create comparison plot by language and task
            pivot_data = comparison_df.pivot_table(
                values=metric_col,
                index=['task', 'model_type'],
                columns='language',
                aggfunc='mean'
            )
            
            if not pivot_data.empty:
                # Create heatmap
                sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                           center=0, ax=ax, cbar_kws={'label': f'{metric_name.upper()} Improvement'})
                ax.set_title(f'{metric_name.upper()} Improvement: Main vs Control (Positive = Main Better)')
                ax.set_xlabel('Language')
                ax.set_ylabel('Task + Model')
            else:
                ax.text(0.5, 0.5, f'No data available for {metric_name}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{metric_name.upper()} Improvement (No Data)')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'main_vs_control_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_by_language_task(self):
        """Plot performance breakdown by language and task."""
        
        successful_df = pd.DataFrame([r for r in self.all_results if r['status'] == 'completed'])
        
        if successful_df.empty:
            return
        
        # Get primary metrics for each task type
        classification_metrics = ['test_accuracy', 'test_f1']
        regression_metrics = ['test_mse', 'test_r2']
        
        # Separate by task type
        classification_df = successful_df[successful_df['task_type'] == 'classification']
        regression_df = successful_df[successful_df['task_type'] == 'regression']
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Performance by Language and Task', fontsize=16, fontweight='bold')
        
        # Classification - Accuracy
        if not classification_df.empty and 'test_accuracy' in classification_df.columns:
            self._create_language_task_plot(classification_df, 'test_accuracy', 'Accuracy', axes[0, 0])
        
        # Classification - F1
        if not classification_df.empty and 'test_f1' in classification_df.columns:
            self._create_language_task_plot(classification_df, 'test_f1', 'F1 Score', axes[0, 1])
        
        # Regression - MSE
        if not regression_df.empty and 'test_mse' in regression_df.columns:
            self._create_language_task_plot(regression_df, 'test_mse', 'MSE (lower is better)', axes[1, 0])
        
        # Regression - R²
        if not regression_df.empty and 'test_r2' in regression_df.columns:
            self._create_language_task_plot(regression_df, 'test_r2', 'R² Score', axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'performance_by_language_task.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_language_task_plot(self, df: pd.DataFrame, metric_col: str, metric_name: str, ax):
        """Create individual language-task performance plot."""
        
        if metric_col not in df.columns:
            ax.text(0.5, 0.5, f'{metric_name} data not available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{metric_name} (No Data)')
            return
        
        # Filter out NaN values
        df_clean = df.dropna(subset=[metric_col])
        
        if df_clean.empty:
            ax.text(0.5, 0.5, f'No valid {metric_name} data', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{metric_name} (No Valid Data)')
            return
        
        # Create box plot by language, separated by experiment type
        if 'experiment_type' in df_clean.columns:
            sns.boxplot(data=df_clean, x='language', y=metric_col, hue='experiment_type', ax=ax)
        else:
            sns.boxplot(data=df_clean, x='language', y=metric_col, ax=ax)
        
        ax.set_title(f'{metric_name} by Language')
        ax.tick_params(axis='x', rotation=45)
        
        # Add mean values as text
        if 'experiment_type' in df_clean.columns:
            for exp_type in df_clean['experiment_type'].unique():
                type_df = df_clean[df_clean['experiment_type'] == exp_type]
                means = type_df.groupby('language')[metric_col].mean()
                for i, (lang, mean_val) in enumerate(means.items()):
                    ax.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_model_performance_comparison(self):
        """Compare model performance across tasks."""
        
        successful_df = pd.DataFrame([r for r in self.all_results if r['status'] == 'completed'])
        
        if successful_df.empty:
            return
        
        # Group by model type and task
        model_task_summary = successful_df.groupby(['model_type', 'task', 'experiment_type']).agg({
            'test_accuracy': 'mean',
            'test_f1': 'mean',
            'test_mse': 'mean',
            'test_r2': 'mean',
            'train_time': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison Across Tasks', fontsize=16, fontweight='bold')
        
        # Accuracy comparison
        if 'test_accuracy' in model_task_summary.columns:
            acc_data = model_task_summary.dropna(subset=['test_accuracy'])
            if not acc_data.empty:
                pivot_acc = acc_data.pivot_table(values='test_accuracy', index='model_type', 
                                               columns=['task', 'experiment_type'], aggfunc='mean')
                sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 0])
                axes[0, 0].set_title('Test Accuracy by Model and Task')
        
        # MSE comparison
        if 'test_mse' in model_task_summary.columns:
            mse_data = model_task_summary.dropna(subset=['test_mse'])
            if not mse_data.empty:
                pivot_mse = mse_data.pivot_table(values='test_mse', index='model_type', 
                                               columns=['task', 'experiment_type'], aggfunc='mean')
                sns.heatmap(pivot_mse, annot=True, fmt='.4f', cmap='viridis_r', ax=axes[0, 1])
                axes[0, 1].set_title('Test MSE by Model and Task (Lower is Better)')
        
        # Training time comparison
        if 'train_time' in model_task_summary.columns:
            time_data = model_task_summary.dropna(subset=['train_time'])
            if not time_data.empty:
                sns.barplot(data=time_data, x='model_type', y='train_time', hue='experiment_type', ax=axes[1, 0])
                axes[1, 0].set_title('Training Time by Model Type')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model usage frequency
        model_counts = successful_df['model_type'].value_counts()
        axes[1, 1].bar(range(len(model_counts)), model_counts.values, color='lightblue')
        axes[1, 1].set_xticks(range(len(model_counts)))
        axes[1, 1].set_xticklabels(model_counts.index, rotation=45)
        axes[1, 1].set_title('Model Usage Frequency')
        axes[1, 1].set_ylabel('Number of Experiments')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_time_analysis(self):
        """Analyze training time patterns."""
        
        successful_df = pd.DataFrame([r for r in self.all_results if r['status'] == 'completed'])
        
        if successful_df.empty or 'train_time' not in successful_df.columns:
            return
        
        # Filter valid training times
        time_df = successful_df.dropna(subset=['train_time'])
        time_df = time_df[time_df['train_time'] > 0]
        
        if time_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Time Analysis', fontsize=16, fontweight='bold')
        
        # Training time by model type
        if 'model_type' in time_df.columns:
            sns.boxplot(data=time_df, x='model_type', y='train_time', ax=axes[0, 0])
            axes[0, 0].set_title('Training Time by Model Type')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_ylabel('Training Time (seconds)')
        
        # Training time by language
        if 'language' in time_df.columns:
            sns.boxplot(data=time_df, x='language', y='train_time', ax=axes[0, 1])
            axes[0, 1].set_title('Training Time by Language')
            axes[0, 1].set_ylabel('Training Time (seconds)')
        
        # Training time vs performance (accuracy)
        if 'test_accuracy' in time_df.columns:
            acc_df = time_df.dropna(subset=['test_accuracy'])
            if not acc_df.empty:
                axes[1, 0].scatter(acc_df['train_time'], acc_df['test_accuracy'], 
                                 c=acc_df['model_type'].astype('category').cat.codes, alpha=0.7)
                axes[1, 0].set_xlabel('Training Time (seconds)')
                axes[1, 0].set_ylabel('Test Accuracy')
                axes[1, 0].set_title('Training Time vs Accuracy')
        
        # Training time distribution
        axes[1, 1].hist(time_df['train_time'], bins=30, color='skyblue', alpha=0.7)
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Training Time Distribution')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'training_time_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_heatmaps(self):
        """Create detailed performance heatmaps."""
        
        successful_df = pd.DataFrame([r for r in self.all_results if r['status'] == 'completed'])
        
        if successful_df.empty:
            return
        
        # Create comprehensive heatmap for main metrics
        metrics_to_plot = []
        
        if 'test_accuracy' in successful_df.columns:
            metrics_to_plot.append(('test_accuracy', 'Test Accuracy', 'viridis'))
        if 'test_f1' in successful_df.columns:
            metrics_to_plot.append(('test_f1', 'Test F1 Score', 'viridis'))
        if 'test_mse' in successful_df.columns:
            metrics_to_plot.append(('test_mse', 'Test MSE', 'viridis_r'))
        if 'test_r2' in successful_df.columns:
            metrics_to_plot.append(('test_r2', 'Test R² Score', 'viridis'))
        
        if not metrics_to_plot:
            return
        
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 6 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle('Detailed Performance Heatmaps', fontsize=16, fontweight='bold')
        
        for i, (metric_col, title, cmap) in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Create pivot table
            pivot_data = successful_df.pivot_table(
                values=metric_col,
                index=['task', 'experiment_type'],
                columns=['language', 'model_type'],
                aggfunc='mean'
            )
            
            if not pivot_data.empty:
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap=cmap, ax=ax,
                           cbar_kws={'label': title})
                ax.set_title(f'{title} Heatmap')
                ax.set_xlabel('Language + Model Type')
                ax.set_ylabel('Task + Experiment Type')
            else:
                ax.text(0.5, 0.5, f'No data for {title}', transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{title} (No Data)')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'performance_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_analysis(self):
        """Create statistical analysis plots."""
        
        if 'main_vs_control' not in self.summaries:
            return
        
        comparison_df = self.summaries['main_vs_control']
        
        if comparison_df.empty:
            return
        
        # Statistical significance testing
        from scipy import stats
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Statistical Analysis: Main vs Control', fontsize=16, fontweight='bold')
        
        # Distribution of improvements
        improvement_cols = [col for col in comparison_df.columns if col.endswith('_improvement')]
        
        for i, metric_col in enumerate(improvement_cols[:4]):  # Limit to 4 plots
            ax = axes[i // 2, i % 2]
            
            improvements = comparison_df[metric_col].dropna()
            
            if len(improvements) > 0:
                # Histogram of improvements
                ax.hist(improvements, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(0, color='red', linestyle='--', label='No Improvement')
                ax.axvline(improvements.mean(), color='green', linestyle='-', label=f'Mean: {improvements.mean():.4f}')
                
                # Add statistical info
                ax.set_xlabel('Improvement (Main - Control)')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{metric_col.replace("test_", "").replace("_improvement", "").upper()} Improvement Distribution')
                ax.legend()
                
                # Add text with statistics
                n_positive = (improvements > 0).sum()
                n_negative = (improvements < 0).sum()
                ax.text(0.02, 0.98, f'Positive: {n_positive}\nNegative: {n_negative}\nMean: {improvements.mean():.4f}\nStd: {improvements.std():.4f}',
                       transform=ax.transAxes, va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_analysis(self):
        """Analyze failed experiments."""
        
        failed_df = pd.DataFrame([r for r in self.all_results if r['status'] == 'failed'])
        
        if failed_df.empty:
            logger.info("No failed experiments to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Failed Experiment Analysis', fontsize=16, fontweight='bold')
        
        # Failures by task
        if 'task' in failed_df.columns:
            task_failures = failed_df['task'].value_counts()
            axes[0, 0].bar(range(len(task_failures)), task_failures.values, color='red', alpha=0.7)
            axes[0, 0].set_xticks(range(len(task_failures)))
            axes[0, 0].set_xticklabels(task_failures.index, rotation=45)
            axes[0, 0].set_title('Failures by Task')
            axes[0, 0].set_ylabel('Number of Failures')
        
        # Failures by model
        if 'model_type' in failed_df.columns:
            model_failures = failed_df['model_type'].value_counts()
            axes[0, 1].bar(range(len(model_failures)), model_failures.values, color='red', alpha=0.7)
            axes[0, 1].set_xticks(range(len(model_failures)))
            axes[0, 1].set_xticklabels(model_failures.index, rotation=45)
            axes[0, 1].set_title('Failures by Model')
            axes[0, 1].set_ylabel('Number of Failures')
        
        # Failures by language
        if 'language' in failed_df.columns:
            lang_failures = failed_df['language'].value_counts()
            axes[1, 0].bar(range(len(lang_failures)), lang_failures.values, color='red', alpha=0.7)
            axes[1, 0].set_xticks(range(len(lang_failures)))
            axes[1, 0].set_xticklabels(lang_failures.index)
            axes[1, 0].set_title('Failures by Language')
            axes[1, 0].set_ylabel('Number of Failures')
        
        # Error type analysis
        if 'error' in failed_df.columns:
            # Extract common error patterns
            error_patterns = {}
            for error in failed_df['error'].dropna():
                error_str = str(error).lower()
                if 'memory' in error_str or 'out of memory' in error_str:
                    error_patterns['Memory'] = error_patterns.get('Memory', 0) + 1
                elif 'timeout' in error_str or 'time' in error_str:
                    error_patterns['Timeout'] = error_patterns.get('Timeout', 0) + 1
                elif 'shape' in error_str or 'dimension' in error_str:
                    error_patterns['Shape/Dimension'] = error_patterns.get('Shape/Dimension', 0) + 1
                elif 'import' in error_str or 'module' in error_str:
                    error_patterns['Import/Module'] = error_patterns.get('Import/Module', 0) + 1
                else:
                    error_patterns['Other'] = error_patterns.get('Other', 0) + 1
            
            if error_patterns:
                axes[1, 1].pie(error_patterns.values(), labels=error_patterns.keys(), autopct='%1.1f%%')
                axes[1, 1].set_title('Error Type Distribution')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed error log
        if 'error' in failed_df.columns:
            error_log_file = self.viz_dir / 'detailed_error_log.txt'
            with open(error_log_file, 'w') as f:
                f.write("DETAILED ERROR LOG\n")
                f.write("=" * 50 + "\n\n")
                
                for i, row in failed_df.iterrows():
                    f.write(f"Experiment ID: {row.get('experiment_id', 'unknown')}\n")
                    f.write(f"Task: {row.get('task', 'unknown')}\n")
                    f.write(f"Model: {row.get('model_type', 'unknown')}\n")
                    f.write(f"Language: {row.get('language', 'unknown')}\n")
                    f.write(f"Error: {row.get('error', 'No error message')}\n")
                    f.write("-" * 30 + "\n\n")
            
            logger.info(f"Detailed error log saved to {error_log_file}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report."""
        
        report_file = self.viz_dir / 'experiment_summary_report.md'
        
        with open(report_file, 'w') as f:
            f.write("# TF-IDF Experiment Results Summary\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            total_experiments = len(self.all_results)
            successful_experiments = len([r for r in self.all_results if r['status'] == 'completed'])
            failed_experiments = total_experiments - successful_experiments
            
            f.write("## Experiment Overview\n\n")
            f.write(f"- **Total Experiments**: {total_experiments}\n")
            f.write(f"- **Successful**: {successful_experiments} ({100*successful_experiments/max(1,total_experiments):.1f}%)\n")
            f.write(f"- **Failed**: {failed_experiments} ({100*failed_experiments/max(1,total_experiments):.1f}%)\n\n")
            
            # Main vs Control summary
            if 'main_vs_control' in self.summaries and not self.summaries['main_vs_control'].empty:
                f.write("## Main vs Control Performance\n\n")
                comparison_df = self.summaries['main_vs_control']
                
                # Best improvements
                improvement_cols = [col for col in comparison_df.columns if col.endswith('_improvement')]
                for metric_col in improvement_cols:
                    metric_name = metric_col.replace('test_', '').replace('_improvement', '').upper()
                    improvements = comparison_df[metric_col].dropna()
                    
                    if len(improvements) > 0:
                        best_improvement = improvements.max()
                        worst_improvement = improvements.min()
                        mean_improvement = improvements.mean()
                        
                        f.write(f"### {metric_name}\n")
                        f.write(f"- **Best Improvement**: {best_improvement:.4f}\n")
                        f.write(f"- **Worst Improvement**: {worst_improvement:.4f}\n")
                        f.write(f"- **Average Improvement**: {mean_improvement:.4f}\n")
                        f.write(f"- **Positive Improvements**: {(improvements > 0).sum()}/{len(improvements)}\n\n")
            
            # Performance by task
            successful_df = pd.DataFrame([r for r in self.all_results if r['status'] == 'completed'])
            if not successful_df.empty:
                f.write("## Performance by Task\n\n")
                
                for task in successful_df['task'].unique():
                    if pd.isna(task):
                        continue
                    
                    task_df = successful_df[successful_df['task'] == task]
                    f.write(f"### {task}\n")
                    f.write(f"- **Experiments**: {len(task_df)}\n")
                    
                    # Key metrics
                    if 'test_accuracy' in task_df.columns:
                        acc_values = task_df['test_accuracy'].dropna()
                        if len(acc_values) > 0:
                            f.write(f"- **Accuracy**: {acc_values.mean():.3f} ± {acc_values.std():.3f} (mean ± std)\n")
                            f.write(f"- **Best Accuracy**: {acc_values.max():.3f}\n")
                    
                    if 'test_mse' in task_df.columns:
                        mse_values = task_df['test_mse'].dropna()
                        if len(mse_values) > 0:
                            f.write(f"- **MSE**: {mse_values.mean():.4f} ± {mse_values.std():.4f} (mean ± std)\n")
                            f.write(f"- **Best MSE**: {mse_values.min():.4f}\n")
                    
                    f.write("\n")
            
            f.write("## Visualizations Generated\n\n")
            viz_files = list(self.viz_dir.glob("*.png"))
            for viz_file in sorted(viz_files):
                f.write(f"- `{viz_file.name}`\n")
        
        logger.info(f"Summary report saved to {report_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize TF-IDF experiment results")
    parser.add_argument("experiment_dir", help="Directory containing experiment results")
    parser.add_argument("--output-format", choices=["png", "pdf", "svg"], default="png",
                       help="Output format for plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")
    
    args = parser.parse_args()
    
    # Validate experiment directory
    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return 1
    
    try:
        # Create visualizer
        visualizer = ExperimentVisualizer(str(experiment_dir))
        
        # Set output format
        if args.output_format != "png":
            plt.rcParams['savefig.format'] = args.output_format
        plt.rcParams['savefig.dpi'] = args.dpi
        
        # Create all visualizations
        visualizer.create_all_visualizations()
        
        # Create summary report
        visualizer.create_summary_report()
        
        logger.info("="*60)
        logger.info("VISUALIZATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Visualizations saved to: {visualizer.viz_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())