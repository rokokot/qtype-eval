#!/usr/bin/env python3
"""
Simplified analysis script for multilingual question probing experiments.

This script:
1. Finds and parses all experiment result files (JSON)
2. Generates key analyses and visualizations
3. Creates a single comprehensive report

Usage:
  python analyze_experiments.py --results-dirs dir1 [dir2 ...] --output-dir output_dir
"""

import os
import sys
import argparse
import json
import glob
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("analysis.log")
    ]
)
logger = logging.getLogger(__name__)

# Define custom color palettes
MODEL_PALETTE = {
    "dummy": "#E0E0E0",  # Light gray
    "logistic": "#4285F4",  # Google blue
    "ridge": "#FBBC05",  # Google yellow
    "xgboost": "#34A853",  # Google green
    "lm_probe": "#EA4335"  # Google red
}

LANGUAGE_PALETTE = {
    "ar": "#E6194B",  # Red
    "en": "#3CB44B",  # Green
    "fi": "#FFE119",  # Yellow
    "id": "#4363D8",  # Blue
    "ja": "#F58231",  # Orange
    "ko": "#911EB4",  # Purple
    "ru": "#42D4F4"   # Cyan
}

SUBMETRIC_PALETTE = {
    "avg_links_len": "#FF9999",  # Light red
    "avg_max_depth": "#66B2FF",  # Light blue
    "avg_subordinate_chain_len": "#99FF99",  # Light green
    "avg_verb_edges": "#FFCC99",  # Light orange
    "lexical_density": "#CC99FF",  # Light purple
    "n_tokens": "#FFD700"  # Gold
}

class ExperimentAnalyzer:
    def __init__(self, results_dirs: List[str], output_dir: str):
        self.results_dirs = results_dirs
        self.output_dir = output_dir
        self.results_df = None
        self.task_summaries = {}
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        
        self.dpi = 150
        self.figsize_standard = (10, 6)
        self.figsize_large = (12, 8)
        self.figsize_square = (8, 8)
        
        sns.set_theme(style="whitegrid")
        
    def find_result_files(self) -> List[str]:
        all_files = []
        for base_dir in self.results_dirs:
            patterns = [
                "**/*_all.json",
                "**/*_control*_all.json",
                "**/results.json",
                "**/all_results.json",
                "**/cross_lingual_results.json"]
            
            for pattern in patterns:
                files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
                all_files.extend(files)
                
        logger.info(f"Found {len(all_files)} result files")
        return all_files
    
    def extract_metadata_from_path(self, file_path: str) -> Dict[str, Any]:
        parts = file_path.split(os.sep)
        filename = os.path.basename(file_path)
        
        metadata = {}
        
        if "dummy_" in filename:
            metadata["model_type"] = "dummy"
        elif "ridge_" in filename:
            metadata["model_type"] = "ridge"
        elif "logistic_" in filename:
            metadata["model_type"] = "logistic"
        elif "xgboost_" in filename:
            metadata["model_type"] = "xgboost"
        elif "lm_probe" in filename or "all_results.json" in filename:
            metadata["model_type"] = "lm_probe"
            
        if "question_type" in filename:
            metadata["task"] = "question_type"
        elif "complexity" in filename:
            if any(submetric in filename for submetric in [
                "avg_links_len", "avg_max_depth", "avg_subordinate_chain_len",
                "avg_verb_edges", "lexical_density", "n_tokens"
            ]):
                for submetric in ["avg_links_len", "avg_max_depth", "avg_subordinate_chain_len",
                                 "avg_verb_edges", "lexical_density", "n_tokens"]:
                    if submetric in filename:
                        metadata["task"] = "single_submetric"
                        metadata["submetric"] = submetric
                        break
            else:
                metadata["task"] = "complexity"
                
        if "control1" in filename:
            metadata["is_control"] = True
            metadata["control_index"] = 1
        elif "control2" in filename:
            metadata["is_control"] = True
            metadata["control_index"] = 2
        elif "control3" in filename:
            metadata["is_control"] = True
            metadata["control_index"] = 3
        else:
            metadata["is_control"] = False
            metadata["control_index"] = None
            
        # Check for layer information in path
        for part in parts:
            if part.startswith("layer_"):
                metadata["layer"] = int(part.split("_")[1])
                break
                
        # Check for cross-lingual information
        if "cross_lingual" in file_path:
            metadata["is_cross_lingual"] = True
        else:
            metadata["is_cross_lingual"] = False
            
        return metadata
    
    def parse_result_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                
            metadata = self.extract_metadata_from_path(file_path)
            
            metadata["file_path"] = file_path
            
            result = {**metadata}
            
            if "model_type" in data:
                result["model_type"] = data["model_type"]
                
            if "task" in data:
                result["task"] = data["task"]
                
            if "task_type" in data:
                result["task_type"] = data["task_type"]
                
            if "language" in data:
                result["language"] = data["language"]
                
            if "train_language" in data and "eval_language" in data:
                result["is_cross_lingual"] = True
                result["train_language"] = data["train_language"]
                result["eval_language"] = data["eval_language"]
                
            if "is_control" in data:
                result["is_control"] = data["is_control"]
                
            if "control_index" in data:
                result["control_index"] = data["control_index"]
                
            if "submetric" in data:
                result["submetric"] = data["submetric"]
                
            for metric_type in ["train_metrics", "val_metrics", "test_metrics"]:
                if metric_type in data and data[metric_type] is not None:
                    for metric_name, metric_value in data[metric_type].items():
                        result[f"{metric_type.split('_')[0]}_{metric_name}"] = metric_value
            
            if "per_language_metrics" in data:
                for lang, lang_data in data["per_language_metrics"].items():
                    for split, split_data in lang_data.items():
                        for metric_name, metric_value in split_data.items():
                            result[f"{lang}_{split}_{metric_name}"] = metric_value
            
            return result
        except Exception as e:
            logger.warning(f"Error parsing {file_path}: {e}")
            return None
    
    def parse_all_results(self) -> pd.DataFrame:
       
        result_files = self.find_result_files()
        all_results = []
        
        for file_path in result_files:
            result = self.parse_result_file(file_path)
            if result:
                all_results.append(result)
                
        logger.info(f"Successfully parsed {len(all_results)} result files")
        
        self.results_df = pd.DataFrame(all_results)
        
        dedup_cols = [col for col in ["model_type", "task", "language", "is_control", "control_index", "layer", "submetric", "train_language", "eval_language"        
        ] if col in self.results_df.columns]
    
        before_count = len(self.results_df)

        self.results_df = self.results_df.drop_duplicates(subset=dedup_cols, keep='first')
        
        after_count = len(self.results_df)
        dupes_removed = before_count - after_count

        logger.info(f'removed {dupes_removed} duplicates')
        logger.info(f'final dataset contains {after_count} uniques')

        # Fill in missing task_type values based on task
        if "task" in self.results_df.columns and "task_type" in self.results_df.columns:
            task_type_map = {
                "question_type": "classification",
                "complexity": "regression",
                "single_submetric": "regression"
            }
            
            mask = self.results_df["task_type"].isna()
            self.results_df.loc[mask, "task_type"] = self.results_df.loc[mask, "task"].map(task_type_map)
            
        # Save parsed results
        self.results_df.to_csv(os.path.join(self.output_dir, "parsed_results.csv"), index=False)
        
        return self.results_df
    
    def get_task_summary(self, task: str) -> Dict[str, Any]:
        
        task_df = self.results_df[self.results_df["task"] == task].copy()
        
        if task_df.empty:
            return {"task": task, "status": "no_data"}
        
        task_type = task_df["task_type"].iloc[0] if "task_type" in task_df.columns else None
        
        metrics = self.get_available_metrics(task_type) if task_type else []
        
        real_df = task_df[task_df["is_control"] == False]
        control_df = task_df[task_df["is_control"] == True]
        
        summary = {
            "task": task,
            "task_type": task_type,
            "n_examples": len(real_df),
            "n_control_examples": len(control_df),
            "models": real_df["model_type"].unique().tolist() if "model_type" in real_df.columns else [],
            "languages": real_df["language"].unique().tolist() if "language" in real_df.columns else [],
            "metrics": metrics
        }
        
        for metric in metrics:
            if metric in real_df.columns:
                higher_is_better = not metric.endswith(("mse", "rmse", "loss"))
                best_idx = real_df[metric].idxmax() if higher_is_better else real_df[metric].idxmin()
                
                if pd.notna(best_idx):
                    best_row = real_df.loc[best_idx]
                    summary[f"best_{metric}"] = {
                        "value": best_row[metric],
                        "model": best_row.get("model_type", "unknown"),
                        "language": best_row.get("language", "unknown")
                    }
                    
        return summary

    def get_available_metrics(self, task_type: str) -> List[str]:
       
        if task_type == "classification":
            return [col for col in self.results_df.columns if 
                    any(col.endswith(suffix) for suffix in ["_accuracy", "_f1", "_precision", "_recall"])]
        else:  
            return [col for col in self.results_df.columns if 
                    any(col.endswith(suffix) for suffix in ["_r2", "_mse", "_rmse", "_mae"])]
    
    def run_analysis(self):
        """Run all analyses and generate a comprehensive report."""
        if self.results_df is None:
            self.parse_all_results()
        
        all_tasks = self.results_df["task"].unique()
        for task in all_tasks:
            self.task_summaries[task] = self.get_task_summary(task)
        
        with open(os.path.join(self.output_dir, "task_summaries.json"), "w") as f:
            json.dump(self.task_summaries, f, indent=2)
            
        self.generate_all_figures()
        
        self.generate_report()
        
        logger.info(f"Analysis complete. Results saved to {self.output_dir}")
        
    def generate_all_figures(self):
        """Generate all figures for the analysis."""
        # Create basic figure directory
        figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Process each task
        for task_name in self.task_summaries.keys():
            # Determine the appropriate metric based on task type
            task_type = self.task_summaries[task_name].get("task_type")
            
            if task_type == "classification":
                primary_metric = "test_accuracy"
                metric_name = "Accuracy"
            else:  # regression
                primary_metric = "test_r2"
                metric_name = "R² Score"
            
            # 1. Model comparison figure
            self.generate_model_comparison_figure(task_name, primary_metric, metric_name)
            
            # 2. Real vs Control tasks figure
            self.generate_real_vs_control_figure(task_name, primary_metric, metric_name)
            
            # 3. Layer-wise performance figure (if applicable)
            self.generate_layer_performance_figure(task_name, primary_metric, metric_name)
            
            # 4. Cross-lingual transfer figure (if applicable)
            self.generate_cross_lingual_figure(task_name, primary_metric, metric_name)
        
        # 5. Submetric comparison figure (if applicable)
        self.generate_submetric_figure(primary_metric="test_r2")
        
    def generate_model_comparison_figure(self, task_name, primary_metric, metric_name):
        """Generate figure comparing model performance."""
        # Filter for the given task and non-control experiments
        task_df = self.results_df[(self.results_df["task"] == task_name) & 
                                 (self.results_df["is_control"] == False)]
        
        if task_df.empty or primary_metric not in task_df.columns:
            return
        
        # Group by model and language
        model_lang_df = task_df.groupby(["model_type", "language"])[primary_metric].mean().reset_index()
        
        # Create figure
        plt.figure(figsize=self.figsize_standard, dpi=self.dpi)
        
        # Create heatmap
        pivot_df = model_lang_df.pivot_table(index="model_type", columns="language", values=primary_metric)
        
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            cbar_kws={"label": metric_name}
        )
        
        # Customize plot
        plt.title(f"{task_name.replace('_', ' ').title()} - Model Performance by Language")
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "figures", f"{task_name}_model_comparison.png")
        plt.savefig(output_path)
        plt.close()
        
    def generate_real_vs_control_figure(self, task_name, primary_metric, metric_name):
        """Generate figure comparing real vs. control tasks."""
        # Filter for the given task
        task_df = self.results_df[self.results_df["task"] == task_name]
        
        if task_df.empty or primary_metric not in task_df.columns:
            return
        
        # Get real and control task data
        real_df = task_df[task_df["is_control"] == False]
        control_df = task_df[task_df["is_control"] == True]
        
        if real_df.empty or control_df.empty:
            return
            
        # Compute average performance
        real_avg = real_df.groupby("model_type")[primary_metric].mean().reset_index()
        real_avg["task_type"] = "Real Task"
        
        control_avg = control_df.groupby("model_type")[primary_metric].mean().reset_index()
        control_avg["task_type"] = "Control Task (Avg)"
        
        # Combine data
        combined_df = pd.concat([real_avg, control_avg])
        
        # Create figure
        plt.figure(figsize=self.figsize_standard, dpi=self.dpi)
        
        # Create bar plot
        ax = sns.barplot(
            data=combined_df,
            x="model_type",
            y=primary_metric,
            hue="task_type",
            palette=["#4CAF50", "#9E9E9E"]  # Green for real, gray for control
        )
        
        # Customize plot
        plt.title(f"{task_name.replace('_', ' ').title()} - Real vs. Control Task Performance")
        plt.xlabel("Model Type")
        plt.ylabel(metric_name)
        plt.legend(title="")
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
            
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "figures", f"{task_name}_real_vs_control.png")
        plt.savefig(output_path)
        plt.close()
        
    def generate_layer_performance_figure(self, task_name, primary_metric, metric_name):
        """Generate figure showing layer-wise performance."""
        # Filter for the given task
        task_df = self.results_df[(self.results_df["task"] == task_name) & 
                                 (self.results_df["is_control"] == False)]
        
        # Check if layer information exists
        if "layer" not in task_df.columns or task_df["layer"].isna().all():
            return
            
        if task_df.empty or primary_metric not in task_df.columns:
            return
        
        # Group by layer and language
        layer_df = task_df.groupby(["layer", "language"])[primary_metric].mean().reset_index()
        
        # Create figure
        plt.figure(figsize=self.figsize_standard, dpi=self.dpi)
        
        # Create line plot
        sns.lineplot(
            data=layer_df,
            x="layer",
            y=primary_metric,
            hue="language",
            palette=LANGUAGE_PALETTE,
            marker="o",
            markersize=8,
            linewidth=2
        )
        
        # Customize plot
        plt.title(f"{task_name.replace('_', ' ').title()} - Performance Across Model Layers")
        plt.xlabel("Layer")
        plt.ylabel(metric_name)
        plt.grid(True, linestyle="--", alpha=0.7)
        
        # Set x-ticks to be specific layer numbers
        layers = sorted(layer_df["layer"].unique())
        plt.xticks(layers)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "figures", f"{task_name}_layer_performance.png")
        plt.savefig(output_path)
        plt.close()
        
    def generate_cross_lingual_figure(self, task_name, primary_metric, metric_name):
        """Generate figure showing cross-lingual transfer."""
        # Filter for the given task and cross-lingual experiments
        task_df = self.results_df[(self.results_df["task"] == task_name) & 
                                 (self.results_df["is_cross_lingual"] == True)]
        
        if task_df.empty or primary_metric not in task_df.columns:
            return
            
        if "train_language" not in task_df.columns or "eval_language" not in task_df.columns:
            return
        
        # Create pivot table
        pivot_df = task_df.pivot_table(
            index="train_language",
            columns="eval_language",
            values=primary_metric,
            aggfunc="mean"
        )
        
        # Create figure
        plt.figure(figsize=self.figsize_square, dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            cbar_kws={"label": metric_name}
        )
        
        # Customize plot
        plt.title(f"{task_name.replace('_', ' ').title()} - Cross-lingual Transfer Performance")
        plt.xlabel("Target Language")
        plt.ylabel("Source Language")
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "figures", f"{task_name}_cross_lingual.png")
        plt.savefig(output_path)
        plt.close()
        
    def generate_submetric_figure(self, primary_metric="test_r2"):
        """Generate figure comparing performance on complexity submetrics."""
        # Filter for submetric experiments
        submetric_df = self.results_df[(self.results_df["task"] == "single_submetric") & 
                                      (self.results_df["is_control"] == False)]
        
        if submetric_df.empty or primary_metric not in submetric_df.columns or "submetric" not in submetric_df.columns:
            return
        
        # Create pivot table
        pivot_df = submetric_df.pivot_table(
            index="model_type",
            columns="submetric",
            values=primary_metric,
            aggfunc="mean"
        )
        
        # Create figure
        plt.figure(figsize=self.figsize_square, dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            cbar_kws={"label": "R² Score"}
        )
        
        # Customize plot
        plt.title("Complexity Submetrics - Model Performance")
        plt.xlabel("Submetric")
        plt.ylabel("Model Type")
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "figures", "submetric_performance.png")
        plt.savefig(output_path)
        plt.close()
        
        # Create language comparison figure
        lang_pivot = submetric_df.pivot_table(
            index="language",
            columns="submetric",
            values=primary_metric,
            aggfunc="mean"
        )
        
        # Create figure
        plt.figure(figsize=self.figsize_square, dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(
            lang_pivot,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            cbar_kws={"label": "R² Score"}
        )
        
        # Customize plot
        plt.title("Complexity Submetrics - Language Performance")
        plt.xlabel("Submetric")
        plt.ylabel("Language")
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, "figures", "submetric_by_language.png")
        plt.savefig(output_path)
        plt.close()
        
    def generate_report(self):
        """Generate a comprehensive Markdown report."""
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start the report
        md_content = f"""# Multilingual Question Probing Experiment Results

Generated on: {timestamp}

## Overview

This report presents a comprehensive analysis of experimental results from our multilingual question probing framework. The experiments investigate how pre-trained multilingual contextual encoder models encode sentence-level linguistic features across diverse languages.

### Experiment Structure

The experiments are organized into the following categories:

1. **Question Type Classification**: Binary classification of questions as polar (yes/no) or content (wh-)
2. **Complexity Regression**: Prediction of an overall complexity score
3. **Submetric Regression**: Prediction of specific complexity submetrics:
   - Average dependency links length
   - Maximum tree depth
   - Subordinate chain length
   - Verb edges
   - Lexical density
   - Number of tokens
4. **Layer-wise Analysis**: Probing performance at different model layers (2, 6, 11, 12)
5. **Cross-lingual Transfer**: Training on one language and evaluating on another
6. **Control Tasks**: Shuffled label versions to test model selectivity

"""

        # Add task summaries
        md_content += "## Task Summaries\n\n"
        
        for task_name, summary in self.task_summaries.items():
            md_content += f"### {task_name.replace('_', ' ').title()}\n\n"
            
            # Basic info
            md_content += f"- **Task Type**: {summary.get('task_type', 'Unknown')}\n"
            md_content += f"- **Number of Examples**: {summary.get('n_examples', 0)}\n"
            md_content += f"- **Number of Control Examples**: {summary.get('n_control_examples', 0)}\n"
            
            # Models and languages
            if "models" in summary and summary["models"]:
                md_content += f"- **Models**: {', '.join(summary['models'])}\n"
            
            if "languages" in summary and summary["languages"]:
                md_content += f"- **Languages**: {', '.join(summary['languages'])}\n"
            
            # Best performances
            md_content += "\n**Best Performances**:\n\n"
            
            metrics = summary.get('metrics', [])
            if metrics:
                md_content += "| Metric | Value | Model | Language |\n"
                md_content += "| --- | --- | --- | --- |\n"
                
                for metric in metrics:
                    best_key = f"best_{metric}"
                    if best_key in summary:
                        best = summary[best_key]
                        md_content += f"| {metric.replace('_', ' ').title()} | {best['value']:.4f} | {best['model']} | {best['language']} |\n"
            
            md_content += "\n"
        
        # Add model comparison section
        md_content += "## Model Comparisons\n\n"
        
        for task_name in self.task_summaries.keys():
            md_content += f"### {task_name.replace('_', ' ').title()}\n\n"
            
            # Add model comparison figure if it exists
            figure_path = f"figures/{task_name}_model_comparison.png"
            if os.path.exists(os.path.join(self.output_dir, figure_path)):
                md_content += f"![Model Performance Comparison]({figure_path})\n\n"
                md_content += "*Figure: Performance heatmap showing results for each model and language combination.*\n\n"
            
            # Add a brief text summary
            task_df = self.results_df[(self.results_df["task"] == task_name) & 
                                     (self.results_df["is_control"] == False)]
            
            if not task_df.empty:
                # Determine the primary metric
                task_type = self.task_summaries[task_name].get("task_type")
                primary_metric = "test_accuracy" if task_type == "classification" else "test_r2"
                metric_name = "Accuracy" if task_type == "classification" else "R² Score"
                
                if primary_metric in task_df.columns:
                    # Find best model and language
                    best_model = task_df.groupby("model_type")[primary_metric].mean().idxmax()
                    best_lang = task_df.groupby("language")[primary_metric].mean().idxmax()
                    
                    overall_avg = task_df[primary_metric].mean()
                    
                    md_content += f"Overall, the **{best_model}** model achieved the highest average {metric_name.lower()} "
                    md_content += f"({overall_avg:.4f}). Among the languages, **{best_lang}** showed the best performance.\n\n"
        
        # Add selectivity analysis section
        md_content += "## Selectivity Analysis (Real vs. Control Tasks)\n\n"
        
        for task_name in self.task_summaries.keys():
            md_content += f"### {task_name.replace('_', ' ').title()}\n\n"
            
            # Add real vs. control figure if it exists
            figure_path = f"figures/{task_name}_real_vs_control.png"
            if os.path.exists(os.path.join(self.output_dir, figure_path)):
                md_content += f"![Real vs. Control Performance]({figure_path})\n\n"
                md_content += "*Figure: Comparison between real task and control task performance.*\n\n"
            
            # Add text summary of selectivity findings
            task_df = self.results_df[self.results_df["task"] == task_name]
            
            if not task_df.empty:
                real_df = task_df[task_df["is_control"] == False]
                control_df = task_df[task_df["is_control"] == True]
                
                if not real_df.empty and not control_df.empty:
                    task_type = self.task_summaries[task_name].get("task_type")
                    primary_metric = "test_accuracy" if task_type == "classification" else "test_r2"
                    
                    if primary_metric in real_df.columns and primary_metric in control_df.columns:
                        # Calculate average performance by model
                        real_avg = real_df.groupby("model_type")[primary_metric].mean()
                        control_avg = control_df.groupby("model_type")[primary_metric].mean()
                        
                        # Calculate selectivity
                        selectivity = {}
                        for model in real_avg.index:
                            if model in control_avg.index:
                                selectivity[model] = real_avg[model] - control_avg[model]
                        
                        if selectivity:
                            best_model = max(selectivity.items(), key=lambda x: x[1])[0]
                            worst_model = min(selectivity.items(), key=lambda x: x[1])[0]
                            
                            md_content += f"The **{best_model}** model showed the highest selectivity, indicating it best captures the true linguistic signal rather than dataset artifacts. "
                            md_content += f"In contrast, the **{worst_model}** model showed the lowest selectivity.\n\n"
        
        # Add layer-wise analysis section
        md_content += "## Layer-wise Analysis\n\n"
        
        for task_name in self.task_summaries.keys():
            # Filter for the given task
            task_df = self.results_df[(self.results_df["task"] == task_name) & 
                                     (self.results_df["is_control"] == False)]
            
            # Check if layer information exists
            if "layer" not in task_df.columns or task_df["layer"].isna().all():
                continue
                
            md_content += f"### {task_name.replace('_', ' ').title()}\n\n"
            
            # Add layer performance figure if it exists
            figure_path = f"figures/{task_name}_layer_performance.png"
            if os.path.exists(os.path.join(self.output_dir, figure_path)):
                md_content += f"![Layer-wise Performance]({figure_path})\n\n"
                md_content += "*Figure: Performance across different model layers.*\n\n"
            
            # Add summary of layer-wise findings
            if not task_df.empty:
                task_type = self.task_summaries[task_name].get("task_type")
                primary_metric = "test_accuracy" if task_type == "classification" else "test_r2"
                
                if primary_metric in task_df.columns:
                    # Get best layer by language
                    best_layers = {}
                    
                    for lang in task_df["language"].unique():
                        lang_df = task_df[task_df["language"] == lang]
                        if not lang_df.empty and primary_metric in lang_df.columns:
                            layer_perf = lang_df.groupby("layer")[primary_metric].mean()
                            if not layer_perf.empty:
                                best_layers[lang] = layer_perf.idxmax()
                    
                    # Calculate most common best layer
                    if best_layers:
                        from collections import Counter
                        layer_counts = Counter(best_layers.values())
                        most_common_layer = layer_counts.most_common(1)[0][0]
                        
                        md_content += f"For most languages, performance peaks at **layer {most_common_layer}**. "
                        md_content += "However, there are language-specific variations:\n\n"
                        
                        md_content += "| Language | Best Layer |\n"
                        md_content += "| --- | --- |\n"
                        
                        for lang, layer in best_layers.items():
                            md_content += f"| {lang} | {layer} |\n"
                        
                        md_content += "\n"
        
        # Add cross-lingual analysis section
        md_content += "## Cross-lingual Transfer\n\n"
        
        for task_name in self.task_summaries.keys():
            # Filter for cross-lingual experiments
            cl_df = self.results_df[(self.results_df["task"] == task_name) & 
                                   (self.results_df["is_cross_lingual"] == True)]
            
            if cl_df.empty or "train_language" not in cl_df.columns or "eval_language" not in cl_df.columns:
                continue
                
            md_content += f"### {task_name.replace('_', ' ').title()}\n\n"
            
            # Add cross-lingual figure if it exists
            figure_path = f"figures/{task_name}_cross_lingual.png"
            if os.path.exists(os.path.join(self.output_dir, figure_path)):
                md_content += f"![Cross-lingual Transfer]({figure_path})\n\n"
                md_content += "*Figure: Cross-lingual transfer performance from source (rows) to target (columns) languages.*\n\n"
            
            # Add text summary of cross-lingual findings
            task_type = self.task_summaries[task_name].get("task_type")
            primary_metric = "test_accuracy" if task_type == "classification" else "test_r2"
                
            if primary_metric in cl_df.columns:
                # Find best and worst transfers
                cl_grouped = cl_df.groupby(["train_language", "eval_language"])[primary_metric].mean().reset_index()
                
                if not cl_grouped.empty:
                    best_idx = cl_grouped[primary_metric].idxmax()
                    worst_idx = cl_grouped[primary_metric].idxmin()
                    
                    best_row = cl_grouped.iloc[best_idx]
                    worst_row = cl_grouped.iloc[worst_idx]
                    
                    md_content += f"The best cross-lingual transfer was observed from **{best_row['train_language']}** to **{best_row['eval_language']}** "
                    md_content += f"with {primary_metric.split('_')[1]} of {best_row[primary_metric]:.4f}. "
                    md_content += f"The worst transfer was from **{worst_row['train_language']}** to **{worst_row['eval_language']}** "
                    md_content += f"with {primary_metric.split('_')[1]} of {worst_row[primary_metric]:.4f}.\n\n"
        
        # Add submetric analysis section
        md_content += "## Complexity Submetric Analysis\n\n"
        
        # Add submetric figure if it exists
        figure_path = "figures/submetric_performance.png"
        if os.path.exists(os.path.join(self.output_dir, figure_path)):
            md_content += f"![Submetric Performance]({figure_path})\n\n"
            md_content += "*Figure: Performance on different complexity submetrics by model type.*\n\n"
            
        # Add language submetric figure if it exists
        figure_path = "figures/submetric_by_language.png"
        if os.path.exists(os.path.join(self.output_dir, figure_path)):
            md_content += f"![Submetric by Language]({figure_path})\n\n"
            md_content += "*Figure: Performance on different complexity submetrics by language.*\n\n"
        
        # Add text summary of submetric findings
        submetric_df = self.results_df[(self.results_df["task"] == "single_submetric") & 
                                      (self.results_df["is_control"] == False)]
        
        if not submetric_df.empty and "submetric" in submetric_df.columns:
            primary_metric = "test_r2"  # Use R² as the primary metric for submetrics
            
            if primary_metric in submetric_df.columns:
                # Find easiest and hardest submetrics
                submetric_perf = submetric_df.groupby("submetric")[primary_metric].mean()
                
                if not submetric_perf.empty:
                    easiest = submetric_perf.idxmax()
                    hardest = submetric_perf.idxmin()
                    
                    md_content += f"Among the complexity submetrics, **{easiest}** was the easiest to predict with an average R² of {submetric_perf[easiest]:.4f}, "
                    md_content += f"while **{hardest}** was the most challenging with an average R² of {submetric_perf[hardest]:.4f}.\n\n"
                    
                    # Find best model for submetrics
                    model_perf = submetric_df.groupby("model_type")[primary_metric].mean()
                    if not model_perf.empty:
                        best_model = model_perf.idxmax()
                        md_content += f"The **{best_model}** model performed best overall on complexity submetric prediction.\n\n"
        
        # Add conclusions section
        md_content += "## Key Findings and Conclusions\n\n"
        
        md_content += "Based on the comprehensive analysis of our experimental results, we can draw the following key findings:\n\n"
        
        # Generate conclusions based on data
        conclusions = []
        
        # 1. Best model conclusion
        model_performances = {}
        for task_name in self.task_summaries.keys():
            task_df = self.results_df[(self.results_df["task"] == task_name) & 
                                     (self.results_df["is_control"] == False)]
            
            if not task_df.empty:
                task_type = self.task_summaries[task_name].get("task_type")
                primary_metric = "test_accuracy" if task_type == "classification" else "test_r2"
                
                if primary_metric in task_df.columns:
                    model_avg = task_df.groupby("model_type")[primary_metric].mean()
                    for model, perf in model_avg.items():
                        if model not in model_performances:
                            model_performances[model] = []
                        model_performances[model].append((task_name, perf))
        
        if model_performances:
            # Calculate overall average performance
            overall_avg = {model: np.mean([p[1] for p in perfs]) for model, perfs in model_performances.items()}
            if overall_avg:
                best_model = max(overall_avg.items(), key=lambda x: x[1])[0]
                conclusions.append(f"1. **Model Performance**: The **{best_model}** model consistently performed best across tasks, indicating its superior capability to encode linguistic signals in multilingual questions.")
        
        # 2. Layer conclusion
        if any("layer" in self.results_df.columns and not pd.isna(self.results_df["layer"]).all()):
            layer_df = self.results_df[(self.results_df["is_control"] == False) & 
                                      (self.results_df["layer"].notna())]
            
            if not layer_df.empty:
                best_layers = {}
                for task in layer_df["task"].unique():
                    task_layer_df = layer_df[layer_df["task"] == task]
                    task_type = self.task_summaries.get(task, {}).get("task_type")
                    primary_metric = "test_accuracy" if task_type == "classification" else "test_r2"
                    
                    if primary_metric in task_layer_df.columns:
                        layer_perf = task_layer_df.groupby("layer")[primary_metric].mean()
                        if not layer_perf.empty:
                            best_layers[task] = layer_perf.idxmax()
                
                if best_layers:
                    from collections import Counter
                    layer_counts = Counter(best_layers.values())
                    most_common_layer = layer_counts.most_common(1)[0][0]
                    
                    conclusions.append(f"2. **Layer Analysis**: Layer **{most_common_layer}** generally yielded the best performance, suggesting this is where the model optimally encodes the linguistic properties relevant to our tasks.")
        
        # 3. Selectivity conclusion
        if any(self.results_df["is_control"] == True):
            selectivity_scores = {}
            
            for task in self.task_summaries.keys():
                task_df = self.results_df[self.results_df["task"] == task]
                real_df = task_df[task_df["is_control"] == False]
                control_df = task_df[task_df["is_control"] == True]
                
                if not real_df.empty and not control_df.empty:
                    task_type = self.task_summaries[task].get("task_type")
                    primary_metric = "test_accuracy" if task_type == "classification" else "test_r2"
                    
                    if primary_metric in real_df.columns and primary_metric in control_df.columns:
                        real_avg = real_df.groupby("model_type")[primary_metric].mean()
                        control_avg = control_df.groupby("model_type")[primary_metric].mean()
                        
                        for model in real_avg.index:
                            if model in control_avg.index:
                                if model not in selectivity_scores:
                                    selectivity_scores[model] = []
                                selectivity_scores[model].append(real_avg[model] - control_avg[model])
            
            if selectivity_scores:
                avg_selectivity = {model: np.mean(scores) for model, scores in selectivity_scores.items()}
                best_selective_model = max(avg_selectivity.items(), key=lambda x: x[1])[0]
                
                conclusions.append(f"3. **Selectivity**: The **{best_selective_model}** model showed the highest selectivity, indicating it best captures the true linguistic properties rather than dataset artifacts.")
        
        # 4. Cross-lingual conclusion
        if any(self.results_df["is_cross_lingual"] == True):
            cl_df = self.results_df[self.results_df["is_cross_lingual"] == True]
            
            if not cl_df.empty and "train_language" in cl_df.columns and "eval_language" in cl_df.columns:
                language_pairs = []
                
                for task in cl_df["task"].unique():
                    task_cl_df = cl_df[cl_df["task"] == task]
                    task_type = self.task_summaries.get(task, {}).get("task_type")
                    primary_metric = "test_accuracy" if task_type == "classification" else "test_r2"
                    
                    if primary_metric in task_cl_df.columns:
                        pairs = task_cl_df.groupby(["train_language", "eval_language"])[primary_metric].mean()
                        top_pairs = pairs.nlargest(3)
                        
                        for (src, tgt), score in top_pairs.items():
                            language_pairs.append((src, tgt, score))
                
                if language_pairs:
                    # Sort by score
                    language_pairs.sort(key=lambda x: x[2], reverse=True)
                    top_pairs = [f"{src}->{tgt}" for src, tgt, _ in language_pairs[:3]]
                    
                    conclusions.append(f"4. **Cross-lingual Transfer**: The best cross-lingual transfer was observed between the following language pairs: {', '.join(top_pairs)}. This suggests stronger linguistic alignment between these languages for the properties being probed.")
        
        # 5. Submetric conclusion
        submetric_df = self.results_df[(self.results_df["task"] == "single_submetric") & 
                                      (self.results_df["is_control"] == False)]
        
        if not submetric_df.empty and "submetric" in submetric_df.columns:
            primary_metric = "test_r2"
            
            if primary_metric in submetric_df.columns:
                submetric_perf = submetric_df.groupby("submetric")[primary_metric].mean()
                
                if not submetric_perf.empty:
                    # Sort submetrics by performance
                    sorted_metrics = submetric_perf.sort_values(ascending=False)
                    
                    easy_metrics = sorted_metrics.index[:2].tolist()  # Top 2 easiest
                    hard_metrics = sorted_metrics.index[-2:].tolist()  # Top 2 hardest
                    
                    easy_str = " and ".join([f"**{m}**" for m in easy_metrics])
                    hard_str = " and ".join([f"**{m}**" for m in hard_metrics])
                    
                    conclusions.append(f"5. **Complexity Submetrics**: The models were most effective at capturing {easy_str}, while struggling more with {hard_str}. This suggests that some complexity features are more readily captured in multilingual representations than others.")
        
        # Add all conclusions
        for conclusion in conclusions:
            md_content += f"{conclusion}\n\n"
        
        # Add missing standard conclusions if needed
        if len(conclusions) < 3:
            md_content += "Based on our experiments, we can conclude that pre-trained multilingual models show varying capabilities in encoding linguistic properties across languages, with performance differences observed between languages, model types, and specific linguistic tasks.\n\n"
        
        # Save the report
        report_path = os.path.join(self.output_dir, "experimental_results_analysis.md")
        with open(report_path, "w") as f:
            f.write(md_content)
        
        logger.info(f"Report saved to {report_path}")
        
        # Try to convert to HTML if possible
        try:
            import markdown
            html_content = markdown.markdown(
                md_content,
                extensions=['markdown.extensions.tables', 'markdown.extensions.fenced_code']
            )
            
            html_report = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Multilingual Question Probing Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3, h4 {{
            color: #2c3e50;
            margin-top: 1.5em;
        }}
        h1 {{
            border-bottom: 2px solid #eaecef;
            padding-bottom: 0.3em;
        }}
        h2 {{
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 1.5em auto;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }}
        th, td {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        code {{
            background-color: #f6f8fa;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
        }}
        pre {{
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 3px;
            overflow: auto;
        }}
        blockquote {{
            border-left: 4px solid #ddd;
            padding: 0 15px;
            color: #777;
            margin: 0;
        }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""
            
            html_path = os.path.join(self.output_dir, "experimental_results_analysis.html")
            with open(html_path, "w") as f:
                f.write(html_report)
                
            logger.info(f"HTML report saved to {html_path}")
        except ImportError:
            logger.warning("Could not create HTML report. Install 'markdown' package if needed.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze multilingual question probing experiments")
    parser.add_argument("--results-dirs", nargs="+", required=True, help="Directories containing result files")
    parser.add_argument("--output-dir", default="analysis_output", help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer(args.results_dirs, args.output_dir)
    analyzer.run_analysis()
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print(f"  - Parsed data: {os.path.join(args.output_dir, 'parsed_results.csv')}")
    print(f"  - Report: {os.path.join(args.output_dir, 'experimental_results_analysis.md')}")
    print(f"  - HTML Report: {os.path.join(args.output_dir, 'experimental_results_analysis.html')}")
    print(f"  - Figures: {os.path.join(args.output_dir, 'figures')}")

if __name__ == "__main__":
    main()