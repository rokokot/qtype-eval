
# Analysis script for multilingual question probing experiments.
# This script processes all experiment results and generates visualizations and tables.

import os
import argparse
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LANGUAGES = ["ar", "en", "fi", "id", "ja", "ko", "ru"]
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "fi": "Finnish",
    "id": "Indonesian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian"}

TASKS = ["question_type", "complexity"]
SUBMETRICS = ["avg_links_len", "avg_max_depth", "avg_subordinate_chain_len",
    "avg_verb_edges", "lexical_density", "n_tokens"]
MODEL_TYPES = ["dummy", "logistic", "ridge", "xgboost", "lm_probe"]

def load_results(results_dir: str) -> List[Dict]:
    """
    Load all results from JSON files in the results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        List of result dictionaries
    """
    # Find all JSON result files
    result_files = glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True)
    result_files += glob.glob(os.path.join(results_dir, "**", "results_with_metadata.json"), recursive=True)
    result_files += glob.glob(os.path.join(results_dir, "**", "all_results.json"), recursive=True)
    result_files += glob.glob(os.path.join(results_dir, "**", "cross_lingual_results.json"), recursive=True)
    
    logger.info(f"Found {len(result_files)} result files")
    
    # Load each file
    results = []
    for file in result_files:
        try:
            with open(file, "r") as f:
                result = json.load(f)
            
            # Add experiment name from directory
            parts = file.split(os.sep)
            experiment_name = parts[-3] if len(parts) >= 3 else parts[-2]
            result["experiment_name"] = experiment_name
            
            # Add file path
            result["file_path"] = file
            
            results.append(result)
                
        except Exception as e:
            logger.warning(f"Error loading {file}: {e}")
    
    logger.info(f"Loaded {len(results)} results")
    return results

def process_results(results: List[Dict]) -> pd.DataFrame:
    """
    Process results into a DataFrame.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        DataFrame with processed results
    """
    processed_results = []
    
    for result in results:
        # Extract basic metadata
        base_info = {
            "file_path": result.get("file_path", ""),
            "language": result.get("language", result.get("train_language", None)),
            "eval_language": result.get("eval_language", None),
            "task": result.get("task", None),
            "task_type": result.get("task_type", None),
            "model_type": result.get("model_type", result.get("model_name", None)),
            "is_control": result.get("is_control", False),
            "control_index": result.get("control_index", None),
            "submetric": result.get("submetric", None),
            "cross_lingual": result.get("eval_language", None) is not None,
        }
        
        # Add metrics
        for metric_type in ["train_metrics", "val_metrics", "test_metrics"]:
            metrics = result.get(metric_type, {})
            if metrics and isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    metric_key = f"{metric_type.split('_')[0]}_{metric_name}"
                    base_info[metric_key] = metric_value
        
        processed_results.append(base_info)
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_results)
    
    # Add full language names
    if "language" in df.columns:
        df["language_name"] = df["language"].map(LANGUAGE_NAMES)
    
    if "eval_language" in df.columns:
        df["eval_language_name"] = df["eval_language"].map(LANGUAGE_NAMES)
    
    return df

def generate_question_type_plots(df: pd.DataFrame, output_dir: str):
    """Generate plots for question type classification task."""
    # Filter for question_type task and non-control experiments
    task_df = df[(df["task"] == "question_type") & (~df["is_control"])]
    
    if task_df.empty:
        logger.warning("No question_type results found, skipping plots")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy by model and language
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=task_df,
        x="language_name",
        y="test_accuracy",
        hue="model_type",
        palette="viridis"
    )
    
    plt.title("Question Type Classification Accuracy by Model and Language", fontsize=16)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.legend(title="Model Type", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "question_type_accuracy.png"), dpi=300)
    plt.close()
    
    # Plot F1 score by model and language
    if "test_f1" in task_df.columns:
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=task_df,
            x="language_name",
            y="test_f1",
            hue="model_type",
            palette="viridis"
        )
        
        plt.title("Question Type Classification F1 Score by Model and Language", fontsize=16)
        plt.xlabel("Language", fontsize=14)
        plt.ylabel("Test F1 Score", fontsize=14)
        plt.xticks(rotation=45)
        plt.ylim(0, 1.05)
        plt.legend(title="Model Type", fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "question_type_f1.png"), dpi=300)
        plt.close()

def generate_complexity_plots(df: pd.DataFrame, output_dir: str):
    """Generate plots for complexity regression task."""
    # Filter for complexity task and non-control experiments
    task_df = df[(df["task"] == "complexity") & (~df["is_control"])]
    
    if task_df.empty:
        logger.warning("No complexity results found, skipping plots")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot R2 by model and language
    if "test_r2" in task_df.columns:
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=task_df,
            x="language_name",
            y="test_r2",
            hue="model_type",
            palette="viridis"
        )
        
        plt.title("Complexity Regression R² Score by Model and Language", fontsize=16)
        plt.xlabel("Language", fontsize=14)
        plt.ylabel("Test R² Score", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Model Type", fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "complexity_r2.png"), dpi=300)
        plt.close()
    
    # Plot MSE by model and language
    if "test_mse" in task_df.columns:
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=task_df,
            x="language_name",
            y="test_mse",
            hue="model_type",
            palette="viridis"
        )
        
        plt.title("Complexity Regression MSE by Model and Language", fontsize=16)
        plt.xlabel("Language", fontsize=14)
        plt.ylabel("Test MSE", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Model Type", fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "complexity_mse.png"), dpi=300)
        plt.close()

def generate_submetric_plots(df: pd.DataFrame, output_dir: str):
    """Generate plots for submetric regression tasks."""
    # Filter for submetric tasks
    submetric_df = df[df["submetric"].isin(SUBMETRICS) & (~df["is_control"])]
    
    if submetric_df.empty:
        logger.warning("No submetric results found, skipping plots")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot R2 for each submetric across languages (for lm_probe model)
    lm_submetric_df = submetric_df[submetric_df["model_type"] == "lm_probe"]
    
    if not lm_submetric_df.empty and "test_r2" in lm_submetric_df.columns:
        plt.figure(figsize=(14, 10))
        
        sns.barplot(
            data=lm_submetric_df,
            x="submetric",
            y="test_r2",
            hue="language_name",
            palette="viridis"
        )
        
        plt.title(f"Submetric R² Scores for LM Probe across Languages", fontsize=16)
        plt.xlabel("Complexity Metric", fontsize=14)
        plt.ylabel("Test R² Score", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Language", fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "submetric_r2_lm_probe_all_languages.png"), dpi=300)
        plt.close()
    
    # Compare models for each submetric (averaged across languages)
    if "test_r2" in submetric_df.columns:
        avg_results = submetric_df.groupby(["model_type", "submetric"])["test_r2"].mean().reset_index()
        
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=avg_results,
            x="submetric",
            y="test_r2",
            hue="model_type",
            palette="viridis"
        )
        
        plt.title("Average R² Score by Model and Submetric", fontsize=16)
        plt.xlabel("Submetric", fontsize=14)
        plt.ylabel("Average R² Score", fontsize=14)
        plt.xticks(rotation=45)
        plt.legend(title="Model Type", fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "submetric_r2_by_model.png"), dpi=300)
        plt.close()

def generate_cross_lingual_plots(df: pd.DataFrame, output_dir: str):
    """Generate plots for cross-lingual experiments."""
    # Filter for cross-lingual experiments
    cross_lingual_df = df[df["cross_lingual"] == True]
    
    if cross_lingual_df.empty:
        logger.warning("No cross-lingual results found, skipping plots")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate heatmaps for each task
    for task in cross_lingual_df["task"].unique():
        task_df = cross_lingual_df[cross_lingual_df["task"] == task]
        
        # For question_type task - accuracy heatmap
        if task == "question_type" and "test_accuracy" in task_df.columns:
            # Create pivot table
            pivot_df = task_df.pivot_table(
                values="test_accuracy",
                index="language",
                columns="eval_language",
                aggfunc="mean"
            )
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            
            ax = sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Accuracy"}
            )
            
            plt.title(f"Cross-Lingual Question Type Accuracy (Train → Evaluate)", fontsize=16)
            plt.xlabel("Evaluation Language", fontsize=14)
            plt.ylabel("Training Language", fontsize=14)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "cross_lingual_question_type_accuracy.png"), dpi=300)
            plt.close()
        
        # For complexity task - R2 heatmap
        elif task == "complexity" and "test_r2" in task_df.columns:
            # Create pivot table
            pivot_df = task_df.pivot_table(
                values="test_r2",
                index="language",
                columns="eval_language",
                aggfunc="mean"
            )
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            
            ax = sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                cbar_kws={"label": "R² Score"}
            )
            
            plt.title(f"Cross-Lingual Complexity R² Score (Train → Evaluate)", fontsize=16)
            plt.xlabel("Evaluation Language", fontsize=14)
            plt.ylabel("Training Language", fontsize=14)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, "cross_lingual_complexity_r2.png"), dpi=300)
            plt.close()

def generate_summary_tables(df: pd.DataFrame, output_dir: str):
    """Generate summary tables with key metrics."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for non-control experiments
    real_df = df[~df["is_control"]]
    
    if real_df.empty:
        logger.warning("No real task results found, skipping summary tables")
        return
    
    # Question type classification summary
    question_type_df = real_df[real_df["task"] == "question_type"]
    
    if not question_type_df.empty and "test_accuracy" in question_type_df.columns:
        # Create pivot table of accuracy by model and language
        accuracy_table = pd.pivot_table(
            question_type_df,
            values="test_accuracy",
            index="model_type",
            columns="language",
            aggfunc="mean"
        )
        
        # Reorder columns by language code
        available_langs = set(accuracy_table.columns) & set(LANGUAGES)
        if available_langs:
            accuracy_table = accuracy_table[[lang for lang in LANGUAGES if lang in available_langs]]
        
        # Add mean column
        accuracy_table["mean"] = accuracy_table.mean(axis=1)
        
        # Save table to CSV
        accuracy_table.to_csv(os.path.join(output_dir, "question_type_accuracy_summary.csv"))
        
        # Create pivot table of F1 score by model and language
        if "test_f1" in question_type_df.columns:
            f1_table = pd.pivot_table(
                question_type_df,
                values="test_f1",
                index="model_type",
                columns="language",
                aggfunc="mean"
            )
            
            # Reorder columns by language code
            available_langs = set(f1_table.columns) & set(LANGUAGES)
            if available_langs:
                f1_table = f1_table[[lang for lang in LANGUAGES if lang in available_langs]]
            
            # Add mean column
            f1_table["mean"] = f1_table.mean(axis=1)
            
            # Save table to CSV
            f1_table.to_csv(os.path.join(output_dir, "question_type_f1_summary.csv"))
    
    # Complexity regression summary
    complexity_df = real_df[real_df["task"] == "complexity"]
    
    if not complexity_df.empty and "test_r2" in complexity_df.columns:
        # Create pivot table of R2 by model and language
        r2_table = pd.pivot_table(
            complexity_df,
            values="test_r2",
            index="model_type",
            columns="language",
            aggfunc="mean"
        )
        
        # Reorder columns by language code
        available_langs = set(r2_table.columns) & set(LANGUAGES)
        if available_langs:
            r2_table = r2_table[[lang for lang in LANGUAGES if lang in available_langs]]
        
        # Add mean column
        r2_table["mean"] = r2_table.mean(axis=1)
        
        # Save table to CSV
        r2_table.to_csv(os.path.join(output_dir, "complexity_r2_summary.csv"))
        
        # Create pivot table of MSE by model and language
        if "test_mse" in complexity_df.columns:
            mse_table = pd.pivot_table(
                complexity_df,
                values="test_mse",
                index="model_type",
                columns="language",
                aggfunc="mean"
            )
            
            # Reorder columns by language code
            available_langs = set(mse_table.columns) & set(LANGUAGES)
            if available_langs:
                mse_table = mse_table[[lang for lang in LANGUAGES if lang in available_langs]]
            
            # Add mean column
            mse_table["mean"] = mse_table.mean(axis=1)
            
            # Save table to CSV
            mse_table.to_csv(os.path.join(output_dir, "complexity_mse_summary.csv"))
    
    # Submetric regression summary
    submetric_df = real_df[real_df["submetric"].isin(SUBMETRICS)]
    
    if not submetric_df.empty and "test_r2" in submetric_df.columns:
        # Create pivot table of R2 by submetric and model
        submetric_table = pd.pivot_table(
            submetric_df,
            values="test_r2",
            index="model_type",
            columns="submetric",
            aggfunc="mean"
        )
        
        # Add mean column
        submetric_table["mean"] = submetric_table.mean(axis=1)
        
        # Save table to CSV
        submetric_table.to_csv(os.path.join(output_dir, "submetric_r2_summary.csv"))
    
    # Cross-lingual summary
    cross_lingual_df = real_df[real_df["cross_lingual"] == True]
    
    if not cross_lingual_df.empty:
        # For question_type task
        q_type_cl_df = cross_lingual_df[cross_lingual_df["task"] == "question_type"]
        if not q_type_cl_df.empty and "test_accuracy" in q_type_cl_df.columns:
            cl_accuracy_table = pd.pivot_table(
                q_type_cl_df,
                values="test_accuracy",
                index="language",
                columns="eval_language",
                aggfunc="mean"
            )
            
            # Save table to CSV
            cl_accuracy_table.to_csv(os.path.join(output_dir, "cross_lingual_question_type_accuracy.csv"))
        
        # For complexity task
        complexity_cl_df = cross_lingual_df[cross_lingual_df["task"] == "complexity"]
        if not complexity_cl_df.empty and "test_r2" in complexity_cl_df.columns:
            cl_r2_table = pd.pivot_table(
                complexity_cl_df,
                values="test_r2",
                index="language",
                columns="eval_language",
                aggfunc="mean"
            )
            
            # Save table to CSV
            cl_r2_table.to_csv(os.path.join(output_dir, "cross_lingual_complexity_r2.csv"))

def main(args):
    """Main function to run the analysis."""
    # Load results
    results = load_results(args.results_dir)
    
    # Process results
    df = process_results(results)
    
    # Save processed results
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(os.path.join(args.output_dir, "processed_results.csv"), index=False)
    
    # Generate plots and tables
    generate_question_type_plots(df, os.path.join(args.output_dir, "question_type"))
    generate_complexity_plots(df, os.path.join(args.output_dir, "complexity"))
    generate_submetric_plots(df, os.path.join(args.output_dir, "submetrics"))
    generate_cross_lingual_plots(df, os.path.join(args.output_dir, "cross_lingual"))
    generate_summary_tables(df, os.path.join(args.output_dir, "tables"))
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing experiment results")
    parser.add_argument("--output-dir", type=str, default="analysis", help="Directory to save analysis outputs")
    args = parser.parse_args()
    
    main(args)