#!/usr/bin/env python3
"""
Analyze feature importance from TF-IDF experiments and create consolidated tables.
Shows top 10 features for regression and classification tasks.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle

def load_tfidf_vocabulary(features_dir):
    """Load TF-IDF vocabulary mapping from metadata."""
    try:
        metadata_path = Path(features_dir) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                vocabulary = metadata.get('vocabulary', {})
                # Reverse mapping: index -> token
                return {v: k for k, v in vocabulary.items()}
        
        print(f"Warning: No metadata found at {metadata_path}")
        return {}
    except Exception as e:
        print(f"Error loading vocabulary: {e}")
        return {}
    


def collect_feature_importance_data(results_dir, vocab_mapping):
    """Collect feature importance data from experiment results."""
    results_path = Path(results_dir)
    
    feature_data = defaultdict(list)
    
    # Find all result files
    result_files = list(results_path.rglob("results.json"))
    if not result_files:
        result_files = list(results_path.rglob("*.json"))
    
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract experiment metadata
            model_type = data.get('model_type', 'unknown')
            task_type = data.get('task_type', 'unknown')
            language = data.get('language', 'unknown')
            task = data.get('task', 'unknown')
            
            # Skip dummy models (no meaningful feature importance)
            if model_type == 'dummy':
                continue
                
            # Extract feature importance
            feature_importance = data.get('feature_importance')
            if not feature_importance:
                continue
                
            values = np.array(feature_importance['values'])
            top_10_indices = feature_importance.get('top_10_indices', [])
            
            # If we don't have pre-computed top 10, compute it
            if not top_10_indices:
                top_10_indices = np.argsort(np.abs(values))[-10:].tolist()[::-1]
            
            # Get top 10 feature information
            top_features = []
            for idx in top_10_indices:
                if idx < len(values):
                    importance_value = values[idx]
                    feature_name = vocab_mapping.get(idx, f"feature_{idx}")
                    top_features.append({
                        'feature_idx': idx,
                        'feature_name': feature_name,
                        'importance': float(importance_value)
                    })
            
            # Store data
            experiment_key = f"{task_type}_{task}_{language}_{model_type}"
            feature_data[experiment_key] = {
                'model_type': model_type,
                'task_type': task_type,
                'task': task,
                'language': language,
                'top_features': top_features,
                'experiment_file': str(result_file.relative_to(results_path))
            }
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue
    
    return feature_data

def create_consolidated_tables(feature_data):
    """Create consolidated feature importance tables by task type."""

    classification_data = []
    regression_data = []
    
    for exp_key, exp_data in feature_data.items():
        task_type = exp_data['task_type']
        
        for i, feature_info in enumerate(exp_data['top_features'], 1):
            row = {
                'rank': i,
                'task': exp_data['task'],
                'language': exp_data['language'],
                'model': exp_data['model_type'],
                'feature_name': feature_info['feature_name'],
                'importance': feature_info['importance'],
                'abs_importance': abs(feature_info['importance'])
            }
            
            if task_type == 'classification':
                classification_data.append(row)
            elif task_type == 'regression':
                regression_data.append(row)
    
    # Create DataFrames
    classification_df = pd.DataFrame(classification_data) if classification_data else pd.DataFrame()
    regression_df = pd.DataFrame(regression_data) if regression_data else pd.DataFrame()
    
    return classification_df, regression_df

def create_summary_tables(classification_df, regression_df):
    """Create summary tables with overall top features."""
    
    summaries = {}
    
    if not classification_df.empty:
        # features all classification tasks
        classification_summary = (classification_df
                                .groupby(['feature_name'])
                                .agg({
                                    'abs_importance': ['mean', 'std', 'count'],
                                    'importance': 'mean'
                                })
                                .round(4))
        
        classification_summary.columns = ['avg_abs_importance', 'std_abs_importance', 'frequency', 'avg_importance']
        classification_summary = classification_summary.sort_values('avg_abs_importance', ascending=False).head(10)
        classification_summary = classification_summary.reset_index()
        summaries['classification'] = classification_summary
    
    if not regression_df.empty:
        # Top features  all regression tasks  
        regression_summary = (regression_df
                            .groupby(['feature_name'])
                            .agg({
                                'abs_importance': ['mean', 'std', 'count'],
                                'importance': 'mean'
                            })
                            .round(4))
        
        regression_summary.columns = ['avg_abs_importance', 'std_abs_importance', 'frequency', 'avg_importance']
        regression_summary = regression_summary.sort_values('avg_abs_importance', ascending=False).head(10)
        regression_summary = regression_summary.reset_index()
        summaries['regression'] = regression_summary
    
    return summaries

def main():
    parser = argparse.ArgumentParser(description="Analyze TF-IDF feature importance from experiments")
    parser.add_argument("results_dir", help="Directory containing experiment results")
    parser.add_argument("--features-dir", default="data/tfidf_features", 
                       help="Directory containing TF-IDF features and vocabulary")
    parser.add_argument("--output-dir", default="analysis_output",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} does not exist")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading TF-IDF vocabulary...")
    vocab_mapping = load_tfidf_vocabulary(features_dir)
    print(f"Loaded vocabulary with {len(vocab_mapping)} tokens")
    
    print("Collecting feature importance data...")
    feature_data = collect_feature_importance_data(results_dir, vocab_mapping)
    print(f"Processed {len(feature_data)} experiments")
    
    if not feature_data:
        print("No feature importance data found")
        return
    
    print("Creating consolidated tables...")
    classification_df, regression_df = create_consolidated_tables(feature_data)
    

    if not classification_df.empty:
        classification_df.to_csv(output_dir / "classification_feature_importance_detailed.csv", index=False)
        print(f"Saved classification details: {len(classification_df)} entries")
    
    if not regression_df.empty:
        regression_df.to_csv(output_dir / "regression_feature_importance_detailed.csv", index=False)
        print(f"Saved regression details: {len(regression_df)} entries")
    
    print("Creating summary tables...")
    summaries = create_summary_tables(classification_df, regression_df)

    for task_type, summary_df in summaries.items():
        print(f"\n{'='*50}")
        print(f"TOP 10 FEATURES FOR {task_type.upper()} TASKS")
        print(f"{'='*50}")
        print(summary_df.to_string(index=False))

        output_file = output_dir / f"top_10_features_{task_type}.csv"
        summary_df.to_csv(output_file, index=False)
        print(f"\nSaved to: {output_file}")
    
    # Save  metadata
    metadata = {
        'total_experiments': len(feature_data),
        'classification_experiments': len([d for d in feature_data.values() if d['task_type'] == 'classification']),
        'regression_experiments': len([d for d in feature_data.values() if d['task_type'] == 'regression']),
        'unique_tasks': list(set(d['task'] for d in feature_data.values())),
        'unique_languages': list(set(d['language'] for d in feature_data.values())),
        'unique_models': list(set(d['model_type'] for d in feature_data.values())),
        'vocab_size': len(vocab_mapping)
    }
    
    with open(output_dir / "analysis_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"results saved to: {output_dir}")
    print(f"Metadata: {metadata}")

if __name__ == "__main__":
    main()