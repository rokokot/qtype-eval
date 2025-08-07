#!/usr/bin/env python3
"""
Utility script to collect and aggregate results from different experiment directories.

Searches for results files and consolidates them into analysis-ready formats.
"""

import pandas as pd
import json
import argparse
from pathlib import Path
import glob
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class ResultsCollector:
    """Collects and aggregates experiment results from various sources."""
    
    def __init__(self):
        pass
    
    def collect_tfidf_results(self, base_dir: str) -> pd.DataFrame:
        """Collect TF-IDF baseline results from experiment directories."""
        base_path = Path(base_dir)
        results = []
        
        # Look for TF-IDF results CSV files
        tfidf_csv_files = list(base_path.glob("**/tfidf_results.csv"))
        
        if tfidf_csv_files:
            logger.info(f"Found {len(tfidf_csv_files)} TF-IDF results files")
            for csv_file in tfidf_csv_files:
                logger.info(f"Loading: {csv_file}")
                df = pd.read_csv(csv_file)
                results.append(df)
        else:
            # Look for individual result files in directory structure
            logger.info("No consolidated CSV found, searching for individual result files...")
            
            pattern = "**/question_type/**/results.json"
            result_files = list(base_path.glob(pattern))
            
            pattern2 = "**/complexity/**/results.json"  
            result_files.extend(list(base_path.glob(pattern2)))
            
            logger.info(f"Found {len(result_files)} individual result files")
            
            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    # Extract metadata from path
                    parts = result_file.parts
                    task = None
                    model = None
                    language = None
                    
                    for i, part in enumerate(parts):
                        if part in ['question_type', 'complexity']:
                            task = part
                            if i + 1 < len(parts):
                                model = parts[i + 1]
                            if i + 2 < len(parts):
                                language = parts[i + 2]
                            break
                    
                    if not all([task, model, language]):
                        logger.warning(f"Could not parse metadata from path: {result_file}")
                        continue
                    
                    # Extract test metrics
                    test_metrics = data.get('test_metrics', {})
                    if not test_metrics:
                        logger.warning(f"No test metrics found in {result_file}")
                        continue
                    
                    # Add each metric as a row
                    for metric, value in test_metrics.items():
                        if value is not None and metric != 'primary_metric':
                            results.append({
                                'experiment_type': 'tfidf',
                                'language': language,
                                'task': task,
                                'model': model,
                                'metric': metric,
                                'value': value
                            })
                    
                except Exception as e:
                    logger.error(f"Error processing {result_file}: {e}")
        
        if results:
            if isinstance(results[0], dict):
                return pd.DataFrame(results)
            else:
                return pd.concat(results, ignore_index=True)
        else:
            logger.warning("No TF-IDF results found")
            return pd.DataFrame()
    
    def collect_probing_results(self, base_dir: str) -> pd.DataFrame:
        """Collect layer-wise probing results from experiment directories."""
        base_path = Path(base_dir)
        results = []
        
        # Look for consolidated probing results CSV
        probing_csv_files = list(base_path.glob("**/*probe_results.csv"))
        probing_csv_files.extend(list(base_path.glob("**/makeup_probe_results.csv")))
        
        if probing_csv_files:
            logger.info(f"Found {len(probing_csv_files)} probing results files")
            for csv_file in probing_csv_files:
                logger.info(f"Loading: {csv_file}")
                df = pd.read_csv(csv_file)
                results.append(df)
        else:
            logger.warning("No consolidated probing results found")
            logger.info("You may need to run the probing experiments first")
        
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_consolidated_results(self, tfidf_df: pd.DataFrame, probing_df: pd.DataFrame,
                                output_dir: str):
        """Save consolidated results to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not tfidf_df.empty:
            tfidf_file = output_path / "consolidated_tfidf_results.csv"
            tfidf_df.to_csv(tfidf_file, index=False)
            logger.info(f"Saved TF-IDF results to: {tfidf_file}")
        
        if not probing_df.empty:
            probing_file = output_path / "consolidated_probing_results.csv" 
            probing_df.to_csv(probing_file, index=False)
            logger.info(f"Saved probing results to: {probing_file}")
        
        return tfidf_file if not tfidf_df.empty else None, probing_file if not probing_df.empty else None
    
    def generate_summary_stats(self, tfidf_df: pd.DataFrame, probing_df: pd.DataFrame) -> Dict:
        """Generate summary statistics for the collected results."""
        summary = {}
        
        if not tfidf_df.empty:
            summary['tfidf'] = {
                'total_experiments': len(tfidf_df),
                'languages': sorted(tfidf_df['language'].unique().tolist()),
                'models': sorted(tfidf_df['model'].unique().tolist()),
                'tasks': sorted(tfidf_df['task'].unique().tolist()),
                'metrics': sorted(tfidf_df['metric'].unique().tolist())
            }
            
            # Best performances
            for task in tfidf_df['task'].unique():
                task_data = tfidf_df[tfidf_df['task'] == task]
                if task == 'question_type':
                    metric_data = task_data[task_data['metric'] == 'accuracy']
                    best_row = metric_data.loc[metric_data['value'].idxmax()]
                    summary['tfidf'][f'best_{task}'] = {
                        'accuracy': best_row['value'],
                        'model': best_row['model'],
                        'language': best_row['language']
                    }
                elif task == 'complexity':
                    metric_data = task_data[task_data['metric'] == 'mse']
                    best_row = metric_data.loc[metric_data['value'].idxmin()]
                    summary['tfidf'][f'best_{task}'] = {
                        'mse': best_row['value'],
                        'model': best_row['model'],
                        'language': best_row['language']
                    }
        
        if not probing_df.empty:
            summary['probing'] = {
                'total_experiments': len(probing_df),
                'languages': sorted(probing_df['language'].unique().tolist()),
                'tasks': sorted(probing_df['task'].unique().tolist()),
                'layers': sorted(probing_df['layer'].unique().tolist()) if 'layer' in probing_df.columns else [],
                'experiment_types': sorted(probing_df['experiment_type'].unique().tolist())
            }
            
            # Best layer performances
            base_experiments = probing_df[
                probing_df['experiment_type'].str.contains('base', na=False)
            ] if 'experiment_type' in probing_df.columns else probing_df
            
            for task in base_experiments['task'].unique():
                task_data = base_experiments[base_experiments['task'] == task]
                if task == 'question_type' and 'accuracy' in task_data['metric'].values:
                    metric_data = task_data[task_data['metric'] == 'accuracy']
                    best_row = metric_data.loc[metric_data['value'].idxmax()]
                    summary['probing'][f'best_{task}'] = {
                        'accuracy': best_row['value'],
                        'layer': best_row.get('layer', 'unknown'),
                        'language': best_row['language']
                    }
                elif task == 'complexity' and 'mse' in task_data['metric'].values:
                    metric_data = task_data[task_data['metric'] == 'mse']
                    best_row = metric_data.loc[metric_data['value'].idxmin()]
                    summary['probing'][f'best_{task}'] = {
                        'mse': best_row['value'],
                        'layer': best_row.get('layer', 'unknown'),
                        'language': best_row['language']
                    }
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Collect and consolidate experiment results")
    parser.add_argument("--input-dirs", nargs='+', required=True,
                       help="Input directories containing experiment results")
    parser.add_argument("--output-dir", default="./consolidated_results",
                       help="Output directory for consolidated results")
    parser.add_argument("--generate-plots", action="store_true",
                       help="Generate visualization plots after collecting results")
    
    args = parser.parse_args()
    
    collector = ResultsCollector()
    
    all_tfidf_results = []
    all_probing_results = []
    
    # Collect results from all input directories
    for input_dir in args.input_dirs:
        logger.info(f"Collecting results from: {input_dir}")
        
        tfidf_df = collector.collect_tfidf_results(input_dir)
        probing_df = collector.collect_probing_results(input_dir)
        
        if not tfidf_df.empty:
            all_tfidf_results.append(tfidf_df)
        
        if not probing_df.empty:
            all_probing_results.append(probing_df)
    
    # Consolidate all results
    final_tfidf_df = pd.concat(all_tfidf_results, ignore_index=True) if all_tfidf_results else pd.DataFrame()
    final_probing_df = pd.concat(all_probing_results, ignore_index=True) if all_probing_results else pd.DataFrame()
    
    # Save consolidated results
    tfidf_file, probing_file = collector.save_consolidated_results(
        final_tfidf_df, final_probing_df, args.output_dir
    )
    
    # Generate summary statistics
    summary = collector.generate_summary_stats(final_tfidf_df, final_probing_df)
    
    def convert_to_serializable(obj):
        """Convert numpy types to JSON serializable types."""
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    summary_file = Path(args.output_dir) / "summary_stats.json"
    with open(summary_file, 'w') as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    
    logger.info(f"Summary statistics saved to: {summary_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    if 'tfidf' in summary:
        print(f"\nTF-IDF Experiments: {summary['tfidf']['total_experiments']}")
        print(f"Languages: {', '.join(summary['tfidf']['languages'])}")
        print(f"Models: {', '.join(summary['tfidf']['models'])}")
        
        for task in ['question_type', 'complexity']:
            if f'best_{task}' in summary['tfidf']:
                best = summary['tfidf'][f'best_{task}']
                metric_name = 'accuracy' if task == 'question_type' else 'mse'
                print(f"Best {task}: {best[metric_name]:.4f} ({best['model']}, {best['language']})")
    
    if 'probing' in summary:
        print(f"\nProbing Experiments: {summary['probing']['total_experiments']}")
        print(f"Layers: {', '.join(map(str, summary['probing']['layers']))}")
        
        for task in ['question_type', 'complexity']:
            if f'best_{task}' in summary['probing']:
                best = summary['probing'][f'best_{task}']
                metric_name = 'accuracy' if task == 'question_type' else 'mse'
                print(f"Best {task}: {best[metric_name]:.4f} (Layer {best['layer']}, {best['language']})")
    
    # Generate plots if requested
    if args.generate_plots and tfidf_file and probing_file:
        logger.info("Generating visualization plots...")
        
        try:
            from scripts.visualize_results import ResultsVisualizer
            
            plot_dir = Path(args.output_dir) / "plots"
            visualizer = ResultsVisualizer(str(plot_dir))
            visualizer.generate_summary_report(str(tfidf_file), str(probing_file))
            
            logger.info(f"Plots generated in: {plot_dir}")
            
        except ImportError as e:
            logger.error(f"Could not import visualization module: {e}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()