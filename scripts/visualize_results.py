#!/usr/bin/env python3
"""
Comprehensive visualization framework for qtype-eval results.

Generates publication-quality plots for:
- TF-IDF baseline comparisons across languages and models
- Layer-wise probing accuracy/MSE by language and task
- Cross-lingual performance analysis
- Submetric control experiment results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-paper')
except OSError:
    # Fallback for older matplotlib/seaborn versions
    try:
        plt.style.use('seaborn-paper')
    except OSError:
        plt.style.use('default')

try:
    sns.set_palette("husl")
except AttributeError:
    # Fallback for newer seaborn versions
    sns.set_palette("husl", n_colors=8)

class ResultsVisualizer:
    """Main class for generating experiment visualization plots."""
    
    def __init__(self, output_dir: str = "./plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes for consistency
        self.language_colors = {
            'ar': '#FF6B6B',  # Red
            'en': '#4ECDC4',  # Teal  
            'fi': '#45B7D1',  # Blue
            'id': '#96CEB4',  # Green
            'ja': '#FFEAA7',  # Yellow
            'ko': '#DDA0DD',  # Plum
            'ru': '#98D8C8'   # Mint
        }
        
        self.task_colors = {
            'question_type': '#3498DB',  # Blue
            'complexity': '#E74C3C'      # Red
        }
        
        # Full language names for labels
        self.language_names = {
            'ar': 'Arabic',
            'en': 'English', 
            'fi': 'Finnish',
            'id': 'Indonesian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ru': 'Russian'
        }
    
    def load_tfidf_results(self, results_file: str) -> pd.DataFrame:
        """Load TF-IDF baseline results from CSV."""
        df = pd.read_csv(results_file)
        
        # Map language codes to full names
        df['language_name'] = df['language'].map(self.language_names)
        
        # Ensure we have the primary metrics
        if 'accuracy' in df['metric'].values and 'mse' in df['metric'].values:
            return df
        else:
            raise ValueError("Results file must contain 'accuracy' and 'mse' metrics")
    
    def load_probing_results(self, results_file: str) -> pd.DataFrame:
        """Load layer-wise probing results from CSV."""
        df = pd.read_csv(results_file)
        
        # Map language codes to full names
        df['language_name'] = df['language'].map(self.language_names)
        
        # Parse layer information if needed
        if 'layer' not in df.columns and 'model_layer' in df.columns:
            df['layer'] = df['model_layer'].str.extract(r'layer_(\d+)').astype(int)
        
        return df
    
    def plot_tfidf_baselines(self, df: pd.DataFrame, save_prefix: str = "tfidf_baselines"):
        """Generate comprehensive TF-IDF baseline comparison plots."""
        
        # 1. Classification accuracy by model and language
        classification_df = df[
            (df['task'] == 'question_type') & 
            (df['metric'] == 'accuracy')
        ].copy()
        
        if not classification_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Create heatmap
            pivot_df = classification_df.pivot(
                index='model', 
                columns='language_name', 
                values='value'
            )
            
            sns.heatmap(
                pivot_df, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                ax=ax,
                cbar_kws={'label': 'Accuracy'}
            )
            
            ax.set_title('TF-IDF Classification Accuracy by Model and Language', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Language', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_classification_heatmap.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f"{save_prefix}_classification_heatmap.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar plot comparison
            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            
            sns.barplot(
                data=classification_df,
                x='language_name',
                y='value',
                hue='model',
                ax=ax
            )
            
            ax.set_title('TF-IDF Classification Accuracy Comparison', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Language', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_classification_bars.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f"{save_prefix}_classification_bars.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Regression MSE by model and language
        regression_df = df[
            (df['task'] == 'complexity') & 
            (df['metric'] == 'mse')
        ].copy()
        
        if not regression_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Create heatmap
            pivot_df = regression_df.pivot(
                index='model', 
                columns='language_name', 
                values='value'
            )
            
            sns.heatmap(
                pivot_df, 
                annot=True, 
                fmt='.4f', 
                cmap='RdYlBu',  # Reversed for MSE (lower is better)
                ax=ax,
                cbar_kws={'label': 'MSE'}
            )
            
            ax.set_title('TF-IDF Regression MSE by Model and Language', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Language', fontsize=12)
            ax.set_ylabel('Model', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_regression_heatmap.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f"{save_prefix}_regression_heatmap.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_layer_wise_probing(self, df: pd.DataFrame, save_prefix: str = "layer_probing"):
        """Generate layer-wise probing analysis plots."""
        
        # 1. Accuracy by layer and language (classification)
        classification_df = df[
            (df['task'] == 'question_type') & 
            (df['metric'] == 'accuracy') &
            (df['experiment_type'].str.contains('base', na=False))
        ].copy()
        
        if not classification_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            for lang in sorted(classification_df['language'].unique()):
                lang_data = classification_df[classification_df['language'] == lang]
                
                ax.plot(
                    lang_data['layer'], 
                    lang_data['value'],
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    label=self.language_names[lang],
                    color=self.language_colors[lang]
                )
            
            ax.set_title('Question-Type Classification Accuracy by Layer', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 13))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_classification_by_layer.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f"{save_prefix}_classification_by_layer.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. MSE by layer and language (regression)
        regression_df = df[
            (df['task'] == 'complexity') & 
            (df['metric'] == 'mse') &
            (df['experiment_type'].str.contains('base', na=False))
        ].copy()
        
        if not regression_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            
            for lang in sorted(regression_df['language'].unique()):
                lang_data = regression_df[regression_df['language'] == lang]
                
                ax.plot(
                    lang_data['layer'], 
                    lang_data['value'],
                    marker='s',
                    linewidth=2,
                    markersize=6,
                    label=self.language_names[lang],
                    color=self.language_colors[lang]
                )
            
            ax.set_title('Complexity Regression MSE by Layer', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('MSE', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 13))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_regression_by_layer.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f"{save_prefix}_regression_by_layer.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Heatmap: Layer vs Language performance
        if not classification_df.empty:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Classification heatmap
            pivot_df = classification_df.pivot(
                index='layer', 
                columns='language_name', 
                values='value'
            )
            
            sns.heatmap(
                pivot_df, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                ax=ax1,
                cbar_kws={'label': 'Accuracy'}
            )
            
            ax1.set_title('Classification Accuracy by Layer and Language', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Language', fontsize=12)
            ax1.set_ylabel('Layer', fontsize=12)
            
            # Regression heatmap
            if not regression_df.empty:
                pivot_df = regression_df.pivot(
                    index='layer', 
                    columns='language_name', 
                    values='value'
                )
                
                sns.heatmap(
                    pivot_df, 
                    annot=True, 
                    fmt='.4f', 
                    cmap='RdYlBu',
                    ax=ax2,
                    cbar_kws={'label': 'MSE'}
                )
                
                ax2.set_title('Regression MSE by Layer and Language', 
                             fontsize=14, fontweight='bold')
                ax2.set_xlabel('Language', fontsize=12)
                ax2.set_ylabel('Layer', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_heatmaps.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f"{save_prefix}_heatmaps.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_submetric_analysis(self, df: pd.DataFrame, save_prefix: str = "submetric_analysis"):
        """Generate submetric control experiment analysis."""
        
        # Filter for submetric control experiments
        submetric_df = df[
            df['experiment_type'].str.contains('control_', na=False) &
            ~df['experiment_type'].str.contains('question_type|complexity', na=False)
        ].copy()
        
        if submetric_df.empty:
            print("No submetric control experiments found")
            return
        
        # Extract submetric names
        submetric_df['submetric'] = submetric_df['experiment_type'].str.extract(
            r'control_([^_]+)_seed'
        )
        
        # Focus on best-performing layers (6-10)
        mid_layer_df = submetric_df[
            submetric_df['layer'].between(6, 10)
        ].copy()
        
        # Classification results
        classification_df = mid_layer_df[
            (mid_layer_df['task'] == 'question_type') & 
            (mid_layer_df['metric'] == 'accuracy')
        ]
        
        if not classification_df.empty:
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            
            # Create boxplot for submetrics across languages
            try:
                sns.boxplot(
                    data=classification_df,
                    x='submetric',
                    y='value',
                    hue='language_name',
                    ax=ax
                )
            except (UnboundLocalError, AttributeError):
                # Fallback for seaborn compatibility issues
                for i, lang in enumerate(classification_df['language_name'].unique()):
                    lang_data = classification_df[classification_df['language_name'] == lang]
                    positions = [j + i*0.1 for j in range(len(lang_data['submetric'].unique()))]
                    ax.boxplot([lang_data[lang_data['submetric'] == sm]['value'].values 
                               for sm in lang_data['submetric'].unique()],
                              positions=positions,
                              widths=0.08,
                              patch_artist=True,
                              label=lang)
            
            ax.set_title('Submetric Control Experiments: Classification Accuracy\n(Layers 6-10)', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Submetric', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.legend(title='Language', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{save_prefix}_classification_boxplot.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.output_dir / f"{save_prefix}_classification_boxplot.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_cross_lingual_comparison(self, tfidf_df: pd.DataFrame, 
                                    probing_df: pd.DataFrame, 
                                    save_prefix: str = "cross_lingual"):
        """Generate cross-lingual performance comparison plots."""
        
        # Get best TF-IDF performance per language/task
        tfidf_best = []
        for (lang, task), group in tfidf_df.groupby(['language', 'task']):
            if task == 'question_type':
                metric_data = group[group['metric'] == 'accuracy']
                best_row = metric_data.loc[metric_data['value'].idxmax()]
            else:  # complexity
                metric_data = group[group['metric'] == 'mse']
                best_row = metric_data.loc[metric_data['value'].idxmin()]
            
            tfidf_best.append({
                'language': lang,
                'task': task,
                'model_type': 'TF-IDF',
                'value': best_row['value'],
                'model_name': best_row['model']
            })
        
        tfidf_best_df = pd.DataFrame(tfidf_best)
        
        # Get best probing performance per language/task (typically middle layers)
        probing_best = []
        probing_base = probing_df[
            probing_df['experiment_type'].str.contains('base', na=False)
        ]
        
        for (lang, task), group in probing_base.groupby(['language', 'task']):
            if task == 'question_type':
                metric_data = group[group['metric'] == 'accuracy']
                best_row = metric_data.loc[metric_data['value'].idxmax()]
            else:  # complexity
                metric_data = group[group['metric'] == 'mse']
                best_row = metric_data.loc[metric_data['value'].idxmin()]
            
            probing_best.append({
                'language': lang,
                'task': task,
                'model_type': 'Probing',
                'value': best_row['value'],
                'layer': best_row['layer']
            })
        
        probing_best_df = pd.DataFrame(probing_best)
        
        # Combine for comparison
        comparison_data = []
        
        for _, tfidf_row in tfidf_best_df.iterrows():
            probing_row = probing_best_df[
                (probing_best_df['language'] == tfidf_row['language']) &
                (probing_best_df['task'] == tfidf_row['task'])
            ]
            
            if not probing_row.empty:
                comparison_data.append({
                    'language': tfidf_row['language'],
                    'task': tfidf_row['task'],
                    'tfidf_performance': tfidf_row['value'],
                    'probing_performance': probing_row.iloc[0]['value'],
                    'best_layer': probing_row.iloc[0]['layer']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Classification comparison
            classification_comp = comparison_df[
                comparison_df['task'] == 'question_type'
            ].copy()
            
            if not classification_comp.empty:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                x_pos = np.arange(len(classification_comp))
                width = 0.35
                
                ax.bar(x_pos - width/2, classification_comp['tfidf_performance'], 
                      width, label='TF-IDF (Best)', alpha=0.8, color='#3498DB')
                ax.bar(x_pos + width/2, classification_comp['probing_performance'], 
                      width, label='Probing (Best Layer)', alpha=0.8, color='#E74C3C')
                
                ax.set_title('Classification Performance: TF-IDF vs Probing', 
                            fontsize=16, fontweight='bold')
                ax.set_xlabel('Language', fontsize=12)
                ax.set_ylabel('Accuracy', fontsize=12)
                ax.set_xticks(x_pos)
                ax.set_xticklabels([self.language_names[lang] for lang in classification_comp['language']])
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (tfidf_val, prob_val) in enumerate(zip(
                    classification_comp['tfidf_performance'], 
                    classification_comp['probing_performance'])):
                    ax.text(i - width/2, tfidf_val + 0.01, f'{tfidf_val:.3f}', 
                           ha='center', va='bottom', fontsize=9)
                    ax.text(i + width/2, prob_val + 0.01, f'{prob_val:.3f}', 
                           ha='center', va='bottom', fontsize=9)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f"{save_prefix}_classification_comparison.pdf", 
                           dpi=300, bbox_inches='tight')
                plt.savefig(self.output_dir / f"{save_prefix}_classification_comparison.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_summary_report(self, tfidf_file: str, probing_file: str):
        """Generate a comprehensive summary report."""
        
        # Load data
        tfidf_df = self.load_tfidf_results(tfidf_file)
        probing_df = self.load_probing_results(probing_file)
        
        print("Generating comprehensive visualization suite...")
        
        # Generate all plot types
        self.plot_tfidf_baselines(tfidf_df)
        self.plot_layer_wise_probing(probing_df)
        self.plot_submetric_analysis(probing_df)
        self.plot_cross_lingual_comparison(tfidf_df, probing_df)
        
        # Create index HTML file
        self.create_html_index()
        
        print(f"All plots saved to: {self.output_dir}")
        print("Open index.html to view all results")
    
    def create_html_index(self):
        """Create an HTML index file to view all generated plots."""
        
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>QType-Eval Results Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .section { margin-bottom: 40px; }
        .plot { margin: 20px 0; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <h1>QType-Eval Results Visualization</h1>
    
    <div class="section">
        <h2>TF-IDF Baseline Results</h2>
        <div class="plot">
            <h3>Classification Accuracy Heatmap</h3>
            <img src="tfidf_baselines_classification_heatmap.png" alt="Classification Heatmap">
        </div>
        <div class="plot">
            <h3>Classification Accuracy Comparison</h3>
            <img src="tfidf_baselines_classification_bars.png" alt="Classification Bars">
        </div>
        <div class="plot">
            <h3>Regression MSE Heatmap</h3>
            <img src="tfidf_baselines_regression_heatmap.png" alt="Regression Heatmap">
        </div>
    </div>
    
    <div class="section">
        <h2>Layer-wise Probing Results</h2>
        <div class="plot">
            <h3>Classification Accuracy by Layer</h3>
            <img src="layer_probing_classification_by_layer.png" alt="Classification by Layer">
        </div>
        <div class="plot">
            <h3>Regression MSE by Layer</h3>
            <img src="layer_probing_regression_by_layer.png" alt="Regression by Layer">
        </div>
        <div class="plot">
            <h3>Performance Heatmaps</h3>
            <img src="layer_probing_heatmaps.png" alt="Performance Heatmaps">
        </div>
    </div>
    
    <div class="section">
        <h2>Submetric Analysis</h2>
        <div class="plot">
            <h3>Submetric Control Experiments</h3>
            <img src="submetric_analysis_classification_boxplot.png" alt="Submetric Analysis">
        </div>
    </div>
    
    <div class="section">
        <h2>Cross-lingual Comparison</h2>
        <div class="plot">
            <h3>TF-IDF vs Probing Performance</h3>
            <img src="cross_lingual_classification_comparison.png" alt="Cross-lingual Comparison">
        </div>
    </div>
    
</body>
</html>
        """
        
        with open(self.output_dir / "index.html", "w") as f:
            f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive visualization suite for qtype-eval results")
    parser.add_argument("--tfidf-results", required=True, help="Path to TF-IDF results CSV")
    parser.add_argument("--probing-results", required=True, help="Path to probing results CSV") 
    parser.add_argument("--output-dir", default="./plots", help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create visualizer and generate plots
    visualizer = ResultsVisualizer(args.output_dir)
    visualizer.generate_summary_report(args.tfidf_results, args.probing_results)

if __name__ == "__main__":
    main()