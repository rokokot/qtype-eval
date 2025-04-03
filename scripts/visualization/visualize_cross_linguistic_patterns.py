#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Define language codes and names
LANGUAGES = ["ar", "en", "fi", "id", "ja", "ko", "ru"]
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "en": "English",
    "fi": "Finnish",
    "id": "Indonesian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
}

# Define complexity metrics
COMPLEXITY_METRICS = [
    "avg_links_len",
    "avg_max_depth",
    "avg_subordinate_chain_len",
    "avg_verb_edges",
    "lexical_density",
    "n_tokens",
]


def load_qtc_dataset(split="train", cache_dir="./data/cache"):
    """Load the QTC dataset from HuggingFace."""
    logger.info(f"Loading {split} split of QTC dataset")
    dataset = load_dataset("rokokot/question-type-and-complexity", name="base", split=split, cache_dir=cache_dir)
    return dataset.to_pandas()


def visualize_complexity_distribution(df, output_dir):
    """Visualize the distribution of complexity across languages."""
    os.makedirs(output_dir, exist_ok=True)

    # Overall complexity distribution by language
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=df, x="language", y="lang_norm_complexity_score", order=LANGUAGES)
    plt.title("Distribution of Complexity Scores by Language", fontsize=16)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Normalized Complexity Score", fontsize=14)
    plt.xticks(range(len(LANGUAGES)), [LANGUAGE_NAMES[lang] for lang in LANGUAGES], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complexity_distribution_by_language.png"), dpi=300)
    plt.close()

    # Complexity by question type and language
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df, x="language", y="lang_norm_complexity_score", hue="question_type", order=LANGUAGES)
    plt.title("Complexity Scores by Question Type and Language", fontsize=16)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Normalized Complexity Score", fontsize=14)
    plt.xticks(range(len(LANGUAGES)), [LANGUAGE_NAMES[lang] for lang in LANGUAGES], rotation=45)
    plt.legend(title="Question Type", labels=["Content", "Polar"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complexity_by_type_and_language.png"), dpi=300)
    plt.close()

    # Individual complexity metrics by language
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
    axes = axes.flatten()

    for i, metric in enumerate(COMPLEXITY_METRICS):
        sns.boxplot(data=df, x="language", y=metric, order=LANGUAGES, ax=axes[i])
        axes[i].set_title(f"{metric} by Language", fontsize=14)
        axes[i].set_xlabel("Language", fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].tick_params(axis="x", rotation=45)

    plt.suptitle("Individual Complexity Metrics by Language", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "individual_metrics_by_language.png"), dpi=300)
    plt.close()

    # Complexity feature correlations
    corr_matrices = {}
    for lang in LANGUAGES:
        lang_df = df[df["language"] == lang]
        corr_matrices[lang] = lang_df[COMPLEXITY_METRICS].corr()

    # Plot correlation matrices
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, (lang, corr_matrix) in enumerate(corr_matrices.items()):
        if i < len(axes):
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[i], fmt=".2f", square=True)
            axes[i].set_title(f"{LANGUAGE_NAMES[lang]}", fontsize=14)

    # Remove extra subplot
    if len(corr_matrices) < len(axes):
        fig.delaxes(axes[-1])

    plt.suptitle("Correlation Between Complexity Metrics by Language", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complexity_correlation_by_language.png"), dpi=300)
    plt.close()


def visualize_question_type_patterns(df, output_dir):
    """Visualize patterns in question types across languages."""
    os.makedirs(output_dir, exist_ok=True)

    # Question type distribution by language
    plt.figure(figsize=(14, 8))

    # Count question types by language
    type_counts = pd.crosstab(df["language"], df["question_type"])
    type_counts.columns = ["Content", "Polar"]
    type_counts.index = [LANGUAGE_NAMES[lang] for lang in type_counts.index]

    # Calculate percentages
    type_percentages = type_counts.div(type_counts.sum(axis=1), axis=0) * 100

    # Plot as stacked bar chart
    type_percentages.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title("Question Type Distribution by Language", fontsize=16)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Percentage", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title="Question Type")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question_type_distribution.png"), dpi=300)
    plt.close()

    # Token length by question type and language
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x="language", y="n_tokens", hue="question_type", order=LANGUAGES)
    plt.title("Question Length by Question Type and Language", fontsize=16)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Number of Tokens", fontsize=14)
    plt.xticks(range(len(LANGUAGES)), [LANGUAGE_NAMES[lang] for lang in LANGUAGES], rotation=45)
    plt.legend(title="Question Type", labels=["Content", "Polar"])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "question_length_by_type_and_language.png"), dpi=300)
    plt.close()

    # Question type and complexity correlations
    complexity_correlations = []

    for lang in LANGUAGES:
        lang_df = df[df["language"] == lang]
        corr = lang_df["question_type"].corr(lang_df["lang_norm_complexity_score"])
        complexity_correlations.append({"language": LANGUAGE_NAMES[lang], "correlation": corr})

    corr_df = pd.DataFrame(complexity_correlations)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=corr_df, x="language", y="correlation")
    plt.title("Correlation Between Question Type and Complexity by Language", fontsize=16)
    plt.xlabel("Language", fontsize=14)
    plt.ylabel("Correlation Coefficient", fontsize=14)
    plt.xticks(rotation=45)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "type_complexity_correlation.png"), dpi=300)
    plt.close()


def visualize_feature_space(df, output_dir, method="tsne", n_samples=5000):
    """
    Visualize the complexity feature space using dimensionality reduction.

    Args:
        df: DataFrame with complexity features
        output_dir: Output directory for visualizations
        method: Dimensionality reduction method ('tsne' or 'umap')
        n_samples: Number of samples to use for visualization
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample data if necessary
    if len(df) > n_samples:
        sampled_df = (
            df.groupby("language")
            .apply(lambda x: x.sample(min(len(x), n_samples // len(LANGUAGES)), random_state=42))
            .reset_index(drop=True)
        )
    else:
        sampled_df = df.copy()

    logger.info(f"Using {len(sampled_df)} samples for feature space visualization")

    # Standardize features
    features = sampled_df[COMPLEXITY_METRICS].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply dimensionality reduction
    if method.lower() == "tsne":
        logger.info("Running t-SNE dimensionality reduction")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embedding = reducer.fit_transform(scaled_features)
        method_name = "t-SNE"
    elif method.lower() == "umap":
        try:
            logger.info("Running UMAP dimensionality reduction")
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(scaled_features)
            method_name = "UMAP"
        except ImportError:
            logger.warning("UMAP not installed. Falling back to t-SNE.")
            reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
            embedding = reducer.fit_transform(scaled_features)
            method_name = "t-SNE"
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    # Add embeddings to DataFrame
    sampled_df["x"] = embedding[:, 0]
    sampled_df["y"] = embedding[:, 1]
    sampled_df["language_name"] = sampled_df["language"].map(LANGUAGE_NAMES)

    # Visualize by language
    plt.figure(figsize=(14, 10))
    for lang in LANGUAGES:
        lang_points = sampled_df[sampled_df["language"] == lang]
        plt.scatter(lang_points["x"], lang_points["y"], label=LANGUAGE_NAMES[lang], alpha=0.7, s=50)

    plt.title(f"{method_name} Visualization of Complexity Features by Language", fontsize=16)
    plt.xlabel(f"{method_name} Dimension 1", fontsize=14)
    plt.ylabel(f"{method_name} Dimension 2", fontsize=14)
    plt.legend(title="Language", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method.lower()}_by_language.png"), dpi=300)
    plt.close()

    # Visualize by question type
    plt.figure(figsize=(14, 10))
    for qtype, name in [(0, "Content"), (1, "Polar")]:
        type_points = sampled_df[sampled_df["question_type"] == qtype]
        plt.scatter(type_points["x"], type_points["y"], label=name, alpha=0.7, s=50)

    plt.title(f"{method_name} Visualization of Complexity Features by Question Type", fontsize=16)
    plt.xlabel(f"{method_name} Dimension 1", fontsize=14)
    plt.ylabel(f"{method_name} Dimension 2", fontsize=14)
    plt.legend(title="Question Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method.lower()}_by_question_type.png"), dpi=300)
    plt.close()

    # Visualize by complexity score
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(
        sampled_df["x"], sampled_df["y"], c=sampled_df["lang_norm_complexity_score"], cmap="viridis", alpha=0.8, s=50
    )
    plt.colorbar(scatter, label="Complexity Score")
    plt.title(f"{method_name} Visualization of Complexity Features by Complexity Score", fontsize=16)
    plt.xlabel(f"{method_name} Dimension 1", fontsize=14)
    plt.ylabel(f"{method_name} Dimension 2", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method.lower()}_by_complexity.png"), dpi=300)
    plt.close()

    # Create separate plots for each language colored by question type
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    for i, lang in enumerate(LANGUAGES):
        if i < len(axes):
            lang_points = sampled_df[sampled_df["language"] == lang]

            for qtype, name, color in [(0, "Content", "blue"), (1, "Polar", "orange")]:
                type_points = lang_points[lang_points["question_type"] == qtype]
                axes[i].scatter(type_points["x"], type_points["y"], label=name, alpha=0.7, s=50, color=color)

            axes[i].set_title(f"{LANGUAGE_NAMES[lang]}", fontsize=14)
            axes[i].set_xlabel(f"{method_name} Dimension 1", fontsize=12)
            axes[i].set_ylabel(f"{method_name} Dimension 2", fontsize=12)
            axes[i].legend(title="Question Type")

    # Remove extra subplot
    if len(LANGUAGES) < 8:
        fig.delaxes(axes[7])

    plt.suptitle(f"Question Type Patterns by Language ({method_name})", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{method.lower()}_by_language_and_type.png"), dpi=300)
    plt.close()


def main(args):
    """Main function to run visualizations."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    df = load_qtc_dataset(args.split, args.cache_dir)
    logger.info(f"Loaded {len(df)} examples from {args.split} split")

    # Run visualizations
    logger.info("Generating complexity distribution visualizations")
    visualize_complexity_distribution(df, os.path.join(args.output_dir, "complexity_distribution"))

    logger.info("Generating question type pattern visualizations")
    visualize_question_type_patterns(df, os.path.join(args.output_dir, "question_types"))

    logger.info("Generating feature space visualizations")
    visualize_feature_space(df, os.path.join(args.output_dir, "feature_space"), args.dim_reduction, args.n_samples)

    logger.info(f"All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cross-linguistic patterns in the QTC dataset")
    parser.add_argument(
        "--output-dir", type=str, default="./visualizations", help="Output directory for visualizations"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "validation", "test"], help="Dataset split to visualize"
    )
    parser.add_argument("--cache-dir", type=str, default="./data/cache", help="Cache directory for datasets")
    parser.add_argument(
        "--dim-reduction", type=str, default="tsne", choices=["tsne", "umap"], help="Dimensionality reduction method"
    )
    parser.add_argument(
        "--n-samples", type=int, default=5000, help="Number of samples to use for feature space visualization"
    )
    args = parser.parse_args()

    main(args)
