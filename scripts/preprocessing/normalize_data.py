import pandas as pd
import numpy as np
import os
import glob
import argparse
from collections import defaultdict
from pathlib import Path
from datetime import datetime

def normalize_scores(df, language_column='language'):
    """
    This script processes the .csv files of the custom profiling-UD output. We make use of the following normalization strategies:
    - language centering for avg_links_len, avg_max_depth, and verbal_head_per_sent
    - log normalization for subordinate_proposition_dist and n_tokens
    - min-max scaling for avg_verb_edges
    - raw features for avg_subordinate_chain_len and lexical_density

    The script takes the following arguments:
      df: DataFrame containing our original csv files
      lang column: Language of the file
    
    to run the script with the following goals:
    > normalize data in csv files: python normalize.py --input "data/*.csv" --output "data/normalized_output_file"
    > combine already normalized files by language:  python normalize.py --combine --tydi-dir "data/normalized_tydi" --ud-dir "data/normalized_ud" --output-combined "data/scores/language_files"   
    """
    normalized_df = df.copy()
    language_groups = df.groupby(language_column)

    for feature in ['avg_links_len', 'avg_max_depth', 'verbal_head_per_sent']:
        language_means = language_groups[feature].mean()
        
        for language, group in language_groups:
            language_mean = language_means[language]
            mask = normalized_df[language_column] == language
            normalized_df.loc[mask, feature] = df.loc[mask, feature] - language_mean

    # Log normalization
    for feature in ['subordinate_proposition_dist', 'n_tokens']:
        normalized_df[feature] = np.log1p(df[feature])

    # Min-max scaling per language
    for feature in ['avg_verb_edges']:
        for language, group in language_groups:
            mask = normalized_df[language_column] == language
            min_val = df.loc[mask, feature].min()
            max_val = df.loc[mask, feature].max()
            range_val = max_val - min_val

            if range_val > 0:
                normalized_df.loc[mask, feature] = (df.loc[mask, feature] - min_val) / range_val
            else:
                normalized_df.loc[mask, feature] = 0

    return normalized_df
  

def token_sampling(df, target_size, token_column='n_tokens', random_state=69):
    "Downsample data based on token count distribution"

    if len(df) <= target_size:
        return df
    
    df_with_bins = df.copy()
    
    df_with_bins['token_bin'] = pd.qcut(df_with_bins[token_column],min(10, len(df_with_bins[token_column].unique())),                               labels=False,duplicates='drop')

    sampled_df = pd.DataFrame()

    for bin_id in df_with_bins['token_bin'].unique():
        bin_df = df_with_bins[df_with_bins['token_bin'] == bin_id]

        bin_proportion = len(bin_df) / len(df_with_bins)
        bin_samples = max(1, int(target_size * bin_proportion))

        if len(bin_df) > bin_samples:
            bin_sampled = bin_df.sample(n=bin_samples, random_state=random_state)
        else:
            bin_sampled = bin_df
            
        sampled_df = pd.concat([sampled_df, bin_sampled])
    
    if len(sampled_df) > target_size:
        sampled_df = sampled_df.sample(n=target_size, random_state=random_state)
    
    if 'token_bin' in sampled_df.columns:
        sampled_df = sampled_df.drop('token_bin', axis=1)
    
    return sampled_df
  

def convert_language_names(df):
    """ convert full language names to ISO codes for easier preprocessing"""
    language_codes = {
        'arabic': 'ar',
        'english': 'en',
        'finnish': 'fi',
        'korean': 'ko',
        'japanese': 'ja',
        'indonesian': 'id',
        'russian': 'ru'
    }
    
    df_copy = df.copy()
    
    if 'language' in df_copy.columns:
        df_copy['language'] = df_copy['language'].apply(
            lambda x: language_codes.get(x.lower(), x) if isinstance(x, str) else x
        )
    
    return df_copy
  

def convert_types(df):
    """
    we represent question types as numeric values:
    - 0 for content questions
    - 1 for polar questions

    this way the model can be understood as answering the question 'is this text a polar question?'
    """
    df_copy = df.copy()
    
    if 'type' in df_copy.columns:
        df_copy['question_type'] = df_copy['type'].apply(
            lambda x: 1 if x.lower() == 'polar' else 0 if x.lower() == 'content' else x
        )
        
        df_copy.rename(columns={'type': 'type_original'}, inplace=True) # decide whether to keep the original type column
    
    return df_copy


def format_decimals(df, decimal_places=3):
    """set float columns to specified number of decimal places"""
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=['float']).columns:
        df_copy[col] = df_copy[col].round(decimal_places)
    return df_copy



def normalize_data(input_patterns, output_dir, dataset_source=None, language_filter=None):
    """normalize data files, with output for each language and combined output"""
    
    # Group files by language
    language_files = defaultdict(list)
    
    if isinstance(input_patterns, str):
        input_patterns = [input_patterns]

    for pattern in input_patterns:
        files = glob.glob(pattern)
        if not files:
            print(f"No files match pattern: {pattern}")
            continue
                
        for file_path in files:
            try:
                file_name = os.path.basename(file_path)
                parts = file_name.replace('.csv', '').split('_')
                language = parts[0] if len(parts) > 0 else None
                
                if language_filter and language.lower() != language_filter.lower():
                    continue
                    
                language_files[language].append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    normalized_dir = os.path.join(output_dir, "normalized")
    normalized_by_lang_dir = os.path.join(normalized_dir, "by_language")
    os.makedirs(normalized_by_lang_dir, exist_ok=True)
    
    all_normalized_dfs = []
    language_specific_dfs = {}
    
    for language, file_paths in language_files.items():
        print(f"Processing language: {language}")
        language_dfs = []
        
        for file_path in file_paths:
            try:
                file_name = os.path.basename(file_path)
                print(f"  Reading {file_name}")
                
                parts = file_name.replace('.csv', '').split('_')
                q_type = parts[1] if len(parts) > 1 else None
                
                df = pd.read_csv(file_path)
                
                if 'language' not in df.columns:
                    df['language'] = language
                if 'type' not in df.columns and q_type:
                    df['type'] = q_type
                
                language_dfs.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if language_dfs:
            combined_language_df = pd.concat(language_dfs, ignore_index=True)
            combined_language_df = convert_language_names(combined_language_df)
            combined_language_df = convert_types(combined_language_df)
            
            normalized_language_df = normalize_scores(combined_language_df)
            
            all_normalized_dfs.append(normalized_language_df)
            
            language_specific_dfs[language] = normalized_language_df
            
            normalized_language_df = format_decimals(normalized_language_df)
            
            lang_output_file = os.path.join(normalized_by_lang_dir, f"{dataset_source}_{language}_normalized.csv")
            normalized_language_df.to_csv(lang_output_file, index=False)
            print(f"  Saved normalized data for {language} to {lang_output_file}")
    
    if not all_normalized_dfs:
        print("No valid data files found.")
        return None, {}
        
    final_normalized_df = pd.concat(all_normalized_dfs, ignore_index=True)
    
    final_normalized_df = format_decimals(final_normalized_df)
    
    combined_output_file = os.path.join(normalized_dir, f"{dataset_source}_all_languages_normalized.csv")
    final_normalized_df.to_csv(combined_output_file, index=False)
    print(f"Saved combined normalized data to {combined_output_file}")
    
    return final_normalized_df, language_specific_dfs


def downsample_dataset(df, target_sizes, dataset_name):
    """
    Downsample the dataset based on specified target sizes per language and question type
    Returns a new dataframe with the downsampled data and a descriptive name
    """
    downsampled_dfs = []
    
    print("\nDownsampling by language and question type:")
    print("--------------------------------------------")
    
    for language in df['language'].unique():
        for q_type in df['type_original'].unique():
            subset = df[(df['language'] == language) & (df['type_original'] == q_type)]
            
            if len(subset) == 0:
                continue
                
            # set target size using fallback logic
            target = (target_sizes.get(language, {}).get(q_type.lower()) or 
                      target_sizes.get(language, {}).get('default') or 
                      target_sizes.get('default', {}).get(q_type.lower()) or 
                      target_sizes.get('default', {}).get('default') or 
                      len(subset))
            
            target = min(target, len(subset))
            print(f"  {language} {q_type}: {len(subset)} → {target}")
            
            # token based sampling we defined above
            sampled = token_sampling(subset, target)
            downsampled_dfs.append(sampled)

    if not downsampled_dfs:
        return df, dataset_name
    
    result = pd.concat(downsampled_dfs, ignore_index=True)
    print(f"Total after downsampling: {len(result)} (from {len(df)})")
    
    return result, f"{dataset_name}_downsampled"


def sum_scores(df, excluded_feature=None, dataset_name=None):
    """
    Function which adds all of the individual feature scores together, and normalizes the final result to [-1,1] value.
    
    excluded_feature : str, optional
        Feature to exclude when calculating the complexity score for ablation datasets
    """

    feature_columns = [
        'avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len',
        'avg_verb_edges', 'lexical_density', 'n_tokens',
        'subordinate_proposition_dist', 'verbal_head_per_sent'
    ]
    
    result_name = dataset_name or "dataset"
    
    if excluded_feature and excluded_feature in feature_columns:
        feature_columns.remove(excluded_feature)
        result_name = f"{result_name}_no_{excluded_feature}"
        print(f"Excluding feature: {excluded_feature}")
    else:
        result_name = f"{result_name}_with_complexity"
    
    result_df = df.copy()
    
    existing_features = [col for col in feature_columns if col in result_df.columns]
    
    if existing_features:

        result_df['complexity_score'] = result_df[existing_features].sum(axis=1)
        
        mean = result_df['complexity_score'].mean()
        std = result_df['complexity_score'].std()
        if std > 0: 
            result_df['complexity_score'] = (result_df['complexity_score'] - mean) / std
    else:
        print("Warning: No feature columns found to calculate complexity score")
    
    return result_df, result_name


def main():
    parser = argparse.ArgumentParser(description='Normalize and downsample linguistic complexity data.')
    parser.add_argument('--input', '-i', required=True, nargs='+', 
                        help='Input file patterns (e.g., "data/*.csv" or "data/en_*.csv data/ja_*.csv")')
    parser.add_argument('--output', '-o', required=True, 
                        help='Output directory for processed files')
    parser.add_argument('--dataset-source','-ds', required=True, choices=['tydi', 'ud'], help='source of the original data: tydi or ud')
    parser.add_argument('--language', '-l', 
                        help='Only process data for a specific language')
    parser.add_argument('--no-downsample', action='store_true',
                        help='Skip downsampling step')
    parser.add_argument('--ablation', '-a', action='store_true',
                        help='Create ablation datasets by excluding one feature at a time')
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    dataset_output_dir = os.path.join(args.output, f'{args.dataset_source}_{timestamp}')
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.dataset_source == 'ud':
        args.no_downsample = True



    print("\nProcessing and normalizing data...")
    normalized_df, language_dfs = normalize_data(args.input, dataset_output_dir, args.dataset_source, args.language)
    
    if normalized_df is None or len(normalized_df) == 0:
        print("No data to process. Exiting.")
        return
    
    stats = {}
    
    if not args.no_downsample:
        print("\nDownsampling data...")
        # Define target sizes for each language and question type
        target_sizes = {
            'ko': {'polar': 380, 'content': 400},
            'id': {'polar': 474, 'content': 600},
            'ar': {'polar': 500, 'content': 500},  
            'en': {'polar': 600, 'content': 600},
            'fi': {'polar': 600, 'content': 600},
            'ja': {'polar': 600, 'content': 600},  
            'ru': {'polar': 600, 'content': 600},
            'default': {'polar': 500, 'content': 500}  # Default for any unlisted languages
        }
        
        downsampled_dir = os.path.join(dataset_output_dir, "downsampled")
        downsampled_by_lang_dir = os.path.join(downsampled_dir, "by_language")
        os.makedirs(downsampled_by_lang_dir, exist_ok=True)

        # Downsample the combined dataset
        downsampled_df, _ = downsample_dataset(normalized_df, target_sizes, "all_languages")
        
        downsampled_language_dfs = {}
        for lang, lang_df in language_dfs.items():
            lang_downsampled, _ = downsample_dataset(lang_df, target_sizes, lang)
            downsampled_language_dfs[lang] = lang_downsampled
            
            lang_downsampled = format_decimals(lang_downsampled)
            lang_output_file = os.path.join(downsampled_by_lang_dir, f"{args.dataset_source}_{lang}_downsampled.csv")
            lang_downsampled.to_csv(lang_output_file, index=False)
            print(f"Saved downsampled data for {lang} to {lang_output_file}")
        
        downsampled_df = format_decimals(downsampled_df)
        combined_output_file = os.path.join(downsampled_dir, f"{args.dataset_source}_all_languages_downsampled.csv")
        downsampled_df.to_csv(combined_output_file, index=False)
        print(f"Saved combined downsampled data to {combined_output_file}")
        
        final_df = downsampled_df
        final_language_dfs = downsampled_language_dfs

    else:
        print("Skipping downsampling step.")
        final_df = normalized_df
        final_language_dfs = language_dfs
    
    print("\nCalculating complexity scores...")
    complexity_dir = os.path.join(dataset_output_dir, "complexity")
    complexity_by_lang_dir = os.path.join(complexity_dir, "by_language")
    os.makedirs(complexity_by_lang_dir, exist_ok=True)
    
    scored_df, _ = sum_scores(final_df.copy(), dataset_name="all_languages")
    
    scored_df = format_decimals(scored_df)
    scored_output_file = os.path.join(complexity_dir, f"{args.dataset_source}_all_languages_complexity.csv")
    scored_df.to_csv(scored_output_file, index=False)
    print(f"Saved data with complexity scores to {scored_output_file}")
    
    for lang, lang_df in final_language_dfs.items():
        scored_lang_df, _ = sum_scores(lang_df.copy(), dataset_name=lang)
        
        scored_lang_df = format_decimals(scored_lang_df)
        lang_output_file = os.path.join(complexity_by_lang_dir, f"{args.dataset_source}_{lang}_complexity.csv")
        scored_lang_df.to_csv(lang_output_file, index=False)
        print(f"Saved scored data for {lang} to {lang_output_file}")
    
    # ablation datasets function flag
    if args.ablation:
        print("\nCreating ablation datasets...")
        ablation_dir = os.path.join(dataset_output_dir, "ablation")
        os.makedirs(ablation_dir, exist_ok=True)
        
        baseline_df = format_decimals(scored_df.copy())
        baseline_file = os.path.join(ablation_dir, f"{args.dataset_source}_baseline_all_features.csv")
        baseline_df.to_csv(baseline_file, index=False)
        print(f"Created baseline dataset with all features: {baseline_file}")
        
        feature_columns = [
            'avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len',
            'avg_verb_edges', 'lexical_density', 'n_tokens',
            'subordinate_proposition_dist', 'verbal_head_per_sent'
        ]
        
        existing_features = [col for col in feature_columns if col in scored_df.columns]
        
        for feature in existing_features:
            ablation_df, _ = sum_scores(scored_df.copy(), excluded_feature=feature)
            ablation_df = format_decimals(ablation_df)
            
            output_file = os.path.join(ablation_dir, f"{args.dataset_source}_ablation_no_{feature}.csv")
            ablation_df.to_csv(output_file, index=False)
            print(f"Created ablation dataset: {output_file}")
    
    summary = scored_df.groupby(['language', 'type_original']).size().reset_index(name='count')
    stats["summary"] = summary
    
    print("\nSummary of final dataset:")
    print(summary)
    
    
    print("\nProcessing complete!")
if __name__ == "__main__":
    main()