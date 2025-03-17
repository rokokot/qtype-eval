#!/usr/bin/env python3
"""
Linguistic Complexity Data Normalization and Pre-processing

This script processes linguistic complexity data from TyDi and UD datasets, creating:

1. Training sets with normalized features based on command-line flags
- normalization procedures are configured in the filter_and_preprocess function

2. Validation sets from leftover data per language

3. Ablation sets for feature importance analysis


4. The script runs the csv files through a cleaning function, removes unused columns, formats the floats, encodes question types, and converts language names to iso codes

5.

Usage:
    python normalize_scores.py --tydi-dir /path/to/tydi --ud-dir /path/to/ud --output-dir /path/to/output [--no-feature-normalization] [--no-score-normalization]

To reproduce the final version of the dataset run: 
    python normalize_scores.py --tydi-dir [path] --ud-dir [path] --output-dir [path] --min-tokens 4 --max-tokens 100 --remove-ablated-features --no-score-normalization

Author: Robin Kokot
Date: 10/03/2025
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
from datetime import datetime
from scipy import stats

# =============================================================================
# Setup and Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("normalize_scores.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("complexity_processor")


LANGUAGE_CODES = {
    'english': 'en', 'russian': 'ru', 'japanese': 'ja', 'arabic': 'ar',
    'finnish': 'fi', 'korean': 'ko', 'indonesian': 'id'
}

FEATURE_COLUMNS = [
    'avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len',
    'avg_verb_edges', 'lexical_density', 'n_tokens'
]

VALIDATION_RATIO = 0.10  

# =============================================================================
# Command Line Arguments
# =============================================================================

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process linguistic complexity data files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--tydi-dir', required=True, 
                       help='Directory containing TyDi data files')
    parser.add_argument('--ud-dir', required=True, 
                       help='Directory containing UD data files')
    parser.add_argument('--output-dir', required=True, 
                       help='Directory to store output files')
    parser.add_argument('--min-tokens', type=int, default=4, 
                       help='Minimum token count')
    parser.add_argument('--max-tokens', type=int, default=100, 
                       help='Maximum token count')
    parser.add_argument('--no-ablation', action='store_true', 
                       help='Skip ablation set creation')
    parser.add_argument('--remove-ablated-features', action='store_true', 
                      help='Remove ablated feature columns from datasets')
    parser.add_argument('--no-feature-normalization', action='store_true',
                       help='Skip normalization of individual features')
    parser.add_argument('--no-score-normalization', action='store_true',
                       help='Skip Z-score normalization of the complexity score')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging')
    return parser.parse_args()

# =============================================================================
# Data Loading and Basic Processing
# =============================================================================

def load_data(directory):

    """Load all CSV files from a directory, grouping by language and type"""


    data_files = {}
    language_type_counts = defaultdict(lambda: defaultdict(int))
    
    logger.info(f"Loading data files from {directory}")
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {directory}")
        return data_files, language_type_counts

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        parts = file_name.replace('.csv', '').split('_')
        
        if len(parts) < 2:
            logger.warning(f"Unexpected file name format: {file_name}, skipping")
            continue
            
        language = parts[0]
        q_type = parts[1]
        
        logger.info(f"Reading {file_name}")
        try:
            df = pd.read_csv(file_path)
            
            if 'n_tokens' not in df.columns:
                logger.warning(f"Missing 'n_tokens' column in {file_name}, skipping")
                continue
                
            if 'language' not in df.columns:
                df['language'] = language
            if 'type' not in df.columns:
                df['type'] = q_type
                
            key = f"{language}_{q_type}"
            data_files[key] = df
            language_type_counts[language][q_type] = len(df)
            logger.info(f"Loaded {len(df)} rows from {file_name}")
            
        except Exception as e:
            logger.error(f"Error loading {file_name}: {e}")
    
    return data_files, language_type_counts





def filter_and_preprocess(df, min_tokens, max_tokens, normalize_features=True, language_code_mapping=None):

    """Filter by token count (min, max); convert types to binary labels (1:polar, 0:content); run normalization, convert lang names to ISO codes"""


    filtered_df = df[(df['n_tokens'] >= min_tokens) & (df['n_tokens'] <= max_tokens)].copy()
    
    filtered_df['original_n_tokens'] = filtered_df['n_tokens'].copy()
    
    if language_code_mapping and 'language' in filtered_df.columns:
        filtered_df['language'] = filtered_df['language'].apply(
            lambda x: language_code_mapping.get(x.lower(), x) if isinstance(x, str) else x
        )
    

    # Convert question types to binary labels (0,1)
    if 'type' in filtered_df.columns:
        filtered_df['type_original'] = filtered_df['type'].copy()
        filtered_df['question_type'] = filtered_df['type'].apply(
            lambda x: 1 if x.lower() == 'polar' else 0 if x.lower() == 'content' else None
        )
    
    if normalize_features:

  
        for feature in ['avg_links_len', 'avg_max_depth']:       # Mean centering for avg_links_len and avg_max_depth
            if feature in filtered_df.columns:
                for language in filtered_df['language'].unique():
                    mask = filtered_df['language'] == language      # We normalize the scores for each language separately

                    feature_mean = filtered_df.loc[mask, feature].mean() # Mean looks only at the masked rows

                    filtered_df.loc[mask, feature] = filtered_df.loc[mask, feature] - feature_mean     # Value of the feature - Language mean
        


        if 'n_tokens' in filtered_df.columns:             # Log normalization for n_tokens
            filtered_df['n_tokens'] = np.log1p(filtered_df['n_tokens'])
        


        if 'avg_verb_edges' in filtered_df.columns:                 # Min-max scaling for avg_verb_edges

            for language in filtered_df['language'].unique():
                mask = filtered_df['language'] == language

                min_val = filtered_df.loc[mask, 'avg_verb_edges'].min()
                max_val = filtered_df.loc[mask, 'avg_verb_edges'].max()
                range_val = max_val - min_val
                
                if range_val > 0:
                    filtered_df.loc[mask, 'avg_verb_edges'] = (
                        (filtered_df.loc[mask, 'avg_verb_edges'] - min_val) / range_val
                    )
    

    return filtered_df




def calculate_complexity_score(df, excluded_feature=None, normalize_score=True, remove_excluded=False):

    """Calculate complexity score and optionally apply Z-score normalization, the combined score is a simple sum of features"""


    feature_columns = [col for col in FEATURE_COLUMNS if col in df.columns]
    if excluded_feature and excluded_feature in feature_columns:        # Handle ablation options
        feature_columns.remove(excluded_feature)
    

    result_df = df.copy()     # Calculate sum

    result_df['complexity_score'] = result_df[feature_columns].sum(axis=1)
    


    if normalize_score and len(result_df) > 1:
        mean = result_df['complexity_score'].mean()
        std = result_df['complexity_score'].std()
        
        if std > 0:
            result_df['complexity_score'] = (result_df['complexity_score'] - mean) / std        # Similar to how we calculated scores avg max depth, and links length
    
    # Exclude feature from score sum, if requested this function will remove the original column form the datset
    if remove_excluded and excluded_feature and excluded_feature in result_df.columns:
        result_df = result_df.drop(columns=[excluded_feature])
    
    # Format decimal values
    for column in result_df.select_dtypes(include=['float']).columns:
        result_df[column] = result_df[column].round(3)
    
    return result_df

def finalize_dataframe(df, drop_original_tokens=True):

    """Remove temporary columns and prepare dataframe for saving"""
    final_df = df.copy()
    

    for col in ['type', 'type_original']:               # Remove type columns
        if col in final_df.columns:
            final_df = final_df.drop(columns=[col])
    


    if drop_original_tokens and 'original_n_tokens' in final_df.columns:           # Remove original_n_tokens if requested
        final_df = final_df.drop(columns=['original_n_tokens'])

    return final_df


# =============================================================================
# Sampling Functions
# =============================================================================

seed = 69


def split_train_validation(df, validation_ratio=VALIDATION_RATIO, random_state=seed):
    """Split dataframe into training and validation sets"""

    if len(df) <= 10:  # Too small to split
        return df, pd.DataFrame()
    
    # Stratify by language and question type if available
    if 'language' in df.columns and 'type_original' in df.columns:
        # Create stratification column
        df['strat'] = df['language'] + '_' + df['type_original']
        
        # Count samples in each stratum
        strat_counts = df['strat'].value_counts()
        
        # Identify strata with enough samples to split
        valid_strata = strat_counts[strat_counts >= 5].index
        
        # Initialize train and validation dataframes
        train_df = pd.DataFrame()
        val_df = pd.DataFrame()
        
        for stratum in valid_strata:
            stratum_df = df[df['strat'] == stratum].copy()
            
            # Determine validation size
            val_size = max(2, int(len(stratum_df) * validation_ratio))
            
            # Sample for validation set
            val_stratum = stratum_df.sample(n=val_size, random_state=random_state)
            train_stratum = stratum_df.drop(val_stratum.index)
            
            # Add to respective dataframes
            train_df = pd.concat([train_df, train_stratum])
            val_df = pd.concat([val_df, val_stratum])
        
        # Add remaining data (from strata too small to split) to training set
        remaining_df = df[~df['strat'].isin(valid_strata)]
        train_df = pd.concat([train_df, remaining_df])
        
        # Drop stratification column
        if 'strat' in train_df.columns:
            train_df = train_df.drop(columns=['strat'])
        if 'strat' in val_df.columns:
            val_df = val_df.drop(columns=['strat'])
    else:
        # Simple random split
        val_size = int(len(df) * validation_ratio)
        val_df = df.sample(n=val_size, random_state=random_state)
        train_df = df.drop(val_df.index)
    
    return train_df, val_df




def token_based_sampling(df, target_size, random_state=seed):

    if len(df) <= target_size:
        return df
    
    token_column = 'original_n_tokens' if 'original_n_tokens' in df.columns else 'n_tokens'
    
    try:

        # Create bins based on token count
        num_bins = min(max(5, len(df) // 10), 10)
        
        df_with_bins = df.copy()
        df_with_bins['token_bin'] = pd.qcut(df_with_bins[token_column], 
                                         q=num_bins, 
                                         labels=False, 
                                         duplicates='drop')
        
        # Sample from each bin proportionally
        sampled_df = pd.DataFrame()
        bin_counts = df_with_bins['token_bin'].value_counts()
        
        for bin_id, bin_count in bin_counts.items():
            bin_df = df_with_bins[df_with_bins['token_bin'] == bin_id]
            
            # Calculate proportional sample size
            bin_proportion = bin_count / len(df_with_bins)
            bin_sample_size = max(1, int(target_size * bin_proportion))
            bin_sample_size = min(bin_sample_size, len(bin_df))
            
            # Sample from this bin
            bin_sample = bin_df.sample(n=bin_sample_size, random_state=random_state)
            sampled_df = pd.concat([sampled_df, bin_sample])
        
        # Adjust to exact target size if needed
        if len(sampled_df) > target_size:
            sampled_df = sampled_df.sample(n=target_size, random_state=random_state)
        
        # Clean up the bin column
        if 'token_bin' in sampled_df.columns:
            sampled_df = sampled_df.drop('token_bin', axis=1)
        
        return sampled_df
    
    except Exception as e:
        logger.error(f"Error during token-based sampling: {e}")
        # Fall back to random sampling
        return df.sample(n=target_size, random_state=random_state)





def complexity_distribution_sampling(ud_df, tydi_df, target_size=None, random_state=seed):
    """Sample UD data to match complexity distribution in TyDi data"""

    if 'complexity_score' not in ud_df.columns or 'complexity_score' not in tydi_df.columns:
        return token_based_sampling(ud_df, target_size or min(len(ud_df), 55))
    
    if not target_size:
        target_size = min(len(ud_df), 55)
    elif target_size > len(ud_df):
        return ud_df
    
    try:

        # Create complexity bins from TyDi data
        n_bins = min(10, len(tydi_df) // 20)
        if n_bins >= 2:
            # Get bin edges from TyDi
            tydi_values = tydi_df['complexity_score'].values
            _, bin_edges = pd.qcut(tydi_values, n_bins, retbins=True, duplicates='drop')
            
            # Apply binning to UD data
            ud_values = ud_df['complexity_score'].values
            ud_bins = np.digitize(ud_values, bin_edges[1:-1])
            
            # Get target distribution from TyDi
            tydi_bin_counts = pd.Series(np.digitize(tydi_values, bin_edges[1:-1])).value_counts(normalize=True)
            
            # Sample from each bin according to target distribution
            sampled_df = pd.DataFrame()
            
            for bin_id, bin_proportion in tydi_bin_counts.items():
                bin_mask = ud_bins == bin_id
                bin_df = ud_df[bin_mask]
                
                if len(bin_df) > 0:
                    bin_target = max(1, int(target_size * bin_proportion))
                    bin_target = min(bin_target, len(bin_df))
                    
                    bin_sample = bin_df.sample(n=bin_target, random_state=random_state)
                    sampled_df = pd.concat([sampled_df, bin_sample])
            
            # Adjust sample size if needed
            if len(sampled_df) < target_size:
                remaining = target_size - len(sampled_df)
                remaining_df = ud_df[~ud_df.index.isin(sampled_df.index)]
                if len(remaining_df) > 0:
                    additional_samples = remaining_df.sample(
                        n=min(remaining, len(remaining_df)), 
                        random_state=random_state
                    )
                    sampled_df = pd.concat([sampled_df, additional_samples])
            elif len(sampled_df) > target_size:
                sampled_df = sampled_df.sample(n=target_size, random_state=random_state)
            
            return sampled_df
        else:
            return token_based_sampling(ud_df, target_size, random_state)
    
    except Exception as e:
        logger.error(f"Error during complexity distribution sampling: {e}")
        return token_based_sampling(ud_df, target_size, random_state)








# =============================================================================
# Main Processing Function
# =============================================================================

def process_files(args):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = args.output_dir
    os.makedirs(output_base, exist_ok=True)
    

    final_dir = os.path.join(output_base, "final")
    train_dir = os.path.join(output_base, "train")  
    test_dir = os.path.join(output_base, "test")    
    dev_dir = os.path.join(output_base, "dev")      
    
    validation_dir = os.path.join(output_base, "validation")
    ablation_dir = os.path.join(output_base, "ablation")
    
    for directory in [train_dir, test_dir, dev_dir, ablation_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Log settings
    logger.info(f"Feature normalization: {'DISABLED' if args.no_feature_normalization else 'ENABLED'}")
    logger.info(f"Score normalization: {'DISABLED' if args.no_score_normalization else 'ENABLED'}")
    



    # === Load data ===
    logger.info("=== Loading TyDi data ===")
    tydi_files, tydi_counts = load_data(args.tydi_dir)
    
    logger.info("=== Loading UD data ===")
    ud_files, ud_counts = load_data(args.ud_dir)
    
    # Group files by language
    tydi_by_lang = defaultdict(dict)
    ud_by_lang = defaultdict(dict)
    
    for key, df in tydi_files.items():
        language, q_type = key.split('_')
        tydi_by_lang[language][q_type] = df
    
    for key, df in ud_files.items():
        language, q_type = key.split('_')
        ud_by_lang[language][q_type] = df
    
    # Track processed data
    processed_tydi_by_lang = {}
    processed_ud_by_lang = {}
    dev_samples_by_lang = {}
    
    all_languages = sorted(set(list(tydi_by_lang.keys()) + list(ud_by_lang.keys())))
    
    # For storing dev set samples
    dev_samples_by_lang = {}
    
    MAX_DEV_SAMPLES_PER_LANGUAGE = 75  
    
    for language in all_languages:
        logger.info(f"\n=== Processing language: {language} ===")
        dev_tydi_samples = []
        dev_ud_samples = []
        


        tydi_processed_types = {} # Keep track of tydi samples per type
        
        if language in tydi_by_lang:
            logger.info(f"Processing TyDi {language} data (for TRAIN set)")
            
            # Target sizes per language and type for TyDi
            target_sizes = {
                'ko': {'polar': 500, 'content': 400},
                'id': {'polar': 474, 'content': 500},
                'ar': {'polar': 500, 'content': 500},
                'en': {'polar': 600, 'content': 600},
                'fi': {'polar': 600, 'content': 600},
                'ja': {'polar': 600, 'content': 600},
                'ru': {'polar': 600, 'content': 600},
                'default': {'polar': 500, 'content': 500}
            }
            
            # Calculate how many dev samples to take from each question type
            tydi_q_types = len(tydi_by_lang[language])
            max_dev_per_type = MAX_DEV_SAMPLES_PER_LANGUAGE // (2 * tydi_q_types)  # Half for TyDi, evenly split by type
            
            # Process each question type
            for q_type, df in tydi_by_lang[language].items():
                logger.info(f"  Processing TyDi {language}_{q_type}")
                
                # Basic preprocessing
                preprocessed_df = filter_and_preprocess(
                    df, 
                    args.min_tokens, 
                    args.max_tokens, 
                    normalize_features=not args.no_feature_normalization,
                    language_code_mapping=LANGUAGE_CODES
                )
                
                # Calculate complexity score
                scored_df = calculate_complexity_score(
                    preprocessed_df,
                    normalize_score=not args.no_score_normalization
                )
                
                # Extract some samples for dev set (before sampling for training)
                # Take a fixed number per question type, not a percentage
                dev_size = min(max_dev_per_type, len(scored_df) // 10)  # No more than 10% of data
                
                if dev_size > 0:
                    dev_samples = scored_df.sample(n=dev_size, random_state=seed)
                    dev_samples['source'] = 'tydi'  # Explicitly mark source
                    dev_tydi_samples.append(dev_samples)
                    
                    # Remove dev samples from the training data
                    train_df = scored_df.drop(dev_samples.index)
                else:
                    train_df = scored_df
                
                # Determine target size for training
                target_size = (target_sizes.get(language, {}).get(q_type) or 
                              target_sizes.get(language, {}).get('default', 500))
                
                # Sample training data
                train_df = token_based_sampling(
                    train_df, 
                    min(target_size, len(train_df)), 
                    random_state=seed
                )
                
                # Store processed dataframes
                tydi_processed_types[q_type] = train_df
                
                logger.info(f"  TyDi {language}_{q_type}: {len(train_df)} train, {dev_size} dev")
            
            # Combine all question types for this language's training set
            if tydi_processed_types:
                combined_train_df = pd.concat(tydi_processed_types.values(), ignore_index=True)
                combined_train_df = combined_train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
                processed_tydi_by_lang[language] = combined_train_df
                
                # Save the combined training file
                final_train_df = finalize_dataframe(combined_train_df)
                final_train_df.to_csv(os.path.join(train_dir, f"tydi_{language}.csv"), index=False)
                logger.info(f"  Saved TyDi {language} training data: {len(final_train_df)} rows")
        



        ud_processed_types = {}        # keep track of ud samples by type
        
        if language in ud_by_lang:
            logger.info(f"Processing UD {language} data (for TEST set)")
            
            # Get TyDi reference for this language if available
            tydi_ref_df = processed_tydi_by_lang.get(language)
            
            # Calculate how many dev samples to take from each question type
            ud_q_types = len(ud_by_lang[language])
            max_dev_per_type = MAX_DEV_SAMPLES_PER_LANGUAGE // (2 * ud_q_types)  # Half for UD, evenly split by type
            
            # Process each question type
            for q_type, df in ud_by_lang[language].items():
                logger.info(f"  Processing UD {language}_{q_type}")
                
                # Basic preprocessing
                preprocessed_df = filter_and_preprocess(
                    df, 
                    args.min_tokens, 
                    args.max_tokens, 
                    normalize_features=not args.no_feature_normalization,
                    language_code_mapping=LANGUAGE_CODES
                )
                
                # Calculate complexity score
                scored_df = calculate_complexity_score(
                    preprocessed_df,
                    normalize_score=not args.no_score_normalization
                )
                
                # Extract some samples for dev set (before sampling for testing)
                # Take a fixed number per question type, not a percentage

                dev_size = min(max_dev_per_type, len(scored_df) // 10)  # No more than 10% of data
                
                if dev_size > 0:
                    dev_samples = scored_df.sample(n=dev_size, random_state=seed)
                    dev_samples['source'] = 'ud'  # Explicitly mark source
                    dev_ud_samples.append(dev_samples)
                    
                    # Remove dev samples from the test data
                    test_df = scored_df.drop(dev_samples.index)
                else:
                    test_df = scored_df
                

                # Match complexity distribution with TyDi if available for test set
                if tydi_ref_df is not None and len(tydi_ref_df) > 0:
                    # Filter TyDi to matching question type if possible
                    if 'type_original' in tydi_ref_df.columns:
                        tydi_type_df = tydi_ref_df[tydi_ref_df['type_original'] == q_type]
                        if len(tydi_type_df) == 0:
                            tydi_type_df = tydi_ref_df
                    else:
                        tydi_type_df = tydi_ref_df
                    
                    # Sample test data based on TyDi complexity distribution
                    target_size = 55  # Max 55 samples per type for UD
                    test_df = complexity_distribution_sampling(
                        test_df, 
                        tydi_type_df, 
                        target_size=min(target_size, len(test_df)),
                        random_state=seed
                    )
                else:
                    # No TyDi reference, use token-based sampling
                    target_size = 55
                    test_df = token_based_sampling(
                        test_df, 
                        min(target_size, len(test_df)),
                        random_state=seed
                    )
                
                # Store processed dataframes for test set
                ud_processed_types[q_type] = test_df
                
                logger.info(f"  UD {language}_{q_type}: {len(test_df)} test, {dev_size} dev")
            
            # Combine all question types for this language's test set
            if ud_processed_types:
                combined_test_df = pd.concat(ud_processed_types.values(), ignore_index=True)
                combined_test_df = combined_test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
                processed_ud_by_lang[language] = combined_test_df
                
                # Save the combined test file
                final_test_df = finalize_dataframe(combined_test_df)
                final_test_df.to_csv(os.path.join(test_dir, f"ud_{language}.csv"), index=False)
                logger.info(f"  Saved UD {language} test data: {len(final_test_df)} rows")
        
        # Combine dev samples from both sources for this language
        if dev_tydi_samples or dev_ud_samples:
            combined_dev_samples = []
            
            if dev_tydi_samples:
                combined_dev_samples.extend(dev_tydi_samples)
                tydi_dev_count = sum(len(df) for df in dev_tydi_samples)
            else:
                tydi_dev_count = 0
                
            if dev_ud_samples:
                combined_dev_samples.extend(dev_ud_samples)
                ud_dev_count = sum(len(df) for df in dev_ud_samples)
            else:
                ud_dev_count = 0
                
            if combined_dev_samples:
                combined_dev_df = pd.concat(combined_dev_samples, ignore_index=True)
                combined_dev_df = combined_dev_df.sample(frac=1, random_state=seed).reset_index(drop=True)
                dev_samples_by_lang[language] = combined_dev_df
                
                # Save language-specific dev set
                final_dev_df = finalize_dataframe(combined_dev_df)
                final_dev_df.to_csv(os.path.join(dev_dir, f"{language}_dev.csv"), index=False)
                logger.info(f"  Created {language} development set with {len(final_dev_df)} rows ({tydi_dev_count} TyDi, {ud_dev_count} UD)")
    



    # === Create combined datasets for all languages ===
    logger.info("\n === Create combined datasets for all languages ===")
    
    # Combine all TyDi training datasets
    if processed_tydi_by_lang:
        all_tydi_df = pd.concat(processed_tydi_by_lang.values(), ignore_index=True)
        all_tydi_df = all_tydi_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        final_tydi_df = finalize_dataframe(all_tydi_df)
        final_tydi_df.to_csv(os.path.join(train_dir, "tydi_all_combined.csv"), index=False)
        logger.info(f"Created combined TyDi training dataset with {len(final_tydi_df)} rows")
    else:
        logger.warning("No processed TyDi data to combine")
        all_tydi_df = None
    
    # Combine all UD test datasets
    if processed_ud_by_lang:
        all_ud_df = pd.concat(processed_ud_by_lang.values(), ignore_index=True)
        all_ud_df = all_ud_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        final_ud_df = finalize_dataframe(all_ud_df)
        final_ud_df.to_csv(os.path.join(test_dir, "ud_all_combined.csv"), index=False)
        logger.info(f"Created combined UD test dataset with {len(final_ud_df)} rows")
    else:
        logger.warning("No processed UD data to combine")
        all_ud_df = None
    
    # Create combined dev set from all languages
    if dev_samples_by_lang:
        all_dev_samples = []
        tydi_total = 0
        ud_total = 0
        
        for lang, dev_df in dev_samples_by_lang.items():
            all_dev_samples.append(dev_df)
            
            # Count by source
            if 'source' in dev_df.columns:
                tydi_count = len(dev_df[dev_df['source'] == 'tydi'])
                ud_count = len(dev_df[dev_df['source'] == 'ud'])
                tydi_total += tydi_count
                ud_total += ud_count
        
        if all_dev_samples:
            all_dev_df = pd.concat(all_dev_samples, ignore_index=True)
            all_dev_df = all_dev_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            final_dev_df = finalize_dataframe(all_dev_df)
            final_dev_df.to_csv(os.path.join(dev_dir, "all_dev.csv"), index=False)
            
            logger.info(f"Created combined development dataset with {len(final_dev_df)} rows total")
            logger.info(f"  - Dev set composition: {tydi_total} TyDi samples, {ud_total} UD samples")
    



    # === Create ablation datasets ===

    if not args.no_ablation and (all_tydi_df is not None or all_ud_df is not None):     
        logger.info("\n=== Create ablation datasets ===")
        
        # Create ablation sets for each language
        for source, data_by_lang in [("tydi", processed_tydi_by_lang), ("ud", processed_ud_by_lang)]:
            for language, df in data_by_lang.items():
                logger.info(f"Creating {source} {language} ablation sets")
                
                for feature in FEATURE_COLUMNS:
                    if feature in df.columns:
                        ablation_df = calculate_complexity_score(
                            df,
                            excluded_feature=feature,
                            normalize_score=not args.no_score_normalization,
                            remove_excluded=args.remove_ablated_features
                        )
                        
                        final_ablation_df = finalize_dataframe(ablation_df)
                        final_ablation_df.to_csv(
                                os.path.join(ablation_dir, f"{source}_{language}_no_{feature}.csv"),
                                index=False
                            )
        
        if all_tydi_df is not None:         # TyDi data ablation sets
            for feature in FEATURE_COLUMNS:
                if feature in all_tydi_df.columns:
                    logger.info(f"Ablating {feature} from combined TyDi dataset")
                    
                    ablation_df = calculate_complexity_score(
                        all_tydi_df,
                        excluded_feature=feature,
                        normalize_score=not args.no_score_normalization,
                        remove_excluded=args.remove_ablated_features
                    )
                    
                    # Finalize and save the ablation dataset
                    final_ablation_df = finalize_dataframe(ablation_df)
                    ablation_filename = f"tydi_all_no_{feature}.csv"
                    ablation_filepath = os.path.join(ablation_dir, ablation_filename)
                    final_ablation_df.to_csv(ablation_filepath, index=False)
                    
                    logger.info(f"    Saved combined TyDi ablation set: {ablation_filename}")
        


        if all_ud_df is not None:       # UD data ablation sets
            for feature in FEATURE_COLUMNS:
                if feature in all_ud_df.columns:
                    logger.info(f"Ablating {feature} from combined UD dataset")
                    
                    ablation_df = calculate_complexity_score(
                        all_ud_df,
                        excluded_feature=feature,
                        normalize_score=not args.no_score_normalization,
                        remove_excluded=args.remove_ablated_features
                    )
                    
                    # Finalize and save the ablation dataset
                    final_ablation_df = finalize_dataframe(ablation_df)
                    ablation_filename = f"ud_all_no_{feature}.csv"
                    ablation_filepath = os.path.join(ablation_dir, ablation_filename)
                    final_ablation_df.to_csv(ablation_filepath, index=False)
                    
                    logger.info(f" Saved combined UD ablation set: {ablation_filename}")
    
    if not args.no_ablation and dev_samples_by_lang:        # Fix the language loops, it only keeps the last language
        logger.info("\n ===== Create ablation sets for dev split ====")

        if 'all_dev_df' in locals() and len(all_dev_df) > 0:
            for feature in FEATURE_COLUMNS:
                if feature in all_dev_df.columns:
                    logger.info(f"Ablating {feature} from combined dev dataset")
                    
                    ablation_df = calculate_complexity_score(
                        all_dev_df,
                        excluded_feature=feature,
                        normalize_score=not args.no_score_normalization,
                        remove_excluded=args.remove_ablated_features
                    )
                    
                    # Finalize and save the ablation dataset
                    final_ablation_df = finalize_dataframe(ablation_df)
                    ablation_filename = f"all_dev_no_{feature}.csv"
                    ablation_filepath = os.path.join(dev_dir, ablation_filename)
                    final_ablation_df.to_csv(ablation_filepath, index=False)
                    
                    logger.info(f"    Saved combined dev ablation set: {ablation_filename}")
    
    # loop through languages, after processing the combined dev set
    # use ISO codes for loop

    for language, dev_df in dev_samples_by_lang.items():
        if len(dev_df) > 0:
            logger.info(f"Creating {language} dev ablation sets")
            
            for feature in FEATURE_COLUMNS:
                if feature in dev_df.columns:
                    ablation_df = calculate_complexity_score(
                        dev_df,
                        excluded_feature=feature,
                        normalize_score=not args.no_score_normalization,
                        remove_excluded=args.remove_ablated_features
                    )
                    
                    final_ablation_df = finalize_dataframe(ablation_df)
                    ablation_filename = f"{language}_dev_no_{feature}.csv"
                    final_ablation_df.to_csv(
                            os.path.join(dev_dir, ablation_filename),
                            index=False
                        )
                    
                    logger.info(f"    Saved {language} dev ablation set: {ablation_filename}")
   
   
   
   
    # === Summary ===
    logger.info("\n=== Processing complete ===")
    logger.info(f"Training data saved to: {train_dir}")
    logger.info(f"Test data saved to: {test_dir}")
    logger.info(f"Development data saved to: {dev_dir}")
    if not args.no_ablation:
        logger.info(f"Ablation files saved to: {ablation_dir}")
    
    if all_ud_df is not None:
        logger.info(f"Total UD training samples: {len(all_ud_df)}")
    if all_tydi_df is not None:
        logger.info(f"Total TyDi training samples: {len(all_tydi_df)}")


def main():

    args = parse_arguments()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info("Starting complexity data processor")
    logger.info(f"TyDi directory: {args.tydi_dir}")
    logger.info(f"UD directory: {args.ud_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Token filter range: [{args.min_tokens}, {args.max_tokens}]")
    
    process_files(args)

if __name__ == "__main__":
    main()