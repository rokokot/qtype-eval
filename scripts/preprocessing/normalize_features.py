import pandas as pd
import numpy as np
import os
import glob
import argparse
from collections import defaultdict
from pathlib import Path


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

    for feature in ['subordinate_proposition_dist', 'n_tokens']:
      normalized_df[feature] = np.log1p(df[feature])

    
    for feature in ['avg_verb_edges']:
      for language, group in language_groups:
        mask = normalized_df[language_column] == language
        min = df.loc[mask, feature].min()
        max = df.loc[mask, feature].max()
        range = max - min

        if range > 0:
          normalized_df.loc[mask, feature] = (df.loc[mask, feature] - min) / range

        else:
          normalized_df.loc[mask, feature] = 0
    return normalized_df
  

def process_files(input_pattern, output_dir=None):

  input_files = glob.glob(input_pattern)
  
  if not input_files:
    print(f'no files match {input_pattern}')
    return

  for input_file in input_files:
    filename = os.path.basename(input_file)

    print(f'processing {filename}')

    df = pd.read_csv(input_file)
    normalized_df = normalize_scores(df)

    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
      output_file = os.path.join(output_dir, f'normalized_{filename}')

    else:
      dir_name = os.path.dirname(input_file)
      output_file = os.path.join(dir_name, f'normalized_{filename}')


    normalized_df.to_csv(output_file, index=False, float_format='%.2f')
    print(f'saved normalized data to {output_file}')

def combine_languages(input_dir, output_dir, dataset_type):
  """
  This function combines the data from multiple csv files into one file per language, including both polar and content questions.

  arguments:
    input_dir: directory of the normalized scores
    output_dir: directory to save combined scores to
    dataset_type: 'tydi' or 'ud'
  
  """
  print(f'combining files in {input_dir}')

  csv_files = glob.glob(os.path.join(input_dir, 'normalized_*.csv'))

  if not csv_files:
    print(f'no CSV files found in {input_dir}')
    return set()
  print(f"found {len(csv_files)} files (csv)")
  
  language_files = defaultdict(list)

  for file_path in csv_files:
    file_name = os.path.basename(file_path)
    parts = file_name.replace('normalized_', '').replace('.csv', '').split('_')

    if len(parts) >= 2:
      language=parts[0]
      language_files[language].append(file_path)
  
  os.makedirs(output_dir, exist_ok=True)

  for language, files in language_files.items():
    print(f'combining data for {language}')

    dfs = []

    for file_path in files:
      file_name = os.path.basename(file_path)
      parts = file_name.replace('normalized_', '').replace('.csv', '').split('_')
      question_type = parts[1] if len(parts) >= 2 else 'uknown'

      try:
        df = pd.read_csv(file_path)

        if 'type' not in df.columns:
          df['type']=question_type
        
        if 'language' not in df.columns:
          df['language']=language

        dfs.append(df)
      
      except Exception as e:
        print(f'error processing {file_path}: {e}')
    
    if dfs:
      combined_df = pd.concat(dfs, ignore_index=True)

      feature_columns = [col for col in combined_df.columns if col not in ['language', 'type']]
      if feature_columns:
        combined_df['complexity_score'] = combined_df[feature_columns].sum(axis=1)
      
      combined_df['dataset'] = dataset_type
      output_file = os.path.join(output_dir, f"{language}_{dataset_type}_combined.csv")
      combined_df.to_csv(output_file, index=False)
      print(f"Saved combined file for {language} to {output_file}")
    
  print(f"Completed processing {input_dir}")
  return set(language_files.keys())

def combine_data(output_dir):

  all_files = glob.glob(os.path.join(output_dir, "*.csv"))
  if all_files:
    dfs = []
    for file_path in all_files:
      try:
        df = pd.read_csv(file_path)
        dfs.append(df)

      except Exception as e:
        print(f'error reading {file_path} > {e}')
  
    if dfs:
      combined_df = pd.concat(dfs, ignore_index=True)
      combined_file = os.path.join(os.path.dirname(output_dir), 'TyCoAnnotatedFullSize.csv')
      combined_df.to_csv(combined_file, index=False)
      print(f'combined all language files: {combined_file}')      




def main():
  parser = argparse.ArgumentParser(description='Run profiled .csv files through the normalization strategies')
  parser.add_argument('--input', '-i', required=True, help='Input fie pattern (e.g., "data/*.csv")')
  parser.add_argument('--output', '-o', help='Output directory (default: same as input)')
  parser.add_argument('--combine', '-c', action='store_true', help='Combine files by language after normalization')
  parser.add_argument('--tydi-dir', type=str, default='data/normalized_tydi', help='Directory with normalized TyDi files')
  parser.add_argument('--ud-dir', type=str, default='data/normalized_ud', help='Directory with normalized UD files')
  parser.add_argument('--output-combined', type=str, default='data/scores/language_files', help='Output directory for combined files')
    
    
  args = parser.parse_args()
    
  process_files(args.input, args.output)

  if args.combine:
    output_dir = args.output_combined
    os.makedirs(output_dir, exist_ok=True)

    tydi_languages = combine_languages(args.tydi_dir, output_dir, 'tydi')
    ud_languages = combine_languages(args.ud_dir, output_dir, 'ud')

    combine_data(output_dir)

    print("\nSummary:")
    print(f"TyDi languages processed: {', '.join(tydi_languages) if tydi_languages else 'None'}")
    print(f"UD languages processed: {', '.join(ud_languages) if ud_languages else 'None'}")
    print(f"Files saved to: {output_dir}")
    
if __name__ == "__main__":
    main()
