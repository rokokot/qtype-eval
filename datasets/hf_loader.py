import os 
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel
from datetime import datetime
from huggingface_hub import upload_file, login, HfApi
import json
import argparse


def process_dataset():
    """
    Main function to load the csv dataset files, configure the HF setup, and process the data
    """

    parser = argparse.ArgumentParser(description='upload question data to HF hub')

    parser.add_argument('--token-env', type=str, default='HF_TOKEN',
                        help='Environment variable containing HF API token')
    
    parser.add_argument('--dataset-name', type=str, default='question-complexity',
                        help='Name for the dataset')
    
    parser.add_argument('--tydi-dir', type=str, required=True,
                        help='Directory containing TyDi processed files')
    
    parser.add_argument('--ud-dir', type=str, required=True,
                        help='Directory containing UD processed files')
    
    parser.add_argument('--add-ablation', action='store_true', 
                        help='Include ablation studies as additional configurations')
    
    
    args = parser.parse_args()
    
    # load token from environment variable
    token = os.environ.get(args.token_env)
    if not token:
        print(f"No token found in environment variable {args.token_env}")
        print("Attempting to use cached credentials")
    else:
        login(token=token)
    
    print("Loading datasets...")
    
    # load tydi data, choose the final set
    tydi_path = os.path.join(args.tydi_dir, "complexity", "tydi_all_languages_complexity.csv")
    tydi_df = pd.read_csv(tydi_path)
    tydi_df['dataset_source'] = 'tydi'
    tydi_df['split'] = 'train'
    
    # load ud data
    ud_path = os.path.join(args.ud_dir, "complexity", "ud_all_languages_complexity.csv")
    ud_df = pd.read_csv(ud_path)
    ud_df['dataset_source'] = 'ud'
    ud_df['split'] = 'test'
    
    # show dataset sizes
    print(f"Loaded {len(tydi_df)} TyDi samples and {len(ud_df)} UD samples")
    
    
    
    # some simple methods to ensure our data is consistently formatted across datasets

    print("Standardizing column names and formats...")
    
    if 'text' not in tydi_df.columns and 'question_text' in tydi_df.columns:
        tydi_df.rename(columns={'question_text': 'text'}, inplace=True)
    
    if 'text' not in ud_df.columns and 'question_text' in ud_df.columns:
        ud_df.rename(columns={'question_text': 'text'}, inplace=True)
        
    if 'unique_id' not in tydi_df.columns:
        tydi_df['unique_id'] = [f"tydi_{lang}_{i}" for i, lang in enumerate(tydi_df['language'])]
    if 'unique_id' not in ud_df.columns:
        ud_df['unique_id'] = [f"ud_{lang}_{i}" for i, lang in enumerate(ud_df['language'])]
    
    if 'question_type' in tydi_df.columns and tydi_df['question_type'].dtype == 'int64':
        pass
    elif 'type_original' in tydi_df.columns:
        tydi_df['question_type'] = tydi_df['type_original'].apply(lambda x: 1 if x.lower() == 'polar' else 0)
    
    if 'question_type' in ud_df.columns and ud_df['question_type'].dtype == 'int64':
        pass
    elif 'type_original' in ud_df.columns:
        ud_df['question_type'] = ud_df['type_original'].apply(lambda x: 1 if x.lower() == 'polar' else 0)
    
    
    
    print("Shuffling data within splits...") # shuffling the rows in final datasets
    tydi_shuffled = tydi_df.sample(frac=1, random_state=42).reset_index(drop=True)
    ud_shuffled = ud_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    
    tydi_columns = set(tydi_shuffled.columns)
    ud_columns = set(ud_shuffled.columns)
    all_columns = tydi_columns.union(ud_columns)
    
    for col in all_columns:
        if col not in tydi_columns:
            tydi_shuffled[col] = None
        if col not in ud_columns:
            ud_shuffled[col] = None
    
    # Feature config for HF metadata
    print("setting features..")
    features_dict = {
        'unique_id': Value('string'),
        'text': Value('string'),
        'language': Value('string'),
        'type_original': Value('string'),
        'question_type': ClassLabel(names=['content', 'polar']),
        'complexity_score': Value('float'),
        'dataset_source': Value('string'),
        'split': Value('string'),
    }
    
    linguistic_features = [
        'avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len',
        'avg_verb_edges', 'lexical_density', 'n_tokens',
        'subordinate_proposition_dist', 'verbal_head_per_sent'
    ]
    
    for feature in linguistic_features:
        if feature in all_columns:
            features_dict[feature] = Value('float')
    
    for col in all_columns:
        if col not in features_dict and col not in ['index']:
            features_dict[col] = Value('string')
    
    combined_features = Features(features_dict)
    
    # create datasets
    print("Creating datasets...")
    tydi_dataset = Dataset.from_pandas(tydi_shuffled, features=combined_features)
    ud_dataset = Dataset.from_pandas(ud_shuffled, features=combined_features)
    
    dataset_dict = DatasetDict({
        'train': tydi_dataset,
        'test': ud_dataset
    })
    
    print(f"Main dataset created with {len(tydi_dataset)} training samples and {len(ud_dataset)} test samples")
    
    if args.add_ablation:
        print("Processing ablation datasets...")
        ablation_datasets = {}
        
        tydi_baseline_path = os.path.join(args.tydi_dir, "ablation", "tydi_baseline_all_features.csv")
        ud_baseline_path = os.path.join(args.ud_dir, "ablation", "ud_baseline_all_features.csv")
        
        if os.path.exists(tydi_baseline_path) and os.path.exists(ud_baseline_path):
            tydi_baseline = pd.read_csv(tydi_baseline_path)
            tydi_baseline['dataset_source'] = 'tydi'
            tydi_baseline['split'] = 'train'
            
            ud_baseline = pd.read_csv(ud_baseline_path)
            ud_baseline['dataset_source'] = 'ud'
            ud_baseline['split'] = 'test'
            
            baseline_combined = pd.concat([tydi_baseline, ud_baseline], ignore_index=True)
            baseline_dataset = Dataset.from_pandas(baseline_combined, features=combined_features)
            ablation_datasets["baseline"] = baseline_dataset
            
            for feature in linguistic_features:
                tydi_ablation_path = os.path.join(args.tydi_dir, "ablation", f"tydi_ablation_no_{feature}.csv")
                ud_ablation_path = os.path.join(args.ud_dir, "ablation", f"ud_ablation_no_{feature}.csv")
                
                if os.path.exists(tydi_ablation_path) and os.path.exists(ud_ablation_path):
                    try:
                        tydi_ablation = pd.read_csv(tydi_ablation_path)
                        tydi_ablation['dataset_source'] = 'tydi'
                        tydi_ablation['split'] = 'train'
                        
                        ud_ablation = pd.read_csv(ud_ablation_path)
                        ud_ablation['dataset_source'] = 'ud'
                        ud_ablation['split'] = 'test'

                        if feature in tydi_ablation.columns:
                            tydi_ablation = tydi_ablation.drop(columns=[feature])
                        if feature in ud_ablation.columns:
                            ud_ablation = ud_ablation.drop(columns=[feature])
                        
                        ablation_combined = pd.concat([tydi_ablation, ud_ablation], ignore_index=True)
                        ablation_dataset = Dataset.from_pandas(ablation_combined, features=combined_features)
                        ablation_datasets[f"no_{feature}"] = ablation_dataset
                        print(f"  Added ablation dataset: no_{feature}")
                    except Exception as e:
                        print(f"  Error processing ablation for {feature}: {e}")
        
        if ablation_datasets:

            ablation_configs = {}

            for variant_name, combined_dataset in ablation_datasets.items():
                train_data = combined_dataset.filter(lambda x: x['split'] == 'train')
                test_data = combined_dataset.filter(lambda x: x['split'] == 'test')

                ablation_configs[variant_name] = DatasetDict({
                    'train': train_data,
                    'test': test_data
                })
            for variant_name, split_dataset in ablation_configs.items():
                split_dataset.push_to_hub(
                    args.dataset_name,
                    config_name=f"ablation-{variant_name}",
                    private=False,
                    token=token)
                
                print(f"pushed ablation variant: {variant_name}")

    
    if not os.path.exists("README.md"):
        print("WARNING: README.md not found. Please create one before uploading.")
        exit(1)
    else:
        readme_size = os.path.getsize("README.md")
        print(f"Using README.md file ({readme_size} bytes)")
    
    # Read the README content to ensure it's valid
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()
        # Print first 100 chars to verify content
            print(f"README preview: {readme_content[:100]}...")
    
    print("Generating dataset card metadata...")
    metadata = {
        "language": ["ar", "en", "fi", "id", "ja", "ko", "ru"],
        "license": "cc-by-sa-4.0",
        "annotations_creators": ["found", "machine-generated"],
        "language_creators": ["found"],
        "task_categories": ["text-classification", "question-answering"],
        "task_ids": ["text-classification-other", "question-complexity"],
        "multilinguality": "multilingual",
        "size_categories": ["1K<n<10K"],
        "source_datasets": ["original", "extended|universal-dependencies", "extended|tydiqa"],
        "paperswithcode_id": "",
        "pretty_name": "Question Type and Complexity Dataset"
    }
    
    with open("dataset_infos.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Uploading to Hugging Face as {args.dataset_name}...")
    
    user = HfApi().whoami(token=token)["name"]

    repo_id = f'{user}/{args.dataset_name}'


    dataset_dict.push_to_hub(
        repo_id,
        private=False,
        token=token
    )

    try:
        print(f'uploading README to {repo_id}')
        api = HfApi()
        
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token
        )
    except Exception as e:
        print(f'{e}')
    
    
    print(f"Successfully uploaded dataset to https://huggingface.co/datasets/{args.dataset_name}")
    print("All OK!")

if __name__ == "__main__":
    process_dataset()

