
import os
import pandas as pd
import yaml
import json
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, load_dataset
from huggingface_hub import HfApi, upload_file, upload_folder, create_repo
import argparse
import time



def process_dataset():
    parser = argparse.ArgumentParser(description='Upload question data to HF hub')
    parser.add_argument('--token-env', type=str, default='HF_TOKEN')
    parser.add_argument('--dataset-name', type=str, default='question-complexity')
    parser.add_argument('--tydi-dir', type=str, required=True)
    parser.add_argument('--ud-dir', type=str, required=True)
    parser.add_argument('--dev-dir', type=str, help='Directory containing dev data')
    parser.add_argument('--config-file', type=str, help='Path to existing configs.yaml file')
    parser.add_argument('--readme-file', type=str, help='Path to existing README.md file')
    parser.add_argument('--random-seed', type=int, default=69)
    args = parser.parse_args()
    
    
    
    token = os.environ.get(args.token_env)
    if not token:
        print("No token provided. set the HF_TOKEN environment variable.")
        exit(1)
    
    api = HfApi(token=token)
    username = api.whoami()["name"]
    repo_id = f"{username}/{args.dataset_name}"
    
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Repository {repo_id} exists")
    except:
        print(f"Creating repository {repo_id}...")
        create_repo(repo_id=repo_id, repo_type="dataset", token=token)
    
    os.makedirs("hf_upload", exist_ok=True)
    
    datasets_info = {}
    
    # Process TyDi train data
    tydi_path = os.path.join(args.tydi_dir, "base", "tydi_train_base.csv")
    if os.path.exists(tydi_path):
        print(f"Processing TyDi base dataset from {tydi_path}")
        tydi_df = pd.read_csv(tydi_path)

        if 'source' in tydi_df.columns:
            tydi_df = tydi_df.drop('source', axis=1)

        tydi_df['dataset_source'] = 'tydi'
        tydi_df = tydi_df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
        
        tydi_csv_path = "hf_upload/tydi_train_base.csv"
        tydi_df.to_csv(tydi_csv_path, index=False)
        
        datasets_info["tydi_train_base"] = {
            "path": tydi_csv_path,
            "num_examples": len(tydi_df)
        }
    
    # Process UD test data
    ud_path = os.path.join(args.ud_dir, "base", "ud_test_base.csv")
    if os.path.exists(ud_path):
        print(f"Processing UD base dataset from {ud_path}")
        ud_df = pd.read_csv(ud_path)

        if 'source' in ud_df.columns:
            ud_df = ud_df.drop('source', axis=1)
            

        ud_df['dataset_source'] = 'ud'
        ud_df = ud_df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
        
        ud_csv_path = "hf_upload/ud_test_base.csv"
        ud_df.to_csv(ud_csv_path, index=False)
        
        datasets_info["ud_test_base"] = {
            "path": ud_csv_path,
            "num_examples": len(ud_df)
        }
    
    # Process Dev data if provided
    if args.dev_dir:
        dev_path = os.path.join(args.dev_dir, "base", "dev_base.csv")
        if os.path.exists(dev_path):
            print(f"Processing dev dataset from {dev_path}")
            dev_df = pd.read_csv(dev_path)

            if 'source' in dev_df.columns:
              dev_df = dev_df.drop('source', axis=1)
            

            if 'dataset_source' not in dev_df.columns:
                dev_df['dataset_source'] = 'dev'
            dev_df = dev_df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
            
            dev_csv_path = "hf_upload/dev_base.csv"
            dev_df.to_csv(dev_csv_path, index=False)
            
            datasets_info["dev_base"] = {
                "path": dev_csv_path,
                "num_examples": len(dev_df)
            }
      

    if args.config_file and os.path.exists(args.config_file):
        print(f"Using existing configs from {args.config_file}")
        with open(args.config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        with open("hf_upload/configs.yaml", "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
    else:
        
        print("No config file provided, generating basic config")
        config_data = {
            "configs": [
                {
                    "config_name": "base",
                    "splits": [
                        {
                            "split_name": "train",
                            "file": "tydi_train_base.csv"
                        },
                        {
                            "split_name": "test",
                            "file": "ud_test_base.csv"
                        }
                    ]
                }
            ]
        }
        
        if "dev_base" in datasets_info:
            config_data["configs"][0]["splits"].append({
                "split_name": "validation",
                "file": "dev_base.csv"
            })
        
        with open("hf_upload/configs.yaml", "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
    
    # Use existing README if provided
    if args.readme_file and os.path.exists(args.readme_file):
        print(f"Using existing README from {args.readme_file}")
        with open(args.readme_file, 'r') as src_file:
            with open("hf_upload/README.md", "w") as dest_file:
                dest_file.write(src_file.read())
    
    print(f"Uploading to {repo_id}...")
    api.upload_folder(
        folder_path="hf_upload",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload dataset with YAML configuration"
    )
    
    print(f"\n✓ Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")
    print(f"The dataset uses a YAML-based configuration system")
    
    print("\nTo load this dataset in Python:")
    print("from datasets import load_dataset")
    print(f"dataset = load_dataset(\"{repo_id}\")")

if __name__ == "__main__":
    process_dataset()