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
    parser.add_argument('--add-ablation', action='store_true')
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()
    
    
    
    token = args.token if args.token else os.environ.get(args.token_env)
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
    
    tydi_path = os.path.join(args.tydi_dir, "complexity", "tydi_all_languages_complexity.csv")
    if os.path.exists(tydi_path):
        print(f"Processing TyDi base dataset from {tydi_path}")
        tydi_df = pd.read_csv(tydi_path)
        tydi_df['dataset_source'] = 'tydi'
        tydi_df = tydi_df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
        
        tydi_csv_path = "hf_upload/tydi_train_base.csv"
        tydi_df.to_csv(tydi_csv_path, index=False)
        
        datasets_info["tydi_train_base"] = {
            "path": tydi_csv_path,
            "num_examples": len(tydi_df)
        }
    
    ud_path = os.path.join(args.ud_dir, "complexity", "ud_all_languages_complexity.csv")
    if os.path.exists(ud_path):
        print(f"Processing UD base dataset from {ud_path}")
        ud_df = pd.read_csv(ud_path)
        ud_df['dataset_source'] = 'ud'
        ud_df = ud_df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
        
        ud_csv_path = "hf_upload/ud_test_base.csv"
        ud_df.to_csv(ud_csv_path, index=False)
        
        datasets_info["ud_test_base"] = {
            "path": ud_csv_path,
            "num_examples": len(ud_df)
        }
    
    linguistic_features = [
        'avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len',
        'avg_verb_edges', 'lexical_density', 'n_tokens'
    ]
    

    if args.add_ablation:
        
        for feature in linguistic_features:
            feature_name = feature.replace('-', '_')
            
            tydi_abl_path = os.path.join(args.tydi_dir, "ablation", f"tydi_ablation_no_{feature}.csv")
            if os.path.exists(tydi_abl_path):
                print(f"Processing TyDi ablation dataset for {feature}")
                tydi_abl = pd.read_csv(tydi_abl_path)
                tydi_abl['dataset_source'] = 'tydi'
                tydi_abl = tydi_abl.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
                
                split_name = f"tydi_train_ablation_no_{feature_name}"
                tydi_abl_csv_path = f"hf_upload/{split_name}.csv"
                tydi_abl.to_csv(tydi_abl_csv_path, index=False)
                
                datasets_info[split_name] = {
                    "path": tydi_abl_csv_path,
                    "num_examples": len(tydi_abl)
                }
            
            ud_abl_path = os.path.join(args.ud_dir, "ablation", f"ud_ablation_no_{feature}.csv")
            if os.path.exists(ud_abl_path):
                print(f"Processing UD ablation dataset for {feature}")
                ud_abl = pd.read_csv(ud_abl_path)
                ud_abl['dataset_source'] = 'ud'
                ud_abl = ud_abl.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
                
                split_name = f"ud_test_ablation_no_{feature_name}"
                ud_abl_csv_path = f"hf_upload/{split_name}.csv"
                ud_abl.to_csv(ud_abl_csv_path, index=False)
                
                datasets_info[split_name] = {
                    "path": ud_abl_csv_path,
                    "num_examples": len(ud_abl)
                }
    
    
    configurations = {
        "base": {
            "train": "tydi_train_base.csv",
            "test": "ud_test_base.csv"
        }
    }
    
    for feature in linguistic_features:
        feature_name = feature.replace('-', '_')
        ablation_config_name = f"ablation_no_{feature_name}"
        
        tydi_ablation_file = f"tydi_train_ablation_no_{feature_name}.csv"
        ud_ablation_file = f"ud_test_ablation_no_{feature_name}.csv"
        
        if os.path.exists(f"hf_upload/{tydi_ablation_file}") and os.path.exists(f"hf_upload/{ud_ablation_file}"):
            configurations[ablation_config_name] = {
                "train": tydi_ablation_file,
                "test": ud_ablation_file
            }
    
    config_data = {
        "configs": []
    }
    
    for config_name, files in configurations.items():
        config_entry = {
            "config_name": config_name,
            "splits": []
        }
        
        for split_name, file_name in files.items():
            if os.path.exists(f"hf_upload/{file_name}"):
                config_entry["splits"].append({
                    "split_name": split_name,
                    "file": file_name
                })
        
        if config_entry["splits"]:  
            config_data["configs"].append(config_entry)
    
    with open("hf_upload/configs.yaml", "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)
    
    
        
        # Give an example with the first config
        if config_data["configs"]:
            first_config = config_data["configs"][0]["config_name"]
            f.write(f"# Load a specific configuration\n")
            f.write(f"dataset = load_dataset(\"{repo_id}\", \"{first_config}\")\n\n")
            
            # If the first config has splits, show how to access them
            if config_data["configs"][0]["splits"]:
                first_split = config_data["configs"][0]["splits"][0]["split_name"]
                f.write(f"# Access a specific split\n")
                f.write(f"data = dataset[\"{first_split}\"]\n")
        
        f.write("```\n")
    
    print(f"Uploading to {repo_id}...")
    api.upload_folder(
        folder_path="hf_upload",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload dataset with YAML configuration"
    )
    
    print(f"\n✓ Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")
    
    print(f"The dataset uses a YAML-based configuration system with {len(config_data['configs'])} configurations:")
    
    for config in config_data["configs"]:
        print(f"- {config['config_name']}")
    
    print("\nTo load this dataset in Python:")
    print("from datasets import load_dataset")
    
    if config_data["configs"]:
        first_config = config_data["configs"][0]["config_name"]
        print(f"dataset = load_dataset(\"{repo_id}\", \"{first_config}\")")

if __name__ == "__main__":
    process_dataset()