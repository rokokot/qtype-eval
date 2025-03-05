import os
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
#import argilla as rg
from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi

#def argilla_config(url: str, key: str):


def hf_config(token: str):
  if token is None:
    token = os.environ.get("HF_TOKEN")

  if token:
    try:
      login(token=token, add_to_git_credential=True)
      api = HfApi()
      user_info = api.whoami(token=token)
      print(f'HF login at {user_info['name']}')
      return True
    except Exception as e:
      print(f'login failed: {e}')
      return False
  


def load_data(input_folder, file: str="*.csv"):
  data = Path(input_folder)
  files = list(data.glob(file))

  if not files:
    raise ValueError(f"no files found in {input_folder}")

  dfs = []
  for file in tqdm(files, desc=f'reading files in {input_folder.name}'):
    df = pd.read_csv(file)
    df['source']=data.name
    dfs.append(df)
  combined_df = pd.concat(dfs, ignore_index=True)
  print(f"combined {len(files)} files with {len(combined_df)} rows")
  return combined_df


def format_dataset(gold_folder: str, silver_folder: str, file: str="*.csv"):
  try:
    gold_df = load_data(gold_folder, file)
    gold_df['split'] = 'gold'
    print(f'Gold(UD) data has {len(gold_df)} rows') 
  except ValueError as e:
    print(f'Error {e}')
    gold_df=pd.DataFrame
  
  try:
    silver_df = load_data(silver_folder, file)
    silver_df['split'] = 'silver'
    print(f'Silver(TyDi) data has {len(silver_df)} examples')
  except ValueError as e:
    print(f'Error {e}')
    silver_df = pd.DataFrame()


def upload_hf(data, repo_id, message):

def save_backup(data, path):

def main()