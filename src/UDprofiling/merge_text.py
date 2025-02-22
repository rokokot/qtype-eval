import pandas as pd
import zipfile
import os
import re
from tqdm import tqdm

def get_index(filename):

    if isinstance(filename, str):
        filename = filename.split('\t')[0].strip()
        match = re.match(r'(\d+)(?:_[12]\.conllu|\.txt)', filename)
        if match:
            return match.group(1)
    return None

def unzip(zip_path):

    sentences = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        txt_files = [f for f in zip_ref.namelist() if f.endswith('.txt')]
        
        print(f"\nFound {len(txt_files)} text files in zip")
        print("Sample of text files in zip:")
        for f in txt_files[:5]:
            print(f"- {f}")
            
        for filename in tqdm(txt_files, desc="Reading text files"):
            file_number = get_index(filename)
            if file_number is None:
                print(f'Error, no index for file: {filename}')
                continue
            with zip_ref.open(filename) as f:
                sentence = f.read().decode('utf-8').strip()
                sentences[file_number] = sentence
    
    return sentences

def merge_to_csv(sentences_dict, csv_path, output_path):
    df = pd.read_csv(csv_path, sep='\t')
    df['base_number'] = df['Filename'].apply(get_index)
    
    number_counts = df['base_number'].value_counts()
    duplicate_numbers = number_counts[number_counts > 1].index.tolist()
    
    print(f"\nFound {len(duplicate_numbers)} base numbers with multiple entries")
    if duplicate_numbers:
        print("Sample of duplicate entries:")
        for num in duplicate_numbers[:3]:
            print(f"\nBase number {num}:")
            print(df[df['base_number'] == num]['Filename'])

    rows_to_keep = ~df['base_number'].isin(duplicate_numbers)
    
    df_filtered = df[rows_to_keep].copy()
    
    df_filtered['sentence'] = df_filtered['base_number'].map(sentences_dict)
    
    missing_matches = df_filtered[df_filtered['sentence'].isna()]
    if not missing_matches.empty:
        print(f"\nWarning: {len(missing_matches)} rows still could not be matched with sentences")
        print("\nSample of unmatched rows:")
        print(missing_matches[['Filename', 'base_number']].head())

    cols = df_filtered.columns.tolist()
    cols = ['sentence'] + [col for col in cols if col != 'sentence']
    df_filtered = df_filtered[cols]
    df_filtered.to_csv(output_path, index=False)
    return df_filtered

def main():
    zip_path = '/home/robin/Research/qtype-eval/src/UDprofiling/UD_profiles/zip_UD_korean_polar.zip'
    csv_path = '/home/robin/Research/qtype-eval/src/UDprofiling/UD_profiles/UD-korean-polar-profile.csv'
    output_path = 'UD-korean-polar-annotated.csv'
    
    print("Extracting sentences from zip file...")
    sentences = unzip(zip_path)
    
    print("\nMerging with CSV data...")

    merged_df = merge_to_csv(sentences, csv_path, output_path)
    print(f"Rows after filtering duplicates: {len(merged_df)}")
    
if __name__ == "__main__":
    main()