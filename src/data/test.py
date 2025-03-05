import argilla as rg
import pandas as pd
from pathlib import Path
import uuid



FEATURES = [
    'n_tokens', 'tokens_per_sent', 'lexical_density',
    'avg_max_depth', 'avg_links_len', 'avg_max_depth', 'avg_token_per_clause',
    'avg_prepositional_chain_len', 'n_prepositional_chains', 'verbal_head_per_sent'
]

def process_file(file_path):
 
    language = Path(file_path).stem.split('-')[1]
    df = pd.read_csv(file_path)
    

    text_col = df.columns[0]
    print(f"Processing {file_path.name} - Text column: {text_col}")
    
    records = []
    for _, row in df.iterrows():
        if pd.isna(row[text_col]):
            continue
            

        metadata = {
            'language': language,
            'file': Path(file_path).stem
        }
        for feature in FEATURES:
            if feature in df.columns:
                try:
                    value = row[feature] if not pd.isna(row[feature]) else 0.0
                    metadata[feature] = round(float(value), 3)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {feature}: {row[feature]}")
                    metadata[feature] = 0.0

        record = rg.Record(
            fields={
                "text": str(row[text_col]).strip()
            },
            metadata=metadata
        )
        records.append(record)
    
    print(f"Created {len(records)} records from {len(df)} rows")
    if records:
        print("\nSample metadata from first record:")
        for k, v in sorted(records[0].metadata.items())[:5]:
            print(f"  {k}: {v}")
    return records

def create_argilla_dataset(name="ud-annotated-questions"):
  
    
    settings = rg.Settings(
        guidelines="Universal Dependencies question sentence",
        allow_extra_metadata=True,
        fields=[
            rg.TextField(name="text", required=True)
        ],
        questions=[
            rg.LabelQuestion(
                name="verified",
                title="Verification Status",
                labels=["yes", "no"],
                description="Has this record been verified?"
            )
        ]
    )

    return rg.Dataset(name=name, settings=settings)

def main():
    # Initialize Argilla client
    client = rg.Argilla(
        api_url="https://rokii3-complexity-annotated-questions.hf.space",
        api_key="CJ83an24doRX_To1xKivqZ0H6KLGBxDftdG0iSWcWtuTiP33VxpZ0z-d2ODQgKYQoc0DQFj_SskZD4jlumdsAiRi18YZ6sdK16p_XOTDl6M"

    )
    
    # Create and initialize dataset
    try:
        dataset = create_argilla_dataset()
        dataset.create()
        print("Successfully created dataset")
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return

    # Process files
    input_dir = Path('/home/robin/Research/qtype-eval/data/annotated_UD_questions')
    files = list(input_dir.glob("*.csv"))
    
    all_records = []
    for file_path in files:
        records = process_file(file_path)
        all_records.extend(records)
    
    if not all_records:
        print("No records found!")
        return
    
    print(f"\nTotal records to upload: {len(all_records)}")
    print("\nSample record:")
    print(all_records[0])

    records_data = []


    
   
    batch_size = 50
    total_batches = (len(all_records) + batch_size - 1) // batch_size
    
    for i in range(0, len(all_records), batch_size):
        batch = all_records[i:i + batch_size]
        try:
            dataset.records.log(records=batch)
            print(f"Uploaded batch {i//batch_size + 1}/{total_batches}")
        except Exception as e:
            print(f"Error uploading batch: {str(e)}")
            print("\nFirst record in failed batch:")
            print(batch[0])
            break

if __name__ == "__main__":
    main()