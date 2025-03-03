import os
import sys
import csv
import glob
import codecs
import argparse
from collections import defaultdict




FEATURES = [
    'n_tokens',
    'lexical_density',
    'verbal_head_per_sent',
    'avg_max_depth',
    'avg_verb_edges'
    'avg_links_len',
    'avg_subordinate_chain_len',
    'subordinate_proposition_dist',
    
]

try:
    from senttok import Sentence, Token
    from compute_features import compute_features
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from senttok import Sentence, Token
    from compute_features import compute_features

def extract_sentence_data(input_file):
    sentences = {}
    sentence_data = {}
    sentence_tokens = []
    
    current_sent_id = None
    current_text = None
    
    with codecs.open(input_file, 'r', 'utf-8') as f:
        for line in f:
            if line.startswith('# sent_id'):
                current_sent_id = line.rstrip('\n').split('= ')[1]
            elif line.startswith('# text'):
                current_text = line.rstrip('\n').split('= ')[1]
            elif line == '\n' and sentence_tokens:
                if current_sent_id:
                    mysent = Sentence(sentence_tokens)
                    sentences[current_sent_id] = mysent
                    sentence_data[current_sent_id] = {
                        'text': current_text,
                        'id': current_sent_id
                    }
                sentence_tokens = []
                current_sent_id = None
                current_text = None
            elif not line.startswith('#') and line.strip():
                line = line.strip().split('\t')
                if '-' not in line[0]:  
                    sentence_tokens.append(Token(line))
    
    if sentence_tokens and current_sent_id:
        mysent = Sentence(sentence_tokens)
        sentences[current_sent_id] = mysent
        sentence_data[current_sent_id] = {
            'text': current_text,
            'id': current_sent_id
        }
    
    return sentences, sentence_data

def analyze_file(input_file, selected_features=None, language="", sentence_type=""):
    
    sentences, sentence_data = extract_sentence_data(input_file)
    

    results = []
    for sent_id, sentence in sentences.items():
        features = compute_features([sentence], None, type_analysis=0)
        
        result = {
            'sentence_id': sent_id,
            'text': sentence_data[sent_id].get('text', ''),
            'language': language,
            'type': sentence_type
        }
        
        for feature_name, feature_value in features.items():
            if selected_features is None or feature_name in selected_features:
                if isinstance(feature_value, list) and feature_value and isinstance(feature_value[0], tuple):
                    for key, value in feature_value:
                        result[key] = value
                else:
                    result[feature_name] = feature_value
        
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Extract linguistic features from sentences in CoNLL-U format')
    parser.add_argument('-p', '--path', required=True, help='path to CoNLL-U file or directory')
    parser.add_argument('-o', '--output', help='output directory (default: current directory)')
    parser.add_argument('-f', '--features', help='comma-separated list of features to include (overrides default features)')
    parser.add_argument('-a', '--all-features', action='store_true', help='include all available features')
    parser.add_argument('-l', '--language', default='unknown', help='language of the texts')
    parser.add_argument('-t', '--type', default='unknown', help='type of sentences (polar or content)')
    
    args = parser.parse_args()
    
    if args.all_features:
        selected_features = None  
    elif args.features:
        selected_features = args.features.split(',')
    else:
        selected_features = FEATURES
    
    input_files = []
    if os.path.isdir(args.path):
        input_files = glob.glob(os.path.join(args.path, '*.conllu'))
    else:
        input_files = [args.path]
    
    output_dir = args.output if args.output else '.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.csv")
        
        print(f"Processing {input_file}...")
        print(f"Extracting {len(selected_features) if selected_features else 'all'} features...")
        
        results = analyze_file(
            input_file, 
            selected_features, 
            args.language, 
            args.type
        )
        
        if results:
            columns = ['sentence_id', 'text', 'language', 'type']
            if selected_features:
                feature_columns = []
                for feature in selected_features:
                    feature_columns.append(feature)

                all_keys = set()
                for result in results:
                  all_keys.update(result.keys())
            
                for key in sorted(all_keys):
                    if key not in columns and any(key.startswith(f) for f in selected_features):
                        feature_columns.append(key)
                columns.extend(feature_columns)

            else:
                
                all_keys = set()
                for result in results:
                    all_keys.update(result.keys())
                
                for key in sorted(list(all_keys)):
                    if key not in columns:
                        columns.append(key)
            
            all_result_keys = set()
            for result in results:
                all_result_keys.update(result.keys())
            
            for key in sorted(all_result_keys):
                if key not in columns:
                    columns.append(key)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"Output written to {output_file}")
        else:
            print(f"No results found for {input_file}")
    
    print("Processing complete.")

if __name__ == '__main__':
    main()