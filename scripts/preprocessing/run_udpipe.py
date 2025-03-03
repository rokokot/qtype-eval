import argparse
import codecs
import json
import requests
import os
import sys
from urllib.parse import quote

# to run the scrip, use: python scripts/preprocessing/run_udpipe.py -i path/to/input.txt -o path/to/output.conllu -m 'model-name'


def list_udpipe_models():
    """Get available UDPipe models from the API"""
    try:
        response = requests.get("https://lindat.mff.cuni.cz/services/udpipe/api/models")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching models: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to UDPipe API: {e}")
        return None

def process_text_with_udpipe(text, model="english", output_format="conllu"):
    """Process text using UDPipe API"""
    url = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
    
    params = {
        "data": text,
        "model": model,
        "tokenizer": "",  
        "tagger": "",     
        "parser": "",   
        "output": output_format
    }
    
    try:
        response = requests.post(url, data=params)
        if response.status_code == 200:
            result = response.json()
            return result['result']
        else:
            print(f"Error processing text: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error connecting to UDPipe API: {e}")
        return None

def process_file(input_file, model, output_file=None):
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".conllu"
    
    try:
        with codecs.open(input_file, 'r', 'utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
    
    processed_text = process_text_with_udpipe(text, model)
    
    if processed_text:
        try:
            with codecs.open(output_file, 'w', 'utf-8') as f:
                f.write(processed_text)
            return True
        except Exception as e:
            print(f"Error writing output file: {e}")
            return False
    else:
        return False

def main():
    parser = argparse.ArgumentParser(description='Process a text file using UDPipe API')
    parser.add_argument('-i', '--input', required=True, help='Input text file')
    parser.add_argument('-o', '--output', help='Output CoNLL-U file (default: input filename with .conllu extension)')
    parser.add_argument('-m', '--model', default='english', help='Language model to use (default: english)')
    parser.add_argument('-l', '--list-models', action='store_true', help='List available UDPipe models and exit')
    
    args = parser.parse_args()
    
    if args.list_models:
        models_info = list_udpipe_models()
        if models_info:
            print("Available UDPipe models:")
            for model_name, components in models_info['models'].items():
                print(f"  - {model_name}: {', '.join(components)}")
            print(f"Default model: {models_info['default_model']}")
        return
    
    if not os.path.isfile(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    if process_file(args.input, args.model, args.output):
        output_file = args.output if args.output else os.path.splitext(args.input)[0] + ".conllu"
        print(f"Successfully processed {args.input} -> {output_file}")
    else:
        print(f"Failed to process {args.input}")

if __name__ == '__main__':
    main()