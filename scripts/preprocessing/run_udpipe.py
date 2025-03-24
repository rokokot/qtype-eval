import argparse
import codecs
import requests
import os
import sys
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("udpipe_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def list_udpipe_models():
    """Get available UDPipe models from the API"""
    try:
        response = requests.get("https://lindat.mff.cuni.cz/services/udpipe/api/models")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error fetching models: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error connecting to UDPipe API: {e}")
        return None

def process_line_with_udpipe(text, model="english", output_format="conllu"):
    """Process a single line of text using UDPipe API"""
    url = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
    
    params = {
        "data": text.strip(),
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
            logger.error(f"Error processing text with {model}: {response.status_code}")
            logger.error(response.text)
            return None
    except Exception as e:
        logger.error(f"Error connecting to UDPipe API: {e}")
        return None

def process_file(input_file, model, output_file=None):
    """Process a file line by line with UDPipe"""
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + ".conllu"
    logger.info(f'starting processing with model: {model}')

    try:
        with codecs.open(input_file, 'r', 'utf-8') as f:
            lines = [line for line in f.readlines() if line.strip()]
        
        logger.info(f"Processing {len(lines)} lines from {input_file}")
        
        all_results = []
        for line in tqdm(lines, desc="Processing lines"):
            result = process_line_with_udpipe(line, model)
            if result:
                all_results.append(result)
        
        if all_results:
            with codecs.open(output_file, 'w', 'utf-8') as f:
                f.write('\n'.join(all_results))
            logger.info(f"Successfully wrote {len(all_results)} processed items to {output_file}")
            return True
        else:
            logger.error("No results were successfully processed")
            return False
    except Exception as e:
        logger.error(f"Error in processing file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Process a text file line by line using UDPipe API')
    parser.add_argument('-i', '--input', required=True, help='Input text file (one sentence per line)')
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
        logger.error(f"Input file not found: {args.input}")
        return
    
    if process_file(args.input, args.model, args.output):
        output_file = args.output if args.output else os.path.splitext(args.input)[0] + ".conllu"
        print(f"Successfully processed {args.input} -> {output_file}")
    else:
        print(f"Failed to process {args.input}")

if __name__ == '__main__':
    main()