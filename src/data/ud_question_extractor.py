import argparse
import conllu
import csv 
import json
import logging
import os
import re
import sys
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
logger = logging.getLogger(__name__)



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("ud_question_extractor.log"), logging.StreamHandler(sys.stdout)])


@dataclass
class LanguageConfig:
  """We begin by defining the language specific parameters for question-extraction"""
  code: str
  punctuation: List[str]
  content_words: List[str] = field(default_factory=list)
  auxiliaries: bool = False
  markers: Dict[str, str] = field(default_factory=dict)

  en_wh_words: str = r'\b(what*|who*|where*|when*|why*|how*|which*)\b'
  en_polar_starters: str = r'^(is|are|do|does|did|have|has|can|could|will|would|should|may|might)'
  embedded_verbs: str = r'\b(know|tell|confirm|explain|understand|think|show|mean|see)\b'
  fi_wh_words: str = r'\b(mikä|mitä|missä|mistä|mihin|milloin|miksi|kuka|ketkä|kumpi|kuinka|miten)\b'
  fi_special_content: str = r'\bmontako\b'
  ko_wh_words: str = r'(무엇|뭐|어디|언제|누구|왜|어떻게|무슨|어느|몇)'

  def is_question(self, sentence: conllu.TokenList) -> bool:

    text = sentence.metadata.get("text", "").strip()

    if any(text.endswith(mark) for mark in self.punctuation):
       return True

    
    
    tokens = list(sentence)
    if not tokens:
      return False
       
    if self.code == "ja" and tokens and tokens[-1]["form"] == "か":
      return True
    
    if self.code == "ko" and tokens:
        last_token = tokens[-1]
        for suffix in ["까", "니", "냐", "나요"]:
            if "form" in last_token and isinstance(last_token["form"], str) and last_token["form"].endswith(suffix):
                return True
        
    if self.code == "fi" and tokens:
        marked = False
        for token in tokens:
            if "form" in token and isinstance(token["form"], str) and token["form"].endswith(("ko", "kö")):
                marked = True
                break
        
        return text.endswith("?") and marked
        
    return False
  
  def classify_question_type(self, sentence: conllu.TokenList) -> str:
    text = sentence.metadata.get("text", "").lower()

    if self.code == "en":
        result = self._classify_english(text)
        return result if result is not None else "polar"
    elif self.code == "fi":
        return self._classify_finnish(text)
    elif self.code == "ko":
        return self._classify_korean(text)
    
    return "polar"
  
  def _classify_english(self, text):
    if re.match(f'^{self.en_wh_words[2:]}', text, re.I):
        return 'content'
        
    if re.match(f'{self.en_polar_starters}.*{self.embedded_verbs}.*{self.en_wh_words}', text, re.I):
        return 'polar'
        
    if re.match(self.en_polar_starters, text, re.I):
        return 'polar'

    if re.search(self.en_wh_words, text, re.I):
        return 'content'
        
    return 'polar'
    
  def _classify_finnish(self, text):
      words = text.replace(',', ' ').replace('?', ' ').split()
      for word in words:
          if word.endswith('ko') or word.endswith('kö'):
              return 'polar'
      if 'vai' in words or 'montako' in words:
          return 'content'
      return 'content'
  
         
  def _classify_korean(self, text):
      if re.search(self.ko_wh_words, text):
          return 'content'
      return 'polar'

class QuestionExtractor:
  """Extract questions from selected UD treebanks"""

  def __init__(self, config_path):
    self.config = self._load_config(config_path)
    self.language_configs = self._initialize_language_configs()
    self.stats = defaultdict(int)
  def _load_config(self, config_path):
    default_config = {
      "languages": {
        "en": {
          "punctuation": ["?"],
          "content_words": ["what", "who", "where", "when", "why", "how", "which"],
          "auxiliaries": True,
        },
        "fi": {
          "punctuation": ["?"],
          "content_question_words": ["mikä", "mitä", "missä", "mistä", "mihin", "milloin","miksi", "kuka", "ketkä", "kumpi", "kuinka", "miten"]
        },
        "ko": {
          "punctuation": ["?"],
          "content_question_words": ["무엇", "뭐", "어디", "언제", "누구", "왜", "어떻게", "무슨", "어느", "몇"]
        }
      },
      "extraction": {
        "min_tokens": 2,
        "max_tokens": 40,
        "exclude_fragmentary": True
      },
      "output": {
        "format": "conllu",
        "separate_by_type": True,
      }
    }
    if config_path and os.path.exists(config_path):
      with open(config_path, 'r', encoding='utf-8') as f:
        user_config = yaml.safe_load(f)
        self._update_config(default_config, user_config)
    return default_config
  
  def _update_config(self, default_config, user_config):
    for key, value in user_config.items():
      if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
        self._update_config(default_config[key], value)
      else:
        default_config[key] = value

  def _initialize_language_configs(self) -> Dict[str, LanguageConfig]:
    """Initialize language configurations from loaded config"""
    language_configs = {}
    for lang_code, lang_config in self.config["languages"].items():
      language_configs[lang_code] = LanguageConfig(
        code=lang_code,
        punctuation=lang_config.get("punctuation",["?"]),
        content_words = lang_config.get("content_words", []),
        auxiliaries = lang_config.get("auxiliaries", False), 
        markers = lang_config.get("markers", {})
      )
    return language_configs

  def extract_from_directory(self, treebank_path: str, output_path: str, language_list: Optional[List[str]]=None) -> Dict[str, int]:
    """
    Extract questions from treebanks.
  
    Args:
        treebank_path: Path to directory containing treebank files
        output_path: Path to directory where outputs will be written
        language_list: List of language codes to process
  
    Returns:
        Dictionary with statistics about extracted questions      
    """
    treebank_dir = Path(treebank_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
  
    print(f"Treebank directory absolute path: {treebank_dir.absolute()}")
    print(f"Directory exists: {treebank_dir.exists()}")
    print(f"Directory is a directory: {treebank_dir.is_dir()}")
    print(f"Directory contents: {list(treebank_dir.iterdir())}")
  
    self.stats = defaultdict(int)

    all_questions_by_lang = {}
  
    
    lang_dir_mapping = {
        'en': '*nglish*',  
        'fi': '*innish*',
        'ko': '*orean*',
        'ja': '*apanese*',
        'ru': '*ussian*',
        'ar': '*rabic*',
        'id': '*ndonesian*',
    }
  
    if not language_list:
        language_list = list(self.language_configs.keys())
    
    logger.info(f'Processing languages: {", ".join(language_list)}')
  
    # Find all conllu files for each language
    for lang_code in language_list:
        all_questions_by_lang[lang_code] = {"polar": [], "content": []}
        
        if lang_code not in self.language_configs:
          logger.warning(f"No configuration for language '{lang_code}', skipping")
          continue
          
        # Try to find the language directory using a more flexible pattern
        pattern = lang_dir_mapping.get(lang_code, f"*{lang_code}*")
        print(f"Looking for directories matching pattern: {pattern}")
        
        # First try with glob
        lang_dirs = list(treebank_dir.glob(pattern))
        print(f"Found directories (glob): {lang_dirs}")
        
        # If that doesn't work, try a manual search
        if not lang_dirs:
            lang_dirs = []
            try:
                for item in treebank_dir.iterdir():
                    if item.is_dir() and (lang_code.lower() in item.name.lower() or 
                                          any(s in item.name.lower() for s in [
                                              'english' if lang_code == 'en' else '',
                                              'finnish' if lang_code == 'fi' else '',
                                              'korean' if lang_code == 'ko' else '',
                                          ])):
                        lang_dirs.append(item)
                print(f"Found directories (manual): {lang_dirs}")
            except Exception as e:
                print(f"Error during manual search: {e}")
        
        if not lang_dirs:
            logger.warning(f"No directory found for language '{lang_code}' with pattern '{pattern}'")
            continue
          
        # Process all conllu files in the language directories
        for lang_dir in lang_dirs:
            logger.info(f"Searching for .conllu files in {lang_dir}")
            
            try:
                # List all files in directory to help debug
                print(f"All files in {lang_dir}: {list(lang_dir.iterdir())}")
                
                # Try with recursive glob
                conllu_files = list(lang_dir.glob("**/*.conllu"))
                print(f"Found .conllu files (recursive): {conllu_files}")
                
                # If that doesn't work, try regular glob
                if not conllu_files:
                    conllu_files = list(lang_dir.glob("*.conllu"))
                    print(f"Found .conllu files (non-recursive): {conllu_files}")


            except Exception as e:
                logger.error(f"Error searching for files in {lang_dir}: {e}")
                continue
              
            if not conllu_files:
                logger.warning(f"No .conllu files found in {lang_dir}")
                continue
              
            logger.info(f"Found {len(conllu_files)} .conllu files in {lang_dir}")
            for file_path in conllu_files:
                self._process_treebank_file(file_path, output_dir, lang_code)

            for file_path in conllu_files:
               polar_q, content_q = self._process_treebank_file(file_path, output_dir, lang_code)

               all_questions_by_lang[lang_code]["polar"].extend(polar_q)
    
               all_questions_by_lang[lang_code]["content"].extend(content_q)

    for lang_code, questions in all_questions_by_lang.items():
      if questions["polar"]:
        self._save_questions(questions["polar"], output_dir/f"all_{lang_code}_polar")
      if questions["content"]:
         self._save_questions(questions["content"], output_dir/f"all_{lang_code}_content")
            

  
    # Log extraction stats
    logger.info(f'Extraction complete: {dict(self.stats)}')
    return dict(self.stats)

  
  def _process_treebank_file(self, file_path: Path, output_dir: Path, lang_code: str):
    """ Run extractor over a single tree bank """
    logger.info(f'processing file: {file_path}')

    try:
      with open(file_path, 'r', encoding='utf-8') as f:
        sentences = list(conllu.parse(f.read()))
      
      logger.info(f"found {len(sentences)} sentences in {file_path}")


      
      polar_questions = []
      content_questions = []



      language_config = self.language_configs[lang_code]
      min_tokens = self.config["extraction"]["min_tokens"]
      max_tokens = self.config["extraction"]["max_tokens"]
      
      for sentence in sentences:
          if len(sentence) < min_tokens or len(sentence) > max_tokens:
              continue
              
          if self.config["extraction"]["exclude_fragmentary"]:
              if "fragmentary" in sentence.metadata.get("text", "").lower():
                  continue
                  
          if language_config.is_question(sentence):
              question_type = language_config.classify_question_type(sentence)
              
              if question_type == "polar":
                  polar_questions.append(sentence)
              else:
                  content_questions.append(sentence)
      
      self.stats[f"{lang_code}_polar"] += len(polar_questions)
      self.stats[f"{lang_code}_content"] += len(content_questions)
      
      base_name = file_path.stem
      
      if self.config["output"]["separate_by_type"]:
          if polar_questions:
              self._save_questions(polar_questions, output_dir / f"polar_{base_name}")
          
          if content_questions:
              self._save_questions(content_questions, output_dir / f"content_{base_name}")
      else:
          all_questions = polar_questions + content_questions
          if all_questions:
              self._save_questions(all_questions, output_dir / f"questions_{base_name}")
              
      logger.info(f"Extracted {len(polar_questions)} polar and {len(content_questions)} content questions from {file_path}")

      return polar_questions, content_questions
      
    except Exception as e:
      logger.error(f"Error processing file {file_path}: {str(e)}")

      return [], []
    

  def _save_questions(self, questions: List[conllu.TokenList], output_path: Path) -> None:

    """Save questions in the configured format."""

    output_format = self.config["output"]["format"]
    
    if output_format == "conllu":
        self._save_conllu(questions, f"{output_path}.conllu")
    elif output_format == "text":
        self._save_text(questions, f"{output_path}.txt")
    elif output_format == "json":
        self._save_json(questions, f"{output_path}.json")
    else:
        logger.warning(f"Unknown output format '{output_format}', defaulting to conllu")
        self._save_conllu(questions, f"{output_path}.conllu")

  def _save_conllu(self, questions: List[conllu.TokenList], output_path: str) -> None:
    """Save questions in CoNLL-U format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(question.serialize())
            f.write("\n")

  def _save_text(self, questions: List[conllu.TokenList], output_path: str) -> None:
    """Save questions as plain text, one per line."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in questions:
            text = question.metadata.get("text", "")
            if text:
                f.write(f"{text.strip()}\n")

  def _save_json(self, questions: List[conllu.TokenList], output_path: str) -> None:
    """Save questions as JSON with metadata."""
    json_data = []
    for question in questions:
        # Convert sentence to serializable format
        json_question = {
            "id": question.metadata.get("sent_id", ""),
            "text": question.metadata.get("text", ""),
            "tokens": [
                {
                    "id": token["id"],
                    "form": token["form"],
                    "lemma": token.get("lemma", ""),
                    "upos": token.get("upos", ""),
                    "xpos": token.get("xpos", ""),
                    "feats": token.get("feats", {}),
                    "head": token.get("head", None),
                    "deprel": token.get("deprel", "")
                }
                for token in question
            ]
        }
        json_data.append(json_question)
      
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)


def main():
  """Main function, testing CLI argument parsing still"""

  parser = argparse.ArgumentParser(description="Extract questions from Universal Dependencies treebanks.")

  parser.add_argument("--treebanks", type=str, required=True,
                      help="Directory containing UD treebank files")
  parser.add_argument("--output", type=str, required=True,
                      help="Directory to write output files")
  parser.add_argument("--config", type=str,
                      help="Path to YAML configuration file")
  parser.add_argument("--languages", type=str,
                      help="Comma-separated list of language codes to process (e.g., 'en,fi,ko')")
  parser.add_argument("--format", type=str, choices=["conllu", "text", "json"],
                      help="Output format (overrides config file)")


  args = parser.parse_args()


  extractor = QuestionExtractor(args.config)

  if args.format:
    extractor.config["output"]["format"] = args.format


  languages = args.languages.split(",") if args.languages else None
  stats = extractor.extract_from_directory(args.treebanks, args.output, languages)

  print("\nExtraction Summary:")
  for lang_type, count in sorted(stats.items()):
    print(f"  {lang_type}: {count}")


if __name__ == "__main__":
  main()






