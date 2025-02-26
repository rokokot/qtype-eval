import argparse
import conllu
import csv 
import json
import loggingimport os
import re
import sys
import yaml
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("ud_question_extractor.log"), logging.StreamHandler(sys.stdout)])



class LanguageConfig:
  """We begin by defining the language specific parameters for question-extraction"""
  code: str
  punctuation: List[str]
  content_words: List[str] = field(default_factory=list)
  auxiliaries: bool = False
  markers: Dict[str, str] = field(default_factory=dict)

  def is_question(self, sentence: conllu.model.Sentence) -> bool:

    # run checks in order
    if sentence.metadata.get("text", "").strip().endswith(tuple(self.question_markers)):
      return True
    
    tokens = list(sentence)
    if tokens and tokens["form"] in self.punctuation:
      return True
    
    if self.code == "ja" and tokens and tokens["form"] == "か":
      return True
    
    if self.code == "ko" and tokens:
      for suffix in ["까", "니", "냐", "나요"]:
        if tokens["form"].endswith(suffix):
          return True
        
    if self.code == "fi" and tokens:
      for token in tokens:
        if token["form"].endswith(("ko", "kö")):
          return True
        
    return False
  
  def class_question_type(self, sentence: conllu.models.Sentence) -> str:
    text = sentence.metadata.get("text", "").lower()

    for word in self.content_words:
      if re.search(rf'b{word}\b', text):
        return "content"

      if self.auxiliaries and len(sentence) > 2:
        first_token = sentence[0]
        if first_token["upos"] == "AUX":
          return "polar"
        
      if self.code == "fi":
        for token in sentence:
          if token["form"].endswith(("ko", "kö")):
            return "polar"
      
      if self.code in ["ja", "ko"] and not any(word in text for word in self.content_words):
        return "polar"

      return "polar"
  

  class QuestionExtractor:
    """Extract questions from selected UD treebanks"""

    def __init__(self, config_path):
      self.config = self._load_config(config_path)
      self.language_configs = self._initialize_language_configs()
      self.stats = defaultdict(int)

    def _load_config(self, config_path):
      default_config = {
        "lanugages": {
          "en": {
            "punctuation": ["?"],
            "content_words": ["what", "who", "where", "when", "why", "how", "which"],
            "auxiliaries": True,
          },
          "fi": {
            "question_markers": ["?"],
            "content_question_words": ["mikä", "mitä", "missä", "mistä", "mihin", "milloin","miksi", "kuka", "ketkä", "kumpi", "kuinka", "miten"]
          },
          "ko": {
            "question_markers": ["?"],
            "content_question_words": ["무엇", "뭐", "어디", "언제", "누구", "왜", "어떻게", "무슨", "어느", "몇"]
          }
        },
        "parameters": {
          "min_tokens": 2,
          "max_tokens": 40,
          "exclude_fragmentary": True
        },
        "output": {
          "format": "conllu",
          "separate_types": True,
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
          punctuation=lang_config["punctuation"],
          content_words = lang_config.get("content_words", []),
          auxiliaries = lang_config.get("markers", {})
        )
      return language_configs
    

    def extract_from_directory(self, treebank: str, ourput_dir: str, language: Optional[List[str]]=None) -> Dict[str, int]:
      """
      Extract questions from treebanks,

      args:
        treebank_dir = Path(treebank_dir)
        output_dir = Path()
        languages: list of codes to process

      returns:
        dictionary with statistics about extracted questions      
      """

      treebank_dir = Path(treebank_dir)
      output_dir = Path(output_dir)
      output_dir.mkdir(parents=True, exist_ok=True)

      self.stats = defaultdict(int)

      if not languages:
        languages = set()
        for file_path in treebank_dir.glob("*.conllu"):
          match = re.search(r'(^[a-z]{2})_', file_path.name)
          if match:
            languages.add(match.group(1))

      logger.info(f'processing languages: {', '.join(languages)}')


      for lang in languages:
        if lang not in self.language_configs:
          logger.warning(f" No configuration for language '{lang}', skipping")
          continue
        
        lang_pattern : f"{lang}_*.conllu"
        lang_files = list(treebank_dir.glob(lang_pattern))

        if not lang_files:
          logger.warning(f'No treebank files found for language '{lang}' with pattern '{lang_pattern}'')
          continue

        for file_path in lang_files:
          self._process_treebank_file(file_path, output_dir, lang)

      # keep basic stats for treebank
      logger.info(f'extraction complete: {dict(self.stats)}')
      return dict(self.stats)

    def _process_treebank_file(self, file_path: Path, output_dir: Path, lang: str):
      """ Run extractor over a single tree bank """
      logger.info(f'processing file: {file_path}')

      try:
        with open(file_path, 'r', encoding='utf-8') as f:
          sentences = list(conllu.parse(f.read()))
        
        logger.info(f"found {len(sentences)} sentences in {file_path}")


        # extract questions
        polar_questions = []
        content_questions = []



        language_config = self.language_configs[lang]
        min_tokens = self.config["extraction"]["min_tokens"]
        max_tokens = self.config["extraction"]["max_tokens"]
        
        for sentence in sentences:
            if len(sentence) < min_tokens or len(sentence) > max_tokens:
                continue
                
            if self.config["extraction"]["exclude_fragmentary"]:
                if "fragmentary" in sentence.metadata.get("text", "").lower():
                    continue
                    
            # Check if a sentence is a question
            if language_config.is_question(sentence):
                question_type = language_config.classify_question_type(sentence)
                
                if question_type == "polar":
                    polar_questions.append(sentence)
                else:
                    content_questions.append(sentence)
        
        self.stats[f"{lang}_polar"] += len(polar_questions)
        self.stats[f"{lang}_content"] += len(content_questions)
        
        base_name = file_path.stem
        
        if self.config["output"]["separate_by_type"]:
            if polar_questions:
                self._save_questions(polar_questions, output_dir / f"polar_{lang}_{base_name}")
            
            if content_questions:
                self._save_questions(content_questions, output_dir / f"content_{lang}_{base_name}")
        else:
            all_questions = polar_questions + content_questions
            if all_questions:
                self._save_questions(all_questions, output_dir / f"questions_{lang}_{base_name}")
                
        logger.info(f"Extracted {len(polar_questions)} polar and {len(content_questions)} content questions from {file_path}")
        
      except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")


def _save_questions(self, questions: List[conllu.models.Sentence], output_path: Path) -> None:
  
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

def _save_conllu(self, questions: List[conllu.models.Sentence], output_path: str) -> None:
    """Save questions in CoNLL-U format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(question.serialize())
            f.write("\n")

def _save_text(self, questions: List[conllu.models.Sentence], output_path: str) -> None:
    """Save questions as plain text, one per line."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for question in questions:
            text = question.metadata.get("text", "")
            if text:
                f.write(f"{text.strip()}\n")

def _save_json(self, questions: List[conllu.models.Sentence], output_path: str) -> None:
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
  parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")

  args = parser.parse_args()


  # Initialize extractor
  extractor = QuestionExtractor(args.config)

  # Override output format if specified
  if args.format:
    extractor.config["output"]["format"] = args.format

  # Process languages 
  languages = args.languages.split(",") if args.languages else None

  # Run extraction
  stats = extractor.extract_from_directory(args.treebanks, args.output, languages)

  print("\nExtraction Summary:")
  for lang_type, count in sorted(stats.items()):
    print(f"  {lang_type}: {count}")


if __name__ == "__main__":
  main()




  

