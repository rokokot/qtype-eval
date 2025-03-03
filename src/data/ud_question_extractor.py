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
  
  def has_question_mark(self, text: str) -> bool:
      return bool(re.search(r'[\?? \؟]$', text.strip()))
  


  def classify_question_type(self, text: str) -> str:
    text = text.lower().strip()

    if self.code == "en":
        return self._classify_english(text)
    elif self.code == "fi":
        return self._classify_finnish(text)
    elif self.code == "ko":
        return self._classify_korean(text)
    elif self.code == "ja":
        return self._classify_japanese(text)
    elif self.code == "ru":
        return self._classify_russian(text)
    elif self.code == "ar":
        return self._classify_arabic(text)
    elif self.code == "id":
        return self._classify_indonesian(text)
    
    return None
  
  
  def _classify_english(self, text):

    qmark = self.has_question_mark(text)

    wh_pattern = r'\b(what|who|where|when|why|how|which)\b'
    polar_starters = r'^(is|Is|are|Are|Do|do|Does|does|Did|Did|Have|have|Has|has|Can|can|Could|could|will|Will|would|Would|should|Should|May|may|Might|might)'
    embedded_question = r'.*(know|tell|ask|explain|understand|wonder|confirm|remember|see|show).*(what|who|whom|whose|where|when|why|how|which).*'
    tag_question = r'.*,\s+(is(n\'t)?|are(n\'t)?|am not|do(n\'t)?|does(n\'t)?|did(n\'t)?|have(n\'t)?|has(n\'t)?|had(n\'t)?|can\'t|could(n\'t)?|wo(n\'t)|would(n\'t)?|should(n\'t)?)\s+(i|you|he|she|it|we|they)\b'
    
    if qmark:
       
        if re.search(embedded_question, text, re.I):
            return 'polar'
        if re.search(tag_question, text, re.I):
            return 'polar'
        elif re.match(polar_starters, text, re.I):
            return 'polar'
        elif re.search(wh_pattern, text, re.I):
            return 'content'
        
        else:
            return 'polar'  
    
    return None
    
  def _classify_finnish(self, text):
      qmark = self.has_question_mark(text)

      wh_pattern = r'\b(mik(?:ä|si)|mit(?:ä|en)|miss(?:ä|tä)|mihin|mill(?:oin|ä)|kuk(?:a|aan)|ket(?:ä|kä)|ken(?:en|eltä)|kumpi|kuinka|montako)\b'
      polar_pattern = r'\b\w+(?:ko|kö)\b'
        
      if qmark:
        if re.search(polar_pattern, text) and not re.search(wh_pattern, text, re.I):
          return 'polar'
        if re.search(wh_pattern, text, re.I) or 'vai' in text.split() or 'montako' in text.split():
          return 'content'
        return 'polar'
              
      return None
  
         
  def _classify_korean(self, text):
      qmark = self.has_question_mark(text)

      wh_pattern = r'(무엇|뭐|뭣|무슨|누구|누가|어디|어느|언제|왜|어째서|어떻게|어떤|몇|얼마)'
      ending_pattern = r'(까요|니까|나요|는가|을까|가요|니|까|냐|가|나)\s*\??$'
        
      if qmark or re.search(ending_pattern, text):
          
        if re.search(wh_pattern, text):
          return 'content'
        return 'polar'
      return None
  
  def _classify_japanese(self, text: str) -> str:
        qmark = self.has_question_mark(text)

        ka_pattern = r'か\s*[\?？]?$'
        wh_pattern = r'(何|なに|なん|どこ|どちら|いつ|誰|だれ|なぜ|どうして|どう|どのよう|どの|どんな|いくつ|いくら)'
        no_pattern = r'の\s*[\?？]?$' 
        deshou_pattern = r'(でしょう|だろう)\s*[\?？]?$'
        wh_pattern = r'(何|なに|なん|どこ|どちら|いつ|誰|だれ|なぜ|どうして|どう|どのよう|どの|どんな|いくつ|いくら|どれ|どういう)'
        
        
        
        
        if qmark or re.search(ka_pattern, text) or re.search(no_pattern, text) or re.search(deshou_pattern, text):
            if re.search(wh_pattern, text):
                # Check if not an embedded question (simplified)
                embedded_patterns = [r'(知って|わかって|教えて)[^か]*か', r'(分かる|知る|教える)[^か]*か']
                for pattern in embedded_patterns:
                    if re.search(pattern, text):
                        return 'polar'
                return 'content'
            return 'polar'
            
        return None
  
  def _classify_russian(self, text: str) -> str:
    qmark = self.has_question_mark(text)

    li_pattern = r'\s+ли\b'
    wh_pattern = r'\b(что|чего|чему|чем|кто|кого|кому|кем|где|куда|откуда|когда|почему|зачем|как|каким\s+образом|который|как(?:ой|ая|ое|ие)|сколько)\b'
      
    if qmark:
        if re.search(li_pattern, text):
            return 'polar'
                
        if re.search(wh_pattern, text):
            return 'content'
                
        return 'polar'
            
    return None
  

  def _classify_arabic(self, text: str) -> str:
    qmark = self.has_question_mark(text)

    polar_pattern = r'^(هل|أ)\b'
    wh_pattern = r'\b(ما(?:ذا)?|من|أين|وين|متى|لماذا|ليش|كيف|أي|كم)\b'
      
    if qmark:
        if re.search(polar_pattern, text):
            return 'polar'
      
        if re.search(wh_pattern, text):
            return 'content'
        
        return 'polar'
    elif re.search(r'^(هل|أ)\b', text):
        return 'polar'
    
    return None
      
  def _classify_indonesian(self, text: str) -> str:
    qmark = self.has_question_mark(text)

    polar_pattern = r'^(apakah|apa\s+kah|apa)\b'
    wh_pattern = r'\b(apa\s+yang|apa\s+saja|siapa(?:kah)?|di\s+mana|dimana|ke\s+mana|kemana|dari\s+mana|darimana|kapan|bila|mengapa|kenapa|bagaimana|yang\s+mana|berapa)\b'
      
    if qmark:
        if re.search(polar_pattern, text) and not re.search(r'\bapa\s+yang\b', text):
            return 'content'
        if re.search(wh_pattern, text):
            return 'content'

        return 'polar'
    return None
  
       

class QuestionExtractor:
    """Extract questions from text files using regex patterns"""
    
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.language_configs = self._initialize_language_configs()
        self.stats = defaultdict(int)
        self.detailed_stats = {
            "total": 0,
            "polar": 0,
            "content": 0,
            "by_language": {}
        }
    
    def _load_config(self, config_path):
        """Load configuration from file or use defaults"""
        default_config = {
            "languages": {
                "en": {
                    "punctuation": ["?"],
                    "content_words": ["what", "who", "where", "when", "why", "how", "which", "whom", "whose"],
                },
                "fi": {
                    "punctuation": ["?"],
                    "content_words": ["mikä", "mitä", "missä", "mistä", "mihin", "milloin", "miksi", "kuka", "ketkä", "kumpi", "kuinka", "miten", "montako"]
                },
                "ko": {
                    "punctuation": ["?"],
                    "content_words": ["무엇", "뭐", "어디", "언제", "누구", "왜", "어떻게", "무슨", "어느", "몇"]
                },
                "ja": {
                    "punctuation": ["?", "？"],
                    "content_words": ["何", "なに", "なん", "どこ", "どちら", "いつ", "誰", "だれ", "なぜ", "どうして", "どう", "どのよう", "どの", "どんな", "いくつ", "いくら"]
                },
                "ru": {
                    "punctuation": ["?"],
                    "content_words": ["что", "чего", "чему", "чем", "кто", "кого", "кому", "кем", "где", "куда", "откуда", "когда", "почему", "зачем", "как", "каким", "который", "какой", "какая", "какое", "какие", "сколько"]
                },
                "ar": {
                    "punctuation": ["?", "؟"],
                    "content_words": ["ما", "ماذا", "من", "أين", "وين", "متى", "لماذا", "ليش", "كيف", "أي", "كم"]
                },
                "id": {
                    "punctuation": ["?"],
                    "content_words": ["apa yang", "apa saja", "siapa", "siapakah", "di mana", "dimana", "ke mana", "kemana", "dari mana", "darimana", "kapan", "bila", "mengapa", "kenapa", "bagaimana", "yang mana", "berapa"]
                }
            },
            "extraction": {
                "min_tokens": 3,  # Now will be used for words instead of tokens
                "max_tokens": 40,  # Now will be used for words instead of tokens
                "exclude_fragmentary": True
            },
            "output": {
                "format": "conllu",  # Default back to conllu
                "separate_by_type": True,
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                self._update_config(default_config, user_config)
        
        return default_config
    
    def _update_config(self, default_config, user_config):
        """Recursively update configuration"""
        for key, value in user_config.items():
            if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                self._update_config(default_config[key], value)
            else:
                default_config[key] = value
    
    def _initialize_language_configs(self) -> Dict[str, LanguageConfig]:
        """Initialize language-specific configuration objects"""
        language_configs = {}
        for lang_code, lang_config in self.config["languages"].items():
            language_configs[lang_code] = LanguageConfig(
                code=lang_code,
                punctuation=lang_config.get("punctuation", ["?"]),
                content_words=lang_config.get("content_words", [])
            )
        return language_configs
    
    def extract_from_directory(self, treebank_path: str, output_path: str, language_list: Optional[List[str]]=None, txt_output_dir: Optional[str]=None) -> Dict[str, int]:
        """Extract questions from files in a directory"""
        treebank_dir = Path(treebank_path)
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        txt_dir = None
        if txt_output_dir:
            txt_dir = Path(txt_output_dir)
            txt_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Will save plain text questions to: {txt_dir}")
        
        logger.info(f"Treebank directory: {treebank_dir.absolute()}")
        
        self.stats = defaultdict(int)
        self.detailed_stats = {
            "total": 0,
            "polar": 0,
            "content": 0,
            "by_language": {}
        }
        
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
        
        for lang_code in language_list:
            all_questions_by_lang[lang_code] = {"polar": [], "content": []}
            
            if lang_code not in self.language_configs:
                logger.warning(f"No configuration for language '{lang_code}', skipping")
                continue
            
            if lang_code not in self.detailed_stats["by_language"]:
                self.detailed_stats["by_language"][lang_code] = {
                    "total": 0,
                    "polar": 0,
                    "content": 0,
                    "by_treebank": {}
                }
            
            pattern = lang_dir_mapping.get(lang_code, f"*{lang_code}*")
            logger.info(f"Looking for directories matching pattern: {pattern}")
            
            lang_dirs = list(treebank_dir.glob(pattern))
            
            if not lang_dirs:
                lang_dirs = []
                try:
                    for item in treebank_dir.iterdir():
                        if item.is_dir() and (
                                lang_code.lower() in item.name.lower() or
                                ('english' in item.name.lower() and lang_code == 'en') or
                                ('finnish' in item.name.lower() and lang_code == 'fi') or
                                ('korean' in item.name.lower() and lang_code == 'ko') or
                                ('japanese' in item.name.lower() and lang_code == 'ja') or
                                ('russian' in item.name.lower() and lang_code == 'ru') or
                                ('arabic' in item.name.lower() and lang_code == 'ar') or
                                ('indonesian' in item.name.lower() and lang_code == 'id')
                            ):
                                lang_dirs.append(item)
                except Exception as e:
                    logger.error(f"Error during manual search: {e}")
            
            if not lang_dirs:
                logger.warning(f"No directory found for language '{lang_code}' with pattern '{pattern}'")
                continue
            
            for lang_dir in lang_dirs:
                logger.info(f"Searching for .conllu files in {lang_dir}")
                conllu_files = []
                
                try:
                    conllu_files = list(lang_dir.glob("**/*.conllu"))
                    if not conllu_files:
                        conllu_files = list(lang_dir.glob("*.conllu"))
                except Exception as e:
                    logger.error(f"Error searching for files in {lang_dir}: {e}")
                    continue
                
                if not conllu_files:
                    logger.warning(f"No .conllu files found in {lang_dir}")
                    continue
                
                logger.info(f"Found {len(conllu_files)} .conllu files in {lang_dir}")
                
                for file_path in conllu_files:
                    treebank_name = file_path.stem
                    if treebank_name not in self.detailed_stats["by_language"][lang_code]["by_treebank"]:
                        self.detailed_stats["by_language"][lang_code]["by_treebank"][treebank_name] = {
                            "total": 0,
                            "polar": 0,
                            "content": 0,
                        }
                    
                    polar_q, content_q = self._process_file(file_path, lang_code, treebank_name)
                    
                    all_questions_by_lang[lang_code]["polar"].extend(polar_q)
                    all_questions_by_lang[lang_code]["content"].extend(content_q)
                    
                    # Save to output directory
                    base_name = file_path.stem
                    if self.config["output"]["separate_by_type"]:
                        if polar_q:
                            self._save_questions(polar_q, output_dir / f"polar_{base_name}")
                        if content_q:
                            self._save_questions(content_q, output_dir / f"content_{base_name}")
                    else:
                        all_q = polar_q + content_q
                        if all_q:
                            self._save_questions(all_q, output_dir / f"questions_{base_name}")
        
        # Save aggregated results
        for lang_code, questions in all_questions_by_lang.items():
            if questions["polar"]:
                self._save_questions(questions["polar"], output_dir / f"all_{lang_code}_polar")
                logger.info(f"Saved {len(questions['polar'])} polar questions for {lang_code}")
            
            if questions["content"]:
                self._save_questions(questions["content"], output_dir / f"all_{lang_code}_content")
                logger.info(f"Saved {len(questions['content'])} content questions for {lang_code}")
            
            if txt_dir and (questions["polar"] or questions["content"]):
                self._save_plain_text_questions(lang_code, questions, txt_dir)
        
        self._print_detailed_stats()
        return self.detailed_stats
    
    def _process_file(self, file_path: Path, lang_code: str, treebank_name: str) -> Tuple[List[any], List[any]]:
        """Process a single file and extract questions using text-based classification"""
        logger.info(f'Processing file: {file_path}')
        
        polar_questions = []
        content_questions = []
        
        try:
            # Parse the CoNLL-U file to maintain full structure
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences = list(conllu.parse(f.read()))
            
            logger.info(f"Found {len(sentences)} sentences in {file_path}")
            
            language_config = self.language_configs[lang_code]
            min_words = self.config["extraction"]["min_tokens"] 
            max_words = self.config["extraction"]["max_tokens"]
            
            for sentence in sentences:
                # Get the text from the sentence metadata
                text = sentence.metadata.get("text", "")
                if not text:
                    continue
                
                # Skip sentences that are too short or too long
                word_count = len(text.split())
                if word_count < min_words or word_count > max_words:
                    continue
                
                # Skip fragmentary sentences
                if self.config["extraction"]["exclude_fragmentary"]:
                    if "fragmentary" in text.lower():
                        continue
                
                # Classify the question using text patterns only
                question_type = language_config.classify_question_type(text)
                
                if question_type == "polar":
                    polar_questions.append(sentence)  # Store the full CoNLL-U structure
                elif question_type == "content":
                    content_questions.append(sentence)  # Store the full CoNLL-U structure
            
            # Update statistics
            treebank_stats = self.detailed_stats["by_language"][lang_code]["by_treebank"][treebank_name]
            treebank_stats["polar"] += len(polar_questions)
            treebank_stats["content"] += len(content_questions)
            treebank_stats["total"] += len(polar_questions) + len(content_questions)
            
            self.detailed_stats["by_language"][lang_code]["polar"] += len(polar_questions)
            self.detailed_stats["by_language"][lang_code]["content"] += len(content_questions)
            self.detailed_stats["by_language"][lang_code]["total"] += len(polar_questions) + len(content_questions)
            
            self.detailed_stats["polar"] += len(polar_questions)
            self.detailed_stats["content"] += len(content_questions)
            self.detailed_stats["total"] += len(polar_questions) + len(content_questions)
            
            self.stats[f"{lang_code}_polar"] += len(polar_questions)
            self.stats[f"{lang_code}_content"] += len(content_questions)
            
            logger.info(f"Extracted {len(polar_questions)} polar and {len(content_questions)} content questions from {file_path}")
            
            return polar_questions, content_questions
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return [], []
    
    def _save_questions(self, questions: List[any], output_path: Path) -> None:
        """Save questions to output file"""
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
        """Save questions in CoNLL-U format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for question in questions:
                f.write(question.serialize())
                f.write("\n")
    
    def _save_text(self, questions: List[any], output_path: str) -> None:
        """Save questions as plain text"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for question in questions:
                # Get text from conllu TokenList
                text = question.metadata.get("text", "")
                if text:
                    f.write(f"{text.strip()}\n")
    
    def _save_json(self, questions: List[any], output_path: str) -> None:
        """Save questions as JSON"""
        json_data = []
        for question in questions:
            json_question = {
                "id": question.metadata.get("sent_id", f"q{len(json_data)+1}"),
                "text": question.metadata.get("text", "").strip(),
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
    
    def _save_plain_text_questions(self, lang_code: str, questions: Dict[str, List[any]], output_dir: Path) -> None:
        """Save questions to plain text files by language and type"""
        if questions["polar"]:
            polar_path = output_dir / f"ud_{lang_code}_polar.txt"
            with open(polar_path, 'w', encoding='utf-8') as f:
                for question in questions["polar"]:
                    text = question.metadata.get("text", "")
                    if text:
                        f.write(f"{text.strip()}\n")
            logger.info(f"Saved {len(questions['polar'])} polar questions to {polar_path}")
        
        if questions["content"]:
            content_path = output_dir / f"ud_{lang_code}_content.txt"
            with open(content_path, 'w', encoding='utf-8') as f:
                for question in questions["content"]:
                    text = question.metadata.get("text", "")
                    if text:
                        f.write(f"{text.strip()}\n")
            logger.info(f"Saved {len(questions['content'])} content questions to {content_path}")
    
    def _print_detailed_stats(self):
        """Print detailed extraction statistics"""
        logger.info("\n=== Question Extraction Statistics ===")
        logger.info(f"Total questions extracted: {self.detailed_stats['total']}")
        
        if self.detailed_stats['total'] > 0:
            polar_percent = (self.detailed_stats['polar']/self.detailed_stats['total']*100)
            content_percent = (self.detailed_stats['content']/self.detailed_stats['total']*100)
            logger.info(f"Polar questions: {self.detailed_stats['polar']} ({polar_percent:.2f}%)")
            logger.info(f"Content questions: {self.detailed_stats['content']} ({content_percent:.2f}%)")
        
        logger.info("\nBreakdown by language:")
        for lang, stats in sorted(self.detailed_stats["by_language"].items()):
            if stats["total"] > 0:
                logger.info(f"\n{lang.upper()}:")
                logger.info(f"  Total questions: {stats['total']}")
                logger.info(f"  Polar questions: {stats['polar']} ({stats['polar']/stats['total']*100:.2f}%)")
                logger.info(f"  Content questions: {stats['content']} ({stats['content']/stats['total']*100:.2f}%)")
                
                logger.info("  By treebank:")
                for treebank, tb_stats in sorted(stats["by_treebank"].items()):
                    if tb_stats["total"] > 0:
                        logger.info(f"    {treebank}:")
                        logger.info(f"      Total: {tb_stats['total']}")
                        logger.info(f"      Polar: {tb_stats['polar']} ({tb_stats['polar']/tb_stats['total']*100:.2f}%)")
                        logger.info(f"      Content: {tb_stats['content']} ({tb_stats['content']/tb_stats['total']*100:.2f}%)")

def main():
    """Main function for CLI operation"""
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
    parser.add_argument("--stats-output", type=str,
                       help="Path to save detailed statistics as JSON")
    parser.add_argument("--txt-output", type=str,
                       help="Directory to save plain text files with only question text")
    
    args = parser.parse_args()
    
    extractor = QuestionExtractor(args.config)
    
    if args.format:
        extractor.config["output"]["format"] = args.format
    
    languages = args.languages.split(",") if args.languages else None
    stats = extractor.extract_from_directory(args.treebanks, args.output, languages, args.txt_output)
    
    if args.stats_output:
        stats_dir = os.path.dirname(args.stats_output)
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        
        with open(args.stats_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved detailed statistics to {args.stats_output}")
    
    print("Summary:")
    print(f"Total questions: {stats['total']}")
    print(f"Polar questions: {stats['polar']}")
    print(f"Content questions: {stats['content']}")
    
    for lang_code, lang_stats in sorted(stats["by_language"].items()):
        if lang_stats["total"] > 0:
            print(f"\n{lang_code}: {lang_stats['total']} total ({lang_stats['polar']} polar, {lang_stats['content']} content)")

if __name__ == "__main__":
    main()