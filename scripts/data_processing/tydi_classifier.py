"""
This script processes the TyDi QA dataset to identify and classify questions across mutliple languages. The core functionality of the file is metadata analysis, question identification, type classifcation, and question filtering. 

The script analyzes both human annotations from the TyDi dataset and linguistic patterns to classify questions. For each question, it:

1. Examines the 'yes_no_answer' annotations to determine if human annotators classified it as a polar question
2. Applies language-specific regex patterns to identify question types based on linguistic markers
4. Validates content questions by checking for the presence of language-specific wh-words
5. Filters out questions with inconsistent annotations or without clear question markers

It supports seven languages: English, Russian, Japanese, Arabic, Finnish, Korean, and Indonesian. For each language, the script outputs plain text files of polar and content questions, along with comprehensive statistics about classification decisions.

The final dataset was generate using the following flags:

    python tydi_classifier.py --cached-dataset data/tydi_validation.pkl --txt-output TyDi-questions/ --output results/classification.csv --simple-output results/tydi_simple.csv

The script takes the following arguments:
    --split              Dataset split to process (train or validation)
    --output             Path to output CSV with detailed classification results
    --languages          Comma-separated list of languages to process
    --txt-output         Directory to save text files with one question per line
    --cached-dataset     Path to a cached version of the dataset
    --save-dataset       Path to save the dataset for future use
    --use-classifier     Use pattern-based classifier for final decisions, instead of relying on the annotations

Author: Robin Kokot
Date: March 2025
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler("tydi_classifier.log"), logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)


LANGUAGE_MAP = {
    "english": "en",
    "russian": "ru",
    "japanese": "ja",
    "arabic": "ar",
    "finnish": "fi",
    "korean": "ko",
    "indonesian": "id"
}

CLASSIFIER_POLAR_SET = {"english", "russian", "japanese", "arabic", "finnish", "korean"}


class QuestionClassifier:
    """Classifier for questions based on patterns from ud_question_extractor"""
    
    def __init__(self, language):
        self.language = language
        
        # English patterns
        self.en_wh_words = r'\b(what*|who*|where*|when*|why*|how*|which*)\b'
        self.en_polar_starters = r'^(is|Is|are|Are|Do|do|Does|does|Did|Did|Have|have|Has|has|Can|can|Could|could|will|Will|would|Would|should|Should|May|may|Might|might)'
        self.embedded_verbs = r'\b(know|tell|confirm|explain|understand|think|show|mean|see)\b'
        
        # Finnish patterns
        self.fi_wh_words = r'\b(mik(?:ä|si)|montako|mit(?:ä|en)|miss(?:ä|tä)|mihin|mill(?:oin|ä)|kuk(?:a|aan)|ket(?:ä|kä)|ken(?:en|eltä)|kumpi|kuinka|montako)\b'
        self.fi_polar = r'\b\w+(?:ko|kö)\b'

        # Korean patterns
        self.ko_wh_words = r'(무엇|뭐|뭣|무슨|누구|누가|어디|어느|언제|왜|어째서|어떻게|어떤|몇|얼마)'
        self.ko_ending_pattern = r'(습니까|읍니까|ㅂ니까|나요|가요|군요|네요|죠|인가요|인가|은가요|는가요|ㄴ가요|까요|을까요|를까요|ㄹ까요|지요|하나요|한가요|할까요|하겠나요|겠나요)\s*\??$'


        # Japanese patterns
        self.ja_polar_pattern = [
            r'(か|かな|のか|のかな|だろうか|でしょうか|ですか)[\s。]*[\?？]*$',  
            r'(ますか|ませんか|ましたか|ませんでしたか)[\s。]*[\?？]*$',         
            r'(ある|ない|いる|いない)(か)[\s。]*[\?？]*$',                     
            r'は.*(ですか|ますか|でしょうか)[\s。]*[\?？]*$',                  
            r'が.*(ですか|ますか|でしょうか)[\s。]*[\?？]*$',                  
            r'を.*(ますか|ませんか)[\s。]*[\?？]*$',                          
            r'(でしょう|だろう)[\s。]*[\?？]*$',                              
        ]

        self.ja_wh_words = [
            r'(何|なに|なん)\b',           
            r'(誰|だれ)\b',               
            r'どこ\b',                    
            r'いつ\b',                    
            r'(なぜ|どうして)\b',          
            r'(どう|どのよう|どのように)\b', 
            r'どの\b',                    
            r'どんな\b',                  
            r'(いくつ|いくら)\b',          
            r'どれ\b',                    
            r'どちら\b',                  
            r'どのくらい\b',              
            r'いかが\b',    
        ]

        # Russian patterns
        self.ru_li_pattern = r'\s+ли\b'
        self.ru_wh_words = r'\b(что|чего|чему|чем|кто|кого|кому|кем|где|куда|откуда|когда|почему|зачем|как|каким\s+образом|который|как(?:ой|ая|ое|ие)|сколько)\b'


        # Arabic patterns
        self.ar_polar_pattern = r'^(هل|أ)\b' 
        self.ar_wh_words = r'\b(ما(?:ذا)?|من|أين|وين|متى|لماذا|ليش|كيف|أي|كم)\b'

        # Indonesian patterns
        self.id_polar_pattern = r'^(apakah|apa\s+kah|apa)\b'  
        self.id_wh_words = r'\b(apa\s+yang|apa\s+saja|siapa(?:kah)?|di\s+mana|dimana|ke\s+mana|kemana|dari\s+mana|darimana|kapan|bila|mengapa|kenapa|bagaimana|yang\s+mana|berapa)\b'
    
    def classify(self, text):
        """Classify a question as polar or content"""
        if self.language == "en":
            return self._classify_english(text)
        elif self.language == "fi":
            return self._classify_finnish(text)
        elif self.language == "ko":
            return self._classify_korean(text)
        elif self.language == "ja":
            return self._classify_japanese(text)
        elif self.language == "ru":
            return self._classify_russian(text)
        elif self.language == "ar":
            return self._classify_arabic(text)
        elif self.language == "id":
            return self._classify_indonesian(text)
        
        return None
    
    def _classify_english(self, text):
        text = text.lower()
        
        if re.match(self.en_wh_words, text, re.I):
            return 'content'
        
        if re.match(f'{self.en_polar_starters}.*{self.embedded_verbs}.*{self.en_wh_words}', text, re.I):
            return 'polar'
        
        if re.match(self.en_polar_starters, text, re.I):
            return 'polar'
        
        if re.search(self.en_wh_words, text, re.I):
            return 'content'
        
        return None
    
    def _classify_finnish(self, text):
        text = text.lower()

        if re.search(self.fi_wh_words, text, re.I):
            return 'content'

        if re.search(self.fi_polar, text):
            return 'polar'
    
        return None
    
    def _classify_korean(self, text):

        if len(text) < 4:
            return None
        
        strong_polar_indicators = [
            r'(인가요|인가)\?$',
            r'(합니까|습니까|읍니까)\?$',
            r'(하나요|나요|가요|군요|네요)\?$',
            r'(하십니까|십니까)\?$',
            r'(한가요|은가요|는가요)\?$',
            r'(할까요|을까요|를까요)\?$',
            r'(하겠\w+가|겠\w+가)\?$',
            r'\?$' ]

        for pattern in strong_polar_indicators:
            if re.search(pattern, text):
                obvious_wh = re.match(r'^(무엇|누구|어디|언제|왜|어떻게) ', text) or re.match(r'^(무슨|어느|몇) \w+ ', text)
                if not obvious_wh:
                    return 'polar'


        if re.search(self.ko_wh_words, text):
            return 'content'
    
        if text.endswith('?'):
            return 'polar'
    
        if re.search(self.ko_ending_pattern, text):
            return 'polar'
    
        wh_in_polar_contexts = [
            r'(무엇|뭐|뭣|무슨).+(인가요|일까요|입니까)',
            r'(누구|누가).+(인가요|일까요|입니까)',
            r'(어디|어느).+(인가요|일까요|입니까)'
        ]

        for pattern in wh_in_polar_contexts:
            if re.search(pattern, text):
                return 'polar'


        return None   
    
    def _classify_japanese(self, text):
      
        wh_patterns = [
            r'(何|なに|なん)\b',           
            r'(誰|だれ)\b',               
            r'どこ\b',                    
            r'いつ\b',                    
            r'(なぜ|どうして)\b',         
            r'(どう|どのよう|どのように)\b',
            r'どの\b',                    
            r'どんな\b',                  
            r'(いくつ|いくら)\b',          
            r'どれ\b',                    
            r'どちら\b',                  
            r'どのくらい\b',              
            r'いかが\b',                  
        ]
        
        has_wh_word = False
        for pattern in wh_patterns:
            if re.search(pattern, text):
                has_wh_word = True
                break

        
        strong_polar_patterns = [
            r'(ですか|ますか|でしょうか)[\s。]*[\?？]*$',
            r'(ある|ない|あります|ありません|います|いません|できる|できない)(か|の|ですか)[\s。]*[\?？]*$',
            r'(〜ですか|〜ますか|〜でしょうか)[\s。]*[\?？]*$',
            
            r'^.+は.+(?<!(なに|なん|だれ|どこ|いつ|なぜ|どう))(ですか|ますか|でしょうか)[\s。]*[\?？]*$',
            r'(ある|ない|あります|ありません|です|ます|ません)(か)[\s。]*[\?？]*$',
            
            r'^.+(は|が|を).+[でし]すか[\?？]?$',
            r'^.+(できます|できる|あります|ある|います|いる)(か|のか|ですか)[\?？]?$',
            
            r'[^か]+(か|の)[\s。]*[\?？]*$',
        ]

        if (text.endswith('?') or text.endswith('？')) and not has_wh_word:
            return 'polar'
        
        for pattern in strong_polar_patterns:
            if re.search(pattern, text):
                if not has_wh_word:
                    return 'polar'
                
                if re.search(r'(ですか|ますか|でしょうか)[\s。]*[\?？]*$', text):
                    if re.search(r'(なに|なん|だれ|どこ|いつ|なぜ|どう).*(ですか|ますか)[\?？]?$', text):
                        return 'content'
                    else:
                        return 'polar'

        if has_wh_word:
            strong_content_patterns = [
                r'(何|なに|なん)(が|を|に|の|は|で)',
                r'(誰|だれ)(が|を|に|の|は|で)',
                r'どこ(が|を|に|の|は|で|へ)',
                r'いつ(から|まで|に|の|は|で)',
                r'(なぜ|どうして)(に|は|で|を)',
                r'どのよう(に|な|は|で)',
                
                r'^(何|なに|なん|誰|だれ|どこ|いつ|なぜ|どうして)',
                r'^.*(何|なに|なん|誰|だれ|どこ|いつ|なぜ|どうして).*[\?？]$',
                
                r'^.+は.*(何|なに|なん|誰|だれ|どこ|いつ|なぜ|どうして)',
            ]
            
            for pattern in strong_content_patterns:
                if re.search(pattern, text):
                    return 'content'
            
            return 'content'

        if (text.endswith('?') or text.endswith('？')):
            if re.search(r'^.+(は|が|を).+', text):
                return 'polar'
        
        if re.search(r'.+か[\s。]*$', text) and not has_wh_word:
            return 'polar'
            
        if re.search(r'(ます|ません|でしょう|ですか|あります|ありません)', text) and not has_wh_word:
            if text.endswith('?') or text.endswith('？'):
                return 'polar'
        
        return None
            
    def _classify_russian(self, text):
        text = text.lower()
        if re.search(self.ru_li_pattern, text):
            return 'polar'
        
        if re.search(self.ru_wh_words, text):
            return 'content'
        
        return None
    
    def _classify_arabic(self, text):
        if re.search(self.ar_polar_pattern, text):
            return 'polar'
        
        if re.search(self.ar_wh_words, text):
            return 'content'
        
        return None
    
    def _classify_indonesian(self, text):
        text = text.lower()

        if re.search(self.id_polar_pattern, text):
            if re.search(r'\bapa\s+yang\b', text):
                return 'content'
            return 'polar'
        
        if re.search(self.id_wh_words, text):
            return 'content'
        
        return None
        

class TyDiClassifier:
    
    def __init__(self):
        self.classifiers = {}
        for tydi_lang, ud_lang in LANGUAGE_MAP.items():
            self.classifiers[tydi_lang] = QuestionClassifier(ud_lang)
        
        self.annotation_counts={
            "total": 0,
            "single_annotator": 0,
            "multiple_annotators": 0,
            "yes_distribution": Counter(),
            "by_language": {}
        }

        for lang in LANGUAGE_MAP:
            self.annotation_counts["by_language"][lang] = {
            "total": 0,
            "single_annotator": 0,
            "multiple_annotators": 0,
            "yes_distribution": Counter()
        }
        
        self.stats = {
            "total": 0,
            "polar_by_annotation": 0,
            "polar_by_classifier": 0,
            "agreed_polar": 0,
            "agreed_content": 0,
            "agreement": 0,
            "disagreement": 0,
            "by_language": {}
        }
        
        for lang in LANGUAGE_MAP:
            self.stats["by_language"][lang] = {
                "total": 0,
                "polar_by_annotation": 0,
                "polar_by_classifier": 0,
                "agreed_polar": 0,
                "agreed_content": 0,
                "agreement": 0,
                "disagreement": 0
            }
    
    def has_unanimous_annotations(self, annotations):
        if isinstance(annotations, dict) and 'yes_no_answer' in annotations:
            yes_no_answers = annotations['yes_no_answer']
            if len(yes_no_answers) <= 1:
                return True
                
            is_polar = [ans == 'YES' or ans == 'NO' for ans in yes_no_answers]
            return all(is_polar) or not any(is_polar)
            
        return True 
    
    def load_tydi_dataset(self, split="validation", cached_path=None):

        pickle_path = None
        if cached_path:
            if cached_path.endswith('.csv'):
                pickle_path = cached_path.replace('.csv', '.pkl')
            else:
                pickle_path = cached_path

            if os.path.exists(pickle_path):
              try:
                  with tqdm(total=1, desc=f"Loading {split} data from cache") as pbar:
                      df = pd.read_pickle(pickle_path)
                      pbar.update(1)
                  logger.info(f"Successfully loaded data from {pickle_path}")
                  return df
              except Exception as e:
                  logger.warning(f"Failed to load cached data from {pickle_path}: {e}")
                  logger.warning("Will download dataset from HuggingFace instead")

        logger.info(f"Download TyDi QA dataset ({split} split) from HuggingFace")

        with tqdm(total=2, desc=f"Downloading {split} split") as pbar:
            dataset = load_dataset("google-research-datasets/tydiqa", "primary_task")
            pbar.update(1)
            
            dataset.set_format("pandas")
            df = dataset[split][:].copy()
            pbar.update(1)

        if pickle_path:
            try:
              os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
              logger.info(f"Saving data to: {pickle_path}")
              with tqdm(total=1, desc=f"Saving {split} data to disk") as pbar:
                  df.to_pickle(pickle_path)
                  pbar.update(1)
            except Exception as e:
              logger.warning(f"Failed to save dataset to {pickle_path}: {e}")
              logger.warning("Continuing without saving cache")
      
        return df
    
    def analyze_annotations(self, df, filter_languages=None):
        
        if filter_languages:
            df = df[df['language'].str.lower().isin(filter_languages)]
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing annotations"):
            language = row['language'].lower()
            
            if language not in self.classifiers:
                continue
            
            yes_no_answers = row['annotations'].get('yes_no_answer', [])
            
            self.annotation_counts["total"] += 1
            
            if len(yes_no_answers) > 1:
                self.annotation_counts["multiple_annotators"] += 1
                
                yes_count = sum(1 for ans in yes_no_answers if ans == 'YES')
                self.annotation_counts["yes_distribution"][yes_count] += 1
                
                lang_counts = self.annotation_counts["by_language"][language]
                lang_counts["total"] += 1
                lang_counts["multiple_annotators"] += 1
                lang_counts["yes_distribution"][yes_count] += 1
            else:
                self.annotation_counts["single_annotator"] += 1
                
                if yes_no_answers and yes_no_answers[0] == 'YES':
                    self.annotation_counts["yes_distribution"][1] += 1
                else:
                    self.annotation_counts["yes_distribution"][0] += 1
                
                lang_counts = self.annotation_counts["by_language"][language]
                lang_counts["total"] += 1
                lang_counts["single_annotator"] += 1
                if yes_no_answers and yes_no_answers[0] == 'YES':
                    lang_counts["yes_distribution"][1] += 1
                else:
                    lang_counts["yes_distribution"][0] += 1
    
    def is_polar_by_annotation(self, annotations):
        if isinstance(annotations, dict) and 'yes_no_answer' in annotations:
            return any(ans == 'YES' or ans == 'NO' for ans in annotations['yes_no_answer'])
        return False
    
    def classify_question(self, question_text, language, annotation_class=None):
    
        if language in ["japanese", "korean"]:
            return annotation_class or "content"
        
        if language not in self.classifiers:
            return annotation_class or "content"
        
        classifier_result = self.classifiers[language].classify(question_text)

        if classifier_result not in ["polar", "content"]:
            return annotation_class or "content"
    
        return classifier_result
    
    def include_question(self, annotation_class, classifier_class, language):
      
        if language in ["japanese", "korean"] and annotation_class in ["polar", "content"]:
            return True
            
        if annotation_class == "content" and classifier_class == "content":
            return True
                
        if annotation_class == "polar" or classifier_class == "polar":
            if language == "indonesian":
                return annotation_class == classifier_class
                
            if language in CLASSIFIER_POLAR_SET:
                return classifier_class == "polar"
                    
        return False
    
    
    def has_any_polar_annotation(self, annotations):
        
        if isinstance(annotations, dict) and 'yes_no_answer' in annotations:
        
            yes_no_answers = annotations['yes_no_answer']
        
            return any(ans == 'YES' or ans == 'NO' for ans in yes_no_answers)
        
        return False

    def process_dataset(self, df, output_path=None, filter_languages=None, txt_output_dir=None, use_classifier=False, split="validation"):
            results = []
            
            self.analyze_annotations(df, filter_languages)

            if filter_languages:
                df = df[df['language'].str.lower().isin(filter_languages)]
            
            filtered_count = 0
            skipped_questions = {lang: 0 for lang in LANGUAGE_MAP}
            filtered_content = {lang: 0 for lang in LANGUAGE_MAP}
            added_polar = {lang: 0 for lang in LANGUAGE_MAP}

            for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying questions"):
                question_text = row['question_text']
                language = row['language'].lower()
                
                if language not in self.classifiers:
                    continue
                
                # Special handling for Japanese and Korean
                if language in ["japanese", "korean"]:
                    # Check if any annotator marked it as polar
                    any_polar = self.has_any_polar_annotation(row['annotations'])
                    
                    # For unanimous agreement
                    unanimous = self.has_unanimous_annotations(row['annotations'])
                    
                    # Get annotation class based on majority (as before)
                    polar_by_annotation = self.is_polar_by_annotation(row['annotations'])
                    annotation_class = "polar" if polar_by_annotation else "content"
                    
                    # RELAXED CRITERIA:
                    # 1. Always include if any annotator marked as polar
                    # 2. For content questions, still verify wh-words
                    
                    # Case 1: Any annotator marked as polar -> include as polar
                    if any_polar:
                        classifier_class = "polar"
                        include_question = True
                        final_class = "polar"
                        
                        # Track how many we're adding that weren't unanimous
                        if not unanimous or annotation_class != "polar":
                            added_polar[language] += 1
                        
                    # Case 2: Content question -> verify unanimous + has wh-words
                    elif annotation_class == "content":
                        # Skip if not unanimous agreement
                        if not unanimous:
                            skipped_questions[language] += 1
                            continue
                        
                        # Check if it contains wh-words
                        has_wh_word = False
                        
                        # Handle different pattern structures for Japanese and Korean
                        if language == "korean":
                            # Korean has a single regex pattern
                            if re.search(self.classifiers[language].ko_wh_words, question_text):
                                has_wh_word = True
                        else:  # Japanese
                            # Japanese has a list of patterns
                            for pattern in self.classifiers[language].ja_wh_words:
                                if re.search(pattern, question_text):
                                    has_wh_word = True
                                    break
                        
                        # Skip content questions without wh-words
                        if not has_wh_word:
                            filtered_content[language] += 1
                            continue
                        
                        classifier_class = "content"
                        include_question = True
                        final_class = "content"
                    
                    # Case 3: Neither polar nor valid content -> skip
                    else:
                        skipped_questions[language] += 1
                        continue
                else:
                    # For other languages, use the regular classification logic
                    polar_by_annotation = self.is_polar_by_annotation(row['annotations'])
                    annotation_class = "polar" if polar_by_annotation else "content"
                    classifier_class = self.classify_question(question_text, language, annotation_class)
                    include_question = self.include_question(annotation_class, classifier_class, language)
                    
                    final_class = None
                    if include_question:
                        if annotation_class == "content" and classifier_class == "content":
                            final_class = "content"
                        elif language in CLASSIFIER_POLAR_SET and classifier_class == "polar":
                            final_class = "polar"
                        elif language == "indonesian" and annotation_class == classifier_class:
                            final_class = annotation_class
                
                # Update statistics
                self.stats["total"] += 1
                if polar_by_annotation:
                    self.stats["polar_by_annotation"] += 1
                if classifier_class == "polar":
                    self.stats["polar_by_classifier"] += 1
                if annotation_class == classifier_class:
                    self.stats["agreement"] += 1
                    if annotation_class == "polar":
                        self.stats["agreed_polar"] += 1
                    else:
                        self.stats["agreed_content"] += 1
                else:
                    self.stats["disagreement"] += 1
                
                lang_stats = self.stats["by_language"][language]
                lang_stats["total"] += 1
                if polar_by_annotation:
                    lang_stats["polar_by_annotation"] += 1
                if classifier_class == "polar":
                    lang_stats["polar_by_classifier"] += 1
                if annotation_class == classifier_class:
                    lang_stats["agreement"] += 1
                    if annotation_class == "polar":
                        lang_stats["agreed_polar"] += 1
                    else:
                        lang_stats["agreed_content"] += 1
                else:
                    lang_stats["disagreement"] += 1
                
                if not include_question:
                    if "filtered_out" not in self.stats:
                        self.stats["filtered_out"] = 0
                    self.stats["filtered_out"] += 1
                    if "filtered_out" not in lang_stats:
                        lang_stats["filtered_out"] = 0
                    lang_stats["filtered_out"] += 1
                    filtered_count += 1
                
                result = {
                    "question_text": question_text,
                    "language": language,
                    "annotation_class": annotation_class,
                    "classifier_class": classifier_class,
                    "agreement": annotation_class == classifier_class,
                    "final_class": final_class,
                    "included": include_question
                }
                results.append(result)
            
            results_df = pd.DataFrame(results)
            
            for lang in ["japanese", "korean"]:
                if (filter_languages and lang in filter_languages) or not filter_languages:
                    if skipped_questions[lang] > 0 or filtered_content[lang] > 0 or added_polar[lang] > 0:
                        logger.info(f"Skipped {skipped_questions[lang]} {lang.capitalize()} questions due to inconsistent annotations")
                        logger.info(f"Filtered {filtered_content[lang]} {lang.capitalize()} content questions without wh-words")
                        logger.info(f"Added {added_polar[lang]} additional {lang.capitalize()} polar questions using relaxed criteria")
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                results_df.to_csv(output_path, index=False)
                logger.info(f"Saved classification results to {output_path}")
            
            if txt_output_dir:
                split_output_dir = self._get_split_output_dir(txt_output_dir, split)
                self.save_as_txt(results_df, split_output_dir)

            return results_df
    
    def _get_split_output_dir(self, base_dir, split, no_suffix=False):
        base_dir = base_dir.rstrip('/')
        if no_suffix:
            return base_dir

        return f'{base_dir}_{split}'
    
    def save_simple_output(self, results_df, output_path, split='validation'):
        included_df = results_df[(results_df['included'] == True) & (results_df['final_class'].notna())]

        simple_df = included_df[['question_text', 'language', 'final_class']]

        simple_output_path = output_path.replace('.csv', f'_{split}.csv')

        os.makedirs(os.path.dirname(simple_output_path), exist_ok=True)
        simple_df.to_csv(simple_output_path, index=False)
        logger.info(f"Saved simplified output with {len(simple_df)} questions to {simple_output_path}")

        return simple_output_path
    
    def save_as_txt(self, results_df, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        included_df = results_df[results_df['included'] == True]

        languages = included_df['language'].unique()

        for language in tqdm(languages, desc="Saving text files by language"):
            language_df = included_df[included_df['language'] == language]
            
            polar_questions = language_df[language_df['final_class'] == 'polar']['question_text']
            if not polar_questions.empty:
                polar_path = Path(output_dir, f"{language}_polar.txt")
                with open(polar_path, 'w', encoding='utf-8') as f:
                    for question in polar_questions:
                        f.write(f"{question.strip()}\n")
                logger.info(f"Saved {len(polar_questions)} polar questions to {polar_path}")
            
            content_questions = language_df[language_df['final_class'] == 'content']['question_text']
            if not content_questions.empty:
                content_path = Path(output_dir, f"{language}_content.txt")
                with open(content_path, 'w', encoding='utf-8') as f:
                    for question in content_questions:
                        f.write(f"{question.strip()}\n")
                logger.info(f"Saved {len(content_questions)} content questions to {content_path}")

    def print_annotation_stats(self):
        logger.info("\n=== Annotation Distribution ===")
        logger.info(f"Total questions analyzed: {self.annotation_counts['total']}")
        logger.info(f"Questions with single annotator: {self.annotation_counts['single_annotator']}")
        logger.info(f"Questions with multiple annotators: {self.annotation_counts['multiple_annotators']}")
        
        logger.info("\nYES answer distribution:")
        for count, occurrences in sorted(self.annotation_counts["yes_distribution"].items()):
            logger.info(f"  Questions with {count} YES annotations: {occurrences}")
        
        logger.info("\nBreakdown by language:")
        for lang, stats in self.annotation_counts["by_language"].items():
            if stats["total"] > 0:
                logger.info(f"\n{lang.capitalize()}:")
                logger.info(f"  Total questions: {stats['total']}")
                logger.info(f"  Single annotator: {stats['single_annotator']}")
                logger.info(f"  Multiple annotators: {stats['multiple_annotators']}")
                
                logger.info("  YES distribution:")
                for count, occurrences in sorted(stats["yes_distribution"].items()):
                    logger.info(f"    {count} YES annotations: {occurrences}")

    def print_stats(self):
        total = self.stats["total"]

        if total == 0:
            logger.warning("No questions processed")
            return
        
        content_by_annotation = total - self.stats["polar_by_annotation"]
        content_by_classifier = total - self.stats["polar_by_classifier"]
        
        logger.info("Classification Statistics:")
        logger.info(f"Total questions processed: {total}")
        logger.info(f"Polar questions by annotation: {self.stats['polar_by_annotation']} ({self.stats['polar_by_annotation']/total*100:.2f}%)")
        logger.info(f"Polar questions by classifier: {self.stats['polar_by_classifier']} ({self.stats['polar_by_classifier']/total*100:.2f}%)")
        logger.info(f"Content questions by annotation: {content_by_annotation} ({content_by_annotation/total*100:.2f}%)")
        logger.info(f"Content questions by classifier: {content_by_classifier} ({content_by_classifier/total*100:.2f}%)")
    
        logger.info(f"\nOverall agreement: {self.stats['agreement']} ({self.stats['agreement']/total*100:.2f}%)")
        logger.info(f"Overall disagreement: {self.stats['disagreement']} ({self.stats['disagreement']/total*100:.2f}%)")
    
        if self.stats['agreement'] > 0:
          agreed_polar_pct = self.stats['agreed_polar'] / self.stats['agreement'] * 100
          agreed_content_pct = self.stats['agreed_content'] / self.stats['agreement'] * 100
        
          logger.info(f"\nAgreement breakdown by question type:")
          logger.info(f"  Agreed on polar questions: {self.stats['agreed_polar']} ({agreed_polar_pct:.2f}% of agreements)")
          logger.info(f"  Agreed on content questions: {self.stats['agreed_content']} ({agreed_content_pct:.2f}% of agreements)")
        
          polar_agreement_rate = self.stats['agreed_polar'] / self.stats['polar_by_annotation'] * 100 if self.stats['polar_by_annotation'] > 0 else 0
          content_agreement_rate = self.stats['agreed_content'] / content_by_annotation * 100 if content_by_annotation > 0 else 0
          
          logger.info(f"\nAgreement rates by question type:")
          logger.info(f"  Polar question agreement rate: {polar_agreement_rate:.2f}%")
          logger.info(f"  Content question agreement rate: {content_agreement_rate:.2f}%")

        logger.info("\nBreakdown by language:")
        for lang, stats in self.stats["by_language"].items():
          if stats["total"] > 0:
              lang_total = stats["total"]
              content_by_annotation_lang = lang_total - stats['polar_by_annotation']

              logger.info(f"\n{lang.capitalize()}:")
              logger.info(f"  Total questions: {lang_total}")
              logger.info(f"  Polar by annotation: {stats['polar_by_annotation']} ({stats['polar_by_annotation']/lang_total*100:.2f}%)")
              logger.info(f"  Polar by classifier: {stats['polar_by_classifier']} ({stats['polar_by_classifier']/lang_total*100:.2f}%)")
              logger.info(f"  Overall agreement: {stats['agreement']} ({stats['agreement']/lang_total*100:.2f}%)")

              if stats['agreement'] > 0:
                  agreed_polar_pct = stats['agreed_polar'] / stats['agreement'] * 100
                  agreed_content_pct = stats['agreed_content'] / stats['agreement'] * 100
                
                  logger.info(f"  Agreement breakdown:")
                  logger.info(f"    Agreed on polar: {stats['agreed_polar']} ({agreed_polar_pct:.2f}% of agreements)")
                  logger.info(f"    Agreed on content: {stats['agreed_content']} ({agreed_content_pct:.2f}% of agreements)")
                
                  polar_agree_rate = stats['agreed_polar'] / stats['polar_by_annotation'] * 100 if stats['polar_by_annotation'] > 0 else 0
                  content_agree_rate = stats['agreed_content'] / content_by_annotation_lang * 100 if content_by_annotation_lang > 0 else 0
                  
                  logger.info(f"  Agreement rates by type:")
                  logger.info(f"    Polar agreement rate: {polar_agree_rate:.2f}%")
                  logger.info(f"    Content agreement rate: {content_agree_rate:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Classify TyDi QA questions using annotations and linguistic patterns.")
    
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                      help="Dataset split to process (default: validation)")
    parser.add_argument("--output", type=str, default="results/tydi_classification.csv",
                      help="Path to output CSV file")
    parser.add_argument("--simple-output", type=str, default="results/tydi_simple.csv",
                        help="Path to save simplified output (question_text, language, final_class)")
    parser.add_argument("--languages", type=str,
                      help="Comma-separated list of languages to process (e.g., 'english,finnish')")
    parser.add_argument("--txt-output", type=str,
                      help="Directory to save text files with one question per line")
    parser.add_argument("--cached-dataset", type=str,
                      help="Path to a cached CSV file of the dataset (to avoid downloading again)")
    parser.add_argument("--save-dataset", type=str,
                      help="Path to save the dataset as CSV for future use")
    parser.add_argument("--use-classifier", action="store_true",
                      help="Use pattern-based classifier for final classification instead of annotations")
    
    args = parser.parse_args()
    
    filter_languages = None
    if args.languages:
        filter_languages = [lang.strip().lower() for lang in args.languages.split(",")]
    
    classifier = TyDiClassifier()
    split = args.split

    output_path = args.output
    if not split.lower() in args.simple_output.lower():
        output_path  = args.output.replace('.csv', f'_{split}.csv')

    simple_output_path = args.simple_output

    dataset_path = args.cached_dataset if args.cached_dataset else args.save_dataset
    df = classifier.load_tydi_dataset(args.split, dataset_path)

    results = classifier.process_dataset(df, output_path, filter_languages, args.txt_output, args.use_classifier, split)

    classifier.save_simple_output(results, simple_output_path, split)
    
    classifier.print_annotation_stats()

    classifier.print_stats()

if __name__ == "__main__":
    main()