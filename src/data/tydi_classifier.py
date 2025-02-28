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


"""

    This script loads and processes the TyDi QA dataset. For example, to download the validation split, run the following commands:

    python src/data/tydi_classifier.py --save-dataset data/tydi_validation.pkl --txt-output TyDi-questions/

    python src/data/tydi_classifier.py --split train --cached-dataset data/tydi_train.pkl --txt-output TyDi-questions-train/
    
    after downloading, the dataset is cached and can be accessed with the --cached-dataset flag

    python src/data/tydi_classifier.py --cached-dataset data/tydi_validation.pkl --txt-output TyDi-questions/

    note the args in main for more options such as:
    
    save data to .csv: python src/data/tydi_classifier.py --cached-dataset data/tydi_validation.pkl --txt-output TyDi-questions/ --output custom/path/results.csv --disagreements custom/path/disagreements.csv


"""






LANGUAGE_MAP = {
    "english": "en",
    "russian": "ru",
    "japanese": "ja",
    "arabic": "ar",
    "finnish": "fi",
    "korean": "ko",
    "indonesian": "id"
}

class QuestionClassifier:
    """Classifier for questions based on patterns from ud_question_extractor"""
    
    def __init__(self, language):
        
        self.language = language
        
        


        self.en_wh_words = r'\b(what*|who*|where*|when*|why*|how*|which*)\b'
        self.en_polar_starters = r'^(is|are|do|does|did|have|has|can|could|will|would|should|may|might)'
        self.embedded_verbs = r'\b(know|tell|confirm|explain|understand|think|show|mean|see)\b'
        
        self.fi_wh_words = r'\b(mik(?:ä|si)|mit(?:ä|en)|miss(?:ä|tä)|mihin|mill(?:oin|ä)|kuk(?:a|aan)|ket(?:ä|kä)|ken(?:en|eltä)|kumpi|kuinka|montako)\b'
        self.fi_polar = r'\b\w+(?:ko|kö)\b'


        self.ko_wh_words = r'(무엇|뭐|뭣|무슨|누구|누가|어디|어느|언제|왜|어째서|어떻게|어떤|몇|얼마)'
        self.ko_ending_pattern = r'(까요|니까|나요|는가|을까|가요|니|까|냐|가|나)\s*\??$'

        self.ja_ka_pattern = r'か\s*[\?？]?$'
        self.ja_wh_words = r'(何|なに|なん|どこ|どちら|いつ|誰|だれ|なぜ|どうして|どう|どのよう|どの|どんな|いくつ|いくら)'

        self.ru_li_pattern = r'\s+ли\b'
        self.ru_wh_words = r'\b(что|чего|чему|чем|кто|кого|кому|кем|где|куда|откуда|когда|почему|зачем|как|каким\s+образом|который|как(?:ой|ая|ое|ие)|сколько)\b'

        self.ar_polar_pattern = r'^(هل|أ)\b' 
        self.ar_wh_words = r'\b(ما(?:ذا)?|من|أين|وين|متى|لماذا|ليش|كيف|أي|كم)\b'


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
        #return "polar"
    
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
        
        return 'polar'
    
    def _classify_finnish(self, text):
        text = text.lower()

        if re.search(self.fi_polar, text):
            return 'polar'
        
        #if re.search(r'\bvai\b', text):
            #return 'content'
        
        if re.search(self.fi_wh_words, text, re.I):
            return 'content'
        
        

    
    def _classify_korean(self, text):
        if re.search(self.ko_wh_words, text):
            return 'content'
        
        if re.search(self.ko_ending_pattern, text):
            return 'polar'
        
    

    def _classify_japanese(self, text):
        if re.search(self.ja_wh_words, text):
            return 'content'
        
        if re.search(self.ja_ka_pattern, text):
            return 'polar'
        
    
    def _classify_russian(self, text):
        text = text.lower()
        if re.search(self.ru_li_pattern, text):
            return 'polar'
        
        if re.search(self.ru_wh_words, text):
            return 'content'

    
    def _classify_arabic(self, text):
        if re.search(self.ar_polar_pattern, text):
            return 'polar'
        
        if re.search(self.ar_wh_words, text):
            return 'content'
    
    def _classify_indonesian(self, text):
        text = text.lower()

        if re.search(self.id_polar_pattern, text):
            if re.search(r'\bapa\s+yang\b', text):
                return 'content'
            return 'polar'
        
        if re.search(self.id_wh_words, text):
            return 'content'
        

class TyDiClassifier:
    """Classifies questions from TyDi QA using both annotations and linguistic patterns"""
    
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
    
    def load_tydi_dataset(self, split="validation", cached_path=None):
        """Load the TyDi QA dataset"""

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
        """Analyze the distribution of yes_no_answer annotations"""
        
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
        """Check if a question is polar based on TyDi annotations"""
        if isinstance(annotations, dict) and 'yes_no_answer' in annotations:
            return any(ans == 'YES' or ans == 'NO' for ans in annotations['yes_no_answer'])
        return False
    
    def classify_question(self, question_text, language, annotation_class=None):
        """Compare result to a pattern-based classifier"""
        if language not in self.classifiers:
           return annotation_class or "content"
        
        classifier_result = self.classifiers[language].classify(question_text)

        if classifier_result not in ["polar", "content"]:
            return annotation_class or "content"
        
        return classifier_result

    
    def process_dataset(self, df, output_path=None, filter_languages=None, txt_output_dir=None, use_classifier=False, split="validation"):
        """Process the dataset and classify questions"""
        results = []
        
        self.analyze_annotations(df, filter_languages)

        if filter_languages:
            df = df[df['language'].str.lower().isin(filter_languages)]
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Classifying questions"):
            question_text = row['question_text']
            language = row['language'].lower()
            
            if language not in self.classifiers:
                continue
            
            polar_by_annotation = self.is_polar_by_annotation(row['annotations'])
            annotation_class = "polar" if polar_by_annotation else "content"
            
            classifier_class = self.classify_question(question_text, language, annotation_class)
            
            final_class = classifier_class if use_classifier else annotation_class

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
                if annotation_class=="polar":
                    lang_stats["agreed_polar"] += 1
                else:
                    lang_stats["agreed_content"] += 1
            else:
                lang_stats["disagreement"] += 1
            
            result = {
                "question_text": question_text,
                "language": language,
                "annotation_class": annotation_class,
                "classifier_class": classifier_class,
                "agreement": annotation_class == classifier_class,
                "final_class": final_class
            }
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved classification results to {output_path}")
        
        if txt_output_dir:
            split_output_dir = self._get_split_output_dir(txt_output_dir, split)
            self.save_as_txt(results_df, split_output_dir)

        
        return results_df
    
    def _get_split_output_dir(self, base_dir, split):
        if split.lower() in base_dir.lower():
            return base_dir
        return os.path.join(base_dir, split)
    
    
    def save_disagreement_examples(self, results_df, output_path, split='validation'):
        """Save examples where annotations and classifier disagree"""
        disagreements = results_df[results_df["agreement"] == False]
        
        split_output_path = output_path
        if not split.lower() in output_path.lower():
            split_output_path = output_path.replace('.csv', f'_{split}.csv')
        
        if len(disagreements) > 0:
            os.makedirs(os.path.dirname(split_output_path), exist_ok=True)
            disagreements.to_csv(split_output_path, index=False)
            logger.info(f"Saved {len(disagreements)} disagreement examples to {split_output_path}")
    
    def save_as_txt(self, results_df, output_dir):
        """Save questions as plain text files, one question per line"""
        os.makedirs(output_dir, exist_ok=True)

        languages = results_df['language'].unique()

        
        for language in tqdm(languages, desc="Saving text files by language"):
            language_df = results_df[results_df['language'] == language]
            
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
        """Print annotation distribution statistics"""
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
        """Print classification statistics"""
        total = self.stats["total"]

        content_by_annotation = total - self.stats["polar_by_annotation"]
        content_by_classifier = total - self.stats["polar_by_classifier"]

        if total == 0:
            logger.warning("No questions processed")
            return
        
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
    
    """Main function for the TyDi classifier"""

    parser = argparse.ArgumentParser(description="Classify TyDi QA questions using annotations and linguistic patterns.")
    
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"],
                      help="Dataset split to process (default: validation)")
    parser.add_argument("--output", type=str, default="results/tydi_classification.csv",
                      help="Path to output CSV file")
    parser.add_argument("--disagreements", type=str, default="results/tydi_disagreements.csv",
                      help="Path to save disagreement examples")
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
    if args.output and not split.lower() in args.output.lower():
        output_path = args.output.replace('.csv', f'_{split}.csv')

    dataset_path = args.cached_dataset if args.cached_dataset else args.save_dataset
    df = classifier.load_tydi_dataset(args.split, dataset_path)
    results = classifier.process_dataset(df, output_path, filter_languages, args.txt_output, args.use_classifier, split)
    classifier.save_disagreement_examples(results, args.disagreements, split)
    classifier.print_annotation_stats()

    classifier.print_stats()

if __name__ == "__main__":
    main()