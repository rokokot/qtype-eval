import argparse
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler("ud_question_extractor.log"), 
                              logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    "en": "english",
    "ru": "russian",
    "ja": "japanese",
    "ar": "arabic",
    "fi": "finnish",
    "ko": "korean",
    "id": "indonesian"
}


class QuestionClassifier:
    
    def __init__(self, language):
        self.language = language
        
        # English patterns
        #self.en_wh_words = r'\b(what|What|who|Who|where|Where|when|When|why|Why|how|How|which|Which|Whose|whose|Whom|whom)\b'
        #self.en_polar_starters = r'^(is|Is|are|Are|Do|do|Does|does|Did|Did|Have|have|Has|has|Can|can|Could|could|will|Will|would|Would|should|Should|May|may|Might|might)'
        self.en_wh_words = r'\b(what|who|where|when|why|how|which|whose|whom)\b'
        self.en_polar_starters = r'^(is|are|am|do|does|did|have|has|had|can|could|will|would|should|may|might|must|shall|isn\'t|aren\'t|don\'t|doesn\'t|didn\'t|won\'t|haven\'t)'
        self.embedded_verbs = r'\b(know|tell|confirm|explain|understand|think|show|mean|see)\b'
        
        # Finnish patterns
        self.fi_wh_words = r'\b(mik(?:ä|si)|montako|mit(?:ä|en)|miss(?:ä|tä)|mihin|mill(?:oin|ä)|kuk(?:a|aan)|ket(?:ä|kä)|ken(?:en|eltä)|kumpi|kuinka|montako)\b'
        self.fi_polar = r'\b\w+(?:ko|kö)\b'

        # Korean patterns
        self.ko_wh_words = r'(무엇|뭐|뭣|무슨|누구|누가|어디|어느|언제|왜|어째서|어떻게|어떤|몇|얼마)'
        self.ko_ending_pattern = r'(까요|니까|나요|는가|을까|가요|니|까|냐|가|나)\s*\??$'

        # Japanese patterns
        self.ja_ka_pattern = r'(か|かな|のか|のかな|だろう|でしょう|ですか|[いな]|の|ん)[\s。]*[\?？]*$'
        self.ja_wh_words = r'(何|なに|なん|どこ|どちら|いつ|誰|だれ|なぜ|どうして|どう|どのよう|どの|どんな|いくつ|いくら|どれ|どんな風|いかが|どのくらい)'

        # Russian patterns
        self.ru_li_pattern = r'\s+ли\b'
        self.ru_wh_words = r'\b(что|чего|чему|чем|кто|кого|кому|кем|где|куда|откуда|когда|почему|зачем|как|каким\s+образом|который|как(?:ой|ая|ое|ие)|сколько)\b'

        # Arabic patterns
        self.ar_polar_pattern = r'^(هل|أ)\b|\bهل\b' 
        self.ar_wh_words = r'\b(ما(?:ذا)?|من|أين|وين|متى|لماذا|ليش|كيف|أي|كم)\b|ف(ماذا|من|أين|متى|لماذا|كيف|أي)'

        # Indonesian patterns
        #self.id_polar_pattern = r'^(apakah|apa\s+kah|apa)\b'  
        self.id_polar_pattern = [r'^(apakah|apa\s+kah|apa)\b', 
        r'(?:^|kah\s+)(apa|ada|akankah|mau|bisa|boleh|sudah|dapat|sanggup|mungkin|perlu|harus|benar|betul|ingin|suka|mampu)',
        r'^(ada|bukan|bukankah|tidakkah|haruskah|bisakah|maukah|dapatkah|mampukah|perlukah|benarkah|betulkah|masakan)',
        r'(kan|bukan|ya|tidak|toh|dong)\s*\?',
        r'([\?\？])\s*$'] 
        #self.id_wh_words = r'\b(apa\s+yang|apa\s+saja|siapa(?:kah)?|di\s+mana|dimana|ke\s+mana|kemana|dari\s+mana|darimana|kapan|bila|mengapa|kenapa|bagaimana|yang\s+mana|berapa)\b'
        self.id_wh_words = [r'\b(apa\s+yang|siapa(?:kah)?|di\s*mana|dimana|ke\s*mana|kemana|dari\s*mana|darimana)',
        r'\b(kapan|bila(?:kah)?|bilamana|mengapa|kenapa|bagaimana(?:kah)?|yang\s+mana|berapa)',
        r'\b(bagaimana(?:kah)?\s+(?:cara|nasib|kisah|kelanjutan|reaksi|hubungan))',
        r'\b(apa\s+(?:saja|yang|arti|makna|maksud|kabar|gunanya|sebab))',
        r'\b(siapa(?:kah)?\s+(?:yang|nama|sebenarnya))',
        r'\b(mengapa\s+(?:tidak|begitu|harus|engkau|kamu))']


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
        if re.search(self.en_wh_words, text, re.I):
            return 'content'
        
        if re.match(f'{self.en_polar_starters}.*{self.embedded_verbs}.*{self.en_wh_words}', text, re.I):
            return 'polar'
        
        if re.match(self.en_polar_starters, text, re.I):
            return 'polar'
        if text.endswith('?'):
            return 'polar'
        
        conjunction_polar = r'^(but|and|so|or|yet|for) (is|are|do|does|did|have|has|can|could|will|would)'
        if re.search(conjunction_polar, text, re.I):
            return 'polar'
        
        return None
    
    def _classify_finnish(self, text):
        if re.search(self.fi_wh_words, text, re.I):
            return 'content'

        if re.search(self.fi_polar, text):
            return 'polar'
    
        return None
    
    def _classify_korean(self, text):
        if re.search(self.ko_wh_words, text):
            return 'content'
        
        if re.search(self.ko_ending_pattern, text):
            return 'polar'
        
        return None        
    
    def _classify_japanese(self, text):
        if re.search(self.ja_wh_words, text):
            return 'content'
        
        if re.search(self.ja_ka_pattern, text):
            return 'polar'
        if re.search(r'[\?? ]$', text):
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
        if text.endswith('؟'):
            return 'content'
        
        return None
    
    def _classify_indonesian(self, text):
        text = text.lower()

        for pattern in self.id_wh_words:
            if re.search(pattern, text):
                return 'content'
    

        for pattern in self.id_polar_pattern:
            if re.search(pattern, text):
                if not any(re.search(word, text) for word in self.id_wh_words):
                    return 'polar'

        if re.search(r'^(adalah|adakah|jadikah|menjadikah|merupakan)', text):
            return 'polar'
    
        if re.search(r'^(boleh|harus|dapat|bisa|akan|mau|perlu|ingin)\b', text):
            return 'polar'
    
        if re.search(r'\b\w+kah\b', text) and not any(re.search(word, text) for word in self.id_wh_words):
            return 'polar'
    
        if re.search(r'\b(atau|ataukah)\b.*[\?\？]$', text):
            return 'polar'
    
        if re.search(r'\b(bertanya|tanya|menanyakan)\b.*[\?\？]$', text):
        
            if any(re.search(word, text) for word in self.id_wh_words):
                return 'content'
            return 'polar'
    
        if re.search(r'^(ini|itu|begitu|begini|demikian)', text) and text.endswith('?'):
            return 'polar'
    
        if re.search(r'\b(tidak|bukan|belum|jangan|tak)\b', text) and text.endswith('?'):
            return 'polar'
    
        if text.endswith('?'):
                   
            if re.match(r'^(me|di|ter|ber)[a-z]+', text.split()[0]):
                return 'polar'
    
        return None


def is_question(text):

 
    question_marks = ['?', '؟', '︖', '﹖', '？']
    
    if any(mark in text for mark in question_marks):
        return True
    
    if 'か。' in text or re.search(r'か[\s。]*$', text):
        return True
    
    korean_endings = ['까요', '니까', '나요', '는가', '을까', '가요', '니', '까', '냐', '가', '나']
    if any(ending in text for ending in korean_endings) and not any(mark in text for mark in question_marks):
        for ending in korean_endings:
            if re.search(f'{ending}\\s*$', text):
                return True
    
    return False

def filter_question(text, min_tokens=3, max_tokens=40):
    tokens = text.split()
    if len(tokens) < min_tokens or len(tokens) > max_tokens:
        return True
    
    if re.search(r'(\?{2,}|\?!|!\?|!{2,})', text):
        return True
    
    
    if text.count('(') != text.count(')') or text.count('[') != text.count(']') or text.count('{') != text.count('}'):
        return True
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F700-\U0001F77F"  # alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    
    if emoji_pattern.search(text):
        return True
    
    return False
    
    

def process_conllu_file(input_file, output_dir, lang, min_tokens=3, max_tokens=40, filter_questions=True):
   
    if not lang:
        logger.error("Language must be specified with --language flag")
        sys.exit(1)
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "total_sentences": 0,
        "questions": 0,
        "filtered_questions": 0,
        "polar": 0,
        "content": 0,
        "unclassified": 0
    }
    
    polar_conllu_path = output_dir / f"{lang}_polar.conllu"
    content_conllu_path = output_dir / f"{lang}_content.conllu"
    polar_txt_path = output_dir / f"{lang}_polar.txt"
    content_txt_path = output_dir / f"{lang}_content.txt"
    unclassified_txt_path = output_dir / f"{lang}_unclassified.txt"
    filtered_txt_path = output_dir / f"{lang}_filtered.txt"

    file_exists = polar_conllu_path.exists() or content_conllu_path.exists()
    mode = 'a' if file_exists else 'w'
    
    polar_conllu = open(polar_conllu_path, mode, encoding='utf-8')
    content_conllu = open(content_conllu_path, mode, encoding='utf-8')
    polar_txt = open(polar_txt_path, mode, encoding='utf-8')
    content_txt = open(content_txt_path, mode, encoding='utf-8')
    unclassified_txt = open(unclassified_txt_path, mode, encoding='utf-8')
    filtered_txt = open(filtered_txt_path, mode, encoding='utf-8')

    try:
        current_sentence = []
        is_question_sentence = False
        sentence_text = ""
        english_translation = ""
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                current_sentence.append(line)
                
                text_match = re.match(r'^# (?:text|sent) = (.+)$', line)
                if text_match:
                    sentence_text = text_match.group(1)
                    if is_question(sentence_text):
                        is_question_sentence = True
                
                if line.strip() == '':
                    stats["total_sentences"] += 1
                    
                    if is_question_sentence:
                        stats["questions"] += 1
                        
                        if filter_questions and filter_question(sentence_text, min_tokens, max_tokens):
                            stats["filtered_questions"] += 1
                            filtered_txt.write(sentence_text + '\n')
                        else:
                            if lang == "id" and english_translation:
                                en_classifier = QuestionClassifier('en')
                                question_type = en_classifier._classify_english(english_translation)

                                if question_type is None:
                                    id_classifier = QuestionClassifier('id')
                                    question_type = id_classifier.classify(sentence_text)
                            else:
                                    
                            
                                classifier = QuestionClassifier(lang)
                                question_type = classifier.classify(sentence_text)
                            
                                if question_type == 'polar':
                                    stats["polar"] += 1
                                    polar_conllu.write(''.join(current_sentence))
                                    polar_txt.write(sentence_text + '\n')
                                elif question_type == 'content':
                                    stats["content"] += 1
                                    content_conllu.write(''.join(current_sentence))
                                    content_txt.write(sentence_text + '\n')
                                else:
                                    stats["unclassified"] += 1
                                    unclassified_txt.write(sentence_text + '\n')
                            
                    # Reset for the next sentence
                    current_sentence = []
                    is_question_sentence = False
                    sentence_text = ""
                    english_translation = ""
                    
        # Handle the last sentence if there's no final newline
        if current_sentence and is_question_sentence and sentence_text:
            stats["questions"] += 1

            if filter_questions and filter_question(sentence_text, min_tokens, max_tokens):
                stats["filtered_questions"] += 1
                filtered_txt.write(sentence_text + '\n')
            else:
                classifier = QuestionClassifier(lang)
                question_type = classifier.classify(sentence_text)
    
                if question_type == 'polar':
                    stats["polar"] += 1
                    polar_conllu.write(''.join(current_sentence))
                    polar_txt.write(sentence_text + '\n')
                elif question_type == 'content':
                    stats["content"] += 1
                    content_conllu.write(''.join(current_sentence))
                    content_txt.write(sentence_text + '\n')
                else:
                    stats["unclassified"] += 1
                    unclassified_txt.write(sentence_text + '\n')
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        raise
        
    finally:
        # Close all output files
        polar_conllu.close()
        content_conllu.close()
        polar_txt.close()
        content_txt.close()
        unclassified_txt.close()
    return stats


def process_directory(input_dir, output_dir, lang, min_tokens=3, max_tokens=40, filter_questions=True):
   
    if not lang:
        logger.error("Language must be specified with --language flag")
        sys.exit(1)
        
    input_dir = Path(input_dir)
    conllu_files = list(input_dir.glob("**/*.conllu"))
    
    logger.info(f"Found {len(conllu_files)} CoNLL-U files in {input_dir}")
    
    # Aggregate statistics
    all_stats = {
        "total_files": len(conllu_files),
        "total_sentences": 0,
        "questions": 0,
        "polar": 0,
        "content": 0,
        "filtered_questions": 0,
        "unclassified": 0,
        "by_file": {}
    }
    
    # Process all files into a single output directory with language-based filenames
    logger.info(f"Processing all files using language: {lang}")
    
    # Process each file separately, but aggregate the results
    for file_path in tqdm(conllu_files, desc="Processing CoNLL-U files"):
        logger.info(f"Processing {file_path}")
        try:
            file_stats = process_conllu_file(file_path, output_dir, lang, min_tokens=min_tokens, max_tokens=max_tokens, filter_questions=filter_questions)
            
            # Update aggregate statistics
            all_stats["total_sentences"] += file_stats["total_sentences"]
            all_stats["questions"] += file_stats["questions"]
            all_stats["polar"] += file_stats["polar"]
            all_stats["content"] += file_stats["content"]
            all_stats["unclassified"] += file_stats["unclassified"]
            all_stats["by_file"][str(file_path)] = file_stats
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.error("Continuing with next file...")
    
    return all_stats


def print_stats(stats):
   
    logger.info("\n=== Question Classification Statistics ===")
    
    if "total_files" in stats:
        logger.info(f"Processed {stats['total_files']} files")
    
    logger.info(f"Total sentences: {stats['total_sentences']}")
    
    if stats["total_sentences"] > 0:
        question_percentage = stats['questions']/stats['total_sentences']*100 if stats['total_sentences'] > 0 else 0
        logger.info(f"Questions found: {stats['questions']} ({question_percentage:.2f}% of sentences)")
    
    if stats["questions"] > 0:
        filtered_percentage = stats['filtered_questions']/stats['questions']*100 if stats['questions'] > 0 else 0
        logger.info(f"Filtered questions: {stats['filtered_questions']} ({filtered_percentage:.2f}% of questions)")
        
        remaining_questions = stats['questions'] - stats['filtered_questions']
        if remaining_questions > 0:
            polar_percentage = stats['polar']/remaining_questions*100 if remaining_questions > 0 else 0
            content_percentage = stats['content']/remaining_questions*100 if remaining_questions > 0 else 0
            unclassified_percentage = stats['unclassified']/remaining_questions*100 if remaining_questions > 0 else 0
            
            logger.info(f"\nClassified questions (after filtering):")
            logger.info(f"  Polar questions: {stats['polar']} ({polar_percentage:.2f}% of remaining questions)")
            logger.info(f"  Content questions: {stats['content']} ({content_percentage:.2f}% of remaining questions)")
            logger.info(f"  Unclassified: {stats['unclassified']} ({unclassified_percentage:.2f}% of remaining questions)")
    
    if "by_file" in stats and stats["by_file"]:
        logger.info("\nBreakdown by file:")
        for file_path, file_stats in stats["by_file"].items():
            if file_stats["questions"] > 0:
                filtered_percentage = file_stats['filtered_questions']/file_stats['questions']*100 if file_stats['questions'] > 0 else 0
                remaining_questions = file_stats['questions'] - file_stats['filtered_questions']
                
                if remaining_questions > 0:
                    polar_percentage = file_stats['polar']/remaining_questions*100 if remaining_questions > 0 else 0
                    content_percentage = file_stats['content']/remaining_questions*100 if remaining_questions > 0 else 0
                    unclassified_percentage = file_stats['unclassified']/remaining_questions*100 if remaining_questions > 0 else 0
                    
                    logger.info(f"\n{os.path.basename(file_path)}:")
                    logger.info(f"  Questions: {file_stats['questions']}")
                    logger.info(f"  Filtered: {file_stats['filtered_questions']} ({filtered_percentage:.2f}%)")
                    logger.info(f"  Polar: {file_stats['polar']} ({polar_percentage:.2f}%)")
                    logger.info(f"  Content: {file_stats['content']} ({content_percentage:.2f}%)")
                    logger.info(f"  Unclassified: {file_stats['unclassified']} ({unclassified_percentage:.2f}%)")



def main():
    parser = argparse.ArgumentParser(
        description="Identify and classify questions from UD treebanks"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input file or directory. If a directory, all .conllu files within it will be processed."
    )
    parser.add_argument(
        "--output-dir", "-o", default="./classified_questions",
        help="Output directory (default: ./classified_questions)"
    )
    parser.add_argument(
        "--language", "-l", required=True,
        help="Language code for classification (e.g., en, fi, ru). This is required and will be used for naming output files."
    )
    
    parser.add_argument(
        "--min-tokens", "-min", type=int, default=3,
        help="Minimum number of tokens required for a question (default: 3)"
    )
    
    parser.add_argument(
        "--max-tokens", "-max", type=int, default=40,
        help="Maximum number of tokens allowed for a question (default: 40)"
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Disable question filtering (by default, filtering is enabled)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not args.language:
        logger.error("Language must be specified with --language flag")
        sys.exit(1)
    
    if args.language not in LANGUAGE_MAP:
        logger.warning(f"Language '{args.language}' is not in the supported languages: {', '.join(LANGUAGE_MAP.keys())}")
        logger.warning("Classification may be inaccurate.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = None
    if input_path.is_dir():
        stats = process_directory(input_path, output_dir, args.language, min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            filter_questions=not args.no_filter)
    elif input_path.is_file():
        stats = process_conllu_file(input_path, output_dir, args.language, min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            filter_questions=not args.no_filter)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    print_stats(stats)
    
    logger.info(f"\nQuestion extraction and classification complete.")
    logger.info(f"Results saved to {output_dir}:")
    logger.info(f"  - {args.language}_filtered.txt (Filtered out questions)")
    logger.info(f"  - {args.language}_unclassified.txt (Unclassified questions)")
    logger.info(f"  - {args.language}_polar.txt (Polar questions)")
    logger.info(f"  - {args.language}_content.txt (Content questions)")
    logger.info(f"  - {args.language}_polar.conllu (Polar questions in CoNLL-U format)")
    logger.info(f"  - {args.language}_content.conllu (Content questions in CoNLL-U format)")


if __name__ == "__main__":
    main()