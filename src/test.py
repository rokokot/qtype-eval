import re
import os
from tqdm import tqdm

def is_english_question(sentence):
 
    sentence = sentence.strip()
    if sentence.endswith('?'):
        return True
        
    question_words = r"^(Who|What|When|Where|Why|How|Which|Whose|Whom|Is|Are|Do|Does|Did|Have|Has|Had|Can|Could|Will|Would|Should|May|Might|Must)\b"
    if re.search(question_words, sentence, re.IGNORECASE):
        return True
    return False

def is_hindi_question(sentence):

    sentence = sentence.strip()
    if sentence.endswith('?'):
        return True
        
    question_words = [
        "क्या", "किसको", "किसके", "किसका", "कौन", "कब", "कहाँ", 
        "कैसी", "क्यों", "कैसे", "कैसा", "कितना", "कितने", "कितनी", 
        "किस", "किसके", "से", "कौन सा", "सी"
    ]
    
    for word in question_words:
        if word in sentence.split():
            return True
    return False

def filter_parallel_questions(english_file_path, hindi_file_path, output_file_path, limit=None):


    with open(english_file_path, 'r', encoding='utf-8') as en_file, \
         open(hindi_file_path, 'r', encoding='utf-8') as hi_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
        
        en_lines = en_file.readlines()
        hi_lines = hi_file.readlines()
        
        if len(en_lines) != len(hi_lines):
            raise ValueError("(different number of lines)")
            
        total_pairs = min(len(en_lines), limit) if limit else len(en_lines)
        count = 0
        
        for i in tqdm(range(total_pairs)):
            en_sentence = en_lines[i].strip()
            hi_sentence = hi_lines[i].strip()
            
            if is_english_question(en_sentence) and is_hindi_question(hi_sentence):
                output_file.write(f"{en_sentence}\t{hi_sentence}\n")
                count += 1
                
        print(f"Found {count} question pairs out of {total_pairs} total pairs")

if __name__ == "__main__":
    en_nllb = os.path.expanduser("~/Research/qtype-eval/data/en-hi.txt/NLLB.en-hi.en")
    hi_nllb = os.path.expanduser("~/Research/qtype-eval/data/en-hi.txt/NLLB.en-hi.hi")
    output_file = os.path.expanduser("~/Research/qtype-eval/data/en_hi_questions_v2.txt")
    
    filter_parallel_questions(en_nllb, hi_nllb, output_file, limit=5000)
    print(f"Filtered question pairs saved to: {output_file}")