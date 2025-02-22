import conllu
from conllu.serializer import serialize
import os

def filter(file, pattern):
    results = []
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = f.read()
            sentences = conllu.parse(data)
            for sentence in sentences:
                if sentence and (any(token['form'] in pattern for token in sentence)) or sentence[-2]['form'] == 'か':
                    results.append(sentence)
    except FileNotFoundError:
        print(f'Error at path {file}') 
        return None
    return results

def save(questions, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(serialize(question))
            f.write('\n')

def run_filter(treebanks):
    pattern = ['?', '؟']

    for file in treebanks:
        print(f'Working on {file}')
        questions = filter(file, pattern)

        if questions:
            print(f'Collected {len(questions)} questions in {file}')
            filename = os.path.basename(file)
            out_file = os.path.join(f'questions_{filename}')
            save(questions, out_file)
            print(f'Saved questions to {out_file}')
        else:
            print(f'No questions found in {file}')

treebanks = [
   '/home/robin/Research/qtype-eval/data/UD_Japanese-PUD/ja_pud-ud-test.conllu'
]

run_filter(treebanks)
