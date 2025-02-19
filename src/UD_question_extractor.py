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
                if sentence and any(token['form'] in pattern for token in sentence):
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
   '/home/robin/Research/qtype-eval/data/UD_Arabic-PADT/ar_padt-ud-dev.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Arabic-PADT/ar_padt-ud-test.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Arabic-PADT/ar_padt-ud-train.conllu',
   '/home/robin/Research/qtype-eval/data/UD_English-EWT/en_ewt-ud-dev.conllu',
   '/home/robin/Research/qtype-eval/data/UD_English-EWT/en_ewt-ud-test.conllu',
   '/home/robin/Research/qtype-eval/data/UD_English-EWT/en_ewt-ud-train.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Finnish-TDT/fi_tdt-ud-dev.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Finnish-TDT/fi_tdt-ud-test.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Finnish-TDT/fi_tdt-ud-train.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Indonesian-GSD/id_gsd-ud-dev.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Indonesian-GSD/id_gsd-ud-test.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Indonesian-GSD/id_gsd-ud-train.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Japanese-GSD/ja_gsd-ud-dev.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Japanese-GSD/ja_gsd-ud-test.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Japanese-GSD/ja_gsd-ud-train.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Korean-Kaist/ko_kaist-ud-dev.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Korean-Kaist/ko_kaist-ud-test.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Korean-Kaist/ko_kaist-ud-train.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Russian-Taiga/ru_taiga-ud-dev.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Russian-Taiga/ru_taiga-ud-test.conllu',
   '/home/robin/Research/qtype-eval/data/UD_Russian-Taiga/ru_taiga-ud-train.conllu'
]

run_filter(treebanks)
