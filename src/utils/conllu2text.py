import os
from pathlib import Path
from conllu import parse_incr


def conllu2text(in_dir, out_dir):
  input = Path(in_dir)
  output = Path(out_dir)
  output.mkdir(parents=True, exist_ok=True)

  for conllu_file in input.glob("*.conllu"):
        output_file = output / f"{conllu_file.stem}.txt"
        with open(conllu_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out_f:
            for sentence in parse_incr(f):
                text = sentence.metadata.get('text', '')

                if len(text.split()) >= 3: # included some pre processing

                  out_f.write(text + '\n')
        print(f"Processed {conllu_file} -> {output_file}")


in_dir = '/home/robin/Research/qtype-eval/src/UD-korean-questions'
out_dir = '/home/robin/Research/qtype-eval/src/UD-ko-questions_txt'
conllu2text(in_dir, out_dir)