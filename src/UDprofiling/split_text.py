import os
from pathlib import Path
import zipfile

def divide_text_to_zip(input_file, output_zip):
    # temporary directory to store individual sentence files
    temp_dir = Path("temp_sentences")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # create individual sentence files
    with open(input_file, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            sentence_file = temp_dir / f"{index}.txt"
            with open(sentence_file, 'w', encoding='utf-8') as sentence_f:
                sentence_f.write(line.strip())

    # a zip file containing sentence files
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for sentence_file in temp_dir.glob("*.txt"):
            zipf.write(sentence_file, arcname=sentence_file.name)

    for sentence_file in temp_dir.glob("*.txt"):
        sentence_file.unlink()
    temp_dir.rmdir()


input_file = '/home/robin/Research/qtype-eval/data/UD-ko-questions_txt/korean_polar_questions_UD.txt'
output_zip = '/home/robin/Research/qtype-eval/src/UDprofiling/zip_UD_korean_polar.zip'
divide_text_to_zip(input_file, output_zip)