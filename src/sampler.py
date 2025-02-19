"""

this simple script creates a sample dataset from our language and split files.
It takes 3 random samples from each language/split file.


"""




import os
import random

def read(filename):
  sentences = []
  index = []
  metadata = []

  with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if line.startswith('#'):
          if 'sent_id' in line or 'text' in line:
             metadata.append(line)

          continue

      if line == '':
        if index:
          sentences.append((metadata, index))
          index = []
          metadata = []
      else:
        index.append(line)

  if index:
    sentences.append((metadata, index))

  return sentences


def write(sentences, out_file, append=False):
  mode = 'a' if append else 'w'
  with open(out_file, 'a' if append else 'w', encoding='utf-8') as f:
    for metadata,sentence in sentences:
      for meta in metadata:
         f.write(meta +'\n')
      for line in sentence:
        f.write(line + '\n')
      f.write('\n')


def sample(directory, output_file, samples_per_file=3):
    if os.path.exists(output_file):
        os.remove(output_file)
    
    samples = []

    # process conllu file
    for filename in os.listdir(directory):
        if filename.endswith('.conllu'):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filename}...")
            
           
            sentences = read(filepath)
            
            if len(sentences) < samples_per_file:
                sampled_sentences = sentences
                print(f"Warning: {filename} has fewer than {samples_per_file} sentences.")
            else:
                sampled_sentences = random.sample(sentences, samples_per_file)
            
            samples.extend(sampled_sentences)
            print(f'added {len(sampled_sentences)} questions from {filename}')

            random.shuffle(samples)

            write(samples, output_file, append=True)
            print(f"Added {len(sampled_sentences)} sentences from {filename}")


directory = "./UD-questions" 
output_file = "sample-data.conllu"
sample(directory, output_file)