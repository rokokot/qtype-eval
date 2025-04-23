# `LingMonitoring` - Linguistic Features extraction

This folder contains scripts to extract the linguistic profile from texts annotated Universal Dependency format.

## Requirements
- Requires Python 3.x

## Type of input
- Text
 * The Linguistic Monitoring works on files containing sentences parsed in CoNLL-U format. More information about this format can be found in the [Universal Dependencies](https://universaldependencies.org/) page.


- Dictionary
 * An optional file can be passed to the app in order to extract features regarding the frequency of the lexicon present in the text. The dictionary should contain a list of lemmas (one per line) followed by a tag indicating the frequency of use of that word in the selected language. You can create your own dictionary and use it in the app, using the following format:  `lemma \t tag \n`.


 For the Italian language, we already provide a Dictionary (NVdB_DeMauro.txt) constructed according to Tullio De Mauro's _Nuovo vocabolario di base della lingua italiana_, which contains 7,500 words that belong to the basic vocabulary of Italian. Each lemma is marked with a tag that indicates its frequency of use in the language:
  * **FO** (_fundamental lexicon_): around 2000 words extremely frequent in the language, used in 86% of texts and discourses;
  * **AU** (_high usage lexicon_): around 3000 words frequently used, they appear in the 6% of texts and discourses;
  * **AD** (_higly available lexicon_): around 2000 words used only in a few contexts, but easily understood by every speaker.


## Usage

To run use:

 `python ling_monitoring.py [-p PATH][-d DICT][-t TYPE]`


* optional arguments

 `-h, --help` show this help message and exit

 `-d DICTIONARY, --dict DICTIONARY` specify dictionary file, if present

  `-p YOUR_PATH, --path YOUR_PATH` specify the path of the directory that contains the file or the files you want to analyse or specify the single file you want to analyse

 `-t {0, 1}, -type {0, 1}` specify if you want to extract features from single sentences [0] or from a document containg one or more sentences [1]. The default value is [1] for documents.


## Input

For document-level analysis the input file(s) must be placed into a folder, while for sentence-level analysis the input can be both a single file or a folder with multiple files.

## Output

All outputs of the analysis are stored in a directory **output_results/** automatically created by the app.

In the case of sentences analysis, for each input file there will be created an output file with name *inputfilename_sent.out*, in which the results will be stored in a tab separated file with this format:

```sentence_id \t feature_name_1 \t feature_name_2 \t feature_name_3 ... \n```

In the case of documents analysis, when a folder with both multiple files and a single file is passed the output will be a single file with name *directoryname_doc.out*, in which the results will be stored in a tab separated file with this format:

`document_id \t feature_name_1 \t feature_name_2 \t feature_name_3 ... \n`
