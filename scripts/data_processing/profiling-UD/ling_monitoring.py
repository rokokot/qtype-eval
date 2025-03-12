# -*- coding: utf-8  -*-
# !/usr/bin/python

import argparse
import codecs
import glob
import os
from hashlib import md5
try:
    from .utils import vectorize
    from .senttok import Sentence, Token
    from .compute_features import compute_features
except:
    from utils import vectorize
    from senttok import Sentence, Token
    from compute_features import compute_features
    

def read_dictionary(dict_file):
    """
    Reads the dictionary file and builds a dict

    :param dict_file: (string) path to dictionary file
    :return: (dict) dictionary
    """
    dictionary = {}
    with codecs.open(dict_file, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            dictionary[line[0]] = line[1].strip()
    return dictionary


def read_and_compute(sentences, type_chosen, *fund_dictionary):
    '''


    :param sentences:
    :param type_chosen:
    :param fund_dictionary:
    :return:
    '''

    if fund_dictionary:
        # Import file of 'Dizionario Fondamentale (De Mauro)' (only for Italian for now)
        dictionary = read_dictionary(fund_dictionary[0])
    else:
        dictionary = None

    sent_features = {}
    doc_sent = []
    if type_chosen == 0:  # analyse single sentences
        for sent_id, sentence in sentences[1].items():
            doc_sent.append(sentence)
            # Compute linguistic features and store them in a dictionary {Key: sentence_id, Value: sentence_features}
            # The function takes in input the sentences of every document and the Dizionario Fondamentale, if present
            features = compute_features(doc_sent, dictionary, type_analysis=type_chosen)
            sent_features[sent_id] = features
            doc_sent = []
    elif type_chosen == 1:
        doc_id = sentences[0]
        for sentence in sentences[1].values():
            doc_sent.append(sentence)
        features = compute_features(doc_sent, dictionary, type_analysis=type_chosen)
        sent_features[doc_id] = features
    return sent_features


def read_file(input_file):
    """
    Reads the input file and build a data structure containing a list of documents (or sentences?)

    :param input_file: (string) path to input file
    :return: ([[Sentence]]) list of documents in file
    """

    docs = {}
    sentences = {}
    sentence = []

    with codecs.open(input_file, 'r', 'utf-8') as f:

        doc_id = os.path.basename(f.name)  # use filename as id of the document

        sent_id = ''
        for line in f:
            if line == '\n' and sentence:
                mysent = Sentence(sentence)  # Create an object Sentence
                # Create identifier
                m = md5()
                identifier = ''
                for token in sentence:
                    identifier += token.form + ' '
                m.update(identifier.encode('utf-8')[:-1])
                #sentences[m.hexdigest()] = mysent
                sentences[sent_id] = mysent
                sentence = []
            elif line == '\n':  # Skip blank lines
                pass
            elif line.startswith('#'):
                if line.startswith('# sent_id'):
                    sent_id = line.rstrip('\n').split('= ')[1]
                pass

            # If a line of text is found, divide it in tokens and create a Token() object
            # Note: text is TAB-separated
            else:
                line = line.strip().split('\t')
                if '-' in line[0]:
                    pass
                else:
                    sentence.append(Token(line))

        if sentence:
            mysent = Sentence(sentence)
            m = md5()
            identifier = ""
            for token in sentence:
                identifier += token.form + " "
            m.update(identifier.encode('utf-8'))
            sentences[m.hexdigest()] = mysent

        docs[doc_id] = sentences
        return docs


def dir_path(path):
    files_path = []

    if os.path.exists(path):
        if os.path.isdir(path):
            files_path = glob.glob(path + '*')
            return files_path
        else:
            files_path.append(path)
            return files_path
    else:
        raise argparse.ArgumentTypeError("readable_dir:{path} is not a valid path")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract linguistic features from sentences or documents parsed in '
                                                 'CoNLL-U format')
    parser.add_argument('-p', '--path', type=dir_path, help='specify the path of the directory that contains the file or'
                                                            'the files you want to analyse or specify the single file'
                                                            'you want to analyse.')
    parser.add_argument('-d', '--dict', type=str, required=False,
                        help='dictionary that contains the categories of the frequency of use of lemmas and words (see '
                             'the README for details on the structure of the dictionary)')
    parser.add_argument('-t', '--type', type=int, default=1, choices=[0, 1], help='select if you want to analyse the '
                                                                                  'single sentences [0] contained in a '
                                                                                  'file or the whole file as a document'
                                                                                  '[1]. Default is 1 for documents.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    files = args.path
    docs_features = {}
    for name in files:
        docs = read_file(name)
        for doc in docs.items():  # here doc is passed as a tuple
            if args.dict:
                computed_features = read_and_compute(doc, args.type, args.dict)
            else:
                computed_features = read_and_compute(doc, args.type)
            if args.type == 1:
                docs_features.update(computed_features)
                print 
        if args.type == 0:
            my_output = name.strip().split('/')
            output_name = my_output[-1]

            try:
                outfile = codecs.open('output_results/' + output_name + '_sent.out', 'w')
            except OSError:
                os.makedirs('output_results')
                outfile = codecs.open('output_results/' + output_name + '_sent.out', 'w')

            outfile.write(vectorize(computed_features))
            outfile.close()

    if args.type == 1:
        my_output = name.strip().split('/')
        try:
            output_name = my_output[-2]
        except:
            print()
            print("ERROR: For document-level analysis the input file(s) must be placed into a folder!!")
            print()
            exit()
        print ("---->", output_name)
        try:
            outfile = codecs.open('output_results/' + output_name + '_doc.out', 'w')
        except OSError:
            os.makedirs('output_results')
            outfile = codecs.open('output_results/' + output_name + '_doc.out', 'w')

        outfile.write(vectorize(docs_features))
        outfile.close()
