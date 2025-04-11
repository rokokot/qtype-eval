# Multilingual Question-Type and Complexity Probing Framework

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.30%2B-green)](https://huggingface.co/docs/transformers/index)
[![Hydra](https://img.shields.io/badge/Hydra-1.3-blue)](https://hydra.cc/)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/rokokot/question-type-and-complexity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repo contains the code, data, and documentation for the experimental framework developed in the scope of the research described in  "Type and Complexity Signals in Multilingual Question Representations: A Diagnostic Study with Selective Control Tasks." [link] We investigate how pre-trained multilingual contextual encoder models encode sentence-level linguistic features across 7 typollogically diverse languages (Arabic, English, Finnish, Indonesian, Japanese, Korean, and Russian).

Our experiments address three research questions:

1. How well do multilingual models encode sentence-level properties compared to traditional feature representations(TF-IDF)?
2. Which linguistic property signals are most expressive at different layers of the model?
3. Do diagnostic classifiers learn true linguistic composition or simply memorize patterns?

We approach these by training diagnostic classifiers on top of frozen model representations in order to predict categorical and continuous linguistic properties of the input text. Diagnostic classification tasks are motivated by the idea that successfully training a classifier model to predict a property of sentences implies the property is encoded in the representation. In contrast, poor performance means the property is not encoded in a way that is useful for the model.

## Dataset

We make extensive use the custom  Question Type and Complexity (QTC) dataset, available at [Hugging Face]([https://huggingface.co/datasets/rokokot/question-type-and-complexity). The dataset was developed by automatically annotating data from two sources:

1. **Training set(silver data)**: Derived from TyDiQA-goldP (cite), automatically annotated using rule-based and regex methods for question types, and UDPipe (cite) with linguistic profiling toolkit (cite) for complexity metrics
2. **Test set (gold data)**: Extracted from Universal Dependency treebanks with high-quality annotations
3. **Development set**: A combination of both sources to provide a balanced validation set

The dataset contains the following labels:

a. Question types: Binary classification of questions as polar (yes/no) or content (wh-)
b. 6 different complexity sub-metrics: Average dependency links length, Average maximum depth, Average subordinate chain length, Average verb edges, Lexical density, Number of tokens
c. Combined complexity metric: One unweighted sum as an abstract complexity score
d. Control sets: Created by intra-language shuffling of labels to test model selectivity

## Experimental Framework



### License

This project is licensed under the MIT License - see the LICENSE file for details.

### References

(Clark et al. (2020))[https://aclanthology.org/2020.tacl-1.30/] - TyDi QA dataset
(Nivre et al. (2020))[https://aclanthology.org/2020.lrec-1.497/] - Universal Dependencies
(Conneau et al. (2018))[https://arxiv.org/abs/1805.01070] - Probing task design
(Sahin et al. (2020))[https://aclanthology.org/2020.cl-2.4/] - LINSPECTOR multilingual probing
(Brunato et al. (2020))[https://aclanthology.org/2020.lrec-1.883/] - Profiling-UD linguistic complexity metric analysis

