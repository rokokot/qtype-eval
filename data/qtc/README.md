---
configs:
  - config_name: base
    data_files:
      - split: train
        path: tydi_train_base.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  # Question Type Control Tasks (3 seeds)
  - config_name: control_question_type_seed1
    data_files:
      - split: train
        path: tydi_train_control_question_type_seed1.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_question_type_seed2
    data_files:
      - split: train
        path: tydi_train_control_question_type_seed2.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_question_type_seed3
    data_files:
      - split: train
        path: tydi_train_control_question_type_seed3.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  # Combined Complexity Control Tasks (3 seeds)
  - config_name: control_complexity_seed1
    data_files:
      - split: train
        path: tydi_train_control_complexity_seed1.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_complexity_seed2
    data_files:
      - split: train
        path: tydi_train_control_complexity_seed2.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_complexity_seed3
    data_files:
      - split: train
        path: tydi_train_control_complexity_seed3.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  # Individual Complexity Metrics Control Tasks (3 seeds each)
  - config_name: control_avg_links_len_seed1
    data_files:
      - split: train
        path: tydi_train_control_avg_links_len_seed1.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_links_len_seed2
    data_files:
      - split: train
        path: tydi_train_control_avg_links_len_seed2.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_links_len_seed3
    data_files:
      - split: train
        path: tydi_train_control_avg_links_len_seed3.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_max_depth_seed1
    data_files:
      - split: train
        path: tydi_train_control_avg_max_depth_seed1.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_max_depth_seed2
    data_files:
      - split: train
        path: tydi_train_control_avg_max_depth_seed2.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_max_depth_seed3
    data_files:
      - split: train
        path: tydi_train_control_avg_max_depth_seed3.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_subordinate_chain_len_seed1
    data_files:
      - split: train
        path: tydi_train_control_avg_subordinate_chain_len_seed1.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_subordinate_chain_len_seed2
    data_files:
      - split: train
        path: tydi_train_control_avg_subordinate_chain_len_seed2.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_subordinate_chain_len_seed3
    data_files:
      - split: train
        path: tydi_train_control_avg_subordinate_chain_len_seed3.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_verb_edges_seed1
    data_files:
      - split: train
        path: tydi_train_control_avg_verb_edges_seed1.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_verb_edges_seed2
    data_files:
      - split: train
        path: tydi_train_control_avg_verb_edges_seed2.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_avg_verb_edges_seed3
    data_files:
      - split: train
        path: tydi_train_control_avg_verb_edges_seed3.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_lexical_density_seed1
    data_files:
      - split: train
        path: tydi_train_control_lexical_density_seed1.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_lexical_density_seed2
    data_files:
      - split: train
        path: tydi_train_control_lexical_density_seed2.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_lexical_density_seed3
    data_files:
      - split: train
        path: tydi_train_control_lexical_density_seed3.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_n_tokens_seed1
    data_files:
      - split: train
        path: tydi_train_control_n_tokens_seed1.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_n_tokens_seed2
    data_files:
      - split: train
        path: tydi_train_control_n_tokens_seed2.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
        
  - config_name: control_n_tokens_seed3
    data_files:
      - split: train
        path: tydi_train_control_n_tokens_seed3.csv
      - split: test
        path: ud_test_base.csv
      - split: validation
        path: dev_base.csv
language: 
  - ar
  - en
  - fi
  - id
  - ja
  - ko
  - ru
license: cc-by-sa-4.0
annotations_creators:
  - found
  - machine-generated
language_creators:
  - found
task_categories:
  - text-classification
  - question-answering
task_ids:
  - text-scoring
  - intent-classification
  - extractive-qa
multilinguality: multilingual
size_categories:
  - 1K<n<10K
source_datasets:
  - original
  - extended|universal-dependencies
  - extended|tydiqa
pretty_name: Question Type and Complexity Dataset
---

# Question Type and Complexity (QTC) Dataset

## Dataset Overview

The Question Type and Complexity (QTC) dataset is a comprehensive resource for linguistics/NLP research focusing on question classification and linguistic complexity analysis across multiple languages. It contains questions from two distinct sources (TyDi QA and Universal Dependencies v2.15), automatically annotated with question types (polar/content) and a set of linguistic complexity features.

**Key Features:**

- 2 question types (polar and content questions) across 7 languages
- 6 numeric linguistic complexity metrics, all normalized using min-max scaling
- Combined/summed complexity scores
- Train(silver)/test(gold)/dev(mix) split using complementary data sources
- Control datasets for evaluating probe selectivity

## Data Sources

### TyDi QA (Training Set)

The primary source for our training data is the TyDi QA dataset (Clark et al., 2020), a typologically diverse question answering benchmark spanning 11 languages. We extracted questions from 7 languages (Arabic, English, Finnish, Indonesian, Japanese, Korean, and Russian), classified them into polar (yes/no) or content (wh-) questions, and analyzed their linguistic complexity.

**Reference:**
Clark, J. H., Choi, E., Collins, M., Garrette, D., Kwiatkowski, T., Nikolaev, V., & Palomaki, J. (2020). TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages. Transactions of the Association for Computational Linguistics, 2020.

### Universal Dependencies (Test Set)

For our test set, we extracted questions from the Universal Dependencies (UD) treebanks (Nivre et al., 2020). UD treebanks provide syntactically annotated sentences across numerous languages, allowing us to identify and extract questions with high precision. We chose UD as our gold standard test set because it provides syntactically annotated data across all our target languages and the universal annotation scheme ensures consistency across languages. Moreover, the high-quality manual annotations make it ideal as a gold standard for evaluation.

**Reference:**
Nivre, J., de Marneffe, M.-C., Ginter, F., Hajič, J., Manning, C. D., Pyysalo, S., Schuster, S., Tyers, F., & Zeman, D. (2020). Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection. In Proceedings of the 12th Language Resources and Evaluation Conference (pp. 4034-4043).

## Data Collection and Processing

### TyDi QA Processing

Data extraction for TyDi began with accessing the dataset via the HuggingFace datasets library. For question classification, we developed language-specific rule-based classifiers using regex and token matching to identify polar and content questions. Languages with well-documented grammatical question markers (Finnish -ko/-kö, Japanese か, English wh-words, etc.) were particularly amenable to accurate classification, as these markers serve as reliable indicators. We verified classification accuracy by cross-checking between our rule-based approach and existing annotations where available.

### Universal Dependencies Processing

The treebanks were chosen partly based on their mean absolute rankings as surveyed by Kulmizev and Nivre (2023). We processed the UD treebanks' CoNLL-U files to extract questions using sentence-final punctuation (?, ？, ؟), language-specific interrogative markers, and syntactic question patterns identifiable through dependency relations. For syntactic processing, we used UDPipe (Straka et al., 2016), which handled tokenization, lemmatization, morphological analysis, and dependency parsing with language-specific models trained on UD treebanks.

Our classification system used the `ud_classifier.py` script to identify and categorize questions from CoNLL-U files based on language-specific pattern matching for interrogative features. Questions were classified as polar or content based on their morphosyntactic properties, with careful filtering to remove incomplete questions, rhetorical questions, and other edge cases that could affect classification accuracy.

**Reference:**
Straka, M., Hajic, J., & Straková, J. (2016). UDPipe: Trainable Pipeline for Processing CoNLL-U Files Performing Tokenization, Morphological Analysis, POS Tagging and Parsing. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16) (pp. 4290-4297).

Kulmizev, A. & Nivre, J. (2023). Investigating UD Treebanks via Dataset Difficulty Measures. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (pp. 1076-1089).  

### Linguistic Complexity Feature Scoring

Our linguistic analysis pipeline consisted of two main components. First, we processed each question through UDPipe to generate CoNLL-U format parse trees using our `scripts/data-processing/run_udpipe.py`. These parsed trees were then analyzed using our `scripts/data_processing/profiling-UD/custom-profile.py` script to extract linguistic features. We normalized the results and aggregated them to provide a single complexity score for each question.

The feature extraction framework extends the approach of Brunato et al. (2020) on linguistic complexity profiling. This allowed us to process parsed sentences and extract a comprehensive set of complexity features that capture different dimensions of linguistic difficulty.

**Reference:**
Brunato, D., Cimino, A., Dell'Orletta, F., Venturi, G., & Montemagni, S. (2020). Profiling-UD: A Tool for Linguistic Profiling of Texts. In Proceedings of The 12th Language Resources and Evaluation Conference (pp. 7145-7151).

## Preprocessing and Feature Extraction

We normalized all linguistic features using min-max scaling per language. This approach ensures cross-linguistic comparability by mapping each feature to a 0-1 range for each language separately.

For the TyDi data, we applied strategic downsampling using token-based stratified sampling. This balances the distribution across languages and question types while preserving the original sentence length distribution, resulting in a more balanced dataset without sacrificing linguistic diversity.

The final step involved calculating a combined complexity score from the normalized features. This provides researchers with a single metric that consolidates multiple dimensions of linguistic complexity into one value for easier analysis and comparison.

## Dataset Structure

The dataset is organized into three main components corresponding to the train/dev/test splits:

```text
QTC-Dataset
├── base                               
│   ├── tydi_train_base.csv           
│   ├── dev_base.csv                   
│   └── ud_test_base.csv               
├── control_question_type_seed1        
│   ├── tydi_train_control_question_type_seed1.csv
│   ├── dev_base.csv
│   └── ud_test_base.csv
├── control_complexity_seed1          
│   ├── tydi_train_control_complexity_seed1.csv
│   ├── dev_base.csv
│   └── ud_test_base.csv
└── control_[metric]_seed[n]           
    ├── tydi_train_control_[metric]_seed[n].csv
    ├── dev_base.csv
    └── ud_test_base.csv
```

## Control Tasks
The dataset includes control task variants for evaluating probe selectivity, following the methodology of Hewitt & Liang (2019). Each control task preserves the structure of the original dataset but with randomized target values:

- Question Type Controls: Three seeds of randomly shuffled question type labels (within each language)
- Complexity Score Controls: Three seeds of randomly shuffled complexity scores (within each language)
- Individual Metric Controls: Three seeds for each of the six linguistic complexity metrics

These control tasks allow researchers to assess whether a probe is truly learning linguistic structure or simply memorizing patterns in the data.

**Reference:**
Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (pp. 2733-2743).


## Features Description

### Core Attributes

| Feature | Type | Description |
|---------|------|-------------|
| `unique_id` | string | Unique identifier for each question |
| `text` | string | The question text |
| `language` | string | ISO language code (ar, en, fi, id, ja, ko, ru) |
| `question_type` | int | Binary encoding (0 = content, 1 = polar) |
| `complexity_score` | float | Combined linguistic complexity score |
| `lang_norm_complexity_score`| float | Language-normalized complexity score (0-1)|

### Linguistic Features

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `avg_links_len` | Average syntactic dependency length | Min-max scaling by language |
| `avg_max_depth` | Average maximum dependency tree depth | Min-max scaling by language |
| `avg_subordinate_chain_len` | Average length of subordinate clause chains | Min-max scaling by language |
| `avg_verb_edges` | Average number of edges connected to verbal nodes | Min-max scaling by language |
| `lexical_density` | Ratio of content words to total words | Min-max scaling by language |
| `n_tokens` | Number of tokens in the question | Min-max scaling by language |

## Linguistic Feature Significance

### Syntactic Complexity

The `avg_links_len` feature captures the average syntactic dependency length, which indicates processing difficulty as syntactically related elements become further apart. Longer dependencies typically correlate with increased cognitive processing load. Similarly, `avg_max_depth` measures the depth of dependency trees, with deeper structures indicating higher levels of embedding and consequently greater syntactic complexity.

### Hierarchical Structure

The `avg_subordinate_chain_len` feature quantifies the length of subordinate clause chains. Longer chains create more dispersed hierarchical structures, which can be harder to process and understand. This feature helps capture how clausal embedding contributes to overall question complexity.

### Lexical and Semantic Load

The `lexical_density` feature measures the ratio of content words to total words. Higher density indicates a greater proportion of information-carrying words relative to function words, resulting in higher information density. The `avg_verb_edges` feature counts the average number of edges connected to verbal nodes, with more edges indicating more complex predicate-argument structures. Finally, `n_tokens` captures sentence length, which correlates with information content and overall processing difficulty.

## Silver and Gold Standard Data

### Silver Standard (TyDi QA)

The TyDi QA component serves as our silver standard training data. It offers a larger volume of questions drawn from real-world information-seeking contexts. These questions were automatically processed and classified through our custom pipeline, then strategically downsampled to balance distribution across languages and question types. The TyDi data represents authentic question complexity in information retrieval scenarios, making it ideal for training models to recognize patterns in question complexity across languages.

### Gold Standard (Universal Dependencies)

The Universal Dependencies component forms our gold standard test set. These questions come with manually annotated syntactic structures, providing high-quality linguistic information. The UD data represents a diverse range of linguistic contexts and genres, and unlike the TyDi data, it was not downsampled to preserve all available gold-standard annotations. While smaller in volume, the UD component offers superior annotation quality and precision, making it an ideal benchmark for evaluating question complexity models.

## Usage Examples

### Basic Usage
```python
from datasets import load_dataset

# Load the base dataset
dataset = load_dataset("rokokot/question-type-and-complexity", name="base")

# Access the training split (TyDi data)
tydi_data = dataset["train"]

# Access the validation split (Dev data)
dev_data = dataset["validation"]

# Access the test split (UD data)
ud_data = dataset["test"]

# Filter for a specific language
finnish_questions = dataset["train"].filter(lambda x: x["language"] == "fi")

# Filter for a specific type
polar_questions = dataset["train"].filter(lambda x: x["question_type"] == 1)
content_questions = dataset["train"].filter(lambda x: x["question_type"] == 0)
```
### Working with Control Tasks
```python
from datasets import load_dataset

# Load the original dataset
original_data = load_dataset("rokokot/question-type-and-complexity", name="base")

# Load question type control tasks
question_control1 = load_dataset("rokokot/question-type-and-complexity", name="control_question_type_seed1")
question_control2 = load_dataset("rokokot/question-type-and-complexity", name="control_question_type_seed2")
question_control3 = load_dataset("rokokot/question-type-and-complexity", name="control_question_type_seed3")

# Load complexity score control tasks
complexity_control1 = load_dataset("rokokot/question-type-and-complexity", name="control_complexity_seed1")

# Load individual metric control tasks
links_control = load_dataset("rokokot/question-type-and-complexity", name="control_avg_links_len_seed1")
depth_control = load_dataset("rokokot/question-type-and-complexity", name="control_avg_max_depth_seed2")
```
## Research Applications

This dataset enables various research directions:

1. **Cross-linguistic question complexity**: Investigate how syntactic complexity varies across languages and question types.
2. **Question answering systems**: Analyze how question complexity affects QA system performance.
3. **Language teaching**: Develop difficulty-aware educational materials for language learners.
4. **Psycholinguistics**: Study processing difficulty predictions for different question constructions.
5. **Machine translation**: Evaluate translation symmetry for questions of varying complexity.

## Citation

If you use this dataset in your research, please cite it as follows:

```bibtex
@dataset{rokokot2025qtc,
  author    = {Robin Kokot},
  title     = {Question Type and Complexity (QTC) Dataset},
  year      = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/rokokot/question-complexity}},
}
```

Additionally, please cite the underlying data sources and tools:

```bibtex
@article{clark2020tydi,
  title={TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
  author={Clark, Jonathan H and Choi, Eunsol and Collins, Michael and Garrette, Dan and Kwiatkowski, Tom and 
  Nikolaev, Vitaly and Palomaki, Jennimaria},  
  journal={Transactions of the Association for Computational Linguistics},
  volume={8},
  pages={454--470},
  year={2020},
  publisher={MIT Press}
}

@inproceedings{nivre2020universal,
  title={Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection},
  author={Nivre, Joakim and de Marneffe, Marie-Catherine and Ginter, Filip and Haji{\v{c}}, Jan and Manning, 
  Christopher D and Pyysalo, Sampo and Schuster, Sebastian and Tyers, Francis and Zeman, Daniel},
  booktitle={Proceedings of the 12th Language Resources and Evaluation Conference},
  pages={4034--4043},
  year={2020}
}

@inproceedings{straka2016udpipe,
  title={UDPipe: Trainable Pipeline for Processing CoNLL-U Files Performing Tokenization, Morphological Analysis, 
  POS Tagging and Parsing},
  author={Straka, Milan and Haji{\v{c}}, Jan and Strakov{\'a}, Jana},
  booktitle={Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16)},
  pages={4290--4297},
  year={2016}
}

@inproceedings{brunato2020profiling,
  title={Profiling-UD: A Tool for Linguistic Profiling of Texts},
  author={Brunato, Dominique and Cimino, Andrea and Dell'Orletta, Felice and Venturi, Giulia and Montemagni, Simonetta},
  booktitle={Proceedings of The 12th Language Resources and Evaluation Conference},
  pages={7145--7151},
  year={2020}
}
```

## License

This dataset is released under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license, in accordance with the licensing of the underlying TyDi QA and Universal Dependencies datasets.

## Acknowledgments

This dataset builds upon the work of the TyDi QA and Universal Dependencies research communities. We are grateful for their contributions to multilingual NLP resources. The linguistic complexity analysis was supported by the tools released by Brunato et al. (2020) and Straka et al.(2016). We acknowledge the critical role of UDPipe in providing robust syntactic parsing across multiple languages, which formed the foundation of our feature extraction pipeline.
