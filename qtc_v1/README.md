---
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
- 8 fine-grained linguistic complexity metrics
- Combined/summed complexity scores
- Train(silver)/test(gold) split using complementary data sources
- Ablation sets for linguistic complexity feature importance analysis

## Data Sources

### TyDi QA (Training Set)

The primary source for our training data is the TyDi QA dataset (Clark et al., 2020), a typologically diverse question answering benchmark spanning 11 languages. We extracted questions from 7 languages (Arabic, English, Finnish, Indonesian, Japanese, Korean, and Russian), classified them into polar (yes/no) or content (wh-) questions, and analyzed their linguistic complexity.

**Reference:**
Clark, J. H., Choi, E., Collins, M., Garrette, D., Kwiatkowski, T., Nikolaev, V., & Palomaki, J. (2020). TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages. Transactions of the Association for Computational Linguistics, 2020.

### Universal Dependencies (Test Set)

For our test set, we extracted questions from the Universal Dependencies (UD) treebanks (Nivre et al., 2020). UD treebanks provide syntactically annotated sentences across numerous languages, allowing us to identify and extract questions with high precision. We specifically chose UD as our gold standard test set because:

1. It provides syntactically annotated data across all our target languages
2. The universal annotation scheme ensures consistency across languages
3. The high-quality manual annotations make it ideal as a gold standard for evaluation
4. The presence of sentence-level annotations allows precise question identification

**Reference:**
Nivre, J., de Marneffe, M.-C., Ginter, F., Hajič, J., Manning, C. D., Pyysalo, S., Schuster, S., Tyers, F., & Zeman, D. (2020). Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection. In Proceedings of the 12th Language Resources and Evaluation Conference (pp. 4034-4043).

## Data Collection and Processing

### Question Extraction and Classification Methodology

#### TyDi QA Processing

1. **Data Extraction:** We utilized the HuggingFace datasets library to access the TyDi QA primary task dataset.
2. **Question Classification:**
   - We developed language-specific rule-based classifiers using regex and token matching for polar and content questions
   - For languages with well-documented grammatical question markers (e.g., Finnish -ko/-kö, Japanese か, English wh-words etc.), we used these as reliable indicators
   - For languages with less explicit markers, we employed combined approaches using question word lists and positional patterns
   - We cross-validated our classifications against the available annotations in TyDi QA's yes_no_answer field
3. **Validation:** We employed a multi-step validation process:
   - Manual review of random samples by language specialists
   - Consistency checking between rule-based classification and existing annotations

#### Universal Dependencies Processing

1. **Dataset Selection:** We curated a selection of UD treebanks across our 7 target languages, prioritizing treebanks with conversational and question-containing content. The treebanks for partly chosen for their mean absolute rankings as surveyed by (Kulmizev, A. and Nivre, J., 2023 )
2. **CoNLL-U Parsing:** We processed the UD treebanks' CoNLL-U files to extract questions using:
   - Sentence-final punctuation (?, ？, ؟)
   - Language-specific interrogative markers
   - Syntactic question patterns identifiable through dependency relations
3. **UDPipe Processing:** For syntactic processing, we employed UDPipe (Straka et al., 2016), a state-of-the-art tool for UD format processing:
   - Tokenization, lemmatization, and morphological analysis
   - Dependency parsing to extract syntactic structures
   - Language-specific models trained on UD treebanks
4. **Question Classification:**
   - We developed the `ud_classifier.py` script to identify and classify questions from CoNLL-U files
   - The script implements language-specific pattern matching for interrogative features
   - Questions were classified as polar or content based on morphosyntactic properties
   - Extensive filtering to remove incomplete questions, rhetorical questions, and other edge cases

**Reference:**
Straka, M., Hajic, J., & Straková, J. (2016). UDPipe: Trainable Pipeline for Processing CoNLL-U Files Performing Tokenization, Morphological Analysis, POS Tagging and Parsing. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16) (pp. 4290-4297).

Kulmizev, A. & Nivre, J. (2023). Investigating UD Treebanks via Dataset Difficulty Measures. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics (pp. 1076-1089).  

### Linguistic Complexity Feature Scoring

We implemented a linguistic analysis pipeline using:

1. **Profiling-UD Framework:**
   - We developed a custom profiling framework for extracting linguistic complexity features
   - Our implementation was inspired by and extends the approach of Brunato et al. (2020) on linguistic complexity profiling
   - The framework processes parsed sentences to extract a wide range of complexity features using the profiling-UD tool developed by Brunato et al. (2020)

2. **Feature Extraction Process for the TyDi Data:**
   - Each question was processed through UDPipe to generate CoNLL-U format parse trees using our `scripts/data-processing/run_udpipe.py`
   - The parsed trees were analyzed using our  `scripts/data_processing/profiling-UD/custom-profile.py` script
   - Results were normalized and aggregated to provide a single complexity score

**References:**
Brunato, D., Cimino, A., Dell'Orletta, F., Venturi, G., & Montemagni, S. (2020). Profiling-UD: A Tool for Linguistic Profiling of Texts. In Proceedings of The 12th Language Resources and Evaluation Conference (pp. 7145-7151).

## Preprocessing and Feature Extraction

Our preprocessing pipeline involved several key stages:

1. **Question Extraction and Classification**
   - TyDi: Questions were classified using a combination of annotations from the original dataset and linguistic pattern matching.
   - UD: Questions were extracted from UD treebanks using sentence-final punctuation and language-specific question markers.

2. **Syntactic Parsing**
   - All questions were parsed using UDPipe with language-specific models
   - Parsed structures were normalized to handle language-specific variations

3. **Linguistic Complexity Analysis**
   - All questions were analyzed using a custom profile of linguistic complexity metrics, focusing on syntactic, morphological, and lexical features.

4. **Normalization**
   - Features were normalized using a combination of methods to ensure cross-linguistic comparability:
     - Language centering: `avg_links_len`, `avg_max_depth`
     - Log normalization: `n_tokens`
     - Min-max scaling per language: `avg_verb_edges`
     - Raw values: `avg_subordinate_chain_len`, `lexical_density`

5. **Downsampling** (TyDi data only)
   - The TyDi data was strategically downsampled using token-based stratified sampling to balance the distribution across languages and question types, while preserving the original sentence length distribution.

6. **Complexity Score Calculation**
   - Combined normalized features with final z-score standardization to produce a single complexity metric centred around 0 as baseline complexity with standard deviation of 1.

7. **Ablation Analysis**
   - Generated variant datasets excluding a single feature each time in order to isolate individual feature importance.

## Dataset Structure

The dataset is organized into two main components with parallel internal structures. Each folder contains language specific files, as well as a combined csv file. The final datasets can be found in QTC-Dataset/{TyDi, UD}/complexity and QTC-Dataset/{TyDi, UD}/ablation

```text
QTC-Dataset
├── tydi_{time_stamp}
│   ├── ablation
│   ├── complexity
│   ├── downsampled
│   └── normalized
└── ud_{time_stamp}
    ├── ablation
    ├── complexity
    └── normalized
```

## Features Description

### Core Attributes

| Feature | Type | Description |
|---------|------|-------------|
| `unique_id` | string | Unique identifier for each question |
| `text` | string | The question text |
| `language` | string | ISO language code (ar, en, fi, id, ja, ko, ru) |
| `type_original` | string | Question type label ("polar" or "content") |
| `question_type` | int | Binary encoding (0 = content, 1 = polar) |
| `complexity_score` | float | Combined linguistic complexity score |

### Linguistic Features

| Feature | Description | Normalization |
|---------|-------------|---------------|
| `avg_links_len` | Average syntactic dependency length | Language centering |
| `avg_max_depth` | Average maximum dependency tree depth | Language centering |
| `avg_subordinate_chain_len` | Average length of subordinate clause chains | Raw values |
| `avg_verb_edges` | Average number of edges connected to verbal nodes | Min-max scaling |
| `lexical_density` | Ratio of content words to total words | Raw values |
| `n_tokens` | Number of tokens in the question | Log normalization |

## Linguistic Feature Significance

### Syntactic Complexity

- **avg_links_len**: Longer dependency links indicate greater processing difficulty because syntactically related elements are further apart.
- **avg_max_depth**: Deeper dependency trees indicate higher embedding levels.

### Hierarchical Structure

- **avg_subordinate_chain_len**: Longer chains of subordinate clauses create more dispersed hierarchical structures.

### Lexical and Semantic Load

- **lexical_density**: Higher density indicates higher ratio of information-carrying words relative to function words.
- **avg_verb_edges**: More edges connected to verbs indicate more complex predicate-argument structures.
- **n_tokens**: Sentence length correlates with information content and processing difficulty.

## Silver and Gold Standard Data

Our dataset contains two complementary components:

### Silver Standard (TyDi QA)

- Larger volume of questions from information-seeking contexts
- Automatically processed and classified using our custom pipeline
- Downsampled to balance distribution across languages and question types
- Primarily used as training data
- Represents real-world question complexity in information retrieval contexts

### Gold Standard (Universal Dependencies)

- Manually annotated for syntactic structure
- Represents a diverse range of linguistic contexts and genres
- Not downsampled to preserve all available gold-standard annotations
- Provides a reliable test benchmark for evaluating question complexity models
- Smaller in volume but higher in annotation quality and precision

## Ablation Studies

The dataset includes ablation sets where a single linguistic feature is excluded from the complexity calculation each time, allowing researchers to evaluate feature importance.

## Usage Examples

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("rokokot/question-complexity")

# Access the training split (TyDi data)
tydi_data = dataset["train"]

# Access the test split (UD data)
ud_data = dataset["test"]

# Filter for a specific language
finnish_questions = dataset.filter(lambda example: example["language"] == "fi")

# Get high complexity questions
complex_questions = dataset.filter(lambda example: example["complexity_score"] > 1.0)

# Compare polar vs. content questions
polar_questions = dataset.filter(lambda example: example["question_type"] == 1)
content_questions = dataset.filter(lambda example: example["question_type"] == 0)
```

## Research Applications

This dataset enables various research directions:

1. **Cross-linguistic question complexity**: Investigate how syntactic complexity varies across languages and question types.
2. **Question answering systems**: Analyze how question complexity affects QA system performance.
3. **Language teaching**: Develop difficulty-aware educational materials for language learners.
4. **Psycholinguistics**: Study processing difficulty predictions for different question constructions.
5. **Machine translation**: Assess translation quality preservation for questions of varying complexity.

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
