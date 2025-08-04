# tests/fixtures/sample_data.py
"""
Sample data generation utilities for testing TF-IDF integration.
Provides realistic test data that mimics the structure of real datasets.
"""

import numpy as np
import scipy.sparse
import json
import random
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Sample texts for different languages
SAMPLE_TEXTS = {
    'en': [
        "What is the capital of France?",
        "How do you solve this problem?",
        "This is a declarative statement.",
        "Where can I find the answer?",
        "The weather is nice today.",
        "Can you help me with this task?",
        "Machine learning is fascinating.",
        "Why is the sky blue?",
        "Programming requires logical thinking.",
        "What time is the meeting?"
    ],
    'ru': [
        "Что является столицей Франции?",
        "Как решить эту проблему?",
        "Это декларативное утверждение.",
        "Где я могу найти ответ?",
        "Сегодня хорошая погода.",
        "Можете ли вы помочь мне с этой задачей?",
        "Машинное обучение увлекательно.",
        "Почему небо голубое?",
        "Программирование требует логического мышления.",
        "Во сколько встреча?"
    ],
    'ar': [
        "ما هي عاصمة فرنسا؟",
        "كيف تحل هذه المشكلة؟",
        "هذا بيان تصريحي.",
        "أين يمكنني العثور على الإجابة؟",
        "الطقس جميل اليوم.",
        "هل يمكنك مساعدتي في هذه المهمة؟",
        "التعلم الآلي رائع.",
        "لماذا السماء زرقاء؟",
        "البرمجة تتطلب التفكير المنطقي.",
        "كم الوقت الاجتماع؟"
    ],
    'fi': [
        "Mikä on Ranskan pääkaupunki?",
        "Miten tämä ongelma ratkaistaan?",
        "Tämä on väittävä lause.",
        "Mistä löydän vastauksen?",
        "Sää on kaunis tänään.",
        "Voitko auttaa minua tässä tehtävässä?",
        "Koneoppiminen on kiehtovaa.",
        "Miksi taivas on sininen?",
        "Ohjelmointi vaatii loogista ajattelua.",
        "Mihin aikaan on kokous?"
    ],
    'id': [
        "Apa ibu kota Prancis?",
        "Bagaimana cara menyelesaikan masalah ini?",
        "Ini adalah pernyataan deklaratif.",
        "Di mana saya bisa menemukan jawabannya?",
        "Cuaca hari ini bagus.",
        "Bisakah Anda membantu saya dengan tugas ini?",
        "Pembelajaran mesin itu menarik.",
        "Mengapa langit biru?",
        "Pemrograman memerlukan pemikiran logis.",
        "Jam berapa pertemuannya?"
    ],
    'ja': [
        "フランスの首都は何ですか？",
        "この問題をどう解決しますか？",
        "これは宣言文です。",
        "答えはどこで見つけられますか？",
        "今日は天気がいいです。",
        "この作業を手伝ってもらえますか？",
        "機械学習は魅力的です。",
        "なぜ空は青いのですか？",
        "プログラミングには論理的思考が必要です。",
        "会議は何時ですか？"
    ],
    'ko': [
        "프랑스의 수도는 무엇입니까?",
        "이 문제를 어떻게 해결합니까?",
        "이것은 선언문입니다.",
        "답을 어디서 찾을 수 있습니까?",
        "오늘 날씨가 좋습니다.",
        "이 작업을 도와주실 수 있습니까?",
        "기계학습은 매력적입니다.",
        "하늘이 왜 파랗습니까?",
        "프로그래밍은 논리적 사고가 필요합니다.",
        "회의는 몇 시입니까?"
    ]
}

# Question type patterns (1 = question, 0 = statement)
QUESTION_PATTERNS = {
    'en': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'ru': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'ar': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'fi': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'id': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'ja': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'ko': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

# Complexity scores (normalized)
COMPLEXITY_SCORES = {
    'en': [0.6, 0.7, 0.4, 0.5, 0.3, 0.6, 0.8, 0.5, 0.9, 0.4],
    'ru': [0.7, 0.8, 0.5, 0.6, 0.4, 0.7, 0.9, 0.6, 1.0, 0.5],
    'ar': [0.8, 0.9, 0.6, 0.7, 0.5, 0.8, 1.0, 0.7, 0.9, 0.6],
    'fi': [0.7, 0.8, 0.5, 0.6, 0.4, 0.7, 0.9, 0.6, 1.0, 0.5],
    'id': [0.6, 0.7, 0.4, 0.5, 0.3, 0.6, 0.8, 0.5, 0.9, 0.4],
    'ja': [0.8, 0.9, 0.6, 0.7, 0.5, 0.8, 1.0, 0.7, 0.9, 0.6],
    'ko': [0.7, 0.8, 0.5, 0.6, 0.4, 0.7, 0.9, 0.6, 1.0, 0.5]
}

# Submetric values (realistic ranges)
SUBMETRIC_VALUES = {
    'avg_links_len': {
        'en': [0.3, 0.4, 0.2, 0.3, 0.1, 0.4, 0.5, 0.3, 0.6, 0.2],
        'ru': [0.4, 0.5, 0.3, 0.4, 0.2, 0.5, 0.6, 0.4, 0.7, 0.3],
        'ar': [0.5, 0.6, 0.4, 0.5, 0.3, 0.6, 0.7, 0.5, 0.8, 0.4]
    },
    'avg_max_depth': {
        'en': [0.4, 0.5, 0.3, 0.4, 0.2, 0.5, 0.6, 0.4, 0.7, 0.3],
        'ru': [0.5, 0.6, 0.4, 0.5, 0.3, 0.6, 0.7, 0.5, 0.8, 0.4],
        'ar': [0.6, 0.7, 0.5, 0.6, 0.4, 0.7, 0.8, 0.6, 0.9, 0.5]
    },
    'avg_subordinate_chain_len': {
        'en': [0.2, 0.3, 0.1, 0.2, 0.0, 0.3, 0.4, 0.2, 0.5, 0.1],
        'ru': [0.3, 0.4, 0.2, 0.3, 0.1, 0.4, 0.5, 0.3, 0.6, 0.2],
        'ar': [0.4, 0.5, 0.3, 0.4, 0.2, 0.5, 0.6, 0.4, 0.7, 0.3]
    },
    'avg_verb_edges': {
        'en': [0.5, 0.6, 0.4, 0.5, 0.3, 0.6, 0.7, 0.5, 0.8, 0.4],
        'ru': [0.6, 0.7, 0.5, 0.6, 0.4, 0.7, 0.8, 0.6, 0.9, 0.5],
        'ar': [0.7, 0.8, 0.6, 0.7, 0.5, 0.8, 0.9, 0.7, 1.0, 0.6]
    },
    'lexical_density': {
        'en': [0.6, 0.7, 0.5, 0.6, 0.4, 0.7, 0.8, 0.6, 0.9, 0.5],
        'ru': [0.7, 0.8, 0.6, 0.7, 0.5, 0.8, 0.9, 0.7, 1.0, 0.6],
        'ar': [0.8, 0.9, 0.7, 0.8, 0.6, 0.9, 1.0, 0.8, 0.9, 0.7]
    },
    'n_tokens': {
        'en': [0.4, 0.5, 0.3, 0.4, 0.2, 0.5, 0.6, 0.4, 0.7, 0.3],
        'ru': [0.5, 0.6, 0.4, 0.5, 0.3, 0.6, 0.7, 0.5, 0.8, 0.4],
        'ar': [0.6, 0.7, 0.5, 0.6, 0.4, 0.7, 0.8, 0.6, 0.9, 0.5]
    }
}


def create_sample_dataset(
    n_samples: int = 100,
    task_type: str = 'classification',
    n_languages: int = 3,
    languages: Optional[List[str]] = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Create a sample dataset for testing.
    
    Args:
        n_samples: Total number of samples to generate
        task_type: 'classification' or 'regression'
        n_languages: Number of languages to include
        languages: Specific languages to use (if None, uses first n_languages)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing sample dataset
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if languages is None:
        available_languages = list(SAMPLE_TEXTS.keys())
        languages = available_languages[:n_languages]
    
    dataset = {
        'text': [],
        'language': [],
        'question_type': [],
        'lang_norm_complexity_score': []
    }
    
    # Add submetric columns
    for submetric in SUBMETRIC_VALUES.keys():
        dataset[submetric] = []
    
    samples_per_lang = n_samples // len(languages)
    
    for lang in languages:
        lang_texts = SAMPLE_TEXTS.get(lang, SAMPLE_TEXTS['en'])
        lang_question_types = QUESTION_PATTERNS.get(lang, QUESTION_PATTERNS['en'])
        lang_complexity = COMPLEXITY_SCORES.get(lang, COMPLEXITY_SCORES['en'])
        
        for i in range(samples_per_lang):
            # Cycle through available texts
            text_idx = i % len(lang_texts)
            
            dataset['text'].append(lang_texts[text_idx])
            dataset['language'].append(lang)
            dataset['question_type'].append(lang_question_types[text_idx])
            
            # Add some noise to complexity scores
            base_complexity = lang_complexity[text_idx]
            noise = np.random.normal(0, 0.1)  # Small amount of noise
            complexity_score = max(0, min(1, base_complexity + noise))
            dataset['lang_norm_complexity_score'].append(complexity_score)
            
            # Add submetric values
            for submetric in SUBMETRIC_VALUES.keys():
                lang_submetric = SUBMETRIC_VALUES[submetric].get(lang, SUBMETRIC_VALUES[submetric]['en'])
                base_value = lang_submetric[text_idx]
                submetric_noise = np.random.normal(0, 0.05)
                submetric_value = max(0, min(1, base_value + submetric_noise))
                dataset[submetric].append(submetric_value)
    
    logger.info(f"Created sample dataset with {len(dataset['text'])} samples across {len(languages)} languages")
    return dataset


def create_sample_tfidf_features(
    n_samples: int = 100,
    vocab_size: int = 1000,
    sparsity: float = 0.95,
    random_seed: int = 42
) -> Dict[str, scipy.sparse.csr_matrix]:
    """
    Create sample TF-IDF features for testing.
    
    Args:
        n_samples: Number of samples per split
        vocab_size: Vocabulary size
        sparsity: Sparsity level (fraction of zeros)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train/val/test sparse matrices
    """
    np.random.seed(random_seed)
    
    features = {}
    splits = {
        'train': int(n_samples * 0.7),
        'val': int(n_samples * 0.15),
        'test': int(n_samples * 0.15)
    }
    
    for split, n_split_samples in splits.items():
        # Create sparse matrix
        n_nonzero = int(n_split_samples * vocab_size * (1 - sparsity))
        
        # Random indices for non-zero elements
        row_indices = np.random.randint(0, n_split_samples, n_nonzero)
        col_indices = np.random.randint(0, vocab_size, n_nonzero)
        
        # Random TF-IDF values (log-normal distribution is realistic)
        values = np.random.lognormal(mean=-1, sigma=1, size=n_nonzero)
        
        # Create sparse matrix
        matrix = scipy.sparse.csr_matrix(
            (values, (row_indices, col_indices)), 
            shape=(n_split_samples, vocab_size)
        )
        
        # Ensure each row has at least one non-zero element
        for i in range(n_split_samples):
            if matrix[i].nnz == 0:
                col = np.random.randint(0, vocab_size)
                matrix[i, col] = np.random.lognormal(-1, 1)
        
        features[split] = matrix
        logger.info(f"Created {split} TF-IDF features: {matrix.shape}, sparsity: {1 - matrix.nnz / np.prod(matrix.shape):.2%}")
    
    return features


def generate_sample_labels(
    n_samples: int,
    task_type: str = 'classification',
    random_seed: int = 42
) -> np.ndarray:
    """
    Generate sample labels for testing.
    
    Args:
        n_samples: Number of labels to generate
        task_type: 'classification' or 'regression'
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of labels
    """
    np.random.seed(random_seed)
    
    if task_type == 'classification':
        # Binary classification with slight imbalance
        labels = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    else:  # regression
        # Continuous values between 0 and 1
        labels = np.random.beta(2, 2, size=n_samples)  # Beta distribution for bounded values
    
    return labels


def create_multilingual_sample_dataset(
    samples_per_language: int = 50,
    languages: Optional[List[str]] = None,
    include_controls: bool = False,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Create a comprehensive multilingual sample dataset.
    
    Args:
        samples_per_language: Number of samples per language
        languages: List of languages to include
        include_controls: Whether to include control experiments data
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing multilingual dataset
    """
    if languages is None:
        languages = ['en', 'ru', 'ar', 'fi']
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    dataset = {
        'text': [],
        'language': [],
        'question_type': [],
        'lang_norm_complexity_score': []
    }
    
    # Add all submetrics
    for submetric in SUBMETRIC_VALUES.keys():
        dataset[submetric] = []
    
    # Generate samples for each language
    for lang in languages:
        lang_texts = SAMPLE_TEXTS.get(lang, SAMPLE_TEXTS['en'])
        lang_question_types = QUESTION_PATTERNS.get(lang, QUESTION_PATTERNS['en'])
        lang_complexity = COMPLEXITY_SCORES.get(lang, COMPLEXITY_SCORES['en'])
        
        for i in range(samples_per_language):
            text_idx = i % len(lang_texts)
            
            # Add some variation to texts
            base_text = lang_texts[text_idx]
            if random.random() < 0.2:  # 20% chance to add variation
                variations = [" Indeed.", " Really?", " Perhaps.", " Certainly.", " Maybe."]
                base_text += random.choice(variations)
            
            dataset['text'].append(base_text)
            dataset['language'].append(lang)
            dataset['question_type'].append(lang_question_types[text_idx])
            
            # Complexity with language-specific bias
            base_complexity = lang_complexity[text_idx]
            lang_bias = {'en': 0.0, 'ru': 0.1, 'ar': 0.15, 'fi': 0.05}.get(lang, 0.0)
            noise = np.random.normal(0, 0.08)
            complexity = max(0, min(1, base_complexity + lang_bias + noise))
            dataset['lang_norm_complexity_score'].append(complexity)
            
            # Submetrics with correlations
            for submetric in SUBMETRIC_VALUES.keys():
                lang_submetric = SUBMETRIC_VALUES[submetric].get(lang, SUBMETRIC_VALUES[submetric]['en'])
                base_value = lang_submetric[text_idx]
                
                # Add correlation with complexity
                correlation_strength = 0.3
                correlated_noise = correlation_strength * (complexity - 0.5)
                independent_noise = np.random.normal(0, 0.05)
                
                submetric_value = max(0, min(1, base_value + correlated_noise + independent_noise))
                dataset[submetric].append(submetric_value)
    
    # Add control experiments data if requested
    if include_controls:
        dataset['control_question_type_1'] = [random.randint(0, 1) for _ in range(len(dataset['text']))]
        dataset['control_question_type_2'] = [random.randint(0, 1) for _ in range(len(dataset['text']))]
        dataset['control_complexity_1'] = [random.random() for _ in range(len(dataset['text']))]
        dataset['control_complexity_2'] = [random.random() for _ in range(len(dataset['text']))]
        
        # Control submetrics
        for submetric in SUBMETRIC_VALUES.keys():
            for control_idx in [1, 2]:
                control_name = f"control_{submetric}_{control_idx}"
                dataset[control_name] = [random.random() for _ in range(len(dataset['text']))]
    
    logger.info(f"Created multilingual dataset with {len(dataset['text'])} samples across {len(languages)} languages")
    return dataset


def save_sample_dataset_splits(
    dataset: Dict[str, Any],
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Save sample dataset as train/validation/test splits.
    
    Args:
        dataset: Dataset dictionary
        output_dir: Output directory
        train_ratio: Fraction for training split
        val_ratio: Fraction for validation split
        test_ratio: Fraction for test split
        random_seed: Random seed for shuffling
        
    Returns:
        Dictionary mapping split names to file paths
    """
    np.random.seed(random_seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_samples = len(dataset['text'])
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    splits = {
        'train': indices[:n_train],
        'validation': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:]
    }
    
    file_paths = {}
    
    for split_name, split_indices in splits.items():
        split_data = {}
        for key, values in dataset.items():
            split_data[key] = [values[i] for i in split_indices]
        
        # Save as JSON
        file_path = output_path / f"{split_name}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        file_paths[split_name] = str(file_path)
        logger.info(f"Saved {split_name} split with {len(split_indices)} samples to {file_path}")
    
    return file_paths


def create_realistic_tfidf_vocabulary(vocab_size: int = 1000, random_seed: int = 42) -> List[str]:
    """
    Create a realistic TF-IDF vocabulary for testing.
    
    Args:
        vocab_size: Size of vocabulary to create
        random_seed: Random seed for reproducibility
        
    Returns:
        List of vocabulary terms
    """
    random.seed(random_seed)
    
    # Base vocabulary with common words, question words, and domain terms
    base_vocab = [
        # Common words
        'the', 'is', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'this', 'that', 'these', 'those', 'a', 'an', 'some', 'any', 'all', 'no', 'not',
        
        # Question words
        'what', 'when', 'where', 'why', 'how', 'who', 'which', 'whose', 'whom',
        
        # Domain-specific terms
        'question', 'answer', 'problem', 'solution', 'method', 'approach', 'technique',
        'analysis', 'result', 'conclusion', 'hypothesis', 'experiment', 'data', 'model',
        'algorithm', 'function', 'variable', 'parameter', 'value', 'system', 'process',
        'structure', 'pattern', 'relationship', 'correlation', 'significance', 'evidence',
        
        # Action words
        'analyze', 'evaluate', 'compare', 'contrast', 'describe', 'explain', 'identify',
        'classify', 'categorize', 'determine', 'calculate', 'measure', 'estimate', 'predict',
        
        # Complexity indicators
        'complex', 'simple', 'difficult', 'easy', 'complicated', 'straightforward',
        'advanced', 'basic', 'fundamental', 'sophisticated', 'elementary', 'intricate'
    ]
    
    # Generate additional vocabulary
    vocab = base_vocab.copy()
    
    # Add numbered variations
    for i in range(100):
        vocab.extend([f'term_{i}', f'concept_{i}', f'item_{i}', f'factor_{i}'])
    
    # Add prefixes and suffixes
    prefixes = ['pre', 'post', 'sub', 'super', 'anti', 'pro', 'multi', 'inter', 'intra', 'trans']
    suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'able', 'ible']
    roots = ['process', 'analyze', 'system', 'method', 'concept', 'theory', 'practice']
    
    for prefix in prefixes:
        for root in roots:
            vocab.append(f'{prefix}{root}')
    
    for root in roots:
        for suffix in suffixes:
            vocab.append(f'{root}{suffix}')
    
    # Add language-specific terms
    multilingual_terms = {
        'en': ['english', 'language', 'grammar', 'syntax', 'semantic'],
        'ru': ['russian', 'cyrillic', 'slavic', 'morphology', 'case'],
        'ar': ['arabic', 'semitic', 'script', 'right_to_left', 'root'],
        'fi': ['finnish', 'uralic', 'agglutinative', 'case_system', 'vowel_harmony'],
        'ja': ['japanese', 'kanji', 'hiragana', 'katakana', 'honorific'],
        'ko': ['korean', 'hangul', 'agglutination', 'honorific_system', 'vowel'],
        'id': ['indonesian', 'malay', 'austronesian', 'prefix', 'suffix']
    }
    
    for lang_terms in multilingual_terms.values():
        vocab.extend(lang_terms)
    
    # Ensure uniqueness and take only required number
    vocab = list(set(vocab))
    
    if len(vocab) < vocab_size:
        # Generate additional generic terms
        for i in range(vocab_size - len(vocab)):
            vocab.append(f'vocab_term_{i:04d}')
    
    # Shuffle and take required size
    random.shuffle(vocab)
    return vocab[:vocab_size]


def create_sample_language_distribution(
    total_samples: int,
    languages: List[str],
    distribution_type: str = 'balanced',
    random_seed: int = 42
) -> Dict[str, int]:
    """
    Create a sample distribution of languages.
    
    Args:
        total_samples: Total number of samples
        languages: List of languages
        distribution_type: 'balanced', 'imbalanced', or 'realistic'
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping languages to sample counts
    """
    np.random.seed(random_seed)
    
    if distribution_type == 'balanced':
        # Equal distribution
        samples_per_lang = total_samples // len(languages)
        distribution = {lang: samples_per_lang for lang in languages}
        
        # Handle remainder
        remainder = total_samples % len(languages)
        for i in range(remainder):
            distribution[languages[i]] += 1
    
    elif distribution_type == 'imbalanced':
        # Imbalanced distribution (some languages dominate)
        weights = np.random.exponential(scale=1.0, size=len(languages))
        weights = weights / weights.sum()
        
        distribution = {}
        assigned = 0
        for i, lang in enumerate(languages[:-1]):
            count = int(total_samples * weights[i])
            distribution[lang] = count
            assigned += count
        
        # Last language gets remainder
        distribution[languages[-1]] = total_samples - assigned
    
    elif distribution_type == 'realistic':
        # Realistic distribution based on language usage
        realistic_weights = {
            'en': 0.4,
            'ru': 0.15,
            'ar': 0.15,
            'fi': 0.1,
            'id': 0.1,
            'ja': 0.05,
            'ko': 0.05
        }
        
        # Normalize weights for selected languages
        selected_weights = [realistic_weights.get(lang, 1.0) for lang in languages]
        selected_weights = np.array(selected_weights)
        selected_weights = selected_weights / selected_weights.sum()
        
        distribution = {}
        assigned = 0
        for i, lang in enumerate(languages[:-1]):
            count = int(total_samples * selected_weights[i])
            distribution[lang] = count
            assigned += count
        
        distribution[languages[-1]] = total_samples - assigned
    
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")
    
    logger.info(f"Created {distribution_type} language distribution: {distribution}")
    return distribution


def validate_sample_data(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate sample dataset for consistency and correctness.
    
    Args:
        dataset: Dataset dictionary to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    try:
        # Check basic structure
        required_fields = ['text', 'language', 'question_type', 'lang_norm_complexity_score']
        for field in required_fields:
            if field not in dataset:
                validation_results['errors'].append(f"Missing required field: {field}")
                validation_results['is_valid'] = False
        
        if not validation_results['is_valid']:
            return validation_results
        
        # Check data consistency
        n_samples = len(dataset['text'])
        for field, values in dataset.items():
            if len(values) != n_samples:
                validation_results['errors'].append(f"Field {field} has {len(values)} samples, expected {n_samples}")
                validation_results['is_valid'] = False
        
        # Check data types and ranges
        for i, qt in enumerate(dataset['question_type']):
            if qt not in [0, 1]:
                validation_results['errors'].append(f"Invalid question_type at index {i}: {qt}")
        
        for i, comp in enumerate(dataset['lang_norm_complexity_score']):
            if not (0 <= comp <= 1):
                validation_results['warnings'].append(f"Complexity score out of range [0,1] at index {i}: {comp}")
        
        # Check submetrics if present
        for submetric in SUBMETRIC_VALUES.keys():
            if submetric in dataset:
                for i, value in enumerate(dataset[submetric]):
                    if not (0 <= value <= 1):
                        validation_results['warnings'].append(f"{submetric} out of range [0,1] at index {i}: {value}")
        
        # Generate statistics
        validation_results['statistics'] = {
            'n_samples': n_samples,
            'n_languages': len(set(dataset['language'])),
            'languages': list(set(dataset['language'])),
            'question_type_distribution': {
                0: dataset['question_type'].count(0),
                1: dataset['question_type'].count(1)
            },
            'complexity_stats': {
                'mean': np.mean(dataset['lang_norm_complexity_score']),
                'std': np.std(dataset['lang_norm_complexity_score']),
                'min': np.min(dataset['lang_norm_complexity_score']),
                'max': np.max(dataset['lang_norm_complexity_score'])
            }
        }
        
        # Add language distribution
        lang_dist = {}
        for lang in dataset['language']:
            lang_dist[lang] = lang_dist.get(lang, 0) + 1
        validation_results['statistics']['language_distribution'] = lang_dist
        
    except Exception as e:
        validation_results['errors'].append(f"Validation failed with exception: {str(e)}")
        validation_results['is_valid'] = False
    
    return validation_results


if __name__ == "__main__":
    # Example usage
    print("Creating sample datasets for testing...")
    
    # Create a basic sample dataset
    dataset = create_sample_dataset(n_samples=100, n_languages=3)
    print(f"Created dataset with {len(dataset['text'])} samples")
    
    # Validate the dataset
    validation = validate_sample_data(dataset)
    print(f"Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    print(f"Statistics: {validation['statistics']}")
    
    # Create TF-IDF features
    tfidf_features = create_sample_tfidf_features(n_samples=100, vocab_size=500)
    print(f"Created TF-IDF features with shapes: {[(k, v.shape) for k, v in tfidf_features.items()]}")
    
    # Create vocabulary
    vocab = create_realistic_tfidf_vocabulary(vocab_size=200)
    print(f"Created vocabulary with {len(vocab)} terms")
    print(f"Sample terms: {vocab[:10]}")