# tests/fixtures/mock_datasets.py
"""
Mock dataset utilities for testing TF-IDF integration.
Provides mock implementations of HuggingFace datasets and other external dependencies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MockDatasetSplit:
    """Mock implementation of a HuggingFace dataset split."""
    
    def __init__(self, data: Dict[str, List[Any]], split_name: str = "train"):
        self.data = data
        self.split_name = split_name
        self._length = len(next(iter(data.values()))) if data else 0
        
        # Validate data consistency
        for key, values in data.items():
            if len(values) != self._length:
                raise ValueError(f"Inconsistent data length for key {key}: {len(values)} vs {self._length}")
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {key: values[idx] for key, values in self.data.items()}
        elif isinstance(idx, slice):
            return MockDatasetSplit({
                key: values[idx] for key, values in self.data.items()
            }, self.split_name)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.data)
    
    def select(self, indices: List[int]) -> 'MockDatasetSplit':
        """Select subset of data by indices."""
        new_data = {}
        for key, values in self.data.items():
            new_data[key] = [values[i] for i in indices]
        return MockDatasetSplit(new_data, self.split_name)
    
    def filter(self, function) -> 'MockDatasetSplit':
        """Filter data using a function."""
        indices = []
        for i in range(len(self)):
            if function(self[i]):
                indices.append(i)
        return self.select(indices)
    
    def map(self, function, **kwargs) -> 'MockDatasetSplit':
        """Apply function to each example."""
        new_data = {key: [] for key in self.data.keys()}
        
        for i in range(len(self)):
            example = self[i]
            processed = function(example)
            
            for key, value in processed.items():
                if key not in new_data:
                    new_data[key] = []
                new_data[key].append(value)
        
        return MockDatasetSplit(new_data, self.split_name)


class MockDataset:
    """Mock implementation of a HuggingFace dataset."""
    
    def __init__(self, splits: Dict[str, MockDatasetSplit]):
        self.splits = splits
    
    def __getitem__(self, split_name: str) -> MockDatasetSplit:
        if split_name not in self.splits:
            raise KeyError(f"Split {split_name} not found")
        return self.splits[split_name]
    
    def keys(self):
        return self.splits.keys()


class MockDatasetLoader:
    """Mock loader for datasets with configurable responses."""
    
    def __init__(self):
        self.datasets = {}
        self._configure_default_datasets()
    
    def _configure_default_datasets(self):
        """Configure default mock datasets."""
        # Default question type dataset
        self.add_dataset(
            "rokokot/question-type-and-complexity",
            config_name="base",
            data={
                'text': [
                    "What is the capital of France?",
                    "Paris is the capital of France.",
                    "How do you solve this problem?",
                    "This is a statement.",
                    "Where can I find the answer?",
                    "The answer is here.",
                    "Why is this important?",
                    "This is important because...",
                    "Can you help me?",
                    "I can help you."
                ] * 10,  # 100 samples total
                'language': ['en'] * 100,
                'question_type': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10,
                'lang_norm_complexity_score': [0.6, 0.4, 0.7, 0.3, 0.5, 0.2, 0.8, 0.1, 0.6, 0.3] * 10,
                'avg_links_len': [0.4, 0.2, 0.5, 0.1, 0.3, 0.1, 0.6, 0.0, 0.4, 0.2] * 10,
                'avg_max_depth': [0.5, 0.3, 0.6, 0.2, 0.4, 0.1, 0.7, 0.1, 0.5, 0.3] * 10,
                'avg_subordinate_chain_len': [0.3, 0.1, 0.4, 0.0, 0.2, 0.0, 0.5, 0.0, 0.3, 0.1] * 10,
                'avg_verb_edges': [0.6, 0.4, 0.7, 0.3, 0.5, 0.2, 0.8, 0.1, 0.6, 0.4] * 10,
                'lexical_density': [0.7, 0.5, 0.8, 0.4, 0.6, 0.3, 0.9, 0.2, 0.7, 0.5] * 10,
                'n_tokens': [0.4, 0.2, 0.5, 0.1, 0.3, 0.1, 0.6, 0.0, 0.4, 0.2] * 10
            }
        )
        
        # Multilingual dataset
        self.add_multilingual_dataset()
        
        # Control experiments dataset
        self.add_control_dataset()
    
    def add_dataset(
        self, 
        dataset_name: str, 
        config_name: str = "base",
        data: Optional[Dict[str, List[Any]]] = None,
        splits: Optional[Dict[str, Dict[str, List[Any]]]] = None
    ):
        """Add a mock dataset."""
        if splits is None:
            # Create train/val/test splits from single data
            if data is None:
                raise ValueError("Must provide either data or splits")
            
            n_total = len(next(iter(data.values())))
            n_train = int(0.7 * n_total)
            n_val = int(0.15 * n_total)
            
            splits = {}
            for split_name, (start, end) in [
                ('train', (0, n_train)),
                ('validation', (n_train, n_train + n_val)),
                ('test', (n_train + n_val, n_total))
            ]:
                splits[split_name] = {
                    key: values[start:end] for key, values in data.items()
                }
        
        dataset_splits = {}
        for split_name, split_data in splits.items():
            dataset_splits[split_name] = MockDatasetSplit(split_data, split_name)
        
        key = f"{dataset_name}_{config_name}"
        self.datasets[key] = MockDataset(dataset_splits)
        
        logger.info(f"Added mock dataset: {key} with splits: {list(dataset_splits.keys())}")
    
    def add_multilingual_dataset(self):
        """Add a multilingual mock dataset."""
        languages = ['en', 'ru', 'ar', 'fi', 'id', 'ja', 'ko']
        
        # Create text samples for each language
        text_templates = {
            'en': ["What is {}?", "This is {}.", "How to {}?", "The {} is here."],
            'ru': ["Что такое {}?", "Это {}.", "Как {}?", "{} здесь."],
            'ar': ["ما هو {}؟", "هذا {}.", "كيف {}؟", "{} هنا."],
            'fi': ["Mikä on {}?", "Tämä on {}.", "Miten {}?", "{} on täällä."],
            'en': ["What is {}?", "This is {}.", "How to {}?", "The {} is here."],
            'ru': ["Что такое {}?", "Это {}.", "Как {}?", "{} здесь."],
            'ar': ["ما هو {}؟", "هذا {}.", "كيف {}؟", "{} هنا."],
            'fi': ["Mikä on {}?", "Tämä on {}.", "Miten {}?", "{} on täällä."],
            'id': ["Apa itu {}?", "Ini adalah {}.", "Bagaimana {}?", "{} ada di sini."],
            'ja': ["{}とは何ですか？", "これは{}です。", "{}の方法は？", "{}はここにあります。"],
            'ko': ["{}는 무엇입니까?", "이것은 {}입니다.", "{}하는 방법은?", "{}가 여기 있습니다."]
        }
        
        topics = ["science", "technology", "language", "culture", "history", "art", "music", "literature"]
        
        multilingual_data = {
            'text': [],
            'language': [],
            'question_type': [],
            'lang_norm_complexity_score': [],
            'avg_links_len': [],
            'avg_max_depth': [],
            'avg_subordinate_chain_len': [],
            'avg_verb_edges': [],
            'lexical_density': [],
            'n_tokens': []
        }
        
        samples_per_lang = 20
        for lang in languages:
            templates = text_templates.get(lang, text_templates['en'])
            
            for i in range(samples_per_lang):
                template_idx = i % len(templates)
                topic_idx = i % len(topics)
                
                text = templates[template_idx].format(topics[topic_idx])
                multilingual_data['text'].append(text)
                multilingual_data['language'].append(lang)
                
                # Question type based on template (questions vs statements)
                is_question = 1 if template_idx in [0, 2] else 0
                multilingual_data['question_type'].append(is_question)
                
                # Language-specific complexity bias
                lang_complexity_bias = {
                    'en': 0.0, 'ru': 0.1, 'ar': 0.15, 'fi': 0.05,
                    'id': 0.0, 'ja': 0.2, 'ko': 0.1
                }.get(lang, 0.0)
                
                base_complexity = 0.5 + (i % 5) * 0.1  # Varies from 0.5 to 0.9
                complexity = min(1.0, base_complexity + lang_complexity_bias)
                multilingual_data['lang_norm_complexity_score'].append(complexity)
                
                # Generate correlated submetrics
                for submetric in ['avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len', 
                                'avg_verb_edges', 'lexical_density', 'n_tokens']:
                    # Base value with some correlation to complexity
                    base_value = 0.3 + 0.4 * complexity + np.random.normal(0, 0.1)
                    submetric_value = max(0, min(1, base_value))
                    multilingual_data[submetric].append(submetric_value)
        
        self.add_dataset(
            "multilingual/question-complexity",
            config_name="base",
            data=multilingual_data
        )
    
    def add_control_dataset(self):
        """Add dataset with control experiment configurations."""
        base_data = {
            'text': [f"Control text sample {i}" for i in range(100)],
            'language': ['en'] * 100,
            'question_type': [i % 2 for i in range(100)],
            'lang_norm_complexity_score': [0.1 + (i % 10) * 0.1 for i in range(100)]
        }
        
        # Add submetrics
        submetrics = ['avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len', 
                     'avg_verb_edges', 'lexical_density', 'n_tokens']
        for submetric in submetrics:
            base_data[submetric] = [0.1 + (i % 8) * 0.1 for i in range(100)]
        
        # Add control versions (randomized)
        for control_idx in [1, 2, 3]:
            control_data = base_data.copy()
            
            # Randomize question types for controls
            np.random.seed(42 + control_idx)
            control_data['question_type'] = np.random.randint(0, 2, 100).tolist()
            control_data['lang_norm_complexity_score'] = np.random.random(100).tolist()
            
            for submetric in submetrics:
                control_data[submetric] = np.random.random(100).tolist()
            
            self.add_dataset(
                "rokokot/question-type-and-complexity",
                config_name=f"control_question_type_seed{control_idx}",
                data=control_data
            )
            
            self.add_dataset(
                "rokokot/question-type-and-complexity", 
                config_name=f"control_complexity_seed{control_idx}",
                data=control_data
            )
            
            # Add submetric controls
            for submetric in submetrics:
                self.add_dataset(
                    "rokokot/question-type-and-complexity",
                    config_name=f"control_{submetric}_seed{control_idx}",
                    data=control_data
                )
    
    def get_dataset(self, dataset_name: str, config_name: str = "base") -> MockDataset:
        """Get a mock dataset."""
        key = f"{dataset_name}_{config_name}"
        if key not in self.datasets:
            raise ValueError(f"Dataset not found: {key}")
        return self.datasets[key]
    
    def load_dataset(self, dataset_name: str, name: str = "base", split: Optional[str] = None, **kwargs):
        """Mock implementation of datasets.load_dataset."""
        dataset = self.get_dataset(dataset_name, name)
        
        if split is None:
            return dataset
        else:
            return dataset[split]


class MockTransformersComponents:
    """Mock implementations of transformers components."""
    
    @staticmethod
    def create_mock_tokenizer(vocab_size: int = 30522, model_name: str = "mock-tokenizer"):
        """Create a mock tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = vocab_size
        mock_tokenizer.model_max_length = 512
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.unk_token_id = 1
        mock_tokenizer.cls_token_id = 2
        mock_tokenizer.sep_token_id = 3
        
        # Mock tokenization methods
        def mock_tokenize(text):
            # Simple word-based tokenization for testing
            tokens = text.lower().split()
            return [f"##{token}" if i > 0 else token for i, token in enumerate(tokens)]
        
        def mock_encode(text, max_length=512, padding=False, truncation=True, return_tensors=None):
            tokens = mock_tokenize(text)
            # Convert to fake token IDs
            token_ids = [hash(token) % vocab_size for token in tokens[:max_length]]
            
            if padding and len(token_ids) < max_length:
                token_ids.extend([0] * (max_length - len(token_ids)))
            
            if return_tensors == 'pt':
                import torch
                return torch.tensor([token_ids])
            return token_ids
        
        mock_tokenizer.tokenize = mock_tokenize
        mock_tokenizer.encode = mock_encode
        mock_tokenizer.__len__ = lambda: vocab_size
        
        return mock_tokenizer
    
    @staticmethod
    def create_mock_model(hidden_size: int = 768, num_layers: int = 12):
        """Create a mock transformer model."""
        mock_model = Mock()
        
        # Mock config
        mock_config = Mock()
        mock_config.hidden_size = hidden_size
        mock_config.num_hidden_layers = num_layers
        mock_config.vocab_size = 30522
        mock_model.config = mock_config
        
        # Mock forward pass
        def mock_forward(input_ids, attention_mask=None, **kwargs):
            import torch
            batch_size, seq_len = input_ids.shape
            
            # Create mock outputs
            last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
            
            # Mock outputs object
            outputs = Mock()
            outputs.last_hidden_state = last_hidden_state
            outputs.hidden_states = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)]
            
            return outputs
        
        mock_model.forward = mock_forward
        mock_model.__call__ = mock_forward
        
        # Mock parameter methods
        def mock_parameters():
            import torch
            # Return some dummy parameters
            return [torch.randn(100, 100, requires_grad=True) for _ in range(10)]
        
        mock_model.parameters = mock_parameters
        mock_model.named_parameters = lambda: [('param_0', p) for p in mock_parameters()]
        
        return mock_model


class MockHuggingFaceDatasets:
    """Comprehensive mock for HuggingFace datasets."""
    
    def __init__(self):
        self.loader = MockDatasetLoader()
    
    def load_dataset(self, *args, **kwargs):
        """Mock load_dataset function."""
        if len(args) >= 1:
            dataset_name = args[0]
            config_name = kwargs.get('name', kwargs.get('config_name', 'base'))
            split = kwargs.get('split', None)
            
            try:
                return self.loader.load_dataset(dataset_name, config_name, split)
            except ValueError:
                # If dataset not found, create a default one
                logger.warning(f"Mock dataset not found: {dataset_name}_{config_name}, creating default")
                return self._create_default_dataset(split)
        
        return self._create_default_dataset(kwargs.get('split', None))
    
    def _create_default_dataset(self, split=None):
        """Create a default dataset when specific one is not found."""
        default_data = {
            'text': [f"Default text {i}" for i in range(50)],
            'language': ['en'] * 50,
            'question_type': [i % 2 for i in range(50)],
            'lang_norm_complexity_score': [(i % 10) * 0.1 for i in range(50)]
        }
        
        # Add submetrics
        submetrics = ['avg_links_len', 'avg_max_depth', 'avg_subordinate_chain_len', 
                     'avg_verb_edges', 'lexical_density', 'n_tokens']
        for submetric in submetrics:
            default_data[submetric] = [(i % 8) * 0.125 for i in range(50)]
        
        if split is None:
            # Return all splits
            n_total = 50
            n_train = 35
            n_val = 8
            
            splits = {}
            for split_name, (start, end) in [
                ('train', (0, n_train)),
                ('validation', (n_train, n_train + n_val)),
                ('test', (n_train + n_val, n_total))
            ]:
                split_data = {key: values[start:end] for key, values in default_data.items()}
                splits[split_name] = MockDatasetSplit(split_data, split_name)
            
            return MockDataset(splits)
        else:
            # Return specific split
            return MockDatasetSplit(default_data, split)


class MockFeatureFiles:
    """Mock TF-IDF feature files for testing."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.feature_files = {}
    
    def create_mock_tfidf_files(
        self,
        n_samples: Dict[str, int] = None,
        vocab_size: int = 100,
        file_format: str = 'sparse'
    ):
        """Create mock TF-IDF feature files."""
        if n_samples is None:
            n_samples = {'train': 70, 'val': 15, 'test': 15}
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata
        metadata = {
            'vocab_size': vocab_size,
            'model_name': 'mock-tfidf-model',
            'max_features': vocab_size,
            'generation_info': {
                'actual_features': vocab_size,
                'sparsity': {'train': 0.9, 'val': 0.9, 'test': 0.9}
            }
        }
        
        with open(self.temp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Create feature files
        import scipy.sparse as sparse
        
        for split, n_split_samples in n_samples.items():
            # Create sparse matrix
            n_nonzero = int(n_split_samples * vocab_size * 0.1)  # 10% non-zero
            row_indices = np.random.randint(0, n_split_samples, n_nonzero)
            col_indices = np.random.randint(0, vocab_size, n_nonzero)
            values = np.random.exponential(scale=0.5, size=n_nonzero)
            
            matrix = sparse.csr_matrix(
                (values, (row_indices, col_indices)),
                shape=(n_split_samples, vocab_size)
            )
            
            if file_format == 'sparse':
                # Save as scipy sparse matrix
                sparse.save_npz(self.temp_dir / f"X_{split}_sparse.npz", matrix)
                
            elif file_format == 'dense':
                # Save as numpy array
                np.save(self.temp_dir / f"X_{split}.npy", matrix.toarray())
                
            elif file_format == 'pickle':
                # Save as pickle (legacy format)
                split_name = 'dev' if split == 'val' else split
                import pickle
                with open(self.temp_dir / f"tfidf_vectors_{split_name}.pkl", 'wb') as f:
                    pickle.dump(matrix, f)
            
            self.feature_files[split] = matrix
        
        # Create language info
        language_info = {}
        for split, n_split_samples in n_samples.items():
            language_info[split] = ['en'] * n_split_samples
        
        with open(self.temp_dir / "language_info.json", 'w') as f:
            json.dump(language_info, f)
        
        return self.feature_files


class MockSparseMatrix:
    """Mock sparse matrix for testing without scipy dependency."""
    
    def __init__(self, shape: Tuple[int, int], nnz: int = None):
        self.shape = shape
        self.nnz = nnz or (shape[0] * shape[1] // 10)  # 10% non-zero
        self._data = np.random.random(self.nnz)
        
    def toarray(self):
        """Convert to dense array."""
        array = np.zeros(self.shape)
        # Add some random non-zero elements
        for _ in range(self.nnz):
            i = np.random.randint(0, self.shape[0])
            j = np.random.randint(0, self.shape[1])
            array[i, j] = np.random.random()
        return array
    
    def __getitem__(self, key):
        """Mock indexing."""
        if isinstance(key, tuple) and len(key) == 2:
            # Return a smaller mock matrix for slicing
            return MockSparseMatrix((1, self.shape[1]))
        return self


def create_mock_environment():
    """Create a complete mock environment for testing."""
    mocks = {}
    
    # Mock HuggingFace datasets
    mock_hf_datasets = MockHuggingFaceDatasets()
    mocks['datasets'] = patch('src.data.datasets.load_dataset', side_effect=mock_hf_datasets.load_dataset)
    
    # Mock transformers components
    mock_tokenizer = MockTransformersComponents.create_mock_tokenizer()
    mock_model = MockTransformersComponents.create_mock_model()
    
    mocks['tokenizer'] = patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer)
    mocks['model'] = patch('transformers.AutoModel.from_pretrained', return_value=mock_model)
    
    # Mock wandb
    mock_wandb = Mock()
    mock_wandb.init.return_value = Mock()
    mock_wandb.run = Mock()
    mocks['wandb'] = patch('wandb', mock_wandb)
    
    return mocks


def apply_mock_patches(mocks: Dict[str, Any]):
    """Apply all mock patches."""
    started_mocks = {}
    for name, mock_patch in mocks.items():
        started_mocks[name] = mock_patch.start()
    return started_mocks


def stop_mock_patches(mocks: Dict[str, Any]):
    """Stop all mock patches."""
    for mock_patch in mocks.values():
        mock_patch.stop()


class MockExperimentEnvironment:
    """Complete mock environment for experiment testing."""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = Path(temp_dir)
        self.dataset_loader = MockDatasetLoader()
        self.feature_mocker = MockFeatureFiles(str(self.temp_dir / "features"))
        self.mocks = {}
        self.started_mocks = {}
    
    def setup(self):
        """Set up the mock environment."""
        # Create mock TF-IDF features
        self.feature_mocker.create_mock_tfidf_files()
        
        # Set up mocks
        self.mocks = create_mock_environment()
        self.mocks['datasets'] = patch('src.data.datasets.load_dataset', 
                                     side_effect=self.dataset_loader.load_dataset)
        
        # Start all mocks
        self.started_mocks = apply_mock_patches(self.mocks)
        
        return self
    
    def teardown(self):
        """Tear down the mock environment."""
        stop_mock_patches(self.mocks)
    
    def __enter__(self):
        return self.setup()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
    
    def get_features_dir(self) -> str:
        """Get the features directory path."""
        return str(self.feature_mocker.temp_dir)
    
    def add_custom_dataset(self, dataset_name: str, config_name: str, data: Dict[str, List[Any]]):
        """Add a custom dataset to the mock environment."""
        self.dataset_loader.add_dataset(dataset_name, config_name, data)


# Utility functions for test setup
def create_test_environment_with_mocks(temp_dir: str) -> MockExperimentEnvironment:
    """Create a complete test environment with mocks."""
    return MockExperimentEnvironment(temp_dir)


def patch_external_dependencies():
    """Patch external dependencies for testing."""
    patches = []
    
    # Patch datasets
    mock_hf = MockHuggingFaceDatasets()
    patches.append(patch('datasets.load_dataset', side_effect=mock_hf.load_dataset))
    
    # Patch transformers
    patches.append(patch('transformers.AutoTokenizer.from_pretrained', 
                        return_value=MockTransformersComponents.create_mock_tokenizer()))
    patches.append(patch('transformers.AutoModel.from_pretrained',
                        return_value=MockTransformersComponents.create_mock_model()))
    
    # Patch wandb
    mock_wandb = Mock()
    patches.append(patch('wandb.init', return_value=Mock()))
    patches.append(patch('wandb.run', Mock()))
    
    return patches


if __name__ == "__main__":
    # Example usage
    print("Creating mock dataset environment...")
    
    # Create mock dataset loader
    loader = MockDatasetLoader()
    
    # Test loading a dataset
    dataset = loader.get_dataset("rokokot/question-type-and-complexity", "base")
    print(f"Mock dataset has splits: {list(dataset.keys())}")
    
    train_split = dataset['train']
    print(f"Train split has {len(train_split)} samples")
    print(f"Sample data: {train_split[0]}")
    
    # Test creating mock TF-IDF files
    with tempfile.TemporaryDirectory() as temp_dir:
        feature_mocker = MockFeatureFiles(temp_dir)
        features = feature_mocker.create_mock_tfidf_files()
        
        print(f"Created mock TF-IDF features:")
        for split, matrix in features.items():
            print(f"  {split}: {matrix.shape}")
        
        # Test complete mock environment
        with MockExperimentEnvironment(temp_dir) as env:
            print(f"Mock environment set up in: {env.get_features_dir()}")
            
            # Add custom dataset
            custom_data = {
                'text': ['Custom text 1', 'Custom text 2'],
                'language': ['en', 'en'],
                'question_type': [1, 0]
            }
            env.add_custom_dataset("custom/dataset", "test", custom_data)
            
            print("Mock environment ready for testing")