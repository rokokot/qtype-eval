# src/models/xlm_roberta_tfidf_configs.py
"""
Model configurations and parameters for XLM-RoBERTa-consistent TF-IDF experiments.
Updated to maintain tokenizer consistency with neural experiments.
"""

from typing import Dict, Any

# Model parameters for XLM-RoBERTa-consistent TF-IDF experiments
XLM_ROBERTA_TFIDF_PARAMS = {
    "dummy": {
        "classification": {"strategy": "most_frequent"},
        "regression": {"strategy": "mean"}
    },
    "logistic": {
        "classification": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "liblinear",  # Good for TF-IDF features
            "penalty": "l2",
            "random_state": 42
        }
    },
    "ridge": {
        "regression": {
            "alpha": 1.0,
            "solver": "auto",
            "random_state": 42
        }
    },
    "xgboost": {
        "classification": {
            "n_estimators": 100,  # Reasonable for TF-IDF
            "max_depth": 6,
            "learning_rate": 0.1,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        },
        "regression": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "eval_metric": ["rmse", "mae"]
        }
    }
}

# Experiment configuration for XLM-RoBERTa-consistent experiments
XLM_ROBERTA_EXPERIMENT_CONFIG = {
    # Random seeds for reproducibility
    "random_seeds": [42],
    
    # Languages from your experiments
    "languages": ["ar", "en", "fi", "id", "ja", "ko", "ru"],
    
    # Tasks
    "tasks": ["question_type", "complexity"],
    
    # Models compatible with TF-IDF
    "models": ["dummy", "logistic", "ridge", "xgboost"],
    
    # Submetrics for complexity analysis
    "submetrics": [
        "avg_links_len",
        "avg_max_depth", 
        "avg_subordinate_chain_len",
        "avg_verb_edges",
        "lexical_density",
        "n_tokens"
    ],
    
    # Control indices
    "control_indices": [1, 2, 3],
    
    # Task type mapping
    "task_types": {
        "question_type": "classification",
        "complexity": "regression",
        "single_submetric": "regression"
    },
    
    # Tokenizer configuration (must match neural experiments)
    "tokenizer_config": {
        "model_name": "xlm-roberta-base",
        "model_max_length": 512,
        "tokenizer_class": "XLMRobertaTokenizer",
        "use_fast": True,
        "special_tokens": {
            "bos_token": "<s>",
            "cls_token": "<s>",
            "eos_token": "</s>",
            "mask_token": "<mask>",
            "pad_token": "<pad>",
            "sep_token": "</s>",
            "unk_token": "<unk>"
        }
    },
    
    # TF-IDF feature parameters
    "tfidf_config": {
        "max_features": 128000,  # Match original implementation
        "min_df": 2,
        "max_df": 0.95,
        "ngram_range": (1, 2),
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": True
    }
}

def get_xlm_roberta_model_params(model_type: str, task_type: str) -> Dict[str, Any]:
    """Get model parameters for XLM-RoBERTa-consistent TF-IDF experiments."""
    if model_type not in XLM_ROBERTA_TFIDF_PARAMS:
        raise ValueError(f"Model type '{model_type}' not found in XLM-RoBERTa TF-IDF params")
    
    model_params = XLM_ROBERTA_TFIDF_PARAMS[model_type]
    
    if task_type in model_params:
        return model_params[task_type].copy()
    
    # For models that work with both tasks, return the first available
    return next(iter(model_params.values())).copy()

def validate_xlm_roberta_compatibility(model_type: str, task_type: str) -> bool:
    """Check if model/task combination is valid for XLM-RoBERTa TF-IDF experiments."""
    if model_type == "logistic" and task_type != "classification":
        return False
    if model_type == "ridge" and task_type != "regression":
        return False
    return True

def get_xlm_roberta_experiment_matrix():
    """Generate the full experiment matrix for XLM-RoBERTa-consistent TF-IDF."""
    experiments = []
    
    config = XLM_ROBERTA_EXPERIMENT_CONFIG
    
    for lang in config["languages"]:
        for task in config["tasks"]:
            task_type = config["task_types"][task]
            
            for model in config["models"]:
                if not validate_xlm_roberta_compatibility(model, task_type):
                    continue
                
                # Main experiment
                experiments.append({
                    "language": lang,
                    "task": task,
                    "task_type": task_type,
                    "model_type": model,
                    "control_index": None,
                    "submetric": None,
                    "experiment_type": "main",
                    "model_params": get_xlm_roberta_model_params(model, task_type)
                })
                
                # Control experiments
                for control_idx in config["control_indices"]:
                    experiments.append({
                        "language": lang,
                        "task": task,
                        "task_type": task_type,
                        "model_type": model,
                        "control_index": control_idx,
                        "submetric": None,
                        "experiment_type": "control",
                        "model_params": get_xlm_roberta_model_params(model, task_type)
                    })
        
        # Submetric experiments
        for submetric in config["submetrics"]:
            for model in config["models"]:
                if not validate_xlm_roberta_compatibility(model, "regression"):
                    continue
                
                # Main submetric experiment
                experiments.append({
                    "language": lang,
                    "task": "single_submetric",
                    "task_type": "regression",
                    "model_type": model,
                    "control_index": None,
                    "submetric": submetric,
                    "experiment_type": "submetric",
                    "model_params": get_xlm_roberta_model_params(model, "regression")
                })
                
                # Control submetric experiments
                for control_idx in config["control_indices"]:
                    experiments.append({
                        "language": lang,
                        "task": "single_submetric", 
                        "task_type": "regression",
                        "model_type": model,
                        "control_index": control_idx,
                        "submetric": submetric,
                        "experiment_type": "control_submetric",
                        "model_params": get_xlm_roberta_model_params(model, "regression")
                    })
    
    return experiments

# Expected statistics for validation (updated for your dataset)
XLM_ROBERTA_STATS = {
    "tokenizer_requirements": {
        "model_name": "xlm-roberta-base",
        "tokenizer_class": "XLMRobertaTokenizer",
        "vocab_size_min": 250000,  # XLM-RoBERTa has large vocab
        "special_tokens_required": ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    },
    
    "feature_expectations": {
        "max_features": 50000,
        "sparsity_range": (0.8, 0.99),  # TF-IDF is typically very sparse
        "min_samples": {
            "train": 1000,  # Expect reasonable training size
            "val": 100,
            "test": 100
        }
    },
    
    "performance_baselines": {
        "classification": {
            "dummy_accuracy_max": 0.6,  # Dummy shouldn't be too good
            "logistic_accuracy_min": 0.5,  # Should beat random
            "xgboost_accuracy_min": 0.5
        },
        "regression": {
            "dummy_r2_max": 0.1,  # Dummy should be poor
            "ridge_r2_min": 0.0,  # Should be non-negative for real tasks
            "xgboost_r2_min": 0.0
        }
    }
}

def validate_xlm_roberta_experiment_results(results: Dict[str, Any]) -> Dict[str, bool]:
    """Validate experiment results for XLM-RoBERTa-consistent TF-IDF."""
    validation = {}
    
    # Check tokenizer consistency
    has_xlm_roberta = False
    for exp_name, result in results.items():
        if "tokenizer_info" in result:
            tokenizer_info = result["tokenizer_info"]
            if tokenizer_info.get("model_name") == "xlm-roberta-base":
                has_xlm_roberta = True
                break
    
    validation["uses_xlm_roberta"] = has_xlm_roberta
    
    # Check experiment coverage
    languages_found = set()
    tasks_found = set()
    models_found = set()
    has_controls = False
    has_submetrics = False
    
    for exp_name, result in results.items():
        if "error" in result:
            continue
            
        languages_found.add(result.get("language"))
        tasks_found.add(result.get("task"))
        models_found.add(result.get("model_type"))
        
        if result.get("control_index") is not None:
            has_controls = True
        
        if result.get("submetric") is not None:
            has_submetrics = True
    
    expected = XLM_ROBERTA_EXPERIMENT_CONFIG
    validation['languages_complete'] = set(expected["languages"]).issubset(languages_found)
    validation['tasks_complete'] = set(expected["tasks"]).issubset(tasks_found)
    validation['models_complete'] = set(expected["models"]).issubset(models_found)
    validation['has_control_experiments'] = has_controls
    validation['has_submetric_experiments'] = has_submetrics
    
    # Check performance baselines
    performance_validation = _validate_performance_baselines(results)
    validation.update(performance_validation)
    
    return validation

def _validate_performance_baselines(results: Dict[str, Any]) -> Dict[str, bool]:
    """Validate that performance metrics meet baseline expectations."""
    baselines = XLM_ROBERTA_STATS["performance_baselines"]
    validation = {}
    
    # Separate results by task type
    classification_results = []
    regression_results = []
    
    for result in results.values():
        if "error" in result:
            continue
        
        task_type = result.get("task_type")
        test_metrics = result.get("test_metrics", {})
        
        if task_type == "classification" and "accuracy" in test_metrics:
            classification_results.append({
                "model_type": result.get("model_type"),
                "accuracy": test_metrics["accuracy"],
                "is_control": result.get("control_index") is not None
            })
        elif task_type == "regression" and "r2" in test_metrics:
            regression_results.append({
                "model_type": result.get("model_type"),
                "r2": test_metrics["r2"],
                "is_control": result.get("control_index") is not None
            })
    
    # Validate classification baselines
    if classification_results:
        dummy_accuracies = [r["accuracy"] for r in classification_results 
                           if r["model_type"] == "dummy" and not r["is_control"]]
        if dummy_accuracies:
            max_dummy_acc = max(dummy_accuracies)
            validation["dummy_classification_reasonable"] = max_dummy_acc <= baselines["classification"]["dummy_accuracy_max"]
        
        logistic_accuracies = [r["accuracy"] for r in classification_results 
                              if r["model_type"] == "logistic" and not r["is_control"]]
        if logistic_accuracies:
            min_logistic_acc = min(logistic_accuracies)
            validation["logistic_classification_reasonable"] = min_logistic_acc >= baselines["classification"]["logistic_accuracy_min"]
    
    # Validate regression baselines
    if regression_results:
        dummy_r2s = [r["r2"] for r in regression_results 
                     if r["model_type"] == "dummy" and not r["is_control"]]
        if dummy_r2s:
            max_dummy_r2 = max(dummy_r2s)
            validation["dummy_regression_reasonable"] = max_dummy_r2 <= baselines["regression"]["dummy_r2_max"]
        
        ridge_r2s = [r["r2"] for r in regression_results 
                     if r["model_type"] == "ridge" and not r["is_control"]]
        if ridge_r2s:
            min_ridge_r2 = min(ridge_r2s)
            validation["ridge_regression_reasonable"] = min_ridge_r2 >= baselines["regression"]["ridge_r2_min"]
    
    return validation

# Utility functions for experiment setup
def get_default_model_params(model_type: str, task_type: str) -> Dict[str, Any]:
    """Get default parameters for different model types and tasks."""
    return get_xlm_roberta_model_params(model_type, task_type)

def get_available_models() -> list:
    """Get list of available model types."""
    return XLM_ROBERTA_EXPERIMENT_CONFIG["models"]

def get_available_languages() -> list:
    """Get list of available languages."""
    return XLM_ROBERTA_EXPERIMENT_CONFIG["languages"]

def get_available_tasks() -> list:
    """Get list of available tasks."""
    return XLM_ROBERTA_EXPERIMENT_CONFIG["tasks"]

def get_available_submetrics() -> list:
    """Get list of available submetrics."""
    return XLM_ROBERTA_EXPERIMENT_CONFIG["submetrics"]

def get_tokenizer_config() -> Dict[str, Any]:
    """Get the tokenizer configuration for validation."""
    return XLM_ROBERTA_EXPERIMENT_CONFIG["tokenizer_config"]

def get_tfidf_config() -> Dict[str, Any]:
    """Get the TF-IDF configuration."""
    return XLM_ROBERTA_EXPERIMENT_CONFIG["tfidf_config"]

def create_experiment_config(
    language: str,
    task: str,
    model_type: str,
    control_index: int = None,
    submetric: str = None
) -> Dict[str, Any]:
    """Create a single experiment configuration."""
    
    # Determine task type
    if submetric:
        task_type = "regression"
        actual_task = "single_submetric"
    else:
        task_type = XLM_ROBERTA_EXPERIMENT_CONFIG["task_types"][task]
        actual_task = task
    
    # Validate compatibility
    if not validate_xlm_roberta_compatibility(model_type, task_type):
        raise ValueError(f"Model '{model_type}' not compatible with task type '{task_type}'")
    
    # Determine experiment type
    if control_index is not None:
        if submetric:
            experiment_type = "control_submetric"
        else:
            experiment_type = "control"
    else:
        if submetric:
            experiment_type = "submetric"
        else:
            experiment_type = "main"
    
    return {
        "language": language,
        "task": actual_task,
        "task_type": task_type,
        "model_type": model_type,
        "control_index": control_index,
        "submetric": submetric,
        "experiment_type": experiment_type,
        "model_params": get_xlm_roberta_model_params(model_type, task_type)
    }

def estimate_experiment_count() -> Dict[str, int]:
    """Estimate the total number of experiments that will be generated."""
    config = XLM_ROBERTA_EXPERIMENT_CONFIG
    
    n_languages = len(config["languages"])
    n_tasks = len(config["tasks"])
    n_submetrics = len(config["submetrics"])
    n_controls = len(config["control_indices"])
    
    # Count compatible models for each task type
    classification_models = len([m for m in config["models"] 
                               if validate_xlm_roberta_compatibility(m, "classification")])
    regression_models = len([m for m in config["models"] 
                           if validate_xlm_roberta_compatibility(m, "regression")])
    
    counts = {
        "main_question_type": n_languages * classification_models,
        "control_question_type": n_languages * classification_models * n_controls,
        "main_complexity": n_languages * regression_models,
        "control_complexity": n_languages * regression_models * n_controls,
        "main_submetrics": n_languages * n_submetrics * regression_models,
        "control_submetrics": n_languages * n_submetrics * regression_models * n_controls
    }
    
    counts["total"] = sum(counts.values())
    
    return counts

# Configuration validation
def validate_config() -> Dict[str, bool]:
    """Validate the configuration for consistency."""
    validation = {}
    
    config = XLM_ROBERTA_EXPERIMENT_CONFIG
    
    # Check that all required keys exist
    required_keys = ["languages", "tasks", "models", "submetrics", "control_indices", 
                    "task_types", "tokenizer_config", "tfidf_config"]
    
    for key in required_keys:
        validation[f"has_{key}"] = key in config
    
    # Check that tokenizer config has required fields
    tokenizer_config = config.get("tokenizer_config", {})
    required_tokenizer_fields = ["model_name", "tokenizer_class", "special_tokens"]
    
    for field in required_tokenizer_fields:
        validation[f"tokenizer_has_{field}"] = field in tokenizer_config
    
    # Check model compatibility
    for model in config.get("models", []):
        validation[f"model_{model}_exists"] = model in XLM_ROBERTA_TFIDF_PARAMS
    
    return validation

if __name__ == "__main__":
    # Test the configuration
    print("XLM-RoBERTa TF-IDF Configuration Test")
    print("=" * 50)
    
    # Validate configuration
    validation = validate_config()
    print("Configuration validation:")
    for check, passed in validation.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    # Print experiment estimates
    print("\nExperiment count estimates:")
    counts = estimate_experiment_count()
    for exp_type, count in counts.items():
        print(f"  {exp_type}: {count}")
    
    # Test experiment generation
    print("\nTesting experiment generation...")
    try:
        experiments = get_xlm_roberta_experiment_matrix()
        print(f"✓ Generated {len(experiments)} experiments")
        
        # Show sample experiment
        if experiments:
            sample = experiments[0]
            print("Sample experiment:")
            for key, value in sample.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"✗ Experiment generation failed: {e}")
    
    print("\nConfiguration test completed!")