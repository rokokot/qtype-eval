import os
import json
import glob
import re
from pathlib import Path

def load_json_file(file_path):
    """Load and return JSON data from a file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def save_json_file(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Successfully saved: {file_path}")

def extract_metadata_from_path(file_path, base_dir):
    """Extract metadata from file path based on directory structure."""
    relative_path = os.path.relpath(file_path, base_dir)
    path_parts = Path(relative_path).parts
    file_name = Path(file_path).stem
    
    metadata = {
        "file_path": relative_path,
        "file_name": file_name
    }
    
    # Extract information from glot500 directory structure
    if "by_layer" in path_parts:
        # Format: by_layer/layer_X/task_or_submetric/language/control/results.json
        layer_idx = next((p for p in path_parts if p.startswith("layer_")), None)
        if layer_idx:
            metadata["layer"] = int(layer_idx.split("_")[1])
        
        # Language and control extraction
        for lang in ["ar", "en", "fi", "id", "ja", "ko", "ru"]:
            if lang in path_parts:
                metadata["language"] = lang
                break
        
        for part in path_parts:
            if part.startswith("control"):
                try:
                    metadata["control_index"] = int(part.replace("control", ""))
                except ValueError:
                    pass
        
        # Task and submetric extraction
        potential_submetrics = ["avg_links_len", "avg_max_depth", "avg_subordinate_chain_len", 
                               "avg_verb_edges", "lexical_density", "n_tokens"]
        potential_tasks = ["question_type", "complexity"]
        
        for part in path_parts:
            if part in potential_submetrics:
                metadata["submetric"] = part
            elif part in potential_tasks:
                metadata["task"] = part
    
    # Extract information from cross_lingual structure
    elif "cross_lingual" in path_parts:
        # Format: cross_lingual/source_to_target/task/results.json
        cross_lingual_parts = None
        for part in path_parts:
            if "_to_" in part:
                cross_lingual_parts = part.split("_to_")
                break
        
        if cross_lingual_parts and len(cross_lingual_parts) == 2:
            metadata["train_language"] = cross_lingual_parts[0]
            metadata["eval_language"] = cross_lingual_parts[1]
        
        for part in path_parts:
            if part in ["question_type", "complexity"]:
                metadata["task"] = part
        
        metadata["experiment_type"] = "cross_lingual"
    
    return metadata

def process_tfidf_results(tfidf_dir):
    """Process TFIDF results directory."""
    combined_results = {}
    print(f"Processing tfidf results from: {tfidf_dir}")
    tfidf_files = glob.glob(os.path.join(tfidf_dir, "**", "*.json"), recursive=True)
    
    for file_path in tfidf_files:
        relative_path = os.path.relpath(file_path, tfidf_dir)
        model_name = Path(file_path).stem
        
        # Skip summary files 
        if "summary" in model_name:
            continue
            
        data = load_json_file(file_path)
        if not data or not isinstance(data, dict):
            continue
            
        # Add source information
        data["source"] = "tfidf"
        data["file_path"] = relative_path
        
        # Extract language information from per_language_metrics if available
        if "per_language_metrics" in data and data["per_language_metrics"]:
            languages = []
            if "test" in data["per_language_metrics"]:
                languages = list(data["per_language_metrics"]["test"].keys())
            elif "val" in data["per_language_metrics"]:
                languages = list(data["per_language_metrics"]["val"].keys())
            elif "train" in data["per_language_metrics"]:
                languages = list(data["per_language_metrics"]["train"].keys())
            
            # Store languages for later use
            if languages:
                data["languages"] = languages
        
        # Use the model name and task as the key
        if "task" in data:
            # Make sure the model_type field is set
            if "model_type" not in data:
                if "Dummy" in model_name:
                    data["model_type"] = "DummyClassifier" if data["task"] == "question_type" else "DummyRegressor"
                elif "XGB" in model_name:
                    data["model_type"] = "XGBClassifier" if data["task"] == "question_type" else "XGBRegressor"
                elif "Logistic" in model_name:
                    data["model_type"] = "LogisticRegression"
                elif "Ridge" in model_name:
                    data["model_type"] = "Ridge"
            
            key = f"tfidf_{data.get('model_type', 'unknown')}_{data['task']}"
            if "control_index" in data and data["control_index"] is not None:
                key += f"_control{data['control_index']}"
            if "submetric" in data and data["submetric"] is not None:
                key += f"_{data['submetric']}"
            if "language" in data and data["language"] != "all" and data["language"] is not None:
                key += f"_{data['language']}"
            
            combined_results[key] = data
        else:
            # Fallback for files without task field
            combined_results[f"tfidf_{model_name}"] = data
    
    print(f"Processed {len(combined_results)} tfidf results")
    return combined_results

def process_glot500_results(glot500_dir):
    """Process glot500 results with improved handling of directory structure."""
    combined_results = {}
    print(f"Processing glot500 results from: {glot500_dir}")
    glot500_files = glob.glob(os.path.join(glot500_dir, "**", "*.json"), recursive=True)
    
    # First pass: collect layer mapping and metadata files
    layer_mapping = None
    for file_path in glot500_files:
        if "layer_mapping.json" in file_path:
            layer_mapping = load_json_file(file_path)
            if layer_mapping:
                layer_mapping["source"] = "glot500"
                layer_mapping["file_path"] = os.path.relpath(file_path, glot500_dir)
                combined_results["glot500_layer_mapping"] = layer_mapping
                break
    
    # Get unique file types to understand the structure
    unique_file_types = set(Path(f).name for f in glot500_files)
    print(f"Found file types: {unique_file_types}")
    
    # Process result files 
    for file_path in glot500_files:
        file_name = Path(file_path).name
        
        # Skip certain files
        if file_name in ["layer_mapping.json"] or "summary" in file_name:
            continue
        
        # Extract metadata from path
        metadata = extract_metadata_from_path(file_path, glot500_dir)
        data = load_json_file(file_path)
        
        if not data or not isinstance(data, dict):
            continue
        
        # Add source and path info
        data["source"] = "glot500"
        data["file_path"] = metadata["file_path"]
        
        # Ensure model_type is set
        if "model_type" not in data:
            # Try to infer from file name
            if "dummy" in file_name.lower():
                data["model_type"] = "DummyClassifier" if "question_type" in file_path else "DummyRegressor"
            elif "lm" in file_path.lower() or "probe" in file_path.lower():
                data["model_type"] = "lm_probe"
            else:
                data["model_type"] = "unknown"
        
        # Handle various types of result files
        if "results.json" in file_path:
            # Regular results file
            # Combine path metadata with file data
            for key, value in metadata.items():
                if key not in data and key not in ["file_path", "file_name"]:
                    data[key] = value
            
            # Create key based on available metadata
            key_parts = ["glot500"]
            
            # Add model type if available
            model_type = data.get("model_type", "lm_probe")
            key_parts.append(model_type)
            
            # Add task information
            if "task" in data:
                key_parts.append(data["task"])
                
                # Add control index if available
                if "control_index" in data and data["control_index"] is not None:
                    key_parts.append(f"control{data['control_index']}")
                
                # Add submetric if available
                if "submetric" in data and data["submetric"] is not None:
                    key_parts.append(data["submetric"])
                
                # Add language if available
                if "language" in data and data["language"] is not None:
                    key_parts.append(data["language"])
                
                # Add layer if available
                if "layer" in data:
                    key_parts.append(f"layer{data['layer']}")
                
                # Add cross-lingual info if available
                if "experiment_type" in data and data["experiment_type"] == "cross_lingual":
                    if "train_language" in data and "eval_language" in data:
                        key_parts.append(f"{data['train_language']}_to_{data['eval_language']}")
            
            key = "_".join(key_parts)
            combined_results[key] = data
            
        elif "cross_lingual_results.json" in file_path:
            # Cross-lingual results
            for key, value in metadata.items():
                if key not in data and key not in ["file_path", "file_name"]:
                    data[key] = value
            
            key = f"glot500_cross_lingual_{data.get('train_language', 'unknown')}_to_{data.get('eval_language', 'unknown')}_{data.get('task', 'unknown')}"
            combined_results[key] = data
            
        elif "all_results.json" in file_path:
            # Results per language file
            languages = ["ar", "en", "fi", "id", "ja", "ko", "ru"]
            
            # Extract main file metadata
            for key, value in metadata.items():
                if key not in data and key not in ["file_path", "file_name"]:
                    data[key] = value
            
            # Process each language entry
            for lang in languages:
                if lang in data:
                    lang_data = data[lang].copy()
                    # Add metadata
                    lang_data["source"] = "glot500"
                    lang_data["file_path"] = metadata["file_path"]
                    for key, value in metadata.items():
                        if key not in lang_data and key not in ["file_path", "file_name", "language"]:
                            lang_data[key] = value
                    lang_data["language"] = lang
                    
                    # Create key
                    key_parts = ["glot500", lang_data.get("model_type", "lm_probe")]
                    
                    if "task" in lang_data:
                        key_parts.append(lang_data["task"])
                        
                        if "control_index" in lang_data and lang_data["control_index"] is not None:
                            key_parts.append(f"control{lang_data['control_index']}")
                        
                        if "submetric" in lang_data and lang_data["submetric"] is not None:
                            key_parts.append(lang_data["submetric"])
                        
                        key_parts.append(lang)
                        
                        if "layer" in lang_data:
                            key_parts.append(f"layer{lang_data['layer']}")
                    
                    key = "_".join(key_parts)
                    combined_results[key] = lang_data
            
        elif "error" in file_name.lower():
            # Error files - capture these too
            for key, value in metadata.items():
                if key not in data and key not in ["file_path", "file_name"]:
                    data[key] = value
            
            key_parts = ["glot500", "error"]
            
            if "task" in data:
                key_parts.append(data["task"])
            
            if "experiment_type" in data:
                key_parts.append(data["experiment_type"])
            
            if "language" in data:
                key_parts.append(data["language"])
            elif "train_language" in data and "eval_language" in data:
                key_parts.append(f"{data['train_language']}_to_{data['eval_language']}")
            
            key = "_".join(key_parts)
            combined_results[key] = data
            
        # Also process any file with "dummy" in the name to ensure we catch all dummy experiments
        elif "dummy" in file_name.lower():
            # Dummy classifier/regressor results
            for key, value in metadata.items():
                if key not in data and key not in ["file_path", "file_name"]:
                    data[key] = value
            
            # Determine if it's a classifier or regressor
            if "question_type" in file_path:
                data["model_type"] = "DummyClassifier"
                data["task"] = "question_type"
            else:
                data["model_type"] = "DummyRegressor"
                data["task"] = "complexity" if "complexity" in file_path else "regression"
            
            # Create key
            key_parts = ["glot500", data["model_type"], data["task"]]
            
            if "control_index" in data and data["control_index"] is not None:
                key_parts.append(f"control{data['control_index']}")
            
            if "submetric" in data and data["submetric"] is not None:
                key_parts.append(data["submetric"])
            
            if "language" in data and data["language"] is not None:
                key_parts.append(data["language"])
            
            key = "_".join(key_parts)
            combined_results[key] = data
    
    print(f"Processed {len(combined_results)} glot500 results")
    return combined_results

def process_directories(tfidf_dir, glot500_dir, output_file):
    """
    Process both directories and combine results into a single JSON file.
    """
    # Process each directory separately
    tfidf_results = process_tfidf_results(tfidf_dir)
    glot500_results = process_glot500_results(glot500_dir)
    
    # Combine results
    combined_results = {**tfidf_results, **glot500_results}
    
    # Save combined results
    save_json_file(combined_results, output_file)
    print(f"Combined {len(combined_results)} experiment results into {output_file}")
    
    return combined_results

def generate_summary(combined_results, summary_file):
    """
    Generate a summary of the combined results focusing on key metrics.
    """
    summary = {}
    
    for key, data in combined_results.items():
        # Skip layer mapping and non-experiment data
        if key == "glot500_layer_mapping" or "task" not in data:
            continue
            
        task = data["task"]
        source = data["source"]
        model_type = data.get("model_type", "unknown")
        control_index = data.get("control_index", None)
        submetric = data.get("submetric", None)
        language = data.get("language", None)
        layer = data.get("layer", None)
        train_language = data.get("train_language", None)
        eval_language = data.get("eval_language", None)
        languages = data.get("languages", None)  # For tfidf multi-language results
        
        # Create a shorter, more readable key
        summary_key = f"{source}_{model_type}_{task}"
        
        if control_index is not None:
            summary_key += f"_control{control_index}"
        
        if submetric is not None:
            summary_key += f"_{submetric}"
        
        if language is not None:
            summary_key += f"_{language}"
        
        if layer is not None:
            summary_key += f"_layer{layer}"
        
        if train_language is not None and eval_language is not None:
            summary_key += f"_{train_language}_to_{eval_language}"
            
        # Extract key metrics based on task type
        if task == "question_type" or task == "classification":
            # Classification task
            summary_entry = {
                "source": source,
                "model_type": model_type,
                "task": task,
                "control_index": control_index,
                "language": language,
                "layer": layer,
                "train_language": train_language,
                "eval_language": eval_language,
                "languages": languages,  # Add languages list for tfidf results
                "train_accuracy": data.get("train_metrics", {}).get("accuracy"),
                "val_accuracy": data.get("val_metrics", {}).get("accuracy"),
                "test_accuracy": data.get("test_metrics", {}).get("accuracy"),
                "train_f1": data.get("train_metrics", {}).get("f1"),
                "val_f1": data.get("val_metrics", {}).get("f1"),
                "test_f1": data.get("test_metrics", {}).get("f1"),
                "training_time": data.get("train_time") or data.get("training_time")
            }
            
            # Handle the case where metrics might be directly in the data for some dummy models
            if summary_entry["train_accuracy"] is None and "accuracy" in data.get("train_metrics", {}):
                summary_entry["train_accuracy"] = data["train_metrics"]["accuracy"]
            if summary_entry["val_accuracy"] is None and "accuracy" in data.get("val_metrics", {}):
                summary_entry["val_accuracy"] = data["val_metrics"]["accuracy"]
            if summary_entry["test_accuracy"] is None and "accuracy" in data.get("test_metrics", {}):
                summary_entry["test_accuracy"] = data["test_metrics"]["accuracy"]
            
            # Add per-language metrics if available
            if "per_language_metrics" in data:
                summary_entry["per_language_metrics"] = data["per_language_metrics"]
                
            summary[summary_key] = summary_entry
            
        elif task == "complexity" or task == "regression" or task == "single_submetric":
            # Regression task
            summary_entry = {
                "source": source,
                "model_type": model_type,
                "task": task,
                "control_index": control_index,
                "submetric": submetric,
                "language": language,
                "layer": layer,
                "train_language": train_language,
                "eval_language": eval_language,
                "languages": languages,  # Add languages list for tfidf results
                "train_mse": data.get("train_metrics", {}).get("mse"),
                "val_mse": data.get("val_metrics", {}).get("mse"),
                "test_mse": data.get("test_metrics", {}).get("mse"),
                "train_r2": data.get("train_metrics", {}).get("r2"),
                "val_r2": data.get("val_metrics", {}).get("r2"),
                "test_r2": data.get("test_metrics", {}).get("r2"),
                "training_time": data.get("train_time") or data.get("training_time")
            }
            
            # Handle the case where metrics might be directly in the data for some dummy models
            if summary_entry["train_mse"] is None and "mse" in data.get("train_metrics", {}):
                summary_entry["train_mse"] = data["train_metrics"]["mse"]
            if summary_entry["val_mse"] is None and "mse" in data.get("val_metrics", {}):
                summary_entry["val_mse"] = data["val_metrics"]["mse"]
            if summary_entry["test_mse"] is None and "mse" in data.get("test_metrics", {}):
                summary_entry["test_mse"] = data["test_metrics"]["mse"]
            
            # Add per-language metrics if available
            if "per_language_metrics" in data:
                summary_entry["per_language_metrics"] = data["per_language_metrics"]
                
            summary[summary_key] = summary_entry
    
    # Save combined summary
    save_json_file(summary, summary_file)
    print(f"Generated combined summary with {len(summary)} entries in {summary_file}")
    
    # Create task-specific summaries only
    tasks = ["question_type", "complexity", "single_submetric"]
    
    # By task
    for task in tasks:
        task_summary = {k: v for k, v in summary.items() if v.get("task") == task}
        if task_summary:
            task_file = f"{task}_summary.json"
            save_json_file(task_summary, task_file)
            print(f"Generated {task} summary with {len(task_summary)} entries") or data.get("training_time")
            
            # Add per-language metrics if available
            if "per_language_metrics" in data:
                summary_entry["per_language_metrics"] = data["per_language_metrics"]
                
            summary[summary_key] = summary_entry
            
        elif task == "complexity" or task == "regression" or task == "single_submetric":
            # Regression task
            summary_entry = {
                "source": source,
                "model_type": model_type,
                "task": task,
                "control_index": control_index,
                "submetric": submetric,
                "language": language,
                "layer": layer,
                "train_language": train_language,
                "eval_language": eval_language,
                "languages": languages,  # Add languages list for tfidf results
                "train_mse": data.get("train_metrics", {}).get("mse"),
                "val_mse": data.get("val_metrics", {}).get("mse"),
                "test_mse": data.get("test_metrics", {}).get("mse"),
                "train_r2": data.get("train_metrics", {}).get("r2"),
                "val_r2": data.get("val_metrics", {}).get("r2"),
                "test_r2": data.get("test_metrics", {}).get("r2"),
                "training_time": data.get("train_time") or data.get("training_time")
            }
            
            # Add per-language metrics if available
            if "per_language_metrics" in data:
                summary_entry["per_language_metrics"] = data["per_language_metrics"]
                
            summary[summary_key] = summary_entry
    
    # Save combined summary
    save_json_file(summary, summary_file)
    print(f"Generated combined summary with {len(summary)} entries in {summary_file}")
    
    # Create task-specific summaries only
    tasks = ["question_type", "complexity", "single_submetric"]
    
    # By task
    for task in tasks:
        task_summary = {k: v for k, v in summary.items() if v.get("task") == task}
        if task_summary:
            task_file = f"{task}_summary.json"
            save_json_file(task_summary, task_file)
            print(f"Generated {task} summary with {len(task_summary)} entries")

if __name__ == "__main__":
    # Define directories and output files
    tfidf_dir = "/home/robin/Research/qtype-eval/tfidf_all_results_summary"  # Update with your actual tfidf results directory
    glot500_dir = "/home/robin/Research/qtype-eval/final_glot500_results"    # Update with your actual glot500 results directory
    output_file = "combined_results.json"
    summary_file = "combined_summary.json"
    
    # Process directories and combine results
    combined_results = process_directories(tfidf_dir, glot500_dir, output_file)
    
    # Generate summary of results
    generate_summary(combined_results, summary_file)