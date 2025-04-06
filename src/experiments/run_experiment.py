# main experiment runner

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import json
import wandb
import numpy as np
import torch
from typing import Optional, List

from src.data.datasets import ensure_string_task, load_sklearn_data, create_lm_dataloaders, TASK_TO_FEATURE
from src.models.model_factory import create_model
from src.training.sklearn_trainer import SklearnTrainer
from src.training.lm_trainer import LMTrainer

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "./data/cache")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def setup_wandb(
    cfg: DictConfig,
    experiment_type: str,
    task: str,
    model_type: str,
    language: Optional[str] = None,
    languages: Optional[List[str]] = None,
    train_language: Optional[str] = None,
    eval_language: Optional[str] = None):

    if cfg.wandb.mode == "disabled":
        return None

    tags = [str(experiment_type)]

    if isinstance(task, (list, dict)) and hasattr(task, '__iter__'):
        task_str = str(task[0]) if task else ""
    else:
        task_str = str(task)
    tags.append(task_str)

    tags.append(str(model_type))



    if language:
        tags.append(f"lang_{language}")
    elif languages:
        for lang in languages:
            tags.append(f"lang_{lang}")

    if train_language and eval_language:
        tags.append(f"cross_{train_language}_to_{eval_language}")

    if cfg.experiment.use_controls:
        tags.append(f"control_{cfg.experiment.control_index}")

    wandb_config = {
        "experiment": {
            "type": experiment_type,
            "task": task,
            "language": language,
            "languages": languages,
            "train_language": train_language,
            "eval_language": eval_language,
            "use_controls": cfg.experiment.use_controls,
            "control_index": cfg.experiment.control_index if cfg.experiment.use_controls else None,
        },
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "training": OmegaConf.to_container(cfg.training, resolve=True),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "seed": cfg.seed,
    }
    try:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.experiment_name,
            config=wandb_config,
            tags=tags,
            mode=cfg.wandb.mode,
            job_type=experiment_type,
        )
        
        # Create artifact only if wandb initialized successfully
        if run is not None:
            try:
                config_artifact = wandb.Artifact(name=f"config_{cfg.experiment_name}", type="config")
                config_path = os.path.join(cfg.output_dir, "config.yaml")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                
                # Write config
                with open(config_path, "w") as f:
                    f.write(OmegaConf.to_yaml(cfg))
                    
                config_artifact.add_file(config_path)
                run.log_artifact(config_artifact)
            except Exception as e:
                logger.warning(f"Failed to log config artifact: {e}")
        
        return run
        
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        return None


def process_task_list(tasks):
    
    if tasks is None:
        return "question_type"
    
    # If tasks is already a string, ensure it's properly formatted
    if isinstance(tasks, str):
        return tasks.strip().lower()
    
    # If tasks is a list, take the first non-empty item
    if isinstance(tasks, list) and len(tasks) > 0:
        valid_tasks = [t for t in tasks if t]
        if valid_tasks:
            # Return the first valid task as a properly formatted string
            return valid_tasks[0].strip().lower()
    
    # Default fallback
    return "question_type"
    
    """
    if tasks is None:
        return "question_type"
    
   
    if isinstance(tasks, str):
        return ensure_string_task(tasks)
    
    
    if isinstance(tasks, list):
        
        valid_tasks = [t for t in tasks if t]
        
        
        if not valid_tasks:
            return "question_type"
        
        # Take the first valid task and normalize
        return ensure_string_task(valid_tasks[0])
    
    return "question_type"
    """
    

def validate_task(task):
    """
    Validate and log warnings about task selection.
    
    Args:
        task (str): Normalized task string
    
    Returns:
        str: Validated task string
    """
    valid_tasks = [
        "question_type", 
        "complexity", 
        "single_submetric",
        "avg_links_len", 
        "avg_max_depth", 
        "avg_subordinate_chain_len", 
        "avg_verb_edges", 
        "lexical_density", 
        "n_tokens"
    ]
    
    # Check if task is valid
    if task not in valid_tasks:
        logger.warning(f"Task '{task}' is not recognized. Valid tasks: {valid_tasks}")
        return "question_type"  # Default fallback
        
    return task



@hydra.main(config_path="../../configs", config_name="config", version_base="1.1")

def main(cfg: DictConfig):
    """Robust main function to run experiments based on configuration."""
    # Set random seeds for reproducibility
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    try:
        task = process_task_list(cfg.experiment.tasks)
        
        # Validate task
        task = validate_task(task)
        
        logger.info(f"Processed Task: {task}")
        
        # Verify task is valid
        valid_tasks = list(TASK_TO_FEATURE.keys()) + ["avg_links_len", "avg_max_depth", 
                            "avg_subordinate_chain_len", "avg_verb_edges", 
                            "lexical_density", "n_tokens"]
                            
        if task not in valid_tasks:
            logger.warning(f"Task '{task}' may not be properly recognized. Valid tasks: {valid_tasks}")
    except Exception as e:
        logger.error(f"Error processing task: {e}")
        task = "question_type"  # Default to question_type as fallback
        logger.info(f"Defaulting to task: {task}")
        
    # Determine task type with more robust logic
    def determine_task_type(task, cfg):
        """Dynamically determine task type based on task and configuration."""
        # Explicit task type override
        if hasattr(cfg.training, 'task_type') and cfg.training.task_type not in ['default', 'auto']:
            return cfg.training.task_type
            
        

        if task == "question_type":
            return "classification"
        elif task in ["complexity", "single_submetric", "avg_links_len", "avg_max_depth", 
                     "avg_subordinate_chain_len", "avg_verb_edges", "lexical_density", 
                     "n_tokens"]:
            return "regression"
    
        # Default fallback
        logger.warning(f"Could not determine task type for '{task}'. Defaulting to classification.")
        return "classification"

    # Determine task type
    task_type = determine_task_type(task, cfg)
    logger.info(f"Determined Task Type: {task_type}")

    # Handle submetric specifically
    submetric = None
    if task == 'single_submetric':
        submetric = getattr(cfg.experiment, 'submetric', None)
        if not submetric:
            
            logger.warning("Submetric task specified without a specific submetric. Defaulting to 'avg_links_len'")
            submetric = 'avg_links_len'
        logger.info(f"Submetric: {submetric}")

    # Experiment type routing with enhanced logging
    try:
        if cfg.experiment.type == "sklearn_baseline":
            wandb_run = setup_wandb(
                cfg=cfg,
                experiment_type=cfg.experiment.type,
                task=submetric or task,
                model_type=cfg.model.model_type,
                languages=cfg.data.languages,
            )
            results = run_sklearn_experiment(cfg, task, task_type, submetric)
            if wandb_run:
                wandb_run.finish()

        elif cfg.experiment.type == "lm_probe":
            results = run_lm_experiment(cfg, task, task_type, submetric)

        elif cfg.experiment.type == "lm_probe_cross_lingual":
            # Cross-lingual specific wandb setup
            wandb_run = setup_wandb(
                cfg=cfg,
                experiment_type=cfg.experiment.type,
                task=task,
                model_type=cfg.model.model_type,
                train_language=cfg.data.train_language,
                eval_language=cfg.data.eval_language,
            )
            results = run_cross_lingual_experiment(cfg, task, task_type)
            if wandb_run:
                wandb_run.finish()

        else:
            raise ValueError(f"Unsupported experiment type: {cfg.experiment.type}")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results = None

    # Final cleanup
    if wandb.run is not None:
        wandb.finish()

    return results


def run_sklearn_experiment(cfg, task, task_type, submetric=None):
    """Run scikit-learn baseline experiment."""
    logger.info(f"Running sklearn experiment with {cfg.model.model_type} for {task}")
    if submetric:
        logger.info(f"Submetric: {submetric}")

    # Get control settings
    control_index = cfg.experiment.control_index if cfg.experiment.use_controls else None

    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
        languages=cfg.data.languages,
        task=task,
        submetric=submetric,
        control_index=control_index,
        vectors_dir=cfg.data.vectors_dir,
    )

    # Create model
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model = create_model(cfg.model.model_type, task_type, **model_params)

    # Create trainer
    trainer = SklearnTrainer(model=model, task_type=task_type, output_dir=cfg.output_dir, wandb_run=wandb.run)

    # Train and evaluate
    results = trainer.train(train_data=(X_train, y_train), val_data=(X_val, y_val), test_data=(X_test, y_test))

    # Add metadata
    metadata = {
        "task": task,
        "task_type": task_type,
        "model_type": cfg.model.model_type,
        "languages": cfg.data.languages,
        "is_control": cfg.experiment.use_controls,
        "control_index": cfg.experiment.control_index,
    }

    if submetric:
        metadata["submetric"] = submetric

    results.update(metadata)

    # Save results
    if cfg.output_dir:
        import json

        with open(os.path.join(cfg.output_dir, "results_with_metadata.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results


def run_lm_experiment(cfg, task, task_type, submetric=None):
    """Run language model probing experiment."""
    
    if task == "single_submetric" and not submetric:
        submetric = getattr(cfg.experiment, "submetric", None)
        logger.info(f"Using submetric from config: {submetric}")
    
    
    if isinstance(task, list):
        task_str = task[0] if task else "question_type"
        logger.info(f"Converting task list {task} to string '{task_str}'")
        task = task_str
        
    logger.info(f"Running LM probe experiment for {task} on languages: {cfg.data.languages}")
    
    if submetric:
        logger.info(f"Submetric: {submetric}")
    
    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    all_results = {}
    
    for language in cfg.data.languages:
        logger.info(f"Processing language: {language}")
        
        # Setup WandB with error handling
        try:
            wandb_run = setup_wandb(
                cfg=cfg,
                experiment_type=cfg.experiment.type,
                task=task if not submetric else submetric,
                model_type=cfg.model.model_type,
                language=language,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb for language {language}: {str(e)}")
            wandb_run = None
        
        try:
            # Get control settings
            control_index = cfg.experiment.control_index if cfg.experiment.use_controls else None
            
            # Create dataloaders with error handling
            try:
                # Handle case where task might be a list or string
                task_str = task[0] if isinstance(task, list) else task
                
                train_loader, val_loader, test_loader = create_lm_dataloaders(
                    language=language,
                    task=task_str,
                    model_name=cfg.model.lm_name,
                    batch_size=cfg.training.batch_size,
                    control_index=control_index,
                    cache_dir=cfg.data.cache_dir,
                    num_workers=cfg.training.num_workers,
                    submetric=submetric
                )
            except Exception as loader_error:
                logger.error(f"Failed to create dataloaders for {language}: {task}")
                logger.error(f"Error details: {loader_error}")
                
                # Save error information
                error_info = {
                    "language": language,
                    "task": str(task),
                    "error": str(loader_error),
                    "error_type": type(loader_error).__name__
                }
                
                error_path = os.path.join(cfg.output_dir, f"error_{language}.json")
                with open(error_path, "w") as f:
                    json.dump(error_info, f, indent=2)
                
                continue  # Skip to next language
            
            # Create model
            try:
                model_params = OmegaConf.to_container(cfg.model, resolve=True)
                model_params_copy = model_params.copy()
                if 'model_type' in model_params_copy:
                    model_params_copy.pop('model_type')
                model = create_model(cfg.model.model_type, task_type, **model_params_copy)
                
                logger.info(f"Successfully created model for {language}")
            except Exception as model_error:
                logger.error(f"Failed to create model for {language}: {model_error}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                continue
            
            # Create language-specific output directory
            language_output_dir = os.path.join(cfg.output_dir, language)
            os.makedirs(language_output_dir, exist_ok=True)
            
            # Train and evaluate
            trainer = LMTrainer(
                model=model,
                task_type=task_type,
                learning_rate=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
                num_epochs=cfg.training.num_epochs,
                patience=cfg.training.patience,
                output_dir=language_output_dir,
                wandb_run=wandb_run,
            )
            
            results = trainer.train(
                train_loader=train_loader, 
                val_loader=val_loader, 
                test_loader=test_loader
            )
            
            # Add metadata
            results.update({
                "language": language,
                "task": str(task),
                "task_type": task_type,
                "model_type": cfg.model.model_type,
                "is_control": cfg.experiment.use_controls,
                "control_index": cfg.experiment.control_index,
            })
            
            if submetric:
                results["submetric"] = submetric
            
            all_results[language] = results
            
            # Save language-specific results
            with open(os.path.join(language_output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
            
            # Log model as an artifact if WandB is enabled
            if wandb_run:
                try:
                    model_artifact = wandb.Artifact(
                        name=f"model_{task}_{language}", 
                        type="model", 
                        description=f"Trained model for {task} on {language}"
                    )
                    model_artifact.add_dir(language_output_dir)
                    wandb_run.log_artifact(model_artifact)
                except Exception as artifact_error:
                    logger.warning(f"Failed to log model artifact: {artifact_error}")
                
                # Finish this language's run
                wandb_run.finish()
        
        except Exception as language_error:
            logger.error(f"Error processing language {language}: {language_error}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Save error information
            error_info = {
                "language": language,
                "task": str(task),
                "error": str(language_error),
                "error_type": type(language_error).__name__,
                "traceback": traceback.format_exc()
            }
            
            error_path = os.path.join(cfg.output_dir, f"error_{language}.json")
            with open(error_path, "w") as f:
                json.dump(error_info, f, indent=2)
                
            # Continue with other languages
            continue
    
    # Save combined results
    results_path = os.path.join(cfg.output_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def run_cross_lingual_experiment(cfg, task, task_type):
    """Run cross-lingual transfer experiment."""
    logger.info(f"Running cross-lingual experiment: {cfg.data.train_language} -> {cfg.data.eval_language}")


    train_loader, val_loader, _ = create_lm_dataloaders(
        language=cfg.data.train_language,
        task=task,
        model_name=cfg.model.lm_name,
        batch_size=cfg.training.batch_size,
        cache_dir=cfg.data.cache_dir,
        num_workers=cfg.training.num_workers,
    )

    # Get target language data (test set)
    _, _, test_loader = create_lm_dataloaders(
        language=cfg.data.eval_language,
        task=task,
        model_name=cfg.model.lm_name,
        batch_size=cfg.training.batch_size,
        cache_dir=cfg.data.cache_dir,
        num_workers=cfg.training.num_workers,
    )

    # Create model
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model_params_copy = model_params.copy()
    if 'model_type' in model_params_copy:
        model_params_copy.pop('model_type')
    model = create_model("lm_probe", task_type, **model_params_copy)

    # Train and evaluate
    trainer = LMTrainer(
        model=model,
        task_type=task_type,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        num_epochs=cfg.training.num_epochs,
        patience=cfg.training.patience,
        output_dir=cfg.output_dir,
        wandb_run=wandb.run,
    )

    results = trainer.train(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    # Add metadata
    results.update(
        {
            "train_language": cfg.data.train_language,
            "eval_language": cfg.data.eval_language,
            "task": task,
            "task_type": task_type,
            "model_type": cfg.model.model_type,
        }
    )

    # Save results
    with open(os.path.join(cfg.output_dir, "cross_lingual_results.json"), "w") as f:
        import json

        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
