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
        
        if run is not None:
            try:
                config_artifact = wandb.Artifact(name=f"config_{cfg.experiment_name}", type="config")
                config_path = os.path.join(cfg.output_dir, "config.yaml")
                
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                
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

def normalize_task_name(task):
    if task is None:
        return "question_type"
    
    if isinstance(task, list):
        if not task:
            return "question_type"
        task = task[0]
    
    task = str(task).strip().lower()
    
    task_mapping = {"question_type": "question_type", "complexity": "complexity", "complexity_score": "complexity", "lang_norm_complexity_score": "complexity"}
    
    return task_mapping.get(task, task)

def determine_task_type(task, cfg=None):
    if cfg and hasattr(cfg.training, 'task_type') and cfg.training.task_type not in ['auto', 'default']:
        logger.info(f"Using explicit task_type from config: {cfg.training.task_type}")
        return cfg.training.task_type
    
    if task == "question_type":
        return "classification"
    
    if task in ["complexity", "single_submetric", "avg_links_len", "avg_max_depth", "avg_subordinate_chain_len", "avg_verb_edges", "lexical_density", "n_tokens"]:
        return "regression"
    
    logger.warning(f"Could not determine task type for '{task}'. Defaulting to classification.")
    return "classification"



@hydra.main(config_path="../../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    task = normalize_task_name(cfg.experiment.tasks)
    logger.info(f"Normalized task: {task}")
    
    submetric = None
    if task == "single_submetric":
        submetric = getattr(cfg.experiment, "submetric", None)
        if not submetric:
            logger.warning("Submetric task specified but no submetric provided. Using avg_links_len.")
            submetric = "avg_links_len"
        logger.info(f"Using submetric: {submetric}")
    
    task_type = determine_task_type(task, cfg)
    logger.info(f"Determined Task Type: {task_type}")
    
    if cfg.output_dir:
        os.makedirs(cfg.output_dir, exist_ok=True)
        with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
    
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
        
        elif cfg.experiment.type == "lm_probe" or cfg.experiment.type == 'lm_finetune':
            results = run_lm_experiment(cfg, task, task_type, submetric)
        
        elif cfg.experiment.type == "lm_probe_cross_lingual":
            wandb_run = setup_wandb(
                cfg=cfg,
                experiment_type=cfg.experiment.type,
                task=submetric or task,
                model_type=cfg.model.model_type,
                train_language=cfg.data.train_language,
                eval_language=cfg.data.eval_language,
            )
            results = run_cross_lingual_experiment(cfg, task, task_type, submetric)
            if wandb_run:
                wandb_run.finish()
        
        else:
            raise ValueError(f"Unknown experiment type: {cfg.experiment.type}")
        
        return results
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Traceback: {error_traceback}")
        
        # Save error information
        if cfg.output_dir:
            error_info = {
                "error": str(e),
                "traceback": error_traceback,
                "task": task,
                "task_type": task_type,
                "submetric": submetric,
                "experiment_type": cfg.experiment.type
            }
            error_path = os.path.join(cfg.output_dir, f"error_info.json")
            with open(error_path, "w") as f:
                json.dump(error_info, f, indent=2)
        
        # Clean up wandb if needed
        if wandb.run is not None:
            wandb.finish()
        
        raise


def run_sklearn_experiment(cfg, task, task_type, submetric=None):
    logger.info(f"Running sklearn experiment with {cfg.model.model_type} for {task}")
    if submetric:
        logger.info(f"Submetric: {submetric}")

    control_index = cfg.experiment.control_index if cfg.experiment.use_controls else None

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(languages=cfg.data.languages,
        task=task,
        submetric=submetric,
        control_index=control_index,
        vectors_dir=cfg.data.vectors_dir,)

    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model = create_model(cfg.model.model_type, task_type, **model_params)

    trainer = SklearnTrainer(model=model, task_type=task_type, output_dir=cfg.output_dir, wandb_run=wandb.run)

    results = trainer.train(train_data=(X_train, y_train), val_data=(X_val, y_val), test_data=(X_test, y_test))

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

    if cfg.output_dir:
        import json

        with open(os.path.join(cfg.output_dir, "results_with_metadata.json"), "w") as f:
            json.dump(results, f, indent=2)

    return results


def run_lm_experiment(cfg, task, task_type, submetric=None):
    logger.info(f"Running LM probe experiment for task '{task}' (type: {task_type}) on languages: {cfg.data.languages}")
    if submetric:
        logger.info(f"Using submetric: {submetric}")
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    all_results = {}
    
    for language in cfg.data.languages:
        logger.info(f"Processing language: {language}")
        
        try:
            wandb_run = setup_wandb(cfg=cfg,experiment_type=cfg.experiment.type,task=submetric or task,model_type=cfg.model.model_type,language=language,)
        except Exception as e:
            logger.warning(f"Failed to initialize wandb for language {language}: {str(e)}")
            wandb_run = None
        
        try:
            control_index = cfg.experiment.control_index if cfg.experiment.use_controls else None
            
            train_loader, val_loader, test_loader = create_lm_dataloaders(language=language,
                task=task,model_name=cfg.model.lm_name,batch_size=cfg.training.batch_size,control_index=control_index,cache_dir=cfg.data.cache_dir,num_workers=cfg.training.num_workers,submetric=submetric)
            
            model_params = OmegaConf.to_container(cfg.model, resolve=True)
            model_params_copy = model_params.copy()
            
            if 'model_type' in model_params_copy:
                model_params_copy.pop('model_type')
            
            model = create_model("lm_probe", task_type, **model_params_copy)
            logger.info(f"Successfully created model for {language}")

            is_finetuning = model_params_copy.get('finetune', False)
            grad_accum_steps = 1
            
            if is_finetuning:
                grad_accum_steps = getattr(cfg.training, 'gradient_accumulation_steps', 1)
                logger.info(f'finetuning with gradient accum steps: {grad_accum_steps}')

            language_output_dir = os.path.join(cfg.output_dir, language)
            os.makedirs(language_output_dir, exist_ok=True)
            
            trainer = LMTrainer(model=model,task_type=task_type,learning_rate=cfg.training.lr,weight_decay=cfg.training.weight_decay,num_epochs=cfg.training.num_epochs,patience=cfg.training.patience,output_dir=language_output_dir,wandb_run=wandb_run,gradient_accumulation_steps=grad_accum_steps)
            
            results = trainer.train(
                train_loader=train_loader, 
                val_loader=val_loader, 
                test_loader=test_loader
            )
            
            # Add metadata
            results.update({
                "language": language,
                "task": task,
                "task_type": task_type,
                "model_type": cfg.model.model_type,
                "is_control": cfg.experiment.use_controls,
                "control_index": cfg.experiment.control_index,
                "is_finetune": is_finetuning,
            })
            
            if submetric:
                results["submetric"] = submetric

            
            
            all_results[language] = results
            
            # Save language-specific results
            with open(os.path.join(language_output_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
            
            # Finish wandb run for this language
            if wandb_run:
                wandb_run.finish()
        
        except Exception as e:
            logger.error(f"Error processing language {language}: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Traceback: {error_traceback}")
            
            # Save error information
            error_info = {
                "language": language,
                "task": task,
                "task_type": task_type,
                "submetric": submetric,
                "error": str(e),
                "traceback": error_traceback
            }
            
            error_path = os.path.join(cfg.output_dir, f"error_{language}.json")
            with open(error_path, "w") as f:
                json.dump(error_info, f, indent=2)
            
            # Finish wandb run if active
            if wandb_run:
                wandb_run.finish()
            
            # Continue with other languages
            continue
    
    # Save combined results
    results_path = os.path.join(cfg.output_dir, "all_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


def run_cross_lingual_experiment(cfg, task, task_type, submetric=None):
    """Run cross-lingual transfer experiment with improved task handling."""
    logger.info(f"Running cross-lingual experiment: {cfg.data.train_language} -> {cfg.data.eval_language}")
    logger.info(f"Task: {task}, Task Type: {task_type}")
    if submetric:
        logger.info(f"Submetric: {submetric}")
    
    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    try:
        # Create dataloaders for source language (train)
        train_loader, val_loader, _ = create_lm_dataloaders(
            language=cfg.data.train_language,
            task=task,
            model_name=cfg.model.lm_name,
            batch_size=cfg.training.batch_size,
            cache_dir=cfg.data.cache_dir,
            num_workers=cfg.training.num_workers,
            submetric=submetric
        )
        
        # Create dataloader for target language (test)
        _, _, test_loader = create_lm_dataloaders(
            language=cfg.data.eval_language,
            task=task,
            model_name=cfg.model.lm_name,
            batch_size=cfg.training.batch_size,
            cache_dir=cfg.data.cache_dir,
            num_workers=cfg.training.num_workers,
            submetric=submetric
        )
        
        # Create model
        model_params = OmegaConf.to_container(cfg.model, resolve=True)
        model_params_copy = model_params.copy()
        
        # Remove model_type from parameters if present
        if 'model_type' in model_params_copy:
            model_params_copy.pop('model_type')
        
        # Create model with the determined task_type
        model = create_model("lm_probe", task_type, **model_params_copy)
        logger.info(f"Successfully created model for cross-lingual experiment")
        
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
        results.update({
            "train_language": cfg.data.train_language,
            "eval_language": cfg.data.eval_language,
            "task": task,
            "task_type": task_type,
            "model_type": cfg.model.model_type,
        })
        
        if submetric:
            results["submetric"] = submetric
        
        # Save results
        with open(os.path.join(cfg.output_dir, "cross_lingual_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in cross-lingual experiment: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Traceback: {error_traceback}")
        
        # Save error information
        error_info = {
            "train_language": cfg.data.train_language,
            "eval_language": cfg.data.eval_language,
            "task": task,
            "task_type": task_type,
            "submetric": submetric,
            "error": str(e),
            "traceback": error_traceback
        }
        
        error_path = os.path.join(cfg.output_dir, f"error_cross_lingual.json")
        with open(error_path, "w") as f:
            json.dump(error_info, f, indent=2)
        
        raise


if __name__ == "__main__":
    main()
