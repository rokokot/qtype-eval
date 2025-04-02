# main experiment runner

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import torch
from datetime import datetime

from src.data.datasets import load_sklearn_data, create_lm_dataloaders
from src.models.model_factory import create_model
from src.training.sklearn_trainer import SklearnTrainer
from src.training.lm_trainer import LMTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Add after the existing imports
import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import torch
from datetime import datetime
from typing import Optional, Dict, Any, List

# ... existing code ...

def setup_wandb(cfg: DictConfig,experiment_type: str,task: str,model_type: str,language: Optional[str] = None,languages: Optional[List[str]] = None,train_language: Optional[str] = None,eval_language: Optional[str] = None) -> Optional[wandb.Run]:
  
    if cfg.wandb.mode == "disabled":
        return None
    
    tags = [experiment_type, task, model_type]
    
    if language:
        tags.append(f"lang_{language}")
    elif languages:
        for lang in languages:
            tags.append(f"lang_{lang}")
    
    if train_language and eval_language:
        tags.append(f"cross_{train_language}_to_{eval_language}")
    
    if cfg.experiment.use_controls:
        tags.append(f"control_{cfg.experiment.control_index}")
    
    wandb_config = {"experiment": {"type": experiment_type,
            "task": task,
            "language": language,
            "languages": languages,
            "train_language": train_language,
            "eval_language": eval_language,
            "use_controls": cfg.experiment.use_controls,
            "control_index": cfg.experiment.control_index if cfg.experiment.use_controls else None,
        },"model": OmegaConf.to_container(cfg.model, resolve=True),
        "training": OmegaConf.to_container(cfg.training, resolve=True),
        "data": OmegaConf.to_container(cfg.data, resolve=True),
        "seed": cfg.seed,}
    
    run = wandb.init(project=cfg.wandb.project,entity=cfg.wandb.entity,name=cfg.experiment_name,config=wandb_config,tags=tags,mode=cfg.wandb.mode,job_type=experiment_type)
    
    config_artifact = wandb.Artifact(name=f"config_{cfg.experiment_name}",type="config")
    
    with open(os.path.join(cfg.output_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    config_artifact.add_file(os.path.join(cfg.output_dir, "config.yaml"))
    run.log_artifact(config_artifact)
    
    return run





@hydra.main(config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    
    # === get task 
    task = cfg.experiment.tasks[0] if isinstance(cfg.experiment.tasks, list) else cfg.experiment.tasks
    task_type = "classification" if task == "question_type" else "regression"
    
    # === submetrics
    submetric = None
    if task == "single_submetric" and hasattr(cfg.experiment, 'submetric'):
        submetric = cfg.experiment.submetric
        task_type = "regression"
    

    if cfg.experiment.type == "sklearn_baseline":
        wandb_run = setup_wandb(cfg=cfg,experiment_type=cfg.experiment.type,task=task if not submetric else submetric,
            model_type=cfg.model.model_type,languages=cfg.data.languages)
        
        results = run_sklearn_experiment(cfg, task, task_type, submetric)
        
        if wandb_run:
            wandb_run.finish()


    elif cfg.experiment.type == "lm_probe":
        results = run_lm_experiment(cfg, task, task_type, submetric)

    elif cfg.experiment.type == "lm_probe_cross_lingual":
        wandb_run = setup_wandb(cfg=cfg,
            experiment_type=cfg.experiment.type,
            task=task,
            model_type=cfg.model.model_type,
            train_language=cfg.data.train_language,
            eval_language=cfg.data.eval_language)
        
        results = run_cross_lingual_experiment(cfg, task, task_type)
    
        if wandb_run:
            wandb_run.finish()
    else:
        logger.error(f"Unknown experiment type: {cfg.experiment.type}")
    

    if cfg.wandb.mode != "disabled":
        wandb.finish()

def run_sklearn_experiment(cfg, task, task_type, submetric=None):
    
    logger.info(f"Running sklearn experiment with {cfg.model.model_type} for {task}")
    if submetric:
        logger.info(f"Submetric: {submetric}")
    
    control_index = cfg.experiment.control_index if cfg.experiment.use_controls else None
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_sklearn_data(
        languages=cfg.data.languages,
        task=task,
        submetric=submetric,
        control_index=control_index,
        vectors_dir=cfg.data.vectors_dir)
    
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model = create_model(cfg.model.model_type, task_type, **model_params)
    
    trainer = SklearnTrainer(
        model=model,
        task_type=task_type,
        output_dir=cfg.output_dir)
    
    results = trainer.train(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        test_data=(X_test, y_test))
    
    metadata = {
        "task": task,
        "task_type": task_type,
        "model_type": cfg.model.model_type,
        "languages": cfg.data.languages,
        "is_control": cfg.experiment.use_controls,
        "control_index": cfg.experiment.control_index}
    
    if submetric:
        metadata["submetric"] = submetric
    
    results.update(metadata)
    
    if cfg.wandb.mode != "disabled":
        wandb.log({
            **metadata,
            **{f"train_{k}": v for k, v in results["train_metrics"].items()},
            **{f"val_{k}": v for k, v in results["val_metrics"].items()},
            **{f"test_{k}": v for k, v in results["test_metrics"].items()}})
    
    if cfg.output_dir:
        import json
        with open(os.path.join(cfg.output_dir, "results_with_metadata.json"), "w") as f:
            json.dump(results, f, indent=2)
    
    return results

def run_lm_experiment(cfg, task, task_type, submetric=None):
    
    logger.info(f"Running LM probe experiment for {task} on languages: {cfg.data.languages}")
    if submetric:
        logger.info(f"Submetric: {submetric}")
    
    all_results = {}
    
    for language in cfg.data.languages:
        logger.info(f"Processing language: {language}")
        
        wandb_run = setup_wandb(
            cfg=cfg,
            experiment_type=cfg.experiment.type,
            task=task if not submetric else submetric,
            model_type=cfg.model.model_type,
            language=language)
        
        control_index = cfg.experiment.control_index if cfg.experiment.use_controls else None
        
        train_loader, val_loader, test_loader = create_lm_dataloaders(
            language=language,
            task=task if not submetric else submetric,
            model_name=cfg.model.lm_name,
            batch_size=cfg.training.batch_size,
            control_index=control_index,
            cache_dir=cfg.data.cache_dir,
            num_workers=cfg.training.num_workers
        )
        
        model_params = OmegaConf.to_container(cfg.model, resolve=True)
        model = create_model("lm_probe", task_type, **model_params)
        
        language_output_dir = os.path.join(cfg.output_dir, language)
        os.makedirs(language_output_dir, exist_ok=True)
        
        trainer = LMTrainer(model=model,task_type=task_type,learning_rate=cfg.training.lr,weight_decay=cfg.training.weight_decay,num_epochs=cfg.training.num_epochs,patience=cfg.training.patience,output_dir=language_output_dir,wandb_run=wandb_run)
        
        results = trainer.train(train_loader=train_loader,val_loader=val_loader,test_loader=test_loader)
        
        results.update({
            "language": language,
            "task": task,
            "task_type": task_type,
            "model_type": cfg.model.model_type,
            "is_control": cfg.experiment.use_controls,
            "control_index": cfg.experiment.control_index})
        
        if submetric:
            results["submetric"] = submetric
        
        all_results[language] = results
        
        if wandb_run:
            model_artifact = wandb.Artifact(
                name=f"model_{task}_{language}",
                type="model",
                description=f"Trained model for {task} on {language}")
            
            model_artifact.add_dir(language_output_dir)
            wandb_run.log_artifact(model_artifact)
            
            wandb_run.finish()
    
    with open(os.path.join(cfg.output_dir, "all_results.json"), "w") as f:
        import json
        json.dump(all_results, f, indent=2)
    
    return all_results

def run_cross_lingual_experiment(cfg, task, task_type):
    logger.info(f"Running cross-lingual experiment: {cfg.data.train_language} -> {cfg.data.eval_language}")
    
    train_loader, val_loader, _ = create_lm_dataloaders(
        language=cfg.data.train_language,
        task=task,
        model_name=cfg.model.lm_name,
        batch_size=cfg.training.batch_size,
        cache_dir=cfg.data.cache_dir,
        num_workers=cfg.training.num_workers)
    
    _, _, test_loader = create_lm_dataloaders(
        language=cfg.data.eval_language,
        task=task,
        model_name=cfg.model.lm_name,
        batch_size=cfg.training.batch_size,
        cache_dir=cfg.data.cache_dir,
        num_workers=cfg.training.num_workers)
    
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model = create_model("lm_probe", task_type, **model_params)
    
    trainer = LMTrainer(
        model=model,
        task_type=task_type,
        learning_rate=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        num_epochs=cfg.training.num_epochs,
        patience=cfg.training.patience,
        output_dir=cfg.output_dir)
    
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader)
    
    results.update({
        "train_language": cfg.data.train_language,
        "eval_language": cfg.data.eval_language,
        "task": task,
        "task_type": task_type,
        "model_type": cfg.model.model_type})
    
    if cfg.wandb.mode != "disabled":
        for metric_type in ["train_metrics", "val_metrics", "test_metrics"]:
            if metric_type in results and results[metric_type]:
                for metric_name, metric_value in results[metric_type].items():
                    wandb.log({
                        f"{metric_type}_{metric_name}": metric_value,
                        "train_language": cfg.data.train_language,
                        "eval_language": cfg.data.eval_language,
                        "task": task
                    })
    
    with open(os.path.join(cfg.output_dir, "cross_lingual_results.json"), "w") as f:
        import json
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    main()