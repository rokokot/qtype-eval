# Helper script to generate hydra command line overrides for a specific experiment ID.
# This is used by the job array script to determine which experiment to run.

import sys
from itertools import product

LANGUAGES = ["ar", "en", "fi", "id", "ja", "ko", "ru"]

MAIN_TASKS = ["question_type", "complexity"]
SUBMETRIC_TASKS = [
    "avg_links_len",
    "avg_max_depth",
    "avg_subordinate_chain_len",
    "avg_verb_edges",
    "lexical_density",
    "n_tokens",
]

ALL_TASKS = MAIN_TASKS + SUBMETRIC_TASKS

SKLEARN_MODELS = ["dummy", "logistic", "ridge", "xgboost"]
NEURAL_MODELS = ["lm_probe"]
ALL_MODELS = SKLEARN_MODELS + NEURAL_MODELS

CONTROL_INDICES = [None, 1, 2, 3]


def generate_experiments():
    experiments = []

    for language, task, model_type in product(LANGUAGES, ALL_TASKS, SKLEARN_MODELS):
        if model_type == "logistic" and task != "question_type":
            continue

        experiments.append(
            {
                "model.model_type": model_type,
                "experiment.type": "sklearn_baseline",
                "experiment.tasks": f"[{task}]",
                "data.languages": f"[{language}]",
                "experiment.use_controls": "false",
                "experiment_name": f"{model_type}_{task}_{language}",
            }
        )

        for control_idx in [1, 2, 3]:
            experiments.append(
                {
                    "model.model_type": model_type,
                    "experiment.type": "sklearn_baseline",
                    "experiment.tasks": f"[{task}]",
                    "data.languages": f"[{language}]",
                    "experiment.use_controls": "true",
                    "experiment.control_index": str(control_idx),
                    "experiment_name": f"{model_type}_{task}_control{control_idx}_{language}",
                }
            )

    for language, task in product(LANGUAGES, ALL_TASKS):
        experiments.append(
            {
                "model.model_type": "lm_probe",
                "experiment.type": "lm_probe",
                "experiment.tasks": f"[{task}]",
                "data.languages": f"[{language}]",
                "experiment.use_controls": "false",
                "training.task_type": "classification" if task == "question_type" else "regression",
                "experiment_name": f"lm_probe_{task}_{language}",
            }
        )

        for control_idx in [1, 2, 3]:
            experiments.append(
                {
                    "model.model_type": "lm_probe",
                    "experiment.type": "lm_probe",
                    "experiment.tasks": f"[{task}]",
                    "data.languages": f"[{language}]",
                    "experiment.use_controls": "true",
                    "experiment.control_index": str(control_idx),
                    "training.task_type": "classification" if task == "question_type" else "regression",
                    "experiment_name": f"lm_probe_{task}_control{control_idx}_{language}",
                }
            )

    for train_lang, eval_lang, task in product(LANGUAGES, LANGUAGES, MAIN_TASKS):
        if train_lang == eval_lang:
            continue

        experiments.append(
            {
                "model.model_type": "lm_probe",
                "experiment.type": "lm_probe_cross_lingual",
                "experiment.tasks": f"[{task}]",
                "data.train_language": train_lang,
                "data.eval_language": eval_lang,
                "training.task_type": "classification" if task == "question_type" else "regression",
                "experiment_name": f"cross_lingual_{task}_{train_lang}_to_{eval_lang}",
            }
        )

    return experiments


def get_experiment_config(experiment_id):
    """
    Get the hydra command line overrides for a specific experiment ID.

    Args:
        experiment_id: Experiment ID (1-based index)

    Returns:
        String of hydra command line overrides
    """
    experiments = generate_experiments()

    if experiment_id < 1 or experiment_id > len(experiments):
        raise ValueError(f"Invalid experiment ID: {experiment_id}. Must be between 1 and {len(experiments)}")

    experiment = experiments[experiment_id - 1]

    overrides = " ".join([f"{k}={v}" for k, v in experiment.items()])

    return overrides


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} EXPERIMENT_ID")
        sys.exit(1)

    try:
        experiment_id = int(sys.argv[1])

        config = get_experiment_config(experiment_id)
        print(config)

        print(f"# Total experiments: {len(generate_experiments())}", file=sys.stderr)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
