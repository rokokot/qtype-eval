#!/bin/bash
EXPERIMENT_TYPE="sklearn_baseline"
MODEL_TYPE="dummy"
TASK="question_type"
SUBMETRIC=""
LANGUAGES="en,fi,id,ja,ko,ru,ar"
USE_CONTROL=false
CONTROL_INDEX=1
CONFIG_OVERRIDE=""

function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -e, --experiment TYPE   Experiment type (sklearn_baseline, lm_probe, lm_probe_cross_lingual)"
    echo "  -m, --model TYPE        Model type (dummy, logistic, ridge, xgboost, lm_probe)"
    echo "  -t, --task TYPE         Task (question_type, complexity, single_submetric)"
    echo "  -s, --submetric NAME    Submetric name (required if task is single_submetric)"
    echo "  -l, --languages LANGS   Comma-separated list of languages (default: all)"
    echo "  -c, --control           Use control task"
    echo "  -i, --index NUM         Control index (1-3)"
    echo "  -o, --override STR      Additional config overrides"
    echo "  -h, --help              Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--experiment)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -s|--submetric)
            SUBMETRIC="$2"
            shift 2
            ;;
        -l|--languages)
            LANGUAGES="$2"
            shift 2
            ;;
        -c|--control)
            USE_CONTROL=true
            shift
            ;;
        -i|--index)
            CONTROL_INDEX="$2"
            shift 2
            ;;
        -o|--override)
            CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate submetric if task is single_submetric
if [[ "$TASK" == "single_submetric" && -z "$SUBMETRIC" ]]; then
    echo "Error: Submetric parameter (-s, --submetric) is required when task is single_submetric"
    show_help
fi

# Construct experiment name
EXPERIMENT_NAME="${MODEL_TYPE}_${TASK}"
if [[ "$TASK" == "single_submetric" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${SUBMETRIC//\//_}"
fi

if [[ "$EXPERIMENT_TYPE" == "lm_probe_cross_lingual" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${TRAIN_LANG}_to_${EVAL_LANG}"
else
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${LANGUAGES//,/_}"
fi

if [[ "$USE_CONTROL" == true ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_control${CONTROL_INDEX}"
fi

# Build Hydra command
CMD="python -m src.experiments.run_experiment"
CMD="${CMD} experiment.type=${EXPERIMENT_TYPE}"
CMD="${CMD} model.model_type=${MODEL_TYPE}"

if [[ "$TASK" == "single_submetric" ]]; then
    CMD="${CMD} experiment=submetrics"
    CMD="${CMD} experiment.submetric=${SUBMETRIC}"
else
    CMD="${CMD} experiment=${TASK}"
fi

if [[ "$EXPERIMENT_TYPE" == "lm_probe_cross_lingual" ]]; then
    CMD="${CMD} data.train_language=${TRAIN_LANG}"
    CMD="${CMD} data.eval_language=${EVAL_LANG}"
else
    CMD="${CMD} data.languages=[${LANGUAGES}]"
fi

if [[ "$USE_CONTROL" == true ]]; then
    CMD="${CMD} experiment.use_controls=true"
    CMD="${CMD} experiment.control_index=${CONTROL_INDEX}"
fi

CMD="${CMD} experiment_name=${EXPERIMENT_NAME}"

if [[ -n "$CONFIG_OVERRIDE" ]]; then
    CMD="${CMD} ${CONFIG_OVERRIDE}"
fi

echo "Running: ${CMD}"
eval ${CMD}