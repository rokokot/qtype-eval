#!/bin/bash
SUBMETRICS=("avg_links_len" "avg_max_depth" "avg_subordinate_chain_len" "avg_verb_edges" "lexical_density" "n_tokens")
LANGUAGES="en,fi,id,ja,ko,ru,ar"
MODEL_TYPE="ridge" 

OUTPUT_DIR="outputs/submetrics_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

LOG_FILE="${OUTPUT_DIR}/run_log.txt"
echo "Starting submetric experiments at $(date)" > $LOG_FILE

for submetric in "${SUBMETRICS[@]}"; do
    echo "Running experiment for submetric: $submetric" | tee -a $LOG_FILE
    
    EXPERIMENT_NAME="${MODEL_TYPE}_${submetric//\//_}"
    
    python -m src.experiments.run_experiment \
        experiment=submetrics \
        experiment.submetric=$submetric \
        model.model_type=$MODEL_TYPE \
        data.languages="[$LANGUAGES]" \
        experiment_name=$EXPERIMENT_NAME \
        output_dir="${OUTPUT_DIR}/${submetric}" \
        2>&1 | tee -a $LOG_FILE
    
    echo "Completed experiment for $submetric" | tee -a $LOG_FILE
    echo "----------------------------------------" | tee -a $LOG_FILE
done

echo "All submetric experiments completed at $(date)" | tee -a $LOG_FILE