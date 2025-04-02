#!/bin/bash
LANGUAGES=("ar" "en" "fi" "id" "ja" "ko" "ru")
TASK="question_type"  # or "complexity"
MODEL_TYPE="lm_probe"

# Create output directory
OUTPUT_DIR="outputs/cross_lingual_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Log file
LOG_FILE="${OUTPUT_DIR}/run_log.txt"
echo "Starting cross-lingual experiments at $(date)" > $LOG_FILE

# Run for each language pair
for src_lang in "${LANGUAGES[@]}"; do
    for tgt_lang in "${LANGUAGES[@]}"; do
        # Skip same language
        if [[ "$src_lang" == "$tgt_lang" ]]; then
            continue
        fi

        echo "Running experiment for $src_lang -> $tgt_lang" | tee -a $LOG_FILE

        # Create experiment name
        EXPERIMENT_NAME="cross_lingual_${src_lang}_to_${tgt_lang}"

        # Run the experiment
        python -m src.experiments.run_experiment \
            experiment=cross_lingual \
            experiment.type=lm_probe_cross_lingual \
            model.model_type=$MODEL_TYPE \
            data.train_language=$src_lang \
            data.eval_language=$tgt_lang \
            experiment_name=$EXPERIMENT_NAME \
            output_dir="${OUTPUT_DIR}/${src_lang}_to_${tgt_lang}" \
            2>&1 | tee -a $LOG_FILE

        echo "Completed experiment for $src_lang -> $tgt_lang" | tee -a $LOG_FILE
        echo "----------------------------------------" | tee -a $LOG_FILE
    done
done

echo "All cross-lingual experiments completed at $(date)" | tee -a $LOG_FILE
