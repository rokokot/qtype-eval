# utils/collect_finetune_results.sh
#!/bin/bash

OUTPUT_DIR="$VSC_DATA/finetune_results"
echo "Collecting fine-tuning results to $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

SOURCE_DIR="$VSC_SCRATCH/finetune_output"

if [ -d "$SOURCE_DIR" ]; then

    for task_dir in "$SOURCE_DIR"/*; do
        if [ ! -d "$task_dir" ]; then continue; fi
        task_name=$(basename "$task_dir")
        
        mkdir -p "$OUTPUT_DIR/$task_name"
        
        for lang_dir in "$task_dir"/*; do
            if [ ! -d "$lang_dir" ]; then continue; fi
            lang_name=$(basename "$lang_dir")
            
            unified_dir="$OUTPUT_DIR/$task_name/$lang_name"
            mkdir -p "$unified_dir"
            
            find "$lang_dir" -maxdepth 1 -name "*.json" -exec cp {} "$unified_dir/" \;
            
            if [ "$task_name" = "submetrics" ]; then
                for submetric_dir in "$lang_dir"/*; do
                    if [ ! -d "$submetric_dir" ]; then continue; fi
                    submetric_name=$(basename "$submetric_dir")
                    
                    submetric_unified_dir="$unified_dir/$submetric_name"
                    mkdir -p "$submetric_unified_dir"
                    
                    find "$submetric_dir" -name "*.json" -exec cp {} "$submetric_unified_dir/" \;
                done
            fi
        done
    done
fi

SUMMARY_FILE="$OUTPUT_DIR/finetune_summary.json"
echo "{" > $SUMMARY_FILE
echo "  \"collection_date\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\"," >> $SUMMARY_FILE
echo "  \"tasks\": [\"question_type\", \"complexity\", \"submetrics\"]," >> $SUMMARY_FILE
echo "  \"languages\": [\"ar\", \"en\", \"fi\", \"id\", \"ja\", \"ko\", \"ru\"]" >> $SUMMARY_FILE
echo "}" >> $SUMMARY_FILE

echo "Results collection complete. Files are in $OUTPUT_DIR"