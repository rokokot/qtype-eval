#!/bin/bash

OUTPUT_DIR="$VSC_DATA/final_results"
echo "collecting to $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

mkdir -p "$OUTPUT_DIR/by_layer"
for layer in 2 6 11 12; do
  mkdir -p "$OUTPUT_DIR/by_layer/layer_$layer"
done

LAYERWISE_DIR="$VSC_SCRATCH/layerwise_output"
if [ -d "$LAYERWISE_DIR" ]; then
  echo "layerwise experiments..."
  
  for lang in ar en fi id ja ko ru; do
    lang_dir="$LAYERWISE_DIR/$lang"
    if [ ! -d "$lang_dir" ]; then continue; fi
    
    for layer_dir in "$lang_dir"/layer_*; do
      if [ ! -d "$layer_dir" ]; then continue; fi
      layer_name=$(basename "$layer_dir")
      layer_num=${layer_name#layer_}
      
      for task_dir in "$layer_dir"/*; do
        if [ ! -d "$task_dir" ]; then continue; fi
        task_name=$(basename "$task_dir")
        
        unified_dir="$OUTPUT_DIR/by_layer/layer_$layer_num/$task_name/$lang"
        mkdir -p "$unified_dir"
        
        find "$task_dir" -maxdepth 1 -name "*.json" -exec cp {} "$unified_dir/" \;
        
        for control_dir in "$task_dir"/control*; do
          if [ ! -d "$control_dir" ]; then continue; fi
          control_name=$(basename "$control_dir")
          
          control_unified_dir="$unified_dir/$control_name"
          mkdir -p "$control_unified_dir"
          
          find "$control_dir" -name "*.json" -exec cp {} "$control_unified_dir/" \;
        done
      done
    done
  done
fi

MAIN_EXPERIMENTS=("complexity_output" "question_type_output" "submetric_output")

for exp_dir in "${MAIN_EXPERIMENTS[@]}"; do
  source_dir="$VSC_SCRATCH/$exp_dir"
  if [ ! -d "$source_dir" ]; then continue; fi
  
  echo "looking in $exp_dir (layer 12)..."
  
  if [[ "$exp_dir" == "complexity_output" ]]; then
    task_name="complexity"
  elif [[ "$exp_dir" == "question_type_output" ]]; then
    task_name="question_type"
  elif [[ "$exp_dir" == "submetric_output" ]]; then
    task_name="submetric"
  else
    task_name=$(basename "$exp_dir")
  fi
  
  for lang in ar en fi id ja ko ru; do
    lang_dir="$source_dir/$lang"
    if [ ! -d "$lang_dir" ]; then continue; fi
    
    echo "processing $lang"
    
    unified_dir="$OUTPUT_DIR/by_layer/layer_12/$task_name/$lang"
    mkdir -p "$unified_dir"
    
    find "$lang_dir" -maxdepth 1 -name "*.json" -exec cp {} "$unified_dir/" \;
    
    for control_dir in "$lang_dir"/control*; do
      if [ ! -d "$control_dir" ]; then continue; fi
      control_name=$(basename "$control_dir")
      
      control_unified_dir="$unified_dir/$control_name"
      mkdir -p "$control_unified_dir"
      
      find "$control_dir" -name "*.json" -exec cp {} "$control_unified_dir/" \;
    done
  done
  
  find "$source_dir" -maxdepth 1 -name "*.json" -exec cp {} "$OUTPUT_DIR/by_layer/layer_12/$task_name/" \;
done

CROSS_LINGUAL_DIR="$VSC_SCRATCH/cross_lingual_output"
if [ -d "$CROSS_LINGUAL_DIR" ]; then
  echo "Processing cross-lingual experiments..."
  mkdir -p "$OUTPUT_DIR/cross_lingual"
  
  for pair_dir in "$CROSS_LINGUAL_DIR"/*_to_*; do
    if [ ! -d "$pair_dir" ]; then continue; fi
    pair_name=$(basename "$pair_dir")
    
    mkdir -p "$OUTPUT_DIR/cross_lingual/$pair_name"
    
    for task_dir in "$pair_dir"/*; do
      if [ ! -d "$task_dir" ]; then continue; fi
      task_name=$(basename "$task_dir")
      
      unified_dir="$OUTPUT_DIR/cross_lingual/$pair_name/$task_name"
      mkdir -p "$unified_dir"
      
      find "$task_dir" -name "*.json" -exec cp {} "$unified_dir/" \;
    done
  done
fi

cat > "$OUTPUT_DIR/layer_mapping.json" << EOF
{
  "layers": [2, 6, 11, 12],
  "layer_sources": {
    "2": "layerwise_output/*/layer_2/*",
    "6": "layerwise_output/*/layer_6/*",
    "11": "layerwise_output/*/layer_11/*",
    "12": ["complexity_output", "question_type_output", "submetric_output"]
  },
  "tasks": ["question_type", "complexity", "submetric"],
  "languages": ["ar", "en", "fi", "id", "ja", "ko", "ru"],
  "collection_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

echo " complete. files in $OUTPUT_DIR/by_layer."