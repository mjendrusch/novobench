#!/bin/bash
# This script provides a convenient way to run novobench with multiple AlphaFold models
# or ESMfold in a single command, with control over the prediction mode.
# Usage examples:
#   ./run_models.sh example_input/pdbs/ example_input/sequences/ example_output/ 1
#   ./run_models.sh example_input/pdbs/ example_input/sequences/ example_output/ 1-5 abinitio
#   ./run_models.sh example_input/pdbs/ example_input/sequences/ example_output/ 6 guess

usage() {
    echo "Usage: ./run_models.sh <pdb_path> <fasta_path> <output_path> <model_spec> [prediction_mode]"
    echo ""
    echo "Parameters:"
    echo "  pdb_path:        Path to directory containing PDB files"
    echo "  fasta_path:      Path to directory containing FASTA sequence files"
    echo "  output_path:     Path where results will be saved"
    echo "  model_spec:      Model specification:"
    echo "                   - Single number (1-5) for a specific AlphaFold model"
    echo "                   - Range (1-5) to run all AlphaFold models"
    echo "                   - 6 for ESMfold"
    echo "  prediction_mode: (optional) Either 'guess' (default) or 'abinitio'"
    exit 1
}

if [ "$#" -lt 4 ]; then
    usage
fi

# Parse arguments
PDB_PATH="$1"
FASTA_PATH="$2"
OUTPUT_PATH="$3"
MODEL_SPEC="$4"
PREDICTION_MODE="${5:-guess}"  # Default to "guess" if not provided

# Validate prediction mode
if [ "$PREDICTION_MODE" != "guess" ] && [ "$PREDICTION_MODE" != "abinitio" ]; then
    echo "ERROR: Invalid prediction mode: $PREDICTION_MODE"
    echo "Must be either 'guess' or 'abinitio'"
    usage
fi

# Function to run models in a given range
run_models() {
    local start=$1
    local end=$2
    
    for model_num in $(seq $start $end); do
        # Determine model name and parameter path
        if [ "$model_num" -le 5 ]; then
            model_name="af_${model_num}"
            model_dir="af${model_num}"
            param_arg="--parameter-path ./"
        else
            model_name="esm"
            model_dir="esm"
            param_arg=""  # ESM doesn't need parameter path
        fi
        
        # Create output directory
        model_output="${OUTPUT_PATH}/${model_dir}"
        mkdir -p "$model_output"
        
        echo "Running novobench with model ${model_name}, prediction mode ${PREDICTION_MODE}..."
        
        # Run the model
        conda activate novobench
        python -m novobench.run \
            --pdb-path "$PDB_PATH" \
            --fasta-path "$FASTA_PATH" \
            --out-path "$model_output" \
            --model "$model_name" \
            --templated none \
            --prediction-mode "$PREDICTION_MODE" \
            $param_arg
        
        echo "Completed analysis with model ${model_name}"
    done
}

# Process model specification and run appropriate models
if [[ "$MODEL_SPEC" =~ ^[0-9]+$ ]]; then
    # Single model number
    run_models "$MODEL_SPEC" "$MODEL_SPEC"
    
elif [[ "$MODEL_SPEC" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    # Range of models
    start="${BASH_REMATCH[1]}"
    end="${BASH_REMATCH[2]}"
    
    if [ "$start" -gt "$end" ]; then
        echo "ERROR: Invalid range. Start model ($start) is greater than end model ($end)."
        exit 1
    fi
    
    run_models "$start" "$end"
    
else
    echo "ERROR: Invalid model specification: ${MODEL_SPEC}"
    echo "Must be a single number (1-6), or a range (e.g., 1-5)"
    usage
fi

echo "All novobench analyses completed."
