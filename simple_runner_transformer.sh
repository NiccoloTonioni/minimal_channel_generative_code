#!/bin/bash
# Convenient scripts for running different experiments

# Full training
# pass two config files: base + model architecture
train() {
    echo "Running full training..."
    python main_transformer.py --gin_config "./configs/base/transformer/conditioned/base_train.gin" 
}

# Inference only (test set)
inference() {
    echo "Running inference on test set..."
    python main_transformer.py --gin_config="./configs/base/transformer/conditioned/base_inference.gin"
}

# Custom run with overrides
custom() {
    echo "Running with custom parameters..."
    python main_transformer.py --gin_config="./configs/base/transformer/conditioned/base_train.gin" \
        --gin_params "$@"
}

# Show usage
usage() {
    echo "Usage: $0 {train|inference|custom|sweep}"
    echo "Examples:"
    echo "  $0 train                    # Full training"
    echo "  $0 custom --gin_params 'main.epochs=500' 'data_loader.batch_size=32' # Custom run with overrides"
}

# Main script logic
case "$1" in
    train)
        train
        ;;
    inference)
        inference
        ;;
    custom)
        shift
        custom "$@"
        ;;
    *)
        usage
        exit 1
        ;;
esac