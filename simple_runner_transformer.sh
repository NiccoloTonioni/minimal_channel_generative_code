#!/bin/bash
# Convenient scripts for running different experiments

# Full training
# pass two config files: base + model architecture
train() {
    echo "Running full training..."
    python main_transformer.py --gin_config "./configs/base/transformer/unconditioned/base_train.gin" 
}

# Inference only (test set)
inference() {
    echo "Running inference on test set..."
    python main_transformer.py --gin_config="./configs/base/transformer/unconditioned/base_inference.gin"
}

# Custom run with overrides
custom() {
    echo "Running with custom parameters..."
    python main_transformer.py --gin_config="./configs/base/transformer/unconditioned/base_train.gin" \
        --gin_params "$@"
}

# Seed sweep example
train_sweep() {
    echo "Running seed sweep..."
    for seed in 40 41 42 43; do
        echo "============================================"
        echo "Training with seed=$seed"
        echo "============================================"
        python main_transformer.py --gin_config="./configs/base/transformer/unconditioned/base_train.gin" \
            --gin_params "SEED=$seed"
    done
}

# Show usage
usage() {
    echo "Usage: $0 {train|inference|train_sweep|custom}"
    echo "Examples:"
    echo "  $0 train                    # Full training"
    echo "  $0 inference                # Inference only"
    echo "  $0 train_sweep              # Seed sweep"
    echo "  $0 custom 'train_transformer.epochs=500' 'latent_data_loader.batch_size=64' # Custom run with overrides"
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
    train_sweep)
        train_sweep
        ;;
    *)
        usage
        exit 1
        ;;
esac