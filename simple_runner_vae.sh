#!/bin/bash
# Convenient scripts for running different experiments

# Full training
train() {
    echo "Running full training..."
    python main_autoencoder.py --gin_config="./configs/base/vae/base_train.gin"
}
# Inference only (test set)
inference() {
    echo "Running inference on test set..."
    python main_autoencoder.py --gin_config="./configs/base/vae/base_inference.gin"
}

# Custom run with overrides
custom() {
    echo "Running with custom parameters..."
    python main_autoencoder.py --gin_config="./configs/base/vae/base_train.gin" \
        --gin_params "$@"
}

# Hyperparameter sweep example
train_sweep() {
    echo "Running hyperparameter sweep..."
    for seed in 40 41 43; do
        for latent_dim in 2 4 6 8 10 12; do
            echo "============================================"
            echo "Training with latent_dim=$latent_dim, seed=$seed"
            echo "============================================"
            python main_autoencoder.py --gin_config="./configs/base/vae/base_train.gin" \
                --gin_params \
                "VAE.latent_dim=$latent_dim" \
                "SEED=$seed" \
                "main_autoencoder.log_dir_relative='pretrained_models/checkpoints_vae_${seed}'"
        done 
    done
}

# Hyperparameter sweep example
inference_sweep() {
    for seed in 40 41 43; do
        for latent_dim in 2 4 6 8 10 12; do
            echo "=============================================================="
            echo "Running reconstructions sweep with latent_dim=$latent_dim, seed=$seed"
            echo "=============================================================="
            python main_autoencoder.py --gin_config="./configs/base/vae/base_inference.gin" \
            --gin_params \
            "VAE.latent_dim=$latent_dim" \
            "main_autoencoder.log_dir_relative='pretrained_models/checkpoints_vae_${seed}/ld_${latent_dim}_fc_6'"
        done
    done
}


# Show usage
usage() {
    echo "Usage: $0 {train|inference|train_sweep|inference_sweep|custom}"
    echo "Examples:"
    echo "  $0 train                    # Full training"
    echo "  $0 inference                # Inference only"
    echo "  $0 train_sweep              # Hyperparameter sweep"
    echo "  $0 inference_sweep          # Inference sweep over latent dimensions"
    echo "  $0 custom --gin_params '@'  # Custom run with overrides"
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
    inference_sweep)
        inference_sweep
        ;;
    *)
        usage
        exit 1
        ;;
esac