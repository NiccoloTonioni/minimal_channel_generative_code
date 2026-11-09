#!/bin/bash
# Convenience runner for the beta-VAE experiments. Run from the repository root.
# Runs are written to checkpoints/vae/<run_name>, with <run_name> = ld_${latent_dim}_fc_6_sd_${seed}

TRAIN_CONFIG="./configs/base/vae/base_train.gin"
INFERENCE_CONFIG="./configs/base/vae/base_inference.gin"
LATENT_DIMS="2 4 6 8 10 12"

# Training with the base configuration (d_zeta = 4, seed 42)
train() {
    python main_autoencoder.py --gin_config "$TRAIN_CONFIG"
}

# Test-set reconstructions with a trained run (default: the run written by `train`)
inference() {
    local run_dir=${1:-checkpoints/vae/ld_4_fc_6_sd_42}
    python main_autoencoder.py --gin_config "$INFERENCE_CONFIG" \
        --gin_params "main_autoencoder.log_dir_relative='$run_dir'"
}

# Training for every latent dimension of the compression study (Fig. 6, 7)
train_latent_dims() {
    local seed=${1:-42}
    for latent_dim in $LATENT_DIMS; do
        echo "=== Training vae: latent_dim=$latent_dim, seed=$seed"
        python main_autoencoder.py --gin_config "$TRAIN_CONFIG" \
            --gin_params "VAE.latent_dim=$latent_dim" "SEED=$seed"
    done
}

# Test-set reconstructions for every latent dimension (inputs of main_postprocessing.py, Fig. 6, 7)
inference_latent_dims() {
    local seed=${1:-42}
    for latent_dim in $LATENT_DIMS; do
        echo "=== Reconstructing the test set with vae: latent_dim=$latent_dim, seed=$seed"
        python main_autoencoder.py --gin_config "$INFERENCE_CONFIG" \
            --gin_params "VAE.latent_dim=$latent_dim" \
                         "main_autoencoder.log_dir_relative='checkpoints/vae/ld_${latent_dim}_fc_6_sd_${seed}'"
    done
}

# Training with gin overrides
custom() {
    python main_autoencoder.py --gin_config "$TRAIN_CONFIG" --gin_params "$@"
}

usage() {
    echo "Usage: $0 {train|inference [run_dir]|train_latent_dims [seed]|inference_latent_dims [seed]|custom <gin_params>}"
    echo "Examples:"
    echo "  $0 train                                   # d_zeta = 4, seed 42"
    echo "  $0 inference                               # test-set reconstructions of checkpoints/vae/ld_4_fc_6_sd_42"
    echo "  $0 inference path/to/run                   # test-set reconstructions of another run"
    echo "  $0 train_latent_dims                       # d_zeta = 2, 4, ..., 12, seed 42"
    echo "  $0 inference_latent_dims 43                # reconstructions of the seed-43 runs"
    echo "  $0 custom 'VAE.latent_dim=8' 'SEED=1'"
}

case "$1" in
    train) train ;;
    inference) shift; inference "$@" ;;
    train_latent_dims) shift; train_latent_dims "$@" ;;
    inference_latent_dims) shift; inference_latent_dims "$@" ;;
    custom) shift; custom "$@" ;;
    *) usage; exit 1 ;;
esac
