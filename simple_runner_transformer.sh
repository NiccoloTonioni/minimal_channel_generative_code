#!/bin/bash
# Convenience runner for the latent-dynamics Transformer experiments. Run from the repository root.
#
# The model is selected with the MODEL environment variable:
#   MODEL=conditioned   (default) sensor-conditioned AdaLN-Zero Transformer used in the paper
#   MODEL=unconditioned           Transformer without sensor conditioning
#
# Runs are written to checkpoints/transformer/<model_type>/<run_name>; the run name encodes the
# architecture, attention mechanism, training loss, input noise and seed.

set -f   # no filename expansion (the variant names contain "*")
MODEL="${MODEL:-conditioned}"
if [[ "$MODEL" != "conditioned" && "$MODEL" != "unconditioned" ]]; then
    echo "Unknown MODEL='$MODEL'. Use 'conditioned' or 'unconditioned'."
    exit 1
fi
CONFIG_DIR="./configs/base/transformer/${MODEL}"
CONDITIONED_RUN_ROOT="checkpoints/transformer/conditioned_adaLN_transformer"

# Variants of Table 3 (conditioned Transformer): attention mechanism, loss, input noise
#   SA      self-attention
#   EA      easy-attention
#   EA*     easy-attention + time-weighted loss + input noise (base configuration, selected model)
#   SA+EA   parallel self- and easy-attention
#   SA+EA*  parallel self- and easy-attention + time-weighted loss + input noise
PAPER_MODELS="SA EA EA* SA+EA SA+EA*"
variant() {
    case "$1" in
        SA)       echo "self_attention mse False" ;;
        EA)       echo "easy_attention mse False" ;;
        EA\*)     echo "easy_attention tw_mse True" ;;
        SA+EA)    echo "self_easy_attention mse False" ;;
        SA+EA\*)  echo "self_easy_attention tw_mse True" ;;
        *)        echo "Unknown variant $1" >&2; exit 1 ;;
    esac
}
variant_params() {
    read -r attention loss noise <<< "$(variant "$1")"
    echo "TimeTransformerAdaLNZero.attention_type='$attention'" \
         "ConditionedTrainerModuleTransformer.loss_type='$loss'" \
         "ConditionedTrainerModuleTransformer.use_input_noise=$noise"
}
variant_run_dir() {
    read -r attention loss noise <<< "$(variant "$1")"
    local suffix=""
    [[ "$noise" == "True" ]] && suffix="noise_0.01_"
    echo "${CONDITIONED_RUN_ROOT}/bs_128_nl_4_nh_4_ne_64_cd_18_attn_${attention}_loss_${loss}_${suffix}sd_${2:-42}"
}

# Training with the base configuration (conditioned: EA*)
train() {
    python main_transformer.py --gin_config "${CONFIG_DIR}/base_train.gin"
}

# Autoregressive rollout over the test set with a trained run (default: the run written by `train`)
inference() {
    local default_run="$(variant_run_dir 'EA*')"
    [[ "$MODEL" == "unconditioned" ]] && \
        default_run="checkpoints/transformer/unconditioned_transformer/bs_128_nl_4_nh_4_ne_64_attn_easy_attention_loss_mse_sd_42"
    local run_dir=${1:-$default_run}
    python main_transformer.py --gin_config "${CONFIG_DIR}/base_inference.gin" \
        --gin_params "main_transformer.checkpoint_dir_relative='$run_dir'"
}

# Training of the five conditioned variants of Table 3
train_paper_models() {
    local seed=${1:-42}
    for name in $PAPER_MODELS; do
        echo "=== Training $name (seed $seed)"
        python main_transformer.py --gin_config "./configs/base/transformer/conditioned/base_train.gin" \
            --gin_params $(variant_params "$name") "SEED=$seed"
    done
}

# Rollouts over the test set of the five conditioned variants (inputs of main_postprocessing.py, Table 3)
inference_paper_models() {
    local seed=${1:-42}
    for name in $PAPER_MODELS; do
        echo "=== Rollout of $name (seed $seed)"
        python main_transformer.py --gin_config "./configs/base/transformer/conditioned/base_inference.gin" \
            --gin_params $(variant_params "$name") \
                         "main_transformer.checkpoint_dir_relative='$(variant_run_dir "$name" "$seed")'"
    done
}

# Seed sweep with the base configuration
train_sweep() {
    for seed in 40 41 42 43; do
        echo "=== Training with seed=$seed ($MODEL)"
        python main_transformer.py --gin_config "${CONFIG_DIR}/base_train.gin" --gin_params "SEED=$seed"
    done
}

# Training with gin overrides
custom() {
    python main_transformer.py --gin_config "${CONFIG_DIR}/base_train.gin" --gin_params "$@"
}

usage() {
    echo "Usage: [MODEL=conditioned|unconditioned] $0 {train|inference [run_dir]|train_paper_models [seed]|inference_paper_models [seed]|train_sweep|custom <gin_params>}"
    echo "Examples:"
    echo "  $0 train                       # train EA* (conditioned, seed 42)"
    echo "  $0 inference                   # rollout of EA* (seed 42) over the test set"
    echo "  $0 inference path/to/run       # rollout of another run"
    echo "  $0 train_paper_models          # train SA, EA, EA*, SA+EA, SA+EA* (Table 3)"
    echo "  $0 inference_paper_models      # rollouts of the five variants"
    echo "  MODEL=unconditioned $0 train   # Transformer without sensors"
    echo "  $0 custom 'train_conditioned_transformer.epochs=500' 'latent_data_loader.batch_size=64'"
}

case "$1" in
    train)                  train ;;
    inference)              shift; inference "$@" ;;
    train_paper_models)     shift; train_paper_models "$@" ;;
    inference_paper_models) shift; inference_paper_models "$@" ;;
    train_sweep)            train_sweep ;;
    custom)                 shift; custom "$@" ;;
    *)                      usage; exit 1 ;;
esac
