# A Hybrid Generative Reduced-Order Model for the Minimal Flow Unit

JAX/Flax implementation of the two-stage generative framework described in:

> **[Paper title]**  
> [Authors]  
> *International Journal of Heat and Fluid Flow*, 2026  
> [DOI / arXiv link]

The pipeline encodes high-fidelity DNS snapshots of a minimal turbulent channel unit into a compact latent space via a VAE or VAE-GAN, then learns temporal dynamics in that latent space with a causal transformer (conditioned on wall-sensor signals or unconditional).

---

## Repository layout

```
Main/
├── main_autoencoder.py                  # Entry point: VAE / VAE-GAN train + inference
├── main_transformer.py                  # Entry point: Transformer train + rollout
├── configs/
│   └── base/
│       ├── vae/                         # VAE gin configs
│       │   ├── model_architecture.gin
│       │   ├── base_train.gin
│       │   └── base_inference.gin
│       ├── vae_gan/                     # VAE-GAN gin configs
│       │   ├── model_architecture.gin
│       │   ├── base_train.gin
│       │   └── base_inference.gin
│       └── transformer/
│           ├── conditioned/             # AdaLN-Zero conditioned transformer configs
│           │   ├── adaLn_transformer_arch.gin
│           │   ├── base_train.gin
│           │   └── base_inference.gin
│           └── unconditioned/           # TimeTransformer (unconditioned) configs
│               ├── transformer_arch.gin
│               ├── base_train.gin
│               └── base_inference_unconditioned.gin
├── src/
│   ├── data/
│   │   ├── flow_dataloader.py           # Spatial-field dataloader (DNS snapshots)
│   │   └── latents_dataloader.py        # Latent-sequence dataloader
│   ├── models/
│   │   ├── autoencoders/
│   │   │   ├── vae.py                   # VAE architecture
│   │   │   └── vae_gan.py               # VAE-GAN architecture
│   │   └── transformers/
│   │       ├── attention.py             # Causal self, easy and self-easy attention
│   │       ├── loss_functions.py        # Teacher-forcing and time-weighted training losses
│   │       ├── conditioned/
│   │       │   └── transformer_adaLNZero.py  # TimeTransformerAdaLNZero
│   │       └── unconditioned/
│   │           └── transformer.py       # TimeTransformer
│   ├── trainers/
│   │   ├── autoencoders/
│   │   │   ├── trainer_vae.py
│   │   │   ├── trainer_vae_gan.py
│   │   │   └── _helper_functions.py     # KL / adversarial weight schedulers
│   │   └── transformers/
│   │       ├── trainer_conditioned.py
│   │       ├── trainer_unconditioned.py
│   │       └── _helper_functions.py     # Curriculum schedulers, sequence truncation
│   ├── inference/
│   │   ├── autoencoders/
│   │   │   ├── inference_vae.py         # Reconstruction + latent encoding (VAE)
│   │   │   └── inference_vae_gan.py     # Reconstruction + latent encoding (VAE-GAN)
│   │   └── transformers/
│   │       ├── inference_adaLNZero.py   # Conditioned autoregressive rollout
│   │       └── inference_unconditioned.py  # Unconditioned autoregressive rollout
│   └── utils/
│       ├── utils.py                     # Collation, device placement, scaling
│       └── loss_functions.py            # Reconstruction, KL and evaluation metrics
├── pretrained_models/
│   ├── checkpoints_vae/                 # VAE, latent dims 2–12
│   ├── checkpoints_vae_gan/             # VAE-GAN, latent dims 2–12
│   └── checkpoints_transformer/         # Conditioned AdaLN-Zero transformer
├── simple_runner_vae.sh                 # Convenience runner for VAE experiments
├── simple_runner_vae_gan.sh             # Convenience runner for VAE-GAN experiments
├── simple_runner_transformer.sh         # Convenience runner for transformer experiments (incl. seed sweep)
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>

# 2. Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install JAX with CUDA 12 support
pip install -U "jax[cuda12]"

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

## Quick start

### Stage 1 — Train the VAE-GAN autoencoder

```bash
# Edit the working_directory in the config first:
# configs/base/vae_gan/base_train.gin  →  main_autoencoder.working_directory = '/your/path'

./simple_runner_vae_gan.sh train
```

### Stage 1 — Generate latent encodings (inference)

```bash
./simple_runner_vae_gan.sh inference
```

### Stage 2 — Train the conditioned transformer

```bash
# Edit working_directory in configs/base/transformer/conditioned/base_train.gin

python main_transformer.py \
  --gin_config configs/base/transformer/conditioned/base_train.gin
```

### Stage 2 — Run a conditioned autoregressive rollout

```bash
python main_transformer.py \
  --gin_config configs/base/transformer/conditioned/base_inference.gin
```

### Stage 2 — Train the unconditioned transformer

```bash
python main_transformer.py \
  --gin_config configs/base/transformer/unconditioned/base_train.gin
```

### Stage 2 — Run an unconditioned autoregressive rollout

```bash
python main_transformer.py \
  --gin_config configs/base/transformer/unconditioned/base_inference_unconditioned.gin
```

### Override any hyperparameter from the command line

```bash
python main_autoencoder.py \
  --gin_config configs/base/vae_gan/base_train.gin \
  --gin_params "VAE_GAN.latent_dim=8" "train_vae_gan.epochs=3000"
```

---

## Configuration system

All hyperparameters are managed by [gin-config](https://github.com/google/gin-config).  
Each config file composes a base architecture file with a task-specific (train/inference) file.

| Config file | Purpose |
|---|---|
| `configs/base/vae/model_architecture.gin` | VAE architecture |
| `configs/base/vae_gan/model_architecture.gin` | VAE-GAN architecture |
| `configs/base/transformer/conditioned/adaLn_transformer_arch.gin` | AdaLN-Zero transformer architecture |
| `configs/base/transformer/unconditioned/transformer_arch.gin` | TimeTransformer architecture |

**You must set `working_directory` to the absolute path of your local clone** in every config you run.

---

## Pretrained models

Pre-trained checkpoints are provided in `pretrained_models/`:

| Path | Description |
|---|---|
| `pretrained_models/checkpoints_vae/ld_{2,4,6,8,10,12}_fc_6/` | VAE, latent dim 2–12 |
| `pretrained_models/checkpoints_vae_gan/ld_{2,4,6,8,10,12}_fc_6_alfa_0.001_beta_0.001/` | VAE-GAN, latent dim 2–12 |
| `pretrained_models/checkpoints_transformer/conditioned_adaLN_transformer/bs_128_nl_4_nh_4_ne_64_cd_18/` | AdaLN-Zero transformer |

---

## Monitoring with TensorBoard

```bash
# VAE / VAE-GAN training
tensorboard --logdir pretrained_models/checkpoints_vae/
tensorboard --logdir pretrained_models/checkpoints_vae_gan/

# Transformer training
tensorboard --logdir pretrained_models/checkpoints_transformer/
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{[citekey],
  title   = {[Title]},
  author  = {[Authors]},
  journal = {International Journal of Heat and Fluid Flow},
  year    = {2026},
  doi     = {[DOI]}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
