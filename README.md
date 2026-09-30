# Latent-Space Generative Modelling for Turbulent Channel Flow

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
├── main_autoencoder.py          # Entry point: VAE / VAE-GAN train + inference
├── main_transformer.py          # Entry point: Transformer train + rollout
├── configs/
│   └── base/
│       ├── vae/                 # VAE gin configs
│       ├── vae_gan/             # VAE-GAN gin configs
│       └── transformer/
│           ├── conditioned/     # AdaLN-Zero conditioned transformer configs
│           └── unconditioned/   # TimeTransformer (unconditioned) configs
├── src/
│   ├── data/
│   │   ├── flow_dataloader.py   # Spatial-field dataloader (DNS snapshots)
│   │   └── latents_dataloader.py # Latent-sequence dataloader
│   ├── models/
│   │   ├── autoencoders/        # VAE and VAE-GAN architectures
│   │   └── transformers/
│   │       ├── conditioned/     # TimeTransformerAdaLNZero
│   │       ├── unconditioned/   # TimeTransformer (transformer.py)
│   │       ├── attention.py     # Self, easy and hybrid attention mechanisms
│   │       └── loss_functions.py
│   ├── trainers/
│   │   ├── autoencoders/        # VAE and VAE-GAN trainers
│   │   └── transformers/        # Conditioned and unconditioned trainers
│   ├── inference/
│   │   ├── autoencoders/        # Reconstruction helpers
│   │   └── transformers/        # Rollout helpers (conditioned & unconditioned)
│   └── utils/
│       ├── utils.py             # Collation, device placement, scaling
│       └── loss_functions.py    # Reconstruction and KL losses
├── pretrained_models/           # Distributed checkpoints (VAE-GAN + Transformer)
├── simple_runner_vae.sh         # Convenience runner for VAE experiments
├── simple_runner_vae_gan.sh     # Convenience runner for VAE-GAN experiments
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Main

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
  --gin_config configs/base/transformer/unconditioned/base_inference.gin
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
| `pretrained_models/checkpoints_vae_gan/ld_4_fc_6_alfa_0.001_beta_0.001/` | VAE-GAN, latent dim 4 |
| `pretrained_models/checkpoints_vae_gan/ld_8_fc_6_alfa_0.001_beta_0.001/` | VAE-GAN, latent dim 8 |
| `pretrained_models/checkpoints_transformer/conditioned_adaLN_transformer/bs_128_nl_4_nh_4_ne_64_cd_18/` | AdaLN-Zero transformer |

---

## Monitoring with TensorBoard

```bash
# VAE-GAN training
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