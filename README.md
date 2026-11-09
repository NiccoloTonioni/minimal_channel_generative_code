# A Hybrid Generative Reduced-Order Model for the Minimal Flow Unit

JAX/Flax implementation of the two-stage generative framework described in:

> **A Hybrid Generative Reduced-Order Model for the Minimal Flow Unit**  
> N. Tonioni, L. Agostini, M. Sanchis-Agudo, M. Umair, F. Kerhervé, L. Cordier, R. Vinuesa  
> *International Journal of Heat and Fluid Flow*, 2026  
> [DOI / arXiv link]

A β-VAE-GAN compresses DNS snapshots of the wall-parallel plane at y<sup>+</sup> = 14 of a Minimal Flow Unit
(Re<sub>τ</sub> = 200) into a four-dimensional latent space. A Transformer conditioned on three velocity probes
(AdaLN-Zero) then forecasts the latent dynamics autoregressively, and the decoder maps the predicted latents back
to velocity fields.

---

## Repository layout

```
├── main_autoencoder.py           # Stage 1: beta-VAE / beta-VAE-GAN training and test-set reconstruction
├── main_transformer.py           # Stage 2: Transformer training and autoregressive latent rollout
├── main_predictive_framework.py  # End-to-end inference: sensor-driven rollout decoded into velocity fields
├── main_postprocessing.py        # Figures and tables of the paper
├── simple_runner_vae.sh          # Convenience runners (training, inference, sweeps)
├── simple_runner_vae_gan.sh
├── simple_runner_transformer.sh
├── configs/base/
│   ├── vae/  vae_gan/            # model_architecture.gin, base_train.gin, base_inference.gin
│   ├── transformer/
│   │   ├── conditioned/          # AdaLN-Zero sensor-conditioned Transformer (used in the paper)
│   │   └── unconditioned/        # Transformer without sensors
│   ├── predictive_framework/     # base_inference.gin
│   └── postprocessing/           # paper_results.gin
├── src/
│   ├── data/
│   │   ├── flow_dataloader.py    # flow-field dataset
│   │   ├── latents_dataloader.py # latent / sensor sequence dataset
│   │   └── datasets/             # sample data (see "Data" below)
│   ├── models/                   # VAE, VAE-GAN, Transformers and attention mechanisms
│   ├── trainers/                 # training loops and schedulers
│   ├── inference/                # reconstruction and rollout modules
│   ├── preprocessing/            # latent and sensor datasets for the Transformer
│   ├── postprocessing/           # metrics, statistics, plotting and loading helpers
│   └── utils/
├── requirements.txt
└── LICENSE
```

Training runs are written to `checkpoints/` and the outputs of the predictive framework and of the
post-processing to `results/`.

---

## Installation

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -U "jax[cuda12]"          # or: pip install -U jax   (CPU only)
pip install -r requirements.txt
```

Training needs a GPU. Inference and post-processing also run on CPU.

---

## Data

The full DNS dataset (two runs of ~17,000 snapshots of size 48 × 24 × 3, about 460 MB) is too large for this
repository and is available at **[dataset link]**. The repository contains:

| Path | Content |
|---|---|
| `src/data/datasets/flow/{train,test}/fields_2d.npy` | **sample**: the first 10 snapshots of each run, `(N, 48, 24, 3)` |
| `src/data/datasets/flow/params/` | normalisation (Eq. 5), friction velocity, grid coordinates |
| `src/data/datasets/latents/{train,validation,test}/` | **full** latent dataset encoded with the d<sub>ζ</sub> = 4 β-VAE-GAN of the paper |
| `src/data/datasets/sensors/{train,validation,test}/` | **full** sensor signals (u′, v′, w′ at three probes) |

The velocity fluctuations are normalised as u′/(3σ<sub>u</sub>) (Eq. 5). The first run is used for training and
validation, the second run is the test set. The autoencoders use a random 90/10 split of the first run; the
latent and sensor sets are split contiguously in time (first 90 % for training, last 10 % for validation).

With the sample data every script runs, so the installation can be checked; the full dataset is needed to
reproduce the results. To use it, replace the two sample files with the full ones (same names), or point
`main_autoencoder.data_root_relative` to another folder.

---

## Workflow

All commands are run from the repository root; paths in the configs are relative to it. Every main script takes
`--gin_config` and optional `--gin_params` overrides, e.g.

```bash
python main_autoencoder.py --gin_config configs/base/vae_gan/base_train.gin \
    --gin_params "VAE_GAN.latent_dim=8" "train_vae_gan.epochs=3000"
```

All hyperparameters are managed by [gin-config](https://github.com/google/gin-config). Each task config
(`base_train.gin`, `base_inference.gin`) includes the corresponding architecture config.

### 1. Autoencoder (β-VAE-GAN)

```bash
./simple_runner_vae_gan.sh train        # d_zeta = 4 -> checkpoints/vae_gan/ld_4_fc_6_gamma_0.001_beta_0.001_sd_42
./simple_runner_vae_gan.sh inference    # test-set reconstructions -> <run_dir>/reconstructions_epoch_<epoch>/
./simple_runner_vae_gan.sh inference path/to/vae_gan_run    # reconstructions of another run
```

`train_latent_dims` and `inference_latent_dims` repeat both steps for d<sub>ζ</sub> = 2, 4, …, 12 (compression
study). The β-VAE baseline works in the same way with `simple_runner_vae.sh`.

### 2. Latent and sensor datasets

Only needed after training a new autoencoder; the distributed latents and sensors were produced in this way with
the d<sub>ζ</sub> = 4 β-VAE-GAN of the paper.

```bash
python -m src.preprocessing.prepare_latent_dataset \
    --vae_gan_dir checkpoints/vae_gan/ld_4_fc_6_gamma_0.001_beta_0.001_sd_42 --latent_dim 4
```

### 3. Sensor-conditioned Transformer

```bash
./simple_runner_transformer.sh train        # EA*: easy-attention, time-weighted loss and input noise
./simple_runner_transformer.sh inference    # rollout over the test set -> <run_dir>/full_rollout_epoch_<epoch>/
./simple_runner_transformer.sh inference path/to/transformer_run    # rollout of another run
```

The rollout is initialised with the first 128 encoded test latents and driven by the sensor signals.
`train_paper_models` and `inference_paper_models` do the same for the five variants of Table 3
(SA, EA, EA\*, SA+EA, SA+EA\*). `MODEL=unconditioned ./simple_runner_transformer.sh train` trains the Transformer
without sensors.

### 4. End-to-end predictive framework

```bash
python main_predictive_framework.py --gin_config configs/base/predictive_framework/base_inference.gin \
    --gin_params "main_predictive_framework.transformer_dir_relative='path/to/transformer_run'" \
                 "main_predictive_framework.autoencoder_dir_relative='path/to/vae_gan_run'"
```

Rolls out the trained EA\* Transformer over the test set and decodes the predicted latents with the β-VAE-GAN into
`results/predictive_framework/{predicted_latents,predicted_fields}.npy`.

### 5. Figures and tables of the paper

```bash
python main_postprocessing.py --gin_config configs/base/postprocessing/paper_results.gin
python main_postprocessing.py --gin_config configs/base/postprocessing/paper_results.gin \
    --gin_params "main_postprocessing.results=['table3', 'fig08_fig09']"
```

Figures are written to `results/figures/` and tables to `results/tables/`. Before running, replace the
`path/to/...` placeholders in `configs/base/postprocessing/paper_results.gin` with the run directories of your
trained models; a result whose inputs are missing is skipped.

| Result | Content | Produced by |
|---|---|---|
| `fig06_fig07` | E<sub>k+</sub>, r<sub>ζ</sub> and two-point correlation vs d<sub>ζ</sub> (POD, β-VAE, β-VAE-GAN) | step 1 for d<sub>ζ</sub> = 2…12 (`*_latent_dims`); POD is computed from the DNS training run |
| `table3` | latent forecasting metrics (NRMSE, NMAE, ρ, d<sub>H</sub>) | step 3, `*_paper_models` |
| `fig08_fig09` | latent time series, joint PDFs, spectra and PDFs of EA\* | step 3 |
| `fig10_fig16` | (−u′v′)<sup>+</sup><sub>max</sub>, u′ at the centre and their spectra | step 4 |
| `fig11_fig12` | low/high-frequency decomposition of ζ<sub>3</sub> and u′ | distributed latents and full DNS test set |
| `fig13` | two-point correlation of u′, v′, w′ | steps 1 and 4 |
| `table4_fig14` | E<sub>k+</sub>, global entropy and singular values | steps 1 and 4 |
| `fig15` | quadrant analysis of the Reynolds shear stress | step 4 |

All results except Table 3 and Figs. 8–9 need the full DNS test set.

---

## Citation

If you use this code, please cite:

```bibtex
@article{tonioni2026hybrid,
  title   = {A Hybrid Generative Reduced-Order Model for the Minimal Flow Unit},
  author  = {Tonioni, Niccol{\`o} and Agostini, Lionel and Sanchis-Agudo, Marcial and Umair, Mohammad and
             Kerherv{\'e}, Franck and Cordier, Laurent and Vinuesa, Ricardo},
  journal = {International Journal of Heat and Fluid Flow},
  year    = {2026},
  doi     = {[DOI]}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
