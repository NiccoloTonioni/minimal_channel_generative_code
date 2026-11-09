"""
Build the latent-space and sensor datasets used to train the Transformer.

Stage 1 -> Stage 2 bridge:
  1. Encodes the DNS snapshots of the train and test runs with a trained VAE-GAN
     (posterior mean and log-variance).
  2. Extracts the wall-parallel velocity probes ("sensors") from the same snapshots.
  3. Splits the training run *contiguously* in time into train / validation
     (first 90% / last 10%), so that both are continuous trajectories.

Output layout (consumed by src/data/latents_dataloader.py):
  <latent_root>/{train,validation,test}/latents_mean.npy     (N, 1, 1, d_zeta)
  <latent_root>/{train,validation,test}/latents_logvar.npy   (N, 1, 1, d_zeta)
  <sensor_root>/{train,validation,test}/sensor_{origin,zero,minimum}.npy   (N, 3)

Usage (from the repository root, after training the beta-VAE-GAN):
  python -m src.preprocessing.prepare_latent_dataset \
      --vae_gan_dir checkpoints/vae_gan/ld_4_fc_6_gamma_0.001_beta_0.001_sd_42 --latent_dim 4
"""
import os
import json
import argparse
import numpy as np
import gin
import jax

from src.models.autoencoders.vae_gan import VAE_GAN
from src.inference.autoencoders.inference_vae_gan import VAEGANInference

# Sensor locations as (streamwise index, spanwise index) on the 48 x 24 plane at y+ = 14.
# 'origin' is the domain centre; 'zero' is the spanwise location where the two-point
# correlation of u' with the origin crosses zero; 'minimum' is where it reaches its minimum.
SENSORS = {
    'origin':  (24, 12),
    'zero':    (24, 19),
    'minimum': (24, 0),
}
TRAIN_FRACTION = 0.9


def encode_fields(inference_module, fields, batch_size):
    """Encode fields (N, H, W, C) in batches; returns posterior mean and log-variance."""
    means, logvars = [], []
    for i in range(0, fields.shape[0], batch_size):
        batch = np.asarray(fields[i:i + batch_size], dtype=np.float32)
        mean, logvar, _ = inference_module.encode(batch)
        means.append(np.asarray(jax.device_get(mean)))
        logvars.append(np.asarray(jax.device_get(logvar)))
        print(f'  encoded {min(i + batch_size, fields.shape[0])}/{fields.shape[0]}', end='\r')
    print()
    return np.concatenate(means, axis=0), np.concatenate(logvars, axis=0)


def save_split(latent_root, sensor_root, split, mean, logvar, fields, model_name):
    """Save latents, sensors and metadata for one split."""
    lat_dir = os.path.join(latent_root, split)
    sen_dir = os.path.join(sensor_root, split)
    os.makedirs(lat_dir, exist_ok=True)
    os.makedirs(sen_dir, exist_ok=True)

    np.save(os.path.join(lat_dir, 'latents_mean.npy'), mean)
    np.save(os.path.join(lat_dir, 'latents_logvar.npy'), logvar)
    with open(os.path.join(lat_dir, 'latents_metadata.json'), 'w') as f:
        json.dump({'model_type': model_name, 'latent_dim': int(mean.shape[-1]),
                   'shape': list(mean.shape), 'n_samples': int(mean.shape[0]),
                   'dtype': str(mean.dtype)}, f, indent=4)

    for name, (ix, iz) in SENSORS.items():
        np.save(os.path.join(sen_dir, f'sensor_{name}.npy'),
                np.asarray(fields[:, ix, iz, :], dtype=np.float32))
    with open(os.path.join(sen_dir, 'sensors_metadata.json'), 'w') as f:
        json.dump({'n_sensors': len(SENSORS),
                   'sensors': [{'name': n, 'grid_idx': list(idx)} for n, idx in SENSORS.items()]},
                  f, indent=4)
    print(f'  [{split}] latents {mean.shape} -> {lat_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--vae_gan_dir', required=True, help='VAE-GAN run directory containing best_model_params/')
    parser.add_argument('--latent_dim', type=int, default=4, help='Latent dimension of the VAE-GAN')
    parser.add_argument('--flow_root', default='src/data/datasets/flow', help='Directory with {train,test}/fields_2d.npy')
    parser.add_argument('--latent_root', default='src/data/datasets/latents', help='Output directory for latents')
    parser.add_argument('--sensor_root', default='src/data/datasets/sensors', help='Output directory for sensors')
    parser.add_argument('--arch_config', default='configs/base/vae_gan/model_architecture.gin')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    gin.parse_config_file(args.arch_config)
    gin.bind_parameter('VAE_GAN.latent_dim', args.latent_dim)

    vae_gan_dir = os.path.abspath(args.vae_gan_dir)
    model_name = os.path.basename(os.path.normpath(vae_gan_dir))
    inference_module = VAEGANInference(generator_model=VAE_GAN(), checkpoint_dir=vae_gan_dir)

    # --- Training run: contiguous train / validation split
    train_fields = np.load(os.path.join(args.flow_root, 'train', 'fields_2d.npy'), mmap_mode='r')
    print(f'Encoding training run {train_fields.shape} ...')
    mean, logvar = encode_fields(inference_module, train_fields, args.batch_size)
    n_train = int(TRAIN_FRACTION * len(mean))
    save_split(args.latent_root, args.sensor_root, 'train',
               mean[:n_train], logvar[:n_train], train_fields[:n_train], model_name)
    save_split(args.latent_root, args.sensor_root, 'validation',
               mean[n_train:], logvar[n_train:], train_fields[n_train:], model_name)

    # --- Test run
    test_fields = np.load(os.path.join(args.flow_root, 'test', 'fields_2d.npy'), mmap_mode='r')
    print(f'Encoding test run {test_fields.shape} ...')
    mean, logvar = encode_fields(inference_module, test_fields, args.batch_size)
    save_split(args.latent_root, args.sensor_root, 'test', mean, logvar, test_fields, model_name)
    print('Done.')


if __name__ == '__main__':
    main()
