"""
End-to-end predictive framework (Section 4.3 of the paper).

  1. The sensor-conditioned Transformer is initialised with the first `block_size` encoded test
     latents and rolled out autoregressively over the test set, driven by the sensor signals.
  2. The decoder of the beta-VAE-GAN maps every predicted latent state back to a velocity field.

Alternatively, an existing latent rollout (e.g. written by main_transformer.py) is decoded by setting
main_predictive_framework.predicted_latents_file.

Output layout (consumed by main_postprocessing.py):
  <output_dir>/predicted_latents.npy   (T, d_zeta)       predicted latents after the context
  <output_dir>/predicted_fields.npy    (T, 48, 24, 3)    decoded velocity fields, normalised as in Eq. (5)

Usage (from the repository root, after training the beta-VAE-GAN and the conditioned Transformer):
  python main_predictive_framework.py --gin_config configs/base/predictive_framework/base_inference.gin \
      --gin_params "main_predictive_framework.transformer_dir_relative='path/to/transformer_run'" \
                   "main_predictive_framework.autoencoder_dir_relative='path/to/vae_gan_run'"
"""
import os
import gin
import jax
import numpy as np
from src.utils.gin_utils import parse_config_file
from src.models.autoencoders.vae_gan import VAE_GAN
from src.models.transformers.conditioned.transformer_adaLNZero import TimeTransformerAdaLNZero
from src.inference.autoencoders.inference_vae_gan import VAEGANInference
from src.inference.transformers.inference_adaLNZero import AdaLNZeroInference

SENSOR_FILES = ['sensor_origin.npy', 'sensor_zero.npy', 'sensor_minimum.npy']


def rollout_latents(transformer_dir, latent_dir, sensor_dir, rollout_length=None):
    """Autoregressive rollout of the conditioned Transformer; returns the predicted latents (T, d_zeta)."""
    transformer = TimeTransformerAdaLNZero(output_mode='full_sequence')
    context = transformer.block_size

    latents = np.load(os.path.join(latent_dir, 'latents_mean.npy'))
    latents = latents.reshape(latents.shape[0], -1)                                 # (T, d_zeta)
    sensors = np.concatenate([np.load(os.path.join(sensor_dir, f)) for f in SENSOR_FILES], axis=-1)

    max_length = sensors.shape[0] - context
    if rollout_length is None or rollout_length > max_length:
        rollout_length = max_length

    inference_module = AdaLNZeroInference(transformer_model=transformer,
                                          checkpoint_dir=transformer_dir,
                                          c_latent=latents.shape[-1],
                                          c_sensor=sensors.shape[-1])
    print(f'Rollout of {rollout_length} steps from a context of {context} encoded latents ...')
    predicted = inference_module.rollout(latents[:context], sensors, steps_to_predict=rollout_length)
    return np.asarray(predicted[0, context:])                                       # future only


def decode_latents(autoencoder_dir, latents, batch_size):
    """Decode latents (T, d_zeta) into velocity fields (T, 48, 24, 3) with the beta-VAE-GAN."""
    inference_module = VAEGANInference(generator_model=VAE_GAN(), checkpoint_dir=autoencoder_dir)
    latents = latents[:, None, None, :]                                             # (T, 1, 1, d_zeta)
    fields = []
    for i in range(0, latents.shape[0], batch_size):
        fields.append(np.asarray(jax.device_get(inference_module.decode(latents[i:i + batch_size]))))
        print(f'  decoded {min(i + batch_size, latents.shape[0])}/{latents.shape[0]}', end='\r')
    print()
    return np.concatenate(fields, axis=0)


@gin.configurable
def main_predictive_framework(working_directory: str = '.',
                              latent_root_relative: str = 'src/data/datasets/latents/test',
                              sensor_root_relative: str = 'src/data/datasets/sensors/test',
                              transformer_dir_relative: str = None,
                              autoencoder_dir_relative: str = None,
                              output_dir_relative: str = 'results/predictive_framework',
                              rollout_length: int = None,
                              predicted_latents_file: str = None,
                              decoder_batch_size: int = 128):
    """
    Args:
        working_directory: Base working directory (repository root)
        latent_root_relative: Encoded test latents (context of the rollout)
        sensor_root_relative: Sensor signals of the test set
        transformer_dir_relative: Run directory of the conditioned Transformer (contains best_model_params/)
        autoencoder_dir_relative: Run directory of the beta-VAE-GAN (contains best_model_params/)
        output_dir_relative: Output directory
        rollout_length: Number of predicted steps (None: until the end of the test set)
        predicted_latents_file: Existing latent rollout to decode instead of running the Transformer
            (e.g. a predicted_latents.npy written by main_transformer.py)
        decoder_batch_size: Batch size of the decoder
    """
    # Absolute paths (required by orbax)
    working_directory = os.path.abspath(working_directory)
    os.chdir(working_directory)
    output_dir = os.path.join(working_directory, output_dir_relative)
    os.makedirs(output_dir, exist_ok=True)
    if autoencoder_dir_relative is None:
        raise ValueError('Set main_predictive_framework.autoencoder_dir_relative.')
    print('Available devices:', jax.devices())

    # 1. Latent rollout
    if predicted_latents_file is None:
        if transformer_dir_relative is None:
            raise ValueError('Set main_predictive_framework.transformer_dir_relative '
                             '(or predicted_latents_file to decode an existing rollout).')
        predicted_latents = rollout_latents(os.path.join(working_directory, transformer_dir_relative),
                                            os.path.join(working_directory, latent_root_relative),
                                            os.path.join(working_directory, sensor_root_relative),
                                            rollout_length)
        path = os.path.join(output_dir, 'predicted_latents.npy')
        np.save(path, predicted_latents)
        print(f'Saved predicted latents {predicted_latents.shape} to {path}')
        jax.clear_caches()
    else:
        predicted_latents = np.load(os.path.join(working_directory, predicted_latents_file))
        predicted_latents = predicted_latents.reshape(predicted_latents.shape[0], -1)
        print(f'Loaded predicted latents {predicted_latents.shape} from {predicted_latents_file}')

    # 2. Decoding into velocity fields
    print(f'Decoding {predicted_latents.shape[0]} latent states in batches of {decoder_batch_size} ...')
    fields = decode_latents(os.path.join(working_directory, autoencoder_dir_relative),
                            predicted_latents, decoder_batch_size)
    path = os.path.join(output_dir, 'predicted_fields.npy')
    np.save(path, fields)
    print(f'Saved predicted fields {fields.shape} to {path}')
    print('Exiting.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='End-to-end predictive framework with Gin Config')
    parser.add_argument('--gin_config', type=str, required=True,
                        help='Path to gin configuration file(s). Can specify multiple files separated by commas.')
    parser.add_argument('--gin_params', type=str, nargs='*', default=[],
                        help='Override gin parameters from command line. Format: param_name=value')
    args = parser.parse_args()

    for gin_file in args.gin_config.split(','):
        parse_config_file(gin_file.strip())
    if args.gin_params:
        gin.parse_config(args.gin_params)

    main_predictive_framework()
