"""
Stage 2: training and autoregressive rollout of the latent-dynamics Transformer.

  1. Training (train=True): the Transformer learns the latent dynamics from the encoded training run,
     split contiguously in time into training / validation (first 90% / last 10%), with scheduled
     sampling and, optionally, the time-weighted loss (Eq. 24) and input-noise injection (Eq. 25).
     Runs are written to <log_dir>/<model_type>/<run_name>, where the run name encodes the architecture,
     attention mechanism, loss, input noise and seed.
  2. Rollout (save_rollout=True): the best checkpoint is initialised with the first `block_size` encoded
     test latents and rolled out autoregressively over the test set (driven by the sensor signals for
     the conditioned model).

Models (main_transformer.model_type):
  conditioned_adaLN_transformer   sensor-conditioned AdaLN-Zero Transformer (used in the paper)
  unconditioned_transformer       Transformer without sensor conditioning

Output layout of the rollout (consumed by main_postprocessing.py):
  <run_dir>/full_rollout_epoch_<epoch>/predicted_latents.npy   (T, d_zeta)  predictions after the context
  <run_dir>/full_rollout_epoch_<epoch>/target_latents.npy      (T, d_zeta)  encoded test latents
  <run_dir>/full_rollout_epoch_<epoch>/initial_latents.npy     (block_size, d_zeta)  context

Usage (from the repository root):
  python main_transformer.py --gin_config configs/base/transformer/conditioned/base_train.gin
  python main_transformer.py --gin_config configs/base/transformer/conditioned/base_inference.gin \
      --gin_params "main_transformer.checkpoint_dir_relative='path/to/transformer_run'"
"""
import os
import jax
import gin
from src.utils.gin_utils import parse_config_file
from src.data.latents_dataloader import latent_data_loader
from src.models.transformers.unconditioned.transformer import TimeTransformer
from src.models.transformers.conditioned.transformer_adaLNZero import TimeTransformerAdaLNZero
from src.trainers.transformers import trainer_unconditioned
from src.trainers.transformers import trainer_conditioned
import src.inference.transformers.inference_adaLNZero as inference_adaLNZero
import src.inference.transformers.inference_unconditioned as inference_unconditioned

@gin.configurable
def main_transformer(train: bool,
                    save_rollout: bool,
                    working_directory: str = '.',
                    rollout_window: int=None, 
                    latent_root_relative:str='src/data/datasets/latents',
                    sensor_root_relative:str='src/data/datasets/sensors',
                    log_dir_relative:str='checkpoints/transformer/',
                    model_type:str='conditioned_adaLN_transformer',
                    checkpoint_dir_relative:str=None,
                ):
    """
    Main function to set up and run Conditioned Transformer training.
    
    Args:
        train: Whether to train the model
        working_directory: Base working directory
        latent_root_relative: Relative path to latent data
        sensor_root_relative: Relative path to sensor data
        save_rollout: Whether to save the autoregressive rollout over the test set
        rollout_window: Length of a shorter rollout window (None: full test set)
        log_dir_relative: Relative path to log directory
        model_type: 'conditioned_adaLN_transformer' or 'unconditioned_transformer'
        checkpoint_dir_relative: Run directory to roll out (None: the run trained in this call)
    """
    # Set working directory (resolved to an absolute path, as required by orbax)
    working_directory = os.path.abspath(working_directory)
    os.chdir(working_directory)
    
    # Construct absolute paths
    latent_root = os.path.join(working_directory, latent_root_relative)
    sensor_root = os.path.join(working_directory, sensor_root_relative)
    
    # Log directory
    log_dir = os.path.join(working_directory, log_dir_relative, model_type)

    # Print available devices and configuration
    if jax.process_index() == 0:
        print("=" * 70)
        print('Available devices:', jax.devices())
        print(f'Total devices: {jax.device_count()}')
        print(f'Current process: {jax.process_index()}')
        print(f'Model selected: {model_type}')
        
    # Initialize the model
    if model_type == 'unconditioned_transformer':
        model = TimeTransformer(
            output_mode='full_sequence' 
        )
        return_sensor = False
    elif model_type == 'conditioned_adaLN_transformer':
        model = TimeTransformerAdaLNZero(
            output_mode='full_sequence' 
        )
        return_sensor = True
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Use 'unconditioned_transformer' for the non-conditioned transformer or "
            f"'conditioned_adaLN_transformer' for AdaLN-Zero conditioning."
        )
    
    # Start training if specified
    if train:
        if jax.process_index() == 0:
            print("=" * 70)
            print(f"Starting conditioned transformer training...")
        
        # Create training dataloader
        trainer_loader = latent_data_loader(
            mode='train',
            root=latent_root,
            sequence_length=model.block_size, 
            shuffle=True,
            drop_last=True,
            return_teacher_forcing=True,
            return_sensor_data=return_sensor,
            sensor_data_directory=sensor_root,
            num_workers=1
        )
        
        # Create validation dataloader
        validation_loader = latent_data_loader(
            mode='validation',
            root=latent_root,
            sequence_length=model.block_size, 
            shuffle=False,
            drop_last=True,
            return_teacher_forcing=True,
            return_sensor_data=return_sensor,
            sensor_data_directory=sensor_root,
            num_workers=1
        )

        # Call conditioned transformer training function
        if model_type == 'unconditioned_transformer':
            trainer = trainer_unconditioned.train_transformer(
                transformer_model=model,
                trainer_loader=trainer_loader,
                validation_loader=validation_loader,
                log_dir=log_dir,
            )
        elif model_type == 'conditioned_adaLN_transformer':
            trainer = trainer_conditioned.train_conditioned_transformer(
                transformer_model=model,
                trainer_loader=trainer_loader,
                validation_loader=validation_loader,
                log_dir=log_dir,
            )        
        else:
            raise ValueError(
                f"Training is not supported for model_type='{model_type}'."
            )
        
        if jax.process_index() == 0:
            print("#" * 70)
            print("Training completed successfully!")
            print("#" * 70)

    if save_rollout:
        if jax.process_index() == 0:
            print("Saving full rollout reconstructions...")

        # Checkpoint to roll out: the given run directory, or the run just trained
        if checkpoint_dir_relative is not None:
            chckpt_dir = os.path.join(working_directory, checkpoint_dir_relative)
        elif train:
            chckpt_dir = trainer.log_dir
        else:
            raise ValueError("Set main_transformer.checkpoint_dir_relative to the run directory to roll out.")
        # The rollout is performed over the test set
        latent_root = os.path.join(latent_root, 'test')
        sensor_root = os.path.join(sensor_root, 'test')

        if model_type == 'conditioned_adaLN_transformer':
            if rollout_window is None:
                inference_adaLNZero.save_full_latent_rollout(
                    transformer_model=model,
                    checkpoint_dir=chckpt_dir,
                    latent_dir=latent_root,
                    sensor_dir=sensor_root,
                    length_initial_context=model.block_size,
                )
            else:
                inference_adaLNZero.save_latent_rollout_window(
                    transformer_model=model,
                    checkpoint_dir=chckpt_dir,
                    latent_dir=latent_root,
                    sensor_dir=sensor_root,
                    length_initial_context=model.block_size,
                    rollout_window=rollout_window,
                )
        elif model_type == 'unconditioned_transformer':
            if rollout_window is None:
                inference_unconditioned.save_full_latent_rollout(
                    transformer_model=model,
                    checkpoint_dir=chckpt_dir,
                    latent_dir=latent_root,
                    length_initial_context=model.block_size,
                )
            else:
                inference_unconditioned.save_latent_rollout_window(
                    transformer_model=model,
                    checkpoint_dir=chckpt_dir,
                    latent_dir=latent_root,
                    length_initial_context=model.block_size,
                    rollout_window=rollout_window,
                )
        else:
            raise ValueError(
                f"save_rollout is not supported for model_type='{model_type}'."
            )
    else:
        if jax.process_index() == 0:
            print("Rollout saving is disabled.")
    
    if jax.process_index() == 0:
        print('Exiting.')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Conditioned Transformer Training with Gin Config'
    )
    parser.add_argument(
        '--gin_config', 
        type=str, 
        required=True,
        help='Path to gin configuration file(s). Can specify multiple files separated by commas.'
    )
    parser.add_argument(
        '--gin_params', 
        type=str, 
        nargs='*', 
        default=[],
        help='Override gin parameters from command line. Format: param_name=value'
    )
    
    args = parser.parse_args()
    
    # Parse gin configuration files
    gin_files = args.gin_config.split(',')
    for gin_file in gin_files:
        gin_file_stripped = gin_file.strip()
        if jax.process_index() == 0:
            print(f"Loading gin config: {gin_file_stripped}")
        parse_config_file(gin_file_stripped)
    
    # Parse gin parameters from command line
    if args.gin_params:
        if jax.process_index() == 0:
            print(f"Overriding gin parameters: {args.gin_params}")
        gin.parse_config(args.gin_params)
    
    # Run the main function with gin configuration
    main_transformer()