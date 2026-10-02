import os
import jax
import gin
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
                    working_directory: str,
                    rollout_window: int=None, 
                    latent_root_relative:str='src/data/datasets/latents',
                    sensor_root_relative:str='src/data/datasets/sensors',
                    log_dir_relative:str='pretrained_models/checkpoints_transformer/',
                    model_type:str='conditioned_adaLN_transformer',
                    checkpoint_dir_relative:str='pretrained_models/checkpoints_transformer/conditioned_adaLN_transformer',
                ):
    """
    Main function to set up and run Conditioned Transformer training.
    
    Args:
        train: Whether to train the model
        working_directory: Base working directory
        latent_root_relative: Relative path to latent data
        sensor_root_relative: Relative path to sensor data
        log_dir_relative: Relative path to log directory
        model_type: Type of model to use
        sensor_names: List of sensor names to load (None loads all sensors)
    """
    # Set working directory
    os.chdir(working_directory)
    
    # Construct absolute paths
    latent_root = os.path.join(working_directory, latent_root_relative)
    sensor_root = os.path.join(working_directory, sensor_root_relative)
    
    # Log directory
    log_dir = os.path.join(working_directory, log_dir_relative, model_type)

    # Print available devices and configuration
    if jax.process_index() == 0:
        print("=" * 70)
        print('Available GPUs:', jax.devices('gpu'))
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
            trainer_unconditioned.train_transformer(
                transformer_model=model,
                trainer_loader=trainer_loader,
                validation_loader=validation_loader,
                log_dir=log_dir,
            )
        elif model_type == 'conditioned_adaLN_transformer':
            trainer_conditioned.train_conditioned_transformer(
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

        chckpt_dir = os.path.join(working_directory, checkpoint_dir_relative)

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
        gin.parse_config_file(gin_file_stripped)
    
    # Parse gin parameters from command line
    if args.gin_params:
        if jax.process_index() == 0:
            print(f"Overriding gin parameters: {args.gin_params}")
        gin.parse_config(args.gin_params)
    
    # Run the main function with gin configuration
    main_transformer()