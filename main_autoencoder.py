import os
import jax
import gin
from src.data.flow_dataloader import data_loader
from src.models.autoencoders.vae import VAE
from src.models.autoencoders.vae_gan import VAE_GAN, Discriminator
import src.trainers.autoencoders.trainer_vae as trainer_vae
import src.trainers.autoencoders.trainer_vae_gan as trainer_vae_gan
import src.inference.autoencoders.inference_vae_gan as inference_vae_gan
import src.inference.autoencoders.inference_vae as inference_vae

@gin.configurable
def main_autoencoder(train: bool,
         save_reconstructions: bool,
         working_directory: str,
         data_root_relative='src/data/dataset/flow/scaled_data',
         log_dir_relative='pretrained_models/checkpoints_vae/',
         use_vae_gan: bool = False,):
    """
    Main function to set up and run VAE or VAE-GAN training with JAX SPMD.
    
    Args:
        train: Whether to train the model
        save_reconstructions: Whether to save reconstructions
        working_directory: Base working directory
        data_root_relative: Relative path to data
        log_dir_relative: Relative path to log directory
        use_vae_gan: If True, use VAE-GAN training; if False, use standard VAE
    """
    # Set working directory
    os.chdir(working_directory)
    
    # Construct absolute paths
    data_root = os.path.join(working_directory, data_root_relative)
    log_dir = os.path.join(working_directory, log_dir_relative)

    # Print available devices and configuration
    if jax.process_index() == 0:
        print("=" * 50)
        print('Available GPUs:', jax.devices('gpu'))
        print(f'Total devices: {jax.device_count()}')
        print(f'Current process: {jax.process_index()}')
        print(f'Training mode: {"VAE-GAN" if use_vae_gan else "VAE"}')

    # Initialize the model(s)
    if not use_vae_gan:
        vae_model = VAE()
        if jax.process_index() == 0:
            print('VAE model initialized.')
            print("=" * 50)
    else:
        vae_model = VAE_GAN()
        if jax.process_index() == 0:
            print('VAE-GAN model initialized.')
            print("=" * 50)
        discriminator_model = Discriminator()
    
    # Start training if specified
    if train:
        if jax.process_index() == 0:
            training_type = "VAE-GAN" if use_vae_gan else "VAE"
            print("#" * 50)
            print(f"Starting {training_type} training...")
        
        trainer_loader, validation_loader = data_loader(
            mode='train',
            root=data_root,
            shuffle=True,
            drop_last=True,
            workers=4,
        )
        if use_vae_gan:
            # Call VAE-GAN training function
            trainer_vae_gan.train_vae_gan(generator_model=vae_model,
                                         discriminator_model=discriminator_model,
                                         train_loader=trainer_loader,
                                         val_loader=validation_loader,
                                         log_dir=log_dir)
        else:
            # Call standard VAE training function
            trainer_vae.train_vae(model=vae_model,
                                train_loader=trainer_loader,
                                val_loader=validation_loader,
                                log_dir=log_dir)
        
        if jax.process_index() == 0:
            print("#" * 50)
    else:
        if jax.process_index() == 0:
            print("Training flag is set to False.")
    
    if save_reconstructions:
        if jax.process_index() == 0:
            print("Saving reconstructions...")
        
        # Initialize the data loader for reconstruction
        test_loader = data_loader(
            mode='test',
            root=data_root,
            shuffle=False,
            drop_last=False,
            workers=1,
        )
        
        if use_vae_gan:
            # Call VAE-GAN reconstruction function
            inference_vae_gan.save_vae_gan_reconstructions(generator_model=vae_model,
                                                        checkpoint_dir=log_dir,
                                                        test_loader=test_loader,)  
        else:
            # Call standard VAE reconstruction function
            inference_vae.save_vae_reconstructions(vae_model=vae_model,
                                                checkpoint_dir=log_dir,
                                                test_loader=test_loader,) 
    else:
        if jax.process_index() == 0:
            print("Reconstruction saving is disabled.")
    
    if jax.process_index() == 0:
        print('Exiting.')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VAE/VAE-GAN Training with Gin Config')
    parser.add_argument('--gin_config', type=str, required=True,
                       help='Path to gin configuration file(s). Can specify multiple files separated by commas.')
    parser.add_argument('--gin_params', type=str, nargs='*', default=[],
                       help='Override gin parameters from command line. Format: param_name=value')
    
    args = parser.parse_args()
    
    # Parse gin configuration files
    gin_files = args.gin_config.split(',')
    for gin_file in gin_files:
        gin.parse_config_file(gin_file.strip())
    
    # Parse gin parameters from command line
    if args.gin_params:
        gin.parse_config(args.gin_params)
    
    # Run the main function with gin configuration
    main_autoencoder()