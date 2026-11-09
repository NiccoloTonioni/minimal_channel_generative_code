"""
Training loop of the beta-VAE (JAX SPMD): optimisation, validation, checkpointing of the best model and
TensorBoard logging.
"""
import os
import json
import numpy as np
from typing import Any, Tuple, Dict
# -- Flax
from flax.training import train_state
from flax.training import orbax_utils
# -- Orbax for checkpoints
import orbax.checkpoint as ocp
# -- Gin
import gin
# -- JAX
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
# -- Loss functions
from src.models.autoencoders.vae import vae_loss
# -- Optax (Optimizers in JAX)
import optax
# -- Time
import time
# -- Torchvision
import torchvision
# -- Utils
from src.utils.utils import jax_to_torch
# -- Summary Writer
from torch.utils.tensorboard import SummaryWriter
# -- Utils schedulers
from src.trainers.autoencoders._helper_functions import ConstantScheduler, LinearScheduler, CyclicLinearScheduler

#################################################################################################
#                            TENSORBOARD CALLBACKS FOR VISUALIZATION                            #
#################################################################################################
# Function to log reconstructions and other metrics during training
class GenerateCallback:
    """Callback to generate and log reconstructions during training
    Args:
        input_imgs (np.ndarray): Images to reconstruct during training
        every_n_epochs (int): Only save those images every N epochs (otherwise tensorboard gets quite large)"""

    def __init__(self, input_imgs, fixed_key, every_n_epochs=100):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.fixed_key = fixed_key

    def log_generations(self, model, state, logger, epoch):
        if epoch % self.every_n_epochs == 0:
            reconst_imgs, _, _ = model.apply({'params': state.params}, self.fixed_key, self.input_imgs, deterministic=True)
            reconst_imgs = jax.device_get(reconst_imgs)
            in_imgs = jax.device_get(self.input_imgs)
            # Plot and add to tensorboard
            imgs = np.stack([in_imgs, reconst_imgs], axis=1).reshape(-1, *self.input_imgs.shape[1:]) # Stack in axis 1. Then flatten axis 0 and 1
            imgs = jax_to_torch(imgs)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, value_range=(0,1))
            logger.add_image("Reconstructions", grid, global_step=epoch)
            logger.add_image("Reconstructions u", grid[0], global_step=epoch, dataformats='HW')
            logger.add_image("Reconstructions v", grid[1], global_step=epoch, dataformats='HW')
            logger.add_image("Reconstructions w", grid[2], global_step=epoch, dataformats='HW')
        

##################################################################################################
#                            TRAINER MODULE FOR AUTOENCODER TRAINING                             #
##################################################################################################
# Trainer module for autoencoder, which includes model initialization, training, evaluation, and logging.
gin.external_configurable(optax.adam, name="adam")
gin.external_configurable(optax.adamw, name="adamw")
gin.external_configurable(optax.sgd, name="sgd")

@gin.configurable
class TrainerModuleVAE:
    """Optimized Trainer module for VAE with bottleneck fixes"""

    def __init__(self,
                 model,        # Pass the model directly
                 init_batch,   # Initial batch for model initialization
                 mesh=None,    # Enable jax to be aware of MultiGPU
                 # Optimizer params
                 optimizer=optax.adam,
                 optimizer_kwargs={},
                 learning_rate=1e-4,
                 learning_schedule='warmup_cosine_decay',
                 warmup_steps=150,
                 end_value=1e-4,
                 gradient_clip=1.0,
                 weight_decay=0.0,
                 # Loss params
                 beta_schedule='cyclic_linear',
                 beta_start_value=0.0,
                 beta_end_value=1e-4,
                 beta_n_cycles=2,
                 beta_warmup=1000,
                 beta_freeze_epochs=0,
                 # Training params
                 seed=42,
                 check_val_every_n_epochs=20,
                 callbacks_every_n_epochs=100,
                 load_checkpoint=True,
                 # Debug and verify params
                 debug=False,
                 print_model_summary=True,
                 verify_sharding=False,
                 # Logging params
                 log_dir=None,
                 base_log_dir='checkpoints/',
                 save_config=True,
                 # Optimization params
                 async_checkpointing=True,
                 log_every_n_steps=10):
        
        # Mesh for sharding (makes jax aware of multiple devices)
        self.mesh = mesh  

        # Store parameters directly as attributes
        self.log_dir = log_dir
        self.base_log_dir = base_log_dir
        self.seed = seed
        self.debug = debug
        self.async_checkpointing = async_checkpointing
        self.log_every_n_steps = log_every_n_steps
        self.seed = seed

        self.model = model
        self.load_checkpoint = load_checkpoint
        
        # Model optimizer params
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.learning_rate = learning_rate
        self.learning_schedule = learning_schedule
        self.warmup_steps = warmup_steps
        self.end_value = end_value
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay

        # Loss params
        self.beta_schedule = beta_schedule
        self.beta_start_value = beta_start_value
        self.beta_end_value = beta_end_value
        self.beta_n_cycles = beta_n_cycles
        self.beta_warmup = beta_warmup
        self.beta_freeze_epochs = beta_freeze_epochs

        # Training schedule
        self.check_val_every_n_epochs = check_val_every_n_epochs
        self.callbacks_every_n_epochs = callbacks_every_n_epochs
        
        # Set params for model initialization
        self.init_imgs = init_batch[:8]
        self.fixed_key = jax.random.PRNGKey(seed)

        # Call initialization functions
        self._init_logger()
        self._init_model(self.init_imgs)
        self._init_orbax_checkpoint_managers()
        self.create_jitted_functions()

        if jax.process_index() == 0: 
            if print_model_summary:
                self._print_tabulate(init_batch) 
            if verify_sharding:
                self._verify_sharding_setup()  
            if save_config:
                self.save_gin_config()
                
    ##############################################################################
    #                           INITIALIZATION FUNCTIONS                         #
    ##############################################################################
    # Funcitons to initialize the logger, model and optimizer.
    def _init_logger(self):
        # Determine logging directory
        log_dir = self.log_dir
        if log_dir is None:
            base_log_dir = self.base_log_dir
            log_dir = os.path.join(base_log_dir, "VAE")
        
        # Prepare logging
        log_dir = os.path.join(log_dir, f'ld_{self.model.latent_dim}_fc_{self.model.filter_chn}_sd_{self.seed}')
        self.log_dir = log_dir

        if jax.process_index() == 0: 
            # Metrics path
            self.metrics_path = os.path.join(log_dir, 'metrics')
            os.makedirs(self.metrics_path, exist_ok=True)
            # Callbacks for tensorboard
            self.generate_callback = GenerateCallback(
                self.init_imgs,
                fixed_key=self.fixed_key, 
                every_n_epochs=self.callbacks_every_n_epochs, 
            )
            self.logger_training = SummaryWriter(log_dir=self.log_dir+"/training")
            self.logger_recon_train = SummaryWriter(log_dir=self.log_dir+"/recon_training")
            self.logger_kl_train = SummaryWriter(log_dir=self.log_dir+"/kl_training")
            self.logger_validation = SummaryWriter(log_dir=self.log_dir+"/validation")
            self.logger_recon_val = SummaryWriter(log_dir=self.log_dir+"/recon_validation")
            self.logger_kl_val = SummaryWriter(log_dir=self.log_dir+"/kl_validation")
        else:
            self.metrics_path = None
            self.generate_callback = None
            self.logger_training = None
            self.logger_recon_train = None
            self.logger_kl_train = None
            self.logger_validation = None
            self.logger_recon_val = None
            self.logger_kl_val = None

    def _init_model(self, init_imgs):
        """Initialize the model parameters"""
        key = self.fixed_key
        self.key, init_rng = jax.random.split(key)

        # Initialize parameters on the host (CPU) first
        params = self.model.init(init_rng, self.key, init_imgs)['params']

        # If we are using multiple devices, replicate the parameters across all of them
        params_sharding = jax.tree_util.tree_map(lambda x: NamedSharding(self.mesh, P()), params)
        params = jax.device_put(params, params_sharding)

        # Create default state
        self.state = train_state.TrainState(
            step=0,
            apply_fn=self.model.apply,
            params=params,
            tx=None,
            opt_state=None
        )

    def _init_orbax_checkpoint_managers(self):
        """Initialize two separate Orbax CheckpointManagers with optional async support"""
        # We create two separate checkpoint managers:
        #   1. A checkpoint manager for saving only the best model parameters (for inference)
        #   2. A checkpoint manager for saving the full training state (for resuming training)

        # 1. Best model parameters checkpoint manager 
        best_params_dir = os.path.join(self.log_dir, 'best_model_params')
        os.makedirs(best_params_dir, exist_ok=True)
        
        best_params_options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.check_val_every_n_epochs,
            max_to_keep=1,  # Only keep the single best checkpoint
            best_fn=lambda metrics: metrics['chckpt_loss'],
            best_mode='min',
            enable_async_checkpointing=self.async_checkpointing  # NEW: async support
        )
        
        # 2. Full training state checkpoint manager (no best tracking)
        full_state_dir = os.path.join(self.log_dir, 'training_checkpoints')
        os.makedirs(full_state_dir, exist_ok=True)
        
        full_state_options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.check_val_every_n_epochs,
            max_to_keep=2,  # Keep latest 2 checkpoints
            enable_async_checkpointing=self.async_checkpointing  # NEW: async support
        )

        handler = ocp.PyTreeCheckpointer()

        # Create both checkpoint managers
        self.best_params_manager = ocp.CheckpointManager(
            best_params_dir,
            checkpointers=handler,
            options=best_params_options
        )
        
        self.full_state_manager = ocp.CheckpointManager(
            full_state_dir,
            checkpointers=handler,
            options=full_state_options
        )

        if jax.process_index() == 0:
            print(f"Best params CheckpointManager initialized at {best_params_dir}")
            print(f"Full state CheckpointManager initialized at {full_state_dir}")
            if self.async_checkpointing:
                print("Async checkpointing enabled for better performance")


    def _init_optimizer_and_scheduler(self,
                                    num_epochs : int,
                                    num_steps_per_epoch : int):
        """Initializes the optimizer, learning rate and beta scheduler."""    

        # Generator optimizer
        optimizer = self._create_optimizer(
            optimizer=self.optimizer,
            optimizer_kwargs=self.optimizer_kwargs,
            learning_rate=self.learning_rate,
            learning_schedule=self.learning_schedule,
            warmup_steps=self.warmup_steps,
            end_value=self.end_value,
            gradient_clip=self.gradient_clip,
            weight_decay=self.weight_decay,
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch
        )
        
        # Create training states
        self.state = train_state.TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.state.params,
            tx=optimizer
        )
        
        # Initialize beta scheduler
        if self.beta_schedule == 'constant':
            self.beta_scheduler = ConstantScheduler(
                value=self.beta_end_value,
                freeze_epochs=self.beta_freeze_epochs
            )
        elif self.beta_schedule == 'linear':
            self.beta_scheduler = LinearScheduler(
                start_value=self.beta_start_value,
                end_value=self.beta_end_value,
                warmup=self.beta_warmup,
                freeze_epochs=self.beta_freeze_epochs
            )
        elif self.beta_schedule == 'cyclic_linear':
            self.beta_scheduler = CyclicLinearScheduler(
                start_value=self.beta_start_value,
                end_value=self.beta_end_value,
                n_epochs=num_epochs,
                n_cycles=self.beta_n_cycles,
                freeze_epochs=self.beta_freeze_epochs
            )
    
    def _create_optimizer(self, optimizer, optimizer_kwargs, learning_rate, learning_schedule, 
                     warmup_steps, end_value, gradient_clip, weight_decay,
                     num_epochs, num_steps_per_epoch):
        """Create an optimizer with the given parameters

        Args:
            optimizer_kwargs: Dict of additional optimizer parameters (e.g., {'b1': 0.9, 'b2': 0.999} 
                             for AdamW, {'momentum': 0.9, 'nesterov': True} for SGD)
        """
        opt_class = optimizer
        optimizer_kwargs = optimizer_kwargs or {}

        # Build learning rate schedule
        if learning_schedule == 'warmup_cosine_decay':
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=num_steps_per_epoch * num_epochs,
                end_value=end_value
            )
        elif learning_schedule == 'cosine_decay':
            lr_schedule = optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=num_steps_per_epoch * num_epochs,
                alpha=end_value
            )
        else:
            lr_schedule = learning_rate

        # Build transformations
        transformations = [optax.clip_by_global_norm(gradient_clip)]

        # Handle weight decay
        if weight_decay > 0.0:
            # Check if optimizer accepts weight_decay parameter
            import inspect
            sig = inspect.signature(opt_class)

            if 'weight_decay' in sig.parameters:
                # Pass weight_decay directly (e.g., AdamW)
                optimizer = optax.chain(
                    *transformations,
                    opt_class(lr_schedule, weight_decay=weight_decay, **optimizer_kwargs)
                )
            else:
                # Use add_decayed_weights for optimizers without native weight_decay
                transformations.append(optax.add_decayed_weights(weight_decay))
                optimizer = optax.chain(*transformations, opt_class(lr_schedule, **optimizer_kwargs))
        else:
            optimizer = optax.chain(*transformations, opt_class(lr_schedule, **optimizer_kwargs))

        return optimizer
            
    ##############################################################################
    #                          TRAIN/VAL STEPS FUNCTIONS                         #
    ##############################################################################
    # Create jitted training and evaluation step functions
    def create_functions(self):
        """Create training and evaluation step functions with type hints"""
        def train_step(state: train_state.TrainState, 
                       key: jax.Array,
                       x: jnp.ndarray, 
                       beta_loss: float) -> Tuple[train_state.TrainState, float, Tuple[float, float]]:
            loss_fn = lambda params: vae_loss(
                params, 
                key, 
                x, 
                self.model.apply,
                beta_loss
            )
            (loss, budgets), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss, budgets


        def eval_step(state: train_state.TrainState,
                      key: jax.Array, 
                      x: jnp.ndarray,
                      beta_loss: float) -> Tuple[float, Tuple[float, float]]:
            
            return vae_loss(
                state.params,
                key,
                x,
                self.model.apply,
                beta_loss,
                deterministic=True,
            )

        return train_step, eval_step
    
    def create_jitted_functions(self):
        """Create jitted training and evaluation functions"""
        train_step, eval_step = self.create_functions()
        
        if self.debug:
            print('Skipping jitting due to debug=True')
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            # Define sharding for VAE functions
            replicated = NamedSharding(self.mesh, P())
            data_sharded = NamedSharding(self.mesh, P('data'))

            self.train_step = jax.jit(train_step,
                in_shardings=(
                    replicated,        # state 
                    replicated,        # key 
                    data_sharded,      # batch.data
                    replicated         # beta_loss 
                ),
                out_shardings=(
                    replicated,        # state 
                    replicated,        # loss 
                    replicated         # budgets 
                )
            )
            self.eval_step = jax.jit(eval_step,
                in_shardings=(
                    replicated,        # state 
                    replicated,        # key 
                    data_sharded,      # batch.data
                    replicated         # beta_loss
                ),
                out_shardings=(
                    replicated,        # loss 
                    replicated         # budgets 
                )
            )

    ##############################################################################
    #                          TRAIN/VAL EPOCH FUNCTIONS                         #
    ##############################################################################
    def train_epoch(self, trainer_loader, epoch):
        """
        OPTIMIZED: Train VAE for one epoch
        
        Key optimizations:
        1. Online metric accumulation instead of list appending
        2. Single device_get per batch instead of accumulating device arrays
        3. Pre-generated keys for the epoch
        4. Pre-computed beta value
        """
        # Use online averaging to avoid accumulating device arrays
        metrics_accum = {
            'loss': 0.0,
            'recon': 0.0,
            'kl': 0.0
        }

        batch_count = 0
        beta_loss = self.beta_scheduler.get_loss_param(epoch)

        # Iterate over trainer_loader
        for batch in trainer_loader:
            self.key, subkey = jax.random.split(self.key)
            
            # Update state
            self.state, loss, (recon_budget, kl_budget) = self.train_step(
                self.state, subkey, batch, beta_loss
            )
            
            # Accumulate on host 
            metrics_accum['loss'] += loss.mean()
            metrics_accum['recon'] += recon_budget.mean()
            metrics_accum['kl'] += kl_budget.mean()
            batch_count += 1

        final_metrics = jax.device_get(metrics_accum)

        # Compute final averages (all on host, no device operations)
        return (
            float(final_metrics['loss'] / max(batch_count, 1)),
            float(final_metrics['recon'] / max(batch_count, 1)),
            float(final_metrics['kl'] / max(batch_count, 1))
        )

    def eval_model(self, data_loader, epoch):
        """
        Evaluate model on validation set.

        Args:
            data_loader: Iterator (already sharded and prefetched)
            epoch: Current epoch number
        """
        # Initialize online accumulation
        weighted_metrics = {
            'loss': 0.0,
            'recon': 0.0,
            'kl': 0.0,
        }
        total_samples = 0

        # Pre-compute beta and alfa for this epoch
        beta_loss = self.beta_scheduler.get_loss_param(epoch)
        
        # Create a separate eval key
        eval_key = jax.random.PRNGKey(42)

        # Iterate over eval set
        for batch in data_loader:
            current_batch_size = batch.shape[0]
            eval_key, batch_key = jax.random.split(eval_key)

            loss, (recon_budget, kl_budget) = self.eval_step(
                self.state, batch_key, batch, beta_loss
            )

            batch_metrics = jax.device_get({
                'loss': loss,
                'recon': recon_budget,
                'kl': kl_budget,
            })

            for key in weighted_metrics:
                weighted_metrics[key] += float(batch_metrics[key]) * current_batch_size

            total_samples += current_batch_size

        # Avoid division by zero if loader is empty
        if total_samples == 0:
            return 0.0, 0.0, 0.0

        # Compute final weighted averages
        return tuple(weighted_metrics[k] / total_samples for k in ['loss', 'recon', 'kl'])

    ##############################################################################
    #                                TRAINING LOOP                               #
    ##############################################################################
    def train_model(self, trainer_loader, validation_loader, num_epochs=500):
        """
        Main training loop for VAE-GAN

        Args:
            trainer_loader: Iterator for training data (already sharded)
            validation_loader: Iterator for validation data (already sharded)
            num_epochs: Number of training epochs
        """
        # Create optimizer and scheduler
        self._init_optimizer_and_scheduler(num_epochs, len(trainer_loader))
        
        start_epoch = 0
        if self.load_checkpoint:
            if jax.process_index() == 0:
                print('# - Attempting to load model from training checkpoint')
            start_epoch = self.load_training_checkpoint()
        else:
            if jax.process_index() == 0:
                print('Option load_from_checkpoint is set to False, starting from scratch.')

        for epoch_idx in range(start_epoch, num_epochs):
            # Train epoch
            start = time.time()
            loss, train_recon, train_kl = self.train_epoch(
                trainer_loader,  
                epoch=epoch_idx
            )
            stop = time.time()

            # Log only every N steps to reduce I/O overhead
            should_log = (epoch_idx % self.log_every_n_steps == 0)
            current_beta = self.beta_scheduler.get_loss_param(epoch_idx)

            if jax.process_index() == 0 and should_log:
                self.logger_training.add_scalar("loss", loss, global_step=epoch_idx)
                self.logger_recon_train.add_scalar('reconstruction', train_recon, global_step=epoch_idx)
                self.logger_kl_train.add_scalar('kl_divergence', train_kl, global_step=epoch_idx)
                self.logger_training.add_scalar("beta_schedule", current_beta, global_step=epoch_idx)

            if jax.process_index() == 0:
                print(f'Epoch {epoch_idx}: Loss: {loss:.4f},\t'
                      f'Recon: {train_recon:.4f}, KL: {train_kl:.4f},\t'
                      f'Weight KL beta: {current_beta*train_kl:.6f}\t'
                      f'Time: {stop-start:.2f} s')

            if epoch_idx % self.check_val_every_n_epochs == 0 and epoch_idx > self.beta_freeze_epochs:
                eval_loss, eval_recon, eval_kl = self.eval_model(
                    validation_loader,  
                    epoch=epoch_idx
                )

                if jax.process_index() == 0:
                    self.logger_validation.add_scalar('loss', eval_loss, global_step=epoch_idx)
                    self.logger_recon_val.add_scalar('reconstruction', eval_recon, global_step=epoch_idx)
                    self.logger_kl_val.add_scalar('kl_divergence', eval_kl, global_step=epoch_idx)

                    print(f'Epoch {epoch_idx}: Eval loss: {eval_loss:.4f}\t'
                          f'Recon: {eval_recon:.4f}, KL: {eval_kl:.4f}')

                # Save to both checkpoint managers (async if enabled)
                chckpt_loss = eval_recon + self.beta_end_value  * eval_kl # Composite metric for checkpointing
                self._save_checkpoints(epoch_idx, chckpt_loss)

                # Generate reconstructions for visualization
                if jax.process_index() == 0 and self.generate_callback:
                    self.generate_callback.log_generations(
                        self.model, self.state, 
                        logger=self.logger_validation, 
                        epoch=epoch_idx
                    )

                # Flush loggers only at validation time (not every epoch)
                if jax.process_index() == 0:
                    for logger in [self.logger_training, self.logger_validation, 
                                   self.logger_recon_train, self.logger_kl_train,
                                   self.logger_recon_val, self.logger_kl_val]:
                        if logger:
                            logger.flush()

        # At the end of training, wait for all saves to complete
        if self.async_checkpointing:
            self.best_params_manager.wait_until_finished()
            self.full_state_manager.wait_until_finished()
            if jax.process_index() == 0:
                print("All async checkpoint saves completed.")

        self.best_params_manager.close()
        self.full_state_manager.close()
    
    ##############################################################################
    #                                SAVING FUNCTIONS                            #
    ##############################################################################
    def _save_checkpoints(self, epoch_idx, chckpt_loss):
        """
        Save to both checkpoint managers (with optional async support)
        
        With async_checkpointing=True, this will return immediately and
        checkpointing happens in the background
        """
        # We created two separate checkpoint managers, one for saving just the best model parameters
        # and another for saving the full training state. Here we save to both, with the model parameters
        # manager tracking the best chckpt_loss metric.

        # 1. Save best model parameters (only parameters, for inference)
        best_params_data = {
            'vae_params': self.state.params,
            'epoch': epoch_idx
        }
        best_params_save_args = orbax_utils.save_args_from_target(best_params_data)
        
        self.best_params_manager.save(
            epoch_idx,
            best_params_data,
            save_kwargs={'save_args': best_params_save_args},
            metrics={'chckpt_loss': chckpt_loss}
        )
        
        # 2. Save full training state (for resuming training)
        full_state_data = {
            'vae_state': self.state,
            'key': self.key,
            'gin_operative_config': gin.operative_config_str(),
            'epoch': epoch_idx
        }
        full_state_save_args = orbax_utils.save_args_from_target(full_state_data)
        
        self.full_state_manager.save(
            epoch_idx,
            full_state_data,
            save_kwargs={'save_args': full_state_save_args}
        )

    def save_gin_config(self):
        # Save the EXACT gin configuration that was used
        config_dir = os.path.join(self.log_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)
        # Save operative config (what was actually used)
        operative_config_path = os.path.join(config_dir, 'operative_config.gin')
        with open(operative_config_path, 'w') as f:
            f.write(gin.operative_config_str())
        # Save original config files used
        self.save_original_configs(config_dir)
        print(f"Configuration saved to: {config_dir}")

    def save_original_configs(self, config_dir):
        """Save copies of the original gin files used"""
        # Get the list of config files that were parsed
        # (You'll need to track this in your main function)
        if hasattr(self, 'gin_files_used'):
            configs_used_dir = os.path.join(config_dir, 'original_files')
            os.makedirs(configs_used_dir, exist_ok=True)
            
            for gin_file in self.gin_files_used:
                if os.path.exists(gin_file):
                    filename = os.path.basename(gin_file)
                    destination = os.path.join(configs_used_dir, filename)
                    import shutil
                    shutil.copy2(gin_file, destination)
                    
    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """Save metrics to JSON file"""
        file_path = os.path.join(self.metrics_path, f'{filename}.json')

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_metrics = json.load(f)
        else:
            existing_metrics = {}

        for key, value in metrics.items():
            if key not in existing_metrics:
                existing_metrics[key] = []
            existing_metrics[key].append(value)

        with open(file_path, 'w') as f:
            json.dump(existing_metrics, f, indent=4)

    ##############################################################################
    #                               LOADING FUNCTIONS                            #
    ##############################################################################
    def load_metrics(self, filename: str) -> Dict[str, Any]:
        """Load metrics from JSON file"""
        file_path = os.path.join(self.log_dir, f'metrics/{filename}.json')

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}
    
    def load_training_checkpoint(self) -> int:
        """Load full training state for resuming training"""
        target_step = self.full_state_manager.latest_step()
        
        if target_step is None:
            if jax.process_index() == 0:
                print("No training checkpoint found. Starting from scratch.")
            return 0

        if jax.process_index() == 0:
            print(f"Attempting to restore training state from step {target_step}")
        
        restore_target = {
            'vae_state': self.state,
            'key': self.key,
            'gin_operative_config': gin.operative_config_str(),
            'epoch': 0
        }

        restore_args = orbax_utils.restore_args_from_target(restore_target)

        restored_data = self.full_state_manager.restore(
            target_step,
            items=restore_target,
            restore_kwargs={'restore_args': restore_args}
        )

        if restored_data is not None:
            self.state = restored_data['vae_state']
            self.key = restored_data['key']
            start_epoch = restored_data['epoch'] + 1
            
            if jax.process_index() == 0:
                print(f"Restored training checkpoint from epoch {restored_data['epoch']}. Resuming from epoch {start_epoch}.")
                if restored_data.get('gin_operative_config'):
                    self._verify_config_compatibility(restored_data['gin_operative_config'])
            
            return start_epoch
        else:
            if jax.process_index() == 0:
                print("Failed to restore training checkpoint. Starting from scratch.")
            return 0
        
    ##############################################################################
    #                              UTILITY FUNCTIONS                             #
    ##############################################################################

    def _print_tabulate(self, exmp_batch):
        """Print model summary"""
        print("\n=== Model Summary ===")
        print(self.model.tabulate(jax.random.PRNGKey(0), self.key, exmp_batch))
        print("=================================\n")

    def _verify_config_compatibility(self, saved_config):
        """Check if current gin config matches saved config"""
        current_config = gin.operative_config_str()
        
        if saved_config != current_config:
            print("⚠️  WARNING: Current gin config differs from saved checkpoint config!")
            print("This might affect model compatibility.")
            
            # Optionally save a diff
            diff_path = os.path.join(self.log_dir, 'config_diff.txt')
            with open(diff_path, 'w') as f:
                f.write("=== SAVED CONFIG ===\n")
                f.write(saved_config)
                f.write("\n\n=== CURRENT CONFIG ===\n")
                f.write(current_config)
            print(f"Config diff saved to: {diff_path}")
    
    def _verify_sharding_setup(self):
        """Simple verification that sharding is set up correctly"""
        print("\n=== Sharding Verification ===")

        # Check model parameters are replicated
        sample_param = jax.tree_util.tree_leaves(self.state.params)[0]
        is_replicated = (hasattr(sample_param, 'sharding') and 
                        sample_param.sharding.spec == P())

        print(f"✓ Model parameters: {'REPLICATED' if is_replicated else 'UNKNOWN'}")
        print(f"  - Same model weights on all {len(self.mesh.devices)} devices")

        # Test data sharding
        dummy_data = jnp.ones((32, 48, 24, 3))  # Example batch
        sharded_data = jax.device_put(dummy_data, NamedSharding(self.mesh, P('data')))

        expected_per_device_size = dummy_data.shape[0] // len(self.mesh.devices)
        actual_per_device_size = sharded_data.shape[0]

        is_data_sharded = (actual_per_device_size == expected_per_device_size)

        print(f"✓ Data batches: {'SHARDED' if is_data_sharded else 'UNKNOWN'}")
        print(f"  - Original batch size: {dummy_data.shape[0]}")
        print(f"  - Per-device batch size: {actual_per_device_size}")

        print("============================\n")

##################################################################################################
# Main training function to be called from the command line or other scripts
##################################################################################################
@gin.configurable
def train_vae(model=None,
            train_loader=None,
            val_loader=None,
            epochs=20000,
            log_dir=None):
    """
    Main training function for VAE
    
    Args:
        model: VAE model instance
        train_loader: Training dataloader (Iterator)
        val_loader: Validation dataloader (Iterator)
        epochs: Number of training epochs
        log_dir: Directory for logs and checkpoints
        mesh: JAX Mesh for SPMD (if None, will create default)
    """
    if jax.process_index() == 0:
        print("=" * 50)
        print(f"VAE Training Configuration")
        print("=" * 50)
        print(f"  model: {type(model).__name__}")
        print(f"  epochs: {epochs}")
        print(f"  log_dir: {log_dir if log_dir else 'default'}")
        print("=====================================")

    if model is None:
        raise ValueError("Model must be provided for training.")
    if train_loader is None or val_loader is None:
        raise ValueError("Both train_loader and val_loader must be provided for training.")
    
    # Create device mesh also if sharding is not used, for loading/saving
    # if we don't do this model trained with sharding cannot be loaded without sharding
    devices = jax.local_devices()
    mesh = Mesh(devices, ('data',))
    
    # Batch for initialization from validation loader
    init_batch = next(iter(val_loader))
    
    # Create trainer with the configured model
    trainer = TrainerModuleVAE(
        model=model,  # Pass the model directly
        init_batch=init_batch,
        log_dir=log_dir,
        mesh=mesh
    )
    
    if jax.process_index() == 0:
        print('# ----- Training the model')

    # Start training
    trainer.train_model(
        trainer_loader=train_loader,
        validation_loader=val_loader,
        num_epochs=epochs
    )

    return trainer
