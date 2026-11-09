"""
Training loop of the beta-VAE-GAN (JAX SPMD): alternating generator / discriminator optimisation,
scheduled beta and gamma weights, validation, checkpointing of the best model and TensorBoard logging.
"""
import os
import numpy as np
from typing import Any, Tuple
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
# -- Loss functions
from src.models.autoencoders.vae_gan import vae_loss, discriminator_loss

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
        

#################################################################################################
#                            EXTENDED TRAIN STATE FOR DISCRIMINATOR                             #
#################################################################################################
class TrainStateDisc(train_state.TrainState):
    """Extended TrainState to include batch statistics for discriminator"""
    batch_stats: Any = None
    @classmethod
    def create(cls, *, apply_fn, params, tx, batch_stats=None, **kwargs):
        """Create a new TrainStateDisc with batch_stats."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            batch_stats=batch_stats,
            **kwargs
        )

##################################################################################################
#                            TRAINER MODULE FOR VAE-GAN TRAINING                                #
##################################################################################################
# Trainer module for autoencoder, which includes model initialization, training, evaluation, and logging.
gin.external_configurable(optax.adam, name="adam")
gin.external_configurable(optax.adamw, name="adamw")
gin.external_configurable(optax.sgd, name="sgd")

@gin.configurable
class TrainerModuleVAEGAN:
    """Optimized Trainer module for VAE-GAN with bottleneck fixes"""

    def __init__(self,
                 generator_model,      
                 discriminator_model,  # Discriminator model
                 init_batch: jnp.array, # Initial batch for model initialization
                 mesh: None,              # Enable jax to be aware of MultiGPU
                 # Generator optimizer params
                 generator_optimizer=optax.adam,
                 generator_optimizer_kwargs={},
                 generator_learning_rate=1e-4,
                 generator_learning_schedule='warmup_cosine_decay',
                 generator_warmup_steps=150,
                 generator_end_value=1e-4,
                 generator_gradient_clip=1.0,
                 generator_weight_decay=0.0,
                 # Discriminator optimizer params
                 discriminator_optimizer=optax.adam,
                 discriminator_optimizer_kwargs={},
                 discriminator_learning_rate=1e-4,
                 discriminator_learning_schedule='warmup_cosine_decay',
                 discriminator_warmup_steps=150,
                 discriminator_end_value=1e-4,
                 discriminator_gradient_clip=1.0,
                 discriminator_weight_decay=0.0,
                 # Loss params
                 beta_schedule='linear',
                 beta_start_value=0.0,
                 beta_end_value=1e-4,
                 beta_n_cycles=2,
                 beta_warmup=1000,
                 beta_freeze_epochs=0,
                 gamma_schedule='linear',
                 gamma_start_value=0.0,
                 gamma_end_value=1e-4,
                 gamma_n_cycles=2,
                 gamma_warmup=1000,
                 gamma_freeze_epochs=0,
                 # Training schedule params
                 update_gen_every_n_steps=1,
                 update_disc_every_n_steps=1,
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
                 log_every_n_steps=10):  # New: reduce logging frequency
        
        # Mesh for sharding (makes jax aware of multiple devices)
        self.mesh = mesh or Mesh(jax.local_devices(), ('data',))

        # Store parameters directly as attributes
        self.log_dir = log_dir
        self.base_log_dir = base_log_dir
        self.seed = seed
        self.debug = debug
        self.async_checkpointing = async_checkpointing
        self.log_every_n_steps = log_every_n_steps
        self.seed = seed

        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.load_checkpoint = load_checkpoint
        
        # Generator optimizer params
        self.generator_optimizer = generator_optimizer
        self.generator_optimizer_kwargs = generator_optimizer_kwargs
        self.generator_learning_rate = generator_learning_rate
        self.generator_learning_schedule = generator_learning_schedule
        self.generator_warmup_steps = generator_warmup_steps
        self.generator_end_value = generator_end_value
        self.generator_gradient_clip = generator_gradient_clip
        self.generator_weight_decay = generator_weight_decay
        
        # Discriminator optimizer params
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_optimizer_kwargs = discriminator_optimizer_kwargs
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_learning_schedule = discriminator_learning_schedule
        self.discriminator_warmup_steps = discriminator_warmup_steps
        self.discriminator_end_value = discriminator_end_value
        self.discriminator_gradient_clip = discriminator_gradient_clip
        self.discriminator_weight_decay = discriminator_weight_decay
        
        # Loss params
        self.beta_schedule = beta_schedule
        self.beta_start_value = beta_start_value
        self.beta_end_value = beta_end_value
        self.beta_n_cycles = beta_n_cycles
        self.beta_warmup = beta_warmup
        self.beta_freeze_epochs = beta_freeze_epochs
        
        self.gamma_schedule = gamma_schedule
        self.gamma_start_value = gamma_start_value
        self.gamma_end_value = gamma_end_value
        self.gamma_n_cycles = gamma_n_cycles
        self.gamma_warmup = gamma_warmup
        self.gamma_freeze_epochs = gamma_freeze_epochs
        
        # Training schedule
        self.update_gen_every_n_steps = update_gen_every_n_steps
        self.update_disc_every_n_steps = update_disc_every_n_steps
        
        self.check_val_every_n_epochs = check_val_every_n_epochs
        self.callbacks_every_n_epochs = callbacks_every_n_epochs
            
        # Set params for model initialization
        self.init_imgs = init_batch[:8]
        self.fixed_key = jax.random.PRNGKey(seed)

        # Call initialization functions
        self._init_logger()
        self._init_models(self.init_imgs)
        # self._init_orbax_checkpoint_managers()
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
    def _init_logger(self):
        """Initialize logging directories and TensorBoard loggers"""
        # Determine logging directory
        log_dir = self.log_dir
        if log_dir is None:
            base_log_dir = self.base_log_dir
            log_dir = os.path.join(base_log_dir, "VAE_GAN")

        # Prepare logging with model parameters
        log_dir = os.path.join(log_dir, f'ld_{self.generator_model.latent_dim}_fc_{self.generator_model.filter_chn}_gamma_{self.gamma_end_value}_beta_{self.beta_end_value}_sd_{self.seed}')
        self.log_dir = log_dir

        # Only process 0 creates loggers
        if jax.process_index() == 0: 
            # Callbacks for tensorboard
            self.generate_callback = GenerateCallback(
                self.init_imgs,
                fixed_key=self.fixed_key,
                every_n_epochs=self.callbacks_every_n_epochs, 
            )

            # Extended logging for VAE-GAN
            self.logger_training = SummaryWriter(log_dir=self.log_dir+"/training")
            self.logger_recon_train = SummaryWriter(log_dir=self.log_dir+"/recon_training")
            self.logger_kl_train = SummaryWriter(log_dir=self.log_dir+"/kl_training")
            self.logger_adv_train = SummaryWriter(log_dir=self.log_dir+"/adv_training")
            self.logger_disc_train = SummaryWriter(log_dir=self.log_dir+"/disc_training")
            self.logger_validation = SummaryWriter(log_dir=self.log_dir+"/validation")
            self.logger_recon_val = SummaryWriter(log_dir=self.log_dir+"/recon_validation")
            self.logger_kl_val = SummaryWriter(log_dir=self.log_dir+"/kl_validation")
            self.logger_adv_val = SummaryWriter(log_dir=self.log_dir+"/adv_validation")
        else:
            # Other processes set everything to None
            self.metrics_path = None
            self.generate_callback = None
            self.logger_training = None
            self.logger_recon_train = None
            self.logger_kl_train = None
            self.logger_adv_train = None
            self.logger_disc_train = None
            self.logger_validation = None
            self.logger_recon_val = None
            self.logger_kl_val = None
            self.logger_adv_val = None

    def _init_models(self, init_imgs):
        """Initialize both generator and discriminator parameters"""
        key = self.fixed_key
        self.key, gen_key, disc_key = jax.random.split(key, 3)

        # Initialize generator (VAE) parameters
        gen_params = self.generator_model.init(gen_key, self.key, init_imgs)['params']

        # Initialize discriminator parameters
        disc_variables = self.discriminator_model.init(disc_key, init_imgs, is_train=True)

        # Handle sharding if using multiple devices
        params_sharding = jax.tree_util.tree_map(
            lambda x: NamedSharding(self.mesh, P()), gen_params
        )
        gen_params = jax.device_put(gen_params, params_sharding)
        
        disc_params_sharding = jax.tree_util.tree_map(
            lambda x: NamedSharding(self.mesh, P()), disc_variables['params']
        )
        disc_params = jax.device_put(disc_variables['params'], disc_params_sharding)
        
        if 'batch_stats' in disc_variables:
            disc_batch_stats_sharding = jax.tree_util.tree_map(
                lambda x: NamedSharding(self.mesh, P()), disc_variables['batch_stats']
            )
            disc_batch_stats = jax.device_put(disc_variables['batch_stats'], disc_batch_stats_sharding)
        else:
            disc_batch_stats = None

        # Create default states
        self.gen_state = train_state.TrainState(
            step=0,
            apply_fn=self.generator_model.apply,
            params=gen_params,
            tx=None,
            opt_state=None
        )

        self.disc_state = TrainStateDisc(
            step=0,
            apply_fn=self.discriminator_model.apply,
            params=disc_params,
            batch_stats=disc_batch_stats,
            tx=None,
            opt_state=None
        )

    def _init_orbax_checkpoint_managers(self):
        """Initialize two separate Orbax CheckpointManagers with optional async support"""
        # We create two separate checkpoint managers:
        #   1. A checkpoint manager for saving only the best model parameters (for inference)
        #   2. A checkpoint manager for saving the full training state (for resuming training)
        best_params_dir = os.path.join(self.log_dir, 'best_model_params')
        full_state_dir = os.path.join(self.log_dir, 'training_checkpoints')      
        
        # 1. Best model parameters checkpoint manager 
        best_params_options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.check_val_every_n_epochs,
            max_to_keep=1,
            best_fn=lambda metrics: metrics['chckpt_loss'],
            best_mode='min',
            enable_async_checkpointing=self.async_checkpointing  
        )
        # 2. Full training state checkpoint manager (no best tracking)    
        full_state_options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.check_val_every_n_epochs,
            max_to_keep=2,
            enable_async_checkpointing=self.async_checkpointing  
        )
    
        handler = ocp.PyTreeCheckpointer()
    
        # Set-up checkpoint menager call
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

        # 1. Template for Best Params
        best_params_template = {
            'gen_params': self.gen_state.params,
            'epoch': 0  # scalar placeholder
        }
        self._best_params_save_args = orbax_utils.save_args_from_target(best_params_template)

        # 2. Template for Full State
        full_state_template = {
            'gen_state': self.gen_state,
            'disc_state': self.disc_state,
            'key': self.key,
            'gin_operative_config': "placeholder_string", 
            'epoch': 0
        }
        self._full_state_save_args = orbax_utils.save_args_from_target(full_state_template)

        if jax.process_index() == 0:
            print(f"Best params CheckpointManager initialized at {best_params_dir}")
            print(f"Full state CheckpointManager initialized at {full_state_dir}")
            if self.async_checkpointing:
                print("Async checkpointing enabled for better performance")
    

    def _init_optimizers_and_schedulers(self, num_epochs: int, num_steps_per_epoch: int):
        """Initialize optimizers and schedulers for both generator and discriminator"""
        
        # Generator optimizer
        gen_optimizer = self._create_optimizer(
            optimizer=self.generator_optimizer,
            optimizer_kwargs=self.generator_optimizer_kwargs,
            learning_rate=self.generator_learning_rate,
            learning_schedule=self.generator_learning_schedule,
            warmup_steps=self.generator_warmup_steps,
            end_value=self.generator_end_value,
            gradient_clip=self.generator_gradient_clip,
            weight_decay=self.generator_weight_decay,
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch
        )
        
        # Discriminator optimizer
        disc_optimizer = self._create_optimizer(
            optimizer=self.discriminator_optimizer,
            optimizer_kwargs=self.discriminator_optimizer_kwargs,
            learning_rate=self.discriminator_learning_rate,
            learning_schedule=self.discriminator_learning_schedule,
            warmup_steps=self.discriminator_warmup_steps,
            end_value=self.discriminator_end_value,
            gradient_clip=self.discriminator_gradient_clip,
            weight_decay=self.discriminator_weight_decay,
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch
        )
        
        # Create training states
        self.gen_state = train_state.TrainState.create(
            apply_fn=self.gen_state.apply_fn,
            params=self.gen_state.params,
            tx=gen_optimizer
        )
        
        self.disc_state = TrainStateDisc.create(
            apply_fn=self.disc_state.apply_fn,
            params=self.disc_state.params,
            batch_stats=self.disc_state.batch_stats,
            tx=disc_optimizer
        )
        
        # Initialize beta scheduler
        if self.beta_schedule == 'linear':
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
        
        # Initialize gamma scheduler
        if self.gamma_schedule == 'constant':
            self.gamma_scheduler = ConstantScheduler(
                value=self.gamma_end_value
            )
            
        if self.gamma_schedule == 'linear':
            self.gamma_scheduler = LinearScheduler(
                start_value=self.gamma_start_value,
                end_value=self.gamma_end_value,
                warmup=self.gamma_warmup,
                freeze_epochs=self.gamma_freeze_epochs
            )
            
        elif self.gamma_schedule == 'cyclic_linear':
            self.gamma_scheduler = CyclicLinearScheduler(
                start_value=self.gamma_start_value,
                end_value=self.gamma_end_value,
                n_epochs=num_epochs,
                n_cycles=self.gamma_n_cycles,
                freeze_epochs=self.gamma_freeze_epochs
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
    def create_functions(self):
        """Create training and evaluation step functions for VAE-GAN"""
        
        # In generator_step function
        def generator_step(gen_state: train_state.TrainState,
                          disc_state: TrainStateDisc,
                          key: jax.Array,
                          in_imgs: jnp.ndarray, 
                          beta_loss: float,
                          gamma_adv: float) -> Tuple[train_state.TrainState, float, Tuple[float, float, float]]:

            loss_fn = lambda params: vae_loss(
                params,
                disc_state.params,
                disc_state.batch_stats,  # This could be None
                key,
                in_imgs,
                self.generator_model.apply,
                self.discriminator_model.apply,
                beta_loss,
                gamma_adv
            )
            (loss, budgets), grads = jax.value_and_grad(loss_fn, has_aux=True)(gen_state.params)
            gen_state = gen_state.apply_gradients(grads=grads)
            return gen_state, loss, budgets

        def discriminator_step(disc_state: TrainStateDisc,
                     gen_state: train_state.TrainState,
                     key: jax.Array,
                     in_imgs: jnp.ndarray) -> Tuple[TrainStateDisc, float, Any]:
            def loss_fn(params, batch_stats):
                return discriminator_loss(
                    params,
                    batch_stats,
                    gen_state.params,
                    key,
                    in_imgs,
                    self.discriminator_model.apply,
                    self.generator_model.apply
                )

            (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                disc_state.params, disc_state.batch_stats
            )

            # Update discriminator state with new batch stats
            disc_state = disc_state.apply_gradients(grads=grads)
            disc_state = disc_state.replace(batch_stats=new_batch_stats)

            return disc_state, loss, new_batch_stats

        def eval_step(gen_state, disc_state, key, in_imgs, beta_loss, gamma_adv):
            return vae_loss(
                gen_state.params,
                disc_state.params,
                disc_state.batch_stats,
                key,
                in_imgs,
                self.generator_model.apply,
                self.discriminator_model.apply,
                beta_loss,
                gamma_adv,
                deterministic=True,
            )

        return generator_step, discriminator_step, eval_step
    
    def create_jitted_functions(self):
        """Create jitted training and evaluation functions"""
        generator_step, discriminator_step, eval_step = self.create_functions()
        
        if self.debug:
            print('Skipping jitting due to debug=True')
            self.generator_step = generator_step
            self.discriminator_step = discriminator_step
            self.eval_step = eval_step
        else:
            # Define sharding for VAE-GAN functions
            replicated = NamedSharding(self.mesh, P())
            data_sharded = NamedSharding(self.mesh, P('data'))    
            self.generator_step = jax.jit(generator_step,
                in_shardings=(
                    replicated,    # gen_state
                    replicated,    # disc_state
                    replicated,    # key
                    data_sharded,  # batch.data
                    replicated,    # beta_loss
                    replicated     # gamma_adv
                ),
                out_shardings=(
                    replicated,    # gen_state
                    replicated,    # loss
                    replicated     # budgets
                )
            )
            self.discriminator_step = jax.jit(discriminator_step,
                in_shardings=(
                    replicated,    # disc_state
                    replicated,    # gen_state
                    replicated,    # key
                    data_sharded   # batch.data
                ),
                out_shardings=(
                    replicated,    # disc_state
                    replicated,    # loss
                    replicated     # new_batch_stats
                )
            )
            self.eval_step = jax.jit(eval_step,
                in_shardings=(
                    replicated,    # gen_state
                    replicated,    # disc_state
                    replicated,    # key
                    data_sharded,  # batch.data
                    replicated,    # beta_loss
                    replicated     # gamma_adv
                ),
                out_shardings=(
                    replicated,    # loss
                    replicated     # budgets
                )
            )


    ##############################################################################
    #                          TRAIN/VAL EPOCH FUNCTIONS                         #
    ##############################################################################
    def train_epoch(self, trainer_loader, epoch):
        """
        Train for one epoch with optimized metric accumulation.

        Args:
            trainer_loader: Iterator (already sharded and prefetched)
            epoch: Current epoch number
        """
        # Initialize metrics as JAX arrays (on device)
        metrics_accum = {
            'gen_loss': jnp.array(0.0),
            'disc_loss': jnp.array(0.0),
            'recon': jnp.array(0.0),
            'kl': jnp.array(0.0),
            'adv': jnp.array(0.0)
        }

        gen_count = 0
        disc_count = 0
        beta_loss = self.beta_scheduler.get_loss_param(epoch)
        gamma_adv = self.gamma_scheduler.get_loss_param(epoch)

        # Iterate over trainer_loader
        for batch in trainer_loader:
            self.key, disc_key, gen_key = jax.random.split(self.key, 3)

            # --- Discriminator Update ---
            if epoch % self.update_disc_every_n_steps == 0:
                self.disc_state, disc_loss, _ = self.discriminator_step(
                    self.disc_state, self.gen_state, disc_key, batch
                )
                metrics_accum['disc_loss'] += disc_loss.mean() 
                disc_count += 1

            # --- Generator Update ---
            if epoch % self.update_gen_every_n_steps == 0:
                self.gen_state, gen_loss, (recon, kl, adv) = self.generator_step(
                    self.gen_state, self.disc_state, gen_key, batch, beta_loss, gamma_adv
                )

                metrics_accum['gen_loss'] += gen_loss.mean()
                metrics_accum['recon'] += recon.mean()
                metrics_accum['kl'] += kl.mean()
                metrics_accum['adv'] += adv.mean()
                gen_count += 1

        # END OF EPOCH: Single synchronization point
        # Transfer all metrics to CPU at once
        final_metrics = jax.device_get(metrics_accum)

        # Normalize and return
        return (
            float(final_metrics['gen_loss']) / max(gen_count, 1),
            float(final_metrics['disc_loss']) / max(disc_count, 1),
            float(final_metrics['recon']) / max(gen_count, 1),
            float(final_metrics['kl']) / max(gen_count, 1),
            float(final_metrics['adv']) / max(gen_count, 1)
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
            'adv': 0.0
        }
        total_samples = 0

        # Pre-compute beta and gamma for this epoch
        beta_loss = self.beta_scheduler.get_loss_param(epoch)
        gamma_adv = self.gamma_scheduler.get_loss_param(epoch)

        # Create a separate eval key
        eval_key = jax.random.PRNGKey(42)

        # Iterate over eval set
        for batch in data_loader:
            current_batch_size = batch.shape[0]
            eval_key, batch_key = jax.random.split(eval_key)

            loss, (recon_budget, kl_budget, adv_budget) = self.eval_step(
                self.gen_state, self.disc_state, batch_key, batch, beta_loss, gamma_adv
            )

            batch_metrics = jax.device_get({
                'loss': loss,
                'recon': recon_budget,
                'kl': kl_budget,
                'adv': adv_budget
            })

            for key in weighted_metrics:
                weighted_metrics[key] += float(batch_metrics[key]) * current_batch_size

            total_samples += current_batch_size

        # Avoid division by zero if loader is empty
        if total_samples == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Compute final weighted averages
        return tuple(weighted_metrics[k] / total_samples for k in ['loss', 'recon', 'kl', 'adv'])

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
        # Create optimizers and schedulers
        self._init_optimizers_and_schedulers(num_epochs, len(trainer_loader))
        # Initialize checkpoint managers
        self._init_orbax_checkpoint_managers()
        # Attempt to load from checkpoint if specified
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
            gen_loss, disc_loss, train_recon, train_kl, train_adv = self.train_epoch(
                trainer_loader,  
                epoch=epoch_idx
            )
            stop = time.time()

            # Log only every N steps to reduce I/O overhead
            should_log = (epoch_idx % self.log_every_n_steps == 0)

            if jax.process_index() == 0 and should_log:
                self.logger_training.add_scalar("generator_loss", gen_loss, global_step=epoch_idx)
                self.logger_training.add_scalar("discriminator_loss", disc_loss, global_step=epoch_idx)
                self.logger_recon_train.add_scalar('reconstruction', train_recon, global_step=epoch_idx)
                self.logger_kl_train.add_scalar('kl_divergence', train_kl, global_step=epoch_idx)
                self.logger_adv_train.add_scalar('adv_loss', train_adv, global_step=epoch_idx)
                self.logger_disc_train.add_scalar('loss', disc_loss, global_step=epoch_idx) 
                # Log current scheduler values
                current_beta = self.beta_scheduler.get_loss_param(epoch_idx)
                current_gamma = self.gamma_scheduler.get_loss_param(epoch_idx)
                self.logger_training.add_scalar("beta_schedule", current_beta, global_step=epoch_idx)
                self.logger_training.add_scalar("gamma_schedule", current_gamma, global_step=epoch_idx)

            # Always print (printing is cheap)
            if jax.process_index() == 0:
                if disc_loss != 0.0:
                    print(f'Epoch {epoch_idx}: Gen loss: {gen_loss:.4f}, Disc loss: {disc_loss:.4f}\t'
                          f'Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Adv: {train_adv:.4f}\t'
                          f'Weighted KL: {train_kl * current_beta:.4f}, Weighted Adv: {train_adv*current_gamma:.4f}\t'
                          f'Time: {stop-start:.2f} s')
                else:
                    print(f'Epoch {epoch_idx}: Gen loss: {gen_loss:.4f}, Disc loss: -\t'
                          f'Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Adv: {train_adv:.4f}\t'
                          f'Weighted KL: {train_kl * current_beta:.4f}, Weighted Adv: {train_adv*current_gamma:.4f}\t'
                          f'Time: {stop-start:.2f} s')

            if epoch_idx % self.check_val_every_n_epochs == 0 and epoch_idx > self.beta_freeze_epochs:
                eval_loss, eval_recon, eval_kl, eval_adv = self.eval_model(
                    validation_loader,  
                    epoch=epoch_idx
                )

                if jax.process_index() == 0:
                    self.logger_validation.add_scalar('loss', eval_loss, global_step=epoch_idx)
                    self.logger_recon_val.add_scalar('reconstruction', eval_recon, global_step=epoch_idx)
                    self.logger_kl_val.add_scalar('kl_divergence', eval_kl, global_step=epoch_idx)
                    self.logger_adv_val.add_scalar('adv_loss', eval_adv, global_step=epoch_idx)

                    print(f'Epoch {epoch_idx}: Eval loss: {eval_loss:.4f}\t'
                          f'Recon: {eval_recon:.4f}, KL: {eval_kl:.4f}, Adv: {eval_adv:.4f}')

                # Save to both checkpoint managers (async if enabled)
                chckpt_loss = eval_recon + 0.0001 * eval_kl
                self._save_checkpoints(epoch_idx, chckpt_loss)

                # Generate reconstructions for visualization
                if jax.process_index() == 0 and self.generate_callback:
                    self.generate_callback.log_generations(
                        self.generator_model, self.gen_state, 
                        logger=self.logger_validation, 
                        epoch=epoch_idx
                    )

                # Flush loggers only at validation time (not every epoch)
                if jax.process_index() == 0:
                    for logger in [self.logger_training, self.logger_validation, 
                                   self.logger_recon_train, self.logger_kl_train,
                                   self.logger_adv_train, self.logger_disc_train,
                                   self.logger_recon_val, self.logger_kl_val, 
                                   self.logger_adv_val]:
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
    def _save_checkpoints(self, epoch_idx, eval_recon):
        """
        Function to save to both checkpoint managers with async support.
        """
        # 1. Save best model parameters
        best_params_data = {
            'gen_params': self.gen_state.params,
            'epoch': epoch_idx
        }

        # Note: We pass the pre-computed self._best_params_save_args
        self.best_params_manager.save(
            epoch_idx,
            best_params_data,
            save_kwargs={'save_args': self._best_params_save_args},
            metrics={'chckpt_loss': eval_recon}
        )

        # 2. Save full training state
        full_state_data = {
            'gen_state': self.gen_state,
            'disc_state': self.disc_state,
            'key': self.key,
            'gin_operative_config': gin.operative_config_str(),
            'epoch': epoch_idx
        }

        # Note: We pass the pre-computed self._full_state_save_args
        self.full_state_manager.save(
            epoch_idx,
            full_state_data,
            save_kwargs={'save_args': self._full_state_save_args}
        )
  
    def save_gin_config(self):
        """Save the exact gin configuration that was used"""
        if jax.process_index() == 0:
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
        if jax.process_index() == 0 and hasattr(self, 'gin_files_used'):
            configs_used_dir = os.path.join(config_dir, 'original_files')
            os.makedirs(configs_used_dir, exist_ok=True)

            for gin_file in self.gin_files_used:
                if os.path.exists(gin_file):
                    filename = os.path.basename(gin_file)
                    destination = os.path.join(configs_used_dir, filename)
                    import shutil
                    shutil.copy2(gin_file, destination)
                    
    ##############################################################################
    #                               LOADING FUNCTIONS                            #
    ##############################################################################
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
            'gen_state': self.gen_state,
            'disc_state': self.disc_state,
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
            self.gen_state = restored_data['gen_state']
            self.disc_state = restored_data['disc_state']
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
        """Print model summary for both generator and discriminator"""
        print("\n=== VAE-GAN Model Summary ===")
        print("Generator (VAE):")
        print(self.generator_model.tabulate(jax.random.PRNGKey(0), self.key, exmp_batch))
        print("\nDiscriminator:")
        print(self.discriminator_model.tabulate(jax.random.PRNGKey(0), exmp_batch, is_train=False))
        print("===============================\n")

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

        # Check generator parameters are replicated
        sample_gen_param = jax.tree_util.tree_leaves(self.gen_state.params)[0]
        is_gen_replicated = (hasattr(sample_gen_param, 'sharding') and 
                            sample_gen_param.sharding.spec == P())

        # Check discriminator parameters are replicated
        sample_disc_param = jax.tree_util.tree_leaves(self.disc_state.params)[0]
        is_disc_replicated = (hasattr(sample_disc_param, 'sharding') and 
                             sample_disc_param.sharding.spec == P())

        print(f"✓ Generator parameters: {'REPLICATED' if is_gen_replicated else 'UNKNOWN'}")
        print(f"✓ Discriminator parameters: {'REPLICATED' if is_disc_replicated else 'UNKNOWN'}")
        print(f"  - Same model weights on all {len(self.mesh.devices)} devices")

        # Test data sharding
        dummy_data = jnp.ones((32, 64, 64, 64, 3))  # Example batch
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
def train_vae_gan(generator_model=None,
                 discriminator_model=None,
                 train_loader=None,
                 val_loader=None,
                 epochs=20000,
                 log_dir=None,
                 mesh=None):  # ✅ ADD: Accept mesh parameter
    """
    Main training function for VAE-GAN
    
    Args:
        generator_model: Generator model instance
        discriminator_model: Discriminator model instance
        train_loader: Training dataloader (Iterator)
        val_loader: Validation dataloader (Iterator)
        epochs: Number of training epochs
        log_dir: Directory for logs and checkpoints
        mesh: JAX Mesh for SPMD (if None, will create default)
    """
    if jax.process_index() == 0:
        print("=" * 50)
        print(f"VAE-GAN Training Configuration")
        print("=" * 50)
        print(f"  generator: {type(generator_model).__name__}")
        print(f"  discriminator: {type(discriminator_model).__name__}")
        print(f"  epochs: {epochs}")
        print(f"  log_dir: {log_dir if log_dir else 'default'}")
        print("=====================================")

    if generator_model is None or discriminator_model is None:
        raise ValueError("Both generator_model and discriminator_model must be provided.")
    if train_loader is None or val_loader is None:
        raise ValueError("Both train_loader and val_loader must be provided.")
    
    # Use provided mesh or fallback in creating default
    if mesh is None:
        devices = jax.local_devices()
        mesh = Mesh(devices, ('data',))
    
    # Batch for initialization from validation loader
    init_batch = next(iter(val_loader))
    
    # Create trainer with the configured models
    trainer = TrainerModuleVAEGAN(
        generator_model=generator_model,
        discriminator_model=discriminator_model,
        init_batch=init_batch,
        log_dir=log_dir,
        mesh=mesh
    )
    
    if jax.process_index() == 0:
        print('# ----- Training the VAE-GAN model')

    # Start training
    trainer.train_model(
        trainer_loader=train_loader,
        validation_loader=val_loader,
        num_epochs=epochs
    )

    return trainer