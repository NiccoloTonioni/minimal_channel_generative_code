"""
Training loop of the sensor-conditioned Transformer: teacher forcing with scheduled sampling, optional
time-weighted loss (Eq. 24) and input-noise injection (Eq. 25), validation, checkpointing and logging.
"""
import os
import numpy as np
from typing import Tuple, Dict
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
# -- Summary Writer
from torch.utils.tensorboard import SummaryWriter
# -- Loss functions
from src.models.transformers.loss_functions import (teacher_forcing_loss, training_loss, split_loss_type,
                                                    per_timestep_loss, per_channel_loss)
from src.trainers.transformers._helper_functions import get_curriculum_scheduler

#################################################################################################
#                            TENSORBOARD CALLBACKS FOR VISUALIZATION                            #
#################################################################################################
class ConditionedPredictionCallback:
    """Callback to log predictions for conditioned transformer"""

    def __init__(self, input_sequences, target_sequences, sensor_sequences, every_n_epochs=100):
        super().__init__()
        self.input_sequences = input_sequences[:4]  # Take first 4 sequences (B=4)
        self.target_sequences = target_sequences[:4]
        self.sensor_sequences = sensor_sequences[:4]
        self.every_n_epochs = every_n_epochs

    def log_predictions(self, model, state, logger, epoch):
        """Log prediction metrics and visualizations"""
        if epoch % self.every_n_epochs == 0:
            
            # 1. CREATE AN ALL-TRUE MASK FOR FULL SEQUENCE EVALUATION
            batch_size, T_max, _ = self.input_sequences.shape
            cb_mask = jnp.ones((batch_size, T_max), dtype=jnp.bool_)

            # 2. PASS THE MASK TO THE MODEL
            predictions = model.apply(
                {'params': state.params}, 
                self.input_sequences,
                self.sensor_sequences,
                mask=cb_mask,            # <-- ADDED MASK HERE
                deterministic=True
            )
            
            # Transfer to host
            predictions = jax.device_get(predictions)
            targets = jax.device_get(self.target_sequences)
            inputs = jax.device_get(self.input_sequences)
            sensors = jax.device_get(self.sensor_sequences)
            
            # Compute per-timestep errors
            per_t_error = np.mean(np.abs(predictions - targets), axis=(0, 2))
            
            # Log to tensorboard
            for t in range(min(len(per_t_error), 10)):  # Log first 10 timesteps
                logger.add_scalar(f"prediction/per_timestep_mae/t{t}", per_t_error[t], global_step=epoch)
            
            # Log overall prediction quality
            mae = np.mean(np.abs(predictions - targets))
            mse = np.mean((predictions - targets) ** 2)
            logger.add_scalar("prediction/mae", mae, global_step=epoch)
            logger.add_scalar("prediction/mse", mse, global_step=epoch)
            
            # Log sensor statistics
            sensor_mean = np.mean(sensors)
            sensor_std = np.std(sensors)
            logger.add_scalar("sensor/mean", sensor_mean, global_step=epoch)
            logger.add_scalar("sensor/std", sensor_std, global_step=epoch)
            
            # Log first latent dimension evolution
            for seq_idx in range(min(2, predictions.shape[0])):
                logger.add_scalar(
                    f"prediction/seq{seq_idx}_dim0_input_mean", 
                    np.mean(inputs[seq_idx, :, 0]), 
                    global_step=epoch
                )
                logger.add_scalar(
                    f"prediction/seq{seq_idx}_dim0_pred_mean", 
                    np.mean(predictions[seq_idx, :, 0]), 
                    global_step=epoch
                )
                logger.add_scalar(
                    f"prediction/seq{seq_idx}_dim0_target_mean", 
                    np.mean(targets[seq_idx, :, 0]), 
                    global_step=epoch
                )

##################################################################################################
#                         TRAINER MODULE FOR CONDITIONED TRANSFORMER                            #
##################################################################################################
gin.external_configurable(optax.adam, name="adam")
gin.external_configurable(optax.adamw, name="adamw")
gin.external_configurable(optax.sgd, name="sgd")


@gin.configurable
class ConditionedTrainerModuleTransformer:
    """Trainer module for Conditioned Latent Space Transformer with AdaLN-Zero"""

    def __init__(self,
                 transformer_model,
                 init_batch_input: jnp.array,    
                 init_batch_target: jnp.array,   
                 init_batch_sensor: jnp.array,   
                 mesh: None,
                 # Optimizer params
                 optimizer=optax.adamw,
                 learning_rate=3e-4,
                 learning_schedule='warmup_cosine_decay',
                 warmup_steps=1000,
                 end_value=0.01,
                 gradient_clip=1.0,
                 weight_decay=0.01,
                 # Loss params
                 loss_type='mse',        # 'mse', 'mae', 'huber' or time-weighted 'tw_mse', 'tw_mae', 'tw_huber'
                 time_weight_delta=0.98, # Temporal weighting factor delta in (0, 1], used only by 'tw_*' losses
                 # Curriculum params
                 curriculum_type=None, 
                 curriculum_min_length=4,
                 curriculum_schedule=None,  
                 curriculum_growth_rate=2.0, 
                 curriculum_warmup_epochs=0, 
                 # Training params
                 seed=42,
                 check_val_every_n_epoch=10,
                 callbacks_every_n_epochs=50,
                 load_checkpoint=True,
                 # Scheduled Sampling Params
                 ss_start_epoch=20,     # When to start mixing predictions
                 ss_max_ratio=0.5,      # Max prob of using model output (don't go to 1.0 for Transf)
                 ss_ramp_epochs=100,    # How long to ramp up to max_ratio
                 # Input Noise Params
                 use_input_noise=False, # Add Gaussian noise to the mixed input after scheduled sampling
                 input_noise_std=0.01,  # Standard deviation sigma_n of the injected noise
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
        
        # Mesh for sharding
        self.mesh = mesh or Mesh(jax.local_devices(), ('data',))

        # Store parameters
        self.log_dir = log_dir
        self.base_log_dir = base_log_dir
        self.seed = seed
        self.debug = debug
        self.transformer_model = transformer_model
        self.load_checkpoint = load_checkpoint
        
        # Optimizer params
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.learning_schedule = learning_schedule
        self.warmup_steps = warmup_steps
        self.end_value = end_value
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay
        
        # Loss params
        if not 0.0 < time_weight_delta <= 1.0:
            raise ValueError(f"time_weight_delta must be in (0, 1], got {time_weight_delta}")
        self.loss_type = loss_type
        self.time_weight_delta = time_weight_delta
        # Validation always uses the un-weighted base loss (e.g. 'tw_mse' -> 'mse')
        self.eval_loss_type, _ = split_loss_type(loss_type)
        self.curriculum_type = curriculum_type
        self.curriculum_min_length = curriculum_min_length
        self.curriculum_schedule = curriculum_schedule
        self.curriculum_growth_rate = curriculum_growth_rate
        self.curriculum_warmup_epochs = curriculum_warmup_epochs

        # Scheduled Sampling Params
        self.ss_start_epoch = ss_start_epoch
        self.ss_max_ratio = ss_max_ratio
        self.ss_ramp_epochs = ss_ramp_epochs

        # Input Noise Params
        self.use_input_noise = use_input_noise
        self.input_noise_std = input_noise_std
        
        # Training params
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.callbacks_every_n_epochs = callbacks_every_n_epochs
            
        # Set params for model initialization
        self.init_input = init_batch_input[:8]
        self.init_target = init_batch_target[:8]
        self.init_sensor = init_batch_sensor[:8]
        self.key = jax.random.PRNGKey(seed)

        # Set optional parameters for checkpointing and logging
        self.async_checkpointing = async_checkpointing
        self.log_every_n_steps = log_every_n_steps

        # Call initialization functions
        self._init_logger()
        self._init_model(self.init_input, self.init_sensor)
        self.create_jitted_functions()

        if jax.process_index() == 0: 
            if print_model_summary:
                self._print_tabulate(init_batch_input, init_batch_sensor) 
            if verify_sharding:
                self._verify_sharding_setup()  
            if save_config:
                self.save_gin_config()

    ##############################################################################
    #                           INITIALIZATION FUNCTIONS                         #
    ##############################################################################
    def _init_logger(self):
        """Initialize tensorboard loggers"""
        log_dir = self.log_dir
        if log_dir is None:
            base_log_dir = self.base_log_dir
            log_dir = os.path.join(base_log_dir, "ConditionedTransformer")
        
        # Prepare logging with model parameters
        log_dir = os.path.join(
            log_dir, 
            f'bs_{self.transformer_model.block_size}_'
            f'nl_{self.transformer_model.n_layer}_'
            f'nh_{self.transformer_model.n_heads}_'
            f'ne_{self.transformer_model.n_embd}_'
            f'cd_{self.transformer_model.cond_dim}_'  # Add conditioning dimension
            f'attn_{self.transformer_model.attention_type}_'  # Attention mechanism
            f'loss_{self.loss_type}_'  # Training loss ('tw_*' = time-weighted)
            + (f'noise_{self.input_noise_std}_' if self.use_input_noise else '')  # Input-noise injection
            + f'sd_{self.seed}'  # Run seed
        )
        self.log_dir = log_dir

        if jax.process_index() == 0: 
            # Metrics path
            self.metrics_path = os.path.join(log_dir, 'metrics')
            os.makedirs(self.metrics_path, exist_ok=True)
            
            # Callbacks for visualization (now includes sensor data)
            self.prediction_callback = ConditionedPredictionCallback(
                self.init_input, 
                self.init_target,
                self.init_sensor, 
                every_n_epochs=self.callbacks_every_n_epochs
            )
            
            # Tensorboard loggers
            self.logger_training = SummaryWriter(log_dir=self.log_dir+"/training")
            self.logger_validation = SummaryWriter(log_dir=self.log_dir+"/validation")
            self.logger_analysis = SummaryWriter(log_dir=self.log_dir+"/analysis")
        else:
            self.metrics_path = None
            self.prediction_callback = None
            self.logger_training = None
            self.logger_validation = None
            self.logger_analysis = None

    def _init_model(self, init_input, init_sensor):
        """Initialize conditioned transformer parameters"""
        self.key, init_key = jax.random.split(self.key)
        
        # Initialize model parameters WITH SENSOR CONDITIONING
        params = self.transformer_model.init(
            init_key, 
            init_input, 
            init_sensor, 
            deterministic=True
        )['params']

        # Handle sharding if using multiple devices
        params_sharding = jax.tree_util.tree_map(
            lambda x: NamedSharding(self.mesh, P()), params
        )
        params = jax.device_put(params, params_sharding)

        # Create default state (optimizer will be added later)
        self.state = train_state.TrainState(
            step=0,
            apply_fn=self.transformer_model.apply,
            params=params,
            tx=None,
            opt_state=None
        )

    def _init_orbax_checkpoint_managers(self):
        """Initialize two separate Orbax CheckpointManagers"""
        # We create two separate checkpoint managers:
        #   1. A checkpoint manager for saving only the best model parameters (for inference)
        #   2. A checkpoint manager for saving the full training state (for resuming training)
        best_params_dir = os.path.join(self.log_dir, 'best_model_params')
        full_state_dir = os.path.join(self.log_dir, 'training_checkpoints')
        
         # 1. Best model parameters checkpoint manager 
        best_params_options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.check_val_every_n_epoch,
            max_to_keep=1, 
            best_fn=lambda metrics: metrics['chckpt_loss'],
            best_mode='min'
        )
        
        # 2. Full training state checkpoint manager (no best tracking)
        full_state_options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.check_val_every_n_epoch,
            max_to_keep=2, 
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

        # 1. Template for Best Params
        best_params_template = {
            'params': self.state.params,
            'epoch': 0  # scalar placeholder
        }
        self._best_params_save_args = orbax_utils.save_args_from_target(best_params_template)

        # 2. Template for Full State
        full_state_template = {
            'state': self.state,
            'key': self.key,
            'gin_operative_config': "placeholder_string", 
            'epoch': 0
        }
        self._full_state_save_args = orbax_utils.save_args_from_target(full_state_template)

        if jax.process_index() == 0:
            print(f"Best params CheckpointManager initialized at {best_params_dir}")
            print(f"Full state CheckpointManager initialized at {full_state_dir}")

    def _init_optimizer(self, num_epochs: int, num_steps_per_epoch: int):
        """Initialize optimizer with learning rate schedule"""
        optimizer = self._create_optimizer(
            optimizer=self.optimizer,
            optimizer_kwargs=None,  # Add if you want to pass specific kwargs for the optimizer
            learning_rate=self.learning_rate,
            learning_schedule=self.learning_schedule,
            warmup_steps=self.warmup_steps,
            end_value=self.end_value,
            gradient_clip=self.gradient_clip,
            weight_decay=self.weight_decay,
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch
        )

        # Create training state with optimizer
        self.state = train_state.TrainState.create(
            apply_fn=self.state.apply_fn,
            params=self.state.params,
            tx=optimizer
        )

    def _init_curriculum_scheduler(self, num_epochs: int):
        """Initialize curriculum learning scheduler if specified"""
        if self.curriculum_type is None:
            self.curriculum_schedule = None
        else:
            self.curriculum_schedule = get_curriculum_scheduler(
                                        curriculum_type=self.curriculum_type,
                                        min_length=self.curriculum_min_length,
                                        max_length=self.transformer_model.block_size,
                                        num_epochs=num_epochs,
                                        warmup_epochs=self.curriculum_warmup_epochs,
                                        schedule=self.curriculum_schedule,
                                        growth_rate=self.curriculum_growth_rate,
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
        """Create training and evaluation step functions"""
        
        def train_step(state: train_state.TrainState,
                      input_seq: jnp.ndarray,
                      target_seq: jnp.ndarray,
                      sensor_seq: jnp.ndarray,  
                      mask: jnp.ndarray,
                      sampling_ratio: float,  
                      key: jax.Array) -> Tuple[train_state.TrainState, float, Dict]:
            
            # Split RNG keys for sampling and input noise
            key, mix_key, noise_key = jax.random.split(key, 3)
             
            def loss_fn(params):
                # Run model on Ground Truth to get "candidate" predictions
                # We stop_gradient because we don't want to backprop through this generation step
                candidate_preds = state.apply_fn(
                    {'params': params}, 
                    input_seq,
                    sensor_seq, 
                    mask=mask,                  
                    deterministic=False, # Use dropout if enabled
                    rngs={'dropout': key}
                )
                candidate_preds = jax.lax.stop_gradient(candidate_preds)

                # Parallel scheluder sampling logic
                # 1. Create a Bernoulli mask (1 = use Prediction, 0 = use Ground Truth)
                B, T, C = input_seq.shape
                swap_mask = jax.random.bernoulli(mix_key, p=sampling_ratio, shape=(B, T, 1))
                # The first token has no previous prediction, always keep the Ground Truth
                swap_mask = swap_mask.at[:, 0].set(False)
                
                # 2. Shift Predictions to align with Inputs
                # Pred at index [t] corresponds to time [t+1]. 
                # [Pred0, Pred1, Pred2] -> [Zero, Pred0, Pred1]
                zeros_start = jnp.zeros((B, 1, C), dtype=candidate_preds.dtype)
                shifted_preds = jnp.concatenate([zeros_start, candidate_preds[:, :-1, :]], axis=1)

                # 3. Create Mixed Input
                mixed_input = jnp.where(swap_mask, shifted_preds, input_seq)

                # 4. Add Gaussian noise to the mixed input (optional)
                if self.use_input_noise:
                    noise = jax.random.normal(noise_key, mixed_input.shape, dtype=mixed_input.dtype)
                    mixed_input = mixed_input + self.input_noise_std * noise

                # Actual forward pass with mixed input
                predictions = state.apply_fn(
                    {'params': params}, 
                    mixed_input, 
                    sensor_seq, 
                    mask=mask,                  
                    deterministic=False,
                    rngs={'dropout': key}
                )
                
                loss = training_loss(
                    predictions, 
                    target_seq, 
                    mask=mask,                  
                    loss_type=self.loss_type,
                    delta=self.time_weight_delta
                )
                return loss, predictions
            
            # Compute gradients
            (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            
            # Update parameters
            state = state.apply_gradients(grads=grads)
            
            # Compute additional metrics
            mae_loss = teacher_forcing_loss(predictions, target_seq, mask=mask, loss_type='mae')
            grad_norm = optax.global_norm(grads)

            return state, (loss, mae_loss, grad_norm)

        def eval_step(state: train_state.TrainState,
                     input_seq: jnp.ndarray,
                     target_seq: jnp.ndarray,
                     sensor_seq: jnp.ndarray,
                     mask: jnp.ndarray) -> Tuple[float, Dict]:
            """Single evaluation step"""
            
            predictions = state.apply_fn(
                {'params': state.params}, 
                input_seq,
                sensor_seq,
                mask=mask,
                deterministic=True
            )
            
            loss = teacher_forcing_loss(predictions, target_seq, mask=mask, loss_type=self.eval_loss_type)
            mae = teacher_forcing_loss(predictions, target_seq, loss_type='mae')
            per_t_loss = per_timestep_loss(predictions, target_seq, loss_type='mse')
            per_c_loss = per_channel_loss(predictions, target_seq, loss_type='mse')
            
            return (loss, mae, per_t_loss, per_c_loss)

        return train_step, eval_step
    
    def create_jitted_functions(self):
        """Create jitted training and evaluation functions"""
        train_step, eval_step = self.create_functions()
        
        if self.debug:
            self.train_step = train_step
            self.eval_step = eval_step
        else:
            replicated = NamedSharding(self.mesh, P())
            data_sharded = NamedSharding(self.mesh, P('data'))
            
            self.train_step = jax.jit(train_step,
                in_shardings=(
                    replicated,    # state
                    data_sharded,  # input_seq
                    data_sharded,  # target_seq
                    data_sharded,  # sensor_seq
                    data_sharded,  # mask
                    replicated,    # sampling_ratio
                    replicated     # key
                ),
                out_shardings=(replicated, replicated)
            )
            
            self.eval_step = jax.jit(eval_step,
                in_shardings=(
                    replicated,    # state
                    data_sharded,  # input_seq
                    data_sharded,  # target_seq
                    data_sharded,  # sensor_seq 
                    data_sharded   # mask
                ),
                out_shardings=(replicated)
            )

    ##############################################################################
    #                          TRAIN/VAL EPOCH FUNCTIONS                         #
    ##############################################################################
    
    def _get_sampling_ratio(self, epoch):
        """Calculate the probability of using model predictions instead of ground truth"""
        if epoch < self.ss_start_epoch:
            return 0.0
        
        # Linear ramp up
        progress = (epoch - self.ss_start_epoch) / self.ss_ramp_epochs
        progress = min(max(progress, 0.0), 1.0)
        
        return progress * self.ss_max_ratio

    def train_epoch(self, trainer_loader, epoch, curriculum_scheduler=None):
        """Train for one epoch with sensor conditioning"""
        losses, maes, grad_norms = [], [], []

        # Get sequence length for this epoch (Curriculum)
        if curriculum_scheduler is not None:
            target_length = curriculum_scheduler.get_sequence_length(epoch)
            if jax.process_index() == 0 and epoch % 10 == 0:
                print(f"  Curriculum: Using sequence length {target_length}")
        else:
            target_length = self.transformer_model.block_size 
        
        # Calculate Scheduled Sampling Ratio
        sampling_ratio = self._get_sampling_ratio(epoch)
        if jax.process_index() == 0 and epoch % 10 == 0:
            print(f"  Scheduled Sampling: mixing ratio {sampling_ratio:.4f}")

        for input_seq, sensor_seq, target_seq in trainer_loader:
            batch_size, T_max, _ = input_seq.shape
            
            seq_mask = jnp.arange(T_max) < target_length
            mask = jnp.tile(seq_mask, (batch_size, 1))

            self.key, step_key = jax.random.split(self.key)

            # Call train step with sampling_ratio instead of noise
            self.state, metrics = self.train_step(
                self.state, input_seq, target_seq, sensor_seq, mask, sampling_ratio, step_key
            )

            losses.append(metrics[0])
            maes.append(metrics[1])
            grad_norms.append(metrics[2])

        return (float(jnp.mean(jnp.stack(losses))), 
                float(jnp.mean(jnp.stack(maes))), 
                float(jnp.mean(jnp.stack(grad_norms))))

    def eval_model(self, validation_loader, epoch):
        """Evaluate model on validation set with sensor conditioning"""
        losses, maes, per_t_losses, per_c_losses = [], [], [], []
        total_samples = 0
        
        for input_seq, sensor_seq, target_seq in validation_loader:
            batch_size, T_max, _ = input_seq.shape

            # Generate all-True mask for evaluation
            eval_mask = jnp.ones((batch_size, T_max), dtype=jnp.bool_)
            # Compute loss and metrics for this batch
            loss, mae, per_t, per_c = self.eval_step(
                self.state, input_seq, target_seq, sensor_seq, eval_mask
            )     
            # Append JAX array to lists 
            # (Weighted by batch size to handle drop_last=False)
            losses.append(loss * batch_size)
            maes.append(mae * batch_size)
            per_t_losses.append(per_t * batch_size)
            per_c_losses.append(per_c * batch_size)
            
            total_samples += batch_size
        
        if total_samples == 0:
            return 0.0, 0.0, np.zeros(1), np.zeros(1)

        # Stack, sum, and average entirely on the device, 
        # then pull to the host.
        final_metrics = jax.device_get({
            'loss': jnp.sum(jnp.stack(losses)) / total_samples,
            'mae': jnp.sum(jnp.stack(maes)) / total_samples,
            'per_t': jnp.sum(jnp.stack(per_t_losses), axis=0) / total_samples,
            'per_c': jnp.sum(jnp.stack(per_c_losses), axis=0) / total_samples,
        })

        # Return scalar floats for overall loss, but keep per_t and per_c as NumPy arrays!
        return (
            float(final_metrics['loss']), 
            float(final_metrics['mae']), 
            final_metrics['per_t'],   # Array of shape (T,)
            final_metrics['per_c']    # Array of shape (C,)
        )
        

    ##############################################################################
    #                                TRAINING LOOP                               #
    ##############################################################################
    def train_model(self, trainer_loader, validation_loader, num_epochs=500):
        """Main training loop"""
        # Create optimizers 
        self._init_optimizer(num_epochs, len(trainer_loader))
        # Initialize checkpoint managers
        self._init_orbax_checkpoint_managers()
        # Initialize curriculum scheduler if specified
        self._init_curriculum_scheduler(num_epochs)
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
            train_loss, train_mae, grad_norm = self.train_epoch(
                                                trainer_loader, 
                                                epoch=epoch_idx,
                                                curriculum_scheduler=self.curriculum_schedule 
                                                )
            stop = time.time()

            # Log only every N steps to reduce I/O overhead
            should_log = (epoch_idx % self.log_every_n_steps == 0)

            # Log training metrics
            if jax.process_index() == 0 and should_log:
                self.logger_training.add_scalar("loss", train_loss, global_step=epoch_idx)
                self.logger_training.add_scalar("mae", train_mae, global_step=epoch_idx)
                self.logger_training.add_scalar("grad_norm", grad_norm, global_step=epoch_idx)
            
            if jax.process_index() == 0:
                print(f'Epoch {epoch_idx}:\t Train Loss: {train_loss:.6f},\t '
                      f'MAE: {train_mae:.6f},\t GradNorm {grad_norm:.6f},\t'
                      f'Time: {stop-start:.2f} s')

            # Validation
            if epoch_idx % self.check_val_every_n_epoch == 0:

                eval_loss, eval_mae, per_t_loss, per_c_loss = self.eval_model(validation_loader, epoch=epoch_idx)
                
                if jax.process_index() == 0:
                    # Log validation metrics
                    self.logger_validation.add_scalar('loss', eval_loss, global_step=epoch_idx)
                    self.logger_validation.add_scalar('mae', eval_mae, global_step=epoch_idx)
                    
                    # Log per-timestep analysis
                    for t, loss_t in enumerate(per_t_loss[:10]):  # First 10 timesteps
                        self.logger_analysis.add_scalar(
                            f'per_timestep_loss/t{t}', loss_t, global_step=epoch_idx
                        )
                    
                    # Log per-channel analysis
                    for c, loss_c in enumerate(per_c_loss):
                        self.logger_analysis.add_scalar(
                            f'per_channel_loss/ch{c}', loss_c, global_step=epoch_idx
                        )
                    
                    print(f'Epoch {epoch_idx}: Val Loss: {eval_loss:.6f}, MAE: {eval_mae:.6f}')
                
                # Save checkpoints
                self._save_checkpoints(epoch_idx, eval_loss)
                
                # Generate predictions for visualization
                if jax.process_index() == 0 and self.prediction_callback:
                    self.prediction_callback.log_predictions(
                        self.transformer_model, self.state, 
                        logger=self.logger_validation, epoch=epoch_idx
                    )
                
                # Flush loggers
                if jax.process_index() == 0:
                    for logger in [self.logger_training, self.logger_validation, self.logger_analysis]:
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
    def _save_checkpoints(self, epoch_idx, eval_loss):
        """
        Function to save to both checkpoint managers with async support.
        """
        # 1. Save best model parameters
        best_params_data = {
            'params': self.state.params,
            'epoch': epoch_idx
        }

        self.best_params_manager.save(
            epoch_idx,
            best_params_data,
            save_kwargs={'save_args': self._best_params_save_args},
            metrics={'chckpt_loss': eval_loss}
        )
        
        # 2. Save full training state
        full_state_data = {
            'state': self.state,
            'key': self.key,
            'gin_operative_config': gin.operative_config_str(),
            'epoch': epoch_idx
        }
        
        self.full_state_manager.save(
            epoch_idx,
            full_state_data,
            save_kwargs={'save_args': self._full_state_save_args}
        )
        
    def save_gin_config(self):
        """Save gin configuration"""
        config_dir = os.path.join(self.log_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        operative_config_path = os.path.join(config_dir, 'operative_config.gin')
        with open(operative_config_path, 'w') as f:
            f.write(gin.operative_config_str())
        
        print(f"Configuration saved to: {config_dir}")

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
            'state': self.state,
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
            self.state = restored_data['state']
            self.key = restored_data['key']
            start_epoch = restored_data['epoch'] + 1
            
            if jax.process_index() == 0:
                print(f"Restored checkpoint from epoch {restored_data['epoch']}. "
                      f"Resuming from epoch {start_epoch}.")
            
            return start_epoch
        else:
            if jax.process_index() == 0:
                print("Failed to restore checkpoint. Starting from scratch.")
            return 0

    ##############################################################################
    #                              UTILITY FUNCTIONS                             #
    ##############################################################################
    def _print_tabulate(self, exmp_input, exmp_sensor):
        """Print model summary with sensor conditioning"""
        print("\n=== Conditioned Transformer Model Summary ===")
        print(self.transformer_model.tabulate(
            jax.random.PRNGKey(0), 
            exmp_input, 
            exmp_sensor,  # Include sensor input
            deterministic=True
        ))
        print("=============================================\n")

    def _verify_sharding_setup(self):
        """Verify sharding is set up correctly"""
        print("\n=== Sharding Verification ===")

        # Check parameters are replicated
        sample_param = jax.tree_util.tree_leaves(self.state.params)[0]
        is_replicated = (hasattr(sample_param, 'sharding') and 
                        sample_param.sharding.spec == P())

        print(f"✓ Model parameters: {'REPLICATED' if is_replicated else 'UNKNOWN'}")
        print(f"  - Same model weights on all {len(self.mesh.devices)} devices")

        # Test data sharding
        dummy_latent = jnp.ones((32, 64, 9))  # Example latent batch
        dummy_sensor = jnp.ones((32, 64, 3))  # Example sensor batch
        
        sharded_latent = jax.device_put(dummy_latent, NamedSharding(self.mesh, P('data')))
        sharded_sensor = jax.device_put(dummy_sensor, NamedSharding(self.mesh, P('data')))

        expected_per_device_size = dummy_latent.shape[0] // len(self.mesh.devices)
        actual_per_device_size = sharded_latent.shape[0]
        is_data_sharded = (actual_per_device_size == expected_per_device_size)

        print(f"✓ Data batches: {'SHARDED' if is_data_sharded else 'UNKNOWN'}")
        print(f"  - Original batch size: {dummy_latent.shape[0]}")
        print(f"  - Per-device batch size: {actual_per_device_size}")
        print(f"  - Latent shape: {dummy_latent.shape}")
        print(f"  - Sensor shape: {dummy_sensor.shape}")
        print("============================\n")


##################################################################################################
# Main training function to be called from the command line or other scripts
##################################################################################################
@gin.configurable
def train_conditioned_transformer(transformer_model=None,
                                  trainer_loader=None,
                                  validation_loader=None,
                                  epochs=1000,
                                  log_dir=None,
                                  mesh=None):
    """
    Main training function for Conditioned Latent Space Transformer'
    
    Args:
        transformer_model: An instance of the conditioned transformer model to be trained.
        trainer_loader: DataLoader for training data, must return (input_seq, target_seq, sensor_seq) tuples.
        validation_loader: DataLoader for validation data, must return (input_seq, target_seq, sensor_seq) tuples.
        epochs: Number of training epochs.
        log_dir: Directory for logging and checkpoints. If None, a default directory will be created
        mesh: Optional JAX Mesh for sharding. If None, a default mesh will be created using local devices.
    """
    if jax.process_index() == 0:
        print(f"\n{'='*70}")
        print(f"  Conditioned Latent Space Transformer Configuration")
        print(f"{'='*70}")
        print(f"  model: {type(transformer_model).__name__}")
        print(f"  block_size: {transformer_model.block_size}")
        print(f"  n_layer: {transformer_model.n_layer}")
        print(f"  n_heads: {transformer_model.n_heads}")
        print(f"  n_embd: {transformer_model.n_embd}")
        print(f"  cond_dim: {transformer_model.cond_dim}")
        print(f"  output_mode: {transformer_model.output_mode}")
        print(f"  epochs: {epochs}")
        print(f"  log_dir: {log_dir if log_dir else 'default'}")
        print(f"{'='*70}")

    if transformer_model is None:
        raise ValueError("transformer_model must be provided for training.")
    if trainer_loader is None or validation_loader is None:
        raise ValueError("Both trainer_loader and validation_loader must be provided for training.")
    
    # Verify output_mode is correct for teacher forcing
    if transformer_model.output_mode != "full_sequence":
        raise ValueError(
            f"Transformer output_mode must be 'full_sequence' for teacher forcing training, "
            f"but got '{transformer_model.output_mode}'. Please set output_mode='full_sequence' "
            f"in your model configuration."
        )
    
    # Use provided mesh or fallback in creating default
    if mesh is None:
        devices = jax.local_devices()
        mesh = Mesh(devices, ('data',))

    # Get init batches for model initialization
    init_input, init_sensor, init_target = next(iter(validation_loader))
    
    # Create trainer
    trainer = ConditionedTrainerModuleTransformer(
        transformer_model=transformer_model,
        init_batch_input=init_input,
        init_batch_target=init_target,
        init_batch_sensor=init_sensor,  
        log_dir=log_dir,
        mesh=mesh
    )
    
    if jax.process_index() == 0:
        print('# ----- Training the Conditioned Transformer model')

    trainer.train_model(
        trainer_loader=trainer_loader,
        validation_loader=validation_loader,
        num_epochs=epochs
    )

    return trainer