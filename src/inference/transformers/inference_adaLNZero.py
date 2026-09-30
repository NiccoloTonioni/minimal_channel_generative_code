import os
import numpy as np
from typing import Tuple
# -- Flax
from flax.training import orbax_utils
# -- Gin
import gin
# -- JAX
import jax
import jax.numpy as jnp
# -- Numpy 
import numpy as np
# -- Orbax for checkpoints
import orbax.checkpoint as ocp

###################################################################################################
# Inference class for loading and using trained VAE-GAN models
###################################################################################################
class AdaLNZeroInference:
    """Lightweight inference class for Conditioned Latent Space Transformer with AdaLN-Zero."""
    
    def __init__(self, 
                 transformer_model, 
                 checkpoint_dir: str,
                 c_latent: int,    # e.g., 9
                 c_sensor: int     # e.g., 9
                ):
        self.transformer_model = transformer_model
        self.checkpoint_dir = checkpoint_dir
        self.c_latent = c_latent
        self.c_sensor = c_sensor
        
        # Initialize checkpoint manager
        best_params_dir = os.path.join(checkpoint_dir, 'best_model_params')
        if not os.path.exists(best_params_dir):
            raise ValueError(f"Best model parameters directory not found: {best_params_dir}")
        
        handler = ocp.PyTreeCheckpointer()
        self.params_manager = ocp.CheckpointManager(best_params_dir, checkpointers=handler)
        
        # Load params and compile
        self._load_best_parameters()
        self._bind_model()
        self._init_jitted_functions()
        
    def _load_best_parameters(self):
        """Load the best model parameters"""
        target_step = self.params_manager.best_step()
        if target_step is None:
            raise ValueError("No best model parameters found")
        
        print(f"Loading best model parameters from step {target_step}")
        
        # Dummy init for shape inference
        dummy_input = jnp.ones((1, 1, self.c_latent))
        dummy_sensor = jnp.ones((1, 1, self.c_sensor))
        dummy_mask = jnp.ones((1, 1), dtype=jnp.bool_) 
        
        init_key = jax.random.PRNGKey(42)
        
        # Initialize structure
        trnsf_params = self.transformer_model.init(
            init_key, 
            dummy_input, 
            dummy_sensor, 
            mask=dummy_mask,
            deterministic=True
        )['params']
        
        restore_target = {'params': trnsf_params, 'epoch': 0}
        restore_args = orbax_utils.restore_args_from_target(restore_target)
        
        restored_data = self.params_manager.restore(
            target_step,
            items=restore_target,
            restore_kwargs={'restore_args': restore_args}
        )
        
        self.trnsf_params = restored_data['params']
        self.epoch = restored_data['epoch']
        print(f"Successfully loaded parameters from epoch {self.epoch}")
    
    def _bind_model(self):
        self.bound_transformer = self.transformer_model.bind({'params': self.trnsf_params})
    
    def _init_jitted_functions(self):
        # Define inner function
        def predict_step_impl(padded_latents, padded_sensors, mask, valid_len):
            preds = self.bound_transformer(
                padded_latents, padded_sensors, mask=mask, deterministic=True
            )
            # Extract last valid prediction
            batch_indices = jnp.arange(preds.shape[0])
            next_step_pred = preds[batch_indices, valid_len - 1, :]
            return next_step_pred[:, None, :] 

        self.jitted_predict_step = jax.jit(predict_step_impl)

    def rollout(self, initial_latents, full_sensor_sequence, steps_to_predict=None):
        """
        Autoregressively generate future steps given single-trajectory inputs (No Batch Dim).
        """
        # 1. Inputs are (T, C). We manually add Batch dimension -> (1, T, C)
        T_init, _ = initial_latents.shape
        T_total, _ = full_sensor_sequence.shape
        
        if steps_to_predict is None:
            steps_to_predict = T_total - T_init
        if steps_to_predict <= 0:
            return initial_latents
        
        # Add batch dim for JAX model
        current_latents = initial_latents[None, :, :]       # (1, T_init, C)
        full_sensors_batched = full_sensor_sequence[None, :, :] # (1, T_total, C)
            
        block_size = self.transformer_model.block_size
        
        for _ in range(steps_to_predict):
            t = current_latents.shape[1]
            
            if t >= block_size:
                # Sliding window
                context_x = current_latents[:, -block_size:]
                context_y = full_sensors_batched[:, t - block_size : t, :]
                mask = jnp.ones((1, block_size), dtype=jnp.bool_)
                valid_len = block_size
            else:
                # Padding
                pad_len = block_size - t
                context_x = jnp.pad(current_latents, ((0, 0), (0, pad_len), (0, 0)))
                context_y = jnp.pad(full_sensors_batched[:, :t, :], ((0, 0), (0, pad_len), (0, 0)))
                
                seq_mask = jnp.arange(block_size) < t
                mask = jnp.tile(seq_mask, (1, 1))
                valid_len = t
                
            next_step = self.jitted_predict_step(context_x, context_y, mask, valid_len)
            current_latents = jnp.concatenate([current_latents, next_step], axis=1)
            
        return current_latents # Returns (1, T_total, C)

###################################################################################################
# Function to save reconstructions using a trained VAE-GAN model
###################################################################################################
def save_full_latent_rollout(transformer_model,
                             checkpoint_dir: str,
                             latent_dir: str,
                             sensor_dir: str,
                             length_initial_context: int,
                             c_latent: int = 4,
                             c_sensor: int = 9
                             ):
    """Generate full rollout of latent prediction and compute metrics."""
    
    print("Setting up for full latent rollout...")
    inference_module = AdaLNZeroInference(
        transformer_model=transformer_model,
        checkpoint_dir=checkpoint_dir,
        c_latent=c_latent,
        c_sensor=c_sensor
    )

    output_dir = os.path.join(checkpoint_dir, f"full_rollout_epoch_{inference_module.epoch}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data (Single Trajectory, No Batch Dim)
    # Latents: (T, C)
    latents = np.squeeze(np.load(os.path.join(latent_dir, 'latents_mean.npy')), axis=(1, 2))
    
    init_latents = latents[:length_initial_context] # Context
    targt_latents = latents[length_initial_context:] # Future Ground Truth
    
    # Sensors: (T, C)
    sensor_files = ['sensor_origin.npy', 'sensor_zero.npy', 'sensor_minimum.npy']
    sensor_paths = [os.path.join(sensor_dir, f) for f in sensor_files]
    full_sensors = np.concatenate([np.load(f) for f in sensor_paths], axis=-1)
    
    print(f"Context Latents: {init_latents.shape}")
    print(f"Target Latents: {targt_latents.shape}")

    # 2. Run Rollout
    # Returns (1, T_total, C)
    predicted_batched = inference_module.rollout(init_latents, full_sensors)
    
    # 3. Squeeze and Slice for Metrics
    # Remove batch dim -> (T_total, C)
    predicted_full = np.array(predicted_batched[0]) 
    
    # Extract only the FUTURE part to compare with targets
    predicted_future = predicted_full[length_initial_context:]
    
    print(f"Predicted Full: {predicted_full.shape}")
    print(f"Predicted Future: {predicted_future.shape}")

    # 4. Compute Metrics on Future only
    mse = np.mean((predicted_future - targt_latents) ** 2)
    mae = np.mean(np.abs(predicted_future - targt_latents))
    print(f"Rollout Future MSE: {mse:.6f}, MAE: {mae:.6f}")

    # 5. Save the FULL prediction (History + Future)
    save_path = os.path.join(output_dir, 'predicted_latents.npy')
    np.save(save_path, predicted_future)
    print(f"Saved predicted latents to: {save_path}")
    save_path = os.path.join(output_dir, 'target_latents.npy')
    np.save(save_path, targt_latents)
    print(f"Saved target latents to: {save_path}")
    save_path = os.path.join(output_dir, 'initial_latents.npy')
    np.save(save_path, init_latents)
    print(f"Saved initial latents to: {save_path}")

def save_latent_rollout_window(transformer_model,
                             checkpoint_dir: str,
                             latent_dir: str,
                             sensor_dir: str,
                             length_initial_context: int,
                             rollout_window: int,   
                             c_latent: int = 4,
                             c_sensor: int = 9
                             ):
    """Generate full rollout of latent prediction and compute metrics."""
    import time 

    print("Setting up for full latent rollout...")
    inference_module = AdaLNZeroInference(
        transformer_model=transformer_model,
        checkpoint_dir=checkpoint_dir,
        c_latent=c_latent,
        c_sensor=c_sensor
    )

    output_dir = os.path.join(checkpoint_dir, f"full_rollout_epoch_{inference_module.epoch}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data (Single Trajectory, No Batch Dim)
    # Latents: (T, C)
    latents = np.squeeze(np.load(os.path.join(latent_dir, 'latents_mean.npy')), axis=(1, 2))
    
    init_latents = latents[:length_initial_context] # Context
    targt_latents = latents[length_initial_context:length_initial_context+rollout_window] # Future Ground Truth
    
    # Sensors: (T, C)
    sensor_files = ['sensor_origin.npy', 'sensor_zero.npy', 'sensor_minimum.npy']
    sensor_paths = [os.path.join(sensor_dir, f) for f in sensor_files]
    full_sensors = np.concatenate([np.load(f) for f in sensor_paths], axis=-1)
    
    print(f"Context Latents: {init_latents.shape}")
    print(f"Target Latents: {targt_latents.shape}")

    # 2. Run Rollout
    # Returns (1, T_total, C)
    t_start = time.time()
    predicted_batched = inference_module.rollout(init_latents, full_sensors, steps_to_predict=rollout_window)
    t_end = time.time()
    print(f"Rollout completed in {t_end - t_start:.2f} seconds")
    
    # 3. Squeeze and Slice for Metrics
    # Remove batch dim -> (T_total, C)
    predicted_full = np.array(predicted_batched[0]) 
    
    # Extract only the FUTURE part to compare with targets
    predicted_future = predicted_full[length_initial_context:]
    
    print(f"Predicted Full: {predicted_full.shape}")
    print(f"Predicted Future: {predicted_future.shape}")

    # 4. Compute Metrics on Future only
    nrmse = np.sqrt(np.mean((predicted_future - targt_latents) ** 2)) / np.sqrt(np.mean(targt_latents ** 2))
    nmae = np.mean(np.abs(predicted_future - targt_latents)) / np.mean(np.abs(targt_latents))
    print(f"Rollout Future NRMSE: {nrmse:.6f}, NMAE: {nmae:.6f}")

    # 5. Save the FULL prediction (History + Future)
    save_path = os.path.join(output_dir, 'predicted_latents.npy')
    np.save(save_path, predicted_future)
    print(f"Saved predicted latents to: {save_path}")
    save_path = os.path.join(output_dir, 'target_latents.npy')
    np.save(save_path, targt_latents)
    print(f"Saved target latents to: {save_path}")
    save_path = os.path.join(output_dir, 'initial_latents.npy')
    np.save(save_path, init_latents)
    print(f"Saved initial latents to: {save_path}")
    


