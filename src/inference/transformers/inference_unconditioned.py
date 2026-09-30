import os
import time
import numpy as np
# -- Flax
from flax.training import orbax_utils
# -- JAX
import jax
import jax.numpy as jnp
# -- Orbax for checkpoints
import orbax.checkpoint as ocp

###################################################################################################
# Inference class for loading and using a trained unconditional (EasyTimeTransformer) model
###################################################################################################
class UnconditionedInference:
    """Lightweight inference class for the unconditional Latent Space Transformer.

    Loads the best checkpoint from ``best_model_params/`` and exposes an
    autoregressive ``rollout`` method.
    """

    def __init__(self,
                 transformer_model,
                 checkpoint_dir: str,
                 c_latent: int,
                 ):
        """
        Args:
            transformer_model: Instantiated ``EasyTimeTransformer`` (not yet initialised).
            checkpoint_dir: Path to the experiment directory that contains
                ``best_model_params/``.
            c_latent: Number of latent channels (e.g. 4 or 8).
        """
        self.transformer_model = transformer_model
        self.checkpoint_dir = checkpoint_dir
        self.c_latent = c_latent

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
        """Load the best model parameters from the checkpoint directory."""
        target_step = self.params_manager.best_step()
        if target_step is None:
            raise ValueError("No best model parameters found")

        print(f"Loading best model parameters from step {target_step}")

        dummy_input = jnp.ones((1, 1, self.c_latent))
        init_key = jax.random.PRNGKey(42)

        trnsf_params = self.transformer_model.init(
            init_key,
            dummy_input,
            deterministic=True,
        )['params']

        restore_target = {'params': trnsf_params, 'epoch': 0}
        restore_args = orbax_utils.restore_args_from_target(restore_target)

        restored_data = self.params_manager.restore(
            target_step,
            items=restore_target,
            restore_kwargs={'restore_args': restore_args},
        )

        self.trnsf_params = restored_data['params']
        self.epoch = restored_data['epoch']
        print(f"Successfully loaded parameters from epoch {self.epoch}")

    def _bind_model(self):
        self.bound_transformer = self.transformer_model.bind({'params': self.trnsf_params})

    def _init_jitted_functions(self):
        def predict_step_impl(padded_latents, mask, valid_len):
            preds = self.bound_transformer(
                padded_latents, mask=mask, deterministic=True
            )
            batch_indices = jnp.arange(preds.shape[0])
            return preds[batch_indices, valid_len - 1, :][:, None, :]

        self.jitted_predict_step = jax.jit(predict_step_impl)

    def rollout(self,
                initial_latents: np.ndarray,
                steps_to_predict: int = None,
                ) -> np.ndarray:
        """Autoregressively generate future latent steps.

        Args:
            initial_latents: Context window of shape ``(T_init, C)``.
            steps_to_predict: Number of future steps to generate.  Defaults
                to ``block_size - T_init``.

        Returns:
            Full latent trajectory of shape ``(1, T_init + steps_to_predict, C)``
            (batch dimension of 1).
        """
        T_init, _ = initial_latents.shape
        block_size = self.transformer_model.block_size

        if steps_to_predict is None:
            steps_to_predict = block_size - T_init
        if steps_to_predict <= 0:
            return initial_latents[None]

        current_latents = initial_latents[None, :, :]  # (1, T_init, C)

        for _ in range(steps_to_predict):
            t = current_latents.shape[1]

            if t >= block_size:
                context_x = current_latents[:, -block_size:]
                mask = jnp.ones((1, block_size), dtype=jnp.bool_)
                valid_len = block_size
            else:
                pad_len = block_size - t
                context_x = jnp.pad(current_latents, ((0, 0), (0, pad_len), (0, 0)))
                seq_mask = jnp.arange(block_size) < t
                mask = jnp.tile(seq_mask, (1, 1))
                valid_len = t

            next_step = self.jitted_predict_step(context_x, mask, valid_len)

            current_latents = jnp.concatenate([current_latents, next_step], axis=1)

        return current_latents  # (1, T_init + steps_to_predict, C)


###################################################################################################
# Convenience function — mirrors save_full_latent_rollout from inference_adaLNZero
###################################################################################################
def save_full_latent_rollout(transformer_model,
                             checkpoint_dir: str,
                             latent_dir: str,
                             length_initial_context: int,
                             c_latent: int = 4,
                             ):
    """Generate a full unconditional latent rollout and compute metrics.

    Args:
        transformer_model: Instantiated ``EasyTimeTransformer``.
        checkpoint_dir: Directory containing ``best_model_params/``.
        latent_dir: Directory containing ``latents_mean.npy``.
        length_initial_context: Number of context steps fed to the model
            (should equal ``transformer_model.block_size``).
        c_latent: Number of latent channels.
    """
    print("Setting up for full unconditional latent rollout...")
    inference_module = UnconditionedInference(
        transformer_model=transformer_model,
        checkpoint_dir=checkpoint_dir,
        c_latent=c_latent,
    )

    output_dir = os.path.join(checkpoint_dir, f"full_rollout_epoch_{inference_module.epoch}")
    os.makedirs(output_dir, exist_ok=True)

    # Load Data (Single Trajectory, No Batch Dim)
    # Latents: (T, C)
    latents = np.squeeze(np.load(os.path.join(latent_dir, 'latents_mean.npy')), axis=(1, 2))
    init_latents = latents[:length_initial_context]
    targt_latents = latents[length_initial_context:]

    print(f"Context latents : {init_latents.shape}")
    print(f"Target latents  : {targt_latents.shape}")

    # Run Rollout
    t_start = time.time()
    predicted_batched = inference_module.rollout(
        init_latents,
        steps_to_predict=len(targt_latents),
    )
    t_end = time.time()
    print(f"Rollout completed in {t_end - t_start:.2f} s")

    # Squeeze and Slice for Metrics
    # Remove batch dim -> (T_total, C)
    predicted_full = np.array(predicted_batched[0])        # (T_total, C)
    predicted_future = predicted_full[length_initial_context:]

    print(f"Predicted full   : {predicted_full.shape}")
    print(f"Predicted future : {predicted_future.shape}")

    # Compute Metrics 
    nrmse = (np.sqrt(np.mean((predicted_future - targt_latents) ** 2))
             / np.sqrt(np.mean(targt_latents ** 2)))
    nmae = (np.mean(np.abs(predicted_future - targt_latents))
            / np.mean(np.abs(targt_latents)))
    print(f"Rollout Future NRMSE: {nrmse:.6f}, NMAE: {nmae:.6f}")

    np.save(os.path.join(output_dir, 'predicted_latents.npy'), predicted_future)
    np.save(os.path.join(output_dir, 'target_latents.npy'), targt_latents)
    np.save(os.path.join(output_dir, 'initial_latents.npy'), init_latents)
    print(f"Saved outputs to: {output_dir}")


def save_latent_rollout_window(transformer_model,
                               checkpoint_dir: str,
                               latent_dir: str,
                               length_initial_context: int,
                               rollout_window: int,
                               c_latent: int = 4,
                               ):
    """Generate a windowed unconditional latent rollout and compute metrics.

    Args:
        transformer_model: Instantiated ``EasyTimeTransformer``.
        checkpoint_dir: Directory containing ``best_model_params/``.
        latent_dir: Directory containing ``latents_mean.npy``.
        length_initial_context: Number of context steps.
        rollout_window: Number of future steps to predict.
        c_latent: Number of latent channels.
    """
    print("Setting up for windowed unconditional latent rollout...")
    inference_module = UnconditionedInference(
        transformer_model=transformer_model,
        checkpoint_dir=checkpoint_dir,
        c_latent=c_latent,
    )

    output_dir = os.path.join(checkpoint_dir, f"full_rollout_epoch_{inference_module.epoch}")
    os.makedirs(output_dir, exist_ok=True)

    # Load Data (Single Trajectory, No Batch Dim)
    # Latents: (T, C)
    latents = np.squeeze(np.load(os.path.join(latent_dir, 'latents_mean.npy')), axis=(1, 2))
    init_latents = latents[:length_initial_context]
    targt_latents = latents[length_initial_context:length_initial_context + rollout_window]

    print(f"Context latents : {init_latents.shape}")
    print(f"Target latents  : {targt_latents.shape}")

    # Run Rollout
    # Returns (1, T_total, C)
    t_start = time.time()
    predicted_batched = inference_module.rollout(
        init_latents,
        steps_to_predict=rollout_window,
    )
    t_end = time.time()
    print(f"Rollout completed in {t_end - t_start:.2f} s")

    # Squeeze and Slice for Metrics
    # Remove batch dim -> (T_total, C)
    predicted_full = np.array(predicted_batched[0])
    predicted_future = predicted_full[length_initial_context:]

    print(f"Predicted full   : {predicted_full.shape}")
    print(f"Predicted future : {predicted_future.shape}")

    nrmse = (np.sqrt(np.mean((predicted_future - targt_latents) ** 2))
             / np.sqrt(np.mean(targt_latents ** 2)))
    nmae = (np.mean(np.abs(predicted_future - targt_latents))
            / np.mean(np.abs(targt_latents)))
    print(f"Rollout Future NRMSE: {nrmse:.6f}, NMAE: {nmae:.6f}")

    np.save(os.path.join(output_dir, 'predicted_latents.npy'), predicted_future)
    np.save(os.path.join(output_dir, 'target_latents.npy'), targt_latents)
    np.save(os.path.join(output_dir, 'initial_latents.npy'), init_latents)
    print(f"Saved outputs to: {output_dir}")
