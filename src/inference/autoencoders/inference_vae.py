import os
from typing import Optional, Tuple
# -- Flax
from flax.training import orbax_utils
# -- JAX
import jax
import jax.numpy as jnp
# -- Numpy
import numpy as np
# -- Orbax for checkpoints
import orbax.checkpoint as ocp


###################################################################################################
# Inference class for loading and using trained VAE models
###################################################################################################
class VAEInference:
    """Lightweight, jitted inference for VAE models.

    Shapes are no longer hard-coded. The input shape (H, W, C) is given at construction
    (save_vae_reconstructions infers it from the first test batch), and the latent
    spatial shape needed by the decoder is computed once from the encoder.
    """

    def __init__(self, vae_model, checkpoint_dir: str,
                 input_shape: Tuple[int, int, int] = (48, 24, 3)):
        """
        Args:
            vae_model: The VAE model architecture
            checkpoint_dir: Path to the run directory containing best_model_params
            input_shape: Shape of one sample (H, W, C), without the batch dimension
        """
        self.vae_model = vae_model
        self.checkpoint_dir = checkpoint_dir
        self.input_shape = tuple(int(s) for s in input_shape)

        best_params_dir = os.path.join(checkpoint_dir, 'best_model_params')
        if not os.path.exists(best_params_dir):
            raise ValueError(f"Best model parameters directory not found: {best_params_dir}")
        print(f"Setting up checkpoint manager for best parameters from: {best_params_dir}")

        # Same best-tracking options as in training, so best_step() uses the metric
        options = ocp.CheckpointManagerOptions(
            best_fn=lambda metrics: metrics['chckpt_loss'],
            best_mode='min',
        )
        self.params_manager = ocp.CheckpointManager(
            best_params_dir,
            checkpointers=ocp.PyTreeCheckpointer(),
            options=options
        )

        self._load_best_parameters()
        self.latent_hw = self._infer_latent_hw()
        self._build_jitted_fns()

    ##############################################################################
    #                                   SETUP                                    #
    ##############################################################################
    def _dummy_batch(self) -> jnp.ndarray:
        return jnp.zeros((1, *self.input_shape), dtype=jnp.float32)

    def _load_best_parameters(self):
        """Load the best model parameters"""
        target_step = self.params_manager.best_step()
        if target_step is None:
            raise ValueError("No best model parameters found in checkpoint directory")

        print(f"Loading best model parameters from step {target_step}")

        # Parameter structure from a dummy batch of the real input shape.
        # If input_shape does not match the training data, the restore fails loudly
        # because the parameter shapes (e.g. the latent conv kernels) will differ.
        key = jax.random.PRNGKey(0)
        vae_params = self.vae_model.init(key, key, self._dummy_batch())['params']

        restore_target = {'vae_params': vae_params, 'epoch': 0}
        restore_args = orbax_utils.restore_args_from_target(restore_target)

        restored_data = self.params_manager.restore(
            target_step,
            items=restore_target,
            restore_kwargs={'restore_args': restore_args}
        )
        if restored_data is None:
            raise ValueError("Failed to restore best model parameters")

        self.vae_params = restored_data['vae_params']
        self.epoch = restored_data['epoch']
        print(f"Successfully loaded parameters from epoch {self.epoch}")

    def _infer_latent_hw(self) -> Tuple[int, int]:
        """Run the encoder once (eagerly) to get the spatial shape the decoder needs."""
        _, _, hw = self.vae_model.apply(
            {'params': self.vae_params},
            self._dummy_batch(),
            method=lambda m, x: m.encoder(x)
        )
        return tuple(int(s) for s in hw)

    def _build_jitted_fns(self):
        """Compile encode/decode/reconstruct/sample. Params are passed as arguments
        (not closed over) so they are not baked into the compiled program as constants."""
        model = self.vae_model
        latent_hw = self.latent_hw  # Python tuple, closed over, hence static under jit

        def _encode(params, x):
            mean, logvar, _ = model.apply(
                {'params': params}, x, method=lambda m, x: m.encoder(x))
            return mean, logvar

        def _decode(params, z):
            return model.apply(
                {'params': params}, z, latent_hw, method=lambda m, z, s: m.decoder(z, s))

        def _reconstruct(params, x):
            mean, _ = _encode(params, x)
            return _decode(params, mean)

        def _sample(params, key, x):
            return model.apply({'params': params}, key, x)[0]

        self._encode_jit = jax.jit(_encode)
        self._decode_jit = jax.jit(_decode)
        self._reconstruct_jit = jax.jit(_reconstruct)
        self._sample_jit = jax.jit(_sample)

    def _check_input(self, data):
        if tuple(data.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Input sample shape {tuple(data.shape[1:])} does not match the "
                f"shape the model was loaded with {self.input_shape}")

    ##############################################################################
    #                                 PUBLIC API                                 #
    ##############################################################################
    def encode(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[int, int]]:
        """Encode data to latent space (returns mean, logvar, and latent spatial shape)"""
        self._check_input(data)
        mean, logvar = self._encode_jit(self.vae_params, data)
        return mean, logvar, self.latent_hw

    def decode(self, latents: jnp.ndarray, shape: Optional[Tuple[int, int]] = None) -> jnp.ndarray:
        """Decode latents to reconstructed data. `shape` is kept for backward compatibility
        and is only checked against the inferred latent shape."""
        if shape is not None and tuple(int(s) for s in shape) != self.latent_hw:
            raise ValueError(f"Latent shape {tuple(shape)} does not match {self.latent_hw}")
        return self._decode_jit(self.vae_params, latents)

    def reconstruct(self, data: jnp.ndarray) -> jnp.ndarray:
        """Full reconstruction using the encoder mean (no sampling)"""
        self._check_input(data)
        return self._reconstruct_jit(self.vae_params, data)

    def sample(self, data: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
        """Full reconstruction with latent sampling"""
        self._check_input(data)
        return self._sample_jit(self.vae_params, key, data)


###################################################################################################
# Function to save reconstructions using a trained VAE model
###################################################################################################
def save_vae_reconstructions(vae_model=None,
                             checkpoint_dir=None,
                             test_loader=None):
    """Generate and save reconstructions of a dataset using a trained VAE"""
    print("Setting up for VAE reconstruction saving...")

    if vae_model is None:
        raise ValueError("vae_model must be provided.")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir must be provided.")
    if test_loader is None:
        raise ValueError("test_loader must be provided.")

    # Infer the sample shape from the data instead of hard-coding it.
    # Assumes test_loader is re-iterable (a fresh iter() restarts from the beginning).
    first_batch = next(iter(test_loader))
    input_shape = tuple(first_batch.shape[1:])

    inference_module = VAEInference(
        vae_model=vae_model,
        checkpoint_dir=checkpoint_dir,
        input_shape=input_shape
    )

    print("Starting reconstruction saving...")

    output_dir_name = f"reconstructions_epoch_{inference_module.epoch}"
    save_base_dir = os.path.join(checkpoint_dir, output_dir_name)
    os.makedirs(save_base_dir, exist_ok=True)

    all_reconstructed_samples = []
    all_encoded_latents = []

    if jax.process_index() == 0:
        print('# ----- Saving reconstructions')

    total_batches = len(test_loader)

    for batch_idx, batch_data in enumerate(test_loader):
        print(f'Processing batch {batch_idx+1}/{total_batches}', end='\r')

        # Jitted encode and decode from the mean (deterministic)
        mean, _, _ = inference_module.encode(batch_data)
        reconstructed_data = inference_module.decode(mean)

        all_encoded_latents.append(jax.device_get(mean))
        all_reconstructed_samples.append(jax.device_get(reconstructed_data))

    if jax.process_index() == 0:
        print("\nConcatenating and saving all collected samples...")

    final_encoded_latents = np.concatenate(all_encoded_latents, axis=0)
    final_reconstructed_data = np.concatenate(all_reconstructed_samples, axis=0)

    latents_filepath = os.path.join(save_base_dir, "all_encoded_latents.npy")
    reconstructed_filepath = os.path.join(save_base_dir, "all_reconstructions.npy")

    np.save(latents_filepath, final_encoded_latents)
    np.save(reconstructed_filepath, final_reconstructed_data)

    if jax.process_index() == 0:
        print(f"Saved all encoded latents to: {latents_filepath}")
        print(f"Saved all reconstructed samples to: {reconstructed_filepath}")
        print("Reconstruction saving complete.")

    return inference_module