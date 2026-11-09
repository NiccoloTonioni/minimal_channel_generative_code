"""
Reconstruction and regularisation losses of the autoencoders (MSE, KL divergence) and auxiliary metrics.
"""
# This script contains all the functions used to define the autoencoder losses.
# -- jax Numpy
import numpy as np
import jax.numpy as jnp
# -- Flax
import flax.linen as nn
# -- Optax
import optax
# -- Scipy
from scipy.stats import wasserstein_distance


# -- Loss functions    
def manattan_loss(x:np.array, y:np.array):
    """Compute the absolute error between the original images and the reconstructed images,
    sum over channels, mean over batches, pixels.
    
    Loss = ||x - y||_1
    
    Args:
        - x: np.array, original images
        - y: np.array, reconstructed images
    """
    loss = jnp.mean(optax.abs(x - y))
    return loss

def mse_loss(x, y):
    """Compute the mean squared error between the original images and the reconstructed images,
    sum over channels, mean over batches, pixels.
    
    Loss = ||x - y||_2
    
    Args:
        - x: np.array, original images
        - y: np.array, reconstructed images
    """
    loss = jnp.mean(optax.squared_error(x,y)) 
    return loss

def mssim_score(x, y):
    """Compute the mean structural similarity index between the original images and the reconstructed images,
    sum over the pixels, mean over batches.
    
    Loss = 1 - MSSIM(x, y)
    
    Args:
        - x: np.array, original images
        - y: np.array, reconstructed images
    """
    def avg_pool_11x11(x):
        return nn.avg_pool(x, window_shape=(11,11), padding='VALID')   
    mu_x = avg_pool_11x11(x)
    mu_y = avg_pool_11x11(y)
    var_x = avg_pool_11x11(x**2) - mu_x**2
    var_y = avg_pool_11x11(y**2) - mu_y**2
    var_xy = avg_pool_11x11(x * y) - mu_x * mu_y
    C1 = 1e-6
    C2 = 1e-6
    # - Compute error in recostructing mean and fluctuations
    SSIM_m = (2 * mu_x * mu_y + C1)/(mu_x ** 2 + mu_y ** 2 + C1)  
    # SSIM_v = (2 * var_x**0.5 * var_y**0.5 +  C2)/(var_x + var_y + C2)
    SSIM_v = (2*var_xy +  C2)/(var_x + var_y + C2)
    SSIM = SSIM_m * SSIM_v
    MSSIM = jnp.mean(SSIM)
    return jnp.clip((1 - MSSIM), 0, 1)

def kl_divergence(mu, logvar):
    """Compute the Kullback-Leibler divergence between the latent space distribution and a standard normal distribution.
    
    Loss = KL(N(mu, logvar) || N(0, 1))
    
    Args:
        - mu: np.array, mean of the latent space distribution
        - logvar: np.array, log-variance of the latent space distribution
    """
    # KL per sample: sum over latent dimensions
    kl_per_sample = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1  )
    # Average over batch
    kl = jnp.mean(kl_per_sample)
    return kl

def compute_wasserstein_1d(images_real, images_reconstructed):
    """Compute the Wasserstein distance between the real images and the reconstructed images."""
    # Flatten images into 1D histograms
    images_real_flat = images_real.reshape(images_real.shape[0], -1)
    images_reconstructed_flat = images_reconstructed.reshape(images_reconstructed.shape[0], -1)
    
    # Compute Wasserstein distance per image and take the average
    distances = [wasserstein_distance(real, rec) for real, rec in zip(images_real_flat, images_reconstructed_flat)]
    
    return np.mean(distances)

