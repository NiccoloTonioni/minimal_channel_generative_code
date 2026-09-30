from typing import Callable
import gin
import jax
import jax.numpy as jnp
from flax import linen as nn
from src.utils.loss_functions import mse_loss, kl_divergence

##############################################################################
#                                  GIN CONFIG                                #
##############################################################################
# Allow gin to register activation functions by name
gin.external_configurable(nn.relu, name="relu")
gin.external_configurable(nn.leaky_relu, name="leaky_relu") 
gin.external_configurable(nn.silu, name="silu")
gin.external_configurable(nn.swish, name="swish")
gin.external_configurable(nn.gelu, name="gelu")

##############################################################################
#                         ARCHITECTURE COMPONENTS                            #
##############################################################################
class Encoder(nn.Module):
    filter_chn: int 
    latent_dim: int
    activation_fn: Callable 

    def setup(self):
        self.conv1 = nn.Conv(features=self.filter_chn, kernel_size=(7,7), padding='CIRCULAR')
        self.conv2 = nn.Conv(features=2*self.filter_chn, kernel_size=(3,3), strides=(2,1), padding='CIRCULAR')
        self.conv3 = nn.Conv(features=4*self.filter_chn, kernel_size=(3,3), strides=2, padding='CIRCULAR')
        self.conv4 = nn.Conv(features=4*self.filter_chn, kernel_size=(3,3), strides=2, padding='CIRCULAR')

    @nn.compact
    def __call__(self, x):
        x = self.conv1(x); x = self.activation_fn(x)
        x = self.conv2(x); x = self.activation_fn(x)
        x = self.conv3(x); x = self.activation_fn(x)
        x = self.conv4(x); x = self.activation_fn(x)

        shape = x.shape[1:3]
        mean = nn.Conv(features=self.latent_dim, kernel_size=(shape[0], shape[1]), padding='VALID')(x)
        logvar = nn.Conv(features=self.latent_dim, kernel_size=(shape[0], shape[1]), padding='VALID')(x)
        return mean, logvar, shape

class Decoder(nn.Module):
    filter_chn: int 
    output_chn: int 
    activation_fn: Callable = nn.silu
    resize_method: str = 'lanczos3'  

    def setup(self):
        self.conv1 = nn.Conv(features=4*self.filter_chn, kernel_size=(3,3), padding='CIRCULAR')
        self.conv2 = nn.Conv(features=2*self.filter_chn, kernel_size=(3,3), padding='CIRCULAR')
        self.conv3 = nn.Conv(features=self.filter_chn, kernel_size=(3,3), padding='CIRCULAR')

    @nn.compact
    def __call__(self, x, shape):
        x = nn.ConvTranspose(features=4*self.filter_chn, kernel_size=(shape[0], shape[1]), padding='VALID')(x)
        x = self.activation_fn(x)

        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[-1]), method=self.resize_method)
        x = self.conv1(x); x = self.activation_fn(x)

        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[-1]), method=self.resize_method)
        x = self.conv2(x); x = self.activation_fn(x)

        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*2, x.shape[2], x.shape[-1]), method=self.resize_method)
        x = self.conv3(x); x = self.activation_fn(x)

        x = nn.Conv(features=self.output_chn, kernel_size=(7,7), padding='CIRCULAR')(x)
        return x

##############################################################################
#                                    MODEL                                   #
##############################################################################
@gin.configurable
class VAE(nn.Module):
    filter_chn: int = 16
    latent_dim: int = 28
    activation_fn: Callable = nn.silu
    resize_method: str = 'lanczos3'  # Made configurable
    output_chn: int = 3

    def setup(self):
        self.encoder = Encoder(
            filter_chn=self.filter_chn, 
            latent_dim=self.latent_dim, 
            activation_fn=self.activation_fn
        )
        self.decoder = Decoder(
            filter_chn=self.filter_chn, 
            output_chn=self.output_chn, 
            activation_fn=self.activation_fn
        )

    def __call__(self, key, x, deterministic: bool = False):
        mean, logvar, shape = self.encoder(x)
        z = mean if deterministic else self.reparameterize(key, mean, logvar)
        x_hat = self.decoder(z, shape)
        return x_hat, mean, logvar

    def reparameterize(self, key, mean, logvar):
        eps = jax.random.normal(key=key, shape=mean.shape)
        return eps * jnp.exp(logvar * 0.5) + mean

    @classmethod
    def __name__(cls):
        return "Beta_Variational_AE"

def vae_loss(vae_params, key, x,
             vae_apply_fn, beta=1e-4):
    """VAE loss function"""
    
    x_hat, mean, logvar = vae_apply_fn({'params': vae_params}, key, x)
    
    # Reconstruction loss
    recon_loss = mse_loss(x, x_hat)
    
    # KL divergence loss
    kl_loss = kl_divergence(mean, logvar)
    
    # Total VAE loss
    loss = recon_loss + beta * kl_loss
    
    return loss, (recon_loss, kl_loss)

def vae_loss(vae_params, key, x, vae_apply_fn, beta=1e-4, deterministic=False):
    """VAE loss function"""
    x_hat, mean, logvar = vae_apply_fn({'params': vae_params}, key, x,
                                       deterministic=deterministic)
    # Reconstruction loss
    recon_loss = mse_loss(x, x_hat)
    # KL divergence loss
    kl_loss = kl_divergence(mean, logvar)
    # Total VAE loss
    loss = recon_loss + beta * kl_loss
    return loss, (recon_loss, kl_loss)