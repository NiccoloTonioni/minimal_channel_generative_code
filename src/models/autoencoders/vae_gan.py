from typing import Callable
import gin
import jax
import jax.numpy as jnp
from flax import linen as nn
from src.utils.loss_functions import mse_loss, kl_divergence
import optax

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

@gin.configurable
class Discriminator(nn.Module):
    filter_chn: int 
    activation_fn: Callable = nn.relu 

    def setup(self):
        self.conv1 = nn.Conv(features=self.filter_chn, kernel_size=(4,4), strides=(2,1), padding='CIRCULAR')
        self.conv2 = nn.Conv(features=2*self.filter_chn, kernel_size=(3,3), strides=2, padding='CIRCULAR')
        self.conv3 = nn.Conv(features=4*self.filter_chn, kernel_size=(3,3), strides=2, padding='CIRCULAR')
        self.conv4 = nn.Conv(features=8*self.filter_chn, kernel_size=(3,3), padding='CIRCULAR')

    @nn.compact
    def __call__(self, x, is_train=True):
        # Layer 1
        x = nn.SpectralNorm(self.conv1)(x, update_stats=is_train)
        x = self.activation_fn(x)
        # Layer 2
        x = nn.SpectralNorm(self.conv2)(x, update_stats=is_train)
        x = self.activation_fn(x)
        # Layer 3
        x = nn.SpectralNorm(self.conv3)(x, update_stats=is_train)
        x = self.activation_fn(x)
        # Layer 3
        x = nn.SpectralNorm(self.conv4)(x, update_stats=is_train)
        x = self.activation_fn(x)
        # Final layer to single output
        logits = nn.Conv(features=1, kernel_size=(1,1), padding='SAME')(x)
        return logits
    
##############################################################################
#                                    MODEL                                   #
##############################################################################
@gin.configurable
class VAE_GAN(nn.Module):
    filter_chn: int = 16
    latent_dim: int = 28
    activation_fn: Callable = nn.silu
    resize_method: str = 'lanczos3'
    output_chn: int = 3

    def setup(self):
        self.encoder = Encoder(
            filter_chn=self.filter_chn, 
            latent_dim=self.latent_dim, 
            activation_fn=self.activation_fn,
        )
        self.decoder = Decoder(
            filter_chn=self.filter_chn, 
            output_chn=self.output_chn, 
            activation_fn=self.activation_fn,
            resize_method=self.resize_method,
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
        return "VAE_GAN"

##############################################################################
#                                   LOSSES                                   #
##############################################################################
def vae_loss(vae_params, discriminator_params, discriminator_batch_stats, key, batch, 
             vae_apply_fn, discriminator_apply_fn, 
             beta=1e-4, gamma=1e-4, deterministic=False):
    """VAE loss with adversarial component"""
    x_hat, mean, logvar = vae_apply_fn({'params': vae_params}, key, batch,
                                       deterministic=deterministic)

    # Reconstruction loss
    distortion_loss = mse_loss(batch, x_hat)
    
    # KL divergence loss
    kl_loss = kl_divergence(mean, logvar)
    
    # Adversarial loss (fool the discriminator)
    # Build discriminator state dict properly
    if discriminator_batch_stats is not None:
        disc_state = {'params': discriminator_params, 'batch_stats': discriminator_batch_stats}
    else:
        disc_state = {'params': discriminator_params}
    
    fake_logits = discriminator_apply_fn(
        disc_state,
        x_hat, 
        is_train=False,  # Don't update batch stats during VAE training
        mutable=False    # Explicitly don't allow mutations
    )
    
    adversarial_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(
        logits=fake_logits, labels=jnp.ones_like(fake_logits)))
    
    # Total loss
    total_loss = distortion_loss + beta * kl_loss + gamma * adversarial_loss
    
    return total_loss, (distortion_loss, kl_loss, adversarial_loss)

def discriminator_loss(discriminator_params, discriminator_batch_stats, vae_params, key, batch, 
                      discriminator_apply_fn, vae_apply_fn):
    """Discriminator loss function with proper spectral norm handling"""
    
    # Real data - update spectral norm stats
    real_output, real_updates = discriminator_apply_fn(
        {'params': discriminator_params, 'batch_stats': discriminator_batch_stats}, 
        batch, 
        is_train=True,
        mutable=['batch_stats']
    )
    
    # Generate fake data
    x_hat, _, _ = vae_apply_fn({'params': vae_params}, key, batch)
    
    # Fake data - use updated batch stats but don't update them further
    fake_logits = discriminator_apply_fn(
        {'params': discriminator_params, 'batch_stats': real_updates['batch_stats']},
        jax.lax.stop_gradient(x_hat), 
        is_train=False,
        mutable=False
    )
    
    # Losses
    real_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(
        logits=real_output, labels=jnp.ones_like(real_output)))
    fake_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(
        logits=fake_logits, labels=jnp.zeros_like(fake_logits)))
    
    total_loss = real_loss + fake_loss
    
    # Return loss and updated batch stats
    return total_loss, real_updates['batch_stats']