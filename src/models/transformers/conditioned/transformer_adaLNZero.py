import flax.linen as nn
import gin
import jax.numpy as jnp
from src.models.transformers.attention import build_attention

##############################################################################
#                            SHARED ARCHITECTURES                            #
##############################################################################

############################################################
# Time Encoding and Decoding with Convolutions
############################################################
class TimeProjection(nn.Module):
    """
    Temporal feature projection using 1D convolutions with residual connection.
    
    Applies a two-layer convolutional network to time series data. The progressive 
    kernel size increase (k → 2k) allows the network to capture both local and broader 
    temporal patterns. The 1x1 residual convolution matches dimensions without mixing 
    temporal information.
    """
    out_features: int
    hidden_features: int
    kernel_size: int
    use_bias: bool

    def setup(self):
        self.conv1 = nn.Conv(
            features=self.hidden_features, 
            kernel_size=self.kernel_size, 
            padding=[(self.kernel_size - 1, 0)], 
            use_bias=self.use_bias
        )
        self.conv2 = nn.Conv(
            features=self.out_features, 
            kernel_size=2*self.kernel_size, 
            padding=[(2*self.kernel_size - 1, 0)], 
            use_bias=self.use_bias
        )
        self.res_proj = nn.Conv(
            features=self.out_features, 
            kernel_size=1, 
            use_bias=self.use_bias
        ) 

    def __call__(self, x):
        y = self.conv1(x)
        y = nn.gelu(y)
        y = self.conv2(y)
        y = nn.gelu(y)
        residual = self.res_proj(x)
        return residual + y

############################################################
# Positional Encoding
############################################################
class PositionalEncoding(nn.Module):
    max_len: int
    embedding_dim: int

    def __call__(self, x):
        position = jnp.arange(0, self.max_len)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.embedding_dim, 2) * -(jnp.log(10000.0) / self.embedding_dim))
        pe = jnp.zeros((self.max_len, self.embedding_dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        pe = pe[jnp.newaxis, ...]  # Shape (1, max_len, embedding_dim)
        return x + pe[:, :x.shape[1], :]

############################################################
# Feed Forward Network 
############################################################
class MLP(nn.Module):
    n_embd: int
    use_bias: bool
    dropout_rate: float

    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(4 * self.n_embd, use_bias=self.use_bias)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.n_embd, use_bias=self.use_bias)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        return x

##############################################################################
#                    ADALN-ZERO CONDITIONING COMPONENTS                      #
##############################################################################

class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization with modulation parameters.
    Each transformer block has its own AdaLNModulation network.
    Returns 6 values: (shift_attn, scale_attn, gate_attn,
                       shift_mlp, scale_mlp, gate_mlp)
    
    The output layer is zero-initialized (adaLN-Zero, DiT paper): at initialization
    shift = scale = gate = 0, so every block starts as the identity function.
    """
    n_embd: int
    use_bias: bool = True
    
    def setup(self):
        # Two-layer MLP for modulation (following DiT paper)
        self.linear1 = nn.Dense(self.n_embd, use_bias=self.use_bias)
        self.linear2 = nn.Dense(
            6 * self.n_embd, 
            use_bias=self.use_bias,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros
        )
    
    def __call__(self, c):
        """
        Args:
            c: Conditioning tensor (B, T, cond_dim)
        
        Returns:
            Tuple of 6 tensors, each (B, T, n_embd)
        """
        # Project through MLP
        h = self.linear1(c)
        h = nn.silu(h)
        modulation = self.linear2(h)
        # Split into 6 components: (shift, scale, gate) for (attn, mlp)
        return jnp.split(modulation, 6, axis=-1)  # Each: (B, T, n_embd)


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization.
    Applies LayerNorm then modulates with scale and shift from conditioning.
    """
    use_bias: bool = False
    
    def setup(self):
        self.ln = nn.LayerNorm(use_bias=self.use_bias)
    
    def __call__(self, x, shift, scale):
        """
        Args:
            x: Input tensor (B, T, C)
            shift: Shift parameter (B, T, C)
            scale: Scale parameter (B, T, C)
        """
        x = self.ln(x)
        return x * (1 + scale) + shift


##############################################################################
#                         ARCHITECTURE COMPONENTS                            #
##############################################################################

class InputEmbedding(nn.Module):
    embedding_dim: int
    max_len: int
    kernel_size: int 
    use_bias: bool

    def setup(self):
        self.time_encoding = TimeProjection(
                            hidden_features=self.embedding_dim*2, 
                            out_features=self.embedding_dim, 
                            kernel_size=self.kernel_size, 
                            use_bias=self.use_bias
                        )
        self.positional_encoding = PositionalEncoding(self.max_len, self.embedding_dim)

    def __call__(self, x):
        embeddings = self.time_encoding(x)
        return self.positional_encoding(embeddings)


class BlockAdaLN(nn.Module):
    """
    Transformer block with adaLN-Zero conditioning.
    Following DiT paper: each block has its own modulation network.
    """
    n_embd: int
    n_max_seq: int
    n_heads: int
    use_bias: bool
    dropout_rate: float
    attention_type: str = 'self_attention'
    ea_sparse_offset: int = None
    ea_init_alpha: str = 'xavier'
    ea_init_wv: str = 'randn'

    def setup(self):
        # Layer norms with adaptive modulation
        self.adaln_attn = AdaLN(use_bias=self.use_bias)
        self.adaln_mlp = AdaLN(use_bias=self.use_bias)
        
        # Attention and MLP
        self.attn = build_attention(
            attention_type=self.attention_type,
            n_embd=self.n_embd,  
            n_heads=self.n_heads,
            n_max_seq=self.n_max_seq,
            use_bias=False,
            dropout_rate=self.dropout_rate,
            ea_sparse_offset=self.ea_sparse_offset,
            ea_init_alpha=self.ea_init_alpha,
            ea_init_wv=self.ea_init_wv
        )
        self.mlp = MLP(
            n_embd=self.n_embd, 
            use_bias=self.use_bias, 
            dropout_rate=self.dropout_rate
        )
        
        # CRITICAL: Each block has its own modulation network
        self.adaLN_modulation = AdaLNModulation(
            n_embd=self.n_embd,
            use_bias=self.use_bias
        )

    def __call__(self, x, c, mask=None, deterministic=True):
        """
        Args:
            x: Input tensor (B, T, n_embd)
            c: Conditioning tensor (B, T, cond_dim) - passed from main model
            deterministic: Whether to use dropout
        """
        # Compute modulation parameters for this block
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)
        # Attention block with adaLN-Zero
        x_norm = self.adaln_attn(x, shift_attn, scale_attn)
        x = x + gate_attn * self.attn(x_norm, mask=mask, deterministic=deterministic) 
        
        # MLP block with adaLN-Zero
        x_norm = self.adaln_mlp(x, shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm, deterministic=deterministic)
        
        return x


##############################################################################
#                                    MODEL                                   #
##############################################################################
@gin.configurable
class TimeTransformerAdaLNZero(nn.Module): 
    """
    Transformer for time series forecasting with adaLN-Zero conditioning.
    Implements conditioning following "Scalable Diffusion Models with Transformers" (DiT).
    
    Each transformer block has its own modulation network that processes
    the conditioning signal, allowing different blocks to extract different features.
    
    Args:
        block_size: Maximum sequence length the model can handle
        n_layer: Number of transformer blocks
        n_heads: Number of attention heads
        n_embd: Embedding dimension
        cond_dim: Dimension of conditioning signal after projection
        use_bias: Whether to use bias in linear layers
        dropout_rate: Dropout probability
        attention_type: 'self_attention', 'easy_attention' or 'self_easy_attention'
        ea_sparse_offset: Easy attention banded alpha offset (None = dense alpha)
        ea_init_alpha: Easy attention alpha initialization ('eye', 'xavier', 'randn')
        ea_init_wv: Easy attention value projection initialization ('xavier', 'randn')
        output_mode: How to return predictions:
            - "last_step": Single next step (B, C)
            - "full_sequence": All positions (B, T, C)
    """
    block_size: int = 1000
    n_layer: int = 12
    n_heads: int = 12
    n_embd: int = 128
    k_emb: int = 3
    cond_dim: int = 64  # Dimension of conditioning signal after embedding
    k_cond : int = 3
    use_bias: bool = False
    dropout_rate: float = 0.0
    attention_type: str = 'self_attention'
    ea_sparse_offset: int = None
    ea_init_alpha: str = 'xavier'
    ea_init_wv: str = 'randn'
    output_mode: str = "last_step"

    def setup(self):
        # Input embedding
        self.signal_embed = InputEmbedding(
            embedding_dim=self.n_embd, 
            max_len=self.block_size,
            kernel_size=self.k_emb,
            use_bias=self.use_bias
        )
        
        # Conditioning embedding - projects sensor channels to cond_dim
        # Dense applies per-timestep automatically for (B, T, C) inputs
        # self.cond_embed = nn.Dense(self.cond_dim, use_bias=self.use_bias)
        self.cond_embed = TimeProjection(
            out_features=self.cond_dim,
            hidden_features=self.cond_dim*2,
            kernel_size=self.k_cond, # Looks at t and the 12 previous steps
            use_bias=self.use_bias
        )
        
        # Transformer blocks with adaLN
        # Each block has its own modulation network inside
        self.h = [
            BlockAdaLN(
                n_embd=self.n_embd, 
                n_max_seq=self.block_size,
                n_heads=self.n_heads, 
                use_bias=self.use_bias, 
                dropout_rate=self.dropout_rate,
                attention_type=self.attention_type,
                ea_sparse_offset=self.ea_sparse_offset,
                ea_init_alpha=self.ea_init_alpha,
                ea_init_wv=self.ea_init_wv
            ) for _ in range(self.n_layer)
        ]
        
        # Final layer norm with conditioning
        self.adaln_final = AdaLN(use_bias=self.use_bias)
        self.final_modulation = AdaLNModulation(
            n_embd=self.n_embd,
            use_bias=self.use_bias
        )
        
        self.dropout = nn.Dropout(self.dropout_rate)

    @nn.compact
    def __call__(self, x, y, mask=None, deterministic=True):
        """
        Forward pass with temporal conditioning.
        
        Args:
            x: Input latent tensor (B, T, C_latent)
            y: Conditioning sensor signal (B, T, C_sensor) - temporal conditioning
            mask: Optional boolean mask for variable-length sequences (B, T)
            deterministic: Whether to use dropout
            
        Returns:
            Predictions with shape depending on output_mode
        """
        B, T, C = x.shape
        assert T <= self.block_size, (
            f"Cannot forward sequence of length {T}, block_size is only: {self.block_size}"
        )
        
        # Verify sensor shape
        assert y.ndim == 3, f"Sensor conditioning must be 3D (B, T, C_sensor), got shape {y.shape}"
        assert y.shape[0] == B and y.shape[1] == T, (
            f"Sensor shape mismatch: expected ({B}, {T}, C_sensor), got {y.shape}"
        )

        # Embed input signal
        x = self.signal_embed(x)
        x = self.dropout(x, deterministic=deterministic)
        
        # Embed conditioning signal - Dense automatically applies per-timestep
        # Input: (B, T, C_sensor) → Output: (B, T, cond_dim)
        c = self.cond_embed(y)

        # Prepare mask for attention (if provided)
        if mask is None:
            mask = jnp.ones((x.shape[0], x.shape[1]), dtype=jnp.bool_)
        
        # Apply transformer blocks with temporal conditioning
        # Each block processes c through its own modulation network
        for block in self.h:
            x = block(x, c, mask=mask, deterministic=deterministic)

        # Final adaptive layer norm with its own modulation
        shift_final, scale_final, _, _, _, _ = self.final_modulation(c)
        x = self.adaln_final(x, shift_final, scale_final)
        
        # Decode to output space
        y_pred = TimeProjection(
            out_features=C,
            hidden_features=self.n_embd,
            kernel_size=3,
            use_bias=self.use_bias
        )(x)  # Shape: (B, T, C_latent)
        
        # Select output based on mode
        if self.output_mode == "last_step":
            return y_pred[:, -1, :]  # (B, C_latent)
        elif self.output_mode == "full_sequence":
            return y_pred  # (B, T, C_latent)
        else:
            raise ValueError(
                f"Invalid output_mode: {self.output_mode}. "
                f"Must be 'last_step' or 'full_sequence'"
            )

    @classmethod
    def __name__(cls):
        return "Transformer-Time-Forecaster-AdaLN"