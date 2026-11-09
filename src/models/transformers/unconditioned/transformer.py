"""
Transformer of the latent dynamics without sensor conditioning.
"""
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
    Strictly uses causal padding to prevent future data leakage.
    """
    out_features: int
    hidden_features: int
    kernel_size: int
    use_bias: bool

    def setup(self):
        # 1. THE LEAK FIX: Explicit CAUSAL padding
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
        pe = pe[jnp.newaxis, ...]  
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
#                         ARCHITECTURE COMPONENTS                            #
##############################################################################
class InputEmbedding(nn.Module):
    embedding_dim: int
    kernel_size: int    
    max_len: int
    use_bias: bool

    def setup(self):
        self.time_encoding = TimeProjection(
                            hidden_features=self.embedding_dim//2, 
                            out_features=self.embedding_dim, 
                            kernel_size=self.kernel_size, 
                            use_bias=self.use_bias
                        )
        self.positional_encoding = PositionalEncoding(self.max_len, self.embedding_dim)

    def __call__(self, x):
        embeddings = self.time_encoding(x)
        return self.positional_encoding(embeddings)

class Block(nn.Module):
    n_embd: int
    n_max_seq: int
    n_heads: int
    use_bias: bool
    dropout_rate: float
    attention_type: str = 'easy_attention'
    ea_sparse_offset: int = None
    ea_init_alpha: str = 'xavier'
    ea_init_wv: str = 'randn'

    def setup(self):
        self.ln_1 = nn.LayerNorm(use_bias=self.use_bias)
        self.attn = build_attention(
                            attention_type=self.attention_type,
                            n_embd=self.n_embd,  
                            n_heads=self.n_heads,
                            n_max_seq=self.n_max_seq,
                            use_bias=self.use_bias,
                            dropout_rate=self.dropout_rate,
                            ea_sparse_offset=self.ea_sparse_offset,
                            ea_init_alpha=self.ea_init_alpha,
                            ea_init_wv=self.ea_init_wv
                    )
        self.ln_2 = nn.LayerNorm(use_bias=self.use_bias)
        self.mlp = MLP(n_embd=self.n_embd, 
                       use_bias=self.use_bias, 
                       dropout_rate=self.dropout_rate
                    )

    def __call__(self, x, mask=None, deterministic=True): 
        x = x + self.attn(self.ln_1(x), mask=mask, deterministic=deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
        return x


##############################################################################
#                                    MODEL                                   #
##############################################################################
@gin.configurable
class TimeTransformer(nn.Module): 
    """
    Transformer for time series forecasting (no conditioning).
    
    Args:
        attention_type: 'self_attention', 'easy_attention' or 'self_easy_attention'
        ea_sparse_offset: Easy attention banded alpha offset (None = dense alpha)
        ea_init_alpha: Easy attention alpha initialization ('eye', 'xavier', 'randn')
        ea_init_wv: Easy attention value projection initialization ('xavier', 'randn')
    """
    block_size: int = 1000
    n_layer: int = 12   
    n_heads: int = 12
    n_embd: int = 12
    emb_kernel_size: int = 6
    use_bias: bool = False
    dropout_rate: float = 0.0
    attention_type: str = 'easy_attention'
    ea_sparse_offset: int = None
    ea_init_alpha: str = 'xavier'
    ea_init_wv: str = 'randn'
    output_mode: str = "last_step"

    def setup(self):
        self.signal_embed = InputEmbedding(
            embedding_dim=self.n_embd, 
            kernel_size=self.emb_kernel_size,
            max_len=self.block_size, 
            use_bias=self.use_bias
        )
        self.h = [
            Block(n_embd=self.n_embd,
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
        self.ln_f = nn.LayerNorm(use_bias=self.use_bias) 
        self.dropout = nn.Dropout(self.dropout_rate)

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True): # <-- 6. ADD MASK ARGUMENT
        B, T, C = x.shape
        assert T <= self.block_size, (
            f"Cannot forward sequence of length {T}, block_size is only: {self.block_size}"
        )
        
        # 7. IF NO MASK PROVIDED, DEFAULT TO ALL TRUE
        if mask is None:
            mask = jnp.ones((B, T), dtype=jnp.bool_)

        x = self.signal_embed(x)
        x = self.dropout(x, deterministic=deterministic)

        for block in self.h:
            # 8. PASS MASK TO BLOCKS
            x = block(x, mask=mask, deterministic=deterministic)

        x = self.ln_f(x)
        
        y = TimeProjection(
            out_features=C,
            hidden_features=self.n_embd,
            kernel_size=3,
            use_bias=self.use_bias
        )(x)  
        
        if self.output_mode == "last_step":
            return y[:, -1, :]  
        elif self.output_mode == "full_sequence":
            return y  
        else:
            raise ValueError(
                f"Invalid output_mode: {self.output_mode}. "
                f"Must be 'last_step', 'last_n', or 'full_sequence'"
            )

    @classmethod
    def __name__(cls):
        return "Transformer-Time-Forecaster"