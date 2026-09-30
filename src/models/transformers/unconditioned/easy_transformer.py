import flax.linen as nn
import gin
import jax
import jax.numpy as jnp

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
        self.conv1 = nn.Conv(features=self.hidden_features, kernel_size=self.kernel_size, use_bias=self.use_bias)
        self.conv2 = nn.Conv(features=self.out_features, kernel_size=2*self.kernel_size, use_bias=self.use_bias)
        self.res_proj = nn.Conv(features=self.out_features, kernel_size=1, use_bias=self.use_bias)

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
# Attention Mechanism 
############################################################
class EasyAttn(nn.Module):
    n_embd: int
    n_max_seq: int
    n_heads: int
    dropout_rate: float
    use_bias : bool
    sparse_offset: int = None
    init_alpha: str = "xavier"
    init_wv: str = "randn"
    
    def setup(self):
        wv_init = nn.initializers.xavier_uniform() if self.init_wv == "xavier" else nn.initializers.normal()
        self.WV = self.param('WV', wv_init, (self.n_embd, self.n_embd))
        
        if self.sparse_offset is None:
            self.Alpha = self._init_dense_alpha()
        else:
            self.diag_params = self._init_sparse_diags()
        
        self.c_proj = nn.Dense(features=self.n_embd, use_bias=self.use_bias)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _init_dense_alpha(self):
        if self.init_alpha == "eye":
            def eye_init(_key, _shape):
                return jnp.stack([jnp.eye(self.n_max_seq) for _ in range(self.n_heads)])
            alpha_init = eye_init
        elif self.init_alpha == "xavier":
            alpha_init = nn.initializers.xavier_uniform()
        else:
            alpha_init = nn.initializers.normal()
        return self.param('Alpha', alpha_init, (self.n_heads, self.n_max_seq, self.n_max_seq))
    
    def _init_sparse_diags(self):
        diag_params = []
        for i in range(self.sparse_offset + 1):
            init_val = 1.0 - i * 0.1
            diag_len = self.n_max_seq - i
            diag_params.append(
                self.param(f'diag_{i}', 
                          lambda key, shape, val=init_val: jnp.ones(shape) * val,
                          (self.n_heads, diag_len))
            )
        return diag_params
    
    def _assemble_sparse_alpha(self, T):
        def build_head_matrix(head_idx):
            matrix = jnp.zeros((T, T))  
            for diag_idx, diag_param in enumerate(self.diag_params):
                diag_vals = diag_param[head_idx, :T-diag_idx]  
                if diag_idx == 0:
                    matrix += jnp.diag(diag_vals, k=0)
                else:
                    matrix += jnp.diag(diag_vals, k=diag_idx) + jnp.diag(diag_vals, k=-diag_idx)
            return matrix
        return jax.vmap(build_head_matrix)(jnp.arange(self.n_heads))

    def __call__(self, x, mask=None, deterministic=True): # <-- 2. ADD MASK ARGUMENT
        B, T, C = x.shape

        x = jnp.dot(x, self.WV.T)
        x = x.reshape(B, self.n_heads, T, C // self.n_heads)

        if self.sparse_offset is None:
            Alpha = self.Alpha[:, :T, :T]
        else:
            Alpha = self._assemble_sparse_alpha(T)  

        # Combine causal and padding masks
        causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
        
        if mask is not None:
            # mask shape (B, T) -> pad_mask shape (B, 1, 1, T_src)
            pad_mask = mask[:, None, None, :]
            valid_mask = causal_mask[None, None, :, :] & pad_mask
        else:
            valid_mask = causal_mask[None, None, :, :]

        # Because EasyAttn doesn't use Softmax, we mask with 0.0 instead of -inf
        Alpha_masked = jnp.where(valid_mask, Alpha[None, ...], 0.0)

        # Einsum for batched data
        # x: (B, H, T_src, HeadDim) -> bhsd
        # Alpha_masked: (B, H, T_dest, T_src) -> bhts
        # Output: (B, H, T_dest, HeadDim) -> bhtd
        x = jnp.einsum('bhsd,bhts->bhtd', x, Alpha_masked)

        x = x.reshape(B, T, C)
        x = self.dropout(self.c_proj(x), deterministic=deterministic)

        return x

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
    sparse_offset: int
    use_bias: bool
    dropout_rate: float

    def setup(self):
        self.ln_1 = nn.LayerNorm(use_bias=self.use_bias)
        self.attn = EasyAttn(n_embd=self.n_embd,  
                            n_max_seq=self.n_max_seq,
                            n_heads=self.n_heads,
                            dropout_rate=self.dropout_rate,
                            use_bias=self.use_bias,
                            sparse_offset=self.sparse_offset, 
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
class EasyTimeTransformer(nn.Module): 
    block_size: int = 1000
    n_layer: int = 12   
    n_heads: int = 12
    n_embd: int = 12
    emb_kernel_size: int = 6
    sparse_offset: int = None
    use_bias: bool = False
    dropout_rate: float = 0.0
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
                    sparse_offset=self.sparse_offset,
                    use_bias=self.use_bias, 
                    dropout_rate=self.dropout_rate
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
        return "Transformer-Time-Forecaster-EasyAttention"