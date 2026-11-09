"""
Attention mechanisms of the Transformers: causal self-attention (SA), easy-attention (EA) and their
parallel combination (SA+EA), selected with build_attention.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import math

##############################################################################
#                      ATTENTION MECHANISMS (SHARED)                         #
##############################################################################
# Available attention types:
#   - 'self_attention'      : Standard causal multi-head self-attention (SA)
#   - 'easy_attention'      : Content-independent causal temporal mixing (EA)
#   - 'self_easy_attention' : Hybrid, SelfAttn(X) + EasyAttn(X) in parallel
ATTENTION_TYPES = ('self_attention', 'easy_attention', 'self_easy_attention')


def _build_valid_mask(T, mask=None):
    """
    Combine the causal mask with the optional padding mask.

    Args:
        T: Sequence length
        mask: Optional boolean padding mask (B, T)

    Returns:
        Boolean mask broadcastable to (B, H, T_dest, T_src)
    """
    # Base Causal Mask (Tokens can only attend to past/current tokens)
    causal_mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))

    # Add Padding Mask for Curriculum Learning
    if mask is not None:
        # mask shape is (B, T) -> reshape to (B, 1, 1, T) to mask the keys
        pad_mask = mask[:, None, None, :]
        # Combine masks: Valid if causal AND not padded
        return causal_mask[None, None, :, :] & pad_mask
    return causal_mask[None, None, :, :]


############################################################
# Self-Attention
############################################################
class CausalSelfAttention(nn.Module):
    """
    Standard causal multi-head self-attention:
        head_h = softmax(Q_h K_h^T / sqrt(d_h) + M) V_h
        SelfAttn(X) = Concat(head_1, ..., head_H) W_O
    """
    n_embd: int
    n_heads: int
    use_bias: bool
    dropout_rate: float

    def setup(self):
        self.c_attn = nn.Dense(features=3 * self.n_embd, use_bias=self.use_bias)
        self.c_proj = nn.Dense(features=self.n_embd, use_bias=self.use_bias)
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, mask=None, deterministic=True):
        B, T, C = x.shape
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)

        # (B, T, C) -> (B, H, T, d_h)
        q = q.reshape(B, T, self.n_heads, C // self.n_heads).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, C // self.n_heads).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, C // self.n_heads).transpose(0, 2, 1, 3)

        att = jnp.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(k.shape[-1])

        # Apply causal + padding mask
        att = jnp.where(_build_valid_mask(T, mask), att, -jnp.inf)
        att = nn.softmax(att, axis=-1)
        att = self.dropout(att, deterministic=deterministic)

        y = jnp.einsum('bhqk,bhkd->bhqd', att, v)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.dropout(self.c_proj(y), deterministic=deterministic)
        return y


############################################################
# Easy Attention
############################################################
class EasyAttention(nn.Module):
    """
    Content-independent causal temporal mixing, attention weights are learned parameters:
        EasyAttn(X) = Concat_h [alpha_h V_h] W_O,   alpha_h in R^{T x T}
    with alpha_h causally masked (upper-triangular entries suppressed).

    Args:
        n_max_seq: Maximum sequence length (size of alpha)
        sparse_offset: If None, alpha is dense. Otherwise alpha is banded and only the
                       diagonals up to sparse_offset are learned.
        init_alpha: 'eye', 'xavier' or 'randn' (dense alpha only)
        init_wv: 'xavier' or 'randn'
    """
    n_embd: int
    n_max_seq: int
    n_heads: int
    dropout_rate: float
    use_bias: bool
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

    def __call__(self, x, mask=None, deterministic=True):
        B, T, C = x.shape

        # Value projection and split into heads: (B, T, C) -> (B, H, T, d_h)
        v = jnp.dot(x, self.WV.T)
        v = v.reshape(B, T, self.n_heads, C // self.n_heads).transpose(0, 2, 1, 3)

        if self.sparse_offset is None:
            Alpha = self.Alpha[:, :T, :T]
        else:
            Alpha = self._assemble_sparse_alpha(T)

        # Because EasyAttention doesn't use Softmax, we mask with 0.0 instead of -inf
        Alpha_masked = jnp.where(_build_valid_mask(T, mask), Alpha[None, ...], 0.0)

        # v: (B, H, T_src, d_h) -> bhsd
        # Alpha_masked: (B, H, T_dest, T_src) -> bhts
        # Output: (B, H, T_dest, d_h) -> bhtd
        y = jnp.einsum('bhsd,bhts->bhtd', v, Alpha_masked)

        # Merge heads: (B, H, T, d_h) -> (B, T, C)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.dropout(self.c_proj(y), deterministic=deterministic)
        return y


############################################################
# Hybrid Self + Easy Attention
############################################################
class SelfEasyAttention(nn.Module):
    """
    Hybrid operator using content-dependent and content-independent mixing in parallel:
        Attn(X) = SelfAttn(X) + EasyAttn(X)
    Each branch has its own output projection W_O.
    """
    n_embd: int
    n_max_seq: int
    n_heads: int
    dropout_rate: float
    use_bias: bool
    sparse_offset: int = None
    init_alpha: str = "xavier"
    init_wv: str = "randn"

    def setup(self):
        self.self_attn = CausalSelfAttention(
            n_embd=self.n_embd,
            n_heads=self.n_heads,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate
        )
        self.easy_attn = EasyAttention(
            n_embd=self.n_embd,
            n_max_seq=self.n_max_seq,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            use_bias=self.use_bias,
            sparse_offset=self.sparse_offset,
            init_alpha=self.init_alpha,
            init_wv=self.init_wv
        )

    def __call__(self, x, mask=None, deterministic=True):
        return (self.self_attn(x, mask=mask, deterministic=deterministic) +
                self.easy_attn(x, mask=mask, deterministic=deterministic))


############################################################
# Factory
############################################################
def build_attention(attention_type, n_embd, n_heads, n_max_seq, use_bias, dropout_rate,
                    ea_sparse_offset=None, ea_init_alpha="xavier", ea_init_wv="randn"):
    """
    Build the attention module selected by attention_type.

    Args:
        attention_type: 'self_attention', 'easy_attention' or 'self_easy_attention'
        n_embd: Embedding dimension
        n_heads: Number of attention heads
        n_max_seq: Maximum sequence length (used by easy attention)
        use_bias: Whether to use bias in linear layers
        dropout_rate: Dropout probability
        ea_sparse_offset: Easy attention banded alpha offset (None = dense alpha)
        ea_init_alpha: Easy attention alpha initialization ('eye', 'xavier', 'randn')
        ea_init_wv: Easy attention value projection initialization ('xavier', 'randn')
    """
    if n_embd % n_heads != 0:
        raise ValueError(f"n_embd ({n_embd}) must be divisible by n_heads ({n_heads})")

    if attention_type == 'self_attention':
        return CausalSelfAttention(
            n_embd=n_embd,
            n_heads=n_heads,
            use_bias=use_bias,
            dropout_rate=dropout_rate
        )

    easy_kwargs = dict(
        n_embd=n_embd,
        n_max_seq=n_max_seq,
        n_heads=n_heads,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        sparse_offset=ea_sparse_offset,
        init_alpha=ea_init_alpha,
        init_wv=ea_init_wv
    )
    if attention_type == 'easy_attention':
        return EasyAttention(**easy_kwargs)
    elif attention_type == 'self_easy_attention':
        return SelfEasyAttention(**easy_kwargs)
    else:
        raise ValueError(
            f"Unknown attention_type: {attention_type}. Must be one of {ATTENTION_TYPES}"
        )
