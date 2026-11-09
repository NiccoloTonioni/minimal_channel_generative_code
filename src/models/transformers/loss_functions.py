"""
Forecasting losses of the Transformers: step-wise MSE / MAE / Huber losses, the time-weighted loss of
Eq. (24) (`tw_*` loss types) and per-time-step / per-channel diagnostics.
"""
import jax.numpy as jnp

##############################################################################
#                    LOSS FUNCTIONS FOR VAE LATENT SPACE                     #
##############################################################################

def get_raw_loss(predictions, targets, loss_type='mse'):
    """Helper function to compute raw unreduced loss."""
    if loss_type == 'mse':
        return (predictions - targets) ** 2
    elif loss_type == 'mae':
        return jnp.abs(predictions - targets)
    elif loss_type == 'huber':
        delta = 1.0
        abs_diff = jnp.abs(predictions - targets)
        quadratic = jnp.minimum(abs_diff, delta)
        linear = abs_diff - quadratic
        return 0.5 * quadratic ** 2 + delta * linear
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'mse', 'mae', or 'huber' "
                         f"(or 'tw_mse', 'tw_mae', 'tw_huber' for time-weighted training loss)")


def teacher_forcing_loss(predictions, targets, mask=None, loss_type='mse', reduce_batch=True):
    """
    Compute loss for teacher forcing training.
    
    Args:
        predictions: Model output (B, T, C)
        targets: Target sequences (B, T, C)
        mask: Boolean attention mask (B, T)
        loss_type: 'mse', 'mae', or 'huber'
        reduce_batch: If True, return scalar. If False, return per-sample loss (B,)
    """
    loss = get_raw_loss(predictions, targets, loss_type)
    
    # Average over channels -> (B, T)
    loss_per_step = jnp.mean(loss, axis=-1)
    
    if mask is not None:
        loss_per_step = loss_per_step * mask
        
        if reduce_batch:
            valid_tokens = jnp.sum(mask)
            total_loss = jnp.sum(loss_per_step)
            return total_loss / jnp.maximum(valid_tokens, 1.0)
        else:
            # Return per-sample loss -> (B,)
            valid_tokens_per_sample = jnp.sum(mask, axis=-1)
            sample_loss = jnp.sum(loss_per_step, axis=-1)
            return sample_loss / jnp.maximum(valid_tokens_per_sample, 1.0)
    else:
        if reduce_batch:
            return jnp.mean(loss_per_step)
        else:
            return jnp.mean(loss_per_step, axis=-1) # -> (B,)


def split_loss_type(loss_type):
    """
    Split a loss_type into (base_loss_type, is_time_weighted).
    E.g. 'tw_mse' -> ('mse', True), 'mse' -> ('mse', False).
    """
    if loss_type.startswith('tw_'):
        return loss_type[len('tw_'):], True
    return loss_type, False


def time_weighted_loss(predictions, targets, mask=None, loss_type='mse', delta=0.98):
    """
    Time-weighted teacher forcing loss: sum_tau delta^(T - tau) * l(pred_tau, target_tau).

    T is the last valid position of each sample (taken from the mask, so it follows
    the curriculum length). Assumes a prefix mask. The loss is normalised by the sum
    of the weights, so delta=1 recovers teacher_forcing_loss.

    Args:
        predictions: Model output (B, T, C)
        targets: Target sequences (B, T, C)
        mask: Boolean attention mask (B, T)
        loss_type: 'mse', 'mae', or 'huber' (step-wise loss metric)
        delta: Temporal weighting factor in (0, 1]
    """
    loss = get_raw_loss(predictions, targets, loss_type)

    # Average over channels -> (B, T)
    loss_per_step = jnp.mean(loss, axis=-1)
    B, T = loss_per_step.shape

    if mask is None:
        mask = jnp.ones((B, T))
    mask = mask.astype(loss_per_step.dtype)

    # Effective length per sample and 1-based time index
    T_eff = jnp.sum(mask, axis=-1, keepdims=True)                     # (B, 1)
    tau = jnp.arange(1, T + 1, dtype=loss_per_step.dtype)[None, :]    # (1, T)
    weights = jnp.power(delta, T_eff - tau) * mask                    # zero past T_eff

    return jnp.sum(weights * loss_per_step) / jnp.maximum(jnp.sum(weights), 1e-8)


def training_loss(predictions, targets, mask=None, loss_type='mse', delta=0.98):
    """
    Dispatch to teacher_forcing_loss or time_weighted_loss depending on loss_type.

    Args:
        loss_type: 'mse', 'mae', 'huber', or their time-weighted versions
                   'tw_mse', 'tw_mae', 'tw_huber'
        delta: Temporal weighting factor, only used for time-weighted losses
    """
    base_loss_type, is_time_weighted = split_loss_type(loss_type)
    if is_time_weighted:
        return time_weighted_loss(predictions, targets, mask=mask,
                                  loss_type=base_loss_type, delta=delta)
    return teacher_forcing_loss(predictions, targets, mask=mask,
                                loss_type=base_loss_type, reduce_batch=True)


def per_timestep_loss(predictions, targets, mask=None, loss_type='mse'):
    """
    Compute loss per timestep for analysis.
    """
    loss = get_raw_loss(predictions, targets, loss_type)
    
    # Average over channels -> (B, T)
    loss_per_step = jnp.mean(loss, axis=-1)
    
    if mask is not None:
        loss_per_step = loss_per_step * mask
        # Count how many valid sequences exist for each timestep -> (T,)
        valid_b_per_t = jnp.sum(mask, axis=0) 
        sum_loss_per_t = jnp.sum(loss_per_step, axis=0)
        return sum_loss_per_t / jnp.maximum(valid_b_per_t, 1.0)
    else:
        # Average over batch -> (T,)
        return jnp.mean(loss_per_step, axis=0)


def per_channel_loss(predictions, targets, mask=None, loss_type='mse'):
    """
    Compute loss per latent channel for analysis.
    """
    loss = get_raw_loss(predictions, targets, loss_type) # (B, T, C)
    
    if mask is not None:
        # Expand mask from (B, T) to (B, T, 1) for broadcasting
        mask_expanded = jnp.expand_dims(mask, axis=-1)
        loss_masked = loss * mask_expanded
        
        valid_tokens = jnp.sum(mask) # scalar: total valid (B, T) pairs
        sum_loss_per_c = jnp.sum(loss_masked, axis=(0, 1)) # (C,)
        return sum_loss_per_c / jnp.maximum(valid_tokens, 1.0)
    else:
        # Average over batch and time -> (C,)
        return jnp.mean(loss, axis=(0, 1))

