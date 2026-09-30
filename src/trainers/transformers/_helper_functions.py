import jax
import jax.numpy as jnp

##############################################################################
#                       CURRICULUM LEARNING SCHEDULERS                       #
##############################################################################

class CurriculumScheduler:
    """Base class for curriculum learning schedulers"""
    
    def get_sequence_length(self, epoch: int) -> int:
        """Return the sequence length for the given epoch"""
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}"


class LinearCurriculum(CurriculumScheduler):
    """Linearly increase sequence length from start to end, then stay constant"""
    
    def __init__(self, min_length: int, max_length: int, warmup_epochs: int):
        """
        Args:
            min_length: Starting sequence length
            max_length: Final sequence length (block_size)
            warmup_epochs: Number of epochs to reach max_length (then stays constant)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.warmup_epochs = warmup_epochs
        
    def get_sequence_length(self, epoch: int) -> int:
        # After warmup, stay at max_length
        if epoch >= self.warmup_epochs:
            return self.max_length
        
        # Linear interpolation during warmup
        progress = epoch / self.warmup_epochs
        length = self.min_length + (self.max_length - self.min_length) * progress
        return int(length)
    
    def __repr__(self):
        return (f"LinearCurriculum(min={self.min_length}, max={self.max_length}, "
                f"warmup={self.warmup_epochs})")


class StepCurriculum(CurriculumScheduler):
    """Increase sequence length in discrete steps"""
    
    def __init__(self, schedule: list):
        """
        Args:
            schedule: List of (epoch, length) tuples
                     Example: [(0, 16), (50, 32), (100, 48), (200, 64)]
        """
        self.schedule = sorted(schedule, key=lambda x: x[0])
        
    def get_sequence_length(self, epoch: int) -> int:
        for epoch_threshold, length in reversed(self.schedule):
            if epoch >= epoch_threshold:
                return length
        return self.schedule[0][1]
    
    def __repr__(self):
        return f"StepCurriculum(schedule={self.schedule})"


class ExponentialCurriculum(CurriculumScheduler):
    """Exponentially increase sequence length, then stay constant"""
    
    def __init__(self, min_length: int, max_length: int, warmup_epochs: int, 
                 growth_rate: float = 2.0):
        """
        Args:
            min_length: Starting sequence length
            max_length: Final sequence length
            warmup_epochs: Number of epochs to reach max_length (then stays constant)
            growth_rate: Exponential growth rate
        """
        self.min_length = min_length
        self.max_length = max_length
        self.warmup_epochs = warmup_epochs
        self.growth_rate = growth_rate
        
    def get_sequence_length(self, epoch: int) -> int:
        # After warmup, stay at max_length
        if epoch >= self.warmup_epochs:
            return self.max_length
        
        # Exponential growth during warmup
        progress = epoch / self.warmup_epochs
        length = self.min_length * (self.growth_rate ** progress)
        length = min(length, self.max_length)
        return int(length)
    
    def __repr__(self):
        return (f"ExponentialCurriculum(min={self.min_length}, max={self.max_length}, "
                f"warmup={self.warmup_epochs}, rate={self.growth_rate})")

##############################################################################
#                    SEQUENCE TRUNCATION UTILITIES                          #
##############################################################################

def truncate_sequences(input_seq: jnp.ndarray, 
                       target_seq: jnp.ndarray, 
                       length: int) -> tuple:
    """
    Truncate sequences to specified length.
    
    Args:
        input_seq: Input sequences (B, T, C) or (B, T, 1, 1, 1, C)
        target_seq: Target sequences (B, T, C) or (B, T, 1, 1, 1, C)
        length: Desired sequence length
        
    Returns:
        Truncated (input_seq, target_seq)
    """
    # Get current sequence length (dimension 1)
    current_length = input_seq.shape[1]
    
    # If already shorter or equal, return as is
    if current_length <= length:
        return input_seq, target_seq
    
    # Truncate to desired length
    if input_seq.ndim == 3:  # (B, T, C)
        return input_seq[:, :length, :], target_seq[:, :length, :]
    elif input_seq.ndim == 6:  # (B, T, 1, 1, 1, C)
        return input_seq[:, :length, :, :, :, :], target_seq[:, :length, :, :, :, :]
    else:
        raise ValueError(f"Unexpected input shape: {input_seq.shape}")


def random_truncate_sequences(input_seq: jnp.ndarray,
                              target_seq: jnp.ndarray,
                              min_length: int,
                              max_length: int,
                              key: jnp.ndarray) -> tuple:
    """
    Randomly truncate sequences to a length between min and max.
    Adds stochasticity to curriculum learning.
    
    Args:
        input_seq: Input sequences
        target_seq: Target sequences
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        key: JAX random key
        
    Returns:
        Truncated (input_seq, target_seq)
    """
    import jax.random as random
    
    # Sample random length
    length = random.randint(key, shape=(), minval=min_length, maxval=max_length + 1)
    
    return truncate_sequences(input_seq, target_seq, int(length))


##############################################################################
#                         TRAIN_EPOCH WITH CURRICULUM                       #
##############################################################################

def train_epoch_with_stochastic_curriculum(self, train_loader, epoch, 
                                           curriculum_scheduler=None,
                                           stochastic_range=4):
    """
    Train for one epoch with stochastic curriculum learning.
    Adds randomness to sequence lengths for better generalization.
    
    Args:
        train_loader: Training data loader
        epoch: Current epoch number
        curriculum_scheduler: CurriculumScheduler instance
        stochastic_range: Range of random variation (±)
    """
    losses = []
    maes = []
    grad_norms = []
    
    # Get base sequence length for this epoch
    if curriculum_scheduler is not None:
        base_length = curriculum_scheduler.get_sequence_length(epoch)
        min_length = max(4, base_length - stochastic_range)
        max_length = base_length + stochastic_range
        
        if jax.process_index() == 0 and epoch % 10 == 0:
            print(f"  Stochastic Curriculum: [{min_length}, {max_length}] (base={base_length})")
    else:
        min_length = None
        max_length = None
    
    for input_seq, target_seq in train_loader:
        # Split key for dropout and random truncation
        self.key, step_key, trunc_key = jax.random.split(self.key, 3)
        
        # Apply stochastic curriculum learning
        if min_length is not None and max_length is not None:
            input_seq, target_seq = random_truncate_sequences(
                input_seq, target_seq, min_length, max_length, trunc_key
            )
        
        # Training step
        self.state, loss, metrics = self.train_step(
            self.state, input_seq, target_seq, step_key
        )
        
        losses.append(metrics['loss'])
        maes.append(metrics['mae'])
        grad_norms.append(metrics['grad_norm'])
    
    # Compute epoch metrics
    losses_jnp = jnp.stack(losses)
    maes_jnp = jnp.stack(maes)
    grad_norms_jnp = jnp.stack(grad_norms)
    
    metrics_on_device = {
        'loss': jnp.mean(losses_jnp),
        'mae': jnp.mean(maes_jnp),
        'grad_norm': jnp.mean(grad_norms_jnp)
    }
    
    metrics_on_host = jax.device_get(metrics_on_device)
    
    return (float(metrics_on_host['loss']), 
            float(metrics_on_host['mae']),
            float(metrics_on_host['grad_norm']))

##############################################################################
#                      GIN-CONFIGURABLE SCHEDULERS                          #
##############################################################################
def get_curriculum_scheduler(
    curriculum_type='linear',  # 'linear', 'step', 'exponential', 'constant'
    min_length=16,
    max_length=64,
    num_epochs=200,
    warmup_epochs=0,
    schedule=None,
    growth_rate=2.0
):
    """
    Factory function for creating curriculum schedulers.
    Can be configured via gin config files.
    
    Args:
        curriculum_type: Type of curriculum scheduler
        min_length: Starting sequence length
        max_length: Final sequence length (block_size)
        num_epochs: Number of epochs to reach max_length
        schedule: For 'step' type: list of (epoch, length) tuples
        growth_rate: For 'exponential' type: growth rate
    """
    if curriculum_type == 'linear':
        return LinearCurriculum(min_length, max_length, warmup_epochs)
    
    elif curriculum_type == 'step':
        if schedule is None:
            # Default schedule
            schedule = [
                (0, min_length),
                (num_epochs // 4, (min_length + max_length) // 2),
                (num_epochs // 2, max_length)
            ]
        return StepCurriculum(schedule)
    
    elif curriculum_type == 'exponential':
        return ExponentialCurriculum(min_length, max_length, warmup_epochs, growth_rate)
    else:
        raise ValueError(f"Unknown curriculum_type: {curriculum_type}")


