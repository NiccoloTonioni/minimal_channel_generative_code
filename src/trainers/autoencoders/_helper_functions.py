"""
Schedulers of the loss weights (beta, gamma) used during the autoencoder training.
"""
# This script contains the scheduler that can be used in VAE trainer to control the beta parameter.
import numpy as np

##############################################################################
#                             PARAM LOSS SCHEDULERS                          #
##############################################################################
# Schedulers to control loss parameters during training
# e.g., beta in VAE loss or lambda in VAE-GAN loss

class ParamScheduler:
    """Base class for loss parameters schedulers"""
    def get_loss_param(self, epoch: int) -> float:
        """Return the value of the corresponding loss parameter for the given epoch"""
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}"


class ConstantScheduler(ParamScheduler):
    """Constant scheduler that returns the same value for all epochs"""
    def __init__(self, value=1e-4, freeze_epochs=0):
        self.value = value
        self.freeze_epochs = freeze_epochs
    
    def get_loss_param(self, epoch):
        if epoch < self.freeze_epochs:
            return 0.0
        else:    
            return self.value
    
    def __repr__(self):
        return f"Constant_Scheduler(value={self.value})"

class StepScheduler(ParamScheduler):
    """Step scheduler that changes value at specified epoch milestones"""
    def __init__(self, milestones, values):
        """
        Args:
            milestones: List of epoch numbers where value changes (must be sorted)
            values: List of values to use. Length should be len(milestones) + 1
                   values[0] is used for epochs < milestones[0]
                   values[i] is used for milestones[i-1] <= epoch < milestones[i]
                   values[-1] is used for epochs >= milestones[-1]
        """
        assert len(values) == len(milestones) + 1, "values must have length len(milestones) + 1"
        assert all(milestones[i] < milestones[i+1] for i in range(len(milestones)-1)), "milestones must be sorted"
        self.milestones = milestones
        self.values = values
    
    def get_loss_param(self, epoch):
        for i, milestone in enumerate(self.milestones):
            if epoch < milestone:
                return self.values[i]
        return self.values[-1]
    
    def __repr__(self):
        return f"Step_Scheduler(milestones={self.milestones})"

class LinearScheduler(ParamScheduler):
    """Schedule with optional initial freeze period, then linear growth to max value"""
    def __init__(self, start_value=0.0, end_value=1e-4, warmup=1000, freeze_epochs=0):
        self.start_value = start_value
        self.end_value = end_value
        self.warmup = warmup
        self.freeze_epochs = freeze_epochs
    
    def get_loss_param(self, epoch):
        if epoch < self.freeze_epochs:
            return 0.0
        elif epoch < self.freeze_epochs + self.warmup:
            adjusted_epoch = epoch - self.freeze_epochs
            param = self.start_value + (self.end_value - self.start_value) * adjusted_epoch / self.warmup
            return param
        else:
            return self.end_value
    
    def __repr__(self):
        return f"Linear_Scheduler(freeze={self.freeze_epochs})"


class CyclicLinearScheduler(ParamScheduler):
    """Schedule with optional initial freeze period, then cyclic linear growth to max value"""
    def __init__(self, start_value=0.0, end_value=1e-4, n_epochs=1000, n_cycles=2, ratio=0.5, freeze_epochs=0):
        self.freeze_epochs = freeze_epochs
        
        # Adjust the cycling schedule to account for freeze period
        effective_epochs = n_epochs - freeze_epochs
        schedule = np.ones(n_epochs) * end_value
        
        # Keep frozen epochs at zero
        schedule[:freeze_epochs] = 0.0
        
        # Build cyclic schedule for remaining epochs
        period = effective_epochs / n_cycles
        step = (end_value - start_value) / (period * ratio)
        
        for c in range(n_cycles):
            v, i = start_value, 0
            while v <= end_value and (int(i + c * period + freeze_epochs) < n_epochs):
                schedule[int(i + c * period + freeze_epochs)] = v
                v += step
                i += 1
        
        self.schedule = schedule
    
    def get_loss_param(self, epoch):
        param = self.schedule[epoch]
        return param
    
    def __repr__(self):
        return f"Cyclic_Linear_Scheduler(freeze={self.freeze_epochs})"