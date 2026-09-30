# -- Directories search 
import os
# -- Numpy
import numpy as np
# -- Torch Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
# -- Gin
import gin
# -- Utils
from src.utils.utils import numpy_collate, load_dataset_metadata

#############################################################################
# Dataset class 
#############################################################################
class LatentSpaceDataset(Dataset):
    def __init__(self, data_directory, mode='train', seed=42,
                 sequence_length=None, use_variance=False, return_teacher_forcing=True,
                 return_sensor_data=False, sensor_data_directory=None):
        """
        Dataset class for latent space data.
        Loads separate .npy files into memory and returns raw numpy arrays.
        If return_sensor_data is True, it also loads the corresponding sensor data and returns it alongside the latents.
        """
        self.mode = mode
        self.sequence_length = sequence_length
        self.return_teacher_forcing = return_teacher_forcing
        self.return_sensor_data = return_sensor_data
        
        if mode not in ['train', 'validation', 'test']:
            raise ValueError('Mode not recognized, choose between train, val, or test')
        
        # Load Mean Data
        mean_file = os.path.join(data_directory, mode, 'latents_mean.npy')
        try:
            mean_raw = np.load(mean_file)
            self.data = np.squeeze(mean_raw, axis=(1, 2))
        except Exception as e:
            raise IOError(f"Error loading {mean_file}: {e}")
            
        # Load Variance Data (Optional)
        if use_variance:
            var_file = os.path.join(data_directory, mode, 'latents_logvar.npy')
            try:
                var_raw = np.load(var_file)
                variance = np.squeeze(var_raw, axis=(1, 2))
                # Sanity Check
                if len(self.data) != len(variance):
                    raise ValueError(f"Length mismatch: Mean {len(self.data)} vs Variance {len(variance)}")

                self.data = np.concatenate([self.data, variance], axis=-1)
                print(f"Loaded {mode} variance, concatenated data shape: {self.data.shape}")
            except Exception as e:
                raise IOError(f"Error loading {var_file}: {e}")

        # Load Sensor Data (Optional)
        if return_sensor_data:
            if sensor_data_directory is None:
                raise ValueError("sensor_data_directory required")
            
            sensor_files = ['sensor_origin.npy', 'sensor_zero.npy', 'sensor_minimum.npy']
            sensor_paths = [os.path.join(sensor_data_directory, mode, f) for f in sensor_files]
            
            try:
                # Load, concat, and cast once
                loaded_sensors = [np.load(f) for f in sensor_paths]
                self.sensor_data = np.concatenate(loaded_sensors, axis=-1)
            except Exception as e:
                raise IOError(f"Error loading sensors: {e}")

        # Length Calculation
        self._length = len(self.data)
        if sequence_length is not None:
            self._length = self._length - sequence_length
            # Sanity Check
            if self._length <= 0:
                raise ValueError(f"Sequence length + 1 ({sequence_length + 1}) is larger than dataset ({len(self.data)})")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # Return No Sequence
        if self.sequence_length is None:
            latent = self.data[idx]
            if self.return_sensor_data:
                return latent, self.sensor_data[idx]
            return latent
        
        # Return Sequence
        if self.return_teacher_forcing:
            # Grab sequence_length + 1 to allow for shift
            end_idx = idx + self.sequence_length + 1
            sequence = self.data[idx : end_idx]
            
            if self.return_sensor_data:
                sensor_seq = self.sensor_data[idx : end_idx]
                return sequence[:-1], sensor_seq[:-1], sequence[1:]
            else:
                return sequence[:-1], sequence[1:]
        
        else:
            # Grab exactly sequence_length
            end_idx = idx + self.sequence_length
            sequence = self.data[idx : end_idx]
            
            if self.return_sensor_data:
                return sequence, self.sensor_data[idx : end_idx]
            else:
                return sequence

#############################################################################
# Main Data Loader Function
#############################################################################
@gin.configurable
def latent_data_loader(mode, root, batch_size, shuffle, drop_last,
                       seed=42, sequence_length=None, 
                       use_variance=False, 
                       return_teacher_forcing=True, 
                       return_sensor_data=False, sensor_data_directory=None,
                       num_workers=4):
    """
    Create PyTorch dataloaders that return Numpy arrays (via numpy_collate).
    No distributed sampler logic.
    """
    
    # Create Dataset
    dataset = LatentSpaceDataset(
        data_directory=root,
        mode=mode,
        seed=seed,
        sequence_length=sequence_length,
        use_variance=use_variance,
        return_teacher_forcing=return_teacher_forcing,
        return_sensor_data=return_sensor_data,
        sensor_data_directory=sensor_data_directory
    )

    # Worker Init Function
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)

    # 3. Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        pin_memory=False
    )
    
    return data_loader