# -- Directories search 
import glob
import os
# -- Numpy
import numpy as np
# -- Torch dataset
import torch
from torch.utils.data import Dataset, DataLoader
# The script contains the definition of the dataset calss and the dataloader.
from typing import Dict, Any
# -- Utils
from src.utils.utils import numpy_collate, load_dataset_metadata
# -- Gin config
import gin

class MinimalChannelFlowDataset(Dataset):
    def __init__(self, data_directory, mode='train', use_float16=False, seed=42):
        """ Minimal Channel dataset. 
        Numerical simulation solver: SOD2D (DNS)
        Data Size (17500,48,24,3) (B,H,W,C).
        Reference case: Numerical investigation on Minimal Flow Unit, domain size follows Jimenez and Moin (1991).
        Args:
            data_directory  (str) : directory containing the piv fields.
            mode (str) : 'train' or 'test'. If 'train', it loads the 17500 samples coming from 4th run and split 10% of it into validation, if 'test' it loads the 17500 samples coming from 5th run.
        """
        self.seed = seed
        self.dtype = np.float16 if use_float16 else np.float32
        
        if mode not in ['train', 'test']:
            raise ValueError('Mode not recognized, choose between train, or test')
        
        # Look for mode-specific metadata
        metadata_pattern = data_directory + f'/{mode}/fields_metadata.json'  # Fixed: single slash
        metadata_file = None
        if os.path.exists(metadata_pattern):
            metadata_file = metadata_pattern
            
        if metadata_file is None:
            raise ValueError(f'No metadata file found in {metadata_pattern}')
            
        self.metadata = load_dataset_metadata(metadata_file)
        
        data_file_list = glob.glob(data_directory + f'/{mode}/*.npy')
        if len(data_file_list) == 0:
            raise ValueError(f'No {mode} file found in the directory {data_directory}/{mode}')
        if len(data_file_list) > 1:
            raise ValueError(f'More than one {mode} file found in the directory {data_directory}/{mode}')
        
        file = data_file_list[0]
        try:
            self._npy_file_data = np.load(file)
        except Exception as e:
            raise IOError(f"Error loading {file}: {e}")
        
    def __len__(self):
        return self.metadata['samples']

    def __getitem__(self, idx):
        grid_data = self._npy_file_data[idx]

        if grid_data.dtype != self.dtype:
            grid_data = grid_data.astype(self.dtype)
 
        return grid_data

def worker_init_fn(worker_id, base_seed=42):
    """Initialize each worker with a unique but reproducible seed."""
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

@gin.configurable
def data_loader(dataset_name, mode, root, batch_size, shuffle, drop_last, workers,
                seed=42, split_seed=42, train_val_split=0.9, use_float16=False):
    """Create dataloaders for training/validation/testing with full reproducibility.

    Args:
        seed (int): Run seed, controls the shuffle order of the training data.
        split_seed (int): Seed for the train/validation split. Keep fixed across runs.
        (other args unchanged)
    """
    print(f"=== Data Loader Configuration ===")
    print(f"  dataset_name: {dataset_name}")
    print(f"  mode: {mode}")
    print(f"  train_val_split: {train_val_split}")
    print(f"  root: {root}")
    print(f"  batch_size: {batch_size}")
    print(f"  shuffle: {shuffle}")
    print(f"  workers: {workers}")
    print(f"  seed: {seed}")
    print(f"  split_seed: {split_seed}")
    print(f"  use_float16: {use_float16}")
    print("=====================================")

    def worker_init_fn_with_seed(worker_id):
        return worker_init_fn(worker_id, base_seed=seed)

    if dataset_name != 'minimal_channel_flow':
        raise ValueError('Dataset name not recognized')

    if mode == 'train':
        data_set = MinimalChannelFlowDataset(data_directory=root, mode='train', use_float16=use_float16)

        # Split: fixed across runs
        split_gen = torch.Generator().manual_seed(split_seed)
        train_size = int(train_val_split * len(data_set))
        train_set, validation_set = torch.utils.data.random_split(
            data_set, [train_size, len(data_set) - train_size], generator=split_gen)

        # Shuffle: follows the run seed
        shuffle_gen = torch.Generator().manual_seed(seed)

        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  drop_last=drop_last,
                                  shuffle=shuffle,
                                  generator=shuffle_gen,
                                  pin_memory=False,
                                  num_workers=workers,
                                  collate_fn=numpy_collate,
                                  persistent_workers=workers > 0,
                                  worker_init_fn=worker_init_fn_with_seed if workers > 0 else None)

        validation_loader = DataLoader(validation_set,
                                       batch_size=batch_size,
                                       drop_last=drop_last,
                                       shuffle=False,
                                       pin_memory=False,
                                       num_workers=workers,
                                       collate_fn=numpy_collate,
                                       persistent_workers=workers > 0,
                                       worker_init_fn=worker_init_fn_with_seed if workers > 0 else None)

        return train_loader, validation_loader

    elif mode == 'test':
        test_set = MinimalChannelFlowDataset(data_directory=root, mode='test', use_float16=use_float16)

        test_loader = DataLoader(test_set,
                                 batch_size=batch_size,
                                 drop_last=drop_last,
                                 shuffle=shuffle,
                                 num_workers=workers,
                                 collate_fn=numpy_collate,
                                 persistent_workers=workers > 0,
                                 worker_init_fn=worker_init_fn_with_seed if workers > 0 else None)
        return test_loader

    else:
        raise ValueError('Loader mode not recognized, choose between train, test')