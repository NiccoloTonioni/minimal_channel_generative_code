"""
Utilities: batch collation, dataset metadata, device placement of JAX batches and tensor scaling.
"""
# This script contains all the utility functions used in the project.
# The functions are divided into the following categories:
# 1. Functions to convert from numpy/jax to torch and back
# 2. Functions to scale tensors
# -- Numpy
import numpy as np
import json
# -- Jax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
# -- Torch
import torch
# -- Typing
from typing import Iterator, Any

################################################################################
#                                                                              #
#                                  Dataloader                                  #
#                                                                              #
################################################################################
def numpy_collate(batch):
    """Numpy version of default_collate for PyTorch"""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        return tuple(numpy_collate(samples) for samples in zip(*batch))
    else:
        return np.array(batch)
    
def load_dataset_metadata(metadata_path):
    """
    Utility function to load metadata for PyTorch DataLoader
    
    Args:
        metadata_path (str): Path to metadata.json file
        
    Returns:
        dict: Metadata dictionary
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

################################################################################
#                                                                              #
#                                  Jax2Torch                                   #
#                                                                              #
################################################################################
def jax_to_torch(imgs):
    """Map JAX or numpy tensors back to PyTorch"""
    imgs = jax.device_get(imgs)
    imgs = torch.from_numpy(imgs.astype(np.float32))
    imgs = imgs.permute(0, 3, 1, 2)
    return imgs



def jax_device_put_iterator(dataloader_iterator: Iterator[Any], mesh: Mesh = None, p_spec: P = None) -> Iterator[Any]:
    """
    Wraps a PyTorch DataLoader iterator to asynchronously put data onto JAX devices.
    """
    for batch_cpu in dataloader_iterator:
        if mesh and p_spec:
            # For sharding, ensure batch_cpu is already a NumPy array or compatible structure.
            # jax.device_put can handle NumPy arrays directly.
            # This operation schedules the transfer to the device.
            yield jax.tree_util.tree_map(lambda x: jax.device_put(x, NamedSharding(mesh, p_spec)), batch_cpu)
        else:
            # Default to replicating data if no mesh/sharding is specified
            yield jax.tree_util.tree_map(jax.device_put, batch_cpu)

################################################################################
#                                                                              #
#                               Paralle Training                               #
#                                                                              #
################################################################################
def print_jax_spmd_configuration(process_count:int,
                                  process_index:int,
                                  num_devices:int,
                                  local_device_count:int,
                                  mode:str,
                                  global_batch_size:int,
                                  model_name:str="Model"):
    """ Utility function to print JAX SPMD configuration. """
    print("=" * 50)
    print("JAX SPMD Configuration")
    print("=" * 50)
    print(f"Number of processes: {process_count}")
    print(f"Process index: {process_index}")
    print(f"Total devices: {num_devices}")
    print(f"Local devices (on this node): {local_device_count}")
    print(f"Device list (local): {jax.local_devices()}")
    if process_count == 1 and num_devices > 1:
        print(f"\n✓ Single-node multi-GPU processing")
        print(f"  - Data will be sharded across {num_devices} GPUs")
        print(f"  - Model parameters will be replicated on each GPU")
        print(f"  - Global batch size ({mode}): {global_batch_size}")
        print(f"  - Per-device batch size: {global_batch_size // num_devices}")
    elif process_count > 1:
        print(f"\n✓ Multi-node processing")
        print(f"  - {process_count} nodes with {local_device_count} GPUs each")
        print(f"  - Data will be sharded across {num_devices} total GPUs")
        print(f"  - Global batch size ({mode}): {global_batch_size}")
        print(f"  - Per-device batch size: {global_batch_size // num_devices}")
    else:
        print(f"\n✓ Single-GPU processing")
        print(f"  - Batch size: {global_batch_size}")
    print("=" * 50)

    if mode == 'train':
        print("\n" + "#" * 50)
        print(f"Starting {model_name} training...")
        print("#" * 50)
    elif mode == 'inference':
        print("\n" + "#" * 50)
        print(f"Starting {model_name} inference...")
        print("#" * 50)

################################################################################
#                                                                              #
#                              Scaling Functions                               #
#                                                                              #
################################################################################

def scale_tensor_jax(array:np.array, range_scaling:list=[-1,1]) -> np.array:
    """Vector scaling centered to the given target_range [x_min,x_max]"""
    if len(range_scaling) != 2:
        raise ValueError('target_range must be of length 2: [x_min,x_max]')
    minimum = jnp.min(array)
    maximum = jnp.max(array)
    scaling_factor = (range_scaling[1]-range_scaling[0])/(maximum-minimum)
    minimum_vector = jnp.ones_like(array)*minimum
    minumum_range = jnp.ones_like(array)*range_scaling[0]
    scaled = (array-minimum_vector)*scaling_factor + minumum_range
    return scaled

def scale_tensor(tensor, min_val, max_val, range_scaling=[0, 1]):
    tensor -= min_val
    tensor /= (max_val - min_val)
    tensor *= (range_scaling[1] - range_scaling[0])
    tensor += range_scaling[0]
    return tensor

def inverse_scale_tensor(scaled_tensor, min_val, max_val, range_scaling=[0, 1]):
    scaled_tensor -= range_scaling[0]
    scaled_tensor /= (range_scaling[1] - range_scaling[0])
    scaled_tensor *= (max_val - min_val)
    scaled_tensor += min_val
    return scaled_tensor
