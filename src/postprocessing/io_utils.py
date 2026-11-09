"""
Data loading and unit conversion for main_postprocessing.py.

All velocity fields produced by the framework (DNS dataset, autoencoder reconstructions,
decoded Transformer rollouts) are stored in the normalised form of Eq. (5) of the paper:
    u'_norm = u' / (3 sigma_u).
The helpers below convert them back to fluctuations in wall units, u'^+ = u' / u_tau.
"""
import os
import glob
import json
import numpy as np

# Context length of the Transformer: predictions start after the first CONTEXT test snapshots
CONTEXT = 128
# Sampling interval between consecutive snapshots (outer units)
DELTA_T = 0.08


# -----------------------------------------------------------------------------
# Physical parameters
# -----------------------------------------------------------------------------
def load_params(params_dir):
    """Scaling parameters, friction quantities and grid of the dataset.

    Returns a dict with:
        utau, retau     : friction velocity and friction Reynolds number
        scale           : 3 sigma_u, factor used to normalise the fluctuations (Eq. 5)
        mean_flow       : (Nx, Nz, 3) mean velocity on the plane (outer units)
        x_plus, z_plus  : streamwise / spanwise cell-centre coordinates in wall units
        dt_plus         : sampling interval in wall units
    """
    scaler = np.load(os.path.join(params_dir, 'scaler_params.npz'))
    with open(os.path.join(params_dir, 'surface_variables.json'), 'r') as f:
        surface = json.load(f)
    grid = np.load(os.path.join(params_dir, 'coordinates.npz'))
    utau, retau = surface['utau'], surface['Retau']
    return {
        'utau': utau,
        'retau': retau,
        'scale': float(np.squeeze(scaler['ref_std']) * scaler['std_factor']),
        'mean_flow': scaler['mean_flow'],
        'x_plus': grid['x'] * retau,
        'z_plus': grid['z'] * retau,
        'dt_plus': DELTA_T * utau * retau,
    }


def to_wall_units(fields_norm, params):
    """Normalised fluctuations (Eq. 5) -> fluctuations in wall units, u'^+."""
    return np.asarray(fields_norm, dtype=np.float64) * params['scale'] / params['utau']


# -----------------------------------------------------------------------------
# Locating outputs of the main scripts
# -----------------------------------------------------------------------------
class MissingInputError(FileNotFoundError):
    """An input of a figure/table is missing or incomplete (e.g. only the sample DNS data is available)."""


def find_unique(pattern):
    """Return the single path matching a glob pattern (raises MissingInputError if none or several)."""
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise MissingInputError(f'Expected exactly one match for {pattern}, found {len(matches)}: {matches}')
    return matches[0]


def check_same_length(name_a, a, name_b, b):
    """Raise MissingInputError if two time series do not have the same number of snapshots."""
    if len(a) != len(b):
        raise MissingInputError(f'{name_a} has {len(a)} snapshots but {name_b} has {len(b)} '
                                f'(the full DNS test set is needed, see README).')


def reconstruction_dir(run_dir):
    """Output folder written by main_autoencoder.py (save_reconstructions=True) in a run directory."""
    return find_unique(os.path.join(run_dir, 'reconstructions_epoch_*'))


def rollout_dir(run_dir):
    """Output folder written by main_transformer.py (save_rollout=True) in a run directory."""
    return find_unique(os.path.join(run_dir, 'full_rollout_epoch_*'))


# -----------------------------------------------------------------------------
# Loaders (all fields returned in wall units, shape (Nt, Nx, Nz, 3))
# -----------------------------------------------------------------------------
def load_dns_test(flow_dir, params):
    """DNS test fields (velocity fluctuations in wall units)."""
    return to_wall_units(np.load(os.path.join(flow_dir, 'test', 'fields_2d.npy')), params)


def load_ae_reconstructions(run_dir, params):
    """Reconstructed test fields (wall units) and encoded latents (Nt, d_zeta) of a VAE / VAE-GAN run."""
    out_dir = reconstruction_dir(run_dir)
    fields = to_wall_units(np.load(os.path.join(out_dir, 'all_reconstructions.npy')), params)
    latents = np.load(os.path.join(out_dir, 'all_encoded_latents.npy'))
    return fields, latents.reshape(latents.shape[0], -1)


def load_split_latents(latent_dir, split):
    """Encoded latents (VAE-GAN posterior mean) of one split, shape (N, d_zeta)."""
    latents = np.load(os.path.join(latent_dir, split, 'latents_mean.npy'), mmap_mode='r')
    return latents.reshape(latents.shape[0], -1)


def load_test_latents(latent_dir):
    """Encoded test latents (VAE-GAN posterior mean), shape (Nt, d_zeta)."""
    return np.asarray(load_split_latents(latent_dir, 'test'))


def load_rollout(run_dir, latent_dir, context=CONTEXT):
    """Predicted latents of a Transformer rollout and the matching encoded (target) latents.

    `run_dir` is a Transformer run directory (containing full_rollout_epoch_*/) or directly a folder
    containing predicted_latents.npy. The rollout is initialised with the first `context` encoded test
    latents; the target is the encoding of the DNS test fields over the predicted horizon.
    """
    folder = run_dir if os.path.isfile(os.path.join(run_dir, 'predicted_latents.npy')) else rollout_dir(run_dir)
    predicted = np.load(os.path.join(folder, 'predicted_latents.npy'))
    predicted = predicted.reshape(predicted.shape[0], -1)
    target = load_test_latents(latent_dir)[context:context + len(predicted)]
    if len(target) != len(predicted):
        raise MissingInputError(f'Rollout length {len(predicted)} exceeds the available test latents.')
    return predicted, target


def sort_by_activity(target, *others):
    """Order latent variables by decreasing latent activity (variance, Eq. 34) of the target."""
    order = np.argsort(np.var(target, axis=0))[::-1]
    return (target[:, order],) + tuple(o[:, order] for o in others)


def load_end_to_end(flow_dir, baseline_run_dir, predicted_fields, params, context=CONTEXT):
    """DNS, beta-VAE-GAN baseline and end-to-end prediction over the forecast horizon (Section 4.3).

    The prediction starts after the `context` snapshots used to initialise the Transformer;
    DNS and baseline are restricted to the same time window. All fields in wall units.
    """
    dns = load_dns_test(flow_dir, params)
    baseline = to_wall_units(np.load(os.path.join(reconstruction_dir(baseline_run_dir), 'all_reconstructions.npy')),
                             params)
    prediction = to_wall_units(np.load(predicted_fields), params)
    n = len(prediction)
    if len(dns) < context + n:
        raise MissingInputError(f'The DNS test set ({len(dns)} snapshots) is shorter than context + prediction '
                                f'({context + n}): the full DNS test set is needed, see README.')
    return {'dns_full': dns,
            'dns': dns[context:context + n],
            'baseline': baseline[context:context + n],
            'prediction': prediction}


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------
def save_json(obj, path):
    """Save a (possibly nested) dict containing numpy types to JSON."""
    def convert(o):
        if isinstance(o, dict):
            return {str(k): convert(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [convert(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return o
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(convert(obj), f, indent=4)
    print(f'Saved {path}')
