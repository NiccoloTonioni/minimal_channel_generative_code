"""
Proper orthogonal decomposition (POD) baseline of the compression study (Figs. 6, 7).

The POD basis is computed on the snapshots of the training run (no mean subtraction, as the fields are
already velocity fluctuations) and the test snapshots are reconstructed by projection onto the first
d_zeta modes.
"""
import numpy as np


def pod_basis(train_fields, n_modes):
    """First `n_modes` spatial POD modes of the training snapshots, shape (n_modes, Nx*Nz*3)."""
    snapshots = np.asarray(train_fields, dtype=np.float64).reshape(len(train_fields), -1)
    _, _, modes = np.linalg.svd(snapshots, full_matrices=False)
    return modes[:n_modes]


def pod_reconstructions(train_fields, test_fields, latent_dims):
    """POD reconstructions of the test snapshots for each latent dimension: {d_zeta: (Nt, Nx, Nz, 3)}."""
    modes = pod_basis(train_fields, max(latent_dims))
    test = np.asarray(test_fields, dtype=np.float64).reshape(len(test_fields), -1)
    coefficients = test @ modes.T                                   # projection on all retained modes
    return {d: (coefficients[:, :d] @ modes[:d]).reshape(test_fields.shape) for d in latent_dims}
