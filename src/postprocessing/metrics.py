"""
Metrics reported in the paper.

  energy_coefficient          E_k+  , Eq. (27)   (Fig. 6, Table 4)
  mean_latent_correlation     r_zeta, Eq. (29)   (Fig. 6)
  global_entropy              H     , Eq. (2)    (Table 4, Fig. 14)
  latent_forecast_metrics     NRMSE, NMAE, rho, d_H, Eqs. (31)-(33)   (Table 3)
"""
import numpy as np
from scipy import stats
from scipy.linalg import svd
from scipy.spatial.distance import directed_hausdorff


# -----------------------------------------------------------------------------
# Compression / reconstruction
# -----------------------------------------------------------------------------
def turbulent_kinetic_energy(fields):
    """k+ of Eq. (26): half the trace of the Reynolds stresses, averaged in time and over the plane."""
    return 0.5 * np.mean(np.sum(fields ** 2, axis=-1))


def energy_coefficient(true_fields, model_fields):
    """Reconstructed energy coefficient E_k+ = 1 - |k+ - k~+| / k+ (Eq. 27)."""
    k_true = turbulent_kinetic_energy(true_fields)
    k_model = turbulent_kinetic_energy(model_fields)
    return 1.0 - np.abs(k_true - k_model) / k_true


def mean_latent_correlation(latents):
    """Mean absolute off-diagonal Pearson correlation between latent variables, r_zeta (Eq. 29)."""
    corr = np.corrcoef(latents.reshape(-1, latents.shape[-1]).T)
    off_diagonal = corr[np.triu_indices_from(corr, k=1)]
    return np.mean(np.abs(off_diagonal))


def global_entropy(fields):
    """POD-based global entropy H (Eq. 2) and the singular values of the snapshot matrix.

    The snapshot matrix has shape (Nx*Nz*C, Nt). Note: the relative contributions p_i are
    computed from the singular values sigma_i of the snapshot matrix, which is how the
    values reported in the paper (Eq. 4, Table 4) were obtained.
    """
    snapshots = fields.reshape(fields.shape[0], -1).T
    singular_values = svd(snapshots, full_matrices=False, compute_uv=False)
    p = singular_values / np.sum(singular_values)
    entropy = -np.sum(p * np.log(p + 1e-15)) / np.log(len(singular_values))
    return entropy, singular_values


# -----------------------------------------------------------------------------
# Latent forecasting (Table 3)
# -----------------------------------------------------------------------------
def nrmse(predicted, target):
    """Normalised root-mean-square error (Eq. 31)."""
    return np.sqrt(np.mean((predicted - target) ** 2)) / (np.sqrt(np.mean(target ** 2)) + 1e-8)


def nmae(predicted, target):
    """Normalised mean absolute error in percent (Eq. 31)."""
    return np.mean(np.abs(target - predicted)) / (np.mean(np.abs(target)) + 1e-8) * 100


def pearson(predicted, target):
    """Pearson correlation between predicted and target trajectories, averaged over latents (Eq. 32)."""
    return np.mean([stats.pearsonr(predicted[:, j], target[:, j])[0] for j in range(target.shape[1])])


def hausdorff(predicted, target):
    """Symmetric Hausdorff distance between the two trajectories in latent space (Eq. 33)."""
    return max(directed_hausdorff(predicted, target)[0], directed_hausdorff(target, predicted)[0])


def latent_forecast_metrics(predicted, target):
    return {
        'nrmse': nrmse(predicted, target),
        'nmae_percent': nmae(predicted, target),
        'pearson_r': pearson(predicted, target),
        'hausdorff': hausdorff(predicted, target),
    }
