"""
Flow statistics: two-point correlations, quadrant analysis and scale decomposition.
"""
import numpy as np
from scipy import signal


# -----------------------------------------------------------------------------
# Two-point correlation (Figs. 7 and 13)
# -----------------------------------------------------------------------------
def two_point_correlation(u, axis=2, max_lag=None):
    """Two-point autocorrelation of a periodic fluctuation field along one axis.

    Args:
        u: fluctuation field, e.g. shape (Nt, Nx, Nz)
        axis: spatial axis along which the separation is taken (2 = spanwise)
        max_lag: largest separation in grid points (default: half the domain)
    Returns:
        lags (grid points), R (normalised to 1 at zero separation)
    """
    if max_lag is None:
        max_lag = u.shape[axis] // 2
    lags = np.arange(0, max_lag + 1)
    variance = np.mean(u ** 2)
    R = np.array([np.mean(u * np.roll(u, shift=s, axis=axis)) / variance for s in lags])
    return lags, R


# -----------------------------------------------------------------------------
# Quadrant analysis of the Reynolds shear stress (Fig. 15)
# -----------------------------------------------------------------------------
def joint_pdf(u, v, bins, lim_u, lim_v):
    """Histogram estimate of P(u', v') and of the covariance integrand u'v' P(u', v')."""
    pdf, u_edges, v_edges = np.histogram2d(u.ravel(), v.ravel(), bins=bins,
                                           range=[lim_u, lim_v], density=True)
    u_c = 0.5 * (u_edges[:-1] + u_edges[1:])
    v_c = 0.5 * (v_edges[:-1] + v_edges[1:])
    U, V = np.meshgrid(u_c, v_c, indexing='ij')
    cell_area = (u_edges[1] - u_edges[0]) * (v_edges[1] - v_edges[0])
    return U, V, pdf, U * V * pdf, cell_area


QUADRANTS = {
    'q1': lambda U, V: (U > 0) & (V > 0),   # outward interactions
    'q2': lambda U, V: (U < 0) & (V > 0),   # ejections
    'q3': lambda U, V: (U < 0) & (V < 0),   # inward interactions
    'q4': lambda U, V: (U > 0) & (V < 0),   # sweeps
}


def quadrant_contributions_outside_hole(u, v, F, uv_ref, bins, lim_u, lim_v):
    """Contribution of each quadrant to <u'v'> from events outside the hyperbolic hole |u'v'| >= F |uv_ref|."""
    U, V, _, weighted, cell_area = joint_pdf(u, v, bins, lim_u, lim_v)
    outside = np.abs(U * V) >= F * np.abs(uv_ref)
    return {q: np.sum(weighted[mask(U, V) & outside]) * cell_area for q, mask in QUADRANTS.items()}


def quadrant_peaks(U, V, Z):
    """(u, v) location of the extremum of Z in each quadrant."""
    peaks = []
    for mask_fn in QUADRANTS.values():
        mask = mask_fn(U, V)
        idx = np.argmax(np.abs(Z[mask]))
        peaks.append((U[mask][idx], V[mask][idx]))
    return peaks


# -----------------------------------------------------------------------------
# Scale decomposition (Figs. 11 and 12)
# -----------------------------------------------------------------------------
def low_pass_filter(data, cutoff, fs, order=4, axis=0):
    """Zero-phase (forward-backward) Butterworth low-pass filter along `axis`."""
    sos = signal.butter(N=order, Wn=cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data, axis=axis)


def hdr_levels(Z, cell_area=None, masses=(0.25, 0.5, 0.75, 0.9)):
    """Density levels enclosing the given fractions of the probability mass (highest-density regions)."""
    z = np.sort(Z.ravel())[::-1]
    cdf = np.cumsum(z)
    if cell_area is None:
        cdf = cdf / cdf[-1]
    else:
        cdf = cdf * cell_area
    return np.sort([z[np.searchsorted(cdf, m)] for m in masses])
