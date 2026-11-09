"""
Reproduction of the figures and tables of the main text of the paper.

  Result         Content                                                     Inputs
  -------------  ----------------------------------------------------------  ----------------------------------------
  fig06_fig07    compression: E_k+, r_zeta, two-point correlation of u'      DNS train/test sets (POD), VAE / VAE-GAN
                                                                             reconstructions for d_zeta = 2..12
  table3         latent forecasting metrics of the Transformer variants      rollouts of SA, EA, EA*, SA+EA, SA+EA*
  fig08_fig09    latent time series, joint PDFs, PSDs and PDFs (EA*)         rollout of the selected model
  fig10_fig16    (-u'v')+_max, u' at the centre and their spectra            DNS test set, predicted fields
  fig11_fig12    low/high-frequency decomposition of zeta_3 and u'           encoded test latents, DNS test set
  fig13          two-point correlation of u', v', w'                         DNS test set, baseline, predicted fields
  table4_fig14   E_k+, global entropy and singular values                    DNS test set, baseline, predicted fields
  fig15          quadrant analysis of the Reynolds shear stress              DNS test set, predicted fields

The inputs are produced by main_autoencoder.py, main_transformer.py and main_predictive_framework.py;
their locations are set in configs/base/postprocessing/paper_results.gin. A result whose inputs are
missing is skipped. All results except Table 3 and Figs. 8-9 need the full DNS dataset.

Output layout:
  <output_dir>/figures/<result>/{png,eps}/
  <output_dir>/tables/

Usage (from the repository root):
  python main_postprocessing.py --gin_config configs/base/postprocessing/paper_results.gin
  python main_postprocessing.py --gin_config configs/base/postprocessing/paper_results.gin \
      --gin_params "main_postprocessing.results=['table3', 'fig08_fig09']"
"""
import os
import types
import gin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator
from scipy import signal, stats
from scipy.ndimage import gaussian_filter

from src.utils.gin_utils import parse_config_file
from src.postprocessing import io_utils as io
from src.postprocessing import metrics
from src.postprocessing.pod import pod_reconstructions
from src.postprocessing.statistics import (low_pass_filter, hdr_levels, joint_pdf, quadrant_peaks,
                                           quadrant_contributions_outside_hole)
from src.postprocessing.plotting import (COLORS, set_style, save_figure, add_inset, joint_pdf_with_marginals,
                                         colormap, plot_two_point_correlation)

CENTRE = (24, 12)          # grid indices of the domain centre (x+ = 188, z+ = 56)
PSD_SEGMENTS = 6           # Welch segment length = N / 6 (T+_seg ~ 2,900), 50% overlap


# =============================================================================
# Figures 6 and 7: compression performance of POD, beta-VAE and beta-VAE-GAN
# =============================================================================
#   Fig. 6 left : reconstructed energy coefficient E_k+ (Eq. 27) vs latent dimension d_zeta
#          right: mean absolute correlation between latent variables r_zeta (Eq. 29) vs d_zeta
#   Fig. 7      : spanwise two-point autocorrelation of u' for d_zeta = 4 and d_zeta = 12
LATENT_DIMS = [2, 4, 6, 8, 10, 12]
FIG7_LATENT_DIMS = [4, 12]
FIG7_ZOOM = ((25, 35), (-0.2, 0.2))
COMPRESSION_STYLES = {
    'pod': {'color': COLORS['pod'], 'marker': '^', 'label': 'pod'},
    'vae': {'color': COLORS['vae'], 'marker': 'o', 'label': 'vae'},
    'vae_gan': {'color': COLORS['vae_gan'], 'marker': 's', 'label': 'vae-gan'},
}


def _load_compression_model(cfg, model, ld, pod=None):
    """Reconstructed test fields (wall units) and latents (None for POD) of one model."""
    if model == 'pod':
        return pod[ld], None
    return io.load_ae_reconstructions(cfg.autoencoder_runs[model].format(ld=ld), cfg.params)


def _plot_compression_metric(results, key, ylabel, legend_loc, skip=()):
    fig, ax = plt.subplots(figsize=(12, 9))
    for model, by_ld in results.items():
        if model in skip:
            continue
        lds = sorted(by_ld)
        st = COMPRESSION_STYLES[model]
        ax.plot(lds, [by_ld[ld][key] for ld in lds], color=st['color'], marker=st['marker'],
                markerfacecolor='white', markeredgecolor=st['color'], markeredgewidth=2,
                label=st['label'], linewidth=3, markersize=12, alpha=0.9)
    ax.set_xlabel(r'$d_{\zeta}$', labelpad=10, fontsize=50)
    ax.set_ylabel(ylabel, labelpad=10, fontsize=50)
    ax.set_xticks(LATENT_DIMS)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3, steps=[1, 2, 5, 10]))
    ax.legend(frameon=False, loc=legend_loc, fontsize=36)
    fig.tight_layout()
    return fig


def fig06_fig07(cfg):
    dns = io.load_dns_test(cfg.flow_dir, cfg.params)
    models = ['vae', 'vae_gan']
    pod = None
    if cfg.include_pod:
        # POD basis from the training run, test snapshots projected on the first d_zeta modes
        train = io.to_wall_units(np.load(os.path.join(cfg.flow_dir, 'train', 'fields_2d.npy')), cfg.params)
        io.check_same_length('DNS training set', train, 'encoded training + validation latents',
                             range(cfg.n_train_snapshots))
        io.check_same_length('DNS test set', dns, 'encoded test latents', io.load_test_latents(cfg.latent_dir))
        print(f'  Computing POD on the training run ({len(train)} snapshots) ...')
        pod = pod_reconstructions(train, dns, LATENT_DIMS)
        del train
        models.insert(0, 'pod')

    # Figure 6: metrics vs latent dimension
    results = {}
    for model in models:
        results[model] = {}
        for ld in LATENT_DIMS:
            fields, latents = _load_compression_model(cfg, model, ld, pod)
            io.check_same_length(f'{model} (d_zeta={ld}) reconstruction', fields, 'DNS test set', dns)
            entry = {'E_k+': 100 * metrics.energy_coefficient(dns, fields)}
            if latents is not None:
                entry['r_zeta'] = metrics.mean_latent_correlation(latents)
            results[model][ld] = entry
            print(f'  {model:>8s}  d_zeta={ld:<3d} ' + '  '.join(f'{k}={v:.4f}' for k, v in entry.items()))
    io.save_json(results, os.path.join(cfg.tables_dir, 'fig06_compression_metrics.json'))

    out = os.path.join(cfg.figures_dir, 'fig06_fig07')
    ylabel = r'$E_{k^+}$ (\%)' if plt.rcParams['text.usetex'] else r'$E_{k^+}$ (%)'
    save_figure(_plot_compression_metric(results, 'E_k+', ylabel, 'lower right'), out, 'fig06_energy_coefficient')
    save_figure(_plot_compression_metric(results, 'r_zeta', r'$\overline{r}_{\zeta}$', 'upper right', skip=('pod',)),
                out, 'fig06_mean_latent_correlation')

    # Figure 7: two-point correlation of u'
    dz_plus = cfg.params['z_plus'][1] - cfg.params['z_plus'][0]
    for ld in FIG7_LATENT_DIMS:
        curves = [(dns, COLORS['dns'], '--')]
        curves += [(_load_compression_model(cfg, m, ld, pod)[0], COMPRESSION_STYLES[m]['color'], '-') for m in models]
        save_figure(plot_two_point_correlation(curves, dz_plus, 0, *FIG7_ZOOM), out,
                    f'fig07_two_point_correlation_ld_{ld}')


# =============================================================================
# Table 3: latent forecasting metrics (NRMSE, NMAE, rho, d_H) of the Transformer variants
# =============================================================================
# Each model is evaluated on its full autoregressive rollout over the test set, initialised with the
# first 128 encoded test latents and compared with the encoded DNS test latents.
def table3(cfg):
    results = {}
    for label, run_dir in cfg.paper_models.items():
        predicted, target = io.load_rollout(run_dir, cfg.latent_dir)
        results[label] = metrics.latent_forecast_metrics(predicted, target)

    table = pd.DataFrame(results).T[['nrmse', 'nmae_percent', 'pearson_r', 'hausdorff']]
    table.columns = ['NRMSE', 'NMAE (%)', 'rho', 'd_H']
    print('\n  Table 3 - latent forecasting metrics')
    print(table.to_string(float_format=lambda x: f'{x:.4f}'))
    os.makedirs(cfg.tables_dir, exist_ok=True)
    table.to_csv(os.path.join(cfg.tables_dir, 'table3_latent_forecast_metrics.csv'))
    io.save_json(results, os.path.join(cfg.tables_dir, 'table3_latent_forecast_metrics.json'))


# =============================================================================
# Figures 8 and 9: latent dynamics of the selected Transformer (EA*) over the full test rollout
# =============================================================================
#   Fig. 8 left : time series of the predicted vs reference latent variables (with zoom)
#          right: joint PDF of reference vs predicted latent variable
#   Fig. 9 left : premultiplied PSD f+ Phi_{zeta_j zeta_j} (Welch)
#          right: marginal PDF of each latent variable
# Latent variables are ordered by decreasing latent activity (Eq. 34) of the reference.
LATENT_ZOOM_T_OUTER = 75
LATENT_ZOOM_Y = [(-2, 2.5), (-2, 2), (-2, 2), (-3, 3)]
LATENT_JPDF_LIM = (-3, 3)
LATENT_PSD_XLIM = (0.001, 0.4)


def _plot_latent_time_series(time, target, predicted, j, zoom_x, zoom_y):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(time, target[:, j], color=COLORS['target'], linewidth=3)
    ax.plot(time, predicted[:, j], color=COLORS['prediction'], alpha=0.8, linewidth=3)
    axins = add_inset(ax, zoom_x, zoom_y, position='upper left')
    axins.plot(time, target[:, j], color=COLORS['target'], linewidth=3)
    axins.plot(time, predicted[:, j], color=COLORS['prediction'], alpha=0.8, linewidth=3)
    # Make room above the signal for the inset
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.5 * (zoom_y[1] - zoom_y[0]))
    ax.set_xlim(0, time[-1])
    ax.set_xlabel(r'$t^+$', fontsize=52, labelpad=20)
    ax.set_ylabel(rf'$\zeta_{j + 1}$', fontsize=52, labelpad=20)
    fig.tight_layout()
    return fig


def _plot_latent_joint_pdf(target, predicted, j, grid_resolution=100):
    x, y = target[:, j], predicted[:, j]
    kde = stats.gaussian_kde(np.vstack([x, y]))
    xp, yp = 0.05 * (x.max() - x.min()), 0.05 * (y.max() - y.min())
    X, Y = np.meshgrid(np.linspace(x.min() - xp, x.max() + xp, grid_resolution),
                       np.linspace(y.min() - yp, y.max() + yp, grid_resolution))
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.contour(X, Y, Z, levels=np.linspace(0, Z.max(), 20), colors='k', linewidths=1)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.plot(LATENT_JPDF_LIM, LATENT_JPDF_LIM, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(rf'$\zeta_{{{j + 1}}}^\mathrm{{vae-gan}}$', fontsize=52, labelpad=20)
    ax.set_ylabel(rf'$\zeta_{{{j + 1}}}^\mathrm{{pred}}$', fontsize=52, labelpad=20)
    ax.set_xlim(LATENT_JPDF_LIM)
    ax.set_ylim(LATENT_JPDF_LIM)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    return fig


def _plot_latent_psd(target, predicted, j, fs, nperseg):
    fig, ax = plt.subplots(figsize=(12, 9))
    for data, color in ((target, COLORS['target']), (predicted, COLORS['prediction_alt'])):
        f, psd = signal.welch(data[:, j], fs=fs, nperseg=nperseg)
        ax.plot(f, f * psd, color=color, linewidth=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(LATENT_PSD_XLIM)
    ax.set_xlabel(r'$f^+$', fontsize=52, labelpad=20)
    ax.set_ylabel(rf'$f^+\,\Phi_{{\zeta_{j + 1} \zeta_{j + 1}}}$', fontsize=52, labelpad=20)
    fig.tight_layout()
    return fig


def _plot_latent_pdf(target, predicted, j):
    lo = min(target[:, j].min(), predicted[:, j].min())
    hi = max(target[:, j].max(), predicted[:, j].max())
    margin = 0.1 * (hi - lo)
    x = np.linspace(lo - margin, hi + margin, 500)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(x, stats.gaussian_kde(target[:, j])(x), linewidth=2, color=COLORS['target'])
    ax.plot(x, stats.gaussian_kde(predicted[:, j])(x), linewidth=2, color=COLORS['prediction'], ls='--')
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(rf'$\zeta_{j + 1}$', fontsize=52, labelpad=20)
    ax.set_ylabel('PDF', fontsize=52, labelpad=20)
    fig.tight_layout()
    return fig


def fig08_fig09(cfg):
    predicted, target = io.load_rollout(cfg.paper_models[cfg.selected_model], cfg.latent_dir)
    target, predicted = io.sort_by_activity(target, predicted)

    dt_plus = cfg.params['dt_plus']
    time = np.arange(len(target)) * dt_plus
    zoom_x = (0, LATENT_ZOOM_T_OUTER * cfg.params['utau'] * cfg.params['retau'])
    nperseg = len(target) // PSD_SEGMENTS
    out = os.path.join(cfg.figures_dir, 'fig08_fig09')

    for j in range(target.shape[1]):
        tag = f'latent_{j + 1}'
        save_figure(_plot_latent_time_series(time, target, predicted, j, zoom_x, LATENT_ZOOM_Y[j % len(LATENT_ZOOM_Y)]),
                    out, f'fig08_time_series_{tag}')
        save_figure(_plot_latent_joint_pdf(target, predicted, j), out, f'fig08_joint_pdf_{tag}')
        save_figure(_plot_latent_psd(target, predicted, j, 1.0 / dt_plus, nperseg), out, f'fig09_psd_{tag}')
        save_figure(_plot_latent_pdf(target, predicted, j), out, f'fig09_pdf_{tag}')


# =============================================================================
# Figures 10 and 16: temporal signals of the flow over the forecast horizon
# =============================================================================
#   Fig. 10 top left    : spatial maximum of the instantaneous Reynolds shear stress, (-u'v')+_max(t), DNS
#           bottom left : premultiplied PSD f+ Phi of (-u'v')+_max (Welch)
#           top right   : u'+ at the domain centre (x+ = 188, z+ = 56), DNS
#           bottom right: premultiplied plane-averaged PSD f+ Phi_(u'u')+ (average of the local spectra)
#   Fig. 16             : (-u'v')+_max(t), DNS vs end-to-end prediction, full signal and three detail windows
SIGNAL_XLIM_UV_PSD = (3e-4, 0.5)
SIGNAL_XLIM_UU_PSD = (1e-3, 0.4)
# Detail windows of Fig. 16 in outer time units (converted to t+), and their y-limits
SIGNAL_ZOOMS_T_OUTER = [(0, 200), (700, 900)]
SIGNAL_ZOOM_LAST_T_PLUS = (14850, 15850)
SIGNAL_ZOOMS_Y = [(-0.1, 18), (-0.1, 15), (-0.5, 15)]
UV_MAX_LABEL = r"$(-u^\prime v^\prime)^+_{\max}$"


def _plot_signal(time, values, ylabel):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(time, values, color=COLORS['dns'], linewidth=3)
    ax.set_xlabel(r'$t^+$', fontsize=46, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=46, labelpad=20)
    ax.set_xlim(time.min(), time.max())
    fig.tight_layout()
    return fig


def _plot_premultiplied_psd(freq, psd, xlim, ylabel):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(freq, freq * psd, color=COLORS['dns'], linewidth=3, alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_xlabel(r'$f^+$', fontsize=50, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=50, labelpad=20)
    fig.tight_layout()
    return fig


def _plot_signal_comparison(time, ref, pred, xlim, ylim=None, legend=False, lw=2.0):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(time, ref, color=COLORS['dns'], linewidth=lw, label='DNS')
    ax.plot(time, pred, color=COLORS['prediction_alt'], linewidth=lw, alpha=0.8, label='Predicted')
    if legend:
        ax.legend(fontsize=30, frameon=False, loc='upper left', handlelength=1.5, borderaxespad=0.5)
    ax.set_xlabel(r'$t^+$', fontsize=46, labelpad=15)
    ax.set_ylabel(UV_MAX_LABEL, fontsize=46, labelpad=20)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    return fig


def fig10_fig16(cfg):
    data = io.load_end_to_end(cfg.flow_dir, cfg.baseline_run, cfg.predicted_fields, cfg.params)
    dns, pred = data['dns'], data['prediction']
    dt_plus = cfg.params['dt_plus']
    nt = dns.shape[0]
    fs = 1.0 / dt_plus
    nperseg = nt // PSD_SEGMENTS
    time = np.arange(nt) * dt_plus
    print(f'  Welch segment: {nperseg} samples, T+_seg = {nperseg * dt_plus:.0f}')

    uv_max = {name: np.max((-f[..., 0] * f[..., 1]).reshape(len(f), -1), axis=1)
              for name, f in (('dns', dns), ('prediction', pred))}
    u_dns = dns[..., 0]
    out = os.path.join(cfg.figures_dir, 'fig10_fig16')

    # Figure 10
    save_figure(_plot_signal(time, uv_max['dns'], UV_MAX_LABEL), out, 'fig10_uv_max_time_series')
    save_figure(_plot_signal(time, u_dns[:, CENTRE[0], CENTRE[1]], r"$u^{\prime +}$"), out, 'fig10_u_centre_time_series')
    f, psd = signal.welch(uv_max['dns'], fs=fs, nperseg=nperseg)
    save_figure(_plot_premultiplied_psd(f, psd, SIGNAL_XLIM_UV_PSD, r"$f^+ \Phi_{(-u^\prime v^\prime)^+_{\max}}$"),
                out, 'fig10_uv_max_psd')
    f, psd_local = signal.welch(u_dns, fs=fs, nperseg=nperseg, axis=0)          # local spectra (Nf, Nx, Nz)
    save_figure(_plot_premultiplied_psd(f, psd_local.mean(axis=(1, 2)), SIGNAL_XLIM_UU_PSD,
                                        r"$f^+ \Phi_{(u^\prime u^\prime)^+}$"),
                out, 'fig10_uu_plane_averaged_psd')

    # Figure 16
    scale = cfg.params['utau'] * cfg.params['retau']                   # outer -> wall time units
    zooms_x = [(a * scale, b * scale) for a, b in SIGNAL_ZOOMS_T_OUTER] + [SIGNAL_ZOOM_LAST_T_PLUS]
    save_figure(_plot_signal_comparison(time, uv_max['dns'], uv_max['prediction'], (time.min(), time.max()),
                                        legend=True), out, 'fig16_uv_max_comparison')
    for i, (zx, zy) in enumerate(zip(zooms_x, SIGNAL_ZOOMS_Y), start=1):
        save_figure(_plot_signal_comparison(time, uv_max['dns'], uv_max['prediction'], zx, zy, lw=2.5),
                    out, f'fig16_uv_max_comparison_zoom{i}')


# =============================================================================
# Figures 11 and 12: scale decomposition with a fourth-order zero-phase Butterworth low-pass filter
# =============================================================================
#   Fig. 11 left : raw and low-pass filtered third latent variable zeta_3 (encoded test latents)
#           right: raw and low-pass filtered u'+ at the domain centre (x+ = 188, z+ = 56)
#   Fig. 12 left : joint PDF p(zeta_3^low, zeta_3^high) with marginals (Gaussian KDE)
#           right: joint PDF p(u'_low, u'_high) over the whole plane (smoothed histogram)
CUTOFF = 1e-2              # f+_cut
LATENT_INDEX = 2           # zeta_3 (latents ordered by activity)


def _plot_raw_and_low(time, raw, low, ylabel, legend_loc, ylim=None):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.plot(time, raw, color='gray', alpha=0.6, linewidth=2)
    ax.plot(time, low, color='tab:blue', linewidth=2, label=rf'Low-pass ($f^+ <$ {CUTOFF})')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlim(0, time[-1])
    ax.set_xlabel(r'$t^+$', fontsize=52, labelpad=20)
    ax.set_ylabel(ylabel, fontsize=52, labelpad=20)
    ax.legend(fontsize=28, loc=legend_loc, frameon=False)
    fig.tight_layout()
    return fig


def fig11_fig12(cfg, n_bins=200, smooth=1.5):
    out = os.path.join(cfg.figures_dir, 'fig11_fig12')
    fs = 1.0 / cfg.params['dt_plus']

    latents = io.load_test_latents(cfg.latent_dir)
    u = io.load_dns_test(cfg.flow_dir, cfg.params)[..., 0]                   # u'+, (Nt, Nx, Nz)
    io.check_same_length('DNS test set', u, 'encoded test latents', latents)

    # Encoded test latents over the forecast horizon (after the Transformer context), ordered by activity
    (target,) = io.sort_by_activity(latents[io.CONTEXT:])
    zeta = target[:, LATENT_INDEX]
    low = low_pass_filter(zeta, CUTOFF, fs)
    high = zeta - low
    time = np.arange(len(zeta)) * cfg.params['dt_plus']
    save_figure(_plot_raw_and_low(time, zeta, low, r'$\zeta_3$', 'lower right'), out, 'fig11_zeta3_low_pass')

    x_grid = np.linspace(low.min(), low.max(), 120)
    y_grid = np.linspace(high.min(), high.max(), 120)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = stats.gaussian_kde(np.vstack([low, high]))(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    fig = joint_pdf_with_marginals(X, Y, Z, x_grid, y_grid, stats.gaussian_kde(low)(x_grid),
                                   stats.gaussian_kde(high)(y_grid), hdr_levels(Z),
                                   r'$\zeta_3^{\mathrm{low}}$', r'$\zeta_3^{\mathrm{high}}$', colormap('bilbao_r'))
    save_figure(fig, out, 'fig12_zeta3_joint_pdf_low_high')

    # Streamwise velocity over the full DNS test set
    low = low_pass_filter(u, CUTOFF, fs, axis=0)
    high = u - low
    time = np.arange(u.shape[0]) * cfg.params['dt_plus']
    raw_c, low_c = u[:, CENTRE[0], CENTRE[1]], low[:, CENTRE[0], CENTRE[1]]
    save_figure(_plot_raw_and_low(time, raw_c, low_c, r"$u^{\prime +}$", 'upper left',
                                  ylim=(1.3 * raw_c.min(), 1.3 * raw_c.max())),
                out, 'fig11_u_centre_low_pass')

    # Joint PDF over the whole plane by a smoothed 2D histogram
    H, x_edges, y_edges = np.histogram2d(low.ravel(), high.ravel(), bins=n_bins, density=True)
    dx, dy = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]
    dens_x = gaussian_filter(H.sum(axis=1) * dy, sigma=smooth, mode='nearest')
    dens_y = gaussian_filter(H.sum(axis=0) * dx, sigma=smooth, mode='nearest')
    H = gaussian_filter(H, sigma=smooth, mode='nearest')
    x_grid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_grid = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = H.T
    fig = joint_pdf_with_marginals(X, Y, Z, x_grid, y_grid, dens_x, dens_y, hdr_levels(Z, cell_area=dx * dy),
                                   r'$u_{\mathrm{low}}$', r'$u_{\mathrm{high}}$', colormap('bilbao_r'))
    save_figure(fig, out, 'fig12_u_joint_pdf_low_high')


# =============================================================================
# Figure 13: spanwise two-point autocorrelation of u', v' and w' (DNS, baseline, end-to-end prediction)
# =============================================================================
TPC_COMPONENTS = ['u', 'v', 'w']
TPC_ZOOM_X = [(25, 35), (15, 25), (35, 45)]          # inset windows around the zero crossing
TPC_ZOOM_Y = [(-0.1, 0.2), (-0.1, 0.1), (-0.1, 0.1)]


def fig13(cfg):
    data = io.load_end_to_end(cfg.flow_dir, cfg.baseline_run, cfg.predicted_fields, cfg.params)
    dz_plus = cfg.params['z_plus'][1] - cfg.params['z_plus'][0]
    curves = [(data['dns'], COLORS['dns'], '--'),
              (data['baseline'], COLORS['baseline'], '-'),
              (data['prediction'], COLORS['prediction'], '-')]
    out = os.path.join(cfg.figures_dir, 'fig13')
    for c, name in enumerate(TPC_COMPONENTS):
        save_figure(plot_two_point_correlation(curves, dz_plus, c, TPC_ZOOM_X[c], TPC_ZOOM_Y[c]),
                    out, f'fig13_two_point_correlation_{name}')


# =============================================================================
# Table 4 and Figure 14: energy and spatio-temporal complexity of the end-to-end prediction
# =============================================================================
#   Table 4 : reconstructed energy coefficient E_k+ (Eq. 27) and global entropy H (Eq. 2) of
#             the DNS test data, the beta-VAE-GAN baseline and the end-to-end prediction
#   Fig. 14 : decay of the singular values of the snapshot matrices (first 50 modes)
# E_k+ and the singular values are computed over the forecast horizon (after the 128-snapshot
# context). As in Eq. (4) of the paper, H_test is computed on the full DNS test set.
SV_STYLES = {
    'dns': {'color': COLORS['dns'], 'marker': 'x', 'ls': '--', 'label': 'DNS'},
    'baseline': {'color': COLORS['baseline'], 'marker': 's', 'ls': '-', 'label': r'$\beta$-VAE-GAN'},
    'prediction': {'color': COLORS['prediction'], 'marker': 'D', 'ls': '-', 'label': 'Prediction'},
}
N_MODES = 50


def _plot_singular_values(singular_values):
    fig, ax = plt.subplots(figsize=(12, 9))
    for name, s in singular_values.items():
        st = SV_STYLES[name]
        ax.plot(s[:N_MODES], color=st['color'], marker=st['marker'], markerfacecolor='white',
                markeredgecolor=st['color'], markeredgewidth=2, linestyle=st['ls'], linewidth=3,
                markersize=12, alpha=0.8, label=st['label'])
    ax.set_yscale('log')
    ax.set_ylim(top=1e4)
    ax.set_xlabel(r'$i$', fontsize=50, labelpad=15)
    ax.set_ylabel(r'$\sigma_i$', fontsize=50, labelpad=15)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    fig.tight_layout()
    return fig


def table4_fig14(cfg):
    data = io.load_end_to_end(cfg.flow_dir, cfg.baseline_run, cfg.predicted_fields, cfg.params)
    entropy_test, _ = metrics.global_entropy(data['dns_full'])
    singular_values, rows = {}, {}
    for name in ('dns', 'baseline', 'prediction'):
        entropy, singular_values[name] = metrics.global_entropy(data[name])
        rows[name] = {
            'E_k+': None if name == 'dns' else metrics.energy_coefficient(data['dns'], data[name]),
            'H': entropy_test if name == 'dns' else entropy,
        }

    table = pd.DataFrame(rows).T.rename(index={'dns': 'Test data', 'baseline': 'Baseline (vae-gan)',
                                               'prediction': 'Inference (pred)'})
    print('\n  Table 4 - energy coefficient and global entropy')
    print(table.to_string(float_format=lambda x: f'{x:.4f}'))
    os.makedirs(cfg.tables_dir, exist_ok=True)
    table.to_csv(os.path.join(cfg.tables_dir, 'table4_energy_entropy.csv'))
    save_figure(_plot_singular_values(singular_values), os.path.join(cfg.figures_dir, 'fig14'), 'fig14_singular_values')


# =============================================================================
# Figure 15: quadrant analysis of the Reynolds shear stress, DNS vs end-to-end prediction
# =============================================================================
#   top left    : joint PDF P(u'+, v'+)
#   top right   : covariance integrand u'+ v'+ P(u'+, v'+) with the hyperbolic hole F = 3
#   bottom left : hole-filtered contribution to <u'v'> (sum over quadrants) vs F, normalised by the DNS <u'v'>
#   bottom right: hole-filtered contribution of each quadrant vs F
QA_BINS = 256                                   # histogram bins per direction (converged, see paper)
QA_LIM_X, QA_LIM_Y = (-7, 7), (-1.5, 1.5)       # plotting window
QA_F_HOLE = 3.0                                 # hole size shown in the top-right panel
QA_F_VALUES = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
QA_QUADRANT_COLORS = {'q1': 'green', 'q2': 'red', 'q3': 'purple', 'q4': 'blue'}


def _format_quadrant_axes(ax):
    ax.axhline(0, color='k', linewidth=0.8)
    ax.axvline(0, color='k', linewidth=0.8)
    (x0, x1), (y0, y1) = QA_LIM_X, QA_LIM_Y
    for label, (x, y) in {'Q1': (x1, y1), 'Q2': (x0, y1), 'Q3': (x0, y0), 'Q4': (x1, y0)}.items():
        ax.text(0.9 * x, 0.9 * y, label, fontsize=28, fontweight='bold', ha='center', va='center')
    ax.set_xlabel(r"$u'^{+}$", fontsize=50, labelpad=15)
    ax.set_ylabel(r"$v'^{+}$", fontsize=50, labelpad=15)
    ax.set_xlim(QA_LIM_X)
    ax.set_ylim(QA_LIM_Y)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3, steps=[1, 2, 5, 10]))


def _plot_quadrant_joint_pdf(qstats):
    fig, ax = plt.subplots(figsize=(12, 9))
    max_density = max(s['pdf'].max() for s in qstats.values())
    levels = np.linspace(0.1 * max_density, max_density, 10)
    for name, s in qstats.items():
        color = COLORS['dns'] if name == 'dns' else COLORS['prediction']
        ax.contour(s['U'], s['V'], s['pdf'], levels=levels, colors=color,
                   linewidths=1.2 if name == 'dns' else 1.5)
        i = np.unravel_index(np.argmax(s['pdf']), s['pdf'].shape)
        ax.scatter(s['U'][i], s['V'][i], marker='x', s=100, color=color, linewidth=2, zorder=10)
    _format_quadrant_axes(ax)
    fig.tight_layout()
    return fig


def _plot_covariance_integrand(qstats, uv_ref):
    fig, ax = plt.subplots(figsize=(12, 9))
    w_max = max(s['weighted'].max() for s in qstats.values())
    w_min = min(s['weighted'].min() for s in qstats.values())
    levels = np.concatenate([np.linspace(w_min, 0.1 * w_min, 8), np.linspace(0.1 * w_max, w_max, 8)])

    # Hyperbolic hole |u'v'| = F |<u'v'>|
    (x0, x1), (y0, y1) = QA_LIM_X, QA_LIM_Y
    Uh, Vh = np.meshgrid(np.linspace(0.9 * x0, 0.9 * x1, 300), np.linspace(0.9 * y0, 0.9 * y1, 300))
    ax.contour(Uh, Vh, np.abs(Uh * Vh), levels=[QA_F_HOLE * uv_ref], colors='gray',
               linewidths=2.5, linestyles=':', zorder=5)
    ax.text(0.3 * x1, 0.9 * y1, rf'$F = {QA_F_HOLE:g}$', fontsize=26, ha='center', va='center', color='gray', zorder=5)

    for name, s in qstats.items():
        color = COLORS['dns'] if name == 'dns' else COLORS['prediction']
        ax.contour(s['U'], s['V'], s['weighted'], levels=levels, colors=color, linewidths=1.5)
        for px, py in quadrant_peaks(s['U'], s['V'], s['weighted']):
            ax.scatter(px, py, marker='+' if name == 'dns' else 'x', s=60, color=color, linewidth=1.5, zorder=10)
    _format_quadrant_axes(ax)
    fig.tight_layout()
    return fig


def _plot_hole_filtered(results, uv_ref):
    styles = {'dns': (COLORS['dns'], '--', 'x'), 'prediction': (COLORS['prediction_alt'], '-', '^')}

    fig_sum, ax = plt.subplots(figsize=(12, 9))
    for name, (color, ls, marker) in styles.items():
        # Normalised by the DNS <u'v'> (negative): positive fraction of the total shear stress
        total = -np.array([sum(r.values()) for r in results[name]]) / uv_ref
        ax.plot(QA_F_VALUES, total, color=color, ls=ls, marker=marker, markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=2, linewidth=3, markersize=12, alpha=0.8)
    ax.set_xlabel(r'$F$', fontsize=48, labelpad=15)
    ax.set_ylabel(r'$\overline{u^\prime v^\prime}^{+}_{q} / \overline{u^\prime v^\prime}^{+}_\mathrm{ref}$',
                  fontsize=48, labelpad=15)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4, steps=[1, 2, 5, 10]))
    fig_sum.tight_layout()

    fig_q, ax = plt.subplots(figsize=(12, 9))
    for name, (color, ls, _) in styles.items():
        for q in ('q1', 'q2', 'q3', 'q4'):
            c = color if name == 'dns' else QA_QUADRANT_COLORS[q]
            ax.plot(QA_F_VALUES, [-r[q] for r in results[name]], color=c, ls=ls,
                    marker='x' if name == 'dns' else 'o', markerfacecolor='white', markeredgecolor=c,
                    markeredgewidth=2, linewidth=3, markersize=12, alpha=0.8)
    ax.set_xlabel(r'$F$', fontsize=46, labelpad=15)
    ax.set_ylabel(r'$-\overline{u^\prime v^\prime}^{+}_{q}$', fontsize=46, labelpad=15)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4, steps=[1, 2, 5, 10]))
    fig_q.tight_layout()
    return fig_sum, fig_q


def fig15(cfg):
    data = io.load_end_to_end(cfg.flow_dir, cfg.baseline_run, cfg.predicted_fields, cfg.params)
    uv = {name: (data[name][..., 0], data[name][..., 1]) for name in ('dns', 'prediction')}
    # Common histogram range covering both datasets
    lim_u = (min(u.min() for u, _ in uv.values()), max(u.max() for u, _ in uv.values()))
    lim_v = (min(v.min() for _, v in uv.values()), max(v.max() for _, v in uv.values()))
    uv_ref = np.abs(np.mean(uv['dns'][0] * uv['dns'][1]))     # |<u'v'>| of the DNS
    out = os.path.join(cfg.figures_dir, 'fig15')

    qstats = {}
    for name, (u, v) in uv.items():
        U, V, pdf, weighted, _ = joint_pdf(u, v, QA_BINS, lim_u, lim_v)
        qstats[name] = {'U': U, 'V': V, 'pdf': pdf, 'weighted': weighted}
    save_figure(_plot_quadrant_joint_pdf(qstats), out, 'fig15_joint_pdf')
    save_figure(_plot_covariance_integrand(qstats, uv_ref), out, 'fig15_covariance_integrand')

    results = {name: [quadrant_contributions_outside_hole(u, v, F, uv_ref, QA_BINS, lim_u, lim_v)
                      for F in QA_F_VALUES]
               for name, (u, v) in uv.items()}
    fig_sum, fig_q = _plot_hole_filtered(results, uv_ref)
    save_figure(fig_sum, out, 'fig15_hole_filtered_sum')
    save_figure(fig_q, out, 'fig15_hole_filtered_quadrants')

    i3 = QA_F_VALUES.index(QA_F_HOLE)
    for name in results:
        frac = -sum(results[name][i3].values()) / uv_ref
        print(f"  {name:>10s}: fraction of DNS <u'v'> carried by events with |u'v'| >= {QA_F_HOLE:g}|<u'v'>|: {frac:.3f}")


# =============================================================================
# Main function
# =============================================================================
RESULTS = {
    'fig06_fig07': (fig06_fig07, 'full DNS dataset and main_autoencoder.py reconstructions '
                                 '(simple_runner_vae*.sh inference_latent_dims)'),
    'table3': (table3, 'main_transformer.py rollouts (simple_runner_transformer.sh inference_paper_models)'),
    'fig08_fig09': (fig08_fig09, 'main_transformer.py rollout of the selected model'),
    'fig10_fig16': (fig10_fig16, 'full DNS test set and main_predictive_framework.py'),
    'fig11_fig12': (fig11_fig12, 'full DNS test set'),
    'fig13': (fig13, 'full DNS test set, main_autoencoder.py and main_predictive_framework.py'),
    'table4_fig14': (table4_fig14, 'full DNS test set, main_autoencoder.py and main_predictive_framework.py'),
    'fig15': (fig15, 'full DNS test set and main_predictive_framework.py'),
}


@gin.configurable
def main_postprocessing(results='all',
                        working_directory: str = '.',
                        flow_dir_relative: str = 'src/data/datasets/flow',
                        latent_dir_relative: str = 'src/data/datasets/latents',
                        output_dir_relative: str = 'results',
                        autoencoder_runs: dict = None,
                        include_pod: bool = True,
                        paper_models: dict = None,
                        selected_model: str = 'EA*',
                        baseline_run_relative: str = None,
                        predicted_fields_relative: str = 'results/predictive_framework/predicted_fields.npy'):
    """
    Args:
        results: 'all' or a list of results (keys of RESULTS)
        working_directory: Base working directory (repository root)
        flow_dir_relative: DNS dataset, with test/fields_2d.npy and params/
        latent_dir_relative: Encoded latents, with test/latents_mean.npy
        output_dir_relative: Output directory for figures and tables
        autoencoder_runs: {'vae': ..., 'vae_gan': ...} run directories with a {ld} placeholder (Fig. 6, 7)
        include_pod: Include the POD baseline in Figs. 6, 7 (computed here from the DNS training run)
        paper_models: {label: run directory} of the Transformer variants of Table 3. A run directory contains
            full_rollout_epoch_*/predicted_latents.npy, or directly predicted_latents.npy
        selected_model: Label of the model shown in Figs. 8 and 9
        baseline_run_relative: Run directory of the beta-VAE-GAN baseline (d_zeta = 4) with its test reconstructions
        predicted_fields_relative: Decoded rollout written by main_predictive_framework.py
    """
    root = os.path.abspath(working_directory)
    path = lambda p: None if p is None else os.path.join(root, p)
    flow_dir = path(flow_dir_relative)
    output_dir = path(output_dir_relative)
    cfg = types.SimpleNamespace(
        flow_dir=flow_dir,
        latent_dir=path(latent_dir_relative),
        params=io.load_params(os.path.join(flow_dir, 'params')),
        figures_dir=os.path.join(output_dir, 'figures'),
        tables_dir=os.path.join(output_dir, 'tables'),
        autoencoder_runs={k: path(v) for k, v in (autoencoder_runs or {}).items()},
        include_pod=include_pod,
        n_train_snapshots=sum(len(io.load_split_latents(path(latent_dir_relative), split))
                              for split in ('train', 'validation')),
        paper_models={k: path(v) for k, v in (paper_models or {}).items()},
        selected_model=selected_model,
        baseline_run=path(baseline_run_relative),
        predicted_fields=path(predicted_fields_relative),
    )
    set_style()

    selected = list(RESULTS) if results == 'all' else list(results)
    unknown = [r for r in selected if r not in RESULTS]
    if unknown:
        raise ValueError(f'Unknown results {unknown}. Available: {list(RESULTS)}')

    done, skipped = [], []
    for name in selected:
        function, needs = RESULTS[name]
        print('=' * 70)
        print(name)
        try:
            function(cfg)
            done.append(name)
        except FileNotFoundError as e:
            print(f'  [SKIPPED] missing input: {e}\n  Needs: {needs}.')
            skipped.append(name)
        plt.close('all')

    print('=' * 70)
    print(f'Done: {done}')
    if skipped:
        print(f'Skipped (missing inputs): {skipped}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Figures and tables of the paper with Gin Config')
    parser.add_argument('--gin_config', type=str, required=True,
                        help='Path to gin configuration file(s). Can specify multiple files separated by commas.')
    parser.add_argument('--gin_params', type=str, nargs='*', default=[],
                        help='Override gin parameters from command line. Format: param_name=value')
    args = parser.parse_args()

    for gin_file in args.gin_config.split(','):
        parse_config_file(gin_file.strip())
    if args.gin_params:
        gin.parse_config(args.gin_params)

    main_postprocessing()
