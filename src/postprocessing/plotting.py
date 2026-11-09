"""
Shared plotting style and helpers.
"""
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Colours used throughout the paper
COLORS = {
    'dns': 'dimgray',
    'target': 'dimgray',
    'pod': '#5789BA',
    'vae': '#EF8636',
    'vae_gan': 'seagreen',          # compression study (Figs. 6, 7)
    'baseline': '#EF8636',          # beta-VAE-GAN baseline in the end-to-end study (Figs. 13, 14)
    'prediction': 'seagreen',
    'prediction_alt': '#009E73',
}


def set_style(usetex=None):
    """Paper figure style. LaTeX rendering is used when a LaTeX installation is found."""
    if usetex is None:
        usetex = shutil.which('latex') is not None
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 46,
        'text.usetex': usetex,
        'xtick.labelsize': 42, 'ytick.labelsize': 42,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 10, 'ytick.major.size': 10,
        'xtick.major.width': 2, 'ytick.major.width': 2,
        'xtick.major.pad': 10, 'ytick.major.pad': 10,
        'xtick.minor.size': 5, 'ytick.minor.size': 5,
        'xtick.minor.width': 1, 'ytick.minor.width': 1,
        'lines.linewidth': 3.0, 'lines.markersize': 12,
        'axes.linewidth': 2,
        'legend.fontsize': 10,
        'figure.figsize': (12, 9),
        'figure.dpi': 100,
    })


def save_figure(fig, out_dir, name, formats=('png', 'eps')):
    """Save a figure as <out_dir>/<fmt>/<name>.<fmt> for each requested format, then close it."""
    for fmt in formats:
        fmt_dir = os.path.join(out_dir, fmt)
        os.makedirs(fmt_dir, exist_ok=True)
        fig.savefig(os.path.join(fmt_dir, f'{name}.{fmt}'), format=fmt, dpi=300)
    print(f'Saved figure {name} ({", ".join(formats)}) in {out_dir}')
    plt.close(fig)


def add_inset(ax, xlim, ylim, position='upper right', width=0.25, keep_slopes=False):
    """Inset axes showing a zoomed region of `ax`.

    keep_slopes: scale the inset height so that slopes look the same as in the main axes.
    """
    height = width
    if keep_slopes:
        main_xlim, main_ylim = ax.get_xlim(), ax.get_ylim()
        height = width * ((ylim[1] - ylim[0]) / (main_ylim[1] - main_ylim[0])) \
                       / ((xlim[1] - xlim[0]) / (main_xlim[1] - main_xlim[0]))
    x0 = 0.95 - width if position == 'upper right' else 0.085
    axins = ax.inset_axes([x0, 0.95 - height, width, height])
    axins.set_xlim(xlim)
    axins.set_ylim(ylim)
    axins.tick_params(axis='both', which='major', labelsize=30, direction='in', length=6, width=1.5)
    for spine in axins.spines.values():
        spine.set_linewidth(1.5)
    return axins


def joint_pdf_with_marginals(X, Y, Z, x_grid, y_grid, dens_x, dens_y, levels, xlabel, ylabel, cmap):
    """Filled joint PDF with highest-density contours and the two marginal PDFs (Fig. 12)."""
    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(4, 4, hspace=0.06, wspace=0.06, left=0.16, right=0.82, bottom=0.14, top=0.92)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_mx = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_my = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    ax_joint.contourf(X, Y, Z, levels=25, cmap=cmap)
    ax_joint.contour(X, Y, Z, levels=levels, colors='k', alpha=0.35, linewidths=1.2)
    ax_joint.axvline(0, color='k', linestyle=':', lw=1.5, alpha=0.45)
    ax_joint.axhline(0, color='k', linestyle=':', lw=1.5, alpha=0.45)
    ax_joint.set_xlabel(xlabel, labelpad=15)
    ax_joint.set_ylabel(ylabel, labelpad=15)

    accent = cmap(0.5)
    ax_mx.plot(x_grid, dens_x, color=accent, lw=3.0)
    ax_mx.fill_between(x_grid, dens_x, alpha=0.25, color=accent)
    ax_mx.tick_params(axis='x', labelbottom=False, bottom=True)
    ax_mx.set_ylim(bottom=0)
    ax_my.plot(dens_y, y_grid, color=accent, lw=3.0)
    ax_my.fill_betweenx(y_grid, dens_y, alpha=0.25, color=accent)
    ax_my.tick_params(axis='y', labelleft=False, left=True)
    ax_my.set_xlim(left=0)
    for ax in (ax_mx, ax_my):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    return fig


def colormap(name='bilbao_r'):
    """Scientific colour map from cmcrameri (falls back to a matplotlib map if not installed)."""
    try:
        from cmcrameri import cm
        return getattr(cm, name)
    except ImportError:
        return plt.get_cmap('pink_r' if name == 'bilbao_r' else 'viridis')



def plot_two_point_correlation(curves, dz_plus, channel, zoom_x, zoom_y):
    """Spanwise two-point autocorrelation with a zoomed inset around the zero crossing (Figs. 7, 13).

    curves: list of (fields (Nt, Nx, Nz, 3), colour, linestyle); the first entry is the reference.
    The inset aspect is set from the reference curve so that slopes look as in the main axes.
    """
    from matplotlib.ticker import MaxNLocator
    from .statistics import two_point_correlation

    comp = [r"u^\prime", r"v^\prime", r"w^\prime"][channel]
    fig, ax = plt.subplots(figsize=(12, 9))
    axins = None
    for i, (field, color, ls) in enumerate(curves):
        lags, R = two_point_correlation(field[..., channel], axis=2)
        r = lags * dz_plus
        ax.plot(r, R, color=color, linestyle=ls, linewidth=3)
        if i == 0:
            axins = add_inset(ax, zoom_x, zoom_y, position='upper right', keep_slopes=True)
        axins.plot(r, R, color=color, linestyle=ls, linewidth=3)
    for a in (ax, axins):
        a.axhline(0, color='k', linewidth=1.5, linestyle=':')
    ax.set_xlabel(r'$r_z^+$', fontsize=50, labelpad=15)
    ax.set_ylabel(rf'$R_{{{comp} {comp}}}^+$', fontsize=50, labelpad=15)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=6, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=6, steps=[1, 2, 5, 10]))
    ax.set_xlim(left=0)
    fig.tight_layout()
    return fig
