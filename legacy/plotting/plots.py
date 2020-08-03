import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter


def subplot_grid(
        n_plots, n_cols, sharex=True, sharey=True, scale=1.0, dpi=None):
    n_cols = min(n_cols, n_plots)
    n_rows = (n_plots // n_cols) + min(1, n_plots % n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, sharex=sharex, sharey=sharey,
        dpi=dpi, figsize=(
            scale * (0.5 + 3.8 * n_cols),
            scale * (0.5 + 2.8 * n_rows)))
    axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.T
    for i in range(n_cols * n_rows):
        i_row, i_col = i // n_cols, i % n_cols
        if i < n_plots:
            axes[i_row, i_col].tick_params(
                "both", direction="in",
                bottom=True, top=True, left=True, right=True)
        else:
            fig.delaxes(axes[i_row, i_col])
    fig.set_facecolor("white")
    return fig, axes


def hist_step_filled(
        data, bins, color, label=None, ax=None, normed=True, rescale=1.0,
        weight=None):
    count = np.histogram(
        data, bins, density=normed, weights=weight)[0] * rescale
    y = np.append(count[0], count)
    if ax is None:
        ax = plt.gca()
    ax.fill_between(
        bins, y, 0.0, color=color, step="pre", alpha=0.4, label=label)
    ax.step(bins, y, color=color)


def hist_smooth_filled(
        data, bins, color, label=None, ax=None, normed=True, kwidth="scott"):
    z = np.linspace(bins[0], bins[-1], 251)
    kde = gaussian_kde(data, bw_method=kwidth)
    y = kde(z)
    if not normed:
        y *= len(data)
    if ax is None:
        ax = plt.gca()
    ax.fill_between(z, y, 0.0, color=color, alpha=0.4, label=label)
    ax.plot(z, y, color=color)


def scatter_dense(xdata, ydata, color, s=1, ax=None, alpha=0.25):
    if ax is None:
        ax = plt.gca()
    handle = ax.scatter(
        xdata, ydata, s=s, color=color, marker=".", alpha=alpha)
    hcolor = handle.get_facecolors()[0].copy()
    hcolor[-1] = 1.0
    handle = plt.scatter([], [], color=hcolor)
    return handle


def subsampling_mask(data, npoints):
    sample_factor = float(npoints) / float(len(data))
    mask = np.ones(len(data), dtype="bool")
    for i in range(1, len(data)):
        inew = i * sample_factor
        iold = inew - sample_factor
        mask[i] = int(inew) > int(iold)
    return mask


def get_plotting_folder(input_cat):
    rootdir, catname = os.path.split(os.path.abspath(input_cat))
    plot_dir = os.path.join(
        rootdir, "plots_" + os.path.splitext(catname)[0])
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    print("write plots to: %s" % (plot_dir + os.sep))
    return plot_dir


def parse_data_labels(dataset, labels, idx):
    if dataset is None:
        data = None
        label = None
    else:
        if hasattr(dataset, "columns"):
            columns = dataset.colnames
            data = dataset if len(columns) == 1 else dataset[columns[idx]]
        elif hasattr(dataset, "shape"):
            data = dataset if len(dataset.shape) == 1 else dataset[idx]
        else:
            data = dataset if len(dataset) == 1 else dataset[idx]
        label = labels if type(labels) is str else labels[idx]
    return data, label


def histogram_per_filter(
        dcols, dlabels, scols, slabels, n_filters,
        bins=50, xlabel=None, label_pos="best", n_cols=3, scale=1.0,
        dweight=None, sweight=None):
    # make a figure grid
    fig, axes = subplot_grid(n_filters, n_cols, scale=scale)
    try:
        n_rows, n_cols = axes.shape
    except ValueError:
        n_rows = 1
    for i in range(n_filters):
        i_row, i_col = i // n_cols, i % n_cols
        ax = axes[i_row, i_col]
        # get the data and labels to plot
        dcol, dlabel = parse_data_labels(dcols, dlabels, idx=i)
        scol, slabel = parse_data_labels(scols, slabels, idx=i)
        # draw histograms
        if dcols is not None:
            hist_step_filled(
                dcol, bins, "C3", label=dlabel, ax=ax, weight=dweight)
        hist_step_filled(
            scol, bins, "C0", label=slabel, ax=ax, weight=sweight)
        ax.legend(loc=label_pos, prop={"size": 8})
    # decorate all axes with labels and grids, even if not used
    for i in range(n_cols * n_rows):
        i_row, i_col = i // n_cols, i % n_cols
        ax = axes[i_row, i_col]
        # add labels to the left-most and bottom axes and set limits
        if xlabel is not None and i_row == n_rows - 1:
            ax.set_xlabel(xlabel)
        if i_col == 0:
            ax.set_ylabel("relative frequency")
        if type(bins) is not int:
            ax.set_xlim([bins[0], bins[-1]])
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
    fig.tight_layout(h_pad=0.0, w_pad=0.0)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    return fig


def scatter_per_filter(
        dydata, dcols, dlabels, sydata, scols, slabels, n_filters,
        xlim=[16.5, 26.5], ylabel=None, label_pos="best", n_cols=3, scale=1.0):
    # make a figure grid
    fig, axes = subplot_grid(n_filters, n_cols, scale=scale)
    try:
        n_rows, n_cols = axes.shape
    except ValueError:
        n_rows = 1
    for i in range(n_filters):
        i_row, i_col = i // n_cols, i % n_cols
        ax = axes[i_row, i_col]
        # get the data and labels to plot
        dcol, dlabel = parse_data_labels(dcols, dlabels, idx=i)
        dy, dlabel = parse_data_labels(dydata, dlabels, idx=i)
        scol, slabel = parse_data_labels(scols, slabels, idx=i)
        sy, slabel = parse_data_labels(sydata, slabels, idx=i)
        # create scatter plot and create fake handles for the legend
        handles = []
        labels = []
        if dcols is not None:
            handles.append(
                scatter_dense(dcol, dy, "C3", ax=ax, alpha=0.25))
            labels.append(dlabel)
        handles.append(
            scatter_dense(scol, sy, "C0", ax=ax, alpha=0.25))
        labels.append(slabel)
        ax.legend(
            handles=handles, labels=labels, loc=label_pos, prop={"size": 8})
    # decorate all axes with labels and grids, even if not used
    for i in range(n_cols * n_rows):
        i_row, i_col = i // n_cols, i % n_cols
        ax = axes[i_row, i_col]
        # add labels to the left-most and bottom axes and set limits
        if ylabel is not None and i_col == 0:
            ax.set_ylabel(ylabel)
        if i_row == n_rows - 1:
            ax.set_xlabel("magnitude")
        ax.set_xlim(xlim)
        ax.grid(alpha=0.25)
    fig.tight_layout(h_pad=0.0, w_pad=0.0)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    return fig


def contour_per_filter(
        dydata, dcols, dlabels, sydata, scols, slabels, n_filters,
        xlim=[16.5, 26.5], ylim=[0.0, 3.0], ylabel=None, label_pos="best",
        n_bins=200, n_cols=3, scale=1.0):
    sigma = 1.5
    # make a figure grid
    fig, axes = subplot_grid(n_filters, n_cols, scale=scale)
    try:
        n_rows, n_cols = axes.shape
    except ValueError:
        n_rows = 1
    for i in range(n_filters):
        i_row, i_col = i // n_cols, i % n_cols
        ax = axes[i_row, i_col]
        # get the data and labels to plot
        dcol, dlabel = parse_data_labels(dcols, dlabels, idx=i)
        dy, dlabel = parse_data_labels(dydata, dlabels, idx=i)
        scol, slabel = parse_data_labels(scols, slabels, idx=i)
        sy, slabel = parse_data_labels(sydata, slabels, idx=i)
        # create scatter plot and create fake handles for the legend
        handles = []
        labels = []
        if dcols is not None:
            counts, x, y = np.histogram2d(
                dcol, dy, bins=[
                    np.linspace(*xlim, n_bins), np.linspace(*ylim, n_bins)])
            X, Y = np.meshgrid(
                (x[1:] + x[:-1]) / 2.0, (y[1:] + y[:-1]) / 2.0)
            plot_data = gaussian_filter(counts.T, sigma)
            levels = np.linspace(0, plot_data.max(), 8)[1:-1:2]
            handles.append(ax.contour(
                X, Y, plot_data, levels=levels, colors="C3").collections[0])
            labels.append(dlabel)
        counts, x, y = np.histogram2d(
                scol, sy, bins=[
                    np.linspace(*xlim, n_bins), np.linspace(*ylim, n_bins)])
        X, Y = np.meshgrid(
            (x[1:] + x[:-1]) / 2.0, (y[1:] + y[:-1]) / 2.0)
        plot_data = gaussian_filter(counts.T, sigma)
        levels = np.linspace(0, plot_data.max(), 8)[1:-1:2]
        handles.append(ax.contour(
            X, Y, plot_data, levels=levels, colors="C0").collections[0])
        labels.append(slabel)
        ax.legend(
            handles=handles, labels=labels, loc=label_pos, prop={"size": 8})
    # decorate all axes with labels and grids, even if not used
    for i in range(n_cols * n_rows):
        i_row, i_col = i // n_cols, i % n_cols
        ax = axes[i_row, i_col]
        # add labels to the left-most and bottom axes and set limits
        if ylabel is not None and i_col == 0:
            ax.set_ylabel(ylabel)
        if i_row == n_rows - 1:
            ax.set_xlabel("magnitude")
        ax.set_xlim(xlim)
        ax.grid(alpha=0.25)
    fig.tight_layout(h_pad=0.0, w_pad=0.0)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    return fig


class photoz_statistics(object):

    def __init__(self, z_model, z_phot):
        # compute bias for each galaxy and classify outliers
        self.z_model = z_model
        self.z_phot = z_phot
        self.delta_z = (z_phot - z_model) / (1.0 + z_model)

    def bias(self, bindata, bins):
        bins = np.asarray(bins)
        bin_centers = np.zeros(len(bins) - 1)
        bias = np.zeros_like(bin_centers)
        for i in range(len(bin_centers)):
            bin_mask = (
                (bindata >= bins[i]) &
                (bindata < bins[i + 1]))
            bin_centers[i] = np.mean(bindata[bin_mask])
            bias[i] = np.mean(self.delta_z[bin_mask])
        return bin_centers, bias

    def MAD(self, bindata, bins):
        bin_centers, bias = self.bias(bindata, bins)
        mad = np.zeros_like(bin_centers)
        for i in range(len(bins) - 1):
            bin_mask = (
                (bindata >= bins[i]) &
                (bindata < bins[i + 1]))
            mad[i] = 1.4826 * np.median(
                np.abs(self.delta_z[bin_mask] - bias[i]))
        return bin_centers, mad

    def nMAD_fraction(self, bindata, bins, n=3):
        bin_centers, mad = self.MAD(bindata, bins)
        frac = np.zeros_like(bin_centers)
        for i in range(len(bins) - 1):
            bin_mask = (
                (bindata >= bins[i]) &
                (bindata < bins[i + 1]))
            # count objects with delta_z > n * mad
            delta_z_bin = self.delta_z[bin_mask]
            try:
                frac[i] = (
                    np.count_nonzero(np.abs(delta_z_bin) > (mad[i] * n)) /
                    float(np.count_nonzero(bin_mask)))
            except ZeroDivisionError:
                frac[i] = None
        return bin_centers, frac

    def outlier_fraction(self, bindata, bins, sigma=0.15):
        bins = np.asarray(bins)
        bin_centers = (bins[1:] + bins[:-1]) / 2.0
        frac = np.zeros_like(bin_centers)
        for i in range(len(bins) - 1):
            bin_mask = (
                (bindata >= bins[i]) &
                (bindata < bins[i + 1]))
            delta_z_bin = self.delta_z[bin_mask]
            try:
                frac[i] = (
                    np.count_nonzero(np.abs(delta_z_bin) > sigma) /
                    float(np.count_nonzero(bin_mask)))
            except ZeroDivisionError:
                frac[i] = None
        return bin_centers, frac

    def generate_stat_text(self, bindata, dmin, dmax, n=3, sigma=0.15):
        strings = (
            "%.2f ≤ z < %.2f:" % (dmin, dmax),
            "μ_Δz:   % .3f" % self.bias(
                bindata, [dmin, dmax])[1],
            "σ_m:    % .3f" % self.MAD(
                bindata, [dmin, dmax])[1],
            "η_%d:    % .3f" % (
                n, self.nMAD_fraction(bindata, [dmin, dmax], n=n)[1]),
            "ζ_%.2f: % .3f" % (
                sigma, self.outlier_fraction(
                    bindata, [dmin, dmax], sigma=sigma)[1]))
        return "\n".join(strings)

    def print(self, bindata, dmin, dmax, n=3, sigma=0.15):
        print(self.generate_stat_text(bindata, dmin, dmax, n, sigma))

    def plot_statistics(
            self, bindata, xlabel, bins=None, n=3, sigma=0.15, axes=None,
            step=False, scale=1.0):
        if axes is None:
            fig, axes = plt.subplots(
                3, 1, sharex=True, figsize=(6 * scale, 12 * scale))
        else:
            assert(len(axes) == 3)
        # compute automatic bins
        if bins is None:
            bins = np.percentile(bindata, np.linspace(0.5, 96.5, 25))
        # plot MAD
        ax = axes[0]
        x, y = self.MAD(bindata, bins)
        if step:
            ax.step(bins, np.append(y[0], y))
        else:
            ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\sigma_\mathrm{m}$")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        # plot bias
        ax = axes[1]
        ax.axhline(y=0, color="k", lw=0.5)
        x, y = self.bias(bindata, bins)
        if step:
            ax.step(bins, np.append(y[0], y))
        else:
            ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\mu_\mathrm{\Delta z}$")
        ax.grid(alpha=0.25)
        # plot outlier fraction
        ax = axes[2]
        x, y = self.outlier_fraction(bindata, bins, sigma=0.15)
        if step:
            ax.step(bins, np.append(y[0], y))
        else:
            ax.plot(x, y)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\zeta_{%.2f}$" % sigma)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)

    def plot_photoz_scatter(self, xdata, xlabel, cmap="plasma", scale=1.0):
        fig, axes = subplot_grid(1, 1, scale=scale)
        ax = axes[0, 0]
        ax.set_aspect("equal")
        ax.grid(alpha=0.33)
        # get the gridsize of the photo-z data
        grid = set(np.diff(np.unique(self.z_phot)))
        dphot = 0.5 * sum(grid) / len(grid)
        # spread the data uniformly between the discrete values
        z_phot_smooth = self.z_phot + np.random.uniform(
            -dphot, dphot, size=self.z_phot.shape)
        ax.scatter(
            xdata, z_phot_smooth, c="C0", marker=".", s=0.25, alpha=0.33)
        z_spec_lim = np.percentile(xdata, 99.75)
        z_phot_lim = np.percentile(self.z_phot.max(), 99.75)
        ax.plot(
            [0.0, z_spec_lim], [0.0, z_spec_lim], color="k", lw=1, alpha=0.67)
        # indicate outlier region for z < 0.9
        mad_lower = self.MAD(self.z_phot, [0.1, 0.9])[1]
        ax.plot(
            [0.0, (0.9 + mad_lower) / (1.0 - mad_lower)],
            [-mad_lower, 0.9],
            color="k", lw=1, ls="--", alpha=0.67)
        ax.plot(
            [0.0, (0.9 - mad_lower) / (1.0 + mad_lower)],
            [+mad_lower, 0.9],
            color="k", lw=1, ls="--", alpha=0.67)
        # indicate outlier region for z > 0.9
        mad_upper = self.MAD(self.z_phot, [0.9, z_phot_lim])[1]
        ax.plot(
            [(0.9 + mad_upper) / (1.0 - mad_upper),
             (z_phot_lim + mad_upper) / (1.0 - mad_upper)],
            [0.9, z_phot_lim],
            color="k", lw=1, ls="--", alpha=0.67)
        ax.plot(
            [(0.9 - mad_upper) / (1.0 + mad_upper),
             (z_phot_lim - mad_upper) / (1.0 + mad_upper)],
            [0.9, z_phot_lim],
            color="k", lw=1, ls="--", alpha=0.67)
        # plot connecting lines
        ax.plot(
            [(0.9 - mad_lower) / (1.0 + mad_lower),
             (0.9 - mad_upper) / (1.0 + mad_upper)],
            [0.9, 0.9],
            color="k", lw=1, ls="--", alpha=0.67)
        ax.plot(
            [(0.9 + mad_lower) / (1.0 - mad_lower),
             (0.9 + mad_upper) / (1.0 - mad_upper)],
            [0.9, 0.9],
            color="k", lw=1, ls="--", alpha=0.67)
        # add box with statistics
        ax.fill_between(
            [0.0, z_spec_lim], [0.9, 0.9], [z_phot_lim, z_phot_lim],
            color="k", alpha=0.125, edgecolor="none")
        props = dict(edgecolor="k", facecolor='white')
        ax.text(
            1.05, 0.8, self.generate_stat_text(self.z_phot, 0.9, z_phot_lim),
            transform=ax.transAxes, fontsize=7, fontfamily="monospace",
            va='center', ha='left', bbox=props)
        ax.text(
            1.05, 0.2, self.generate_stat_text(self.z_phot, 0.1, 0.9),
            transform=ax.transAxes, fontsize=7, fontfamily="monospace",
            va='center', ha='left', bbox=props)
        # set x-axis ticks, labels, limits
        xmin = 0.0  # xdata.min()
        ax.set_xlim([0.0, z_spec_lim])
        xticks = [
            f for f in np.arange(0.0, 20.0, 0.25) if xmin <= f <= z_spec_lim]
        ax.set_xticks(xticks, minor=False)
        ax.set_xlabel(xlabel)
        # set x-axis ticks, labels, limits
        ymin = 0.0  # self.z_phot.min()
        ax.set_ylim([ymin, z_phot_lim])
        yticks = [
            f for f in np.arange(0.0, 20.0, 0.25) if ymin <= f <= z_phot_lim]
        ax.set_yticks(yticks, minor=False)
        ax.set_ylabel("Z_B")
        return fig
