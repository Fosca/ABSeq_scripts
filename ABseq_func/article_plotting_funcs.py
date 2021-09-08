import sys

sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
import os.path as op
import config
from ABseq_func import *
import matplotlib

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from ABseq_func.stats_funcs import stats
from scipy import stats
import matplotlib.ticker as ticker
import mne.stats


# ----------------------------------------------------------------------------------------------------------------------
def heatmap_avg_subj(data_subjs, times, xlims=None, ylims=[-.5, .5], filter=False, fig_name='', figsize=(10 * 0.8, 1)):
    """
    Function to plot the data_subjs as a heatmap.
    data_subjs is of the shape n_subjects X n_times
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if xlims:
        idx = np.where((times >= xlims[0]) & (times <= xlims[1]))[0]
        data_subjs = data_subjs[:, idx]
        times = times[idx]

    # ---- determine the significant time-windows ----
    mean_data = np.mean(data_subjs, axis=0)
    if filter == True:
        mean = savgol_filter(mean_data, 11, 3)
    extent = [min(times), max(times), 0, 0.03]
    plt.imshow(mean_data[np.newaxis, :], aspect="auto", cmap="RdBu_r", extent=extent, vmin=ylims[0], vmax=ylims[1])
    plt.gca().set_yticks([])
    plt.colorbar()
    if fig_name is not None:
        plt.gcf().savefig(fig_name, dpi=300)


# ----------------------------------------------------------------------------------------------------------------------
def plot_timecourses(data_seq_subjs, times, filter=False, fig_name='', color='b', chance=0.5, pos_sig=None, plot_shaded_vertical=False, xlims=None):
    """
    param data_seq_subjs: n_subject X n_times array that you want to plot as mean + s.e.m in shaded bars
    param pos_sig: If you want to plot the significant time-points as a line under the graph, set this value to the y position of the line
    param plot_shaded_vertical: True if you want to plot a grey zone where the temporal cluster test is significan
    param plot_shaded_vertical: xlims: if not None, will crop the data according to time before plotting and computing significant clusters
    """
    plt.gcf()

    # ---- crop data if necessary
    if xlims:
        idx = np.where((times >= xlims[0]) & (times <= xlims[1]))[0]
        data_seq_subjs = data_seq_subjs[:, idx]
        times = times[idx]

    # ---- determine the significant time-windows ----
    if chance is not None:
        t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(data_seq_subjs[:, times > 0] - chance, n_permutations=2 ** 12, out_type='mask')  # If threshold is None, t-threshold equivalent to p < 0.05 (if t-statistic))
        good_cluster_inds = np.where(cluster_pv < 0.05)[0]

    n_subj = data_seq_subjs.shape[0]
    # ----- average the data and determine the s.e.m -----
    mean_data = np.mean(data_seq_subjs, axis=0)
    ub = (mean_data + np.std(data_seq_subjs, axis=0) / (np.sqrt(n_subj)))
    lb = (mean_data - np.std(data_seq_subjs, axis=0) / (np.sqrt(n_subj)))

    if filter == True:
        mean_data = savgol_filter(mean_data, 11, 3)
        ub = savgol_filter(ub, 11, 3)
        lb = savgol_filter(lb, 11, 3)

    ylims = plt.gca().get_ylim()
    stat_times = times[times > 0]  # since stats were done on times > 0 (time index of clusters is based on this)
    if plot_shaded_vertical:
        if len(good_cluster_inds) > 0:
            for i_clu, clu_idx in enumerate(good_cluster_inds):
                clu_times = stat_times[clusters[clu_idx]]
                # plt.gca().fill_between([clu_times[0], clu_times[-1]], ylims[1], ylims[0], color='black', alpha=.1)
                plt.gca().fill_between([clu_times[0], clu_times[-1]], -99999, 99999, color='black', alpha=.08, linewidth=0.0)
                print("The p-value of the cluster number %i" % (i_clu) + " is {:.5f}".format(cluster_pv[clu_idx]))
        plt.gca().set_ylim(ylims)
        return True

    plt.fill_between(times, ub, lb, alpha=.2, color=color)
    plt.plot(times, mean_data, linewidth=1.5, color=color)

    if chance is not None:
        if len(good_cluster_inds) > 0:
            for i_clu, clu_idx in enumerate(good_cluster_inds):
                clu_times = times[clusters[clu_idx]]
                sig_mean = mean_data[times > 0]
                sig_mean = sig_mean[clusters[clu_idx]]
                if (pos_sig is not None):
                    plt.plot(clu_times, [pos_sig] * len(clu_times), linestyle='-', color=color, linewidth=2)
                else:
                    plt.plot(clu_times, sig_mean, linewidth=3, color=color)

    if fig_name is not None:
        plt.gcf().savefig(fig_name)


# ----------------------------------------------------------------------------------------------------------------------
def compute_corr_comp(data):
    """
    Function that takes data in the shape of n_seq X n_subjects X n_times and returns n_subjects time courses of the pearson
    correlation with complexity
    """
    complexity = [4, 6, 6, 6, 12, 15, 28]
    n_seq, n_subj, n_times = data.shape
    pearson = []
    for nn in range(n_subj):
        # ---- for 1 subject, diagonal of the GAT for all the 7 sequences through time ---
        dd = data[:, nn, :]
        r = []
        # Pearson correlation
        for t in range(n_times):
            r_t, _ = stats.pearsonr(dd[:, t], complexity)
            r.append(r_t)
        pearson.append(r)
    pearson = np.asarray(pearson)

    return pearson


# ----------------------------------------------------------------------------------------------------------------------
def plot_7seq_timecourses(data_7seq, times, save_fig_path='SVM/standard_vs_deviant/', fig_name='All_sequences_standard_VS_deviant_cleaned_', suffix='',
                          pos_horizontal_bar=0.47, plot_pearson_corrComplexity=True, chance=0, xlims=None, ymin=None, ylabel=None, filter=False):
    """
    param data_7seq: data in the shape of 7 X n_subjects X n_times
    param times: the times for the plot
    """
    if xlims == None:
        xlims = [times[0], times[-1]]

    NUM_COLORS = 7
    cm = plt.get_cmap('viridis')
    colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(10 * 0.8, 7 * 0.8))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    # plt.axhline(0.5, linestyle='-', color='black', linewidth=1)  # ligne horizontale à 0.5 pas applicable pour valeurs GFP à 1e-25!
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)

    for ii, SeqID in enumerate(range(1, 8)):
        plot_timecourses(data_7seq[ii, :, :], times, filter=filter, color=colorslist[SeqID - 1], pos_sig=pos_horizontal_bar - 0.005 * ii, chance=chance)  #

    if plot_pearson_corrComplexity:
        pearsonr = compute_corr_comp(data_7seq)
        plot_timecourses(pearsonr, times, chance=0, plot_shaded_vertical=True, xlims=xlims)

    # Set limits
    ax.set_xlim(xlims)
    if ymin is not None:
        ax.set_ylim(ymin=ymin)

    # # Remove some spines?
    # for key in ('top', 'right'):
    #     ax.spines[key].set(visible=False)

    # Add ylabel and format x10^...
    if ylabel == 'GFP':
        ax.set_ylabel(ylabel, fontsize=14)
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.get_yaxis().set_major_formatter(fmt)
    plt.gca().set_xlabel('Time (ms)', fontsize=14)
    utils.create_folder(op.join(config.fig_path, save_fig_path))
    plt.gcf().savefig(op.join(config.fig_path, save_fig_path, fig_name + suffix + '.svg'))
    plt.gcf().savefig(op.join(config.fig_path, save_fig_path, fig_name + suffix + '.png'), dpi=300)
    plt.close('all')
