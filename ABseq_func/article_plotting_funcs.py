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

# ----------------------------------------------------------------------------------------------------------------------
def heatmap_avg_subj(data_subjs, times, filter=True, fig_name='',figsize=(10*0.8, 1)):
    """
    Function to plot the data_subjs as a heatmap.
    data_subjs is of the shape n_subjects X n_times
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # ---- determine the significant time-windows ----
    mean_data = np.mean(data_subjs, axis=0)
    if filter == True:
        mean = savgol_filter(mean_data, 11, 3)
    extent = [min(times), max(times), 0, 0.03]
    plt.imshow(mean_data[np.newaxis, :], aspect="auto", cmap="viridis", extent=extent)
    plt.gca().set_yticks([])
    plt.colorbar()
    if fig_name is not None:
        plt.gcf().savefig(fig_name)

# ----------------------------------------------------------------------------------------------------------------------
def plot_timecourses(data_seq_subjs, times, filter=False, fig_name='', color='b', chance = 0.5, pos_sig = None, plot_shaded_vertical = False):

    """
    param data_seq_subjs: n_subject X n_times array that you want to plot as mean + s.e.m in shaded bars
    param pos_sig: If you want to plot the significant time-points as a line under the graph, set this value to the y position of the line
    param plot_shaded_vertical: True if you want to plot a grey zone where the temporal cluster test is significan
    """
    plt.gcf()
    # ---- determine the significant time-windows ----
    if chance is not None:
        sig = stats_funcs.stats(data_seq_subjs[:, times > 0] - chance)
        # ---- determine the significant times ----
        times_sig = times[times > 0]
        times_sig = times_sig[sig<0.05]
    n_subj = data_seq_subjs.shape[0]
    # ----- average the data and determine the s.e.m -----
    mean_data = np.mean(data_seq_subjs, axis=0)
    ub = (mean_data + np.std(data_seq_subjs, axis=0) / (np.sqrt(n_subj)))
    lb = (mean_data - np.std(data_seq_subjs, axis=0) / (np.sqrt(n_subj)))

    if filter == True:
        mean_data = savgol_filter(mean_data, 11, 3)
        ub = savgol_filter(ub, 11, 3)
        lb = savgol_filter(lb, 11, 3)

    if plot_shaded_vertical and len(times_sig)!=0:
        ylims = plt.gca().get_ylim()
        plt.gca().fill_between([times_sig[0],times_sig[-1]],ylims[1], ylims[0], color='black', alpha=.1)
        return True

    if chance is not None:
        sig_mean = mean_data[times>0]
        sig_mean = sig_mean[sig<0.05]
    plt.fill_between(times, ub, lb, alpha=.2,color=color)
    plt.plot(times, mean_data, linewidth=1.5,color=color)
    if (chance is not None) and (pos_sig is not None):
        plt.plot(times_sig,[pos_sig]*len(times_sig), linestyle='-', color=color, linewidth=2)
    elif (chance is not None):
        plt.plot(times_sig,sig_mean,linewidth=3,color=color)
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
def plot_7seq_timecourses(data_7seq,times, save_fig_path='SVM/standard_vs_deviant/',fig_name='All_sequences_standard_VS_deviant_cleaned_', suffix= '',
                          pos_horizontal_bar = 0.47,plot_pearson_corrComplexity=True):

    """
    param data_7seq: data in the shape of 7 X n_subjects X n_times
    param times: the times for the plot
    """

    NUM_COLORS = 7
    cm = plt.get_cmap('viridis')
    colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])

    fig, ax = plt.subplots(1, 1, figsize=(10 * 0.8, 7 * 0.8))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    plt.axhline(0.5, linestyle='-', color='black', linewidth=1)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax.set_xlim(np.min(times), np.max(times))
    for ii, SeqID in enumerate(range(1, 8)):
        plot_timecourses(data_7seq[ii,:,:], times, filter=True, color=colorslist[SeqID - 1], pos_sig=pos_horizontal_bar - 0.005 * ii)  #

    if plot_pearson_corrComplexity:
        pearsonr = compute_corr_comp(data_7seq)
        plot_timecourses(pearsonr, times, chance=0, plot_shaded_vertical=True)
    plt.gca().set_xlabel('Time (ms)', fontsize=14)
    plt.gcf().savefig(op.join(config.fig_path,save_fig_path,fig_name+ suffix + '.svg'))
    plt.gcf().savefig(op.join(config.fig_path, save_fig_path,fig_name + suffix + '.png'), dpi=300)
    plt.close('all')

