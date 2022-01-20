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
    plt.imshow(mean_data[np.newaxis, :], aspect="auto", cmap="PRGn", extent=extent, vmin=ylims[0], vmax=ylims[1])
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.colorbar(label='Pearsor r')
    if fig_name is not None:
        plt.gcf().savefig(fig_name, dpi=300, bbox_inches='tight')


# ----------------------------------------------------------------------------------------------------------------------
def plot_timecourses(data_seq_subjs, times, filter=False, fig_name='', color='b', chance=0.5, pos_sig=None, plot_shaded_vertical=False, xlims=None,logger=None):
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
                plt.gca().fill_between([clu_times[0], clu_times[-1]], ylims[1], ylims[0], color='black', alpha=.08, linewidth=0.0)
                sp = "The p-value of the cluster number %i" % (i_clu) + " is {:.5f}".format(cluster_pv[clu_idx])
                st = "The T-value of the cluster number %i" % (i_clu) + " is {:.5f}".format(t_obs [clu_idx])
                print(sp)
                print(st)
                if logger is not None:
                    logger.info(sp)
                    logger.info(st)
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
                          pos_horizontal_bar=0.47, plot_pearson_corrComplexity=True, chance=0, xlims=None, ymin=None, ylabel=None, filter=False,logger = None):
    """
    param data_7seq: data in the shape of 7 X n_subjects X n_times
    param times: the times for the plot
    """
    if xlims == None:
        xlims = [times[0], times[-1]]

    NUM_COLORS = 7
    # cm = plt.get_cmap('viridis')
    # colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
    # OR USE PREDEFINED COLORS:
    colorslist = config.seqcolors
    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(10 * 0.5, 7 * 0.5))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    # plt.axhline(0.5, linestyle='-', color='black', linewidth=1)  # ligne horizontale à 0.5 pas applicable pour valeurs GFP à 1e-25!
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)

    if logger is not None:
        logger.debug("----- Analysis %s ------"%fig_name)

    for ii, SeqID in enumerate(range(1, 8)):
        plot_timecourses(data_7seq[ii, :, :], times, filter=filter, color=colorslist[SeqID - 1], pos_sig=pos_horizontal_bar - 0.005 * ii, chance=chance,logger=logger)  #

    if plot_pearson_corrComplexity:
        pearsonr = compute_corr_comp(data_7seq)
        plot_timecourses(pearsonr, times, chance=0, plot_shaded_vertical=True, xlims=xlims,logger=logger)

    # Set limits
    ax.set_xlim(xlims)
    if ymin is not None:
        ax.set_ylim(ymin=ymin)

    # # Remove some spines?
    for key in ('top', 'right'):
        ax.spines[key].set(visible=False)

    # Add ylabel and format x10^...
    if ylabel == 'GFP':
        ax.set_ylabel(ylabel, fontsize=14)
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.get_yaxis().set_major_formatter(fmt)
    plt.gca().set_xlabel('Time (ms)', fontsize=14)

    utils.create_folder(op.join(config.fig_path, save_fig_path))
    plt.gcf().savefig(op.join(config.fig_path, save_fig_path, fig_name + suffix + '.svg'), bbox_inches='tight')
    plt.gcf().savefig(op.join(config.fig_path, save_fig_path, fig_name + suffix + '.png'), dpi=300, bbox_inches='tight')
    plt.close('all')

# ----------------------------------------------------------------------------------------------------------------------
def load_epochs_explained_signal_and_residuals_and_plot(
        regressors_names=['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'],
        filter_name='Hab', suffix='--remapped_gtmbaselined_clean-epo.fif', compute=True):
    """
    The goal of this function is to see how much signal coming from the epochs is modeled by the intercept, the explained signals coming from the regressors and from the residuals
    """

    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)

    results_path = op.join(config.result_path, 'linear_models')
    epochs_all = []
    explained_signal_all = []
    residuals_all = []
    intercept_all = []

    to_append_to_results_path = ''
    for name in regressors_names:
        to_append_to_results_path += '_' + name
    results_path = op.join(results_path, filter_name, to_append_to_results_path[1:])

    if compute:
        for subject in config.subjects_list:
            print(subject)
            subj_path = op.join(results_path, subject)
            epochs = mne.read_epochs(op.join(subj_path, 'epochs' + suffix))
            intercept = mne.read_epochs(op.join(subj_path, 'intercept' + suffix))
            explained_signal = mne.read_epochs(op.join(subj_path, 'explained_signal' + suffix))
            residuals = mne.read_epochs(op.join(subj_path, 'residuals' + suffix))
            epochs_all.append(epochs.average()._data)
            explained_signal_all.append(explained_signal.average()._data)
            residuals_all.append(residuals.average()._data)
            intercept_all.append(intercept.average()._data)

        epo = mne.EpochsArray(np.asarray(epochs_all), tmin=epochs.tmin, info=epochs.info)
        expl = mne.EpochsArray(np.asarray(explained_signal_all), tmin=epochs.tmin, info=epochs.info)
        resid = mne.EpochsArray(np.asarray(residuals_all), tmin=epochs.tmin, info=epochs.info)
        interc = mne.EpochsArray(np.asarray(intercept_all), tmin=epochs.tmin, info=epochs.info)
        epo.save(op.join(results_path, 'epochs_allsubjects-epo.fif'), overwrite=True)
        expl.save(op.join(results_path, 'explained_signal_allsubjects-epo.fif'), overwrite=True)
        resid.save(op.join(results_path, 'residuals_allsubjects-epo.fif'), overwrite=True)
        interc.save(op.join(results_path, 'intercept_allsubjects-epo.fif'), overwrite=True)
    else:
        epo = mne.read_epochs(op.join(results_path, 'epochs_allsubjects-epo.fif'))
        expl = mne.read_epochs(op.join(results_path, 'explained_signal_allsubjects-epo.fif'))
        resid = mne.read_epochs(op.join(results_path, 'residuals_allsubjects-epo.fif'))
        interc = mne.read_epochs(op.join(results_path, 'intercept_allsubjects-epo.fif'))

    figure_path = op.join(config.result_path, 'linear_models', 'plot_joint_regressions/') + filter_name + '_'

    print("==== NOW PLOTTING ===")

    fig = epo.crop(tmax=0.35).average().plot(ylim=dict(mag=[-100, 100]), time_unit='ms')
    ax = fig.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title("")
    fig.savefig((figure_path + 'epochs_allsubjects.svg'))
    plt.close(fig)

    fig = interc.crop(tmax=0.35).average().plot(ylim=dict(mag=[-100, 100]), time_unit='ms')
    fig.savefig((figure_path + 'intercept_allsubjects.svg'))

    plt.close(fig)

    if filter_name != 'Viol':
        fig = expl.crop(tmax=0.35).average().plot(time_unit='ms', ylim=dict(mag=[-2, 2]))
    else:
        fig = expl.crop(tmax=0.35).average().plot(time_unit='ms', ylim=dict(mag=[-10, 10]))

    fig.savefig((figure_path + 'explained_signal_allsubjects.svg'))
    plt.close(fig)

    fig = resid.crop(tmax=0.35).average().plot(time_unit='ms', ylim=dict(mag=[-0.15, 0.15]))
    fig.savefig((figure_path + 'residuals_allsubjects.svg'))
    plt.close(fig)
