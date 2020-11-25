from __future__ import division
from matplotlib import pyplot as plt
import mne
import numpy as np
import os.path as op
from mne.stats import permutation_cluster_1samp_test
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import sem
from scipy.signal import savgol_filter
import matplotlib.ticker as ticker
from ABseq_func import evoked_funcs
import copy


def run_cluster_permutation_test_1samp(data, ch_type='eeg', nperm=2 ** 12, threshold=None, n_jobs=6, tail=0):
    # If threshold is None, it will choose a t-threshold equivalent to p < 0.05 for the given number of observations
    # (only valid when using an t-statistic).

    # compute connectivity
    connectivity = mne.channels.find_ch_connectivity(data.info, ch_type=ch_type)[0]

    # subset of the data, as array
    if ch_type == 'eeg':
        data.pick_types(meg=False, eeg=True)
        data_array_chtype = np.array([data[c].get_data() for c in range(len(data))])
    else:
        data.pick_types(meg=ch_type, eeg=False)
        data_array_chtype = np.array([data[c].get_data() for c in range(len(data))])
    data_array_chtype = np.transpose(np.squeeze(data_array_chtype), (0, 2, 1))  # transpose for clustering

    # stat func
    cluster_stats = permutation_cluster_1samp_test(data_array_chtype, threshold=threshold, n_jobs=n_jobs, verbose=True,
                                                   tail=tail, n_permutations=nperm, connectivity=connectivity,
                                                   out_type='indices')
    return cluster_stats, data_array_chtype, ch_type


def extract_info_cluster(cluster_stats, p_threshold, data, data_array_chtype, ch_type):
    """
    This function takes the output of
        cluster_stats = permutation_cluster_1samp_test(...)
        and returns all the useful things for the plots

    :return: dictionnary containing all the information:
    - position of the sensors
    - number of clusters
    - The T-value of the cluster

    """
    cluster_info = {'times': data.times * 1e3, 'p_threshold': p_threshold, 'ch_type': ch_type}

    T_obs, clusters, p_values, _ = cluster_stats
    good_cluster_inds = np.where(p_values < p_threshold)[0]
    pos = mne.find_layout(data.info, ch_type=ch_type).pos
    print("We found %i positions for ch_type %s" % (len(pos), ch_type))
    times = data.times * 1e3

    if len(good_cluster_inds) > 0:

        # loop over significant clusters
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            # unpack cluster information, get unique indices
            time_inds, space_inds = np.squeeze(clusters[clu_idx])
            ch_inds = np.unique(space_inds)
            time_inds = np.unique(time_inds)
            signals = data_array_chtype[..., ch_inds].mean(axis=-1)  # is this correct ??
            # signals = data_array_chtype[..., ch_inds].mean(axis=1)  # is this correct ??
            sig_times = times[time_inds]
            p_value = p_values[clu_idx]

            cluster_info[i_clu] = {'sig_times': sig_times, 'time_inds': time_inds, 'signal': signals,
                                   'channels_cluster': ch_inds, 'p_values': p_value}

        cluster_info['pos'] = pos
        cluster_info['ncluster'] = i_clu + 1
        cluster_info['T_obs'] = T_obs
        if ch_type == 'eeg':
            tmp = data.copy().pick_types(meg=False, eeg=True)
            cluster_info['data_info'] = tmp.info
        else:
            tmp = data.copy().pick_types(meg=ch_type, eeg=False)
            cluster_info['data_info'] = tmp.info

    return cluster_info


def plot_clusters_old(cluster_stats, p_threshold, data, data_array_chtype, ch_type, T_obs_max=5., fname='', figname_initial=''):
    """
    This function plots the clusters

    :param cluster_stats: the output of permutation_cluster_1samp_test
    :param p_threshold: the thresholding p value
    :param data: the dataset with all chtypes (required to get correct channel indices)
    :param data_array_chtype: the data fed to the cluster permutation test
    :param T_obs_max:
    :param fname:
    :param figname_initial:
    :return:
    """

    color = 'r'
    linestyle = '-'
    T_obs_min = -T_obs_max

    cluster_info = extract_info_cluster(cluster_stats, p_threshold, data, data_array_chtype, ch_type)

    if cluster_info is not None:
        for i_clu in range(cluster_info['ncluster']):
            cinfo = cluster_info[i_clu]
            T_obs_map = cluster_info['T_obs'][cinfo['time_inds'], ...].mean(axis=0)
            mask = np.zeros((T_obs_map.shape[0], 1), dtype=bool)
            # mask[cinfo['channel_inds'], :] = True
            mask[cinfo['channels_cluster'], :] = True

            fig, ax_topo = plt.subplots(1, 1, figsize=(7, 2.))
            image, _ = plot_topomap(T_obs_map, cluster_info['pos'], mask=mask, axes=ax_topo,
                                    vmin=T_obs_min, vmax=T_obs_max,
                                    show=False)
            divider = make_axes_locatable(ax_topo)
            # add axes for colorbar
            ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(image, cax=ax_colorbar, format='%0.1f')
            ax_topo.set_xlabel('Averaged t-map\n({:0.1f} - {:0.1f} ms)'.format(
                *cinfo['sig_times'][[0, -1]]))
            ax_topo.set(title=fname)

            # signal average & sem (over subjects)
            ax_signals = divider.append_axes('right', size='300%', pad=1.2)
            # for signal, name, col, ls in zip(cinfo['signal'], [fname], colors, linestyles):
            #     ax_signals.plot(cluster_info['times'], signal * 1e6, color=col, linestyle=ls, label=name)  # why  signal*1e6 ??
            mean = np.mean(cinfo['signal'], axis=0)
            ub = mean + sem(cinfo['signal'], axis=0)
            lb = mean - sem(cinfo['signal'], axis=0)
            ax_signals.fill_between(cluster_info['times'], ub, lb, color=color, alpha=.2)
            ax_signals.plot(cluster_info['times'], mean, color=color, linestyle=linestyle, label=fname)

            # ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
            ax_signals.axhline(0, color='k', linestyle='-', linewidth=0.5)
            ax_signals.set_xlim([cluster_info['times'][0], cluster_info['times'][-1]])
            ax_signals.set_xlabel('Time [ms]')
            ax_signals.set_ylabel('Amplitude')

            ymin, ymax = ax_signals.get_ylim()
            ax_signals.fill_betweenx((ymin, ymax), cinfo['sig_times'][0], cinfo['sig_times'][-1], color='orange', alpha=0.3)
            # ax_signals.legend(loc='lower right')
            title = 'Cluster #{0} (p < {1:0.3f})'.format(i_clu + 1, cinfo['p_values'])
            ax_signals.set(ylim=[ymin, ymax], title=title)

            fig.tight_layout(pad=0.5, w_pad=0)
            fig.subplots_adjust(bottom=.05)
            fig_name = figname_initial + '_clust_' + str(i_clu + 1) + '.png'
            print('Saving ' + fig_name)
            plt.savefig(fig_name, dpi=300)

    return True


def plot_clusters(cluster_info, ch_type, T_obs_max=5., fname='', figname_initial='', filter_smooth=False):
    """
    This function plots the clusters

    :param cluster_info:
    :param good_cluster_inds: indices of the cluster to plot
    :param T_obs_max: colormap limit
    :param fname:
    :param figname_initial:
    :return:
    """
    color = 'r'
    linestyle = '-'
    T_obs_min = -T_obs_max

    for i_clu in range(cluster_info['ncluster']):
        cinfo = cluster_info[i_clu]
        T_obs_map = cluster_info['T_obs'][cinfo['time_inds'], ...].mean(axis=0)
        mask = np.zeros((T_obs_map.shape[0], 1), dtype=bool)
        mask[cinfo['channels_cluster'], :] = True

        fig, ax_topo = plt.subplots(1, 1, figsize=(7, 2.))
        # image, _ = plot_topomap(T_obs_map, cluster_info['data_info'], extrapolate='head',  mask=mask, axes=ax_topo, vmin=T_obs_min, vmax=T_obs_max, show=False)
        image, _ = plot_topomap(T_obs_map, cluster_info['pos'], extrapolate='head', mask=mask,  axes=ax_topo, vmin=T_obs_min, vmax=T_obs_max, show=False)


        divider = make_axes_locatable(ax_topo)
        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar, format='%0.1f')
        ax_topo.set_xlabel('Averaged t-map\n({:0.1f} - {:0.1f} ms)'.format(*cinfo['sig_times'][[0, -1]]))
        ax_topo.set(title=ch_type+': '+fname)

        # signal average & sem (over subjects)
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)
        # for signal, name, col, ls in zip(cinfo['signal'], [fname], colors, linestyles):
        #     ax_signals.plot(cluster_info['times'], signal * 1e6, color=col, linestyle=ls, label=name)
        mean = np.mean(cinfo['signal'], axis=0)
        ub = mean + sem(cinfo['signal'], axis=0)
        lb = mean - sem(cinfo['signal'], axis=0)
        if filter_smooth:
            mean = savgol_filter(mean, 9, 3)
            ub = savgol_filter(ub, 9, 3)
            lb = savgol_filter(lb, 9, 3)
        ax_signals.fill_between(cluster_info['times'], ub, lb, color=color, alpha=.2)
        ax_signals.plot(cluster_info['times'], mean, color=color, linestyle=linestyle, label=fname)

        # ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
        ax_signals.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax_signals.set_xlim([cluster_info['times'][0], cluster_info['times'][-1]])
        ax_signals.set_xlabel('Time [ms]')
        ax_signals.set_ylabel('Amplitude')

        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), cinfo['sig_times'][0], cinfo['sig_times'][-1], color='orange', alpha=0.3)
        # ax_signals.legend(loc='lower right')
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax_signals.get_yaxis().set_major_formatter(fmt)
        ax_signals.get_yaxis().get_offset_text().set_position((-0.07, 0))  # move 'x10-x', does not work with y
        title = 'Cluster #{0} (p < {1:0.3f})'.format(i_clu + 1, cinfo['p_values'])
        ax_signals.set(ylim=[ymin, ymax], title=title)

        fig.tight_layout(pad=0.5, w_pad=0)
        fig.subplots_adjust(bottom=.05)
        fig_name = figname_initial + '_clust_' + str(i_clu + 1) + '.png'
        print('Saving ' + fig_name)
        plt.savefig(fig_name, dpi=300)
    plt.close('all)')

    return True


def plot_clusters_evo(evoked_dict, cinfo, ch_type, i_clu=0, analysis_name='', filter_smooth=False, legend=False, blackfig=False):
    units = dict(eeg='uV', grad='fT/cm', mag='fT')

    if legend:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    if blackfig:
        textcolor = 'white'
        linecolor = 'white'
        ax.set_facecolor((.2, .2, .2))
        fig.patch.set_facecolor((.2, .2, .2))
    else:
        textcolor = 'black'
        linecolor = 'black'
    plt.axvline(0, linestyle='-', color=linecolor, linewidth=2)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color=linecolor, linewidth=1)
    ax.set_xlabel('Time (ms)', color=textcolor)
    condnames = list(evoked_dict.keys())
    if len(condnames) == 2:
        colorslist = ['r', 'b']
    else:
        NUM_COLORS = len(condnames)
        cm = plt.get_cmap('viridis')
        colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
    for ncond, condname in enumerate(condnames):
        data = evoked_dict[condname].copy()
        evoked_funcs.plot_evoked_with_sem_1cond(data, condname[:-1], ch_type, cinfo['channels_cluster'], color=colorslist[ncond], filter=filter_smooth, axis=None)
    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx((ymin, ymax), cinfo['sig_times'][0], cinfo['sig_times'][-1], color='orange', alpha=0.2)
    if legend:
        # plt.legend(loc='best', fontsize=6)
        l = plt.legend(fontsize=7, bbox_to_anchor=(0., 1.25, 1., .08), loc=2, ncol=3, mode="expand", borderaxespad=.8, frameon=False)
        for text in l.get_texts():
            text.set_color(textcolor)
    for key in ('top', 'right'):  # Remove spines
        ax.spines[key].set(visible=False)
    # ax.spines['bottom'].set_position('zero')
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.get_yaxis().set_major_formatter(fmt)
    ax.get_yaxis().get_offset_text().set_position((-0.08, 0))  # move 'x10-x', does not work with y
    ax.set_xlim([-100, 600])
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel(units[ch_type], color=textcolor)
    ax.spines['bottom'].set_color(linecolor)
    ax.spines['left'].set_color(linecolor)
    ax.tick_params(axis='x', colors=textcolor)
    ax.tick_params(axis='y', colors=textcolor)
    plt.title(ch_type + '_' + analysis_name + '_clust_' + str(i_clu + 1), fontsize=10, weight='bold', color=textcolor)
    fig.tight_layout(pad=0.5, w_pad=0)

    return fig


def plot_clusters_evo_bars(evoked_dict, cinfo, ch_type, i_clu=0, analysis_name='', filter_smooth=False, legend=False, blackfig=False):
    units = dict(eeg='uV', grad='fT/cm', mag='fT')

    if legend:
        fig, ax = plt.subplots(1, 1, figsize=(3, 4))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if blackfig:
        textcolor = 'white'
        linecolor = 'white'
        ax.set_facecolor((.2, .2, .2))
        fig.patch.set_facecolor((.2, .2, .2))
    else:
        textcolor = 'black'
        linecolor = 'black'

    plt.axhline(0, linestyle='-', color=linecolor, linewidth=1)
    # for xx in range(3):
    #     plt.axvline(250 * xx, linestyle='--', color=linecolor, linewidth=1)
    # ax.set_xlabel('Time (ms)', color=textcolor)

    ch_inds = cinfo['channels_cluster']  # channel indices from 366 (mag+grad+eeg) ???
    t_inds = cinfo['time_inds']
    condnames = list(evoked_dict.keys())
    if len(condnames) == 2:
        colorslist = ['r', 'b']
    else:
        NUM_COLORS = len(condnames)
        cm = plt.get_cmap('viridis')
        colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
    for ncond, condname in enumerate(condnames):
        data = evoked_dict[condname].copy()
        times = data[0][0].times * 1000
        group_data_seq = []
        for nn in range(len(data)):
            sub_data = data[nn][0].copy()
            if ch_type == 'eeg':
                sub_data = np.array(sub_data.pick_types(meg=False, eeg=True)._data)
            elif ch_type == 'mag':
                sub_data = np.array(sub_data.pick_types(meg='mag', eeg=False)._data)
            elif ch_type == 'grad':
                sub_data = np.array(sub_data.pick_types(meg='grad', eeg=False)._data)
            if np.size(ch_inds) > 1:
                sub_data = sub_data[:, t_inds].mean(axis=1)  # average times indices
                sub_data = sub_data[ch_inds].mean(axis=0)  # average channel indices
                group_data_seq.append(sub_data)
            else:
                sub_data = sub_data[:, t_inds].mean(axis=1)  # average times indices
                group_data_seq.append(sub_data)
        mean = np.mean(group_data_seq, axis=0)
        ub = mean + sem(group_data_seq, axis=0)
        lb = mean - sem(group_data_seq, axis=0)
        plt.bar(ncond, mean, color=colorslist[ncond])
        plt.errorbar(ncond, mean, yerr=sem(group_data_seq, axis=0), ecolor='black', capsize=5)

    ymin, ymax = ax.get_ylim()
    if legend:
        # plt.legend(loc='best', fontsize=6)
        l = plt.legend(fontsize=7, bbox_to_anchor=(0., 1.25, 1., .08), loc=2, ncol=3, mode="expand", borderaxespad=.8, frameon=False)
        for text in l.get_texts():
            text.set_color(textcolor)
    for key in ('top', 'right', 'bottom'):  # Remove spines
        ax.spines[key].set(visible=False)
    # ax.spines['bottom'].set_position('zero')
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.get_yaxis().set_major_formatter(fmt)
    ax.get_yaxis().get_offset_text().set_position((-0.08, 0))  # move 'x10-x', does not work with y
    # ax.set_xlim([-100, 600])
    # ax.set_ylim([ymin, ymax])
    plt.xticks(np.arange(len(condnames)), condnames, rotation=45)
    ax.set_ylabel(units[ch_type], color=textcolor)
    ax.spines['bottom'].set_color(linecolor)
    ax.spines['left'].set_color(linecolor)
    ax.tick_params(axis='x', colors=textcolor)
    ax.tick_params(axis='y', colors=textcolor)
    plt.title(ch_type + '_' + analysis_name + '_clust_' + str(i_clu + 1) + '_[%d-%dms]' % (times[t_inds[0]], times[t_inds[-1]]), fontsize=9, color=textcolor)
    fig.tight_layout(pad=0.5, w_pad=0)

    return fig




