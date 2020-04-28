from __future__ import division
from matplotlib import pyplot as plt
import mne
import numpy as np
from mne.stats import permutation_cluster_1samp_test
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ======================================================================================================================
# Coucou Samuel, utilises ca si tu veux raccourcir tes scripts de plot des clusters. Maintenant tu n'as qu'a faire:

# ch_type = 'eeg'
# p_threshold = 0.05
# cluster_stats, _ = run_cluster_permutation_test_1samp(data,ch_type = ch_type)
# plot_clusters(cluster_stats,p_threshold,data,ch_type,T_obs_max = 5.,fname='analysis_name',figname_initial='the_basic_name_for_figure')


# ======================================================================================================================


def run_cluster_permutation_test_1samp(data, ch_type='eeg', nperm=2 ** 12, threshold=None, n_jobs=6, tail=0):
    # If threshold is None, it will choose a t-threshold equivalent to p < 0.05 for the given number of observations
    # (only valid when using an t-statistic).

    # compute connectivity
    connectivity = mne.channels.find_ch_connectivity(data.info, ch_type=ch_type)[0]

    # subset of the data, as array
    if ch_type == 'eeg':
        data_array_chtype = np.array([data.pick_types(meg=False, eeg=True)[c].get_data() for c in range(len(data))])
    elif ch_type == 'mag':
        data_array_chtype = np.array([data.pick_types(meg=ch_type, eeg=False)[c].get_data() for c in range(len(data))])
    elif ch_type == 'grad':
        data_array_chtype = np.array([data.pick_types(meg=ch_type, eeg=False)[c].get_data() for c in range(len(data))])
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

    :return: dictionnary containing all the information

    """
    cluster_info = {'times': data.times * 1e3, 'p_threshold': p_threshold, 'ch_type': ch_type}

    T_obs, clusters, p_values, _ = cluster_stats
    good_cluster_inds = np.where(p_values < p_threshold)[0]
    pos = mne.find_layout(data.info).pos
    times = data.times * 1e3

    if len(good_cluster_inds) > 0:

        # loop over significant clusters
        for i_clu, clu_idx in enumerate(good_cluster_inds):
            # unpack cluster information, get unique indices
            time_inds, space_inds = np.squeeze(clusters[clu_idx])
            ch_inds = np.unique(space_inds)
            time_inds = np.unique(time_inds)
            signals = data_array_chtype[..., ch_inds].mean(axis=-1)
            sig_times = times[time_inds]
            p_value = p_values[clu_idx]

            cluster_info[i_clu] = {'sig_times': sig_times, 'time_inds': time_inds, 'signal': signals,
                                   'channels_cluster': ch_inds, 'p_values': p_value}

        cluster_info['pos'] = pos
        cluster_info['ncluster'] = i_clu + 1
        cluster_info['T_obs'] = T_obs

    return cluster_info


def plot_clusters(cluster_stats, p_threshold, data, data_array_chtype, ch_type, T_obs_max=5., fname='', figname_initial=''):
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

    colors = 'r', 'steelblue'
    linestyles = '-', '--'
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

            ax_signals = divider.append_axes('right', size='300%', pad=1.2)
            for signal, name, col, ls in zip(cinfo['signal'], [fname], colors, linestyles):
                ax_signals.plot(cluster_info['times'], signal * 1e6, color=col, linestyle=ls, label=name)

            ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
            ax_signals.set_xlim([cluster_info['times'][0], cluster_info['times'][-1]])
            ax_signals.set_xlabel('Time [ms]')
            ax_signals.set_ylabel('Amplitude [uV]')

            ymin, ymax = ax_signals.get_ylim()
            ax_signals.fill_betweenx((ymin, ymax), cinfo['sig_times'][0], cinfo['sig_times'][-1], color='orange', alpha=0.3)
            ax_signals.legend(loc='lower right')
            title = 'Cluster #{0} (p < {1:0.3f})'.format(i_clu + 1, cinfo['p_values'])
            ax_signals.set(ylim=[ymin, ymax], title=title)

            fig.tight_layout(pad=0.5, w_pad=0)
            fig.subplots_adjust(bottom=.05)
            fig_name = figname_initial + '_clust_' + str(i_clu + 1) + '.png'
            print('Saving ' + fig_name)
            plt.savefig(fig_name, dpi=300)

    return True
