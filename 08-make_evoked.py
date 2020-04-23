import config
import os
import os.path as op
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import sem
from ABseq_func import *
from mne.parallel import parallel_func
from importlib import reload

# # make less parallel runs to limit memory usage
# N_JOBS = max(config.N_JOBS // 4, 1)
N_JOBS = config.N_JOBS


# ----------------------------------------------------------- #
# ---------- COMPUTE AND SAVE EVOKED OF INTEREST ------------ #
# ----------------------------------------------------------- #

parallel, run_func, _ = parallel_func(evoked_funcs.create_evoked, n_jobs=N_JOBS)
parallel(run_func(subject, cleaned=True) for subject in config.subjects_list)
parallel(run_func(subject, cleaned=False) for subject in config.subjects_list)

# ----------------------------------------------------------- #
# -------- GENERATE EVOKED GROUP FIGURES (GFP...) ----------- #
# ----------------------------------------------------------- #
script_group_avg_and_plot_gfp()
script_generate_heatmap_gfp_figures()


def script_group_avg_and_plot_gfp():

    # ----------------------------------------------------------- #
    # ------------------ LOAD THE EVOKED OF INTEREST ------------ #
    # ----------------------------------------------------------- #

    # all sequences pooled together
    evoked_all_standard = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_all', filter_not=None)
    evoked_all_viol = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_all', filter_not=None)
    evoked_full_seq_all_standard = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_all', filter_not=None)

    # one key per sequence ID
    evoked_standard_seq = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_seq', filter_not='pos')  #
    evoked_viol_seq = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_seq', filter_not='pos')  #
    evoked_full_seq_standard_seq = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq', filter_not=None)
    evoked_full_seq_teststandard_seq = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_teststandard_seq', filter_not=None)
    evoked_full_seq_habituation_seq = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_habituation_seq', filter_not=None)

    evoked_viol_seq1_pos = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq1_pos', filter_not=None)  #
    evoked_viol_seq2_pos = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq2_pos', filter_not=None)  #
    evoked_viol_seq3_pos = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq3_pos', filter_not=None)  #
    evoked_viol_seq4_pos = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq4_pos', filter_not=None)  #
    evoked_viol_seq5_pos = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq5_pos', filter_not=None)  #
    evoked_viol_seq6_pos = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq6_pos', filter_not=None)  #
    evoked_viol_seq7_pos = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq7_pos', filter_not=None)  #

    evoked_standard_seq1 = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq1', filter_not='pos')  #
    evoked_standard_seq2 = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq2', filter_not='pos')  #
    evoked_standard_seq3 = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq3', filter_not='pos')  #
    evoked_standard_seq4 = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq4', filter_not='pos')  #
    evoked_standard_seq5 = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq5', filter_not='pos')  #
    evoked_standard_seq6 = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq6', filter_not='pos')  #
    evoked_standard_seq7 = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq7', filter_not='pos')  #


    def butterfly_grand_avg():

        evoked_funcs.plot_butterfly_items_allsubj(evoked_standard_seq, violation_or_not=0)
        evoked_funcs.plot_butterfly_items_allsubj(evoked_viol_seq, violation_or_not=1)


    # ----------------------------------------------------------- #
    # ------------------ EXTRACT GROUP GFP DATA ----------------- #
    # ----------------------------------------------------------- #


    def plot_gfp_super_cool(evoked_list, full_sequence=True, ch_type='eeg', save_path='', fig_name='', labels=None):
        if full_sequence:
            fig, ax = plt.subplots(1, 1, figsize=(18, 4))
            plt.axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(16):
                plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            plt.axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(3):
                plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        fig.suptitle('GFP (%s)' % ch_type + ', N subjects=' + str(len(evoked_list[next(iter(evoked_list))])), fontsize=12)
        fig_name_save = save_path + op.sep + 'GFP_' + fig_name + '_' + ch_type + '.png'
        plot_evoked_list(evoked_list, ch_type=ch_type, labels=labels)
        plt.legend(loc='upper right', fontsize=9)

        if full_sequence:
            ax.set_xlim([-500, 4250])
            ax.set_ylim([0, 8e-25])
        else:
            ax.set_xlim([-100, 750])
        fig.savefig(fig_name_save, bbox_inches='tight', dpi=300)
        plt.close('all')


    def plot_evoked_list(evoked_list, ch_type='eeg', labels=None):
        for ll, cond in enumerate(evoked_list.keys()):
            NUM_COLORS = len(evoked_list.keys())
            if NUM_COLORS > 1:
                cm = plt.get_cmap('viridis')
                colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
            else:
                colorslist = 'k'
            gfp_cond, times = GFP_funcs.gfp_evoked(evoked_list[cond])
            GFP_funcs.plot_GFP_with_sem(gfp_cond[ch_type], times * 1000, color_mean=colorslist[ll], label=labels[ll],
                                        filter=True)

        return plt.gcf()


    for ch_type in ['eeg', 'mag', 'grad']:
        plot_gfp_super_cool(evoked_viol_seq, full_sequence=False, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP', fig_name='viol',
                            labels=['Seq_%i' % r for r in range(1, 8)])
        plot_gfp_super_cool(evoked_standard_seq, full_sequence=False, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='standard', labels=['Seq_%i' % r for r in range(1, 8)])
        plot_gfp_super_cool(evoked_full_seq_standard_seq, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard', labels=['Seq_%i' % r for r in range(1, 8)])

        plot_gfp_super_cool(evoked_all_standard, full_sequence=False, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='all_standard', labels=['all_sequences'])
        plot_gfp_super_cool(evoked_all_viol, full_sequence=False, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='all_viol', labels=['all_sequences'])
        plot_gfp_super_cool(evoked_full_seq_all_standard, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_all_standard', labels=['all_sequences'])

    for ch_type in ['eeg', 'mag', 'grad']:
        plot_gfp_super_cool(evoked_viol_seq1_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq1_pos', labels=list(evoked_viol_seq1_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq2_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq2_pos', labels=list(evoked_viol_seq2_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq3_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq3_pos', labels=list(evoked_viol_seq3_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq4_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq4_pos', labels=list(evoked_viol_seq4_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq5_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq5_pos', labels=list(evoked_viol_seq5_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq6_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq6_pos', labels=list(evoked_viol_seq6_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq7_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq7_pos', labels=list(evoked_viol_seq7_pos.keys()))

    for ch_type in ['eeg', 'mag', 'grad']:
        plot_gfp_super_cool(evoked_standard_seq1, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq1', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq2, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq2', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq3, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq3', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq4, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq4', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq5, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq5', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq6, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq6', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq7, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq7', labels=[''])


def script_generate_heatmap_gfp_figures():

    # ----------------------------------------------------------- #
    # ---------------- GROUP GFP HEATMAP FIGURES ---------------- #
    # ----------------------------------------------------------- #

    filterdata = True
    # Load required evoked data
    evoked_full_seq_teststandard_seq = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_teststandard_seq', filter_not=None)
    evoked_full_seq_habituation_seq = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_habituation_seq', filter_not=None)
    evoked_viol_seq_pos = dict()
    for seqID in range(1, 8):
        evoked_viol_seq_pos['seq' + str(seqID)] = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq' + str(seqID) + '_pos', filter_not=None)

    for ch_type in ['mag', 'grad', 'eeg']:
        # Compute & store average gfp vectors in a data_to_plot dict (gfp per subject, then group average)
        data_to_plot = {}
        data_to_plot['hab'] = {}
        data_to_plot['teststand'] = {}
        data_to_plot['violpos1'] = {}
        data_to_plot['violpos2'] = {}
        data_to_plot['violpos3'] = {}
        data_to_plot['violpos4'] = {}
        for seqID in range(1, 8):
            # Habituation trials
            gfp_all_subs, times = GFP_funcs.gfp_evoked(evoked_full_seq_habituation_seq['full_seq_habituation_seq'+str(seqID)+'-'])
            data_to_plot['hab']['seq'+str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
            if filterdata:
                data_to_plot['hab']['seq'+str(seqID)] = savgol_filter(np.mean(gfp_all_subs[ch_type], axis=0), window_length=13, polyorder=3)
            else:
                data_to_plot['hab']['seq'+str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
            # TestStandard trials
            gfp_all_subs, times = GFP_funcs.gfp_evoked(evoked_full_seq_teststandard_seq['full_seq_teststandard_seq'+str(seqID)+'-'])
            if filterdata:
                data_to_plot['teststand']['seq'+str(seqID)] = savgol_filter(np.mean(gfp_all_subs[ch_type], axis=0), window_length=13, polyorder=3)
            else:
                data_to_plot['teststand']['seq'+str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
            # Violation trials
            seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqID)
            for n, devpos in enumerate(violation_positions):
                gfp_all_subs, times = GFP_funcs.gfp_evoked(evoked_viol_seq_pos['seq' + str(seqID)]['full_seq_viol_seq'+str(seqID)+'_pos' + str(devpos) + '-'])
                data_to_plot['violpos'+str(n+1)]['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
                if filterdata:
                    data_to_plot['violpos' + str(n + 1)]['seq' + str(seqID)] = savgol_filter(np.mean(gfp_all_subs[ch_type], axis=0), window_length=13, polyorder=3)
                else:
                    data_to_plot['violpos' + str(n + 1)]['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)

        # Draw & save heatmap figure
        evoked_funcs.allsequences_heatmap_figure(data_to_plot, times, cmap_style='unilateral', fig_title='GFP ' + ch_type,
                                                 file_name=op.join(config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP', 'GFP_full_seq_allconds_heatmap_' + ch_type + '.png'))


def various():
    # TEST EEG GROUP STATS
    from scipy import stats
    from mne.stats import permutation_cluster_1samp_test
    from mne.viz import plot_topomap
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    evs = list()
    fname ='viol_seq7'
    for subj in config.subjects_list:
        path_evo = op.join(config.meg_dir, subj, 'evoked')
        ev_name = path_evo + op.sep + fname + '-ave.fif'
        ev = mne.read_evokeds(ev_name)
        ev = ev[0]
        ev.pick_types(meg=False, eeg=True)
        ev.apply_baseline((-0.100, 0.0))
        evs.append(ev)
    data = np.array([c.data for c in evs])

    connectivity = None
    tail = 0.  # for two sided test
    # set cluster threshold
    # p_thresh = 0.01 / (1 + (tail == 0))
    p_thresh = 0.01
    n_samples = len(data)
    threshold = -stats.t.ppf(p_thresh, n_samples - 1)
    if np.sign(tail) < 0:
        threshold = -threshold
    # Make a triangulation between EEG channels locations to
    # use as connectivity for cluster level stat
    connectivity = mne.channels.find_ch_connectivity(ev.info, 'eeg')[0]
    data = np.transpose(data, (0, 2, 1))  # transpose for clustering
    cluster_stats = permutation_cluster_1samp_test(
        data, threshold=threshold, n_jobs=2, verbose=True, tail=0,
        connectivity=connectivity, out_type='indices',
        check_disjoint=True, step_down_p=0.05)
    T_obs, clusters, p_values, _ = cluster_stats
    good_cluster_inds = np.where(p_values < 0.05)[0]
    print("Good clusters: %s" % good_cluster_inds)

    # Visualize the spatio - temporal clusters
    plt.close('all')
    # set_matplotlib_defaults()
    times = ev.times * 1e3
    colors = 'r', 'steelblue'
    linestyles = '-', '--'
    pos = mne.find_layout(ev.info).pos
    T_obs_max = 5.
    T_obs_min = -T_obs_max

    # loop over significant clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):

        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for T0 stat
        T_obs_map = T_obs[time_inds, ...].mean(axis=0)

        # get signals at significant sensors
        signals = data[..., ch_inds].mean(axis=-1)
        sig_times = times[time_inds]

        # create spatial mask
        mask = np.zeros((T_obs_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(7, 2.))

        # plot average test statistic and mark significant sensors
        image, _ = plot_topomap(T_obs_map, pos, mask=mask, axes=ax_topo,
                                vmin=T_obs_min, vmax=T_obs_max,
                                show=False)

        # advanced matplotlib for showing image with figure and colorbar
        # in one plot
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar, format='%0.1f')
        ax_topo.set_xlabel('Averaged t-map\n({:0.1f} - {:0.1f} ms)'.format(
            *sig_times[[0, -1]]
        ))
        # ax_topo.annotate(chr(65 + 2 * i_clu), (0.1, 1.1), **annot_kwargs)

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)
        for signal, name, col, ls in zip(signals, [fname], colors,
                                         linestyles):
            ax_signals.plot(times, signal * 1e6, color=col,
                            linestyle=ls, label=name)

        # add information
        ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
        ax_signals.set_xlim([times[0], times[-1]])
        ax_signals.set_xlabel('Time [ms]')
        ax_signals.set_ylabel('Amplitude [uV]')

        # plot significant time range
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                 color='orange', alpha=0.3)
        ax_signals.legend(loc='lower right')
        title = 'Cluster #{0} (p < {1:0.3f})'.format(i_clu + 1, p_values[clu_idx])
        ax_signals.set(ylim=[ymin, ymax], title=title)
        # ax_signals.annotate(chr(65 + 2 * i_clu + 1), (-0.125, 1.1), **annot_kwargs)

        # clean up viz
        fig.tight_layout(pad=0.5, w_pad=0)
        fig.subplots_adjust(bottom=.05)
        # plt.savefig(op.join('..', 'figures',
        #                     'spatiotemporal_stats_cluster_highpass-%sHz-%02d.pdf'
        #                     % (l_freq, i_clu)))
        plt.show()



    contrast = mne.combine_evoked(contrasts, 'equal')
    data_evoked = evoked_all_standard['all_standard-']
    all_evokeds=[]
    for n in range(len(data_evoked)):
        all_evokeds.append(data_evoked[n][0])
    # all_evokeds = mne.combine_evoked(all_evokeds, 'equal')
    grand_average = mne.grand_average(all_evokeds)
    grand_average.plot(spatial_colors=True, gfp=True)
    times = np.arange(-0.050, 0.251, 0.025)
    grand_average.plot_joint()
    grand_average.plot_topomap(times=times, ch_type='mag', time_unit='s')
    grand_average.plot_topomap(times=times, ch_type='eeg', time_unit='s')

    mne.stats.permutation_cluster_test(all_evokeds)

