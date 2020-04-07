from __future__ import division
import os.path as op
from matplotlib import pyplot as plt
import config
from ABseq_func import *
import mne
import numpy as np
from mne.stats import linear_regression, fdr_correction, bonferroni_correction, permutation_cluster_1samp_test
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from importlib import reload

# ======================== REGRESSORS TO INCLUDE AND RESULTS FOLDER ========================= #
Only_group_part = True
analysis_name = 'test2'
regress_path = op.join(config.result_path, 'linear_models', analysis_name)
names = ["ChunkBeginning", "ChunkEnd", "ChunkDepth", "ChunkSize", "Identity"]

if Only_group_part == False:
    # ========================= RUN LINEAR REGRESSION FOR EACH SUBJECT ========================== #
    for subject in config.subjects_list:
        linear_reg_funcs.run_linear_regression_v2(analysis_name, names, subject, cleaned=True)

    # Create evoked with each (unique) level of the regressor
    for subject in config.subjects_list:
        evoked_funcs.create_evoked_for_regression_factors(names, subject, cleaned=True)

    # =========== LOAD INDIVIDUAL REGRESSION RESULTS AND SAVE THEM AS GROUP FIF FILES =========== #
    # ============================= (necessary only the first time) ============================= #
    names = ["Intercept"] + names  # intercept was added when running the regression
    # Load data from all subjects
    tmpdat = dict()
    for name in names:
        tmpdat[name] = evoked_funcs.load_evoked('all', filter_name=name, root_path=regress_path)

    # Store as epo objects
    for name in names:
        dat = tmpdat[name][name[0:-3]]  # the dict key has 3 less characters than the original... because I did not respect MNE naming conventions ??
        exec(name + "_epo = mne.EpochsArray(np.asarray([dat[i][0].data for i in range(len(dat))]), dat[0][0].info, tmin=-0.1)")

    # Save group fif files
    out_path = op.join(regress_path, 'group')
    utils.create_folder(out_path)
    for name in names:
        exec(name + "_epo.save(op.join(out_path, '" + name + "_epo.fif'), overwrite=True)")

# ============================ (RE)LOAD GROUP REGRESSION RESULTS =========================== #
path = op.join(regress_path, 'group')
if names[0] != 'Intercept':
    names = ["Intercept"] + names  # intercept was added when running the regression

betas = dict()
for name in names:
    exec(name + "_epo = mne.read_epochs(op.join(path, '" + name + "_epo.fif'))")
    betas[name] = globals()[name + '_epo']

# ================= PLOT THE HEATMAPS OF THE GROUP-AVERAGED BETAS / CHANNEL ================ #
fig_path = op.join(config.fig_path, 'Linear_regressions', analysis_name)
utils.create_folder(fig_path)

# Loop over the 3 ch_types
plt.close('all')
ch_types = ['eeg', 'grad', 'mag']
for ch_type in ch_types:
    fig, axes = plt.subplots(1, len(betas.keys()), figsize=(len(betas.keys())*4, 6), sharex=False, sharey=False, constrained_layout=True)
    fig.suptitle(ch_type, fontsize=12, weight='bold')
    ax = axes.ravel()[::1]
    # Loop over the different betas
    for x, regressor_name in enumerate(betas.keys()):
        # ---- Data
        evokeds = betas[regressor_name].average()
        if ch_type == 'eeg':
            betadata = evokeds.copy().pick_types(eeg=True, meg=False).data
        elif ch_type == 'mag':
            betadata = evokeds.copy().pick_types(eeg=False, meg='mag').data
        elif ch_type == 'grad':
            betadata = evokeds.copy().pick_types(eeg=False, meg='grad').data
        minT = min(evokeds.times) * 1000
        maxT = max(evokeds.times) * 1000
        # ---- Plot
        im = ax[x].imshow(betadata, origin='upper', extent=[minT, maxT, betadata.shape[0],0], aspect='auto', cmap='viridis') #cmap='RdBu_r'
        ax[x].axvline(0, linestyle='-', color='black', linewidth=1)
        for xx in range(3):
            ax[x].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        ax[x].set_xlabel('Time (ms)')
        ax[x].set_ylabel('Channels')
        ax[x].set_title(regressor_name, loc='center', weight='normal')
        fig.colorbar(im, ax = ax[x], shrink=1, location='bottom')
    fig_name = fig_path + op.sep + ('betas_' + ch_type + '.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, dpi=300)
    plt.close(fig)

# =========================== PLOT THE BUTTERFLY OF THE REGRESSORS ========================== #
ylim_eeg = 0.3
ylim_mag = 20
ylim_grad = 4
# Butterfly plots for violations (one graph per sequence) - in EEG/MAG/GRAD
ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
ts_args = dict(gfp=True, time_unit='s', ylim=ylim)
topomap_args = dict(time_unit='s')
times = 'peaks'
for x, regressor_name in enumerate(betas.keys()):
    evokeds = betas[regressor_name].average()
    # EEG
    fig = evokeds.plot_joint(ts_args=ts_args, title='EEG_' + regressor_name ,
                                   topomap_args=topomap_args, picks='eeg', times=times, show=False)
    fig_name = fig_path + op.sep + ('EEG_' + regressor_name + '.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name)
    plt.close(fig)
    # MAG
    fig = evokeds.plot_joint(ts_args=ts_args, title='MAG_' + regressor_name,
                                   topomap_args=topomap_args, picks='mag', times=times, show=False)
    fig_name = fig_path + op.sep + ('MAG_' + regressor_name+'.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name)
    plt.close(fig)
    # #GRAD
    fig = evokeds.plot_joint(ts_args=ts_args, title='GRAD_' + regressor_name,
                                   topomap_args=topomap_args, picks='grad', times=times, show=False)
    fig_name = fig_path + op.sep + ('GRAD_' + regressor_name + '.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name)
    plt.close(fig)

# ======================= RUN CLUSTER STATISTICS ====================== #
nperm = 5000  # number of permutations
p1 = 0.05
p2 = 0.01  # second p threshold, step-down-in-jumps test?
# To perform a step-down-in-jumps test, pass a p-value for clusters to exclude from each successive iteration. Default is zero, perform no
# step-down test (since no clusters will be smaller than this value). Setting this to a reasonable value, e.g. 0.05, can increase sensitivity
# but costs computation time.
tmin = 0.0  # timewindow to test (crop data)
tmax = 0.600 # timewindow to test (crop data)
fig_path = op.join(config.fig_path, 'Linear_regressions', analysis_name)
utils.create_folder(fig_path)

# Loop over the 3 ch_types
ch_types = ['eeg', 'grad', 'mag']
for ch_type in ch_types:
    # Loop over the different regressors
    for x, regressor_name in enumerate(betas.keys()):
        data_condition = betas[regressor_name].copy()
        fname = regressor_name
        # data_condition.apply_baseline(baseline=(-0.100, 0.0))  # baseline ?? (probably not at this step - betas)
        data_condition.crop(tmin=tmin, tmax=tmax)  # crop
        connectivity = mne.channels.find_ch_connectivity(data_condition.info, ch_type=ch_type)[0]
        if ch_type == 'eeg':
            data = np.array([data_condition.pick_types(meg=False, eeg=True)[c].get_data() for c in range(len(data_condition))])
        elif ch_type == 'mag':
            data = np.array([data_condition.pick_types(meg=ch_type, eeg=False)[c].get_data() for c in range(len(data_condition))])
        elif ch_type == 'grad':
            data = np.array([data_condition.pick_types(meg=ch_type, eeg=False)[c].get_data() for c in range(len(data_condition))])
        data = np.transpose(np.squeeze(data), (0, 2, 1))  # transpose for clustering


        cluster_stats = permutation_cluster_1samp_test(data, threshold=None, n_jobs=6, verbose=True, tail=0, n_permutations=nperm,
                                                   connectivity=connectivity, out_type='indices', check_disjoint=True, step_down_p=p1)


        T_obs, clusters, p_values, _ = cluster_stats
        good_cluster_inds = np.where(p_values < p2)[0]



        print("Good clusters: %s" % good_cluster_inds)

        # PLOT CLUSTERS
        plt.close('all')
        # set_matplotlib_defaults()
        times = data_condition.times * 1e3
        colors = 'r', 'steelblue'
        linestyles = '-', '--'
        pos = mne.find_layout(data_condition.info).pos
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
                *sig_times[[0, -1]] ))
            # ax_topo.annotate(chr(65 + 2 * i_clu), (0.1, 1.1), **annot_kwargs)
            # add new axis for time courses and plot time courses
            ax_signals = divider.append_axes('right', size='300%', pad=1.2)
            for signal, name, col, ls in zip(signals, [fname], colors, linestyles):
                ax_signals.plot(times, signal * 1e6, color=col, linestyle=ls, label=name)
            # add information
            ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
            ax_signals.set_xlim([times[0], times[-1]])
            ax_signals.set_xlabel('Time [ms]')
            ax_signals.set_ylabel('Amplitude [uV]')
            # plot significant time range
            ymin, ymax = ax_signals.get_ylim()
            ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1], color='orange', alpha=0.3)
            ax_signals.legend(loc='lower right')
            title = 'Cluster #{0} (p < {1:0.3f})'.format(i_clu + 1, p_values[clu_idx])
            ax_signals.set(ylim=[ymin, ymax], title=title)
            # ax_signals.annotate(chr(65 + 2 * i_clu + 1), (-0.125, 1.1), **annot_kwargs)
            # clean up viz
            fig.tight_layout(pad=0.5, w_pad=0)
            fig.subplots_adjust(bottom=.05)
            fig_name = fig_path + op.sep + 'stats_' + ch_type + '_' + regressor_name + '_clust_' + str(i_clu+1) + '.png'
            print('Saving ' + fig_name)
            plt.savefig(fig_name, dpi=300)

        # =========================================================== #
        # ==========  cluster evoked data plot
        # =========================================================== #

        # ------------------ LOAD THE EVOKED FOR THE CURRENT REGRESSOR ------------ #
        evoked_reg = evoked_funcs.load_evoked(subject='all', filter_name=regressor_name, filter_not=None, cleaned=True)

        for i_clu, clu_idx in enumerate(good_cluster_inds):

            # ----------------- SELECT CHANNELS OF INTEREST ----------------- #
            time_inds, space_inds = np.squeeze(clusters[clu_idx])
            ch_inds = np.unique(space_inds)  # list of channels we want to average and plot (!? after pick_types or from all ?!)

            # ----------------- PLOT ----------------- #
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            fig.suptitle(ch_type, fontsize=12, weight='bold')
            plt.axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(3):
                plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            ax.set_xlabel('Time (ms)')
            condnames = list(evoked_reg.keys())
            for condname in condnames:
                data = evoked_reg[condname].copy()
                evoked_funcs.plot_evoked_with_sem_1cond(data, condname, ch_inds, color=None, filter=True, axis=None)
            plt.legend(loc='upper right', fontsize=9)
            ax.set_xlim([-100, 750])
            plt.title(ch_type + '_' + regressor_name + '_clust_' + str(i_clu+1))
            fig_name = fig_path + op.sep + 'stats_' + ch_type + '_' + regressor_name + '_clust_' + str(i_clu+1) + '_evo.png'
            print('Saving ' + fig_name)
            plt.savefig(fig_name, dpi=300)