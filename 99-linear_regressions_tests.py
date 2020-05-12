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
Only_group_part = True  # False means running the regressions at the individual level, True to just analyze and plot the group results
analysis_name = 'RepeatAlter&Viol'
regress_path = op.join(config.result_path, 'linear_models', analysis_name)
names = ["RepeatAlter", "ViolationOrNot"]

if Only_group_part == False:
    # ========================= RUN LINEAR REGRESSION FOR EACH SUBJECT ========================== #
    for subject in config.subjects_list:
        linear_reg_funcs.run_linear_regression_v2(analysis_name, names, subject, cleaned=True)

    # Create evoked with each (unique) level of the regressor / involves loading data again
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
nperm = 10000  # number of permutations
threshold = None  # If threshold is None, t-threshold equivalent to p < 0.05 (if t-statistic)
p_threshold = 0.05
tmin = 0.000  # timewindow to test (crop data)
tmax = 0.500  # timewindow to test (crop data)
fig_path = op.join(config.fig_path, 'Linear_regressions', analysis_name)
utils.create_folder(fig_path)

# Loop over the 3 ch_types
ch_types = ['eeg', 'grad', 'mag']
for ch_type in ch_types:
    # Loop over the different regressors
    for x, regressor_name in enumerate(betas.keys()):
        data_condition = betas[regressor_name].copy()
        # fname = regressor_name
        # data_condition.apply_baseline(baseline=(-0.100, 0.0))  # baseline ?? (probably not at this step - betas)
        data_condition.crop(tmin=tmin, tmax=tmax)  # crop

        # RUN THE STATS
        cluster_stats, data_array_chtype, _ = stats_funcs.run_cluster_permutation_test_1samp(data_condition, ch_type=ch_type, nperm=nperm, threshold=threshold, n_jobs=6, tail=0)
        cluster_info = stats_funcs.extract_info_cluster(cluster_stats, p_threshold, data_condition, data_array_chtype, ch_type)

        # Significant clusters
        T_obs, clusters, p_values, _ = cluster_stats
        good_cluster_inds = np.where(p_values < p_threshold)[0]
        print("Good clusters: %s" % good_cluster_inds)

        # PLOT CLUSTERS
        if len(good_cluster_inds) > 0:
            figname_initial = fig_path + op.sep + 'stats_' + ch_type + '_' + regressor_name
            # stats_funcs.plot_clusters(cluster_stats, p_threshold, data_stat, data_array_chtype, ch_type, T_obs_max=5., fname=analysis_name, figname_initial=figname_initial)
            stats_funcs.plot_clusters(cluster_info, ch_type, T_obs_max=5., fname=analysis_name+'_'+regressor_name, figname_initial=figname_initial)

        # =========================================================== #
        # ==========  cluster evoked data plot
        # =========================================================== #

        if len(good_cluster_inds) > 0:
            # ------------------ LOAD THE EVOKED FOR THE CURRENT REGRESSOR ------------ #
            evoked_reg = evoked_funcs.load_evoked(subject='all', filter_name=regressor_name, filter_not='seqID', cleaned=True)

            # T_obs, clusters, p_values, _ = cluster_stats
            # good_cluster_inds = np.where(p_values < p_threshold)[0]
            # print("Good clusters: %s" % good_cluster_inds)

            for i_clu, clu_idx in enumerate(good_cluster_inds):

                cinfo = cluster_info[i_clu]

                # ----------------- SELECT CHANNELS OF INTEREST ----------------- #
                time_inds, space_inds = np.squeeze(clusters[clu_idx])
                ch_inds = np.unique(space_inds)  # list of channels we want to average and plot (!should be from one ch_type!)

                # ----------------- PLOT ----------------- #
                fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                # fig.suptitle(ch_type, fontsize=12, weight='bold')
                plt.axvline(0, linestyle='-', color='black', linewidth=2)
                for xx in range(3):
                    plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
                ax.set_xlabel('Time (ms)')
                condnames = list(evoked_reg.keys())
                if len(condnames) == 2:
                    colorslist = ['r', 'b']
                else:
                    NUM_COLORS = len(condnames)
                    cm = plt.get_cmap('jet')
                    colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
                for ncond, condname in enumerate(condnames):
                    data = evoked_reg[condname].copy()
                    evoked_funcs.plot_evoked_with_sem_1cond(data, condname, ch_type, ch_inds, color=colorslist[ncond], filter=True, axis=None)
                ymin, ymax = ax.get_ylim()
                ax.fill_betweenx((ymin, ymax), cinfo['sig_times'][0], cinfo['sig_times'][-1], color='orange', alpha=0.2)
                plt.legend(loc='upper right', fontsize=9)
                ax.set_xlim([-100, 750])
                ax.set_ylim([ymin, ymax])
                plt.title(ch_type + '_' + regressor_name + '_clust_' + str(i_clu+1), fontsize=10, weight='bold')
                fig.tight_layout(pad=0.5, w_pad=0)
                fig_name = fig_path + op.sep + 'stats_' + ch_type + '_' + regressor_name + '_clust_' + str(i_clu+1) + '_evo.png'
                print('Saving ' + fig_name)
                plt.savefig(fig_name, dpi=300)


def tmp_script_repeataltern_surprise_regression_figures():

    analysis_name = 'reg_repeataltern_surpriseOmegainfinity'
    regress_path = op.join(config.result_path, 'linear_models', analysis_name)
    names = ["intercept", "RepeatAlter", "RepeatAlternp1", "surpriseN", "surpriseNp1"]

    # =========== LOAD INDIVIDUAL REGRESSION RESULTS =========== #
    # Load data from all subjects
    betas = dict()
    for name in names:
        import glob
        evoked_dict = []
        for subj in config.subjects_list:
            path_evo = op.join(regress_path, subj)
            mne.read_evokeds(op.join(path_evo, 'beta_' + name + '-ave.fif'))
            evoked_dict.append(mne.read_evokeds(op.join(path_evo, 'beta_' + name + '-ave.fif')))

        betas[name] = mne.combine_evoked([evoked_dict[i][0] for i in range(len(config.subjects_list))], weights='equal')

    # ================= PLOT THE HEATMAPS OF THE GROUP-AVERAGED BETAS / CHANNEL ================ #
    fig_path = op.join(config.fig_path, 'Linear_regressions', analysis_name)
    utils.create_folder(fig_path)

    # Loop over the 3 ch_types
    plt.close('all')
    ch_types = ['eeg', 'grad', 'mag']
    for ch_type in ch_types:
        fig, axes = plt.subplots(1, len(betas.keys()), figsize=(len(betas.keys()) * 4, 6), sharex=False, sharey=False, constrained_layout=True)
        fig.suptitle(ch_type, fontsize=12, weight='bold')
        ax = axes.ravel()[::1]
        # Loop over the different betas
        for x, regressor_name in enumerate(betas.keys()):
            # ---- Data
            evokeds = betas[regressor_name]
            if ch_type == 'eeg':
                betadata = evokeds.copy().pick_types(eeg=True, meg=False).data
            elif ch_type == 'mag':
                betadata = evokeds.copy().pick_types(eeg=False, meg='mag').data
            elif ch_type == 'grad':
                betadata = evokeds.copy().pick_types(eeg=False, meg='grad').data
            minT = min(evokeds.times) * 1000
            maxT = max(evokeds.times) * 1000
            # ---- Plot
            im = ax[x].imshow(betadata, origin='upper', extent=[minT, maxT, betadata.shape[0], 0], aspect='auto', cmap='viridis')  # cmap='RdBu_r'
            ax[x].axvline(0, linestyle='-', color='black', linewidth=1)
            for xx in range(3):
                ax[x].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            ax[x].set_xlabel('Time (ms)')
            ax[x].set_ylabel('Channels')
            ax[x].set_title(regressor_name, loc='center', weight='normal')
            fig.colorbar(im, ax=ax[x], shrink=1, location='bottom')
        fig_name = fig_path + op.sep + ('betas_' + ch_type + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)

    # =========================== PLOT THE BUTTERFLY OF THE REGRESSORS ========================== #
    ylim_eeg = 20
    ylim_mag = 600
    ylim_grad = 200

    # Butterfly plots for violations (one graph per sequence) - in EEG/MAG/GRAD
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
    ts_args = dict(gfp=True, time_unit='s', ylim=ylim)
    topomap_args = dict(time_unit='s')
    times = 'peaks'
    for x, regressor_name in enumerate(betas.keys()):
        evokeds = betas[regressor_name]
        # EEG
        fig = evokeds.plot_joint(ts_args=ts_args, title='EEG_' + regressor_name,
                                 topomap_args=topomap_args, picks='eeg', times=times, show=False)
        fig_name = fig_path + op.sep + ('EEG_' + regressor_name + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)
        # MAG
        fig = evokeds.plot_joint(ts_args=ts_args, title='MAG_' + regressor_name,
                                 topomap_args=topomap_args, picks='mag', times=times, show=False)
        fig_name = fig_path + op.sep + ('MAG_' + regressor_name + '.png')
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
