"""
======================
10. linear_regression
======================

"FirstLevel": Performs linear regression for each subject & saves results as fif files
"SecondLevel": Performs (group-level) cluster-corrected permutation test for each beta of the regression

Regressors to include (to indicate in "names=") are taken from epochs.metadata
Can use the residuals of a previous regression (with surprise) instead of original epochs (resid_epochs = True/False)
Uses filters (using epochs.metadata) to include/exclude some epochs in the regression (e.g. filters[analysis_name] = ['ViolationOrNot == 1'] for deviant items only)
Regressors are always normalized with "scale" (sklearn.preprocessing)

"""

from __future__ import division
import os.path as op
from matplotlib import pyplot as plt
import config
from ABseq_func import *
import mne
import numpy as np
from mne.stats import linear_regression
from sklearn.preprocessing import scale
import copy
import matplotlib.ticker as ticker

# =========================================================== #
# Options
# =========================================================== #
analysis_name = 'ViolComplexity'
names = ["Complexity"]  # Factors included in the regression
cleaned = True  # epochs cleaned with autoreject or not, only when using original epochs (resid_epochs=False)
resid_epochs = True  # use epochs created by regressing out surprise effects, instead of original epochs
if resid_epochs:
    resid_epochs_type = 'reg_repeataltern_surpriseOmegainfinity'  # 'residual_surprise'  'residual_model_constant' 'reg_repeataltern_surpriseOmegainfinity'
    # /!\ if 'reg_repeataltern_surpriseOmegainfinity', epochs wil be loaded from '/results/linear_models' instead of '/data/MEG/'
DoFirstLevel = True  # To compute the regression and evoked for each subject
DoSecondLevel = True  # Run the group level statistics

# Filter (for each analysis_name) to keep or exlude some epochs
filters = dict()
filters['StandComplexity'] = ['ViolationInSequence == 0 and StimPosition > 1']
filters['ViolComplexity'] = ['ViolationOrNot == 1']

print('\n#=====================================================================#\n                 Analysis: '
      + analysis_name + '\n#=====================================================================#\n')
# Results folder
if resid_epochs:
    results_path = op.join(config.result_path, 'linear_models', resid_epochs_type, analysis_name)
else:
    results_path = op.join(config.result_path, 'linear_models', analysis_name)
utils.create_folder(results_path)

if DoFirstLevel:

    # ========================= RUN LINEAR REGRESSION FOR EACH SUBJECT ========================== #
    for subject in config.subjects_list:
        # linear_reg_funcs.run_linear_regression_v2(analysis_name, names, subject, cleaned=True)

        print('\n#------------------------------------------------------------------#\n          Linear regression: ' + subject +
               '\n#------------------------------------------------------------------#\n')
        sub_results_path = op.join(results_path, subject)
        utils.create_folder(sub_results_path)

        # Load epochs data (& update metadata in case new things were added)
        if resid_epochs and resid_epochs_type == 'reg_repeataltern_surpriseOmegainfinity':
            resid_path = op.join(config.result_path, 'linear_models', 'reg_repeataltern_surpriseOmegainfinity', subject)
            fname_in = op.join(resid_path, 'residuals-epo.fif')
            print("Input: ", fname_in)
            epochs = mne.read_epochs(fname_in, preload=True)
        elif resid_epochs:
            epochs = epoching_funcs.load_resid_epochs_items(subject, type=resid_epochs_type)
        else:
            if cleaned:
                epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
                epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
            else:
                epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
                epochs = epoching_funcs.update_metadata_rejected(subject, epochs)

        # ====== filter items ====== #
        before = len(epochs)
        epochs = epochs[filters[analysis_name]]
        print('Keeping %.1f%% of epochs' % (len(epochs)/before*100))
        # ====== normalization ? ====== #
        for name in names:
            epochs.metadata[name] = scale(epochs.metadata[name])

        # ====== Linear model (all items) ====== #
        df = epochs.metadata
        epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
        regressors_names = ["Intercept"] + names
        res = linear_regression(epochs, epochs.metadata[regressors_names], names=regressors_names)

        # Save regression results
        for name in regressors_names:
            res[name].beta.save(op.join(sub_results_path, name + '.fif'))

        # ===== create evoked for each level of each regressor ===== #
        path_evo = op.join(config.meg_dir, subject, 'evoked')
        if cleaned:
            path_evo = path_evo + '_cleaned'
        for name in names:
            levels = np.unique(epochs.metadata[name])
            for x, level in enumerate(levels):
                if resid_epochs:
                    fname = op.join(path_evo, 'residepo_' + name + '_level' + str(x))
                else:
                    fname = op.join(path_evo,               name + '_level' + str(x))
                epochs[name + ' == "' + str(level) + '"'].average().save(fname + '-ave.fif')

    # # Create evoked with each (unique) level of the regressor / involves loading data again
    # for subject in config.subjects_list:
    #     evoked_funcs.create_evoked_for_regression_factors(names, subject, cleaned=True)

    # =========== LOAD INDIVIDUAL REGRESSION RESULTS AND SAVE THEM AS GROUP FIF FILES =========== #
    # ============================= (necessary only the first time) ============================= #
    regressors_names = ["Intercept"] + names  # intercept was added when running the regression
    # Load data from all subjects
    tmpdat = dict()
    for name in regressors_names:
        tmpdat[name] = evoked_funcs.load_evoked('all', filter_name=name, root_path=results_path)

    # Store as epo objects
    for name in regressors_names:
        dat = tmpdat[name][name[0:-3]]  # the dict key has 3 less characters than the original... because I did not respect MNE naming conventions ??
        exec(name + "_epo = mne.EpochsArray(np.asarray([dat[i][0].data for i in range(len(dat))]), dat[0][0].info, tmin=-0.1)")

    # Save group fif files
    out_path = op.join(results_path, 'group')
    utils.create_folder(out_path)
    for name in regressors_names:
        exec(name + "_epo.save(op.join(out_path, '" + name + "_epo.fif'), overwrite=True)")

if DoSecondLevel:
    # ============================ (RE)LOAD GROUP REGRESSION RESULTS =========================== #
    path = op.join(results_path, 'group')
    if names[0] != 'Intercept':
        names = ["Intercept"] + names  # intercept was added when running the regression

    betas = dict()
    for name in names:
        exec(name + "_epo = mne.read_epochs(op.join(path, '" + name + "_epo.fif'))")
        betas[name] = globals()[name + '_epo']

    # Results figures path
    if resid_epochs:
        fig_path = op.join(config.fig_path, 'Linear_regressions', resid_epochs_type, analysis_name)
    else:
        fig_path = op.join(config.fig_path, 'Linear_regressions', analysis_name)
    utils.create_folder(fig_path)

    # ================= PLOT THE HEATMAPS OF THE GROUP-AVERAGED BETAS / CHANNEL ================ #
    plt.close('all')
    for ch_type in ['eeg', 'grad', 'mag']:
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
    # Butterfly plots - in EEG/MAG/GRAD
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
    ts_args = dict(gfp=True, time_unit='s', ylim=ylim)
    topomap_args = dict(time_unit='s')
    times = 'peaks'
    for x, regressor_name in enumerate(betas.keys()):
        evokeds = betas[regressor_name].average()
        # EEG
        fig = evokeds.plot_joint(ts_args=ts_args, title='EEG_' + regressor_name, topomap_args=topomap_args, picks='eeg', times=times, show=False)
        fig_name = fig_path + op.sep + ('EEG_' + regressor_name + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)
        # MAG
        fig = evokeds.plot_joint(ts_args=ts_args, title='MAG_' + regressor_name, topomap_args=topomap_args, picks='mag', times=times, show=False)
        fig_name = fig_path + op.sep + ('MAG_' + regressor_name+'.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)
        # #GRAD
        fig = evokeds.plot_joint(ts_args=ts_args, title='GRAD_' + regressor_name, topomap_args=topomap_args, picks='grad', times=times, show=False)
        fig_name = fig_path + op.sep + ('GRAD_' + regressor_name + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

    # =========================================================== #
    # Group stats
    # =========================================================== #
    nperm = 5000  # number of permutations
    threshold = None  # If threshold is None, t-threshold equivalent to p < 0.05 (if t-statistic)
    p_threshold = 0.05
    tmin = 0.000  # timewindow to test (crop data)
    tmax = 0.500  # timewindow to test (crop data)

    for ch_type in ['eeg', 'grad', 'mag']:
        for x, regressor_name in enumerate(betas.keys()):
            data_stat = copy.deepcopy(betas[regressor_name])
            data_stat.crop(tmin=tmin, tmax=tmax)  # crop

            print('\n\n' + regressor_name + ', ch_type ' + ch_type)
            cluster_stats, data_array_chtype, _ = stats_funcs.run_cluster_permutation_test_1samp(data_stat, ch_type=ch_type, nperm=nperm, threshold=threshold, n_jobs=6, tail=0)
            cluster_info = stats_funcs.extract_info_cluster(cluster_stats, p_threshold, data_stat, data_array_chtype, ch_type)

            # Significant clusters
            T_obs, clusters, p_values, _ = cluster_stats
            good_cluster_inds = np.where(p_values < p_threshold)[0]
            print("Good clusters: %s" % good_cluster_inds)

            # PLOT CLUSTERS
            if len(good_cluster_inds) > 0:
                figname_initial = fig_path + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type
                stats_funcs.plot_clusters(cluster_info, ch_type, T_obs_max=5., fname=regressor_name, figname_initial=figname_initial, filter_smooth=True)

            # =========================================================== #
            # ==========  cluster evoked data plot
            # =========================================================== #

            if len(good_cluster_inds) > 0 and regressor_name != 'Intercept':
                # ------------------ LOAD THE EVOKED FOR THE CURRENT CONDITION ------------ #
                filter_name = regressor_name + '_level'
                if resid_epochs:
                    filter_name = 'residepo_' + filter_name
                evoked_reg = evoked_funcs.load_evoked(subject='all', filter_name=filter_name, filter_not=None, cleaned=True)

                for i_clu, clu_idx in enumerate(good_cluster_inds):
                    cinfo = cluster_info[i_clu]
                    # ----------------- PLOT ----------------- #
                    fig = stats_funcs.plot_clusters_evo(evoked_reg, cinfo, ch_type, i_clu, analysis_name=analysis_name + '_' + regressor_name, filter_smooth=True, legend=True, blackfig=True)
                    fig_name = fig_path + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + '_clust_' + str(i_clu + 1) + '_evo.jpg'
                    print('Saving ' + fig_name)
                    fig.savefig(fig_name, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
                    plt.close('all')

            # =========================================================== #
            # ==========  heatmap betas plot
            # =========================================================== #
            if len(good_cluster_inds) > 0 and regressor_name != 'Intercept':
                beta_average = copy.deepcopy(betas[regressor_name]).average()
                if ch_type == 'eeg':
                    betadata = beta_average.copy().pick_types(eeg=True, meg=False).data
                elif ch_type == 'mag':
                    betadata = beta_average.copy().pick_types(eeg=False, meg='mag').data
                elif ch_type == 'grad':
                    betadata = beta_average.copy().pick_types(eeg=False, meg='grad').data
                times = (beta_average.times) * 1000
                minT = min(times)
                maxT = max(times)
                # ---- Plot
                fig, ax = plt.subplots(1, 1, figsize=(4, 6), constrained_layout=True)
                fig.suptitle(ch_type, fontsize=12, weight='bold')
                im = ax.imshow(betadata, origin='upper', extent=[minT, maxT, betadata.shape[0], 0], aspect='auto', cmap='viridis')  # cmap='RdBu_r'
                ax.axvline(0, linestyle='-', color='black', linewidth=1)
                for xx in range(3):
                    ax.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Channels')
                ax.set_title(regressor_name, loc='center', weight='normal')
                # fig.colorbar(im, ax=ax, shrink=1, location='bottom')
                fmt = ticker.ScalarFormatter(useMathText=True)
                fmt.set_powerlimits((0, 0))
                cb = fig.colorbar(im, ax=ax, location='bottom', format=fmt, shrink=1, aspect=30, pad=.005)
                cb.ax.yaxis.set_offset_position('left')
                cb.set_label('Beta')
                # # ---- Add clusters v1
                # for i_clu, clu_idx in enumerate(good_cluster_inds):
                #     cinfo = cluster_info[i_clu]
                #     xstart = times[cinfo['time_inds'][0]]
                #     xend = times[cinfo['time_inds'][-1]]
                #     signif_chans = cinfo['channels_cluster']
                #     for ii in signif_chans:
                #         rect = patches.Rectangle((xstart, ii), xend-xstart, 1, facecolor='red', alpha=.5)
                #         ax.add_patch(rect)
                # ---- Add clusters v2
                mask = np.ones(betadata.shape)
                for i_clu, clu_idx in enumerate(good_cluster_inds):
                    cinfo = cluster_info[i_clu]
                    xstart = np.where(times == cinfo['sig_times'][0])[0][0]
                    xend = np.where(times == cinfo['sig_times'][-1])[0][0]
                    chanidx = cinfo['channels_cluster']
                    for yidx in range(len(chanidx)):
                        mask[chanidx[yidx], xstart:xend] = 0
                ax.imshow(mask, origin='upper', extent=[minT, maxT, betadata.shape[0], 0], aspect='auto', cmap='gray', alpha=.3)
                fig_name = fig_path + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + '_allclust_heatmap.jpg'
                print('Saving ' + fig_name)
                fig.savefig(fig_name, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close('all')




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
