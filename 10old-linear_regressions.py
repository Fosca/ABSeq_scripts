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
import pandas as pd
import glob
import os
from importlib import reload
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn import linear_model

# Exclude some subjects
# config.exclude_subjects.append('sub10-gp_190568')
# config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
# config.subjects_list.sort()

print('############------- Running analysis with ' + str(len(config.subjects_list)) + ' subjects ! -------############')

# =========================================================== #
# Options
# =========================================================== #
analysis_name = 'HabComplexity'
# names = ['StimID', 'ChunkBeginning', 'ChunkEnd', 'WithinChunkPosition']  # error if 'WithinChunkPositionReverse' also included // Factors included in the regression
names = ['Complexity']  # error if 'WithinChunkPositionReverse' also included // Factors included in the regression
exclude_Repeat_and_Alternate = False
cross_validate = True
cleaned = True  # epochs cleaned with autoreject or not, only when using original epochs (resid_epochs=False)
resid_epochs = False  # use epochs created by regressing out surprise effects, instead of original epochs
use_baseline = True  # apply baseline to the epochs before running the regression
lowpass_epochs = False  # option to filter epochs with  30Hz lowpass filter
remap_grads = True  # combine mag et grads, cf https://mne.tools/stable/auto_examples/preprocessing/virtual_evoked.html
suffix = ''
if cross_validate:
    suffix = '_cv'
Do3Dplot = True
RunStats = True
if resid_epochs:
    resid_epochs_type = 'reg_repeataltern_surpriseOmegainfinity'  # 'residual_surprise'  'residual_model_constant' 'reg_repeataltern_surpriseOmegainfinity'
    # /!\ if 'reg_repeataltern_surpriseOmegainfinity', epochs wil be loaded from '/results/linear_models' instead of '/data/MEG/'
DoFirstLevel = False  # To compute the regression and evoked for each subject
DoSecondLevel = True  # Run the group level statistics



# Filter (for each analysis_name) to keep or exclude some epochs
filters = dict()
filters['StandComplexity'] = ['TrialNumber > 10 and ViolationInSequence == 0 and StimPosition > 1']
filters['ViolComplexity'] = ['ViolationOrNot == 1']
filters['StandMultiStructure'] = ['ViolationInSequence == 0 and StimPosition > 1']
filters['HabComplexity'] = ['TrialNumber <= 10 and StimPosition > 1']

if exclude_Repeat_and_Alternate:
    for key in filters.keys():
        filters[key] = filters[key][0] + ' and SequenceID >= 3'
    number_of_sequences = 5
else:
    number_of_sequences = 7

print('\n#=====================================================================#\n                 Analysis: ' + analysis_name + '\n#=====================================================================#\n')
# Results folder
if resid_epochs:
    results_path = op.join(config.result_path, 'linear_models', analysis_name, 'TP_corrected_data', 'Signals')
else:
    results_path = op.join(config.result_path, 'linear_models', analysis_name, 'Original_data', 'Signals')
if use_baseline:
    results_path = results_path + op.sep + 'With_baseline_correction'
utils.create_folder(results_path)

if DoFirstLevel:
    # ========================= RUN LINEAR REGRESSION FOR EACH SUBJECT ========================== #
    for subject in config.subjects_list:
        # linear_reg_funcs.run_linear_regression_v2(analysis_name, names, subject, cleaned=True)

        print('\n#------------------------------------------------------------------#\n          Linear regression: ' + subject + '\n#------------------------------------------------------------------#\n')
        sub_results_path = op.join(results_path, 'data', subject)
        utils.create_folder(sub_results_path)

        # Load epochs data (& update metadata in case new things were added)
        if resid_epochs and resid_epochs_type == 'reg_repeataltern_surpriseOmegainfinity':
            resid_path = op.join(config.result_path, 'linear_models', 'reg_repeataltern_surpriseOmegainfinity', subject)
            fname_in = op.join(resid_path, 'residuals-epo.fif'); print('!!!!! OLD VERSION OF RESID !!!!!!')
            # fname_in = op.join(resid_path, suffix + 'residuals-epo.fif')
            print("Input: ", fname_in)
            epochs = mne.read_epochs(fname_in, preload=True)
        elif resid_epochs:
            epochs = epoching_funcs.load_resid_epochs_items(subject, type=resid_epochs_type)
        else:
            if cleaned:
                epochs = epoching_funcs.load_epochs_items(subject, cleaned=True, AR_type='global')
                # epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
            else:
                epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
                epochs = epoching_funcs.update_metadata_rejected(subject, epochs)

        # ====== REMAP GRAD TO MAGS !! ====== #
        if remap_grads:
            print('Remapping grads to mags')
            epochs = epochs.as_type('mag')
            print(str(len(epochs.ch_names)) + ' remaining channels!')

        if lowpass_epochs:
            print('Low pass filtering...')
            epochs = epochs.filter(l_freq=None, h_freq=30)  # default parameters (maybe should filter raw data instead of epochs...)

        # ====== filter items ====== #
        before = len(epochs)
        epochs = epochs[filters[analysis_name]]
        print('Keeping %.1f%% of epochs' % (len(epochs) / before * 100))

        # ====== Updating complexity values !! ====== #
        print('Updating complexity values in metadata using config.complexity...')
        for ii in epochs.metadata.index:
            seqID = epochs.metadata.at[ii, 'SequenceID']
            epochs.metadata.at[ii,'Complexity'] = config.complexity[seqID]

        # ====== apply baseline ? ====== #
        if use_baseline:
            epochs = epochs.apply_baseline(baseline=(-0.050, 0))
        # ====== normalization of regressors ====== #
        for name in names:
            epochs.metadata[name] = scale(epochs.metadata[name])

        # ====== Linear model (all items) ====== #
        if config.noEEG:
            epochs = epochs.pick_types(meg=True, eeg=False)
        else:
            epochs = epochs.pick_types(meg=True, eeg=True)

        df = epochs.metadata
        epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later

        regressors_names = ["Intercept"] + names
        res = linear_regression(epochs, epochs.metadata[regressors_names], names=regressors_names)

        if cross_validate:
            skf = StratifiedKFold(n_splits=4)
            y_balancing = epochs.metadata["SequenceID"].values * 100 + epochs.metadata["StimPosition"].values
            betas = []
            scores = []
            fold_number = 1
            for train_index, test_index in skf.split(np.zeros(len(y_balancing)), y_balancing):
                print("======= running regression for fold %i ======="%fold_number)

                # predictor matrix
                preds_matrix_train = np.asarray(epochs[train_index].metadata[regressors_names].values)
                preds_matrix_test = np.asarray(epochs[test_index].metadata[regressors_names].values)
                betas_matrix = np.zeros((len(regressors_names), epochs.get_data().shape[1], epochs.get_data().shape[2]))
                scores_cv = np.zeros((epochs.get_data().shape[1], epochs.get_data().shape[2]))

                for tt in range(epochs.get_data().shape[2]):
                    # for each time-point, we run a regression for each channel
                    reg = linear_model.LinearRegression(fit_intercept=False)
                    data_train = epochs[train_index].get_data()
                    data_test = epochs[test_index].get_data()

                    reg.fit(y=data_train[:, :, tt], X=preds_matrix_train)
                    betas_matrix[:, :, tt] = reg.coef_.T
                    y_preds = reg.predict(preds_matrix_test)
                    scores_cv[:, tt] = r2_score(y_true=data_test[:, :, tt], y_pred=y_preds)

                betas.append(betas_matrix)
                scores.append(scores_cv)
                fold_number += 1

            # MEAN ACROSS CROSS-VALIDATION FOLDS
            betas = np.mean(betas, axis=0)
            scores = np.mean(scores, axis=0)

            np.save(op.join(sub_results_path, 'score' + suffix + '.npy'),scores)

            for ii, name_reg in enumerate(regressors_names):
                res[name_reg].beta._data = np.asarray(betas[ii, :, :])

        # Save regression results
        for name in regressors_names:
            res[name].beta.save(op.join(sub_results_path, name + suffix + '.fif'))

        # ===== create evoked for each level of each regressor ===== #
        if config.noEEG:
            path_evo0 = op.join(config.meg_dir, subject, 'noEEG')
        else:
            path_evo0 = op.join(config.meg_dir, subject)
        if cleaned:
            path_evo = op.join(path_evo0, 'evoked_cleaned')
        if resid_epochs:
            path_evo = op.join(path_evo0, 'evoked_resid')
        for name in names:
            # remove files from a previous version of the analysis
            for filename in glob.glob(op.join(path_evo, name) + '*'):
                os.remove(filename)
            # save data for each level level
            levels = np.unique(epochs.metadata[name])
            for x, level in enumerate(levels):
                fname = op.join(path_evo, analysis_name + '_analysis_' + name + '_level%02.0f' % x)
                epochs[name + ' == ' + str(level)].average().save(fname + '-ave.fif')

        # ===== also create evoked for each sequence (i.e. only with epochs used in the regression) ===== #
        for nseq in range(number_of_sequences):
            fname = op.join(path_evo, analysis_name + '_analysis_SequenceID_%02.0f' % (nseq+1))
            epochs['SequenceID == ' + str(nseq+1)].average().save(fname + '-ave.fif')

    # =========== LOAD INDIVIDUAL REGRESSION RESULTS AND SAVE THEM AS GROUP FIF FILES =========== #
    # ============================= (necessary only the first time) ============================= #
    regressors_names = ["Intercept"] + names  # intercept was added when running the regression
    # Load data from all subjects
    tmpdat = dict()
    for name in regressors_names:
        tmpdat[name], path_evo = evoked_funcs.load_evoked('all', filter_name=name+ suffix, root_path=op.join(results_path, 'data'))

    # Store as epo objects
    for name in regressors_names:
        dat = tmpdat[name][name]
        exec(name + "_epo = mne.EpochsArray(np.asarray([dat[i][0].data for i in range(len(dat))]), dat[0][0].info, tmin="+str(np.round(dat[0][0].times[0],3))+")")

    # Save group fif files
    out_path = op.join(results_path, 'data', 'group')
    utils.create_folder(out_path)
    for name in regressors_names:
        exec(name + "_epo.save(op.join(out_path, '" + name + "_epo.fif'), overwrite=True)")

if DoSecondLevel:

    # ============================ (RE)LOAD GROUP REGRESSION RESULTS =========================== #
    path = op.join(results_path, 'data', 'group')
    if names[0] != 'Intercept':
        names = ["Intercept"] + names  # intercept was added when running the regression

    betas = dict()
    for name in names:
        exec(name + "_epo = mne.read_epochs(op.join(path, '" + name + "_epo.fif'))")
        betas[name] = globals()[name + '_epo']
        print('There is ' + str(len(betas[name])) + ' betas for ' + name)

    # Results figures path
    fig_path = op.join(results_path, 'figures')
    utils.create_folder(fig_path)

    if remap_grads:
        ch_types = ['mag']
    else:
        ch_types = config.ch_types

    # ================= PLOT THE GROUP-AVERAGED SOURCES OF THE BETAS  ================ #
    if Do3Dplot:
        all_stcs, all_betasevoked = linear_reg_funcs.plot_average_betas_with_sources(betas, analysis_name, fig_path, remap_grads=remap_grads)

    # ================= PLOT THE HEATMAPS OF THE GROUP-AVERAGED BETAS / CHANNEL ================ #
    savepath = op.join(fig_path, 'Signals')
    utils.create_folder(savepath)
    plt.close('all')
    for ch_type in ch_types:
        fig, axes = plt.subplots(1, len(betas.keys()), figsize=(len(betas.keys()) * 4, 6), sharex=False, sharey=False, constrained_layout=True)
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
            # minT = min(evokeds.times) * 1000
            minT = -0.050 * 1000
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
        fig_name = op.join(savepath, 'betas_' + ch_type + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)

    # =========================== PLOT THE BUTTERFLY OF THE REGRESSORS ========================== #
    ylim_eeg = 0.3
    ylim_mag = 30
    ylim_grad = 6
    # Butterfly plots - in EEG/MAG/GRAD
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
    ts_args = dict(gfp=True, time_unit='s', ylim=ylim)
    topomap_args = dict(time_unit='s')
    times = 'peaks'
    for x, regressor_name in enumerate(betas.keys()):
        evokeds = betas[regressor_name].average()
        if 'eeg' in ch_types: # EEG
            fig = evokeds.plot_joint(ts_args=ts_args, title='EEG_' + regressor_name, topomap_args=topomap_args, picks='eeg', times=times, show=False)
            fig_name = savepath + op.sep + ('EEG_' + regressor_name + '.png')
            print('Saving ' + fig_name)
            plt.savefig(fig_name)
            plt.close(fig)
        if 'mag' in ch_types: # MAG
            fig = evokeds.plot_joint(ts_args=ts_args, title='MAG_' + regressor_name, topomap_args=topomap_args, picks='mag', times=times, show=False)
            fig_name = savepath + op.sep + ('MAG_' + regressor_name + '.png')
            print('Saving ' + fig_name)
            plt.savefig(fig_name)
            plt.close(fig)
        if 'grad' in ch_types: # GRAD
            fig = evokeds.plot_joint(ts_args=ts_args, title='GRAD_' + regressor_name, topomap_args=topomap_args, picks='grad', times=times, show=False)
            fig_name = savepath + op.sep + ('GRAD_' + regressor_name + '.png')
            print('Saving ' + fig_name)
            plt.savefig(fig_name)
            plt.close(fig)

    # =========================================================== #
    # Group stats
    # =========================================================== #
    savepath = op.join(fig_path, 'Stats')
    utils.create_folder(savepath)
    nperm = 5000  # number of permutations
    threshold = None  # If threshold is None, t-threshold equivalent to p < 0.05 (if t-statistic)
    p_threshold = 0.05
    tmin = 0.000  # timewindow to test (crop data)
    tmax = 0.350  # timewindow to test (crop data)
    for ch_type in ch_types:
        for x, regressor_name in enumerate(betas.keys()):
            data_stat = copy.deepcopy(betas[regressor_name])
            data_stat.crop(tmin=tmin, tmax=tmax)  # crop

            print('\n\n' + regressor_name + ', ch_type ' + ch_type)
            cluster_stats = []
            data_array_chtype = []
            cluster_stats, data_array_chtype, _ = stats_funcs.run_cluster_permutation_test_1samp(data_stat, ch_type=ch_type, nperm=nperm, threshold=threshold, n_jobs=6, tail=0)
            cluster_info = stats_funcs.extract_info_cluster(cluster_stats, p_threshold, data_stat, data_array_chtype, ch_type)

            # Significant clusters
            T_obs, clusters, p_values, _ = cluster_stats
            good_cluster_inds = np.where(p_values < p_threshold)[0]
            print("Good clusters: %s" % good_cluster_inds)

            # PLOT CLUSTERS
            if len(good_cluster_inds) > 0:
                figname_initial = op.join(savepath, analysis_name + '_' + regressor_name + '_stats_' + ch_type)
                stats_funcs.plot_clusters(cluster_info, ch_type, T_obs_max=5., fname=regressor_name, figname_initial=figname_initial, filter_smooth=False)

            if Do3Dplot:
                # SOURCES FIGURES FROM CLUSTERS TIME WINDOWS
                if len(good_cluster_inds) > 0:
                    # Group mean stc (all_stcs loaded before)
                    n_subjects = len(all_stcs[regressor_name])
                    mean_stc = all_stcs[regressor_name][0].copy()  # get copy of first instance
                    for sub in range(1, n_subjects):
                        mean_stc._data += all_stcs[regressor_name][sub].data
                    mean_stc._data /= n_subjects

                    for i_clu in range(cluster_info['ncluster']):
                        cinfo = cluster_info[i_clu]
                        twin_min = cinfo['sig_times'][0] / 1000
                        twin_max = cinfo['sig_times'][-1] / 1000
                        stc_timewin = mean_stc.copy()
                        stc_timewin.crop(tmin=twin_min, tmax=twin_max)
                        stc_timewin = stc_timewin.mean()
                        # max_t_val = mean_stc.get_peak()[1]
                        brain = stc_timewin.plot(views=['lat'], surface='inflated', hemi='split', size=(1200, 600), subject='fsaverage', clim='auto',
                                                   subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'), smoothing_steps=5, time_viewer=False)
                        screenshot = brain.screenshot()
                        brain.close()
                        nonwhite_pix = (screenshot != 255).any(-1)
                        nonwhite_row = nonwhite_pix.any(1)
                        nonwhite_col = nonwhite_pix.any(0)
                        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
                        plt.close('all')
                        fig = plt.imshow(cropped_screenshot)
                        plt.axis('off')
                        info = analysis_name + '_' + regressor_name + ' [%d - %d ms]' % (twin_min*1000, twin_max*1000)
                        # figname_initial = savepath + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type
                        plt.title(info)
                        plt.savefig(op.join(savepath, info + '_sources.png'), bbox_inches='tight', dpi=600)
                        plt.close('all')


            # =========================================================== #
            # ==========  cluster evoked data plot
            # =========================================================== #

            if len(good_cluster_inds) > 0 and regressor_name != 'Intercept':
                # ------------------ LOAD THE EVOKED FOR THE CURRENT CONDITION ------------ #
                filter_name = analysis_name + '_analysis_' + regressor_name + '_level'
                if resid_epochs:
                    evoked_reg, _ = evoked_funcs.load_evoked(subject='all', filter_name=filter_name, filter_not=None, cleaned=True, evoked_resid=True)
                else:
                    evoked_reg, _ = evoked_funcs.load_evoked(subject='all', filter_name=filter_name, filter_not=None, cleaned=True, evoked_resid=False)
                # ----------------- PLOTS ----------------- #
                for i_clu, clu_idx in enumerate(good_cluster_inds):
                    cinfo = cluster_info[i_clu]
                    fig = stats_funcs.plot_clusters_evo(evoked_reg, cinfo, ch_type, i_clu, analysis_name=analysis_name + '_' + regressor_name, filter_smooth=False, legend=True, blackfig=False)
                    fig_name = savepath + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + '_clust_' + str(i_clu + 1) + '_evo.jpg'
                    print('Saving ' + fig_name)
                    fig.savefig(fig_name, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
                    plt.close('all')

            # =========================================================== #
            # ==========  cluster evoked data plot --> per sequence
            # =========================================================== #
            if len(good_cluster_inds) > 0:
                # ------------------ LOAD THE EVOKED FOR EACH SEQUENCE ------------ #
                filter_name = analysis_name + '_analysis_SequenceID_'
                if resid_epochs:
                    evoked_reg, _ = evoked_funcs.load_evoked(subject='all', filter_name=filter_name, filter_not=None, cleaned=True, evoked_resid=True)
                else:
                    evoked_reg, _ = evoked_funcs.load_evoked(subject='all', filter_name=filter_name, filter_not=None, cleaned=True, evoked_resid=False)
                # ----------------- PLOTS ----------------- #
                for i_clu, clu_idx in enumerate(good_cluster_inds):
                    cinfo = cluster_info[i_clu]
                    fig = stats_funcs.plot_clusters_evo(evoked_reg, cinfo, ch_type, i_clu, analysis_name=analysis_name + '_eachSeq', filter_smooth=False, legend=True, blackfig=False)
                    fig_name = savepath + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + '_clust_' + str(i_clu + 1) + '_eachSeq_evo.jpg'
                    print('Saving ' + fig_name)
                    fig.savefig(fig_name, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
                    fig = stats_funcs.plot_clusters_evo_bars(evoked_reg, cinfo, ch_type, i_clu, analysis_name=analysis_name + '_eachSeq', filter_smooth=False, legend=False, blackfig=False)
                    fig_name = savepath + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + '_clust_' + str(i_clu + 1) + '_eachSeq_evo_bars.jpg'
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
                fig_name = savepath + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + '_allclust_heatmap.jpg'
                print('Saving ' + fig_name)
                fig.savefig(fig_name, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close('all')
