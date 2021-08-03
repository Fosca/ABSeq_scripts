"""
Used to regress out suprise (& more)
"""

from __future__ import division
from mne.stats import linear_regression, fdr_correction, bonferroni_correction, permutation_cluster_1samp_test
import os.path as op
import numpy as np
import config
from ABseq_func import *
from sklearn.preprocessing import scale
import os
import mne
import matplotlib.pyplot as plt
import copy

# # Exclude some subjects
# config.exclude_subjects.append('sub10-gp_190568')
# config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
# config.subjects_list.sort()

# Recompute surprise ?
for subject in config.subjects_list:
    # remove old files
    meg_subject_dir = op.join(config.meg_dir, subject)
    if config.noEEG:
        meg_subject_dir = op.join(meg_subject_dir, 'noEEG')
    metadata_path = op.join(meg_subject_dir, 'metadata_item.pkl')
    if op.exists(metadata_path):
        os.remove(metadata_path)
    metadata_path = op.join(meg_subject_dir, 'metadata_item_clean.pkl')
    if op.exists(metadata_path):
        os.remove(metadata_path)

    list_omegas = np.logspace(-1, 2, 50)
    TP_funcs.from_epochs_to_surprise(subject, list_omegas)
    TP_funcs.append_surprise_to_metadata_clean(subject)

# Run the regression
for subject in config.subjects_list:

    # =========== correction of the metadata with the surprise for the clean epochs ============
    # TP_funcs.append_surprise_to_metadata_clean(subject)  # already done above

    # ====== load the data , remove the first item for which the surprise is not computed ==========
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
    metadata = epoching_funcs.update_metadata(subject, clean=True, new_field_name=None, new_field_values=None, recompute=False)
    metadata["surprise_100"] = metadata["surprise_100.00000"]  # "rename" the variable
    # metadata.to_csv(r'tmp.csv')

    # ============ build the repeatAlter and the surprise 100 for n+1 ==================
    metadata_notclean = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None, recompute=False)
    metadata_notclean["surprise_100"] = metadata_notclean["surprise_100.00000"]  # "rename" the variable
    RepeatAlternp1_notclean = metadata_notclean["RepeatAlter"].values[1:].tolist()
    RepeatAlternp1_notclean.append(np.nan)
    Surprisenp1_notclean = metadata_notclean["surprise_100"].values[1:].tolist()
    Surprisenp1_notclean.append(np.nan)
    good_idx = np.where([len(epochs.drop_log[i]) == 0 for i in range(len(epochs.drop_log))])[0]
    RepeatAlternp1 = np.asarray(RepeatAlternp1_notclean)[good_idx]
    Surprisenp1 = np.asarray(Surprisenp1_notclean)[good_idx]
    # ======================================================================================

    metadata = metadata.assign(Intercept=1)  # Add an intercept for later
    metadata = metadata.assign(RepeatAlternp1=RepeatAlternp1)
    metadata = metadata.assign(Surprisenp1=Surprisenp1)  # Add an intercept for later

    epochs.metadata = metadata
    epochs.pick_types(meg=True, eeg=True)

    np.unique(metadata[np.isnan(epochs.metadata['RepeatAlter'])]['StimPosition'].values)
    np.unique(metadata[np.isnan(epochs.metadata['surprise_100'])]['StimPosition'].values)
    np.unique(metadata[np.isnan(metadata['RepeatAlternp1'])]['StimPosition'].values)
    np.unique(metadata[np.isnan(metadata['Surprisenp1'])]['StimPosition'].values)

    epochs = epochs[np.where(1 - np.isnan(epochs.metadata["surprise_100"].values))[0]]
    epochs = epochs[np.where(1 - np.isnan(epochs.metadata["RepeatAlternp1"].values))[0]]

    # =============== define the regressors =================
    # Repetition and alternation for n (not defined for the 1st item of the 16)
    # Repetition and alternation for n+1 (not defined for the last item of the 16)
    # Omega infinity for n (not defined for the 1st item of the 16)
    # Omega infinity for n+1 (not defined for the last item of the 16)

    names = ["Intercept", "surprise_100", "Surprisenp1", "RepeatAlter", "RepeatAlternp1"]
    for name in names:
        print(name)
        print(np.unique(epochs.metadata[name].values))

    # ====== normalization ? ====== #
    for name in names[1:]:  # all but intercept
        epochs.metadata[name] = scale(epochs.metadata[name])

    # ====== baseline correction ? ====== #
    print('Baseline correction...')
    epochs = epochs.apply_baseline(baseline=(-0.050, 0))

    lin_reg = linear_regression(epochs, epochs.metadata[names], names=names)

    # Save surprise regression results
    out_path = op.join(config.result_path, 'linear_models', 'reg_repeataltern_surpriseOmegainfinity', subject)
    utils.create_folder(out_path)
    lin_reg['Intercept'].beta.save(op.join(out_path, 'beta_intercept-ave.fif'))
    lin_reg['surprise_100'].beta.save(op.join(out_path, 'beta_surpriseN-ave.fif'))
    lin_reg['Surprisenp1'].beta.save(op.join(out_path, 'beta_surpriseNp1-ave.fif'))
    lin_reg['RepeatAlternp1'].beta.save(op.join(out_path, 'beta_RepeatAlternp1-ave.fif'))
    lin_reg['RepeatAlter'].beta.save(op.join(out_path, 'beta_RepeatAlter-ave.fif'))

    # save the residuals epoch in the same folder
    residuals = epochs.get_data() - lin_reg['Intercept'].beta.data
    for nn in ["surprise_100", "Surprisenp1", "RepeatAlter", "RepeatAlternp1"]:
        residuals = residuals - np.asarray([epochs.metadata[nn].values[i] * lin_reg[nn].beta._data for i in range(len(epochs))])

    residual_epochs = epochs.copy()
    residual_epochs._data = residuals

    # save the residuals epoch in the same folder
    residual_epochs.save(out_path + op.sep + 'residuals-epo.fif', overwrite=True)

# Create evoked from the residuals epochs
for subject in config.subjects_list:
    evoked_funcs.create_evoked_resid(subject, resid_epochs_type='reg_repeataltern_surpriseOmegainfinity')

# =========== 2nd Level statistics =========== #
DoSecondLevel = True
Do3Dplot = True
if DoSecondLevel:

    analysis_name = 'reg_repeataltern_surpriseOmegainfinity'
    regressors_names = ["intercept", "surpriseN", "surpriseNp1", "RepeatAlter", "RepeatAlternp1"]
    results_path = op.join(config.result_path, 'linear_models', analysis_name)

    # =========== LOAD INDIVIDUAL REGRESSION RESULTS AND SAVE THEM AS GROUP FIF FILES =========== #
    # Load data from all subjects
    tmpdat = dict()
    for name in regressors_names:
        tmpdat[name]=[]
        for subj in config.subjects_list:
            file_name = op.join(results_path, subj, 'beta_'+name+'-ave.fif')
            tmpdat[name].append(mne.read_evokeds(file_name))
            # tmpdat[name], path_evo = evoked_funcs.load_evoked('all', filter_name=name+suffix, root_path=op.join(results_path, 'data'))

    # Store as epo objects
    for name in regressors_names:
        # dat = tmpdat[name][name[0:-3]]  # the dict key has 3 less characters than the original... because I did not respect MNE naming conventions ??
        dat = tmpdat[name]
        exec(name + "_epo = mne.EpochsArray(np.asarray([dat[i][0].data for i in range(len(dat))]), dat[0][0].info, tmin="+str(np.round(dat[0][0].times[0],3))+")")

    # Save group fif files
    out_path = op.join(results_path, 'data', 'group')
    utils.create_folder(out_path)
    for name in regressors_names:
        exec(name + "_epo.save(op.join(out_path, '" + name + "_epo.fif'), overwrite=True)")

    # ============================ (RE)LOAD GROUP REGRESSION RESULTS =========================== #
    path = op.join(results_path, 'data', 'group')
    betas = dict()
    for name in regressors_names:
        exec(name + "_epo = mne.read_epochs(op.join(path, '" + name + "_epo.fif'))")
        betas[name] = globals()[name + '_epo']
        print('There is ' + str(len(betas[name])) + ' betas for ' + name)

    # Results figures path
    fig_path = op.join(results_path, 'figures')
    utils.create_folder(fig_path)

    # ================= PLOT THE GROUP-AVERAGED SOURCES OF THE BETAS /  ================ #
    if Do3Dplot:
        linear_reg_funcs.average_betas_plot_with_sources(betas, analysis_name, fig_path)

    # ================= PLOT THE HEATMAPS OF THE GROUP-AVERAGED BETAS / CHANNEL ================ #
    savepath = op.join(fig_path, 'Signals')
    utils.create_folder(savepath)
    plt.close('all')
    for ch_type in config.ch_types:
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
        if 'eeg' in config.ch_types: # EEG
            fig = evokeds.plot_joint(ts_args=ts_args, title='EEG_' + regressor_name, topomap_args=topomap_args, picks='eeg', times=times, show=False)
            fig_name = savepath + op.sep + ('EEG_' + regressor_name + '.png')
            print('Saving ' + fig_name)
            plt.savefig(fig_name)
            plt.close(fig)
        # MAG
        fig = evokeds.plot_joint(ts_args=ts_args, title='MAG_' + regressor_name, topomap_args=topomap_args, picks='mag', times=times, show=False)
        fig_name = savepath + op.sep + ('MAG_' + regressor_name + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)
        # #GRAD
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
    for ch_type in config.ch_types:
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
                filter_name = regressor_name + '_level'
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
