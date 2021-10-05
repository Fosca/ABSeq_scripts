# This module contains all the functions related to the linear regression analyses
from __future__ import division
import sys

sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
from scipy.io import loadmat
from mne.stats import linear_regression, fdr_correction, bonferroni_correction, permutation_cluster_1samp_test
import os.path as op
import numpy as np
import config
from ABseq_func import *
from sklearn.preprocessing import scale
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from ABseq_func import source_estimation_funcs
import mne
import copy


def run_linear_regression(subject, cleaned=True):
    # Load data
    if cleaned:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
        epochs = epoching_funcs.update_metadata(subject, epochs)
    else:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
        epochs = epoching_funcs.update_metadata(subject, epochs)

    # add surprise_dynamic in metadata (excluding removed items/trials)
    print('We merge the dynamical model of surprise with the metadata')
    run_info_subject_dir = op.join(config.run_info_dir, subject)
    surprise = loadmat(op.join(run_info_subject_dir, 'surprise.mat'))
    surprise = list(surprise['Surprise'])
    badidx = np.where(epochs.drop_log)
    badidx = badidx[0]
    [surprise.pop(i) for i in badidx[::-1]]
    surprise = np.asarray(surprise)
    epochs.metadata['surprise_dynamic'] = surprise

    # ====== remove the first item of each sequence in the linear model ==========
    print('We remove the first sequence item for which the surprise is not well computed')
    epochs = epochs["StimPosition > 1"]

    # ====== normalization ?
    epochs.metadata['surprise_dynamic'] = scale(epochs.metadata['surprise_dynamic'])
    epochs.metadata['Complexity'] = scale(epochs.metadata['Complexity'])
    epochs.metadata['ViolationOrNot'] = scale(epochs.metadata['ViolationOrNot'])

    # ====== let's add in the metadata a term of violation_or_not X complexity ==========
    print('We remove the first sequence item for which the surprise is not well computed')
    # epochs.metadata['violation_X_complexity'] = np.asarray([epochs.metadata['ViolationOrNot'][i]*epochs.metadata['Complexity'][i] for i in range(len(epochs.metadata))])  # does not work, replaced by the next line (correct?)
    epochs.metadata['violation_X_complexity'] = scale(epochs.metadata['ViolationOrNot'] * epochs.metadata['Complexity'])

    # Linear model (all items)
    df = epochs.metadata
    epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
    names = ["Intercept", "Complexity", "surprise_dynamic", "ViolationOrNot", "violation_X_complexity"]
    res = linear_regression(epochs, epochs.metadata[names], names=names)

    # Save regression results
    out_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic', subject)
    utils.create_folder(out_path)
    res['Intercept'].beta.save(op.join(out_path, 'beta_intercept-ave.fif'))
    res['Complexity'].beta.save(op.join(out_path, 'beta_Complexity-ave.fif'))
    res['surprise_dynamic'].beta.save(op.join(out_path, 'beta_surprise_dynamic-ave.fif'))
    res['ViolationOrNot'].beta.save(op.join(out_path, 'beta_violation_or_not-ave.fif'))
    res['violation_X_complexity'].beta.save(op.join(out_path, 'beta_violation_X_complexity-ave.fif'))


def run_linear_regression_v2(analysis_name, regressor_names, subject, cleaned=True):
    # Load data & update metadata (in case new things were added)
    if cleaned:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
        epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
    else:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
        epochs = epoching_funcs.update_metadata_rejected(subject, epochs)

    # # ====== remove some items from the linear model ==========
    print('We remove the first sequence item for which the surprise is not well computed and for which there is no RepeatAlter')
    epochs = epochs["StimPosition > 1"]
    # print('We remove items from trials with violation')
    # epochs = epochs["ViolationInSequence == 0"]

    # ====== regressors
    names = regressor_names

    # ====== normalization ?
    for name in names:
        epochs.metadata[name] = scale(epochs.metadata[name])

    # ====== Linear model (all items)
    df = epochs.metadata
    epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
    names = ["Intercept"] + names
    res = linear_regression(epochs, epochs.metadata[names], names=names)

    # Save regression results
    out_path = op.join(config.result_path, 'linear_models', analysis_name, subject)
    utils.create_folder(out_path)
    for name in names:
        res[name].beta.save(op.join(out_path, name + '.fif'))

    # # TO REVIEW RESULTS...
    # import matplotlib.pyplot as plt
    # path = op.join(config.result_path, 'linear_models', 'test', subject)
    # Intercept = mne.read_evokeds(op.join(path, 'Intercept.fif'))
    # StimPosition = mne.read_evokeds(op.join(path, 'StimPosition.fif'))
    # RepeatAlter = mne.read_evokeds(op.join(path, 'RepeatAlter.fif'))
    # OpenedChunks = mne.read_evokeds(op.join(path, 'OpenedChunks.fif'))
    # Intercept[0].plot_joint()
    # StimPosition[0].plot_joint()
    # RepeatAlter[0].plot_joint()
    # OpenedChunks[0].plot_joint()
    # plt.close('all')


def run_linear_reg_surprise_repeat_alt(subject, with_complexity=False, cross_validate=True):
    TP_funcs.append_surprise_to_metadata_clean(subject)

    # ====== load the data , remove the first item for which the surprise is not computed ==========
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
    metadata = epoching_funcs.update_metadata(subject, clean=True, new_field_name=None, new_field_values=None)

    # ============ build the repeatAlter and the surprise 299 for n+1 ==================
    metadata_notclean = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None)

    # ====== attention il faut que je code à la main la présence de répétition ou d'alternance ===========

    metadata_notclean = repeat_alternate_from_metadata(metadata_notclean)
    RepeatAlternp1_notclean = metadata_notclean["RepeatAlter"].values[1:].tolist()
    RepeatAlternp1_notclean.append(np.nan)
    Surprisenp1_notclean = metadata_notclean["surprise_299"].values[1:].tolist()
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

    epochs = epochs[np.where(1 - np.isnan(epochs.metadata["surprise_299"].values))[0]]
    epochs = epochs[np.where(1 - np.isnan(epochs.metadata["RepeatAlternp1"].values))[0]]

    # =============== define the regressors =================
    # Repetition and alternation for n (not defined for the 1st item of the 16)
    # Repetition and alternation for n+1 (not defined for the last item of the 16)
    # Omega infinity for n (not defined for the 1st item of the 16)
    # Omega infinity for n+1 (not defined for the last item of the 16)

    names = ["Intercept", "surprise_299", "Surprisenp1", "RepeatAlter", "RepeatAlternp1"]
    if with_complexity:
        names.append("Complexity")
    for name in names:
        print(name)
        print(np.unique(epochs.metadata[name].values))

    # ============== define the output paths ======

    out_path = op.join(config.result_path, 'linear_models', 'reg_repeataltern_surpriseOmegainfinity', subject)
    if with_complexity:
        out_path = op.join(config.result_path, 'linear_models', 'reg_repeataltern_surpriseOmegainfinity_complexity',
                           subject)
    utils.create_folder(out_path)

    # ------------------- implementing the 4 folds CV -----------------

    if cross_validate:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=4)
        y_balancing = epochs.metadata["SequenceID"].values * 100 + epochs.metadata["StimPosition"].values

        Intercept = []
        surprise_299 = []
        Surprisenp1 = []
        RepeatAlternp1 = []
        RepeatAlter = []
        Complexity = []

        for train_index, test_index in skf.split(np.zeros(len(y_balancing)), y_balancing):
            print("======= running a new fold =======")
            lin_reg_cv = linear_regression(epochs[train_index], epochs[train_index].metadata[names], names=names)
            # score with cross validation
            Intercept.append(lin_reg_cv['Intercept'].beta)
            surprise_299.append(lin_reg_cv['surprise_299'].beta)
            Surprisenp1.append(lin_reg_cv['Surprisenp1'].beta)
            RepeatAlternp1.append(lin_reg_cv['RepeatAlternp1'].beta)
            RepeatAlter.append(lin_reg_cv['RepeatAlter'].beta)
            if with_complexity:
                Complexity.append(lin_reg_cv['Complexity'].beta)
        lin_reg_cv['Intercept'].beta = np.asarray(np.mean(Intercept, axis=0))
        lin_reg_cv['surprise_299'].beta = np.asarray(np.mean(surprise_299, axis=0))
        lin_reg_cv['Surprisenp1'].beta = np.asarray(np.mean(Surprisenp1, axis=0))
        lin_reg_cv['RepeatAlternp1'].beta = np.asarray(np.mean(RepeatAlternp1, axis=0))
        lin_reg_cv['RepeatAlter'].beta = np.asarray(np.mean(RepeatAlter, axis=0))
        if with_complexity:
            lin_reg_cv['Complexity'].beta = np.asarray(np.mean(Complexity, axis=0))
        lin_reg = lin_reg_cv

    # ------ end of the CV option -------

    else:
        lin_reg = linear_regression(epochs, epochs.metadata[names], names=names)
        # Save surprise regression results

    lin_reg['Intercept'].beta.save(op.join(out_path, 'beta_intercept-ave.fif'))
    lin_reg['surprise_299'].beta.save(op.join(out_path, 'beta_surpriseN-ave.fif'))
    lin_reg['Surprisenp1'].beta.save(op.join(out_path, 'beta_surpriseNp1-ave.fif'))
    lin_reg['RepeatAlternp1'].beta.save(op.join(out_path, 'beta_RepeatAlternp1-ave.fif'))
    lin_reg['RepeatAlter'].beta.save(op.join(out_path, 'beta_RepeatAlter-ave.fif'))
    if with_complexity:
        lin_reg['Complexity'].beta.save(op.join(out_path, 'beta_Complexity-ave.fif'))

    # save the residuals epoch in the same folder

    residuals = epochs.get_data() - lin_reg['Intercept'].beta.data
    for nn in ["surprise_299", "Surprisenp1", "RepeatAlter", "RepeatAlternp1"]:
        residuals = residuals - np.asarray([epochs.metadata[nn].values[i] * lin_reg[nn].beta._data for i in range(len(epochs))])
    if with_complexity:
        residuals = residuals - np.asarray([epochs.metadata["Complexity"].values[i] * lin_reg["Complexity"].beta._data for i in range(len(epochs))])

    residual_epochs = epochs.copy()
    residual_epochs._data = residuals

    # save the residuals epoch in the same folder
    residual_epochs.save(out_path + op.sep + 'residuals-epo.fif', overwrite=True)

    return True


def repeat_alternate_from_metadata(metadata):
    stimuli = metadata['StimID'].values.tolist()
    ra = [1 * (x != 0) for x in np.diff(stimuli)]
    ra = [np.nan] + ra
    count = 0
    while count < len(stimuli):
        print(count)
        if (count % 16) == 0:
            ra[count] = np.nan
        count += 1

    metadata["RepeatAlter"] = np.asarray(ra)
    return metadata


def run_linear_reg_surprise_repeat_alt_latest(subject, cross_validate=True):
    # remove old files
    meg_subject_dir = op.join(config.meg_dir, subject)
    metadata_path = op.join(meg_subject_dir, 'metadata_item.pkl')
    if op.exists(metadata_path):
        os.remove(metadata_path)
    metadata_path = op.join(meg_subject_dir, 'metadata_item_clean.pkl')
    if op.exists(metadata_path):
        os.remove(metadata_path)

    list_omegas = np.logspace(-1, 2, 50)
    TP_funcs.from_epochs_to_surprise(subject, list_omegas)
    TP_funcs.append_surprise_to_metadata_clean(subject)

    # =========== correction of the metadata with the surprise for the clean epochs ============
    # TP_funcs.append_surprise_to_metadata_clean(subject)  # already done above

    # ====== load the data , remove the first item for which the surprise is not computed ==========
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
    metadata = epoching_funcs.update_metadata(subject, clean=True, new_field_name=None, new_field_values=None)
    metadata["surprise_100"] = metadata["surprise_100.00000"]  # "rename" the variable
    # metadata.to_csv(r'tmp.csv')

    # ============ build the repeatAlter and the surprise 100 for n+1 ==================
    metadata_notclean = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None)
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

    # np.unique(metadata[np.isnan(epochs.metadata['RepeatAlter'])]['StimPosition'].values)
    # np.unique(metadata[np.isnan(epochs.metadata['surprise_100'])]['StimPosition'].values)
    # np.unique(metadata[np.isnan(metadata['RepeatAlternp1'])]['StimPosition'].values)
    # np.unique(metadata[np.isnan(metadata['Surprisenp1'])]['StimPosition'].values)

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
    out_path = op.join(config.result_path, 'linear_models', 'reg_repeataltern_surpriseOmegainfinity', subject)
    utils.create_folder(out_path)

    suffix = ''
    if cross_validate:
        #  ---- we replace the data in lin_reg ----
        suffix = '_cv'
        skf = StratifiedKFold(n_splits=4)
        y_balancing = epochs.metadata["SequenceID"].values * 100 + epochs.metadata["StimPosition"].values

        betas = []
        scores = []

        fold_number = 1
        for train_index, test_index in skf.split(np.zeros(len(y_balancing)), y_balancing):
            print("======= running a new fold =======")

            # predictor matrix
            preds_matrix_train = np.asarray(epochs[train_index].metadata[names].values)
            preds_matrix_test = np.asarray(epochs[test_index].metadata[names].values)
            betas_matrix = np.zeros((len(names), epochs.get_data().shape[1], epochs.get_data().shape[2]))
            scores_cv = np.zeros((epochs.get_data().shape[2]))
            residuals_cv = np.zeros(epochs[test_index].get_data().shape)

            for tt in range(epochs.get_data().shape[2]):
                # for each time-point, we run a regression for each channel
                reg = linear_model.LinearRegression(fit_intercept=False)
                data_train = epochs[train_index].get_data()
                data_test = epochs[test_index].get_data()

                reg.fit(y=data_train[:, :, tt], X=preds_matrix_train)
                betas_matrix[:, :, tt] = reg.coef_.T
                y_preds = reg.predict(preds_matrix_test)
                scores_cv[tt] = r2_score(y_true=data_test[:, :, tt], y_pred=y_preds)

                # build the residuals by removing the betas computed on the training set to the data from the testing set

                residuals_cv[:, :, tt] = data_test[:, :, tt] - y_preds
            residual_epochs_cv = epochs[test_index].copy()
            residual_epochs_cv._data = residuals_cv
            residual_epochs_cv.save(out_path + op.sep + 'fold_' + str(fold_number) + 'residuals-epo.fif', overwrite=True)

            betas.append(betas_matrix)
            scores.append(scores_cv)
            fold_number += 1

        # MEAN ACROSS CROSS-VALIDATION FOLDS
        betas = np.mean(betas, axis=0)
        scores = np.mean(scores, axis=0)

        lin_reg['Intercept'].beta._data = np.asarray(betas[0, :, :])
        lin_reg['surprise_100'].beta._data = np.asarray(betas[1, :, :])
        lin_reg['Surprisenp1'].beta._data = np.asarray(betas[2, :, :])
        lin_reg['RepeatAlter'].beta._data = np.asarray(betas[3, :, :])
        lin_reg['RepeatAlternp1'].beta._data = np.asarray(betas[4, :, :])

    # Save surprise regression results

    lin_reg['Intercept'].beta.save(op.join(out_path, suffix + 'beta_intercept-ave.fif'))
    lin_reg['surprise_100'].beta.save(op.join(out_path, suffix + 'beta_surpriseN-ave.fif'))
    lin_reg['Surprisenp1'].beta.save(op.join(out_path, suffix + 'beta_surpriseNp1-ave.fif'))
    lin_reg['RepeatAlternp1'].beta.save(op.join(out_path, suffix + 'beta_RepeatAlternp1-ave.fif'))
    lin_reg['RepeatAlter'].beta.save(op.join(out_path, suffix + 'beta_RepeatAlter-ave.fif'))

    if cross_validate:
        np.save(op.join(out_path, 'scores_linear_reg_CV.npy'), scores)
    # save the residuals epoch in the same folder

    residuals = epochs.get_data() - lin_reg['Intercept'].beta.data
    for nn in ["surprise_100", "Surprisenp1", "RepeatAlter", "RepeatAlternp1"]:
        residuals = residuals - np.asarray([epochs.metadata[nn].values[i] * lin_reg[nn].beta._data for i in range(len(epochs))])

    residual_epochs = epochs.copy()
    residual_epochs._data = residuals
    # save the residuals epoch in the same folder
    residual_epochs.save(out_path + op.sep + suffix + 'residuals-epo.fif', overwrite=True)
    # evoked_funcs.create_evoked_resid(subject, resid_epochs_type='reg_repeataltern_surpriseOmegainfinity')


def plot_average_betas_with_sources(betas, analysis_name, fig_path, remap_grads=False):
    savepath = op.join(fig_path, 'Sources')
    utils.create_folder(savepath)

    all_stcs = dict()
    all_betasevoked = dict()
    for x, regressor_name in enumerate(betas.keys()):
        all_stcs[regressor_name] = []
        all_betasevoked[regressor_name] = []
        for nsub, subject in enumerate(config.subjects_list):
            print(regressor_name + ' regressor: sources for subject ' + str(nsub))
            data = betas[regressor_name][nsub].average()  # 'fake' average since evoked was stored as 1 epoch
            stc = source_estimation_funcs.normalized_sources_from_evoked(subject, data, remap_grads=remap_grads)
            all_stcs[regressor_name].append(stc)
            all_betasevoked[regressor_name].append(data)

        # Group mean stc + betas
        n_subjects = len(all_stcs[regressor_name])
        mean_stc = all_stcs[regressor_name][0].copy()  # get copy of first instance
        for sub in range(1, n_subjects):
            mean_stc._data += all_stcs[regressor_name][sub].data
        mean_stc._data /= n_subjects
        mean_betas = mne.grand_average(all_betasevoked[regressor_name])

        # Create figures
        output_file = op.join(savepath, 'Sources_' + regressor_name + '.png')
        figure_title = analysis_name + ' regression: ' + regressor_name
        source_estimation_funcs.sources_evoked_figure(mean_stc, mean_betas, output_file, figure_title, timepoint='max', ch_type='mag', colormap='hot', colorlims='auto', signallims=None)
        output_file = op.join(savepath, 'Sources_' + regressor_name + '_at70ms.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc, mean_betas, output_file, figure_title, timepoint=0.070, ch_type='mag', colormap='viridis', colorlims='auto', signallims=None)
        output_file = op.join(savepath, 'Sources_' + regressor_name + '_at140ms.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc, mean_betas, output_file, figure_title, timepoint=0.140, ch_type='mag', colormap='viridis', colorlims='auto', signallims=None)

        # Timecourse source figure
        output_file = op.join(savepath, 'Sources_' + regressor_name + '_timecourse.png')
        times_to_plot = [.0, .050, .100, .150, .200, .250, .300]
        win_size = .050
        stc = mean_stc
        maxval = np.max(stc._data)
        colorlims = [maxval * .30, maxval * .40, maxval * .80]
        # plot and screenshot for each timewindow
        stc_screenshots = []
        for t in times_to_plot:
            twin_min = t
            twin_max = t + win_size
            stc_timewin = stc.copy()
            stc_timewin.crop(tmin=twin_min, tmax=twin_max)
            stc_timewin = stc_timewin.mean()
            brain = stc_timewin.plot(views=['lat'], surface='inflated', hemi='split', size=(1200, 600), subject='fsaverage', clim=dict(kind='value', lims=colorlims),
                                     subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'), background='w', smoothing_steps=5,
                                     colormap='hot', colorbar=False, time_viewer=False, backend='mayavi')
            screenshot = brain.screenshot()
            brain.close()
            nonwhite_pix = (screenshot != 255).any(-1)
            nonwhite_row = nonwhite_pix.any(1)
            nonwhite_col = nonwhite_pix.any(0)
            cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
            plt.close('all')
            stc_screenshots.append(cropped_screenshot)
        # main figure
        fig, axes = plt.subplots(len(times_to_plot), 1, figsize=(len(times_to_plot) * 1.1, 4))
        fig.suptitle(regressor_name, fontsize=8, fontweight='bold')
        for idx in range(len(times_to_plot)):
            axes[idx].imshow(stc_screenshots[idx])
            axes[idx].axis('off')
            twin_min = times_to_plot[idx]
            twin_max = times_to_plot[idx] + win_size
            axes[idx].set_title('[%d - %d ms]' % (twin_min * 1000, twin_max * 1000), fontsize=6)
        # tweak margins and spacing
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.01, top=0.9, wspace=1, hspace=0.5)
        fig.savefig(output_file, bbox_inches='tight', dpi=600)
        print('========> ' + output_file + " saved !")
        plt.close(fig)

        # Or explore sources activation
        # maxvtx,  max_t_val = mean_stc.get_peak()
        # brain = mean_stc.plot(views=['lat'], surface='pial', hemi='split', size=(1200, 600), subject='fsaverage', clim='auto',
        #                       subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'), initial_time=max_t_val, smoothing_steps=5, time_viewer=True) # show_traces=True, (not available with MNE 0.19 ?)
        # mni_max = mne.vertex_to_mni(maxvtx, 0,  subject='fsaverage', subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'))[0]
        # print('Peak at {:.0f}'.format(mni_max[0]) + ',{:.0f}'.format(mni_max[1]) + ', {:.0f}'.format(mni_max[2]))
        # plot the peak activation
        # plt.figure()
        # plt.axes([.1, .275, .85, .625])
        # hl = plt.plot(stc.times, stc.data[maxvtx], 'b')[0]
        # lt.xlabel('Time (s)')
        # plt.ylabel('Source amplitude (dSPM)')
        # plt.xlim(stc.times[0], stc.times[-1])
        # plt.figlegend([hl], ['Peak at = %s' % mni_max.round(2)], 'lower center')
        # plt.show()

    return all_stcs, all_betasevoked  # can be useful


def plot_betas_heatmaps(betas, ch_types, fig_path,suffix=''):
    savepath = op.join(fig_path, 'Signals')
    utils.create_folder(savepath)
    for ch_type in ch_types:
        fig, axes = plt.subplots(1, len(betas.keys()), figsize=(len(betas.keys()) * 4, 6), sharex=False, sharey=False, constrained_layout=True)
        fig.suptitle(ch_type, fontsize=12, weight='bold')
        # ax = axes.ravel()[::1]

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
            if len(betas.keys()) == 1:
                subplots_ax = axes
            else:
                subplots_ax = axes[x]
            im = subplots_ax.imshow(betadata, origin='upper', extent=[minT, maxT, betadata.shape[0], 0], aspect='auto', cmap='viridis')  # cmap='RdBu_r'
            subplots_ax.axvline(0, linestyle='-', color='black', linewidth=1)
            for xx in range(3):
                subplots_ax.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            subplots_ax.set_xlabel('Time (ms)')
            subplots_ax.set_ylabel('Channels')
            subplots_ax.set_title(regressor_name, loc='center', weight='normal')
            fig.colorbar(im, ax=subplots_ax, shrink=1, location='bottom')
        fig_name = op.join(savepath, 'betas_' + ch_type +suffix+ '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)
    # return betadata, plt, savepath


def plot_betas_heatmaps_with_clusters(analysis_name, betas, ch_type, regressor_name, cluster_info, good_cluster_inds, savepath,suffix=''):
    beta_average = betas[regressor_name].copy().average()
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
    # ---- Add clusters
    mask = np.ones(betadata.shape)
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        cinfo = cluster_info[i_clu]
        xstart = np.where(times == cinfo['sig_times'][0])[0][0]
        xend = np.where(times == cinfo['sig_times'][-1])[0][0]
        chanidx = cinfo['channels_cluster']
        for yidx in range(len(chanidx)):
            mask[chanidx[yidx], xstart:xend] = 0
    ax.imshow(mask, origin='upper', extent=[minT, maxT, betadata.shape[0], 0], aspect='auto', cmap='gray', alpha=.3)
    fig_name = savepath + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + suffix + '_allclust_heatmap.jpg'
    print('Saving ' + fig_name)
    fig.savefig(fig_name, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close('all')


def plot_betas_butterfly(betas, ch_types, fig_path,suffix=''):
    savepath = op.join(fig_path, 'Signals')
    utils.create_folder(savepath)

    ylim_eeg = 0.3
    ylim_mag = 30
    if suffix == 'mag_to_grad':
        ylim_mag = 100

    ylim_grad = 6
    # Butterfly plots - in EEG/MAG/GRAD
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
    ts_args = dict(gfp=True, time_unit='s', ylim=ylim)
    topomap_args = dict(time_unit='s')
    times = 'peaks'
    for x, regressor_name in enumerate(betas.keys()):
        evokeds = betas[regressor_name].average()
        if 'eeg' in ch_types:  # EEG
            fig = evokeds.plot_joint(ts_args=ts_args, title='EEG_' + regressor_name, topomap_args=topomap_args, picks='eeg', times=times, show=False)
            fig_name = savepath + op.sep + ('EEG_' + regressor_name + suffix+'.png')
            print('Saving ' + fig_name)
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close(fig)
        if 'mag' in ch_types:  # MAG
            fig = evokeds.plot_joint(ts_args=ts_args, title='MAG_' + regressor_name, topomap_args=topomap_args, picks='mag', times=times, show=False)
            fig_name = savepath + op.sep + ('MAG_' + regressor_name +suffix+ '.png')
            print('Saving ' + fig_name)
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close(fig)
        if 'grad' in ch_types:  # GRAD
            fig = evokeds.plot_joint(ts_args=ts_args, title='GRAD_' + regressor_name, topomap_args=topomap_args, picks='grad', times=times, show=False)
            fig_name = savepath + op.sep + ('GRAD_' + regressor_name +suffix+ '.png')
            print('Saving ' + fig_name)
            plt.savefig(fig_name, dpi=300, bbox_inches='tight')
            plt.close(fig)
