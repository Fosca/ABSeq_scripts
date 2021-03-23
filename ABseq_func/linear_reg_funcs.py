# This module contains all the functions related to the linear regression analyses
from __future__ import division
import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
from scipy.io import loadmat
import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
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
    epochs.metadata['violation_X_complexity'] = scale(epochs.metadata['ViolationOrNot']*epochs.metadata['Complexity'])

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
   good_idx = np.where([len(epochs.drop_log[i])==0 for i in range(len(epochs.drop_log))])[0]
   RepeatAlternp1 = np.asarray(RepeatAlternp1_notclean)[good_idx]
   Surprisenp1= np.asarray(Surprisenp1_notclean)[good_idx]
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

   names = ["Intercept", "surprise_299","Surprisenp1","RepeatAlter","RepeatAlternp1"]
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
       y_balancing = epochs.metadata["SequenceID"].values*100+epochs.metadata["StimPosition"].values

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
       lin_reg_cv['Intercept'].beta = np.asarray(np.mean(Intercept,axis=0))
       lin_reg_cv['surprise_299'].beta = np.asarray(np.mean(surprise_299,axis=0))
       lin_reg_cv['Surprisenp1'].beta = np.asarray(np.mean(Surprisenp1,axis=0))
       lin_reg_cv['RepeatAlternp1'].beta = np.asarray(np.mean(RepeatAlternp1,axis=0))
       lin_reg_cv['RepeatAlter'].beta = np.asarray(np.mean(RepeatAlter,axis=0))
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

   residuals = epochs.get_data()-lin_reg['Intercept'].beta.data
   for nn in ["surprise_299","Surprisenp1","RepeatAlter","RepeatAlternp1"]:
       residuals = residuals - np.asarray([epochs.metadata[nn].values[i]*lin_reg[nn].beta._data for i in range(len(epochs))])
   if with_complexity:
       residuals = residuals - np.asarray([epochs.metadata["Complexity"].values[i]*lin_reg["Complexity"].beta._data for i in range(len(epochs))])

   residual_epochs = epochs.copy()
   residual_epochs._data = residuals

   # save the residuals epoch in the same folder
   residual_epochs.save(out_path+op.sep+'residuals-epo.fif', overwrite=True)

   return True


def repeat_alternate_from_metadata(metadata):

    stimuli = metadata['StimID'].values.tolist()
    ra = [1*(x!=0) for x in np.diff(stimuli)]
    ra = [np.nan]+ra
    count = 0
    while count < len(stimuli):
        print(count)
        if (count%16)==0:
            ra[count]=np.nan
        count +=1

    metadata["RepeatAlter"] = np.asarray(ra)
    return metadata


def run_linear_reg_surprise_repeat_alt_latest(subject,cross_validate=True):

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
        y_balancing = epochs.metadata["SequenceID"].values*100+epochs.metadata["StimPosition"].values

        betas = []
        scores = []

        fold_number = 1
        for train_index, test_index in skf.split(np.zeros(len(y_balancing)), y_balancing):
            print("======= running a new fold =======")

            # predictor matrix
            preds_matrix_train = np.asarray(epochs[train_index].metadata[names].values)
            preds_matrix_test = np.asarray(epochs[test_index].metadata[names].values)
            betas_matrix = np.zeros((len(names),epochs.get_data().shape[1],epochs.get_data().shape[2]))
            scores_cv = np.zeros((epochs.get_data().shape[1],epochs.get_data().shape[2]))

            for tt in range(epochs.get_data().shape[2]):
                # for each time-point, we run a regression for each channel
                reg = linear_model.LinearRegression()
                data_train = epochs[train_index].get_data()
                data_test = epochs[test_index].get_data()

                reg.fit(y = data_train[:,:,tt], X = preds_matrix_train)
                betas_matrix[:,:,tt] = reg.coef_.T
                y_preds = reg.predict(preds_matrix_test)
                scores_cv[:,tt] = r2_score(y_true = data_test[:,:,tt],y_pred = y_preds)

                # build the residuals by removing the betas computed on the training set to the data from the testing set

                residuals_cv = data_test - y_preds
                residual_epochs_cv = epochs[test_index].copy()
                residual_epochs_cv._data = residuals_cv
                residual_epochs_cv.save(out_path + op.sep + 'fold_' + str(fold_number) + 'residuals-epo.fif', overwrite=True)

            betas.append(betas_matrix)
            scores.append(scores_cv)
            fold_number += 1

        # MEAN ACROSS CROSS-VALIDATION FOLDS
        betas = np.mean(betas,axis=0)
        scores = np.mean(scores,axis=0)

        lin_reg['Intercept'].beta._data = np.asarray(betas[0,:,:])
        lin_reg['surprise_100'].beta._data = np.asarray(betas[1,:,:])
        lin_reg['Surprisenp1'].beta._data = np.asarray(betas[2,:,:])
        lin_reg['RepeatAlternp1'].beta._data = np.asarray(betas[3,:,:])
        lin_reg['RepeatAlter'].beta._data = np.asarray(betas[4,:,:])

    # Save surprise regression results

    lin_reg['Intercept'].beta.save(op.join(out_path,suffix+ 'beta_intercept-ave.fif'))
    lin_reg['surprise_100'].beta.save(op.join(out_path,suffix+ 'beta_surpriseN-ave.fif'))
    lin_reg['Surprisenp1'].beta.save(op.join(out_path,suffix+ 'beta_surpriseNp1-ave.fif'))
    lin_reg['RepeatAlternp1'].beta.save(op.join(out_path,suffix+ 'beta_RepeatAlternp1-ave.fif'))
    lin_reg['RepeatAlter'].beta.save(op.join(out_path, suffix+'beta_RepeatAlter-ave.fif'))

    if cross_validate:
        np.save(op.join(out_path, 'scores_linear_reg_CV.npy'),scores)
    # save the residuals epoch in the same folder

    residuals = epochs.get_data() - lin_reg['Intercept'].beta.data
    for nn in ["surprise_100", "Surprisenp1", "RepeatAlter", "RepeatAlternp1"]:
        residuals = residuals - np.asarray([epochs.metadata[nn].values[i] * lin_reg[nn].beta._data for i in range(len(epochs))])

    residual_epochs = epochs.copy()
    residual_epochs._data = residuals

    # save the residuals epoch in the same folder
    residual_epochs.save(out_path + op.sep +suffix+ 'residuals-epo.fif', overwrite=True)

    # evoked_funcs.create_evoked_resid(subject, resid_epochs_type='reg_repeataltern_surpriseOmegainfinity')


