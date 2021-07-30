import os.path as op
import os
import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
import config
import numpy as np
from ABseq_func import epoching_funcs, regression_funcs, utils, TP_funcs, epoching_funcs
import os.path as op
import os
from sklearn.preprocessing import scale
from mne.stats import linear_regression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn import linear_model
import numpy as np
import mne

# ----------------------------------------------------------------------------------------------------------------------
def update_metadata_epochs_and_save_epochs(subject):
    """
    This function updates the metadata fields for the epochs such that they contain all the useful information for
    the complexity and surprise regressions.
    """

    # update the metadata for the non-clean epochs by adding the surprise computed for an observer that has 100 items in memory.
    metadata_notclean = TP_funcs.from_epochs_to_surprise(subject, [100])
    epochs_notclean, fname = epoching_funcs.load_epochs_items(subject, cleaned=False,return_fname=True)

    # load the metadata for the non-cleaned epochs, remove the bad ones, and this becomes the metadata for the cleaned epochs
    epochs_clean, fname_clean = epoching_funcs.load_epochs_items(subject, cleaned=True,return_fname=True)

    # ============ build the repeatAlter and the surprise 100 for n+1 ==================
    # 1 - update the full epochs (not_clean) metadata with the new fields
    RepeatAlternp1_notclean = metadata_notclean["RepeatAlter"].values[1:].tolist()
    RepeatAlternp1_notclean.append(np.nan)
    Surprisenp1_notclean = metadata_notclean["surprise_100"].values[1:].tolist()
    Surprisenp1_notclean.append(np.nan)
    metadata_notclean = metadata_notclean.assign(Intercept=1)
    metadata_notclean = metadata_notclean.assign(RepeatAlternp1=RepeatAlternp1_notclean)
    metadata_notclean = metadata_notclean.assign(Surprisenp1=Surprisenp1_notclean)
    epochs_notclean.metadata = metadata_notclean
    epochs_notclean.save(fname,overwrite = True)

    # 2 - subselect only the good epochs indices to filter the metadata
    good_idx = [len(epochs_clean.drop_log[i]) == 0 for i in range(len(epochs_clean.drop_log))]
    where_good = np.where(good_idx)[0]
    RepeatAlternp1 = np.asarray(RepeatAlternp1_notclean)[where_good]
    Surprisenp1 = np.asarray(Surprisenp1_notclean)[where_good]
    metadata_clean = metadata_notclean[good_idx]

    metadata_clean = metadata_clean.assign(Intercept=1)  # Add an intercept for later
    metadata_clean = metadata_clean.assign(RepeatAlternp1=RepeatAlternp1)
    metadata_clean = metadata_clean.assign(Surprisenp1=Surprisenp1)  # Add an intercept for later

    epochs_clean.metadata = metadata_clean
    epochs_clean.save(fname_clean,overwrite = True)

    return True


def filter_good_epochs_for_regression_analysis(subject,clean=True,fields_of_interest = ['surprise_100','RepeatAlternp1']):
    """
    This function removes the epochs that have Nans in the fields of interest specified in the list
    """
    epochs = epoching_funcs.load_epochs_items(subject,cleaned=clean)
    if fields_of_interest is not None:
        for field in fields_of_interest:
            epochs = epochs[np.where(1 - np.isnan(epochs.metadata[field].values))[0]]
            print("--- removing the epochs that have Nan values for field %s ----\n"%field)

    if config.noEEG:
        epochs = epochs.pick_types(meg=True, eeg=False)
    else:
        epochs = epochs.pick_types(meg=True, eeg=True)

    return epochs


def filter_string_for_metadata():
    """
    function that generates a dictionnary for conveniant selection of type of epochs
    """

    filters = dict()
    filters['Stand'] = 'TrialNumber > 10 and ViolationInSequence == 0 and StimPosition > 1'
    filters['Viol'] = 'ViolationOrNot == 1'
    filters['StandMultiStructure'] = 'ViolationInSequence == 0 and StimPosition > 1'
    filters['Hab'] = 'TrialNumber <= 10 and StimPosition > 1'

    filters['Stand_excluseRA'] = 'TrialNumber > 10 and ViolationInSequence == 0 and StimPosition > 1 and SequenceID >= 3'
    filters['Viol_excluseRA'] = 'ViolationOrNot == 1 and SequenceID >= 3'
    filters['StandMultiStructure_excluseRA'] = 'ViolationInSequence == 0 and StimPosition > 1 and SequenceID >= 3'
    filters['Hab_excluseRA'] = 'TrialNumber <= 10 and StimPosition > 1 and SequenceID >= 3'

    return filters



def prepare_epochs_for_regression(subject,cleaned,epochs_fname,regressors_names,filter_name,remap_grads,lowpass_epochs,apply_baseline,suffix,linear_reg_path):
    """

    """
    epo_fname = linear_reg_path + epochs_fname
    results_path = os.path.dirname(epo_fname) + '/'
    if epochs_fname == '':
        epochs = regression_funcs.filter_good_epochs_for_regression_analysis(subject, clean=cleaned,
                                                                             fields_of_interest=regressors_names)
    else:
        print("----- loading the data from %s ------" % epo_fname)
        epochs = mne.read_epochs(epo_fname)
    # ====== normalization of regressors ====== #
    for name in regressors_names:
        epochs.metadata[name] = scale(epochs.metadata[name])
        results_path += '_' + name
    results_path +='/'
    # - - - - OPTIONNAL STEPS - - - -
    if remap_grads:
        print('Remapping grads to mags')
        epochs = epochs.as_type('mag')
        print(str(len(epochs.ch_names)) + ' remaining channels!')
        suffix += 'remapped_'
    if lowpass_epochs:
        print('Low pass filtering...')
        epochs = epochs.filter(l_freq=None,
                               h_freq=30)  # default parameters (maybe should filter raw data instead of epochs...)
        suffix += 'lowpassed_'
    if apply_baseline:
        epochs = epochs.apply_baseline(baseline=(-0.050, 0))
        suffix += 'baselined_'
    if cleaned:
        suffix += 'clean_'
    # ====== filter epochs according to the hab, test, including repeat alternate or not etc. ====== #
    before = len(epochs)
    filters = regression_funcs.filter_string_for_metadata()
    if filter_name is not None:
        suffix = filter_name + '-' + suffix
        epochs = epochs[filters[filter_name]]
    print('Keeping %.1f%% of epochs' % (len(epochs) / before * 100))

    return epochs, results_path, suffix


def run_regression_CV(epochs, regressors_names):
    # cross validate 4 folds
    skf = StratifiedKFold(n_splits=4)
    y_balancing = epochs.metadata["SequenceID"].values * 100 + epochs.metadata["StimPosition"].values

    betas = []
    scores = []
    fold_number = 1

    for train_index, test_index in skf.split(np.zeros(len(y_balancing)), y_balancing):
        print("======= running regression for fold %i =======" % fold_number)
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

    return betas, scores


def save_regression_outputs(subject, results_path, regressors_names, betas, scores):
    utils.create_folder(results_path)
    np.save(op.join(results_path, subject + '-' + 'scores' + suffix + '.npy'), scores)
    # save betas
    for ii, name_reg in enumerate(regressors_names):
        beta = epochs.average().copy()
        beta._data = np.asarray(betas[ii, :, :])
        beta.save(op.join(results_path, 'beta_' + name_reg + '--' + suffix[:-1] + '-ave.fif'))

    # save the residuals
    residuals = epochs.get_data()
    for nn in regressors_names:
        residuals = residuals - np.asarray(
            [epochs.metadata[nn].values[i] * lin_reg[nn].beta._data for i in range(len(epochs))])

    residual_epochs = epochs.copy()
    residual_epochs._data = residuals
    residual_epochs.save(op.join(results_path, 'residuals' + suffix + '-epo.fif'), overwrite=True)
