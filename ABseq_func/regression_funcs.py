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

# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
def prepare_epochs_for_regression(subject,cleaned,epochs_fname,regressors_names,filter_name,remap_grads,apply_baseline,suffix):
    """
    This function loads and removes the epochs that have nan fields in the metadata for the regressors of interest in the analysis
    It performs the additional modifications to the epochs (filtering, remapping, baselining) that are asked
    It generates the results path string where the results should be stored

    :param subject: subject's NIP
    :param cleaned: Set it to True if you want to perform the analysis on cleaned (AR global) data
    :param regressors_names: List of fieds that exist in the metadata of the epochs
    :param filter_name: 'Stand', 'Viol', 'StandMultiStructure', 'Hab', 'Stand_excluseRA', 'Viol_excluseRA', 'StandMultiStructure_excluseRA', 'Hab_excluseRA'
    :param remap_grads: True if you want to remaps the 306 channels onto 102 virtual mags
    :param apply_baseline: Set it to True if initially the epochs are not baselined and you want to baseline them.
    :param suffix: Initial suffix value if your want to specify something in particular. In any case it may be updated according to the steps you do to the epochs.
    :return:

    """
    linear_reg_path = config.result_path + '/linear_models/' +filter_name+'/'
    epo_fname = linear_reg_path + epochs_fname
    results_path = os.path.dirname(epo_fname) + '/'

    if epochs_fname == '':
        epochs = regression_funcs.filter_good_epochs_for_regression_analysis(subject, clean=cleaned,
                                                                             fields_of_interest=regressors_names)
    else:
        print("----- loading the data from %s ------" % epo_fname)
        epochs = mne.read_epochs(epo_fname)
    # ====== normalization of regressors ====== #
    to_append_to_results_path = ''
    for name in regressors_names:
        epochs.metadata[name] = scale(epochs.metadata[name])
        to_append_to_results_path += '_' + name
    results_path = results_path + to_append_to_results_path[1:]+ '/'
    # - - - - OPTIONNAL STEPS - - - -
    if remap_grads:
        print('Remapping grads to mags')
        epochs = epochs.as_type('mag')
        print(str(len(epochs.ch_names)) + ' remaining channels!')
        suffix += 'remapped_'

    if apply_baseline:
        epochs = epochs.apply_baseline(baseline=(-0.050, 0))
        suffix += 'baselined_'
    if cleaned:
        suffix += 'clean_'
    # ====== filter epochs according to the hab, test, including repeat alternate or not etc. ====== #
    before = len(epochs)
    filters = regression_funcs.filter_string_for_metadata()
    if filter_name is not None:
        epochs = epochs[filters[filter_name]]
    print('Keeping %.1f%% of epochs' % (len(epochs) / before * 100))

    return epochs, results_path, suffix

# ----------------------------------------------------------------------------------------------------------------------
def run_regression_CV(epochs, regressors_names):
    """
    Wrapper function to run the linear regression on the epochs for the list of regressors contained in regressors names
    It does it by cross validating 4 times
    """

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

# ----------------------------------------------------------------------------------------------------------------------
def save_regression_outputs(subject,epochs,suffix, results_path, regressors_names, betas, scores):
    """
    This function saves in the results_path the regression score, betas and residuals.
    """
    results_path = results_path+'/'+subject+'/'
    utils.create_folder(results_path)
    np.save(op.join(results_path, 'scores--'+suffix[:-1] + '.npy'), scores)

    # save betas and residuals
    residuals = epochs.get_data()
    for ii, name_reg in enumerate(regressors_names):
        beta = epochs.average().copy()
        beta._data = np.asarray(betas[ii, :, :])
        beta.save(op.join(results_path,'beta_' + name_reg + '--' + suffix[:-1] + '-ave.fif'))
        residuals = residuals - np.asarray(
            [epochs.metadata[name_reg].values[i] * beta._data for i in range(len(epochs))])

    residual_epochs = epochs.copy()
    residual_epochs._data = residuals
    residual_epochs.save(op.join(results_path,'residuals' + '--' +  suffix[:-1] + '-epo.fif'), overwrite=True)

# ----------------------------------------------------------------------------------------------------------------------
def compute_regression(subject, regressors_names, epochs_fname, filter_name, cleaned=True, remap_grads=True,
                       apply_baseline=False, suffix='',save_evoked_for_regressor_level=True):
    """
    This function computes and saves the regression results when regressing on the epochs (or residuals if specified in epochs_fname)
    :param subject: subject's NIP
    :param regressors_names: List of fieds that exist in the metadata of the epochs
    :epochs_fname: '' if you want to load the normal epochs otherwise specify what you want to load (path starting in the linear_model folder of the results)
    :param cleaned: Set it to True if you want to perform the analysis on cleaned (AR global) data
    :param filter_name: 'Stand', 'Viol', 'StandMultiStructure', 'Hab', 'Stand_excluseRA', 'Viol_excluseRA', 'StandMultiStructure_excluseRA', 'Hab_excluseRA'
    :param remap_grads: True if you want to remaps the 306 channels onto 102 virtual mags
    :param apply_baseline: Set it to True if initially the epochs are not baselined and you want to baseline them.
    :param suffix: Initial suffix value if your want to specify something in particular. In any case it may be updated according to the steps you do to the epochs.

    """

    # - prepare the epochs (removing the ones that have nans for the fields of interest) and define the results path and suffix ---
    epochs, results_path, suffix = prepare_epochs_for_regression(subject, cleaned, epochs_fname, regressors_names,
                                                                 filter_name, remap_grads, apply_baseline, suffix)

    if save_evoked_for_regressor_level:
        save_evoked_levels_regressors(epochs, subject, regressors_names, results_path, suffix)

    # --- run the regression with 4 folds ----
    betas, scores = run_regression_CV(epochs, regressors_names)
    #  save the outputs of the regression : score, betas and residuals
    save_regression_outputs(subject, epochs, suffix, results_path, regressors_names, betas, scores)

# ----------------------------------------------------------------------------------------------------------------------
def save_evoked_levels_regressors(epochs,subject, regressors_names,results_path, suffix):
    """
    This function computes and saves the regression results when regressing on the epochs (or residuals if specified in epochs_fname)
    :param epochs: subject's NIP
    :param regressors_names: List of fieds that exist in the metadata of the epochs
    """

    for reg_name in regressors_names:
        save_reg_levels_evoked_path = results_path+ subject+'/'+reg_name+'/'
        utils.create_folder(save_reg_levels_evoked_path)
        # --- these are the different values of the regressor ----
        levels = np.unique(epochs.metadata[reg_name])
        if len(levels)>10:
            bins = np.linspace(np.min(levels),np.max(levels),11)
            for ii in range(10):
                epochs["%s >= %0.02f and %s < %0.02f"%(reg_name, bins[ii], reg_name,bins[ii+1])].average().save(
                    save_reg_levels_evoked_path + str(ii) + '-' + suffix[:-1]+ '-ave.fif')
        else:
            for lev in levels:
                epochs["%s == %0.02f"%(reg_name,lev)].average().save(save_reg_levels_evoked_path+'/'+str(np.round(lev,2))+'-'+suffix[:-1]+'-ave.fif')

    save_reg_levels_evoked_path = results_path+ subject+'/SequenceID/'
    utils.create_folder(save_reg_levels_evoked_path)
    levels = np.unique(epochs.metadata['SequenceID'])
    for lev in levels:
        epochs["%s == %0.02f"%(reg_name,lev)].average().save(save_reg_levels_evoked_path+'/'+str(np.round(lev,2))+'-'+suffix[:-1]+'-ave.fif')

    return True