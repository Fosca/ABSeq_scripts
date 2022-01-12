import os.path as op
import os
import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
import config
import numpy as np
from ABseq_func import epoching_funcs, regression_funcs, utils, TP_funcs, evoked_funcs, linear_reg_funcs, stats_funcs
import os.path as op
import os
from sklearn.preprocessing import scale
from mne.stats import linear_regression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn import linear_model
import numpy as np
import mne
import copy
import warnings

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
    epochs_clean.get_data()
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
    if subject == 'sub16-ma_190185':
        # in the case of sub16, no epochs are removed in the process of cleaning
        metadata_clean = metadata_notclean
    else:
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
def prepare_epochs_for_regression(subject, cleaned, epochs_fname, regressors_names, filter_name, remap_channels, apply_baseline, suffix):
    """
    This function loads and removes the epochs that have nan fields in the metadata for the regressors of interest in the analysis
    It performs the additional modifications to the epochs (filtering, remapping, baselining) that are asked
    It generates the results path string where the results should be stored

    :param subject: subject's NIP
    :param cleaned: Set it to True if you want to perform the analysis on cleaned (AR global) data
    :param regressors_names: List of fieds that exist in the metadata of the epochs
    :param filter_name: 'Stand', 'Viol', 'StandMultiStructure', 'Hab', 'Stand_excluseRA', 'Viol_excluseRA', 'StandMultiStructure_excluseRA', 'Hab_excluseRA'
    :param remap_channels: 'grad_to_mag' if you want to remaps the 306 channels onto 102 virtual mags and 'mag_to_grad' is you want to remap the 306 sensors into 102 sensors with the norm(rms) of the grads
    :param apply_baseline: Set it to True if initially the epochs are not baselined and you want to baseline them.
    :param suffix: Initial suffix value if your want to specify something in particular. In any case it may be updated according to the steps you do to the epochs.
    :return:

    subject = config.subjects_list[0]
    cleaned = True
    epochs_fname = ''
    regressors_names = ['Complexity']
    filter_name = 'Hab'
    remap_channels = 'mag_to_grad'
    suffix = ''

    """
    linear_reg_path = config.result_path + '/linear_models/' +filter_name+'/'
    epo_fname = linear_reg_path + epochs_fname

    if epochs_fname == '':
        epochs = regression_funcs.filter_good_epochs_for_regression_analysis(subject, clean=cleaned,
                                                                             fields_of_interest=regressors_names)
        results_path = os.path.dirname(epo_fname) + '/'
    else:
        print("----- loading the data from %s ------" % epo_fname)
        epochs = mne.read_epochs(epo_fname)
        results_path = os.path.dirname(epo_fname) + '/'
        results_path = op.abspath(op.join(results_path, os.pardir, os.pardir, 'from_' + results_path.split(op.sep)[-3] + '--'))

    # ====== normalization of regressors ====== #
    to_append_to_results_path = ''
    for name in regressors_names:
        if name != 'Intercept':
            epochs.metadata[name] = scale(epochs.metadata[name])
        to_append_to_results_path += '_' + name
    results_path = results_path + to_append_to_results_path[1:]+ '/'

    # - - - - OPTIONNAL STEPS - - - -
    if remap_channels =='grad_to_mag' and epochs_fname == '':
        print('Remapping grads to mags')
        # ---- build fake epochs with only mags ----
        epochs_final = epochs.copy()
        epochs_final.pick_types(meg='mag')
        epochs = epochs.as_type('mag',mode="accurate")
        print(str(len(epochs.ch_names)) + ' remaining channels!')
        suffix += 'remapped_gtm'
        epochs_final._data = epochs._data

    elif remap_channels =='mag_to_grad' and epochs_fname == '':
        print('Remapping mags to grads and taking the rms. The final type of channels will be mag but actually it is rms of grads')
        from mne.channels.layout import _merge_grad_data as rms_grad
        epochs_final = epochs.copy()
        epochs_final.pick_types(meg='mag')

        epochs = epochs.as_type(ch_type='grad',mode='accurate')
        data_good_shape = np.transpose(epochs._data,(1,0,2))
        data_good_shape = rms_grad(data_good_shape)
        data_good_shape = np.transpose(data_good_shape,(1,0,2))

        epochs_final._data = data_good_shape
        suffix += 'remapped_mtg'

    elif remap_channels == 'mag_to_grad' and epochs_fname != '':
        print("-- LOADING THE RESIDUALS THAT HAVE BEEN COMPUTED ON RMS OF MAG_TO_GRAD. NO NEED TO TAKE THE MAG_TO_GRAD AGAIN AND RMS. --")
        epochs_final = epochs

    if apply_baseline:
        epochs_final = epochs_final.apply_baseline(baseline=(-0.050, 0))
        suffix += 'baselined_'
    if cleaned:
        suffix += 'clean_'
    # ====== filter epochs according to the hab, test, including repeat alternate or not etc. ====== #
    before = len(epochs_final)
    filters = regression_funcs.filter_string_for_metadata()
    if filter_name is not None:
        epochs_final = epochs_final[filters[filter_name]]
    print('Keeping %.1f%% of epochs' % (len(epochs_final) / before * 100))

    return epochs_final, results_path, suffix

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

    explained = []

    for ii, name_reg in enumerate(regressors_names):
        explained_signal = np.asarray(
            [epochs.metadata[name_reg].values[i] * beta._data for i in range(len(epochs))])
        if name_reg == 'Intercept':
            intercept = np.asarray(
            [epochs.metadata[name_reg].values[i] * beta._data for i in range(len(epochs))])
        else:
            explained.append(explained_signal)

    if 'Intercept' in regressors_names:
        residuals = residuals - intercept - np.mean(explained,axis=0)
        intercept_epochs = epochs.copy()
        intercept_epochs._data = intercept
        intercept_epochs.save(op.join(results_path, 'intercept' + '--' + suffix[:-1] + '-epo.fif'), overwrite=True)
    else:
        residuals = residuals - np.mean(explained, axis=0)

    epochs.save(op.join(results_path, 'epochs' + '--' + suffix[:-1] + '-epo.fif'), overwrite=True)

    residual_epochs = epochs.copy()
    residual_epochs._data = residuals
    residual_epochs.save(op.join(results_path,'residuals' + '--' +  suffix[:-1] + '-epo.fif'), overwrite=True)

    explained_signal_epochs = epochs.copy()
    explained_signal_epochs._data = explained_signal
    explained_signal_epochs.save(op.join(results_path,'explained_signal' + '--' +  suffix[:-1] + '-epo.fif'), overwrite=True)

# ----------------------------------------------------------------------------------------------------------------------
def compute_regression(subject, regressors_names, epochs_fname, filter_name, cleaned=True, remap_channels='mag_to_grad',
                       apply_baseline=False, suffix='', save_evoked_for_regressor_level=True):
    """
    This function computes and saves the regression results when regressing on the epochs (or residuals if specified in epochs_fname)
    :param subject: subject's NIP
    :param regressors_names: List of fieds that exist in the metadata of the epochs
    :epochs_fname: '' if you want to load the normal epochs otherwise specify what you want to load (path starting in the linear_model folder of the results)
    :param cleaned: Set it to True if you want to perform the analysis on cleaned (AR global) data
    :param filter_name: 'Stand', 'Viol', 'StandMultiStructure', 'Hab', 'Stand_excluseRA', 'Viol_excluseRA', 'StandMultiStructure_excluseRA', 'Hab_excluseRA'
    :param remap_channels: 'grad_to_mag' if you want to remaps the 306 channels onto 102 virtual mags and 'mag_to_grad' is you want to remap the 306 sensors into 102 sensors with the norm(rms) of the grads
    :param apply_baseline: Set it to True if initially the epochs are not baselined and you want to baseline them.
    :param suffix: Initial suffix value if your want to specify something in particular. In any case it may be updated according to the steps you do to the epochs.

    apply_baseline = False
    suffix=''

    """

    # - prepare the epochs (removing the ones that have nans for the fields of interest) and define the results path and suffix ---
    epochs, results_path, suffix = prepare_epochs_for_regression(subject, cleaned, epochs_fname, regressors_names,
                                                                 filter_name, remap_channels, apply_baseline, suffix)

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
        save_reg_levels_evoked_path = results_path+ subject+op.sep+reg_name+'_evo/'
        utils.create_folder(save_reg_levels_evoked_path)
        # --- these are the different values of the regressor ----
        levels = np.unique(epochs.metadata[reg_name])
        if len(levels)>10:
            bins = np.linspace(np.min(levels),np.max(levels),6)  ## changed from 11 to 6 --> to recompute
            for ii in range(5):
                epochs["%s >= %0.02f and %s < %0.02f"%(reg_name, bins[ii], reg_name,bins[ii+1])].average().save(
                    save_reg_levels_evoked_path + str(ii) + '-' + suffix[:-1]+ '-ave.fif')
        else:
            for k, lev in enumerate(levels):
                # epochs["%s == %s"%(reg_name,lev)].average().save(save_reg_levels_evoked_path+op.sep+str(np.round(lev,2))+'-'+suffix[:-1]+'-ave.fif')
                epochs["%s == %s"%(reg_name,lev)].average().save(save_reg_levels_evoked_path+op.sep+str(k)+'-'+suffix[:-1]+'-ave.fif')

    save_reg_levels_evoked_path = results_path+ subject+'/SequenceID_evo/'
    utils.create_folder(save_reg_levels_evoked_path)
    levels = np.unique(epochs.metadata['SequenceID'])
    for lev in levels:
        epochs["%s == %s"%('SequenceID',lev)].average().save(save_reg_levels_evoked_path+op.sep+str(np.round(lev,2))+'-'+suffix[:-1]+'-ave.fif')

    return True

# ----------------------------------------------------------------------------------------------------------------------
def merge_individual_regression_results(regressors_names, epochs_fname, filter_name,suffix = ''):

    """
    This function loads individual regression results (betas, computed by 'compute_regression' function)
     and saves them as an epochs object, with Nsubjects betas, per regressor
    :param regressors_names: regressors used in the regression (required to find path and files)
    :epochs_fname: '' empty unless regresssions was conducted with the residuals of a previous regression
    :param filter_name: 'Stand', 'Viol', 'StandMultiStructure', 'Hab', 'Stand_excluseRA', 'Viol_excluseRA', 'StandMultiStructure_excluseRA', 'Hab_excluseRA'
    """

    # Results path
    results_path = op.join(config.result_path, 'linear_models', filter_name)
    if epochs_fname != '':
        results_path = op.abspath(op.join(results_path, 'from_' + epochs_fname + '--'))
        to_append_to_results_path = ''
        for name in regressors_names:
            to_append_to_results_path += '_' + name
        results_path = results_path + to_append_to_results_path[1:]
    else:
        to_append_to_results_path = ''
        for name in regressors_names:
            to_append_to_results_path += '_' + name
        results_path = op.join(results_path, to_append_to_results_path[1:])

    # Load data from all subjects
    tmpdat = dict()
    for name in regressors_names:
        tmpdat[name], path_evo = evoked_funcs.load_evoked('all', filter_name='beta_' + name + suffix, root_path=results_path)

    # Store as epo objects
    for name in regressors_names:
        dat = tmpdat[name][next(iter(tmpdat[name]))]
        exec(name + "_epo = mne.EpochsArray(np.asarray([dat[i][0].data for i in range(len(dat))]), dat[0][0].info, tmin="+str(np.round(dat[0][0].times[0],3))+")", locals(), globals())

    # Save group fif files
    out_path = op.join(results_path, 'group')
    utils.create_folder(out_path)
    for name in regressors_names:
        exec(name + "_epo.save(op.join(out_path, '" + name + suffix + "_epo.fif'), overwrite=True)")

# ----------------------------------------------------------------------------------------------------------------------
def regression_group_analysis(regressors_names, epochs_fname, filter_name, suffix='', Do3Dplot=True, ch_types = ['mag'],suffix_evoked = ''):

    """
    This function loads individual regression results merged as epochs arrays (with 'merge_individual_regression_results' function)
     and compute group level statistics (with various figures)
    :param regressors_names: regressors used in the regression (required to find path and files)
    :epochs_fname: '' empty unless regresssions was conducted with the residuals of a previous regression
    :param filter_name: 'Stand', 'Viol', 'StandMultiStructure', 'Hab', 'Stand_excluseRA', 'Viol_excluseRA', 'StandMultiStructure_excluseRA', 'Hab_excluseRA'
    :param suffix: '' or 'remapped_mtg' or 'remapped_gtm'
    :param Do3Dplot: create the sources figures (may not work, depending of the computer config)
    regressors_names = reg_names
    epochs_fname = ''
    filter_name = 'Hab'
    suffix='--remapped_mtgclean'
    Do3Dplot=False
    ch_types = ['mag']

    """

    # ===================== LOAD GROUP REGRESSION RESULTS & SET PATHS ==================== #

    # Results (data) path
    results_path = op.join(config.result_path, 'linear_models', filter_name)
    if epochs_fname != '':
        results_path = op.abspath(op.join(results_path, 'from_' + epochs_fname + '--'))
        to_append_to_results_path = ''
        for name in regressors_names:
            to_append_to_results_path += '_' + name
        results_path = results_path + to_append_to_results_path[1:]
    else:
        to_append_to_results_path = ''
        for name in regressors_names:
            to_append_to_results_path += '_' + name
        results_path = op.join(results_path, to_append_to_results_path[1:])
    results_path = op.join(results_path, 'group')

    # Ch_types
    if suffix == 'mag_to_grad' or 'mtg' in suffix:
        ch_types = ['mag']
        print('The grads we obtained are actually the RMS of grads so they should be considered as mags for the plots.')
    elif suffix == 'grad_to_mag' or 'gtm' in suffix:
        ch_types = ['mag']
    else:
        if ch_types =='':
            ch_types = config.ch_types

    # Load data
    betas = dict()
    for name in regressors_names:
        exec(name + "_epo = mne.read_epochs(op.join(results_path, '" + name + suffix + "_epo.fif'))")
        # betas[name] = globals()[name + '_epo']
        betas[name] = locals()[name + '_epo']
        print('There is ' + str(len(betas[name])) + ' betas for ' + name)

    # Results figures path
    fig_path = op.join(results_path, 'figures')
    utils.create_folder(fig_path)

    # Analysis name
    analysis_name = ''
    for name in regressors_names:
        analysis_name += '_' + name
    analysis_name = analysis_name[1:]

    # ====================== PLOT THE GROUP-AVERAGED SOURCES OF THE BETAS  ===================== #
    if Do3Dplot:
        all_stcs, all_betasevoked = linear_reg_funcs.plot_average_betas_with_sources(betas, analysis_name, fig_path, remap_grads=suffix)

    # ================= PLOT THE HEATMAPS OF THE GROUP-AVERAGED BETAS / CHANNEL ================ #
    linear_reg_funcs.plot_betas_heatmaps(betas, ch_types, fig_path, suffix=suffix)

    # =========================== PLOT THE BUTTERFLY OF THE REGRESSORS ========================== #
    linear_reg_funcs.plot_betas_butterfly(betas, ch_types, fig_path, suffix=suffix)

    # =========================================================== #
    # Group stats
    # =========================================================== #
    import matplotlib.pyplot as plt
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
                figname_initial = op.join(savepath, analysis_name + '_' + regressor_name + '_stats_' + ch_type+suffix)
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
                        plt.savefig(op.join(savepath, info + suffix + '_sources.svg'), bbox_inches='tight', dpi=600)
                        plt.close('all')

            # =========================================================== #
            # ==========  cluster evoked data plot --> per regressor level
            # =========================================================== #
            filter_evo = suffix.replace('-', '')  # we load all files in the subjects folder, we need an additional filter after importing

            if len(good_cluster_inds) > 0 and regressor_name != 'Intercept':
                # ------------------ LOAD THE EVOKED FOR THE CURRENT CONDITION ------------ #
                path = op.abspath(op.join(results_path, os.pardir))
                subpath = regressor_name + '_evo'
                # evoked_reg = evoked_funcs.load_regression_evoked(subject='all', path=path, subpath=subpath,filter=suffix_evoked)
                evoked_reg = evoked_funcs.load_regression_evoked(subject='all', path=path, subpath=subpath)
                warnings.warn("Keeping only evoked containing \"" + filter_evo + "\" ")
                evoked_reg = {k: v for (k, v) in evoked_reg.items() if filter_evo in k}

               # ----------------- PLOTS ----------------- #
                for i_clu, clu_idx in enumerate(good_cluster_inds):
                    cinfo = cluster_info[i_clu]
                    fig = stats_funcs.plot_clusters_evo(evoked_reg, cinfo, ch_type, i_clu, analysis_name=analysis_name + '_' + regressor_name, filter_smooth=False, legend=True, blackfig=False)
                    fig_name = savepath + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + '_clust_' + str(i_clu + 1) + suffix + '_evo.svg'
                    print('Saving ' + fig_name)
                    fig.savefig(fig_name, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
                    plt.close('all')

            # =========================================================== #
            # ==========  cluster evoked data plot --> per sequence
            # =========================================================== #
            if len(good_cluster_inds) > 0:
                # ------------------ LOAD THE EVOKED FOR EACH SEQUENCE ------------ #
                path = op.abspath(op.join(results_path, os.pardir))
                subpath = 'SequenceID' + '_evo'
                evoked_reg = evoked_funcs.load_regression_evoked(subject='all', path=path, subpath=subpath)
                warnings.warn("Keeping only evoked containing \"" + filter_evo + "\" ")
                evoked_reg = {k: v for (k, v) in evoked_reg.items() if filter_evo in k}

                # ----------------- PLOTS ----------------- #
                for i_clu, clu_idx in enumerate(good_cluster_inds):
                    cinfo = cluster_info[i_clu]
                    fig = stats_funcs.plot_clusters_evo(evoked_reg, cinfo, ch_type, i_clu, analysis_name=analysis_name + '_eachSeq', filter_smooth=False, legend=False, blackfig=False)
                    fig_name = savepath + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + '_clust_' + str(i_clu + 1) + suffix + '_eachSeq_evo.svg'
                    print('Saving ' + fig_name)
                    fig.savefig(fig_name, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
                    fig = stats_funcs.plot_clusters_evo_bars(evoked_reg, cinfo, ch_type, i_clu, analysis_name=analysis_name + '_eachSeq', filter_smooth=False, legend=False, blackfig=False)
                    fig_name = savepath + op.sep + analysis_name + '_' + regressor_name + '_stats_' + ch_type + '_clust_' + str(i_clu + 1) + suffix + '_eachSeq_evo_bars.svg'
                    print('Saving ' + fig_name)
                    fig.savefig(fig_name, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
                    plt.close('all')

            # =========================================================== #
            # ==========  heatmap betas plot
            # =========================================================== #
            if len(good_cluster_inds) > 0 and regressor_name != 'Intercept':
                linear_reg_funcs.plot_betas_heatmaps_with_clusters(analysis_name, betas, ch_type, regressor_name, cluster_info, good_cluster_inds, savepath,suffix)
