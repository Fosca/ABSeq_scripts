# This module contains all the functions related to the decoding analysis
import sys

sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
from initialization_paths import initialization_paths
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import time
from sklearn.model_selection import StratifiedKFold
from ABseq_func import *
import config
from scipy.stats import sem
from ABseq_func import utils  # why do we need this now ?? (error otherwise)
import matplotlib.ticker as ticker

from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVR
from sklearn.model_selection import KFold
import numpy as np
import random
from jr.plot import base, gat_plot, pretty_gat, pretty_decod, pretty_slices
from sklearn.linear_model import LinearRegression

from sklearn.base import TransformerMixin
from ABseq_func import stats_funcs

# ______________________________________________________________________________________________________________________
def SVM_decoder():
    """
    Builds an SVM decoder that will be able to output the distance to the hyperplane once trained on data.
    It is meant to generalize across time by construction.
    :return:
    """
    clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear', probability=True))
    time_gen = GeneralizingEstimator(clf, scoring=None, n_jobs=8, verbose=True)

    return time_gen


# ______________________________________________________________________________________________________________________
def regression_decoder():
    """
    Builds an SVM decoder that will be able to output the distance to the hyperplane once trained on data.
    It is meant to generalize across time by construction.
    :return:
    """
    clf = make_pipeline(StandardScaler(), LinearSVR())
    time_gen = GeneralizingEstimator(clf, scoring=None, n_jobs=8, verbose=True)

    return time_gen


# ______________________________________________________________________________________________________________________
def leave_one_sequence_out(epochs, list_sequences):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for seqID in list_sequences:
        X_train.append(epochs["SequenceID != " + str(seqID)].get_data())
        y_train.append(epochs["SequenceID != " + str(seqID)].events[:, 2])
        X_test.append(epochs["SequenceID == " + str(seqID)].get_data())
        y_test.append(epochs["SequenceID == " + str(seqID)].events[:, 2])

    return X_train, y_train, X_test, y_test


# ______________________________________________________________________________________________________________________
def train_quads_test_others(epochs, list_sequences):
    X_train = epochs["SequenceID == 4 "].get_data()
    y_train = epochs["SequenceID == 4 "].events[:, 2]
    X_test = epochs["SequenceID != 4 "].get_data()
    y_test = epochs["SequenceID != 4 "].events[:, 2]

    return [X_train], [y_train], [X_test], [y_test]

# ----------------------------------------------------------------------------------------------------------------------
def train_test_different_blocks(epochs,return_per_seq = False):

    import random

    """
    For each sequence, check that there are two run_numbers for the sequence.
    If there are 2, then randomly put one in the training set and the other one in the test set.
    If there is just one, split the trials of that one into two sets, one for training the other for testing
    :param epochs:
    :return:
    """

    train_test_dict = {i:{'train':[],'test':[]} for i in range(1,8)}
    train_inds_fold1 = []
    train_inds_fold2 = []
    test_inds_fold1 = []
    test_inds_fold2 = []

    for seqID in range(1,8):
        epochs_Seq = epochs["SequenceID == %i "%seqID]
        n_runs = np.unique(epochs_Seq.metadata['RunNumber'].values)
        if len(n_runs) == 1:
            print('There is only one run for sequence ID %i'%(seqID))
            inds_seq = np.where(epochs.metadata['RunNumber'].values==n_runs)[0]
            np.random.shuffle(inds_seq)
            inds_1 = inds_seq[:int(np.floor(len(inds_seq)/2))]
            inds_2 = inds_seq[int(np.floor(len(inds_seq)/2)):]
        else:
            pick_run = random.randint(0, 1)
            run_train = n_runs[pick_run]
            run_test = n_runs[1-pick_run]
            inds_1 = np.where(epochs.metadata['RunNumber'].values==run_train)[0]
            inds_2 = np.where(epochs.metadata['RunNumber'].values==run_test)[0]

        train_inds_fold1.append(inds_1)
        train_inds_fold2.append(inds_2)

        test_inds_fold1.append(inds_2)
        test_inds_fold2.append(inds_1)

        train_test_dict[seqID]['train']= [inds_1,inds_2]
        train_test_dict[seqID]['test']= [inds_2,inds_1]

    if return_per_seq:
        return train_test_dict
    else:
        return [np.concatenate(train_inds_fold1),np.concatenate(train_inds_fold2)], [np.concatenate(test_inds_fold1),np.concatenate(test_inds_fold2)]



def SVM_ordinal_code_train_test_quads(subject, SVM_dec=SVM_decoder(),decim=1,sliding_window=True,crop=[0.1,0.3],dobaseline=True):
    """
    Train on the quads sequences (trials coming from the standards of the  test part), remove the first, second, 15th and 16th item of quads.
    Test on all the other sequences. It makes especially sense for 2 pairs, shrink, complex
    :param subject:
    :param load_residuals_regression:
    :param SVM_dec: The classifier type
    :param decim:
    :return:
    """

    # ------------------------------------------------------------------------------------------------------------------
    # ---------------- LOAD THE EPOCHS ON SINGLE ITEMS IN ORDER TO TRAIN THE DECODER  ----------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    suffix = ''

    epochs_train = epoching_funcs.load_epochs_items(subject, cleaned=False)
    metadata = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None,
                                              recompute=True)
    epochs_train.metadata = metadata

    if dobaseline:
        print("--- we baselined the data from %i ms to 0 ----"%(int(epochs_train.tmin*1000)))
        epochs_train.apply_baseline()
        suffix += 'baselined_training_'

    # --- for training: quad sequences without the first 2 and last 2 positions. Bon
    epochs_train = epochs_train["SequenceID == 4 and StimPosition > 2 and StimPosition < 15 and TrialNumber > 10 and ViolationInSequence == 0 "]
    if sliding_window:
        epochs_train = epoching_funcs.sliding_window(epochs_train)
    if decim is not None:
        epochs_train.decimate(decim)


    # ------------------------------------------------------------------------------------------------------------------
    # ---------------- LOAD THE EPOCHS ON FULL SEQUENCES TO TEST THE DECODER  ------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    epochs_full = epoching_funcs.load_epochs_full_sequence(subject,cleaned=False)
    epochs_full = epochs_full["ViolationInSequence == 0 and TrialNumber > 10"]

    if sliding_window:
        epochs_full = epoching_funcs.sliding_window(epochs_full)

    folds_CV = train_test_different_blocks(epochs_full, return_per_seq=True)
    folds_CV = folds_CV[4]

    # ------------------------------------------------------------------------------------------------------------------
    # ----------------  TRAIN AND TEST DOING CROSS-VALIDATION ACROSS RUNS  ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    proj = []

    for cv in range(2):
        print(" fitting and scoring the decoder for the fold number %i"%cv)

        # test data : full sequence
        test_inds =  folds_CV['test'][cv]
        epochs_full_test = epochs_full[test_inds]

        # find the good params to select the training data that corresponds to the complementary of the test data
        inds_for_train = folds_CV['train'][cv]
        epo_train = epochs_full[inds_for_train]
        epochs_all_train = []
        for k in range(len(epo_train)):
            # extract the training run numbers and trial numbers
            run_k = epo_train[k].metadata["RunNumber"].values
            trial_k = epo_train[k].metadata["TrialNumber"].values
            string = "RunNumber == %i and TrialNumber == %i"%(run_k[0],trial_k[0])
            print(string)
            epochs_all_train.append(epochs_train[string])

        # concatenate all the training data into one single epoch object
        epochs_all_train = mne.concatenate_epochs(epochs_all_train)
        if crop is not None:
            epochs_all_train.crop(crop[0], crop[1])

        #  fit on the training data
        SVM_dec.fit(epochs_all_train.get_data(), epochs_all_train.metadata['WithinChunkPosition'].values)
        # test on the full sequence data
        proj.append(SVM_dec.decision_function(epochs_full_test.get_data()))


    save_path = config.result_path+'/SVM/ordinal_code_16items/'+subject+'/'
    utils.create_folder(save_path)
    np.save(save_path+'/'+suffix + 'ordinal_code_quads_tested_quads.npy',{'projection':proj,'times':epochs_full_test.times})




def SVM_ordinal_code_train_quads_test_others(subject,load_residuals_regression=False, SVM_dec=SVM_decoder(),decim=1,sliding_window=True,crop=[0.1,0.3],dobaseline=True):
    """
    Train on the quads sequences (trials coming from the standards of the  test part), remove the first, second, 15th and 16th item of quads.
    Test on all the other sequences. It makes especially sense for 2 pairs, shrink, complex
    :param subject:
    :param load_residuals_regression:
    :param SVM_dec: The classifier type
    :param decim:
    :return:
    """

    ordinal_code_projection_decision_axis = {'SeqID_%i'%i:{} for i in [1,2,3,5,6,7]}

    # ==================================================================================================================
    # --------------------------------------- select the training epochs : quads ---------------------------------------
    # ==================================================================================================================
    suffix = ''
    if load_residuals_regression:
        epochs_train = epoching_funcs.load_resid_epochs_items(subject)
        suffix = 'resid_'
    else:
        epochs_train = epoching_funcs.load_epochs_items(subject, cleaned=False)
        metadata = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None,
                                                  recompute=True)
        epochs_train.metadata = metadata

    if dobaseline:
        print("--- we baselined the data from %i ms to 0 ----"%(int(epochs_train.tmin*1000)))
        epochs_train.apply_baseline()
        suffix += 'baselined_training_'

    # --- for training: quad sequences without the first 2 and last 2 positions. Bon
    epochs_train = epochs_train["SequenceID == 4 and StimPosition > 2 and StimPosition < 15 and TrialNumber > 10 and ViolationInSequence == 0 "]
    if sliding_window:
        epochs_train = epoching_funcs.sliding_window(epochs_train)
    if decim is not None:
        epochs_train.decimate(decim)
    if crop is not None:
        epochs_train.crop(crop[0], crop[1])

    # ==================================================================================================================
    # --------------------------------------------------- Fit the decoder  ---------------------------------------------
    # ==================================================================================================================

    SVM_dec.fit(epochs_train.get_data(), epochs_train.metadata['WithinChunkPosition'].values)

    # ==================================================================================================================
    # --------------------------------- Apply it to the 16 item sequences  ---------------------------------------------
    # ==================================================================================================================

    epochs_test = epoching_funcs.load_epochs_full_sequence(subject,cleaned=False)
    epochs_test = epochs_test["ViolationInSequence == 0 and TrialNumber > 10"]

    if sliding_window:
        epochs_test = epoching_funcs.sliding_window(epochs_test)

    for SeqID in [1,2,3,5,6,7]:
        epochs_test_seq = epochs_test["SequenceID == %i"%SeqID]
        proj = SVM_dec.decision_function(epochs_test_seq.get_data())
        ordinal_code_projection_decision_axis['SeqID_%i'%SeqID] = {'projection':proj,'times':epochs_test_seq.times}

    # ==================================================================================================================
    # ----------------------------------------------- Save it ! --------------------------------------------------------
    # ==================================================================================================================

    save_path = config.result_path+'/SVM/ordinal_code_16items/'+subject+'/'
    utils.create_folder(save_path)
    np.save(save_path+'/'+suffix + 'ordinal_code_quads_tested_others.npy',ordinal_code_projection_decision_axis)


# ______________________________________________________________________________________________________________________
def SVM_decode_feature(subject, feature_name, load_residuals_regression=True, SVM_dec=SVM_decoder(),
                       list_sequences=[1, 2, 3, 4, 5, 6, 7], decim=1, crop=None, cross_val_func=None,
                       balance_features=True, meg=True, eeg=True, distance=True,filter_from_metadata = None):

    """
    Builds an SVM decoder that will be able to output the distance to the hyperplane once trained on data.
    It is meant to generalize across time by construction.
    :return:

    SVM_dec=SVM_decoder()
    subject = 'sub06-kc_160388'
    feature_name = 'ChunkBeginning'
    load_residuals_regression = True
    list_sequences=[3,4,5,6,7]
    crop = [-0.1,0.4]

    decim = 10
    cross_val_func=SVM_funcs.leave_one_sequence_out
    balance_features=True
    meg=True
    eeg=True
    distance = True

    """

    if load_residuals_regression:
        epochs = epoching_funcs.load_resid_epochs_items(subject)
        # metadata = epoching_funcs.update_metadata(subject, clean=True,recompute=True)
    else:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
        metadata = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None,
                                                  recompute=True)
        epochs.metadata = metadata


    epochs = epoching_funcs.sliding_window(epochs,sliding_window_step=2)

    if decim is not None:
        epochs.decimate(decim)
    if crop is not None:
        epochs.crop(crop[0], crop[1])

    # We remove the habituation trials
    epochs = epochs["TrialNumber>10 and ViolationOrNot == 0"]
    if filter_from_metadata is not None:
        epochs = epochs[filter_from_metadata]
    # remove the stim channel from decoding
    epochs.pick_types(meg=meg, eeg=eeg, stim=False)

    print('-- The values of the metadata for the feature %s are : ' % feature_name)
    print(np.unique(epochs.metadata[feature_name].values))

    if balance_features:
        epochs = balance_epochs_for_feature(epochs, feature_name, list_sequences)
    else:
        filter_epochs = np.where(1 - np.isnan(epochs.metadata[feature_name].values))[0]
        epochs = epochs[filter_epochs]
        epochs.events[:, 2] = epochs.metadata[feature_name].values

    scores = []
    dec = []
    y_tests = []
    if cross_val_func is not None:
        X_train, y_train, X_test, y_test = cross_val_func(epochs, list_sequences)
        y_tests.append(y_test)
        n_folds = len(list_sequences)
        for k in range(n_folds):
            SVM_dec.fit(X_train[k], y_train[k])
            scores.append(SVM_dec.score(X_test[k], y_test[k]))
            if distance:
                dec.append(SVM_dec.decision_function(X_test[k]))
    else:
        kf = KFold(n_splits=4)
        y = epochs.events[:, 2]
        X = epochs._data
        nfold = 1
        for train_index, test_index in kf.split(X):
            print("fold number %i"%nfold)
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            SVM_dec.fit(X_train, y_train)
            scores.append(SVM_dec.score(X_test, y_test))
            y_tests.append(y_test)
            if distance:
                dec.append(SVM_dec.decision_function(X_test))
            nfold += 1
    score = np.mean(scores, axis=0)

    dec = np.vstack(dec)
    y_tests =  np.vstack(y_tests)
    times = epochs.times
    results_dict = {'score':score,'times':times,'y_test':y_tests,'distance':dec}

    return results_dict


def balance_epochs_for_feature(epochs, feature_name, list_sequences):
    # concatenate the epochs belonging to the different sequences from the list_sequences
    epochs_concat1 = []
    # count the number of epochs that contribute per sequence in order later to balance this
    n_epochs = []
    for seqID in list_sequences:
        epo = epochs["SequenceID == " + str(seqID)].copy()
        # ---- remove the epochs that have nan values for the feature we are decoding ----
        filter_epochs = np.where(1 - np.isnan(epo.metadata[feature_name].values))[0]
        epo = epo[filter_epochs]
        epo.events[:, 2] = epo.metadata[feature_name].values
        epo.event_id = {'%i' % i: i for i in np.unique(epo.events[:, 2])}
        epo.equalize_event_counts(epo.event_id)
        n_epochs.append(len(epo))
        print("---- there are %i epochs that contribute from sequence %i -----" % (len(epo), seqID))
        epochs_concat1.append(epo)
    # now determine the minimum number of epochs that come from a sequence and append the same number of epochs from each
    # sequence type
    epochs_concat2 = []
    n_min = np.min(n_epochs)
    n_max = np.max(n_epochs)
    if n_min != n_max:
        for k in range(len(list_sequences)):
            n_epo_seq = len(epochs_concat1[k])
            inds = np.random.permutation(n_epo_seq)
            inds = inds[:n_min]
            epochs_concat2.append(mne.concatenate_epochs([epochs_concat1[k][i] for i in inds]))
    else:
        epochs_concat2 = epochs_concat1
    epochs = mne.concatenate_epochs(epochs_concat2)
    return epochs


# ______________________________________________________________________________________________________________________
def generate_SVM_all_sequences(subject, load_residuals_regression=False, train_different_blocks=True,
                               sliding_window=False):
    """
    Generates the SVM decoders for all the channel types using 4 folds. We save the training and testing indices as well as the epochs
    in order to be flexible for the later analyses.

    :param epochs:
    :param saving_directory:
    :return:
    """

    # ----------- set the directories ----------
    saving_directory = op.join(config.SVM_path, subject)
    utils.create_folder(saving_directory)

    # ----------- load the epochs ---------------
    suf = ''
    if load_residuals_regression:
        epochs = epoching_funcs.load_resid_epochs_items(subject)
        epochs.pick_types(meg=True, eeg=True, stim=False)
        suf = 'resid_'
    else:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
        epochs.pick_types(meg=True, eeg=True, stim=False)

    # ----------- balance the position of the standard and the deviants -------
    epochs_balanced = epoching_funcs.balance_epochs_violation_positions(epochs, balance_violation_standards=True)

    # ----------- do a sliding window to smooth the data if neeeded -------
    if sliding_window:
        epochs_balanced = epoching_funcs.sliding_window(epochs_balanced)
        suf += 'SW_'

    # =============================================================================================
    epochs_balanced_mag = epochs_balanced.copy().pick_types(meg='mag')
    epochs_balanced_grad = epochs_balanced.copy().pick_types(meg='grad')
    epochs_balanced_eeg = epochs_balanced.copy().pick_types(eeg=True, meg=False)
    epochs_balanced_all_chans = epochs_balanced.copy().pick_types(eeg=True, meg=True)

    # ==============================================================================================
    y_violornot = np.asarray(epochs_balanced.metadata['ViolationOrNot'].values)
    epochs_all = [epochs_balanced_mag, epochs_balanced_grad, epochs_balanced_eeg, epochs_balanced_all_chans]
    sensor_types = ['mag', 'grad', 'eeg', 'all_chans']
    SVM_results = {'mag': [], 'grad': [], 'eeg': [], 'all_chans': []}

    for l, senso in enumerate(sensor_types):
        epochs_senso = epochs_all[l]
        X_data = epochs_senso.get_data()

        All_SVM = []

        if train_different_blocks:
            training_inds , testing_inds = train_test_different_blocks(epochs_senso,return_per_seq = False)
            for k in range(2):
                SVM_dec = SVM_decoder()
                SVM_dec.fit(X_data[training_inds[k],:,:], y_violornot[training_inds[k]])
                All_SVM.append(SVM_dec)
        else:
            training_inds = []
            testing_inds = []
            metadata_epochs = epochs_balanced.metadata
            y_tmp = [
                int(metadata_epochs['SequenceID'].values[i] * 1000 + metadata_epochs['StimPosition'].values[i] * 10 +
                    metadata_epochs['ViolationOrNot'].values[i]) for i in range(len(epochs_balanced))]
            for train, test in StratifiedKFold(n_splits=4).split(X_data, y_tmp):
                SVM_dec = SVM_decoder()
                SVM_dec.fit(X_data[train], y_violornot[train])
                All_SVM.append(SVM_dec)
                training_inds.append(train)
                testing_inds.append(test)

        SVM_results[senso] = {'SVM': All_SVM, 'train_ind': training_inds, 'test_ind': testing_inds,
                              'epochs': epochs_all[l]}

    if train_different_blocks:
        suf += 'train_different_blocks'
    np.save(op.join(saving_directory, suf + 'SVM_results.npy'), SVM_results)


# ______________________________________________________________________________________________________________________
def generate_SVM_separate_sequences(subject, load_residuals_regression=False, train_different_blocks=True,
                               sliding_window=False):
    """
    Generates the SVM decoders for all the channel types using 4 folds. We save the training and testing indices as well as the epochs
    in order to be flexible for the later analyses.

    :param epochs:
    :param saving_directory:
    :return:
    """

    # ----------- set the directories ----------
    saving_directory = op.join(config.SVM_path, subject)
    utils.create_folder(saving_directory)

    # ----------- load the epochs ---------------
    suf = ''
    if load_residuals_regression:
        epochs = epoching_funcs.load_resid_epochs_items(subject)
        epochs.pick_types(meg=True, eeg=True, stim=False)
        suf = 'resid_'
    else:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
        epochs.pick_types(meg=True, eeg=True, stim=False)

    # ----------- balance the position of the standard and the deviants -------
    epochs_balanced = epoching_funcs.balance_epochs_violation_positions(epochs)

    # ----------- do a sliding window to smooth the data if neeeded -------
    if sliding_window:
        epochs_balanced = epoching_funcs.sliding_window(epochs_balanced)
        suf += 'SW_'
    # =============================================================================================
    epochs_balanced_mag = epochs_balanced.copy().pick_types(meg='mag')
    epochs_balanced_grad = epochs_balanced.copy().pick_types(meg='grad')
    epochs_balanced_eeg = epochs_balanced.copy().pick_types(eeg=True, meg=False)
    epochs_balanced_all_chans = epochs_balanced.copy().pick_types(eeg=True, meg=True)

    # ==============================================================================================
    y_violornot = np.asarray(epochs_balanced.metadata['ViolationOrNot'].values)
    epochs_all = [epochs_balanced_mag, epochs_balanced_grad, epochs_balanced_eeg, epochs_balanced_all_chans]
    sensor_types = ['mag', 'grad', 'eeg', 'all_chans']
    SVM_results = {senso:{i:{'SVM':[],'train_inds':[],'test_inds':[],'epochs_seq':[]} for i in range(1,8)} for senso in sensor_types}

    for l, senso in enumerate(sensor_types):
        epochs_senso = epochs_all[l]
        X_data = epochs_senso.get_data()

        train_test_dict = train_test_different_blocks(epochs_senso, return_per_seq=True)
        for seqID in range(1,8):
            train_inds = train_test_dict[seqID]['train']
            test_inds = train_test_dict[seqID]['test']
            SVM_results[senso][seqID]['train_inds'] = train_test_dict[seqID]['train']
            SVM_results[senso][seqID]['test_inds'] = train_test_dict[seqID]['test']
            SVM_results[senso][seqID]['epochs_seq_train'] = [epochs_senso[train_inds[k]] for k in range(2)]
            SVM_results[senso][seqID]['epochs_seq_test'] = [epochs_senso[test_inds[k]] for k in range(2)]

            for k in range(2):
                SVM_dec = SVM_decoder()
                SVM_dec.fit(X_data[train_test_dict[seqID]['train'][k]], y_violornot[train_test_dict[seqID]['train'][k]])
                SVM_results[senso][seqID]['SVM'].append(SVM_dec)

    if train_different_blocks:
        suf += 'train_different_blocks_and_sequences'
    np.save(op.join(saving_directory, suf + 'SVM_results.npy'), SVM_results)



# ______________________________________________________________________________________________________________________
def GAT_SVM_trained_all_sequences(subject, load_residuals_regression=False, train_different_blocks=True,
            sliding_window=False):
    """
    The SVM at a training times are tested at testing times. Allows to obtain something similar to the GAT from decoding.
    Dictionnary contains the GAT for each sequence separately. GAT_all contains the average over all the sequences
    :param SVM_results: output of generate_SVM_all_sequences
    :return: GAT averaged over the 4 classification folds
    """

    saving_directory = op.join(config.SVM_path, subject)

    # ----- build the right suffix to load the correct matrix -----
    suf = ''
    if sliding_window:
        suf += 'SW_'
    if load_residuals_regression:
        suf = 'resid_'
    if train_different_blocks:
        suf += 'train_different_blocks'
        n_folds = 2
    else:
        n_folds = 4

    # ---------- load the data ------------
    SVM_results = np.load(op.join(saving_directory, suf + 'SVM_results.npy'), allow_pickle=True).item()

    # ----- initialize the results dictionnary ------
    GAT_sens_seq = {sens: [] for sens in ['eeg', 'mag', 'grad', 'all_chans']}

    for sens in ['eeg', 'mag', 'grad', 'all_chans']:
        GAT_all = []
        GAT_per_sens_and_seq = {'SeqID_%i' % i: [] for i in range(1, 8)}

        epochs_sens = SVM_results[sens]['epochs']
        n_times = epochs_sens.get_data().shape[-1]
        SVM_sens = SVM_results[sens]['SVM']

        for sequence_number in range(1, 8):
            seqID = 'SeqID_%i' % sequence_number
            GAT_seq = np.zeros((n_folds, n_times, n_times))

            for fold_number in range(n_folds):

                test_indices = SVM_results[sens]['test_ind'][fold_number]
                epochs_sens_test = epochs_sens[test_indices]
                epochs_sens_and_seq_test = epochs_sens_test["SequenceID == %i"%sequence_number]
                y_sens_and_seq_test = epochs_sens_and_seq_test.metadata["ViolationOrNot"].values

                GAT_seq[fold_number,:,:] = SVM_sens[fold_number].score(epochs_sens_and_seq_test.get_data(),y_sens_and_seq_test)

                # inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == sequence_number) & (
                #         epochs_sens_test.metadata['ViolationOrNot'].values == 0))[0]
                # inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == sequence_number) & (
                #         epochs_sens_test.metadata['ViolationOrNot'].values == 1))[0]
                # X = epochs_sens_test.get_data()

                # if score_or_decisionfunc == 'score':
                #     GAT_each_epoch = SVM_sens[fold_number].predict(X)
                # else:
                #     GAT_each_epoch = SVM_sens[fold_number].decision_function(X)
                #
                # GAT_seq[fold_number, :, :] = np.mean(
                #     GAT_each_epoch[inds_seq_noviol, :, :], axis=0) - np.mean(
                #     GAT_each_epoch[inds_seq_viol, :, :], axis=0)
                # print('The shape of GAT_seq[fold_number, :, :] is')
                # print(GAT_seq[fold_number, :, :].shape)

            #  --------------- now average across the folds ---------------
            GAT_seq_avg = np.mean(GAT_seq, axis=0)
            GAT_per_sens_and_seq[seqID] = GAT_seq_avg
            GAT_all.append(GAT_seq_avg)

        GAT_sens_seq[sens] = GAT_per_sens_and_seq
        GAT_sens_seq[sens]['average_all_sequences'] = np.mean(GAT_all, axis=0)
        times = epochs_sens_test.times

    GAT_results = {'GAT': GAT_sens_seq, 'times': times}
    np.save(op.join(saving_directory, suf + 'GAT_results.npy'), GAT_results)



# ______________________________________________________________________________________________________________________
def GAT_SVM_trained_separate_sequences(subject, load_residuals_regression=False,
            sliding_window=False):
    """
    The SVM at a training times are tested at testing times. Allows to obtain something similar to the GAT from decoding.
    Dictionnary contains the GAT for each sequence separately. GAT_all contains the average over all the sequences
    :param SVM_results: output of generate_SVM_all_sequences
    :return: GAT averaged over the 4 classification folds
    """

    saving_directory = op.join(config.SVM_path, subject)

    # ----- build the right suffix to load the correct matrix -----
    suf = ''
    if sliding_window:
        suf += 'SW_'
    if load_residuals_regression:
        suf = 'resid_'
    suf += 'train_different_blocks_and_sequences'
    n_folds = 2

    # ---------- load the data ------------
    SVM_results = np.load(op.join(saving_directory, suf + 'SVM_results.npy'), allow_pickle=True).item()

    # ----- initialize the results dictionnary ------
    GAT_sens_seq = {sens: [] for sens in ['eeg', 'mag', 'grad', 'all_chans']}

    for sens in ['eeg', 'mag', 'grad', 'all_chans']:
        GAT_all = []
        GAT_per_sens_and_seq = {'SeqID_%i' % i: [] for i in range(1, 8)}

        for sequence_number in range(1, 8):

            epochs_sens_and_seq_test = SVM_results[sens][sequence_number]['epochs_seq_test']
            n_times = epochs_sens_and_seq_test[0].get_data().shape[-1]
            SVM_sens = SVM_results[sens][sequence_number]['SVM']

            seqID = 'SeqID_%i' % sequence_number
            GAT_seq = np.zeros((n_folds, n_times, n_times))

            for fold_number in range(n_folds):
                y_sens_and_seq_test = epochs_sens_and_seq_test[fold_number].metadata["ViolationOrNot"].values
                GAT_seq[fold_number,:,:] = SVM_sens[fold_number].score(epochs_sens_and_seq_test[fold_number].get_data(),y_sens_and_seq_test)

                # inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == sequence_number) & (
                #         epochs_sens_test.metadata['ViolationOrNot'].values == 0))[0]
                # inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == sequence_number) & (
                #         epochs_sens_test.metadata['ViolationOrNot'].values == 1))[0]
                # X = epochs_sens_test.get_data()

                # if score_or_decisionfunc == 'score':
                #     GAT_each_epoch = SVM_sens[fold_number].predict(X)
                # else:
                #     GAT_each_epoch = SVM_sens[fold_number].decision_function(X)
                #
                # GAT_seq[fold_number, :, :] = np.mean(
                #     GAT_each_epoch[inds_seq_noviol, :, :], axis=0) - np.mean(
                #     GAT_each_epoch[inds_seq_viol, :, :], axis=0)
                # print('The shape of GAT_seq[fold_number, :, :] is')
                # print(GAT_seq[fold_number, :, :].shape)

            #  --------------- now average across the folds ---------------
            GAT_seq_avg = np.mean(GAT_seq, axis=0)
            GAT_per_sens_and_seq[seqID] = GAT_seq_avg
            GAT_all.append(GAT_seq_avg)

        GAT_sens_seq[sens] = GAT_per_sens_and_seq
        GAT_sens_seq[sens]['average_all_sequences'] = np.mean(GAT_all, axis=0)
        times = epochs_sens_and_seq_test[0].times

    GAT_results = {'GAT': GAT_sens_seq, 'times': times}
    np.save(op.join(saving_directory, suf + 'GAT_results.npy'), GAT_results)


# ______________________________________________________________________________________________________________________
def GAT_SVM_4pos(subject, load_residuals_regression=False, score_or_decisionfunc='score',
                 train_test_different_blocks=True):
    """
    The SVM at a training times are tested at testing times. Allows to obtain something similar to the GAT from decoding.
    Dictionnary contains the GAT for each sequence separately and for each violation position.
    The difference with the previous function is that it tests only the positions that could be violated for a given sequence.
    GAT_all contains the average over all the sequences
    :param SVM_results: output of generate_SVM_all_sequences
    :return: GAT averaged over the 4 classification folds
    """

    n_folds = 4
    suf = ''
    if load_residuals_regression:
        suf = 'resid_'

    if train_test_different_blocks:
        n_folds = 2
        suf += 'train_test_different_blocks'

    saving_directory = op.join(config.SVM_path, subject)
    SVM_results = np.load(op.join(saving_directory, suf + 'SVM_results.npy'), allow_pickle=True).item()

    GAT_sens_seq = {sens: [] for sens in ['eeg', 'mag', 'grad', 'all_chans']}

    for sens in ['eeg', 'mag', 'grad', 'all_chans']:
        print(sens)
        GAT_all = []
        GAT_per_sens_and_seq = {'SeqID_%i' % i: [] for i in range(1, 8)}
        epochs_sens = SVM_results[sens]['epochs']
        n_times = epochs_sens.get_data().shape[-1]
        SVM_sens = SVM_results[sens]['SVM']

        for k in range(1, 8):
            seqID = 'SeqID_%i' % k
            # extract the 4 positions of violation
            violpos_list = np.unique(epochs_sens['SequenceID == %i' % k].metadata['ViolationInSequence'])[1:]
            GAT_seq = np.zeros((4, n_times, n_times, 4))
            for fold_number in range(n_folds):
                print("====== running for fold number %i =======\n" % fold_number)
                test_indices = SVM_results[sens]['test_ind'][fold_number]
                epochs_sens_test = epochs_sens[test_indices]
                X = epochs_sens_test.get_data()
                if score_or_decisionfunc == 'score':
                    GAT_each_epoch = SVM_sens[fold_number].predict(X)
                else:
                    GAT_each_epoch = SVM_sens[fold_number].decision_function(X)
                for nn, pos_viol in enumerate(violpos_list):
                    print("===== RUNNING for SEQ %i and position violation %i" % (k, nn))
                    inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                            epochs_sens_test.metadata['ViolationOrNot'].values == 0) & (
                                                       epochs_sens_test.metadata['StimPosition'].values == pos_viol))[0]
                    inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                            epochs_sens_test.metadata['ViolationOrNot'].values == 1) & (
                                                     epochs_sens_test.metadata['StimPosition'].values == pos_viol))[0]
                    GAT_seq[fold_number, :, :, nn] = np.mean(
                        GAT_each_epoch[inds_seq_noviol, :, :], axis=0) - np.mean(
                        GAT_each_epoch[inds_seq_viol, :, :], axis=0)
                    print("======== finished the loop ==========")
            # now average across the 4 folds
            GAT_seq_avg = np.mean(GAT_seq, axis=0)
            GAT_per_sens_and_seq[seqID] = GAT_seq_avg
            GAT_all.append(GAT_seq_avg)

        print('coucou1')
        GAT_sens_seq[sens] = GAT_per_sens_and_seq
        print('coucou2')
        GAT_sens_seq[sens]['average_all_sequences'] = np.mean(GAT_all, axis=0)
        times = epochs_sens_test.times

    print('coucou3')
    GAT_results = {'GAT': GAT_sens_seq, 'times': times}
    if score_or_decisionfunc == 'score':
        np.save(op.join(saving_directory, suf + 'GAT_results_4pos_score.npy'), GAT_results)
    else:
        np.save(op.join(saving_directory, suf + 'GAT_results_4pos.npy'), GAT_results)
    print(" ======== job done ============")


# ______________________________________________________________________________________________________________________
def SVM_applied_to_epochs(SVM_results, sequenceID=None):
    """
    This function applies the spatial filters to the epochs for all the sequence IDS or just some of them.
    It returns the projection, for each sensor type, and the codes indicating if it was violated or not, for each sensor type.
    The third thing returned are the times.

    :param SVM_results:
    :param sequenceID:
    :return:
    """

    y_violornot = {'eeg': [], 'grad': [], 'mag': [], 'all_chans': []}
    X_transform = {'eeg': [], 'grad': [], 'mag': [], 'all_chans': []}

    for sens in {'eeg', 'grad', 'mag', 'all_chans'}:
        print(sens)
        SVM_sens = SVM_results[sens]['SVM']
        epochs_sens = SVM_results[sens]['epochs']
        n_epochs, n_channels, n_times = epochs_sens.get_data().shape
        X_transform[sens] = np.zeros((n_epochs, n_times))
        for fold_number in range(4):
            test_indices = SVM_results[sens]['test_ind'][fold_number]
            if sequenceID is not None:
                epochs_sens_test = epochs_sens[test_indices]['SequenceID == %i' % sequenceID]
            else:
                epochs_sens_test = epochs_sens[test_indices]

            y_violornot[sens].append(np.asarray(epochs_sens.metadata['ViolationOrNot'].values))
            X = epochs_sens_test.get_data()
            X_transform[sens][test_indices] = SVM_sens[fold_number].decision_function(X)

    times = epochs_sens_test.times

    return X_transform, y_violornot, times


# ______________________________________________________________________________________________________________________
def plot_GAT_SVM(GAT_avg,chance,times, sens='mag', save_path=None, figname='GAT_', vmin=None, vmax=None,sig=None):

    pretty_gat(-GAT_avg,chance=chance,times=times,sig=sig,clim=[vmin,vmax])

    #
    # minT = np.min(times) * 1000
    # maxT = np.max(times) * 1000
    # fig = plt.figure()
    # plt.imshow(-GAT_avg, origin='lower', extent=[minT, maxT, minT, maxT], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    # # -----# ADD LINES ?
    # plt.axvline(0, linestyle='-', color='black', linewidth=1)
    # plt.axhline(0, linestyle='-', color='black', linewidth=1)
    # plt.plot([minT, maxT], [minT, maxT], linestyle='--', color='black', linewidth=1)
    # # -----#
    # plt.ylabel('Training time (ms)')  # NO TRANSPOSE
    # plt.xlabel('Testing time (ms)')  # NO TRANSPOSE
    # plt.colorbar()
    if save_path is not None:
        plt.savefig(op.join(save_path, figname + sens), dpi=300)

    return plt.gcf()


# ______________________________________________________________________________________________________________________
def plot_SVM_average(X_transform, y_violornot, times, fig_path=None, figname='Average_SVM_all_sequences_'):
    fig = plt.figure()
    plt.title('Average SVM signal - ')
    plt.axhline(0, linestyle='-', color='black', linewidth=1)
    plt.plot(times, X_transform[y_violornot == 0].mean(0), label="standard", linestyle='--', linewidth=0.5)
    plt.plot(times, X_transform[y_violornot == 1].mean(0), label="deviant", linestyle='--', linewidth=0.5)
    plt.plot(times, X_transform[y_violornot == 0].mean(0) - X_transform[y_violornot == 1].mean(0),
             label="standard vs deviant")
    plt.xlabel('Time (ms)')
    plt.ylabel('a.u.')
    plt.legend(loc='best')
    plt.show()
    if fig_path is not None:
        plt.savefig(fig_path + figname)
        plt.close('all')

    return fig


# ______________________________________________________________________________________________________________________
def plot_single_trials(X_transform, y_violornot, times, fig_path=None, figname='all_trials'):
    fig = plt.figure()
    plt.title('single trial surrogates')
    plt.imshow(X_transform[y_violornot.argsort()], origin='lower', aspect='auto',
               extent=[times[0], times[-1], 1, len(X_transform)],
               cmap='RdBu_r', vmin=-20, vmax=20)
    plt.colorbar()
    plt.xlabel('Time (ms)')
    plt.ylabel('Trials (reordered by condition)')
    plt.show()

    if fig_path is not None:
        plt.savefig(fig_path + figname)
        plt.close('all')

    return fig


# ______________________________________________________________________________________________________________________
def apply_SVM_filter_16_items_epochs(subject, times=[x / 1000 for x in range(0, 750, 50)], window=False,
                                     train_test_different_blocks=True, sliding_window=False):
    """
    Function to apply the SVM filters built on all the sequences the 16 item sequences
    :param subject:
    :param times:the different times at which we want to apply the filter (if window is False). Otherwise (window = True),
    min(times) and max(times) define the time window on which we average the spatial filter.
    :param window: set to True if you want to average the spatial filter over a window.
    :return:
    """

    # ==== load the ems results ==============
    SVM_results_path = op.join(config.SVM_path, subject)
    suf = ''
    n_folds = 4

    if sliding_window:
        suf += 'SW_'

    if train_test_different_blocks:
        n_folds = 2
        suf += 'train_test_different_blocks'

    SVM_results = np.load(op.join(SVM_results_path, suf + 'SVM_results.npy'), allow_pickle=True).item()

    # ==== define the paths ==============
    meg_subject_dir = op.join(config.meg_dir, subject)
    fig_path = op.join(config.study_path, 'Figures', 'SVM') + op.sep
    extension = subject + '_1st_element_epo'
    fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Input: ", fname_in)

    # ====== loading the 16 items sequences epoched on the first element ===================
    epochs_1st_element = mne.read_epochs(fname_in, preload=True)

    if sliding_window:
        epochs_1st_element = epoching_funcs.sliding_window(epochs_1st_element)

    epochs_1st_element = epochs_1st_element["TrialNumber > 10"]
    epochs_1st = {'mag': epochs_1st_element.copy().pick_types(meg='mag'),
                  'grad': epochs_1st_element.copy().pick_types(meg='grad'),
                  'eeg': epochs_1st_element.copy().pick_types(eeg=True, meg=False),
                  'all_chans': epochs_1st_element.copy().pick_types(eeg=True, meg=True)}

    # ====== compute the projections for each of the 3 types of sensors ===================
    for sens in ['mag', 'grad', 'eeg', 'all_chans']:

        print(sens)
        SVM_sens = SVM_results[sens]['SVM']
        epochs_sens = SVM_results[sens]['epochs']
        epochs_1st_sens = epochs_1st[sens]

        # = we initialize the metadata
        data_frame_meta = pd.DataFrame([])
        elapsed = 0
        data_for_epoch_object = np.zeros(
            (epochs_sens.get_data().shape[0] * len(times), epochs_1st_sens.get_data().shape[2]))
        if window:
            data_for_epoch_object = np.zeros(
                (SVM_results[sens]['epochs'].get_data().shape[0], epochs_1st_sens.get_data().shape[2]))

        print("The shape of data_for_epoch_object is ")
        print(data_for_epoch_object.shape)
        # ===============================
        counter = 0
        for fold_number in range(n_folds):
            print('Fold ' + str(fold_number + 1) + ' on %i...' % n_folds)
            start = time.time()
            test_indices = SVM_results[sens]['test_ind'][fold_number]
            epochs_sens_test = epochs_sens[test_indices]
            points = epochs_sens_test.time_as_index(times)

            for m in test_indices:

                seqID_m = epochs_sens[m].metadata['SequenceID'].values[0]
                run_m = epochs_sens[m].metadata['RunNumber'].values[0]
                trial_number_m = epochs_sens[m].metadata['TrialNumber'].values[
                    0]  # this is the number of the trial, that will allow to determine which sequence within the run of 46 is the one that was left apart
                epochs_1st_sens_m = epochs_1st_sens[
                    'SequenceID == "%i" and RunNumber == %i and TrialNumber == %i' % (seqID_m, run_m, trial_number_m)]

                # if sens =="all_chans":
                #    epochs_1st_sens_m.pick_types(meg=True,eeg=True)

                if len(epochs_1st_sens_m.events) != 0:
                    data_1st_el_m = epochs_1st_sens_m.get_data()
                    SVM_to_data = np.squeeze(SVM_sens[fold_number].decision_function(data_1st_el_m))
                    print("The shape of SVM_to_data is ")
                    print(SVM_to_data.shape)
                    if not window:
                        for mm, point_of_interest in enumerate(points):
                            print(" The point of interest has index %i" % point_of_interest)
                            print(
                                " === MAKE SURE THAT WHEN SELECTING SVM_to_data[point_of_interest,:] WE ARE INDEED CHOOSING THE TRAINING TIMES ===")
                            epochs_1st_sens_m_filtered_data = SVM_to_data[point_of_interest, :]
                            print('epochs_1st_sens_m_filtered_data has shape')
                            print(epochs_1st_sens_m_filtered_data.shape)
                            print("data_for_epoch_object has shape")
                            print(data_for_epoch_object.shape)
                            data_for_epoch_object[counter, :] = np.squeeze(epochs_1st_sens_m_filtered_data)

                            metadata_m = epochs_1st_sens_m.metadata
                            metadata_m['SVM_filter_datapoint'] = int(point_of_interest)
                            metadata_m['SVM_filter_time'] = times[mm]
                            data_frame_meta = data_frame_meta.append(metadata_m)
                            counter += 1
                    else:
                        print(
                            " === MAKE SURE THAT WHEN SELECTING SVM_to_data[np.min(points):np.max(points),:] WE ARE INDEED CHOOSING THE TRAINING TIMES ===")
                        print(SVM_to_data.shape)
                        epochs_1st_sens_m_filtered_data = np.mean(SVM_to_data[np.min(points):np.max(points), :], axis=0)
                        print(epochs_1st_sens_m_filtered_data.shape)
                        print('This was the shape of epochs_1st_sens_m_filtered_data')

                        data_for_epoch_object[counter, :] = np.squeeze(epochs_1st_sens_m_filtered_data)
                        metadata_m = epochs_1st_sens_m.metadata
                        metadata_m['SVM_filter_min_datapoint'] = np.min(points)
                        metadata_m['SVM_filter_max_datapoint'] = np.max(points)
                        metadata_m['SVM_filter_tmin_window'] = times[0]
                        metadata_m['SVM_filter_tmax_window'] = times[-1]
                        data_frame_meta = data_frame_meta.append(metadata_m)
                        counter += 1

                else:
                    print(
                        '========================================================================================================================================')
                    print(
                        ' Epoch on first element for sequence %s Run number %i and Trial number %i was excluded by autoreject' % (
                        seqID_m, run_m, trial_number_m))
                    print(
                        '========================================================================================================================================')
            end = time.time()
            elapsed = end - start
            print('... lasted: ' + str(elapsed) + ' s')

        dat = np.expand_dims(data_for_epoch_object, axis=1)
        info = mne.create_info(['SVM'], epochs_1st_sens.info['sfreq'])
        epochs_proj_sens = mne.EpochsArray(dat, info, tmin=-0.5)
        epochs_proj_sens.metadata = data_frame_meta

        if window:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + suf + '_SVM_on_16_items_test_window-epo.fif',
                                  overwrite=True)
        else:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + suf + '_SVM_on_16_items_test-epo.fif',
                                  overwrite=True)

    return True


def apply_SVM_filter_16_items_epochs_habituation(subject, times=[x / 1000 for x in range(0, 750, 50)], window=False,
                                                 train_test_different_blocks=True, sliding_window=False):
    """
    Function to apply the SVM filters on the habituation trials. It is simpler than the previous function as we don't have to select the specific
    trials according to the folds.
    :param subject:
    :param times:
    :return:
    """

    # ==== load the ems results ==============
    SVM_results_path = op.join(config.SVM_path, subject)
    suf = ''
    n_folds = 4

    if sliding_window:
        suf += 'SW_'

    if train_test_different_blocks:
        n_folds = 2
        suf += 'train_test_different_blocks'

    SVM_results = np.load(op.join(SVM_results_path, suf + 'SVM_results.npy'), allow_pickle=True).item()

    # ==== define the paths ==============
    meg_subject_dir = op.join(config.meg_dir, subject)
    fig_path = op.join(config.study_path, 'Figures', 'SVM') + op.sep
    extension = subject + '_1st_element_epo'
    fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Input: ", fname_in)

    # ====== loading the 16 items sequences epoched on the first element ===================
    epochs_1st_element = mne.read_epochs(fname_in, preload=True)
    if sliding_window:
        epochs_1st_element = epoching_funcs.sliding_window(epochs_1st_element)
    epochs_1st_element = epochs_1st_element["TrialNumber < 11"]
    epochs_1st = {'mag': epochs_1st_element.copy().pick_types(meg='mag'),
                  'grad': epochs_1st_element.copy().pick_types(meg='grad'),
                  'eeg': epochs_1st_element.copy().pick_types(eeg=True, meg=False),
                  'all_chans': epochs_1st_element.copy().pick_types(eeg=True, meg=True)}

    # ====== compute the projections for each of the 3 types of sensors ===================
    for sens in ['all_chans', 'mag', 'grad', 'eeg']:
        # for sens in ['all_chans','mag', 'grad', 'eeg']:

        SVM_sens = SVM_results[sens]['SVM']
        points = SVM_results[sens]['epochs'][0].time_as_index(times)

        epochs_1st_sens = epochs_1st[sens]

        # = we initialize the metadata
        data_frame_meta = pd.DataFrame([])
        n_habituation = epochs_1st_element.get_data().shape[0]
        data_for_epoch_object = np.zeros(
            (n_habituation * len(times), epochs_1st_sens.get_data().shape[2]))
        if window:
            data_for_epoch_object = np.zeros(
                (n_habituation, epochs_1st_sens.get_data().shape[2]))

        # ========== les 4 filtres peuvent etre appliquees aux sequences d habituation sans souci, selection en fonction des indices ========
        data_1st_el_m = epochs_1st_sens.get_data()
        if not window:
            for mm, point_of_interest in enumerate(points):
                epochs_1st_sens_filtered_data_4folds = []
                for fold_number in range(n_folds):
                    SVM_to_data = np.squeeze(SVM_sens[fold_number].decision_function(data_1st_el_m))
                    print("The shape of SVM_to_data is ")
                    print(SVM_to_data.shape)
                    epochs_1st_sens_filtered_data_4folds.append(SVM_to_data[point_of_interest, :])

                    # ==== now that we projected the 4 filters, we can average over the 4 folds ================
                epochs_1st_sens_filtered_data = np.mean(epochs_1st_sens_filtered_data_4folds, axis=0).T
                data_for_epoch_object[n_habituation * mm:n_habituation * (mm + 1), :] = epochs_1st_sens_filtered_data
                metadata_m = epochs_1st_sens.metadata
                metadata_m['SVM_filter_datapoint'] = int(point_of_interest)
                metadata_m['SVM_filter_time'] = times[mm]
                data_frame_meta = data_frame_meta.append(metadata_m)
        else:
            epochs_1st_sens_filtered_data_4folds = []
            for fold_number in range(n_folds):
                SVM_to_data = np.squeeze(SVM_sens[fold_number].decision_function(data_1st_el_m))
                print("The shape of SVM_to_data is ")
                print(SVM_to_data.shape)
                print(
                    " === MAKE SURE THAT WHEN SELECTING SVM_to_data[point_of_interest,:] WE ARE INDEED CHOOSING THE TRAINING TIMES ===")
                epochs_1st_sens_filtered_data_4folds.append(
                    np.mean(SVM_to_data[:, np.min(points):np.max(points), :], axis=1))

            # ==== now that we projected the 4 filters, we can average over the 4 folds ================
            data_for_epoch_object = np.mean(epochs_1st_sens_filtered_data_4folds, axis=0)

            metadata = epochs_1st_sens.metadata
            print("==== the length of the epochs_1st_sens.metadata to append is %i ====" % len(metadata))
            metadata['SVM_filter_min_datapoint'] = np.min(points)
            metadata['SVM_filter_max_datapoint'] = np.max(points)
            metadata['SVM_filter_tmin_window'] = times[0]
            metadata['SVM_filter_tmax_window'] = times[-1]
            data_frame_meta = data_frame_meta.append(metadata)

        dat = np.expand_dims(data_for_epoch_object, axis=1)
        info = mne.create_info(['SVM'], epochs_1st_sens.info['sfreq'])
        epochs_proj_sens = mne.EpochsArray(dat, info, tmin=-0.5)
        print("==== the total number of epochs is %i ====" % len(epochs_proj_sens))
        print("==== the total number of metadata fields is %i ====" % len(data_frame_meta))

        epochs_proj_sens.metadata = data_frame_meta
        if window:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + suf + '_SVM_on_16_items_habituation_window-epo.fif',
                                  overwrite=True)
        else:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + suf + '_SVM_on_16_items_habituation-epo.fif',
                                  overwrite=True)

    return True


# ______________________________________________________________________________________________________________________
def plot_SVM_projection_for_seqID(epochs_list, sensor_type, seqID=1,
                                  SVM_filter_times=[x / 1000 for x in range(100, 700, 50)], save_path=None,
                                  color_mean=None, plot_noviolation=True):
    """
    This will allow to plot the Projections of the SVM on the 4 positions of the violations.

    :param epochs:
    :param sensor_type:
    :param seqID:
    :param SVM_filter_times:
    :return:
    """

    # this provides us with the position of the violations and the times
    epochs_seq_subset = epochs_list[0]['SequenceID == "' + str(seqID) + '"']
    times = epochs_seq_subset.times
    violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])[1 - 1 * plot_noviolation:]

    fig, axes = plt.subplots(4 + 1 * plot_noviolation, 1, figsize=(12, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    fig.suptitle('SVM %s - SequenceID_' % sensor_type + str(seqID) + ' N subjects = ' + str(len(epochs_list)),
                 fontsize=12)
    ax = axes.ravel()[::1]
    n = 0
    for viol_pos in violpos_list:
        for ii, point_of_interest in enumerate(SVM_filter_times):
            y_list = []
            for epochs in epochs_list:
                epochs_subset = epochs['SequenceID == "' + str(seqID)
                                       + '" and SVM_filter_time == "' + str(point_of_interest)
                                       + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='SVM').get_data()))

            mean = np.mean(y_list, axis=0)
            ub = mean + sem(y_list, axis=0)
            lb = mean - sem(y_list, axis=0)

            for xx in range(16):
                ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            ax[n].fill_between(times * 1000, ub, lb, color=color_mean[ii], alpha=.2)
            ax[n].plot(times * 1000, mean, color=color_mean[ii], linewidth=1.5, label=str(point_of_interest))
            ax[n].set_xlim(-500, 4250)
            # ax[n].legend(loc='upper left', fontsize=10)
            ax[n].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=len(SVM_filter_times),
                         mode="expand", borderaxespad=0.)
            ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
            if viol_pos != 0:
                ax[n].axvline(250 * (viol_pos - 1), linestyle='-', color='red', linewidth=2)
        n += 1
    axes.ravel()[-1].set_xlabel('Time (ms)')

    figure = plt.gcf()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    return figure


# ______________________________________________________________________________________________________________________
def plot_SVM_projection_for_seqID_window(epochs_list, sensor_type, seqID=1, save_path=None):
    # this provides us with the position of the violations and the times
    epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
    times = epochs_seq_subset.times
    violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])

    # window info, just for figure title
    win_tmin = epochs_list['test'][0][0].metadata.SVM_filter_tmin_window[0] * 1000
    win_tmax = epochs_list['test'][0][0].metadata.SVM_filter_tmax_window[0] * 1000

    fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle('SVM %s - window %d-%dms - SequenceID_%d; N subjects = %d' % (
    sensor_type, win_tmin, win_tmax, seqID, len(epochs_list['test'])), fontsize=12)
    ax = axes.ravel()[::1]
    ax[0].set_title('Habituation trials', loc='right', weight='bold')
    ax[1].set_title('Standard test trials', loc='right', weight='bold')
    ax[2].set_title('Trials with deviant pos %d' % violpos_list[1], loc='right', weight='bold')
    ax[3].set_title('Trials with deviant pos %d' % violpos_list[2], loc='right', weight='bold')
    ax[4].set_title('Trials with deviant pos %d' % violpos_list[3], loc='right', weight='bold')
    ax[5].set_title('Trials with deviant pos %d' % violpos_list[4], loc='right', weight='bold')

    n = 0
    # First, plot habituation trials
    y_list = []
    for epochs in epochs_list['hab']:
        epochs_subset = epochs['SequenceID == "' + str(seqID) + '"']
        y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='SVM').get_data()))
    mean_hab = np.mean(y_list, axis=0)
    ub_hab = mean_hab + sem(y_list, axis=0)
    lb_hab = mean_hab - sem(y_list, axis=0)
    for xx in range(16):
        ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax[n].axhline(0, linestyle='-', color='black', linewidth=0.5)
    ax[n].fill_between(times * 1000, ub_hab, lb_hab, color='blue', alpha=.2)
    ax[n].plot(times * 1000, mean_hab, color='blue', linewidth=1.5)
    ax[n].set_xlim(-500, 4250)

    ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
    n += 1

    for viol_pos in violpos_list:
        y_list = []
        for epochs in epochs_list['test']:
            epochs_subset = epochs[
                'SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(viol_pos) + '"']
            y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='SVM').get_data()))
        mean = np.mean(y_list, axis=0)
        ub = mean + sem(y_list, axis=0)
        lb = mean - sem(y_list, axis=0)
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        ax[n].axhline(0, linestyle='-', color='black', linewidth=0.5)
        ax[n].fill_between(times * 1000, ub, lb, color='blue', alpha=.2)
        ax[n].plot(times * 1000, mean, color='blue', linewidth=1.5)
        ax[n].set_xlim(-500, 4250)

        ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
        if viol_pos != 0:
            ax[n].axvline(250 * (viol_pos - 1), linestyle='-', color='red', linewidth=2)
        n += 1
    axes.ravel()[-1].set_xlabel('Time (ms)')

    figure = plt.gcf()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    return figure


# ______________________________________________________________________________________________________________________
def plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type, save_path=None, vmin=-1, vmax=1,compute_reg_complexity = False, window_CBPT_violation = None):
    import matplotlib.colors as mcolors

    colors = [(0, 0, 0, c) for c in np.linspace(0, 1, 2)]
    cmapsig = mcolors.LinearSegmentedColormap.from_list('significance_cmpa', colors, N=5)

    # window info, just for figure title
    win_tmin = epochs_list['test'][0][0].metadata.SVM_filter_tmin_window[0] * 1000
    win_tmax = epochs_list['test'][0][0].metadata.SVM_filter_tmax_window[0] * 1000
    n_plots = 7
    if compute_reg_complexity:
        n_plots = 8
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 12), sharex=True, sharey=False, constrained_layout=True)
    fig.suptitle('SVM %s - window %d-%dms; N subjects = %d' % (
        sensor_type, win_tmin, win_tmax, len(epochs_list['test'])), fontsize=12)
    ax = axes.ravel()[::1]
    ax[0].set_title('Repeat', loc='left', weight='bold')
    ax[1].set_title('Alternate', loc='left', weight='bold')
    ax[2].set_title('Pairs', loc='left', weight='bold')
    ax[3].set_title('Quadruplets', loc='left', weight='bold')
    ax[4].set_title('Pairs+Alt', loc='left', weight='bold')
    ax[5].set_title('Shrinking', loc='left', weight='bold')
    ax[6].set_title('Complex', loc='left', weight='bold')

    seqtxtXY = ['xxxxxxxxxxxxxxxx',
                'xYxYxYxYxYxYxYxY',
                'xxYYxxYYxxYYxxYY',
                'xxxxYYYYxxxxYYYY',
                'xxYYxYxYxxYYxYxY',
                'xxxxYYYYxxYYxYxY',
                'xYxxxYYYYxYYxxxY']

    if compute_reg_complexity:
        ax[7].set_title('Beta_complexity', loc='left', weight='bold')
        seqtxtXY.append('')

    print("vmin = %0.02f, vmax = %0.02f" % (vmin, vmax))

    n = 0

    violation_significance = {i:[] for i in range(1, 8)}
    epochs_data_hab_allseq = []
    epochs_data_test_allseq = []

    for seqID in range(1, 8):
        #  this provides us with the position of the violations and the times
        epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
        times = epochs_seq_subset.times
        times = times + 0.3
        violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])
        violation_significance[seqID] = {'times':times,'window_significance':window_CBPT_violation}

        #  ----------- habituation trials -----------
        epochs_data_hab_seq = []
        y_list_epochs_hab = []
        data_mean = []
        mean_alpha = []
        for epochs in epochs_list['hab']:
            epochs_subset = epochs['SequenceID == "' + str(seqID) + '"']
            avg_epo = np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0)
            y_list_epochs_hab.append(avg_epo)
            epochs_data_hab_seq.append(avg_epo)
        epochs_data_hab_allseq.append(epochs_data_hab_seq)
        mean_hab = np.mean(y_list_epochs_hab, axis=0)
        data_mean.append(mean_hab)
        mean_alpha.append(np.zeros(mean_hab.shape))

        #  ----------- test trials -----------
        epochs_data_test_seq = []

        for viol_pos in violpos_list:
            y_list = []
            y_list_alpha = []
            contrast_viol_pos = []
            for epochs in epochs_list['test']:
                epochs_subset = epochs[
                    'SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                avg_epo = np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0)
                y_list.append(avg_epo)
                if viol_pos==0:
                    avg_epo_standard = np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0)
                    epochs_data_test_seq.append(avg_epo_standard)
                    y_list_alpha.append(np.zeros(avg_epo_standard.shape))
                if viol_pos !=0 and window_CBPT_violation is not None:
                    epochs_standard = epochs[
                        'SequenceID == "' + str(seqID) + '" and ViolationInSequence == 0']
                    avg_epo_standard = np.mean(np.squeeze(epochs_standard.savgol_filter(20).get_data()), axis=0)
                    contrast_viol_pos.append(avg_epo - avg_epo_standard)

            # --------------- CBPT to test for significance ---------------
            if window_CBPT_violation is not None and viol_pos !=0:
                time_start_viol = 0.250 * (viol_pos - 1)
                time_stop_viol = time_start_viol + window_CBPT_violation
                inds_stats = np.where(np.logical_and(times>time_start_viol,times<=time_stop_viol))
                contrast_viol_pos = np.asarray(contrast_viol_pos)
                p_vals = np.asarray([1]*contrast_viol_pos.shape[1])
                p_values = stats_funcs.stats(contrast_viol_pos[:,inds_stats[0]],tail=1)
                p_vals[inds_stats[0]] = p_values
                violation_significance[seqID][int(viol_pos)] = p_vals
                y_list_alpha.append(1*(p_vals<0.05))

            mean = np.mean(y_list, axis=0)
            mean_alpha_seq = np.mean(y_list_alpha, axis=0)
            data_mean.append(mean)
            mean_alpha.append(mean_alpha_seq)
        epochs_data_test_allseq.append(epochs_data_test_seq)

        width = 75
        # Add vertical lines, and "xY"
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
            txt = seqtxtXY[n][xx]
            ax[n].text(250 * (xx + 1) - 125, width * 6 + (width / 3), txt, horizontalalignment='center', fontsize=16)

        # return data_mean
        im = ax[n].imshow(data_mean, extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap='RdBu_r',
                          vmin=vmin, vmax=vmax)
        # add colorbar
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.50, aspect=10, pad=.005)
        cb.ax.yaxis.set_offset_position('left')
        cb.set_label('a. u.')
        if window_CBPT_violation:
            masked = np.ma.masked_where(mean_alpha == 0, mean_alpha)
            im = ax[n].imshow(masked, extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap=cmapsig,
                              vmin=vmin, vmax=vmax,alpha=0.7)
        ax[n].set_yticks(np.arange(width / 2, 6 * width, width))
        ax[n].set_yticklabels(['Violation (pos. %d)' % violpos_list[4], 'Violation (pos. %d)' % violpos_list[3],
                               'Violation (pos. %d)' % violpos_list[2], 'Violation (pos. %d)' % violpos_list[1],
                               'Standard', 'Habituation'])
        ax[n].axvline(0, linestyle='-', color='black', linewidth=2)

        # add deviant marks
        for k in range(4):
            viol_pos = violpos_list[k + 1]
            x = 250 * (viol_pos - 1)
            y1 = (4 - k) * width
            y2 = (4 - 1 - k) * width
            ax[n].plot([x, x], [y1, y2], linestyle='-', color='black', linewidth=6)
            ax[n].plot([x, x], [y1, y2], linestyle='-', color='yellow', linewidth=3)

        n += 1

    if compute_reg_complexity:
        epochs_data_hab_allseq = np.asarray(epochs_data_hab_allseq)
        epochs_data_test_allseq = np.asarray(epochs_data_test_allseq)
        coeff_const_hab, coeff_complexity_hab = compute_regression_complexity(epochs_data_hab_allseq)
        coeff_const_test, coeff_complexity_test = compute_regression_complexity(epochs_data_test_allseq)

        for xx in range(16):
            ax[7].axvline(250 * xx, linestyle='--', color='black', linewidth=1)

        # return data_mean
        im = ax[7].imshow(np.asarray([np.mean(coeff_complexity_hab,axis=0),np.mean(coeff_complexity_test,axis=0)]), extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap='RdBu_r',
                          vmin=-0.5, vmax=0.5)
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.50, aspect=10, pad=.005)
        cb.ax.yaxis.set_offset_position('left')
        width = width*3
        ax[7].set_yticks(np.arange(width / 2, 2 * width, width))
        ax[7].set_yticklabels(['Standard', 'Habituation'])
        ax[7].axvline(0, linestyle='-', color='black', linewidth=2)

    axes.ravel()[-1].set_xlabel('Time (ms)')

    figure = plt.gcf()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    return figure


# ______________________________________________________________________________________________________________________
def plot_SVM_projection_for_seqID_heatmap(epochs_list, sensor_type, seqID=1,
                                          SVM_filter_times=[x / 1000 for x in range(100, 700, 50)], save_path=None,
                                          vmin=None, vmax=None):
    # this provides us with the position of the violations and the times
    epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
    times = epochs_seq_subset.times
    violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])

    fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle('SVM %s - SequenceID_' % sensor_type + str(seqID) + ' N subjects = ' + str(len(epochs_list['test'])),
                 fontsize=12)
    ax = axes.ravel()[::1]
    ax[0].set_title('Habituation trials', loc='left', weight='bold')
    ax[1].set_title('Standard test trials', loc='left', weight='bold')
    ax[2].set_title('Trials with deviant pos %d' % violpos_list[1], loc='left', weight='bold')
    ax[3].set_title('Trials with deviant pos %d' % violpos_list[2], loc='left', weight='bold')
    ax[4].set_title('Trials with deviant pos %d' % violpos_list[3], loc='left', weight='bold')
    ax[5].set_title('Trials with deviant pos %d' % violpos_list[4], loc='left', weight='bold')

    n = 0
    # First, plot habituation trials
    mean_all_SVM_times = np.empty((0, len(times)), int)
    for ii, point_of_interest in enumerate(SVM_filter_times):
        y_list = []
        for epochs in epochs_list['hab']:
            epochs_subset = epochs['SequenceID == "' + str(seqID)
                                   + '" and SVM_filter_time == "' + str(point_of_interest) + '"']
            y_list.append(np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0))
            # y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='SVM').data))
            # y_list.append(np.squeeze(epochs_subset.average(picks='SVM').data))
        mean = np.mean(y_list, axis=0)
        mean_all_SVM_times = np.vstack([mean_all_SVM_times, mean])
    width = 50
    for xx in range(16):
        ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
    ax[n].imshow(mean_all_SVM_times, origin='lower',
                 extent=[min(times) * 1000, max(times) * 1000, 0, len(SVM_filter_times) * width], cmap='RdBu_r',
                 vmin=vmin, vmax=vmax)
    ax[n].set_yticks(
        np.arange(width / 2, len(SVM_filter_times) * width, len(SVM_filter_times) * width / len(SVM_filter_times)))
    ax[n].set_yticklabels(SVM_filter_times)
    ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
    n += 1

    # Then, plot all other trials
    for viol_pos in violpos_list:
        mean_all_SVM_times = np.empty((0, len(times)), int)
        for ii, point_of_interest in enumerate(SVM_filter_times):
            y_list = []
            for epochs in epochs_list['test']:
                epochs_subset = epochs['SequenceID == "' + str(seqID)
                                       + '" and SVM_filter_time == "' + str(point_of_interest)
                                       + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                y_list.append(np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0))
                # y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='SVM').data))
                # y_list.append(np.squeeze(epochs_subset.average(picks='SVM').data))
            mean = np.mean(y_list, axis=0)
            mean_all_SVM_times = np.vstack([mean_all_SVM_times, mean])

        width = 50
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
        ax[n].imshow(mean_all_SVM_times, origin='lower',
                     extent=[min(times) * 1000, max(times) * 1000, 0, len(SVM_filter_times) * width], cmap='RdBu_r',
                     vmin=vmin, vmax=vmax)
        # ax[n].set_xlim(-500, 4250)
        # ax[n].legend(loc='upper left', fontsize=10)
        ax[n].set_yticks(
            np.arange(width / 2, len(SVM_filter_times) * width, len(SVM_filter_times) * width / len(SVM_filter_times)))
        ax[n].set_yticklabels(SVM_filter_times)
        ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
        if viol_pos != 0:
            ax[n].axvline(250 * (viol_pos - 1), linestyle='-', color='black', linewidth=4)
            ax[n].axvline(250 * (viol_pos - 1), linestyle='-', color='yellow', linewidth=2)
        n += 1
    axes.ravel()[-1].set_xlabel('Time (ms)')

    figure = plt.gcf()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    return figure


# =========================================================================================================

class ZScoreEachChannel(TransformerMixin):
    """
    Z-score the data of each channel separately

    Input matrix: Epochs x Channels x TimePoints
    Output matrix: Epochs x Channels x TimePoints (same size as input)
    """

    # --------------------------------------------------
    def __init__(self, debug=False):
        self._debug = debug

    # --------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self

    # --------------------------------------------------
    def transform(self, x):
        result = np.zeros(x.shape)
        n_epochs, nchannels, ntimes = x.shape
        for c in range(nchannels):
            channel_data = x[:, c, :]
            m = np.mean(channel_data)
            sd = np.std(channel_data)
            if self._debug:
                print('ZScoreEachChannel: channel {:} m={:}, sd={:}'.format(c, m, sd))
            result[:, c, :] = (x[:, c, :] - m) / sd

        return result


# =========================================================================================================

class SlidingWindow(TransformerMixin):
    """
    Aggregate time points in a "sliding window" manner

    Input: Anything x Anything x Time points
    Output - if averaging: Unchanged x Unchanged x Windows
    Output - if not averaging: Windows x Unchanged x Unchanged x Window size
                Note that in this case, the output may not be a real matrix in case the last sliding window is smaller than the others
    """

    # --------------------------------------------------
    def __init__(self, window_size, step, min_window_size=None, average=True, debug=False):
        """
        :param window_size: The no. of time points to average
        :param step: The no. of time points to slide the window to get the next result
        :param min_window_size: The minimal number of time points acceptable in the last step of the sliding window.
                                If None: min_window_size will be the same as window_size
        :param average: If True, just reduce the number of time points by averaging over each window
                        If False, each window is copied as-is to the output, without averaging
        """
        self._window_size = window_size
        self._step = step
        self._min_window_size = min_window_size
        self._average = average
        self._debug = debug

    # --------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self

    # --------------------------------------------------
    def transform(self, x):
        x = np.array(x)
        assert len(x.shape) == 3
        n1, n2, n_time_points = x.shape

        # -- Get the start-end indices of each window
        min_window_size = self._min_window_size or self._window_size
        window_start = np.array(range(0, n_time_points - min_window_size + 1, self._step))
        if len(window_start) == 0:
            # -- There are fewer than window_size time points
            raise Exception('There are only {:} time points, but at least {:} are required for the sliding window'.
                            format(n_time_points, self._min_window_size))
        window_end = window_start + self._window_size
        window_end[-1] = min(window_end[-1],
                             n_time_points)  # make sure that the last window doesn't exceed the input size

        if self._debug:
            win_info = [(s, e, e - s) for s, e in zip(window_start, window_end)]
            print('SlidingWindow transformer: the start,end,length of each sliding window: {:}'.
                  format(win_info))
            if len(win_info) > 1 and win_info[0][2] != win_info[-1][2] and not self._average:
                print(
                    'SlidingWindow transformer: note that the last sliding window is smaller than the previous ones, ' +
                    'so the result will be a list of 3-dimensional matrices, with the last list element having ' +
                    'a different dimension than the previous elements. ' +
                    'This format is acceptable by the RiemannDissimilarity transformer')

        if self._average:
            # -- Average the data in each sliding window
            result = np.zeros((n1, n2, len(window_start)))
            for i in range(len(window_start)):
                result[:, :, i] = np.mean(x[:, :, window_start[i]:window_end[i]], axis=2)

        else:
            # -- Don't average the data in each sliding window - just copy it
            result = []
            for i in range(len(window_start)):
                result.append(x[:, :, window_start[i]:window_end[i]])

        return result


# =========================================================================================================

class AveragePerEvent(TransformerMixin):
    """
    This transformer averages all epochs that have the same label.
    It can also create several averages per event ID (based on independent sets of trials)

    Input matrix: Epochs x Channels x TimePoints
    Output matrix: Labels x Channels x TimePoints. If asked to create N results per event ID, the "labels"
                   dimension is multiplied accordingly.
    """

    # --------------------------------------------------
    def __init__(self, event_ids=None, n_results_per_event=1, max_events_with_missing_epochs=0, debug=False):
        """
        :param event_ids: The event IDs to average on. If None, compute average for all available events.
        :param n_results_per_event: The number of aggregated stimuli to create per event type.
               The event's epochs are distributed randomly into N groups, and each group is averaged, creating
               N independent results.
        :param max_events_with_missing_epochs: The maximal number of event IDs for which we allow the number
               of epochs to be smaller than 'n_results_per_event'. For such events, randomly-selected epochs
               will be duplicated.
        """
        assert isinstance(n_results_per_event, int) and n_results_per_event > 0
        assert isinstance(max_events_with_missing_epochs, int) and max_events_with_missing_epochs >= 0

        self._event_ids = None if event_ids is None else np.array(event_ids)
        self._curr_event_ids = None
        self._n_results_per_event = n_results_per_event
        self._max_events_with_missing_epochs = max_events_with_missing_epochs
        self._debug = debug

        if debug:
            if event_ids is None:
                print('AveragePerEvent: will create averages for all events.')
            else:
                print('AveragePerEvent: will create averages for these events: {:}'.format(event_ids))

    # --------------------------------------------------
    # noinspection PyUnusedLocal,PyAttributeOutsideInit
    def fit(self, x, y, *_):

        self._y = np.array(y)

        if self._event_ids is None:
            self._curr_event_ids = np.unique(y)
            if self._debug:
                print('AveragePerEvent: events IDs are {:}'.format(self._event_ids))
        else:
            self._curr_event_ids = self._event_ids

        return self

    # --------------------------------------------------
    def transform(self, x):

        x = np.array(x)

        result = []

        # -- Split the epochs by event ID.
        # -- x_per_event_id has a 3-dim matrix for each event ID
        x_per_event_id = [x[self._y == eid] for eid in self._curr_event_ids]

        # -- Check if there are enough epochs per event ID
        too_few_epochs = [len(e) < self._n_results_per_event for e in x_per_event_id]  # list of bool - one per event ID
        if sum(too_few_epochs) > self._max_events_with_missing_epochs:
            raise Exception('There are {:} event IDs with fewer than {:} epochs: {:}'.
                            format(sum(too_few_epochs), self._n_results_per_event,
                                   self._curr_event_ids[np.where(too_few_epochs)[0]]))
        elif sum(too_few_epochs) > 0:
            print('WARNING (AveragePerEvent): There are {:} event IDs with fewer than {:} epochs: {:}'.
                  format(sum(too_few_epochs), self._n_results_per_event,
                         self._curr_event_ids[np.where(too_few_epochs)[0]]))

        # -- Do the actual aggregation
        for i in range(len(x_per_event_id)):
            # Get a list whose length is n_results_per_event; each list entry is a 3-dim matrix to average
            agg = self._aggregate(x_per_event_id[i])
            if self._debug:
                print('AveragePerEvent: event={:}, #epochs={:}'.format(self._curr_event_ids[i], [len(a) for a in agg]))
            result.extend([np.mean(a, axis=0) for a in agg])

        result = np.array(result)

        if self._debug:
            print('AveragePerEvent: transformed from shape={:} to shape={:}'.format(x.shape, result.shape))

        return result

    # --------------------------------------------------
    def _aggregate(self, one_event_x):
        """
        Distribute the epochs of one_event_x into separate sets

        The function returns a list with self._n_results_per_event different sets.
        """

        if self._n_results_per_event == 1:
            # -- Aggregate all epochs into one result
            return [one_event_x]

        if len(one_event_x) >= self._n_results_per_event:

            # -- The number of epochs is sufficient to have at least one different epoch per result

            one_event_x = np.array(one_event_x)

            result = [[]] * self._n_results_per_event

            # -- First, distribute an equal number of epochs to each result
            n_in_epochs = len(one_event_x)
            in_epochs_inds = range(len(one_event_x))
            random.shuffle(in_epochs_inds)
            n_take_per_result = int(np.floor(n_in_epochs / self._n_results_per_event))
            for i in range(self._n_results_per_event):
                result[i] = list(one_event_x[in_epochs_inds[:n_take_per_result]])
                in_epochs_inds = in_epochs_inds[n_take_per_result:]

            # -- If some epochs remained, add each of them to a different result set
            n_remained = len(in_epochs_inds)
            for i in range(n_remained):
                result[i].append(one_event_x[in_epochs_inds[i]])

        else:

            # -- The number of epochs is too small: each result will consist of a single epoch, and epochs some will be duplicated

            # -- First, take all events that we have
            result = list(one_event_x)

            # -- Then pick random some epochs and duplicate them
            n_missing = self._n_results_per_event - len(result)
            epoch_inds = range(len(one_event_x))
            random.shuffle(epoch_inds)
            duplicated_inds = epoch_inds[:n_missing]
            result.extend(np.array(result)[duplicated_inds])

            result = [[x] for x in result]

        random.shuffle(result)

        return result


# ---------------------------------------------------------------------------------------------------------------------
def plot_gat_simple(analysis_name,subjects_list,fig_name,score_field='GAT',folder_name = 'GAT',sensors = ['all_chans'],vmin=-0.1,vmax=.1):
    GAT_all = []
    fig_path = op.join(config.fig_path, 'SVM', folder_name)
    count = 0
    for subject in subjects_list:
        count += 1
        SVM_path = op.join(config.SVM_path, subject)
        GAT_path = op.join(SVM_path, analysis_name + '.npy')
        GAT_results = np.load(GAT_path, allow_pickle=True).item()
        print(op.join(SVM_path, analysis_name + '.npy'))
        times = GAT_results['times']
        GAT_all.append(GAT_results[score_field])

    plot_GAT_SVM(np.mean(GAT_all,axis=0), times, sens=sensors, save_path=fig_path, figname=fig_name,vmin=vmin,vmax=vmax)

    print("============ THE AVERAGE GAT WAS COMPUTED OVER %i PARTICIPANTS ========"%count)

    return plt.gcf()

def plot_all_subjects_results_SVM(analysis_name,subjects_list,fig_name,plot_per_sequence=False,plot_individual_subjects=False,score_field='GAT',folder_name = 'GAT',sensors = ['eeg', 'mag', 'grad','all_chans'],vmin=-0.1,vmax=.1,analysis_type='',compute_significance=None):

    GAT_sens_all = {sens: [] for sens in sensors}

    for sens in sensors:
        print("==== running the plotting for sensor type %s"%sens)
        count = 0
        for subject in subjects_list:
            SVM_path = op.join(config.SVM_path, subject)
            GAT_path = op.join(SVM_path,analysis_name+'.npy')
            if op.exists(GAT_path):
                count +=1

                GAT_results = np.load(GAT_path, allow_pickle=True).item()
                print(op.join(SVM_path, analysis_name+'.npy'))
                times = GAT_results['times']
                GAT_results = GAT_results[score_field]
                fig_path = op.join(config.fig_path, 'SVM', folder_name)
                sub_fig_path = op.join(fig_path,subject)
                utils.create_folder(sub_fig_path)
                if plot_per_sequence:
                    if not GAT_sens_all[sens]:  # initialize the keys and empty lists only the first time
                        GAT_sens_all[sens] = {'SeqID_%i' % i: [] for i in range(1, 8)}
                        GAT_sens_all[sens]['average_all_sequences'] = []
                    for key in ['SeqID_%i' % i for i in range(1, 8)]:
                        GAT_sens_all[sens][key].append(GAT_results[sens][key])
                        # ================ Plot & save each subject / each sequence figures ???
                        if plot_individual_subjects:
                            plot_GAT_SVM(GAT_results[sens][key],0, times, sens=sens, save_path=sub_fig_path, figname=fig_name+key); plt.close('all')
                    # ================ Plot & save each subject / average of all sequences figures ???
                    GAT_sens_all[sens]['average_all_sequences'].append(GAT_results[sens]['average_all_sequences'])
                    plot_GAT_SVM(GAT_results[sens]['average_all_sequences'],0, times, sens=sens, save_path=sub_fig_path, figname=fig_name+'_all_seq',vmin=vmin,vmax=vmax)
                    plt.close('all')
                else:
                    GAT_sens_all[sens].append(GAT_results)
                    if plot_individual_subjects:
                        print('plotting for subject:%s'%subject)
                        print("the shape of the GAT result is ")
                        print(GAT_results.shape)
                        plot_GAT_SVM(GAT_results,0, times, sens=sens, save_path=sub_fig_path,
                                     figname=fig_name,vmin=vmin,vmax=vmax)
                        plt.close('all')
        # return GAT_sens_all

        plt.close('all')
        print("plotting in %s"%config.fig_path)
        if plot_per_sequence:
            for key in ['SeqID_%i' % i for i in range(1, 8)]:
                sig_all = None
                if compute_significance is not None:
                    GAT_all = np.asarray(GAT_sens_all[sens][key])
                    tmin_sig = compute_significance[0]
                    tmax_sig = compute_significance[1]
                    times_sig = np.where(np.logical_and(times <= tmax_sig, times > tmin_sig))[0]
                    sig_all = np.ones(GAT_all[0].shape)
                    GAT_all_for_sig = GAT_all[:, times_sig, :]
                    GAT_all_for_sig = GAT_all_for_sig[:, :, times_sig]
                    sig = stats_funcs.stats(GAT_all_for_sig, tail=-1)
                    sig_all = replace_submatrix(sig_all, times_sig, times_sig, sig)

                plot_GAT_SVM(np.nanmean(GAT_sens_all[sens][key],axis=0),0, times, sens=sens, save_path=fig_path,sig=sig_all<0.05,
                             figname=fig_name+key,vmin=vmin, vmax=vmax)
                plt.close('all')

            plot_GAT_SVM(np.nanmean(GAT_sens_all[sens]['average_all_sequences'],axis=0),0,  times, sens=sens,
                         save_path=fig_path, figname=fig_name + '_all_seq' + '_',
                         vmin=vmin, vmax=vmax)
            plt.close('all')
        else:
            plot_GAT_SVM(np.mean(GAT_sens_all[sens],axis=0), times, sens=sens, save_path=fig_path, figname=fig_name,vmin=vmin,vmax=vmax)

    print("============ THE AVERAGE GAT WAS COMPUTED OVER %i PARTICIPANTS ========"%count)

    # ===== GROUP AVG FIGURES ===== #
    if analysis_type == 'perSeq':
        plt.close('all')
        for sens in ['eeg', 'mag', 'grad', 'all_chans']:
            GAT_avg_sens = GAT_sens_all[sens]
            for seqID in range(1, 8):
                GAT_avg_sens_seq = GAT_avg_sens['SeqID_%i' % seqID]
                if compute_significance is not None:
                    GAT_all = np.asarray(GAT_avg_sens_seq)
                    tmin_sig = compute_significance[0]
                    tmax_sig = compute_significance[1]
                    times_sig = np.where(np.logical_and(times <= tmax_sig, times > tmin_sig))[0]
                    sig_all = np.ones(GAT_all[0].shape)
                    GAT_all_for_sig = GAT_all[:, times_sig, :]
                    GAT_all_for_sig = GAT_all_for_sig[:, :, times_sig]
                    sig = stats_funcs.stats(GAT_all_for_sig, tail=-1)
                    sig_all = replace_submatrix(sig_all, times_sig, times_sig, sig)
                    plot_GAT_SVM(np.mean(GAT_all,axis=0),0, times, sens=sens, save_path=fig_path,sig=sig_all,
                                 figname= 'GAT_' + str(seqID) +'_allparticipants_',vmin=vmin, vmax=vmax)
                else:
                    GAT_avg_sens_seq_groupavg = np.mean(GAT_avg_sens_seq, axis=0)
                    plot_GAT_SVM(GAT_avg_sens_seq_groupavg,0, times, sens=sens,
                                 save_path=op.join(config.fig_path, 'SVM', 'GAT'),
                                 figname='GAT_' + str(seqID) + '_allparticipants_')
                plt.close('all')
            GAT_avg_sens_allseq_groupavg = np.mean(GAT_avg_sens['average_all_sequences'], axis=0)
            plot_GAT_SVM(GAT_avg_sens_allseq_groupavg,0, times, sens=sens,
                         save_path=op.join(config.fig_path, 'SVM', 'GAT'),
                         figname= 'GAT_all_seq'  + '_allparticipants_',vmin=vmin, vmax=vmax)

    return GAT_sens_all, times

def SVM_GAT_linear_reg_sequence_complexity(subject,suffix = 'SW_train_test_different_blocksGAT_results_score.npy'):

    # load the participants GAT results for the decoding of standard VS deviant for all the different sequences =========
    SVM_path = op.join(config.SVM_path, subject)
    GAT_path = op.join(SVM_path,suffix)
    GAT_results = np.load(GAT_path, allow_pickle=True).item()
    times = GAT_results['times']
    GAT = GAT_results['GAT']['all_chans']
    # We concatenate the data from all the sequences for that participant =========
    GAT_all_sequences = []
    for seqID in range(1, 8):
        GAT_all_sequences.append(GAT['SeqID_%i' % seqID])

    GAT_all_sequences = np.asarray(GAT_all_sequences)

    # ====== select a training and a testing time and compute the regression coeffs =====
    from sklearn.linear_model import LinearRegression
    complexities = np.asarray([4,6,6,6,12,14,23])
    coeff_constant = np.zeros((GAT_all_sequences.shape[1],GAT_all_sequences.shape[1]))

    coeff_complexity = np.zeros((GAT_all_sequences.shape[1],GAT_all_sequences.shape[1]))
    for train_ind in range(GAT_all_sequences.shape[1]):
        for test_ind in range(GAT_all_sequences.shape[1]):
            data = GAT_all_sequences[:,train_ind,test_ind]
            reg = LinearRegression().fit(complexities.reshape(-1,1), data)
            coeff_complexity[train_ind,test_ind] = reg.coef_
            coeff_constant[train_ind,test_ind] = reg.intercept_

    np.save(SVM_path+'/GAT_lin_reg_complexity.npy',{'coeff_complexity':coeff_complexity,'coeff_constant':coeff_constant,'times':times})

    return coeff_complexity, coeff_constant, times


def plot_gat_simple(analysis_name, subjects_list, fig_name,chance, score_field='GAT', folder_name='GAT',vmin=-0.1, vmax=.1,compute_significance=None):
    GAT_all = []
    fig_path = op.join(config.fig_path, 'SVM', folder_name)
    count = 0
    for subject in subjects_list:
        count += 1
        SVM_path = op.join(config.SVM_path, subject)
        GAT_path = op.join(SVM_path, analysis_name + '.npy')
        GAT_results = np.load(GAT_path, allow_pickle=True).item()
        print(op.join(SVM_path, analysis_name + '.npy'))
        times = GAT_results['times']
        GAT_all.append(GAT_results[score_field])

    if compute_significance is not None:
        GAT_all = np.asarray(GAT_all)
        tmin_sig = compute_significance[0]
        tmax_sig = compute_significance[1]
        times_sig = np.where(np.logical_and(times<=tmax_sig,times>tmin_sig))[0]
        sig_all = np.ones(GAT_all[0].shape)
        GAT_all_for_sig = GAT_all[:,times_sig,:]
        GAT_all_for_sig = GAT_all_for_sig[:,:,times_sig]
        sig = stats_funcs.stats(GAT_all_for_sig-chance,tail=1)
        sig_all= replace_submatrix(sig_all, times_sig, times_sig, sig)

    if vmin is not None:
        if compute_significance is not None:
            pretty_gat(np.mean(GAT_all, axis=0), times, chance=chance, clim=[vmin, vmax],sig=sig_all<0.05)
        else:
            pretty_gat(np.mean(GAT_all, axis=0), times, chance=chance, clim=[vmin, vmax])
    else:
        if compute_significance is not None:
            pretty_gat(np.mean(GAT_all, axis=0), times, chance=chance, sig=1*(sig_all<0.05))
        else:
            pretty_gat(np.mean(GAT_all, axis=0), times, chance=chance)

    plt.gcf().savefig(fig_path+'/'+fig_name)
    plt.close('all')

    print("============ THE AVERAGE GAT WAS COMPUTED OVER %i PARTICIPANTS ========" % count)

    return plt.gcf()

def replace_submatrix(mat, ind1, ind2, mat_replace):
  for i, index in enumerate(ind1):
    mat[index, ind2] = mat_replace[i, :]
  return mat

def check_missing_GAT_data(subjects):
    for subject in subjects:
        SVM_path = op.join(config.SVM_path, subject)
        GAT_path = op.join(SVM_path, 'SW_train_test_different_blocksGAT_results_score.npy')
        GAT_results = np.load(GAT_path, allow_pickle=True).item()
        times = GAT_results['times']
        GAT = GAT_results['GAT']['all_chans']
        # We concatenate the data from all the sequences for that participant =========
        for seqID in range(1, 8):
            if np.sum(np.isnan(GAT['SeqID_%i' % seqID])) != 0:
                print("----------- there is no data for subject %s and sequence %i" % (subject, seqID))



def compute_regression_complexity(data):
    """
    Data has to be obtained as the mean over the epochs for each participant for the different sequences in the same order as
    the corresponding complexities
    :param
    :return:
    """
    complexities = np.asarray([4,6,6,6,12,14,28])
    n_times = data.shape[2]

    Constant_coeff = []
    Complexity_coeff = []

    for ii in range(data.shape[1]):
        data_reg_subject = data[:,ii,:]
        coeff_constant = [0]*n_times
        coeff_complexity = [0]*n_times
        for tt in range(n_times):
            data_reg = data_reg_subject[:,tt]
            reg = LinearRegression().fit(complexities.reshape(-1,1), data_reg)
            coeff_constant[tt] = reg.coef_[0]
            coeff_complexity[tt] = reg.intercept_
        Constant_coeff.append(coeff_constant)
        Complexity_coeff.append(coeff_complexity)

    return np.asarray(Constant_coeff), np.asarray(Complexity_coeff)



def compute_regression_complexity_epochs(epochs_name):
    """
    This function computes the regression for each participant of the data as a function of complexity
    :param epochs_name:
    :return:
    """

    from sklearn.linear_model import LinearRegression
    complexities = np.asarray([4,6,6,6,12,14,28])

    Constant_coeff = []
    Complexity_coeff = []

    for nsubj, subject in enumerate(config.subjects_list):
        epoch = mne.read_epochs(op.join(config.meg_dir, subject, epochs_name))
        n_times = len(epoch.times)
        coeff_constant = np.zeros((1,n_times))
        coeff_complexity = np.zeros((1,n_times))
        data = []
        for seqID in range(1, 8):
            data.append(epoch['SequenceID == "' + str(seqID) + '" and ViolationInSequence == 0'])
        data = np.asarray(data)
        for tt in range(n_times):
            data_reg = data[:,tt]
            reg = LinearRegression().fit(complexities.reshape(-1,1), data_reg)
            coeff_constant[tt] = reg.coef_
            coeff_complexity[tt] = reg.intercept_
        Constant_coeff.append(coeff_constant)
        Complexity_coeff.append(coeff_complexity)

    return np.asarray(Constant_coeff), np.asarray(Complexity_coeff)
