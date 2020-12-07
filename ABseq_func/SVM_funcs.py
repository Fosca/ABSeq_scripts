# This module contains all the functions related to the decoding analysis
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import time
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage.filters import gaussian_filter1d
from ABseq_func import *
import config
from scipy.stats import sem
from ABseq_func import utils  # why do we need this now ?? (error otherwise)
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mne.decoding import GeneralizingEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np
import random

from sklearn.base import TransformerMixin


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
def SVM_decode_feature(subject,feature_name,load_residuals_regression=False, list_sequences = None, decim = 1,crop=None):
    """
    Builds an SVM decoder that will be able to output the distance to the hyperplane once trained on data.
    It is meant to generalize across time by construction.
    :return:
    """

    SVM_dec = SVM_decoder()
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
    if decim is not None:
        epochs.decimate(decim)
    metadata = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None)
    if crop is not None:
        epochs.crop(crop[0],crop[1])
    epochs.metadata = metadata
    epochs = epochs["TrialNumber>10 and ViolationOrNot ==0"]
    # remove the stim channel from decoding
    epochs.pick_types(meg=True,eeg=True,stim=False)
    suf = ''
    if load_residuals_regression:
        epochs = epoching_funcs.load_resid_epochs_items(subject)
        suf = 'resid_'

    print('-- The values of the metadata for the feature %s are : ' % feature_name)
    print(np.unique(epochs.metadata[feature_name].values))

    if list_sequences is not None:
        # concatenate the epochs belonging to the different sequences from the list_sequences
        epochs_concat1 = []
        # count the number of epochs that contribute per sequence in order later to balance this
        n_epochs = []
        for seqID in list_sequences:
            epo = epochs["SequenceID == " + str(seqID)].copy()
            filter_epochs = np.where(1-np.isnan(epo.metadata[feature_name].values))[0]
            epo = epo[filter_epochs]
            epo.events[:, 2] = epo.metadata[feature_name].values
            epo.event_id = {'%i' % i: i for i in np.unique(epo.events[:, 2])}
            epo.equalize_event_counts(epo.event_id)
            n_epochs.append(len(epo))
            print("---- there are %i epochs that contribute from sequence %i -----" % (len(epo), seqID))
            epochs_concat1.append(epo)
        # now determine the minimum number of epochs that come from a sequence
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
    else:
        epochs.events[:, 2] = epochs.metadata[feature_name].values
        epochs.event_id = {'%i' % i: i for i in np.unique(epochs.events[:, 2])}
        epochs.equalize_event_counts(epochs.event_id)

    import time
    before_decoding = time.time()
    kf = KFold(n_splits=4)

    y = epochs.events[:, 2]
    X = epochs._data
    scores = []
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        SVM_dec.fit(X_train, y_train)
        scores.append(SVM_dec.score(X_test, y_test))

    score = np.mean(scores, axis=0)
    times = epochs.times
    # then use plot_GAT_SVM to plot the gat matrix
    after_decoding = time.time()-before_decoding
    print("================ the decoding of feature %s took %i seconds ====="%(feature_name,int(after_decoding)))


    return score, times


# ______________________________________________________________________________________________________________________
def generate_SVM_all_sequences(subject,load_residuals_regression=False,train_test_different_blocks=True,sliding_window=False):
    """
    Generates the SVM decoders for all the channel types using 4 folds. We save the training and testing indices as well as the epochs
    in order to be flexible for the later analyses.

    :param epochs:
    :param saving_directory:
    :return:
    """

    epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
    epochs.pick_types(meg=True,eeg=True,stim=False)

    suf = ''
    if load_residuals_regression:
        epochs = epoching_funcs.load_resid_epochs_items(subject)
        suf = 'resid_'

    saving_directory = op.join(config.SVM_path, subject)
    utils.create_folder(saving_directory)

    epochs_balanced = epoching_funcs.balance_epochs_violation_positions(epochs,balance_violation_standards=True)
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
    epochs_all = [epochs_balanced_mag, epochs_balanced_grad, epochs_balanced_eeg,epochs_balanced_all_chans]
    sensor_types = ['mag', 'grad', 'eeg','all_chans']
    SVM_results = {'mag': [], 'grad': [], 'eeg': [],'all_chans':[]}

    for l in range(4):
        senso = sensor_types[l]
        epochs_senso = epochs_all[l]
        X_data = epochs_senso.get_data()
        # ======= create the 4 SVM spatial filters ========
        All_SVM = []

        if train_test_different_blocks:
            run_numbers = epochs_senso.metadata['RunNumber'].values
            training_inds = [np.where(run_numbers < 8)[0], np.where(run_numbers >= 8)[0]]
            testing_inds = [np.where(run_numbers >= 8)[0], np.where(run_numbers < 8)[0]]
            for k in range(2):
                SVM_dec = SVM_decoder()
                SVM_dec.fit(X_data[training_inds[k]], y_violornot[training_inds[k]])
                All_SVM.append(SVM_dec)

        else:
            training_inds = []
            testing_inds = []
            metadata_epochs = epochs_balanced.metadata
            y_tmp = [
                int(metadata_epochs['SequenceID'].values[i] * 1000 + metadata_epochs['StimPosition'].values[i] * 10 +
                    metadata_epochs['ViolationOrNot'].values[i]) for i in range(len(epochs_balanced))]
            for train, test in StratifiedKFold(n_splits=4).split(X_data, y_tmp):
                # we split in training and testing sets using y_tmp because it allows us to balance with respect to all our constraints
                SVM_dec = SVM_decoder()
                # Fit and store the spatial filters
                SVM_dec.fit(X_data[train], y_violornot[train])
                # Store filters for future plotting
                All_SVM.append(SVM_dec)
                training_inds.append(train)
                testing_inds.append(test)

        SVM_results[senso] = {'SVM': All_SVM, 'train_ind': training_inds, 'test_ind': testing_inds,
                              'epochs': epochs_all[l]}

    if train_test_different_blocks:
        suf += 'train_test_different_blocks'
    np.save(op.join(saving_directory, suf+'SVM_results.npy'), SVM_results)


# ______________________________________________________________________________________________________________________
def GAT_SVM(subject,load_residuals_regression=False,score_or_decisionfunc = 'score',train_test_different_blocks=True,sliding_window=False):
    """
    The SVM at a training times are tested at testing times. Allows to obtain something similar to the GAT from decoding.
    Dictionnary contains the GAT for each sequence separately. GAT_all contains the average over all the sequences
    :param SVM_results: output of generate_SVM_all_sequences
    :return: GAT averaged over the 4 classification folds
    """
    suf = ''

    if sliding_window:
        suf += 'SW_'

    saving_directory = op.join(config.SVM_path, subject)
    n_folds = 4
    if load_residuals_regression:
        suf = 'resid_'

    if train_test_different_blocks:
        suf += 'train_test_different_blocks'
        n_folds = 2



    SVM_results = np.load(op.join(saving_directory, suf+'SVM_results.npy'), allow_pickle=True).item()

    GAT_sens_seq = {sens: [] for sens in ['eeg', 'mag', 'grad','all_chans']}

    for sens in ['eeg', 'mag', 'grad','all_chans']:
    # for sens in ['eeg']:
        print(sens)
        GAT_all = []
        GAT_per_sens_and_seq = {'SeqID_%i' % i: [] for i in range(1, 8)}

        epochs_sens = SVM_results[sens]['epochs']
        n_times = epochs_sens.get_data().shape[-1]
        SVM_sens = SVM_results[sens]['SVM']

        for k in range(1, 8):
            print('The value of k is %i'%k)
            seqID = 'SeqID_%i' % k
            GAT_seq = np.zeros((4,n_times,n_times))
            for fold_number in range(n_folds):
                test_indices = SVM_results[sens]['test_ind'][fold_number]
                epochs_sens_test = epochs_sens[test_indices]
                inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                        epochs_sens_test.metadata['ViolationOrNot'].values == 0))[0]
                inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                        epochs_sens_test.metadata['ViolationOrNot'].values == 1))[0]
                X = epochs_sens_test.get_data()
                if score_or_decisionfunc == 'score':
                    GAT_each_epoch = SVM_sens[fold_number].predict(X)
                else:
                    GAT_each_epoch = SVM_sens[fold_number].decision_function(X)
                GAT_seq[fold_number, :, :] = np.mean(
                    GAT_each_epoch[inds_seq_noviol,:,:],axis=0) - np.mean(
                    GAT_each_epoch[inds_seq_viol,:,:],axis=0)
                print('The shape of GAT_seq[fold_number, :, :] is')
                print(GAT_seq[fold_number, :, :].shape)

            # now average across the 4 folds
            GAT_seq_avg = np.mean(GAT_seq, axis=0)
            GAT_per_sens_and_seq[seqID] = GAT_seq_avg
            GAT_all.append(GAT_seq_avg)

        GAT_sens_seq[sens] = GAT_per_sens_and_seq
        GAT_sens_seq[sens]['average_all_sequences'] = np.mean(GAT_all, axis=0)
        times = epochs_sens_test.times


    GAT_results = {'GAT': GAT_sens_seq, 'times': times}
    if score_or_decisionfunc == 'score':
        np.save(op.join(saving_directory, suf+'GAT_results_score.npy'), GAT_results)
    else:
        np.save(op.join(saving_directory, suf+'GAT_results.npy'), GAT_results)


# ______________________________________________________________________________________________________________________
def GAT_SVM_4pos(subject,load_residuals_regression=False,score_or_decisionfunc = 'score',train_test_different_blocks=True):
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
    SVM_results = np.load(op.join(saving_directory, suf+'SVM_results.npy'), allow_pickle=True).item()

    GAT_sens_seq = {sens: [] for sens in ['eeg', 'mag', 'grad','all_chans']}

    for sens in ['eeg', 'mag', 'grad','all_chans']:
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
                print("====== running for fold number %i =======\n"%fold_number)
                test_indices = SVM_results[sens]['test_ind'][fold_number]
                epochs_sens_test = epochs_sens[test_indices]
                X = epochs_sens_test.get_data()
                if score_or_decisionfunc =='score':
                    GAT_each_epoch = SVM_sens[fold_number].predict(X)
                else:
                    GAT_each_epoch = SVM_sens[fold_number].decision_function(X)
                for nn, pos_viol in enumerate(violpos_list):
                    print("===== RUNNING for SEQ %i and position violation %i"%(k,nn))
                    inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                            epochs_sens_test.metadata['ViolationOrNot'].values == 0) & (
                                                       epochs_sens_test.metadata['StimPosition'].values == pos_viol))[0]
                    inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                            epochs_sens_test.metadata['ViolationOrNot'].values == 1) & (
                                                     epochs_sens_test.metadata['StimPosition'].values == pos_viol))[0]
                    GAT_seq[fold_number, :, :,nn] = np.mean(
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
        np.save(op.join(saving_directory, suf+'GAT_results_4pos_score.npy'), GAT_results)
    else:
        np.save(op.join(saving_directory, suf+'GAT_results_4pos.npy'), GAT_results)
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

    y_violornot = {'eeg': [], 'grad': [], 'mag': [],'all_chans':[]}
    X_transform = {'eeg': [], 'grad': [], 'mag': [],'all_chans':[]}

    for sens in {'eeg', 'grad', 'mag','all_chans'}:
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
def plot_GAT_SVM(GAT_avg, times, sens='mag', save_path=None, figname='GAT_', vmin=None, vmax=None):
    minT = np.min(times) * 1000
    maxT = np.max(times) * 1000
    fig = plt.figure()
    plt.imshow(-GAT_avg, origin='lower', extent=[minT, maxT, minT, maxT], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    # -----# ADD LINES ?
    plt.axvline(0, linestyle='-', color='black', linewidth=1)
    plt.axhline(0, linestyle='-', color='black', linewidth=1)
    plt.plot([minT, maxT], [minT, maxT], linestyle='--', color='black', linewidth=1)
    # -----#
    plt.title('Generalization across time performance - ' + sens)
    plt.ylabel('Training time (ms)')  # NO TRANSPOSE
    plt.xlabel('Testing time (ms)')  # NO TRANSPOSE
    plt.colorbar()
    if save_path is not None:
        plt.savefig(op.join(save_path, figname + sens), dpi=300)

    return fig

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
def apply_SVM_filter_16_items_epochs(subject, times=[x / 1000 for x in range(0, 750, 50)],window=False,train_test_different_blocks=True ):
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
    if train_test_different_blocks:
        n_folds = 2
        suf += 'train_test_different_blocks'

    SVM_results = np.load(op.join(SVM_results_path, suf+'SVM_results.npy'),allow_pickle=True).item()

    # ==== define the paths ==============
    meg_subject_dir = op.join(config.meg_dir, subject)
    fig_path = op.join(config.study_path, 'Figures', 'SVM') + op.sep
    extension = subject + '_1st_element_epo'
    fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Input: ", fname_in)

    # ====== loading the 16 items sequences epoched on the first element ===================
    epochs_1st_element = mne.read_epochs(fname_in, preload=True)
    epochs_1st_element = epochs_1st_element["TrialNumber > 10"]
    epochs_1st = {'mag': epochs_1st_element.copy().pick_types(meg='mag'),
                  'grad': epochs_1st_element.copy().pick_types(meg='grad'),
                  'eeg': epochs_1st_element.copy().pick_types(eeg=True, meg=False),
                  'all_chans':epochs_1st_element.copy().pick_types(eeg=True, meg=True)}

    # ====== compute the projections for each of the 3 types of sensors ===================
    for sens in ['mag', 'grad', 'eeg','all_chans']:

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
            print('Fold ' + str(fold_number + 1) + ' on %i...'%n_folds)
            start = time.time()
            test_indices = SVM_results[sens]['test_ind'][fold_number]
            epochs_sens_test = epochs_sens[test_indices]
            points = epochs_sens_test.time_as_index(times)

            for m in test_indices:

                seqID_m = epochs_sens[m].metadata['SequenceID'].values[0]
                run_m = epochs_sens[m].metadata['RunNumber'].values[0]
                trial_number_m = epochs_sens[m].metadata['TrialNumber'].values[0]  # this is the number of the trial, that will allow to determine which sequence within the run of 46 is the one that was left apart
                epochs_1st_sens_m = epochs_1st_sens['SequenceID == "%i" and RunNumber == %i and TrialNumber == %i' % (seqID_m, run_m, trial_number_m)]

                #if sens =="all_chans":
                #    epochs_1st_sens_m.pick_types(meg=True,eeg=True)

                if len(epochs_1st_sens_m.events) != 0:
                    data_1st_el_m = epochs_1st_sens_m.get_data()
                    SVM_to_data = np.squeeze(SVM_sens[fold_number].decision_function(data_1st_el_m))
                    print("The shape of SVM_to_data is ")
                    print(SVM_to_data.shape)
                    if not window:
                        for mm, point_of_interest in enumerate(points):
                            print(" The point of interest has index %i"%point_of_interest)
                            print(" === MAKE SURE THAT WHEN SELECTING SVM_to_data[point_of_interest,:] WE ARE INDEED CHOOSING THE TRAINING TIMES ===" )
                            epochs_1st_sens_m_filtered_data = SVM_to_data[point_of_interest,:]
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
                        epochs_1st_sens_m_filtered_data = np.mean(SVM_to_data[np.min(points):np.max(points),:],axis=0)
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
                    print('========================================================================================================================================')
                    print(' Epoch on first element for sequence %s Run number %i and Trial number %i was excluded by autoreject' % (seqID_m, run_m, trial_number_m))
                    print('========================================================================================================================================')
            end = time.time()
            elapsed = end - start
            print('... lasted: ' + str(elapsed) + ' s')

        dat = np.expand_dims(data_for_epoch_object, axis=1)
        info = mne.create_info(['SVM'], epochs_1st_sens.info['sfreq'])
        epochs_proj_sens = mne.EpochsArray(dat, info, tmin=-0.5)
        epochs_proj_sens.metadata = data_frame_meta

        if window:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + suf + '_SVM_on_16_items_test_window-epo.fif',overwrite=True)
        else:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + suf + '_SVM_on_16_items_test-epo.fif',overwrite=True)

    return True


def apply_SVM_filter_16_items_epochs_habituation(subject, times=[x / 1000 for x in range(0, 750, 50)],window = False,train_test_different_blocks=True ):
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
    if train_test_different_blocks:
        n_folds = 2
        suf += 'train_test_different_blocks'

    SVM_results = np.load(op.join(SVM_results_path, suf+'SVM_results.npy'),allow_pickle=True).item()

    # ==== define the paths ==============
    meg_subject_dir = op.join(config.meg_dir, subject)
    fig_path = op.join(config.study_path, 'Figures', 'SVM') + op.sep
    extension = subject + '_1st_element_epo'
    fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Input: ", fname_in)

    # ====== loading the 16 items sequences epoched on the first element ===================
    epochs_1st_element = mne.read_epochs(fname_in, preload=True)
    epochs_1st_element= epochs_1st_element["TrialNumber < 11"]
    epochs_1st = {'mag': epochs_1st_element.copy().pick_types(meg='mag'),
                  'grad': epochs_1st_element.copy().pick_types(meg='grad'),
                  'eeg': epochs_1st_element.copy().pick_types(eeg=True, meg=False),
                  'all_chans': epochs_1st_element.copy().pick_types(eeg=True, meg=True)}

    # ====== compute the projections for each of the 3 types of sensors ===================
    for sens in ['all_chans','mag', 'grad', 'eeg']:
    # for sens in ['all_chans','mag', 'grad', 'eeg']:

        SVM_sens = SVM_results[sens]['SVM']
        points = SVM_results[sens]['epochs'][0].time_as_index(times)

        epochs_1st_sens = epochs_1st[sens]

        # = we initialize the metadata
        data_frame_meta = pd.DataFrame([])
        n_habituation = epochs_1st_element.get_data().shape[0]
        data_for_epoch_object = np.zeros(
            (n_habituation* len(times), epochs_1st_sens.get_data().shape[2]))
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
                epochs_1st_sens_filtered_data = np.mean(epochs_1st_sens_filtered_data_4folds,axis=0).T
                data_for_epoch_object[n_habituation*mm:n_habituation*(mm+1),:] = epochs_1st_sens_filtered_data
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
                    np.mean(SVM_to_data[:,np.min(points):np.max(points), :], axis=1))

            # ==== now that we projected the 4 filters, we can average over the 4 folds ================
            data_for_epoch_object = np.mean(epochs_1st_sens_filtered_data_4folds, axis=0)

            metadata = epochs_1st_sens.metadata
            print("==== the length of the epochs_1st_sens.metadata to append is %i ===="%len(metadata))
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
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens +suf+ '_SVM_on_16_items_habituation_window-epo.fif',overwrite=True)
        else:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens +suf+ '_SVM_on_16_items_habituation-epo.fif',overwrite=True)

    return True


# ______________________________________________________________________________________________________________________
def plot_SVM_projection_for_seqID(epochs_list, sensor_type, seqID=1, SVM_filter_times=[x / 1000 for x in range(100, 700, 50)], save_path=None, color_mean=None, plot_noviolation=True):
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
            ax[n].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=len(SVM_filter_times), mode="expand", borderaxespad=0.)
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
    fig.suptitle('SVM %s - window %d-%dms - SequenceID_%d; N subjects = %d' % (sensor_type, win_tmin, win_tmax, seqID, len(epochs_list['test'])), fontsize=12)
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
            epochs_subset = epochs['SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(viol_pos) + '"']
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
def plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type, save_path=None):

    # window info, just for figure title
    win_tmin = epochs_list['test'][0][0].metadata.SVM_filter_tmin_window[0] * 1000
    win_tmax = epochs_list['test'][0][0].metadata.SVM_filter_tmax_window[0] * 1000

    fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True, sharey=False, constrained_layout=True)
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

    if sensor_type == 'mag':
        vmin = -1
        vmax = 1
        print("vmin = %0.02f, vmax = %0.02f"%(vmin, vmax))
    elif sensor_type == 'grad':
        vmin = -1
        vmax = 1
        print("vmin = %0.02f, vmax = %0.02f"%(vmin, vmax))
    elif sensor_type == 'eeg':
        vmin = -1
        vmax = 1
        print("vmin = %0.02f, vmax = %0.02f"%(vmin, vmax))
    elif sensor_type == 'all_chans':
        vmin = -1
        vmax = 1
        print("vmin = %0.02f, vmax = %0.02f"%(vmin, vmax))

    n = 0

    for seqID in range(1, 8):

        # this provides us with the position of the violations and the times
        epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
        times = epochs_seq_subset.times
        times = times + 0.3
        violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])

        # Average data from habituation trials
        y_list = []
        data_mean = []
        for epochs in epochs_list['hab']:
            epochs_subset = epochs['SequenceID == "' + str(seqID) + '"']
            y_list.append(np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()),axis=0))
            # y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='SVM').data))
            # y_list.append(np.squeeze(epochs_subset.average(picks='SVM').data))
        mean_hab = np.mean(y_list, axis=0)
        data_mean.append(mean_hab)

        # Average data from other trials
        for viol_pos in violpos_list:
            y_list = []
            for epochs in epochs_list['test']:
                epochs_subset = epochs[
                    'SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                y_list.append(np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()),axis=0))
                # y_list.append(np.squeeze(epochs_subset.average(picks='SVM').data))
            mean = np.mean(y_list, axis=0)
            data_mean.append(mean)

        width = 75
        # Add vertical lines, and "xY"
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
            txt = seqtxtXY[n][xx]
            ax[n].text(250*(xx+1)-125, width*6+(width/3), txt, horizontalalignment='center', fontsize=16)

        # return data_mean
        im = ax[n].imshow(data_mean, extent=[min(times)*1000, max(times)*1000, 0, 6*width], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        # ax[n].set_xlim(-500, 4250)
        # ax[n].legend(loc='upper left', fontsize=10)
        ax[n].set_yticks(np.arange(width/2, 6*width, width))
        ax[n].set_yticklabels(['Violation (pos. %d)' % violpos_list[4], 'Violation (pos. %d)' % violpos_list[3], 'Violation (pos. %d)' % violpos_list[2], 'Violation (pos. %d)' % violpos_list[1], 'Standard','Habituation'])
        ax[n].axvline(0, linestyle='-', color='black', linewidth=2)

        # add deviant marks
        for k in range(4):
            viol_pos = violpos_list[k+1]
            x = 250 * (viol_pos - 1)
            y1 = (4-k)*width
            y2 = (4-1-k)*width
            ax[n].plot([x, x], [y1, y2], linestyle='-', color='black', linewidth=6)
            ax[n].plot([x, x], [y1, y2], linestyle='-', color='yellow', linewidth=3)
        # add colorbar
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        # cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.5, pad=.2, aspect=10)
        cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.50, aspect=10, pad=.005)
        cb.ax.yaxis.set_offset_position('left')
        cb.set_label('a. u.')
        n += 1
    axes.ravel()[-1].set_xlabel('Time (ms)')

    figure = plt.gcf()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    return figure

# ______________________________________________________________________________________________________________________
def plot_SVM_projection_for_seqID_heatmap(epochs_list, sensor_type, seqID=1, SVM_filter_times=[x / 1000 for x in range(100, 700, 50)], save_path=None,vmin=None,vmax=None):

    # this provides us with the position of the violations and the times
    epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
    times = epochs_seq_subset.times
    violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])

    fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle('SVM %s - SequenceID_' % sensor_type + str(seqID) + ' N subjects = ' + str(len(epochs_list['test'])), fontsize=12)
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
            y_list.append(np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()),axis=0))
            # y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='SVM').data))
            # y_list.append(np.squeeze(epochs_subset.average(picks='SVM').data))
        mean = np.mean(y_list, axis=0)
        mean_all_SVM_times = np.vstack([mean_all_SVM_times, mean])
    width = 50
    for xx in range(16):
        ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
    ax[n].imshow(mean_all_SVM_times, origin='lower', extent=[min(times)*1000, max(times)*1000, 0, len(SVM_filter_times)*width], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax[n].set_yticks(np.arange(width/2, len(SVM_filter_times)*width, len(SVM_filter_times)*width/len(SVM_filter_times)))
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
                y_list.append(np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()),axis=0))
                # y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='SVM').data))
                # y_list.append(np.squeeze(epochs_subset.average(picks='SVM').data))
            mean = np.mean(y_list, axis=0)
            mean_all_SVM_times = np.vstack([mean_all_SVM_times, mean])

        width = 50
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
        ax[n].imshow(mean_all_SVM_times, origin='lower', extent=[min(times)*1000, max(times)*1000, 0, len(SVM_filter_times)*width], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        # ax[n].set_xlim(-500, 4250)
        # ax[n].legend(loc='upper left', fontsize=10)
        ax[n].set_yticks(np.arange(width/2, len(SVM_filter_times)*width, len(SVM_filter_times)*width/len(SVM_filter_times)))
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



#=========================================================================================================

class ZScoreEachChannel(TransformerMixin):
    """
    Z-score the data of each channel separately

    Input matrix: Epochs x Channels x TimePoints
    Output matrix: Epochs x Channels x TimePoints (same size as input)
    """

    #--------------------------------------------------
    def __init__(self, debug=False):
        self._debug = debug

    #--------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self

    #--------------------------------------------------
    def transform(self, x):
        result = np.zeros(x.shape)
        n_epochs, nchannels, ntimes = x.shape
        for c in range(nchannels):
            channel_data = x[:, c, :]
            m = np.mean(channel_data)
            sd = np.std(channel_data)
            if self._debug:
                print('ZScoreEachChannel: channel {:} m={:}, sd={:}'.format(c, m, sd))
            result[:, c, :] = (x[:, c, :]-m)/sd

        return result


#=========================================================================================================

class SlidingWindow(TransformerMixin):
    """
    Aggregate time points in a "sliding window" manner

    Input: Anything x Anything x Time points
    Output - if averaging: Unchanged x Unchanged x Windows
    Output - if not averaging: Windows x Unchanged x Unchanged x Window size
                Note that in this case, the output may not be a real matrix in case the last sliding window is smaller than the others
    """

    #--------------------------------------------------
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


    #--------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self

    #--------------------------------------------------
    def transform(self, x):
        x = np.array(x)
        assert len(x.shape) == 3
        n1, n2, n_time_points = x.shape

        #-- Get the start-end indices of each window
        min_window_size = self._min_window_size or self._window_size
        window_start = np.array(range(0, n_time_points-min_window_size+1, self._step))
        if len(window_start) == 0:
            #-- There are fewer than window_size time points
            raise Exception('There are only {:} time points, but at least {:} are required for the sliding window'.
                            format(n_time_points, self._min_window_size))
        window_end = window_start + self._window_size
        window_end[-1] = min(window_end[-1], n_time_points)  # make sure that the last window doesn't exceed the input size

        if self._debug:
            win_info = [(s, e, e-s) for s, e in zip(window_start, window_end)]
            print('SlidingWindow transformer: the start,end,length of each sliding window: {:}'.
                  format(win_info))
            if len(win_info) > 1 and win_info[0][2] != win_info[-1][2] and not self._average:
                print('SlidingWindow transformer: note that the last sliding window is smaller than the previous ones, ' +
                      'so the result will be a list of 3-dimensional matrices, with the last list element having ' +
                      'a different dimension than the previous elements. ' +
                      'This format is acceptable by the RiemannDissimilarity transformer')

        if self._average:
            #-- Average the data in each sliding window
            result = np.zeros((n1, n2, len(window_start)))
            for i in range(len(window_start)):
                result[:, :, i] = np.mean(x[:, :, window_start[i]:window_end[i]], axis=2)

        else:
            #-- Don't average the data in each sliding window - just copy it
            result = []
            for i in range(len(window_start)):
                result.append(x[:, :, window_start[i]:window_end[i]])

        return result


#=========================================================================================================

class AveragePerEvent(TransformerMixin):
    """
    This transformer averages all epochs that have the same label.
    It can also create several averages per event ID (based on independent sets of trials)

    Input matrix: Epochs x Channels x TimePoints
    Output matrix: Labels x Channels x TimePoints. If asked to create N results per event ID, the "labels"
                   dimension is multiplied accordingly.
    """

    #--------------------------------------------------
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


    #--------------------------------------------------
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


    #--------------------------------------------------
    def transform(self, x):

        x = np.array(x)

        result = []

        #-- Split the epochs by event ID.
        #-- x_per_event_id has a 3-dim matrix for each event ID
        x_per_event_id = [x[self._y == eid] for eid in self._curr_event_ids]

        #-- Check if there are enough epochs per event ID
        too_few_epochs = [len(e) < self._n_results_per_event for e in x_per_event_id]  # list of bool - one per event ID
        if sum(too_few_epochs) > self._max_events_with_missing_epochs:
            raise Exception('There are {:} event IDs with fewer than {:} epochs: {:}'.
                            format(sum(too_few_epochs), self._n_results_per_event,
                            self._curr_event_ids[np.where(too_few_epochs)[0]]))
        elif sum(too_few_epochs) > 0:
            print('WARNING (AveragePerEvent): There are {:} event IDs with fewer than {:} epochs: {:}'.
                  format(sum(too_few_epochs), self._n_results_per_event,
                         self._curr_event_ids[np.where(too_few_epochs)[0]]))

        #-- Do the actual aggregation
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

    #--------------------------------------------------
    def _aggregate(self, one_event_x):
        """
        Distribute the epochs of one_event_x into separate sets

        The function returns a list with self._n_results_per_event different sets.
        """

        if self._n_results_per_event == 1:
            #-- Aggregate all epochs into one result
            return [one_event_x]

        if len(one_event_x) >= self._n_results_per_event:

            #-- The number of epochs is sufficient to have at least one different epoch per result

            one_event_x = np.array(one_event_x)

            result = [[]] * self._n_results_per_event

            #-- First, distribute an equal number of epochs to each result
            n_in_epochs = len(one_event_x)
            in_epochs_inds = range(len(one_event_x))
            random.shuffle(in_epochs_inds)
            n_take_per_result = int(np.floor(n_in_epochs / self._n_results_per_event))
            for i in range(self._n_results_per_event):
                result[i] = list(one_event_x[in_epochs_inds[:n_take_per_result]])
                in_epochs_inds = in_epochs_inds[n_take_per_result:]

            #-- If some epochs remained, add each of them to a different result set
            n_remained = len(in_epochs_inds)
            for i in range(n_remained):
                result[i].append(one_event_x[in_epochs_inds[i]])

        else:

            #-- The number of epochs is too small: each result will consist of a single epoch, and epochs some will be duplicated

            #-- First, take all events that we have
            result = list(one_event_x)

            #-- Then pick random some epochs and duplicate them
            n_missing = self._n_results_per_event - len(result)
            epoch_inds = range(len(one_event_x))
            random.shuffle(epoch_inds)
            duplicated_inds = epoch_inds[:n_missing]
            result.extend(np.array(result)[duplicated_inds])

            result = [[x] for x in result]

        random.shuffle(result)

        return result



