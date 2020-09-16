import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import time
from mne.decoding import EMS


from sklearn.model_selection import StratifiedKFold
from scipy.ndimage.filters import gaussian_filter1d
from ABseq_func import *
import config
from scipy.stats import sem
from ABseq_func import utils  # why do we need this now ?? (error otherwise)
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

def generate_EMS_all_sequences(subject):
    """
    Generates the EMS filters for all the channel types using 4 folds. We save the training and testing indices as well as the epochs
    in order to be flexible for the later analyses.

    :param epochs:
    :param saving_directory:
    :return:
    """

    epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
    saving_directory = op.join(config.EMS_path, subject)
    utils.create_folder(saving_directory)

    epochs_balanced = epoching_funcs.balance_epochs_violation_positions(epochs)

    # ===== to train the filter do not consider the habituation trials to later test on them separately ================

    epochs_balanced = epochs_balanced["TrialNumber > 10"]

    # ==================================================================================================================

    epochs_balanced_mag = epochs_balanced.copy().pick_types(meg='mag')
    epochs_balanced_grad = epochs_balanced.copy().pick_types(meg='grad')
    epochs_balanced_eeg = epochs_balanced.copy().pick_types(eeg=True, meg=False)

    # ============== create the temporary labels that will allow us to create the training and testing sets in a nicely balanced way =======================

    metadata_epochs = epochs_balanced.metadata



    y_tmp = [int(metadata_epochs['SequenceID'].values[i] * 1000 + metadata_epochs['StimPosition'].values[i] * 10 +
                 metadata_epochs['ViolationOrNot'].values[i]) for i in range(len(epochs_balanced))]
    y_violornot = np.asarray(epochs_balanced.metadata['ViolationOrNot'].values)

    epochs_all = [epochs_balanced_mag, epochs_balanced_grad, epochs_balanced_eeg]
    sensor_types = ['mag', 'grad', 'eeg']
    EMS_results = {'mag': [], 'grad': [], 'eeg': []}

    for l in range(3):
        senso = sensor_types[l]
        epochs_senso = epochs_all[l]
        X_data = epochs_senso.get_data()

        # ======= create the 4 EMS spatial filters ========
        All_EMS = []
        training_inds = []
        testing_inds = []

        for train, test in StratifiedKFold(n_splits=4).split(X_data, y_tmp):
            # we split in training and testing sets using y_tmp because it allows us to balance with respect to all our constraints

            # Initialize EMS transformer
            ems = EMS()
            X_scaled = X_data / np.std(X_data[train])
            # Fit and store the spatial filters
            ems.fit(X_scaled[train], y_violornot[train])
            # Store filters for future plotting
            All_EMS.append(ems)
            training_inds.append(train)
            testing_inds.append(test)

        EMS_results[senso] = {'EMS': All_EMS, 'train_ind': training_inds, 'test_ind': testing_inds,
                              'epochs': epochs_all[l]}
    np.save(op.join(saving_directory, 'EMS_results.npy'), EMS_results)


def GAT_EMS(subject):
    """
    The EMS at a training times are tested at testing times. Allows to obtain something similar to the GAT from decoding.
    Dictionnary contains the GAT for each sequence separately. GAT_all contains the average over all the sequences
    :param EMS_results: output of generate_EMS_all_sequences
    :return: GAT averaged over the 4 classification folds
    """

    saving_directory = op.join(config.EMS_path, subject)
    EMS_results = np.load(op.join(saving_directory, 'EMS_results.npy'), allow_pickle=True).item()

    GAT_sens_seq = {sens: [] for sens in ['eeg', 'mag', 'grad']}

    for sens in ['eeg', 'mag', 'grad']:
        print(sens)
        GAT_all = []
        GAT_per_sens_and_seq = {'SeqID_%i' % i: [] for i in range(1, 8)}

        epochs_sens = EMS_results[sens]['epochs']
        n_times = epochs_sens.get_data().shape[-1]
        EMS_sens = EMS_results[sens]['EMS']

        for k in range(1, 8):
            seqID = 'SeqID_%i' % k
            GAT_seq = np.zeros((4, n_times, n_times))
            for fold_number in range(4):
                test_indices = EMS_results[sens]['test_ind'][fold_number]
                epochs_sens_test = epochs_sens[test_indices]
                inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                        epochs_sens_test.metadata['ViolationOrNot'].values == 0))[0]
                inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                        epochs_sens_test.metadata['ViolationOrNot'].values == 1))[0]
                X = epochs_sens_test.get_data()
                X_scaled = X / np.std(X)
                for time_train in range(n_times):
                    for time_test in range(n_times):
                        GAT_each_epoch = np.dot(EMS_sens[fold_number].filters_[:, time_train],
                                                X_scaled[:, :, time_test].T)
                        GAT_seq[fold_number, time_train, time_test] = np.mean(
                            GAT_each_epoch[inds_seq_noviol]) - np.mean(
                            GAT_each_epoch[inds_seq_viol])
            # now average across the 4 folds
            GAT_seq_avg = np.mean(GAT_seq, axis=0)
            GAT_per_sens_and_seq[seqID] = GAT_seq_avg
            GAT_all.append(GAT_seq_avg)

        GAT_sens_seq[sens] = GAT_per_sens_and_seq
        GAT_sens_seq[sens]['average_all_sequences'] = np.mean(GAT_all, axis=0)
        times = epochs_sens_test.times

    GAT_results = {'GAT': GAT_sens_seq, 'times': times}
    np.save(op.join(saving_directory, 'GAT_results.npy'), GAT_results)


def GAT_EMS_4pos(subject):
    """
    The EMS at a training times are tested at testing times. Allows to obtain something similar to the GAT from decoding.
    Dictionnary contains the GAT for each sequence separately. GAT_all contains the average over all the sequences
    :param EMS_results: output of generate_EMS_all_sequences
    :return: GAT averaged over the 4 classification folds
    """

    saving_directory = op.join(config.EMS_path, subject)
    EMS_results = np.load(op.join(saving_directory, 'EMS_results.npy'), allow_pickle=True).item()

    GAT_sens_seq = {sens: [] for sens in ['eeg', 'mag', 'grad']}

    for sens in ['eeg', 'mag', 'grad']:
        print(sens)
        GAT_all = []
        GAT_per_sens_and_seq = {'SeqID_%i' % i: [] for i in range(1, 8)}
        epochs_sens = EMS_results[sens]['epochs']
        n_times = epochs_sens.get_data().shape[-1]
        EMS_sens = EMS_results[sens]['EMS']

        for k in range(1, 8):
            seqID = 'SeqID_%i' % k
            # extract the 4 positions of violation
            violpos_list = np.unique(epochs_sens['SequenceID == %i' % k].metadata['ViolationInSequence'])[1:]
            GAT_seq = np.zeros((4, n_times, n_times, 4))
            for fold_number in range(4):
                for nn, pos_viol in enumerate(violpos_list):
                    test_indices = EMS_results[sens]['test_ind'][fold_number]
                    epochs_sens_test = epochs_sens[test_indices]
                    inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                            epochs_sens_test.metadata['ViolationOrNot'].values == 0) & (
                                                       epochs_sens_test.metadata['StimPosition'].values == pos_viol))[0]
                    inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == k) & (
                            epochs_sens_test.metadata['ViolationOrNot'].values == 1) & (
                                                     epochs_sens_test.metadata['StimPosition'].values == pos_viol))[0]
                    X = epochs_sens_test.get_data()
                    X_scaled = X / np.std(X)
                    for time_train in range(n_times):
                        for time_test in range(n_times):
                            GAT_each_epoch = np.dot(EMS_sens[fold_number].filters_[:, time_train],
                                                    X_scaled[:, :, time_test].T)
                            GAT_seq[fold_number, time_train, time_test, nn] = np.mean(
                                GAT_each_epoch[inds_seq_noviol]) - np.mean(
                                GAT_each_epoch[inds_seq_viol])
            # now average across the 4 folds
            GAT_seq_avg = np.mean(GAT_seq, axis=0)
            GAT_per_sens_and_seq[seqID] = GAT_seq_avg
            GAT_all.append(GAT_seq_avg)

        GAT_sens_seq[sens] = GAT_per_sens_and_seq
        GAT_sens_seq[sens]['average_all_sequences'] = np.mean(GAT_all, axis=0)
        times = epochs_sens_test.times

    GAT_results = {'GAT': GAT_sens_seq, 'times': times}
    np.save(op.join(saving_directory, 'GAT_results_4pos.npy'), GAT_results)


def EMS_applied_to_epochs(EMS_results, sequenceID=None):
    """
    This function applies the spatial filters to the epochs for all the sequence IDS or just some of them.
    It returns the projection, for each sensor type, and the codes indicating if it was violated or not, for each sensor type.
    The third thing returned are the times.

    :param EMS_results:
    :param sequenceID:
    :return:
    """

    y_violornot = {'eeg': [], 'grad': [], 'mag': []}
    X_transform = {'eeg': [], 'grad': [], 'mag': []}

    for sens in {'eeg', 'grad', 'mag'}:
        print(sens)
        EMS_sens = EMS_results[sens]['EMS']
        epochs_sens = EMS_results[sens]['epochs']
        n_epochs, n_channels, n_times = epochs_sens.get_data().shape
        X_transform[sens] = np.zeros((n_epochs, n_times))
        for fold_number in range(4):
            test_indices = EMS_results[sens]['test_ind'][fold_number]
            if sequenceID is not None:
                epochs_sens_test = epochs_sens[test_indices]['SequenceID == %i' % sequenceID]
            else:
                epochs_sens_test = epochs_sens[test_indices]

            y_violornot[sens].append(np.asarray(epochs_sens.metadata['ViolationOrNot'].values))
            X = epochs_sens_test.get_data()
            X_scaled = X / np.std(X)
            # Generate the transformed data
            X_transform[sens][test_indices] = EMS_sens[fold_number].transform(X_scaled)

    times = epochs_sens_test.times

    return X_transform, y_violornot, times


def plot_GAT_EMS(GAT_avg, times, sens='mag', save_path=None, figname='GAT_', vmin=-1.5, vmax=1.5):
    minT = np.min(times) * 1000
    maxT = np.max(times) * 1000
    fig = plt.figure()
    # plt.imshow(GAT_avg.T, origin='lower', extent=[minT,maxT,minT,maxT],cmap='RdBu_r',vmin=vmin,vmax=vmax) # NO TRANSPOSE
    plt.imshow(GAT_avg, origin='lower', extent=[minT, maxT, minT, maxT], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    # -----# ADD LINES ?
    plt.axvline(0, linestyle='-', color='black', linewidth=1)
    plt.axhline(0, linestyle='-', color='black', linewidth=1)
    # for xx in range(3):
    #     plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    #     plt.axhline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    plt.plot([minT, maxT], [minT, maxT], linestyle='--', color='black', linewidth=1)
    # -----#
    plt.title('Average EMS signal - ' + sens)
    plt.ylabel('Training time (ms)')  # NO TRANSPOSE
    plt.xlabel('Testing time (ms)')  # NO TRANSPOSE
    # plt.xlabel('Training time (ms)')  # NO TRANSPOSE
    # plt.ylabel('Testing time (ms)')  # NO TRANSPOSE
    plt.colorbar()
    # plt.show()
    if save_path is not None:
        plt.savefig(op.join(save_path, figname + sens), dpi=300)

    return fig


def plot_EMS_average(X_transform, y_violornot, times, fig_path=None, figname='Average_EMS_all_sequences_'):
    fig = plt.figure()
    plt.title('Average EMS signal - ')
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


def apply_EMS_filter_16_items_epochs(subject, times=[x / 1000 for x in range(0, 750, 50)],window=False):
    """
    Function to apply the EMS filters built on all the sequences the 16 item sequences
    :param subject:
    :param times:the different times at which we want to apply the filter (if window is False). Otherwise (window = True),
    min(times) and max(times) define the time window on which we average the spatial filter.
    :param window: set to True if you want to average the spatial filter over a window.
    :return:
    """

    # ==== load the ems results ==============
    EMS_results_path = op.join(config.EMS_path, subject)
    EMS_results = np.load(op.join(EMS_results_path, 'EMS_results.npy'),allow_pickle=True).item()

    # ==== define the paths ==============
    meg_subject_dir = op.join(config.meg_dir, subject)
    fig_path = op.join(config.study_path, 'Figures', 'EMS') + op.sep
    extension = subject + '_1st_element_epo'
    fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Input: ", fname_in)

    # ====== loading the 16 items sequences epoched on the first element ===================
    epochs_1st_element = mne.read_epochs(fname_in, preload=True)
    epochs_1st_element = epochs_1st_element["TrialNumber > 10"]
    epochs_1st = {'mag': epochs_1st_element.copy().pick_types(meg='mag'),
                  'grad': epochs_1st_element.copy().pick_types(meg='grad'),
                  'eeg': epochs_1st_element.copy().pick_types(eeg=True, meg=False)}

    # ====== compute the projections for each of the 3 types of sensors ===================
    for sens in ['mag', 'grad', 'eeg']:

        EMS_sens = EMS_results[sens]['EMS']
        epochs_sens = EMS_results[sens]['epochs']
        epochs_1st_sens = epochs_1st[sens]

        # = we initialize the metadata
        data_frame_meta = pd.DataFrame([])
        elapsed = 0
        data_for_epoch_object = np.zeros(
            (epochs_sens.get_data().shape[0] * len(times), epochs_1st_sens.get_data().shape[2]))
        if window:
            data_for_epoch_object = np.zeros(
                (EMS_results[sens]['epochs'].get_data().shape[0], epochs_1st_sens.get_data().shape[2]))

        # ===============================
        counter = 0
        for fold_number in range(4):

            print('Fold ' + str(fold_number + 1) + ' on 4...')
            start = time.time()
            test_indices = EMS_results[sens]['test_ind'][fold_number]
            epochs_sens_test = epochs_sens[test_indices]
            points = epochs_sens_test.time_as_index(times)

            for m in test_indices:

                seqID_m = epochs_sens[m].metadata['SequenceID'].values[0]
                run_m = epochs_sens[m].metadata['RunNumber'].values[0]
                trial_number_m = epochs_sens[m].metadata['TrialNumber'].values[0]  # this is the number of the trial, that will allow to determine which sequence within the run of 46 is the one that was left apart
                epochs_1st_sens_m = epochs_1st_sens['SequenceID == "%i" and RunNumber == %i and TrialNumber == %i' % (seqID_m, run_m, trial_number_m)]

                if len(epochs_1st_sens_m.events) != 0:
                    data_1st_el_m = epochs_1st_sens_m.get_data()
                    if not window:
                        for mm, point_of_interest in enumerate(points):
                            epochs_1st_sens_m_filtered_data = np.dot(EMS_sens[fold_number].filters_[:, point_of_interest],data_1st_el_m.T)
                            data_for_epoch_object[counter, :] = np.squeeze(epochs_1st_sens_m_filtered_data)
                            metadata_m = epochs_1st_sens_m.metadata
                            metadata_m['EMS_filter_datapoint'] = int(point_of_interest)
                            metadata_m['EMS_filter_time'] = times[mm]
                            data_frame_meta = data_frame_meta.append(metadata_m)
                            counter += 1
                    else:
                        filter_fold = np.mean(EMS_sens[fold_number].filters_[:, np.min(points):np.max(points)],axis=-1)
                        epochs_1st_sens_m_filtered_data = np.dot(filter_fold[np.newaxis,:],
                                                                 data_1st_el_m)
                        data_for_epoch_object[counter, :] = np.squeeze(epochs_1st_sens_m_filtered_data)
                        metadata_m = epochs_1st_sens_m.metadata
                        metadata_m['EMS_filter_min_datapoint'] = np.min(points)
                        metadata_m['EMS_filter_max_datapoint'] = np.max(points)
                        metadata_m['EMS_filter_tmin_window'] = times[0]
                        metadata_m['EMS_filter_tmax_window'] = times[-1]
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
        info = mne.create_info(['EMS'], epochs_1st_sens.info['sfreq'])
        epochs_proj_sens = mne.EpochsArray(dat, info, tmin=-0.5)
        epochs_proj_sens.metadata = data_frame_meta
        if window:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + '_filter_on_16_items_test_window-epo.fif',overwrite=True)
        else:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + '_filter_on_16_items_test-epo.fif',overwrite=True)


    return True


def apply_EMS_filter_16_items_epochs_habituation(subject, times=[x / 1000 for x in range(0, 750, 50)],window = False):
    """
    Function to apply the EMS filters on the habituation trials. It is simpler than the previous function as we don't have to select the specific
    trials according to the folds.
    :param subject:
    :param times:
    :return:
    """

    # ==== load the ems results ==============
    EMS_results_path = op.join(config.EMS_path, subject)
    EMS_results = np.load(op.join(EMS_results_path, 'EMS_results.npy'), allow_pickle=True).item()

    # ==== define the paths ==============
    meg_subject_dir = op.join(config.meg_dir, subject)
    fig_path = op.join(config.study_path, 'Figures', 'EMS') + op.sep
    extension = subject + '_1st_element_epo'
    fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Input: ", fname_in)

    # ====== loading the 16 items sequences epoched on the first element ===================
    epochs_1st_element = mne.read_epochs(fname_in, preload=True)
    epochs_1st_element= epochs_1st_element["TrialNumber < 11"]
    epochs_1st = {'mag': epochs_1st_element.copy().pick_types(meg='mag'),
                  'grad': epochs_1st_element.copy().pick_types(meg='grad'),
                  'eeg': epochs_1st_element.copy().pick_types(eeg=True, meg=False)}

    # ====== compute the projections for each of the 3 types of sensors ===================
    for sens in ['mag', 'grad', 'eeg']:

        EMS_sens = EMS_results[sens]['EMS']
        points = EMS_results[sens]['epochs'][0].time_as_index(times)

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
                for fold_number in range(4):
                    epochs_1st_sens_filtered_data_4folds.append(np.dot(EMS_sens[fold_number].filters_[:, point_of_interest],data_1st_el_m.T))
                # ==== now that we projected the 4 filters, we can average over the 4 folds ================
                epochs_1st_sens_filtered_data = np.mean(epochs_1st_sens_filtered_data_4folds,axis=0).T
                data_for_epoch_object[n_habituation*mm:n_habituation*(mm+1),:] = epochs_1st_sens_filtered_data
                metadata_m = epochs_1st_sens.metadata
                metadata_m['EMS_filter_datapoint'] = int(point_of_interest)
                metadata_m['EMS_filter_time'] = times[mm]
                data_frame_meta = data_frame_meta.append(metadata_m)
        else:
            epochs_1st_sens_filtered_data_4folds = []
            for fold_number in range(4):
                filter_fold = np.mean(EMS_sens[fold_number].filters_[:, np.min(points):np.max(points)],axis=-1)
                epochs_1st_sens_filtered_data_4folds.append(
                    np.dot(filter_fold, data_1st_el_m.T))
            # ==== now that we projected the 4 filters, we can average over the 4 folds ================
            data_for_epoch_object = np.mean(epochs_1st_sens_filtered_data_4folds, axis=0).T

            metadata = epochs_1st_sens.metadata
            metadata['EMS_filter_min_datapoint'] = np.min(points)
            metadata['EMS_filter_max_datapoint'] = np.max(points)
            metadata['EMS_filter_tmin_window'] = times[0]
            metadata['EMS_filter_tmax_window'] = times[-1]
            data_frame_meta = data_frame_meta.append(metadata)

        dat = np.expand_dims(data_for_epoch_object, axis=1)
        info = mne.create_info(['EMS'], epochs_1st_sens.info['sfreq'])
        epochs_proj_sens = mne.EpochsArray(dat, info, tmin=-0.5)
        epochs_proj_sens.metadata = data_frame_meta
        if window:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + '_filter_on_16_items_habituation_window-epo.fif',overwrite=True)
        else:
            epochs_proj_sens.save(meg_subject_dir + op.sep + sens + '_filter_on_16_items_habituation-epo.fif',overwrite=True)

    return True


def plot_EMS_projection_for_seqID_old(epochs_list, sensor_type, seqID=1,EMS_filter_times=[x / 1000 for x in range(-100, 701, 50)], save_path=None,color_mean=None):
    """
    This will allow to plot the Projections of the EMS on the 4 positions of the violations.

    :param epochs:
    :param sensor_type:
    :param seqID:
    :param EMS_filter_times:
    :return:
    """

    # this provides us with the position of the violations and the times
    epochs_seq_subset = epochs_list[0]['SequenceID == "' + str(seqID) + '"']
    times = epochs_seq_subset.times
    violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])[1:]

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle('EMS %s - SequenceID_' % sensor_type + str(seqID) + ' N subjects = ' + str(len(epochs_list)),
                 fontsize=12)
    ax = axes.ravel()[::1]
    n = 0
    for viol_pos in violpos_list:
        for ii, point_of_interest in enumerate(EMS_filter_times):
            y_list = []
            for epochs in epochs_list:
                epochs_subset = epochs['SequenceID == "' + str(seqID)
                                       + '" and EMS_filter_time == "' + str(point_of_interest)
                                       + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
                # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))

            mean = np.mean(y_list, axis=0)
            ub = mean + sem(y_list, axis=0)
            lb = mean - sem(y_list, axis=0)

            for xx in range(16):
                ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            ax[n].fill_between(times * 1000, ub, lb, color=color_mean[ii], alpha=.2)
            ax[n].plot(times * 1000, mean, color=color_mean[ii], linewidth=1.5, label=str(point_of_interest))
            ax[n].set_xlim(-500, 4250)
            ax[n].legend(loc='upper left', fontsize=10)
            ax[n].axvline(0, linestyle='-', color='black', linewidth=2)

            ax[n].axvline(250 * (viol_pos - 1), linestyle='-', color='red', linewidth=2)
        n += 1
    axes.ravel()[-1].set_xlabel('Time (ms)')

    figure = plt.gcf()
    if save_path is not None:
        print('Saving ' + save_path)
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    return figure


def plot_EMS_projection_for_seqID(epochs_list, sensor_type, seqID=1, EMS_filter_times=[x / 1000 for x in range(100, 700, 50)], save_path=None, color_mean=None, plot_noviolation=True):
    """
    This will allow to plot the Projections of the EMS on the 4 positions of the violations.

    :param epochs:
    :param sensor_type:
    :param seqID:
    :param EMS_filter_times:
    :return:
    """

    # this provides us with the position of the violations and the times
    epochs_seq_subset = epochs_list[0]['SequenceID == "' + str(seqID) + '"']
    times = epochs_seq_subset.times
    violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])[1 - 1 * plot_noviolation:]

    fig, axes = plt.subplots(4 + 1 * plot_noviolation, 1, figsize=(12, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    fig.suptitle('EMS %s - SequenceID_' % sensor_type + str(seqID) + ' N subjects = ' + str(len(epochs_list)),
                 fontsize=12)
    ax = axes.ravel()[::1]
    n = 0
    for viol_pos in violpos_list:
        for ii, point_of_interest in enumerate(EMS_filter_times):
            y_list = []
            for epochs in epochs_list:
                epochs_subset = epochs['SequenceID == "' + str(seqID)
                                       + '" and EMS_filter_time == "' + str(point_of_interest)
                                       + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
                # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))

            mean = np.mean(y_list, axis=0)
            ub = mean + sem(y_list, axis=0)
            lb = mean - sem(y_list, axis=0)

            for xx in range(16):
                ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            ax[n].fill_between(times * 1000, ub, lb, color=color_mean[ii], alpha=.2)
            ax[n].plot(times * 1000, mean, color=color_mean[ii], linewidth=1.5, label=str(point_of_interest))
            ax[n].set_xlim(-500, 4250)
            # ax[n].legend(loc='upper left', fontsize=10)
            ax[n].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=len(EMS_filter_times), mode="expand", borderaxespad=0.)
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


def plot_EMS_projection_for_seqID_window(epochs_list, sensor_type, seqID=1, save_path=None):
    # this provides us with the position of the violations and the times
    epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
    times = epochs_seq_subset.times
    violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])

    # window info, just for figure title
    win_tmin = epochs_list['test'][0][0].metadata.EMS_filter_tmin_window[0] * 1000
    win_tmax = epochs_list['test'][0][0].metadata.EMS_filter_tmax_window[0] * 1000

    fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle('EMS %s - window %d-%dms - SequenceID_%d; N subjects = %d' % (sensor_type, win_tmin, win_tmax, seqID, len(epochs_list['test'])), fontsize=12)
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
        y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
        # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))
    mean_hab = np.mean(y_list, axis=0)
    ub_hab = mean_hab + sem(y_list, axis=0)
    lb_hab = mean_hab - sem(y_list, axis=0)
    for xx in range(16):
        ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax[n].axhline(0, linestyle='-', color='black', linewidth=0.5)
    ax[n].fill_between(times * 1000, ub_hab, lb_hab, color='blue', alpha=.2)
    ax[n].plot(times * 1000, mean_hab, color='blue', linewidth=1.5)
    ax[n].set_xlim(-500, 4250)
    # ax[n].legend(loc='upper left', fontsize=10)
    # ax[n].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=len(EMS_filter_times), mode="expand", borderaxespad=0.)
    ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
    n += 1


# #==== TEST:  ADDING HAB AND STAND CURVES IN ALL GRAPHS...====#
# y_list = []
# for epochs in epochs_list['test']:
#     epochs_subset = epochs['SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(0) + '"']
#     y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
#     # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))
# mean_std = np.mean(y_list, axis=0)
# ub_std = mean_std + sem(y_list, axis=0)
# lb_std = mean_std - sem(y_list, axis=0)
# for viol_pos in violpos_list:
#     y_list = []
#     for epochs in epochs_list['test']:
#         epochs_subset = epochs['SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(viol_pos) + '"']
#         y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
#         # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))
#     mean = np.mean(y_list, axis=0)
#     ub = mean + sem(y_list, axis=0)
#     lb = mean - sem(y_list, axis=0)
#     # mean -= mean_std  # SUBSTRACT STANDARDS?
#     for xx in range(16):
#         ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
#     ax[n].axhline(0, linestyle='-', color='black', linewidth=0.5)
#     # ax[n].fill_between(times * 1000, ub, lb, color='blue', alpha=.2)
#     ax[n].plot(times * 1000, mean_hab, color='blue', linewidth=1)
#     ax[n].plot(times * 1000, mean_std, color='green', linewidth=1)
#     ax[n].plot(times * 1000, mean, color='red', linewidth=1.5)
#     ax[n].set_xlim(-500, 4250)
#     # ax[n].legend(loc='upper left', fontsize=10)
#     # ax[n].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=len(EMS_filter_times), mode="expand", borderaxespad=0.)
#     ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
#     if viol_pos != 0:
#         ax[n].axvline(250 * (viol_pos - 1), linestyle='-', color='red', linewidth=2)
#     n += 1
# axes.ravel()[-1].set_xlabel('Time (ms)')
# #=========================#

    for viol_pos in violpos_list:
        y_list = []
        for epochs in epochs_list['test']:
            epochs_subset = epochs['SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(viol_pos) + '"']
            y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
            # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))
        mean = np.mean(y_list, axis=0)
        ub = mean + sem(y_list, axis=0)
        lb = mean - sem(y_list, axis=0)
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        ax[n].axhline(0, linestyle='-', color='black', linewidth=0.5)
        ax[n].fill_between(times * 1000, ub, lb, color='blue', alpha=.2)
        ax[n].plot(times * 1000, mean, color='blue', linewidth=1.5)
        ax[n].set_xlim(-500, 4250)
        # ax[n].legend(loc='upper left', fontsize=10)
        # ax[n].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=len(EMS_filter_times), mode="expand", borderaxespad=0.)
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


def plot_EMS_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type, save_path=None):

    # window info, just for figure title
    win_tmin = epochs_list['test'][0][0].metadata.EMS_filter_tmin_window[0] * 1000
    win_tmax = epochs_list['test'][0][0].metadata.EMS_filter_tmax_window[0] * 1000

    fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True, sharey=False, constrained_layout=True)
    fig.suptitle('EMS %s - window %d-%dms; N subjects = %d' % (
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
        vmin = -5e-13
        vmax = 5e-13
    elif sensor_type == 'grad':
        vmin = -1.8e-11
        vmax = 1.8e-11
    elif sensor_type == 'eeg':
        vmin = -1e-5
        vmax = 1e-5
    n = 0

    for seqID in range(1, 8):

        # this provides us with the position of the violations and the times
        epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
        times = epochs_seq_subset.times
        violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])

        # Average data from habituation trials
        y_list = []
        data_mean = []
        for epochs in epochs_list['hab']:
            epochs_subset = epochs['SequenceID == "' + str(seqID) + '"']
            y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
            # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))
        mean_hab = np.mean(y_list, axis=0)
        data_mean.append(mean_hab)

        # Average data from other trials
        for viol_pos in violpos_list:
            y_list = []
            for epochs in epochs_list['test']:
                epochs_subset = epochs[
                    'SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
                # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))
            mean = np.mean(y_list, axis=0)
            data_mean.append(mean)

        width = 75
        # Add vertical lines, and "xY"
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
            txt = seqtxtXY[n][xx]
            ax[n].text(250*(xx+1)-125, width*6+(width/3), txt, horizontalalignment='center', fontsize=16)
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

def plot_EMS_projection_for_seqID_heatmap(epochs_list, sensor_type, seqID=1, EMS_filter_times=[x / 1000 for x in range(100, 700, 50)], save_path=None,):

    # this provides us with the position of the violations and the times
    epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
    times = epochs_seq_subset.times
    violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])

    fig, axes = plt.subplots(6, 1, figsize=(12, 9), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle('EMS %s - SequenceID_' % sensor_type + str(seqID) + ' N subjects = ' + str(len(epochs_list['test'])), fontsize=12)
    ax = axes.ravel()[::1]
    ax[0].set_title('Habituation trials', loc='left', weight='bold')
    ax[1].set_title('Standard test trials', loc='left', weight='bold')
    ax[2].set_title('Trials with deviant pos %d' % violpos_list[1], loc='left', weight='bold')
    ax[3].set_title('Trials with deviant pos %d' % violpos_list[2], loc='left', weight='bold')
    ax[4].set_title('Trials with deviant pos %d' % violpos_list[3], loc='left', weight='bold')
    ax[5].set_title('Trials with deviant pos %d' % violpos_list[4], loc='left', weight='bold')
    if sensor_type == 'mag':
        vmin = -5e-13
        vmax = 5e-13
    elif sensor_type == 'grad':
        vmin = -1.8e-11
        vmax = 1.8e-11
    elif sensor_type == 'eeg':
        vmin = -1e-5
        vmax = 1e-5
    n = 0
    # First, plot habituation trials
    mean_all_ems_times = np.empty((0, len(times)), int)
    for ii, point_of_interest in enumerate(EMS_filter_times):
        y_list = []
        for epochs in epochs_list['hab']:
            epochs_subset = epochs['SequenceID == "' + str(seqID)
                                   + '" and EMS_filter_time == "' + str(point_of_interest) + '"']
            y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
            # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))
        mean = np.mean(y_list, axis=0)
        mean_all_ems_times = np.vstack([mean_all_ems_times, mean])
    width = 50
    for xx in range(16):
        ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
    ax[n].imshow(mean_all_ems_times, origin='lower', extent=[min(times)*1000, max(times)*1000, 0, len(EMS_filter_times)*width], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax[n].set_yticks(np.arange(width/2, len(EMS_filter_times)*width, len(EMS_filter_times)*width/len(EMS_filter_times)))
    ax[n].set_yticklabels(EMS_filter_times)
    ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
    n += 1

    # Then, plot all other trials
    for viol_pos in violpos_list:
        mean_all_ems_times = np.empty((0, len(times)), int)
        for ii, point_of_interest in enumerate(EMS_filter_times):
            y_list = []
            for epochs in epochs_list['test']:
                epochs_subset = epochs['SequenceID == "' + str(seqID)
                                       + '" and EMS_filter_time == "' + str(point_of_interest)
                                       + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
                # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))
            mean = np.mean(y_list, axis=0)
            mean_all_ems_times = np.vstack([mean_all_ems_times, mean])

        width = 50
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
        ax[n].imshow(mean_all_ems_times, origin='lower', extent=[min(times)*1000, max(times)*1000, 0, len(EMS_filter_times)*width], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        # ax[n].set_xlim(-500, 4250)
        # ax[n].legend(loc='upper left', fontsize=10)
        ax[n].set_yticks(np.arange(width/2, len(EMS_filter_times)*width, len(EMS_filter_times)*width/len(EMS_filter_times)))
        ax[n].set_yticklabels(EMS_filter_times)
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

def plot_EMS_projection_for_seqID_heatmap_old(epochs_list, sensor_type, seqID=1, EMS_filter_times=[x / 1000 for x in range(100, 700, 50)], save_path=None, plot_noviolation=True):
    """
    This will allow to plot the Projections of the EMS on the 4 positions of the violations.

    :param epochs:
    :param sensor_type:
    :param seqID:
    :param EMS_filter_times:
    :return:
    """

    # this provides us with the position of the violations and the times
    epochs_seq_subset = epochs_list[0]['SequenceID == "' + str(seqID) + '"']
    times = epochs_seq_subset.times
    violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])[1 - 1 * plot_noviolation:]

    fig, axes = plt.subplots(4 + 1 * plot_noviolation, 1, figsize=(12, 8), sharex=True, sharey=True,
                             constrained_layout=True)
    fig.suptitle('EMS %s - SequenceID_' % sensor_type + str(seqID) + ' N subjects = ' + str(len(epochs_list)), fontsize=12)
    ax = axes.ravel()[::1]
    ax[0].set_title('Standard trials', loc='left', weight='bold')
    ax[1].set_title('Trials with deviant pos %d' % violpos_list[1], loc='left', weight='bold')
    ax[2].set_title('Trials with deviant pos %d' % violpos_list[2], loc='left', weight='bold')
    ax[3].set_title('Trials with deviant pos %d' % violpos_list[3], loc='left', weight='bold')
    ax[4].set_title('Trials with deviant pos %d' % violpos_list[4], loc='left', weight='bold')
    if sensor_type == 'mag':
        vmin = -5e-13
        vmax = 5e-13
    elif sensor_type == 'grad':
        vmin = -1.8e-11
        vmax = 1.8e-11
    elif sensor_type == 'eeg':
        vmin = -1e-5
        vmax = 1e-5
    n = 0
    for viol_pos in violpos_list:
        mean_all_ems_times = np.empty((0, len(times)), int)
        for ii, point_of_interest in enumerate(EMS_filter_times):
            y_list = []
            for epochs in epochs_list:
                epochs_subset = epochs['SequenceID == "' + str(seqID)
                                       + '" and EMS_filter_time == "' + str(point_of_interest)
                                       + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                y_list.append(np.squeeze(epochs_subset.savgol_filter(20).average(picks='EMS').data))
                # y_list.append(np.squeeze(epochs_subset.average(picks='EMS').data))

            mean = np.mean(y_list, axis=0)
            mean_all_ems_times = np.vstack([mean_all_ems_times, mean])

        width = 50
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
        ax[n].imshow(mean_all_ems_times, origin='lower', extent=[min(times)*1000, max(times)*1000, 0, len(EMS_filter_times)*width], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        # ax[n].set_xlim(-500, 4250)
        # ax[n].legend(loc='upper left', fontsize=10)
        ax[n].set_yticks(np.arange(width/2, len(EMS_filter_times)*width, len(EMS_filter_times)*width/len(EMS_filter_times)))
        ax[n].set_yticklabels(EMS_filter_times)
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
