# This module contains functions for epoching the data and for loading the saved epochs.
import csv
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import config
import os.path as op
import mne
import glob
import warnings
from autoreject import AutoReject
import pickle


def read_info_csv(filename):
    presented_seq = []
    violation_position = []

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            presented_seq.append(row['Presented_sequence'])
            violation_position.append(int(row['Position_Violation']))

    return presented_seq, violation_position


def load_surprise_mat(config):
    path = config.study_path + os.path.sep + 'surprise_values' + os.path.sep

    surprise_not_violated = sio.loadmat(path + 'surprise_not_viol.mat')
    surprise_viol = sio.loadmat(path + 'surprise_viol.mat')

    surprise_not_violated = surprise_not_violated['origSurp']
    surprise_viol = surprise_viol['devSurp']

    return surprise_not_violated, surprise_viol


def get_seqID(presented_sequences):
    if presented_sequences[0] == '0000000000000000' or presented_sequences[0] == '1111111111111111':
        seqID = 1
        complexity = 4
        sequence_entropy = np.nan
        violation_positions = [9, 12, 13, 15]
    elif presented_sequences[0] == '0101010101010101' or presented_sequences[0] == '1010101010101010':
        seqID = 2
        complexity = 6
        sequence_entropy = np.nan
        violation_positions = [9, 12, 14, 15]
    elif presented_sequences[0] == '0011001100110011' or presented_sequences[0] == '1100110011001100':
        seqID = 3
        complexity = 6
        sequence_entropy = 1.988
        violation_positions = [10, 11, 14, 15]
    elif presented_sequences[0] == '0000111100001111' or presented_sequences[0] == '1111000011110000':
        seqID = 4
        complexity = 6
        sequence_entropy = 1.617
        violation_positions = [9, 12, 13, 15]
    elif presented_sequences[0] == '0011010100110101' or presented_sequences[0] == '1100101011001010':
        seqID = 5
        complexity = 12
        sequence_entropy = 1.837
        violation_positions = [10, 11, 14, 15]
    elif presented_sequences[0] == '0000111100110101' or presented_sequences[0] == '1111000011001010':
        seqID = 6
        complexity = 14
        sequence_entropy = 1.988
        violation_positions = [10, 11, 14, 15]
    elif presented_sequences[0] == '0100011110110001' or presented_sequences[0] == '1011100001001110':
        seqID = 7
        complexity = 23
        sequence_entropy = 1.988
        violation_positions = [9, 12, 14, 15]
    else:
        print('This sequence was not recognized!!!! ')

    return seqID, complexity, sequence_entropy, violation_positions


def get_seqInfo(seqID):
    if seqID == 1:
        seqname = 'Repeat'
        seqtxtXY = 'xxxxxxxxxxxxxxxx'
        violation_positions = [9, 12, 13, 15]
    elif seqID == 2:
        seqname = 'Alternate'
        seqtxtXY = 'xYxYxYxYxYxYxYxY'
        violation_positions = [9, 12, 14, 15]
    elif seqID == 3:
        seqname = 'Pairs'
        seqtxtXY = 'xxYYxxYYxxYYxxYY'
        violation_positions = [10, 11, 14, 15]
    elif seqID == 4:
        seqname = 'Quadruplets'
        seqtxtXY = 'xxxxYYYYxxxxYYYY'
        violation_positions = [9, 12, 13, 15]
    elif seqID == 5:
        seqname = 'Pairs+Alt'
        seqtxtXY = 'xxYYxYxYxxYYxYxY'
        violation_positions = [10, 11, 14, 15]
    elif seqID == 6:
        seqname = 'Shrinking'
        seqtxtXY = 'xxxxYYYYxxYYxYxY'
        violation_positions = [10, 11, 14, 15]
    elif seqID == 7:
        seqname = 'Complex'
        seqtxtXY = 'xYxxxYYYYxYYxxxY'
        violation_positions = [9, 12, 14, 15]
    else:
        print('This sequence was not recognized!!!! ')

    return seqname, seqtxtXY, violation_positions


def update_metadata_rejected(subject, epochs_items):
    run_info_subject_dir = op.join(config.run_info_dir, subject)

    # Find removed epochs to exclude them for the metadata structure
    tokeep = [i for i, x in enumerate(epochs_items.drop_log) if not x]

    metadata = convert_csv_info_to_metadata(run_info_subject_dir)
    metadata_pandas = pd.DataFrame.from_dict(metadata, orient='index')
    metadata_pandas = pd.DataFrame.transpose(metadata_pandas)
    metadata_pandas = metadata_pandas.loc[tokeep, :]
    epochs_items.metadata = metadata_pandas
    return epochs_items


# ====================== change 31/3/2020 ========================

def update_metadata(subject, clean=False, new_field_name=None, new_field_values=None,recompute=False):
    """
    This function appends data to the metadata
    :param subject:
    :param autoreject:
    :param new_field_name:
    :param new_field_values:
    :return:
    """
    run_info_subject_dir = op.join(config.run_info_dir, subject)
    meg_subject_dir = op.join(config.meg_dir, subject)

    if clean:
        metadata_path = os.path.join(meg_subject_dir, 'metadata_item_clean.pkl')
        if op.exists(metadata_path) and not recompute:
            with open(metadata_path, 'rb') as fid:
                metadata = pickle.load(fid)
        else:
            epochs_items = load_epochs_items(subject, cleaned=True)
            epochs_items_cleaned = update_metadata_rejected(subject, epochs_items)
            metadata_path = os.path.join(meg_subject_dir, 'metadata_item_clean.pkl')
            metadata = epochs_items_cleaned.metadata
    else:
        metadata_path = os.path.join(meg_subject_dir, 'metadata_item.pkl')
        if op.exists(metadata_path) and not recompute:
            with open(metadata_path,'rb') as fid:
                metadata = pickle.load(fid)
        else:
            metadata = convert_csv_info_to_metadata(run_info_subject_dir)
            metadata = pd.DataFrame.from_dict(metadata, orient='index')
            metadata = pd.DataFrame.transpose(metadata)

    if new_field_name is not None:
        metadata[new_field_name] = new_field_values

    with open(metadata_path, "wb") as fid:
        pickle.dump(metadata, fid)

    return metadata


def convert_csv_info_to_metadata(csv_path):
    """
    This function reads the information from the csv files contained in csv_path.
    It generates a metadata dictionnary that will be inserted in the epochs object.
    It is particularly relevant when we epoch on each item
    :param csv_path:
    :return:

    'SequenceID': Goes from 1 to 7 and indices the considered sequence
    'Complexity': Complexity value attributed to the sequence
    'GlobalEntropy': Entropy from all the statistics of the sequence
    'RunNumber': From 1 to 14, corresponds to the number of the run during the MEEG acquisition
    'TrialNumber': From 1 to 46, corresponds to the index of the sequence presentation within a run
    'StimID': Was it the sound A or B that was presented (i.e. AFTER violation, if any)
    'Identity': Was it the sound A or B /!\ BEFORE violation, if any /!\
    'ViolationOrNot': 0 if no violation, 1 if violation
    'StimPosition': From 1 to 16, corresponds to the ordinal position of a sound within a sequence
    'ViolationInSequence': 0 if the sequence was not violated, the position of the violation for all the 16 sequence items if the sequence was violated
    'Violation_position_1234': 0 when no violation, relative position among the 4 possible violation positions that vary across SequenceIDs
    'Surprise': Local surprise computed from Maxime's model
    others...
    """

    nrun = []
    stimpos = []
    seqID = []
    complexity = []
    seq_entropy = []
    is_there_violation_in_seq = []

    ntrial = []  # numero de l essai dans le run
    stimID = []
    viol = []  # 0 if no violation, 1 if it is a violation
    surprise = []
    Violation_position_1234 = []

    surprise_not_violated, surprise_viol = load_surprise_mat(config)

    files_to_exclude = glob.glob(csv_path + os.path.sep + '*_missed.csv')

    # =============== load position dependent metadata for all runs, stored in mat files
    IdentityAllRuns = sio.loadmat(csv_path + os.path.sep + 'Identity.mat')
    RepeatAlterAllRuns = sio.loadmat(csv_path + os.path.sep + 'RepeatAlter.mat')
    ChunkNumberAllRuns = sio.loadmat(csv_path + os.path.sep + 'ChunkNumber.mat')
    WithinChunkPositionAllRuns = sio.loadmat(csv_path + os.path.sep + 'WithinChunkPosition.mat')
    WithinChunkPositionReverseAllRuns = sio.loadmat(csv_path + os.path.sep + 'WithinChunkPositionReverse.mat')
    ChunkDepthAllRuns = sio.loadmat(csv_path + os.path.sep + 'ChunkDepth.mat')
    OpenedChunksAllRuns = sio.loadmat(csv_path + os.path.sep + 'OpenedChunks.mat')
    ChunkSizeAllRuns = sio.loadmat(csv_path + os.path.sep + 'ChunkSize.mat')

    Identity = []
    RepeatAlter = []
    ChunkNumber = []
    WithinChunkPosition = []
    WithinChunkPositionReverse = []
    ChunkDepth = []
    OpenedChunks = []
    ChunkSize = []

    for i in range(1, 15):

        load_path = csv_path + os.path.sep + 'info_run%i' % i + '.csv'

        if load_path[:-4] + '_missed.csv' not in files_to_exclude:
            # we enter the loop only if we have the data for that run
            presented_seq, violation_position = read_info_csv(load_path)
            nrun += [i] * 736
            stimpos += [k for k in range(1, 17)] * 46

            thisrun_seqID, this_run_complexity, this_seq_entropy, violation_positions_for_this_sequence = get_seqID(
                presented_seq)

            # =============== select the correspond surprises ===========
            surprise_not_violated_this_sequence = surprise_not_violated[thisrun_seqID - 1, :]
            surprise_viol_this_sequence = surprise_viol[thisrun_seqID - 1, :, :]

            seqID += [thisrun_seqID] * 736
            complexity += [this_run_complexity] * 736
            seq_entropy += [this_seq_entropy] * 736

            Identity += np.transpose(IdentityAllRuns['Identity'][i - 1, :]).tolist()
            RepeatAlter += np.transpose(RepeatAlterAllRuns['RepeatAlter'][i - 1, :]).tolist()
            ChunkNumber += np.transpose(ChunkNumberAllRuns['ChunkNumber'][i - 1, :]).tolist()
            WithinChunkPosition += np.transpose(WithinChunkPositionAllRuns['WithinChunkPosition'][i - 1, :]).tolist()
            WithinChunkPositionReverse += np.transpose(WithinChunkPositionReverseAllRuns['WithinChunkPositionReverse'][i - 1, :]).tolist()
            ChunkDepth += np.transpose(ChunkDepthAllRuns['ChunkDepth'][i - 1, :]).tolist()
            OpenedChunks += np.transpose(OpenedChunksAllRuns['OpenedChunks'][i - 1, :]).tolist()
            ChunkSize += np.transpose(ChunkSizeAllRuns['ChunkSize'][i - 1, :]).tolist()

            for k in range(46):
                ntrial += [k + 1] * 16
                seq_pres = presented_seq[k]
                stimID += [int(seq_element) for seq_element in seq_pres]
                viol_to_append = [0] * 16
                viol_1234 = [0] * 16

                viol_in_trial = violation_position[k]
                if viol_in_trial != 0:
                    viol_to_append[viol_in_trial - 1] = 1
                    viol_1234[viol_in_trial - 1] = violation_positions_for_this_sequence.index(viol_in_trial) + 1
                    is_there_violation_in_seq += [viol_in_trial] * 16
                    surprise += list(surprise_viol_this_sequence[viol_in_trial - 1, :])

                else:
                    is_there_violation_in_seq += [0] * 16
                    surprise += list(surprise_not_violated_this_sequence)

                viol += viol_to_append
                Violation_position_1234 += viol_1234

    ChunkBeginning = (np.asarray(WithinChunkPosition) == 1) * 1
    ChunkEnd = (np.asarray(WithinChunkPositionReverse) == 1) * 1

    metadata = {'SequenceID': np.asarray(seqID),
                'Complexity': np.asarray(complexity),
                'GlobalEntropy': np.asarray(seq_entropy),
                'RunNumber': np.asarray(nrun),
                'TrialNumber': np.asarray(ntrial),
                'StimID': np.asarray(stimID),
                'ViolationOrNot': np.asarray(viol),
                'StimPosition': np.asarray(stimpos),
                'ViolationInSequence': np.asarray(is_there_violation_in_seq),
                'Violation_position_1234': np.asarray(Violation_position_1234),
                'Surprise': np.asarray(surprise),
                'Identity': np.asarray(Identity),
                'RepeatAlter': np.asarray(RepeatAlter),
                'ChunkNumber': np.asarray(ChunkNumber),
                'WithinChunkPosition': np.asarray(WithinChunkPosition),
                'WithinChunkPositionReverse': np.asarray(WithinChunkPositionReverse),
                'ChunkDepth': np.asarray(ChunkDepth),
                'OpenedChunks': np.asarray(OpenedChunks),
                'ChunkSize': np.asarray(ChunkSize),
                'ChunkBeginning': ChunkBeginning,
                'ChunkEnd': ChunkEnd
                }

    return metadata


def load_epochs_items(subject, cleaned=True, AR_type='local'):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    if cleaned:
        if AR_type == 'local':
            extension = subject + '_clean_epo'
        elif AR_type == 'global':
            extension = subject + '_ARglob_epo'
    else:
        extension = subject + '_epo'
        warnings.warn('\nLoading pre-autoreject epochs for subject ' + subject)
    fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Input: ", fname_in)
    epochs = mne.read_epochs(fname_in, preload=True)
    return epochs


def load_resid_epochs_items(subject, resid_epochs_type='reg_repeataltern_surpriseOmegainfinity'):
    resid_path = op.join(config.result_path, 'linear_models', resid_epochs_type, subject)
    fname_in = op.join(resid_path, 'residuals-epo.fif')
    print("Input: ", fname_in)
    epochs = mne.read_epochs(fname_in, preload=True)

    return epochs


def load_epochs_full_sequence(subject, cleaned=True, AR_type='local'):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    if cleaned:
        if AR_type == 'local':
            extension = subject + '_1st_element_clean_epo'
        elif AR_type == 'global':
            extension = subject + '_1st_element_ARglob_epo'
    else:
        extension = subject + '_1st_element_epo'
        warnings.warn('\nLoading pre-autoreject epochs for subject ' + subject)
    fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Input: ", fname_in)
    epochs = mne.read_epochs(fname_in, preload=True)
    return epochs


def balance_epochs_violation_positions(epochs,balance_violation_standards=True):
    """
    This function balances violations and standards by position for each sequence
    :param epochs:
    :return:
    """

    epochs_balanced_allseq = []

    events_IDS = []
    for seqID in range(1, 8):


        epochs_seq = epochs['SequenceID == "' + str(seqID) + '" and TrialNumber>10'].copy()
        tmp = epochs_seq['ViolationOrNot == "1"']  # Deviant trials
        devpos = np.unique(tmp.metadata.StimPosition)  # Position of deviants
        epochs_seq = epochs_seq['StimPosition == "' + str(devpos[0]) +
                                '" or StimPosition == "' + str(devpos[1]) +
                                '" or StimPosition == "' + str(devpos[2]) +
                                '" or StimPosition == "' + str(devpos[3]) + '"']

        epochs_seq_noviol = epochs_seq["ViolationInSequence == 0"]
        epochs_seq_viol = epochs_seq["ViolationInSequence > 0 and ViolationOrNot ==1"]

        epochs_balanced_allseq.append([epochs_seq_noviol, epochs_seq_viol])
        print('We appended the balanced epochs for SeqID%i' % seqID)

        events_IDS.append([[seqID*1000+dev*100,seqID*1000+dev*100+10]  for dev in devpos])

    epochs_balanced = mne.concatenate_epochs(list(np.hstack(epochs_balanced_allseq)))

    events_IDS = np.concatenate(events_IDS)

    all_combinations = [list(zip(each_permutation, list2)) for each_permutation in itertools.permutations(list1, len(list2))]
    if balance_violation_standards:
        print("we are balancing the number of standards and of violations")
        metadata_epochs = epochs_balanced.metadata
        events = [int(metadata_epochs['SequenceID'].values[i] * 10000 + metadata_epochs['StimPosition'].values[i] * 100 +
                     metadata_epochs['ViolationOrNot'].values[i]*10 + metadata_epochs['StimID'].values[i]) for i in range(len(epochs_balanced))]
        epochs_balanced.events[:, 2] = events
        epochs_balanced.event_id = {'%i'%i:i for i in np.unique(events)}
        epochs_balanced.equalize_event_counts(epochs_balanced.event_id)    # ===== to train the filter do not consider the habituation trials to later test on them separately ================

    return epochs_balanced

def metadata_balance_epochs_violation_positions(metadata):
    """
    This function balances violations and standards by position for each sequence by adding a yes/no column in metadata, for standard with matched positions
    /!\ careful with indexes of pd dataframe (after removal of bad epochs)...
    """
    metadata['balanced_standard'] = 'no'
    all_idx = []
    for seqID in range(1, 8):
        tmp = metadata[(metadata['ViolationOrNot'] == 1) & (metadata['SequenceID'] == seqID)]  # Deviant trials for this sequence
        devpos = np.unique(tmp['StimPosition'])  # Position of deviants
        idx = (metadata['SequenceID'] == seqID) & (metadata['TrialNumber'] > 10) & (metadata['ViolationInSequence'] == 0) & (metadata['StimPosition'] == devpos[0])
        all_idx.extend(np.where(idx)[0].tolist())
        idx = (metadata['SequenceID'] == seqID) & (metadata['TrialNumber'] > 10) & (metadata['ViolationInSequence'] == 0) & (metadata['StimPosition'] == devpos[1])
        all_idx.extend(np.where(idx)[0].tolist())
        idx = (metadata['SequenceID'] == seqID) & (metadata['TrialNumber'] > 10) & (metadata['ViolationInSequence'] == 0) & (metadata['StimPosition'] == devpos[2])
        all_idx.extend(np.where(idx)[0].tolist())
        idx = (metadata['SequenceID'] == seqID) & (metadata['TrialNumber'] > 10) & (metadata['ViolationInSequence'] == 0) & (metadata['StimPosition'] == devpos[3])
        all_idx.extend(np.where(idx)[0].tolist())
    for ii, idx in enumerate(all_idx):  # tried to use a loop and one value at a time to avoid Pandas SettingWithCopyWarning, but no luck...
        metadata['balanced_standard'].iloc[idx] = 'yes'
        # metadata.iloc[idx, 'balanced_standard'] = 'yes'
    return metadata

# ---------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------------- #


def run_epochs(subject, epoch_on_first_element, baseline=True):
    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    run_info_subject_dir = op.join(config.run_info_dir, subject)
    raw_list = list()
    events_list = list()

    print("  Loading raw data")
    runs = config.runs_dict[subject]
    for run in runs:
        extension = run + '_ica_raw'
        raw_fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
        raw = mne.io.read_raw_fif(raw_fname_in, preload=True)

        # ---------------------------------------------------------------------------------------------------------------- #
        # RESAMPLING EACH RUN BEFORE CONCAT & EPOCHING
        # Resampling the raw data while keeping events from original raw data, to avoid potential loss of
        # events when downsampling: https://www.nmr.mgh.harvard.edu/mne/dev/auto_examples/preprocessing/plot_resample.html
        # Find events
        events = mne.find_events(raw, stim_channel=config.stim_channel,
                                 consecutive=True,
                                 min_duration=config.min_event_duration,
                                 shortest_event=config.shortest_event)

        print('  Downsampling raw data')
        raw, events = raw.resample(config.resample_sfreq, npad='auto', events=events)
        if len(events) != 46 * 16:
            raise Exception('We expected %i events but we got %i' % (46 * 16, len(events)))

        raw_list.append(raw)
        # ---------------------------------------------------------------------------------------------------------------- #

    if subject == 'sub08-cc_150418':
        # For this participant, we had some problems when concatenating the raws for run08. The error message said that raw08._cals didn't match the other ones.
        # We saw that it is the 'calibration' for the channel EOG061 that was different with respect to run09._cals.
        raw_list[7]._cals = raw_list[8]._cals
        print('Warning: corrected an issue with subject08 run08 ica_raw data file...')

    print('Concatenating runs')
    raw = mne.concatenate_raws(raw_list)
    if "eeg" in config.ch_types:
        raw.set_eeg_reference(projection=True)
    del raw_list

    # Save resampled, concatenated runs (in case we need it)
    print('Saving concatenated runs')
    fname = op.join(meg_subject_dir, subject + '_allruns_final_raw.fif')
    raw.save(fname, overwrite=True)

    meg = False
    if 'meg' in config.ch_types:
        meg = True
    elif 'grad' in config.ch_types:
        meg = 'grad'
    elif 'mag' in config.ch_types:
        meg = 'mag'
    eeg = 'eeg' in config.ch_types
    picks = mne.pick_types(raw.info, meg=meg, eeg=eeg, stim=True, eog=True, exclude=())

    # Construct metadata from csv events file
    metadata = convert_csv_info_to_metadata(run_info_subject_dir)
    metadata_pandas = pd.DataFrame.from_dict(metadata, orient='index')
    metadata_pandas = pd.DataFrame.transpose(metadata_pandas)

    # ====== Epoching the data
    print('  Epoching')

    # Events
    events = mne.find_events(raw, stim_channel=config.stim_channel, consecutive=True,
                             min_duration=config.min_event_duration, shortest_event=config.shortest_event)

    if epoch_on_first_element:
        # fosca 06012020
        config.tmin = -0.200
        config.tmax = 0.25 * 17
        config.baseline = (config.tmin, 0)
        if baseline is None:
            config.baseline = None
        for k in range(len(events)):
            events[k, 2] = k % 16 + 1
        epochs = mne.Epochs(raw, events, {'sequence_starts': 1}, config.tmin, config.tmax,
                            proj=True, picks=picks, baseline=config.baseline,
                            preload=False, decim=config.decim,
                            reject=None)
        epochs.metadata = metadata_pandas[metadata_pandas['StimPosition'] == 1.0]
    else:
        config.tmin = -0.050
        config.tmax = 0.600
        config.baseline = (config.tmin, 0)
        if baseline is None:
            config.baseline = None
        epochs = mne.Epochs(raw, events, None, config.tmin, config.tmax,
                            proj=True, picks=picks, baseline=config.baseline,
                            preload=False, decim=config.decim,
                            reject=None)

        # Add metadata to epochs
        epochs.metadata = metadata_pandas

    # Save epochs (before AutoReject)
    print('  Writing epochs to disk')
    if epoch_on_first_element:
        extension = subject + '_1st_element_epo'
    else:
        extension = subject + '_epo'
    epochs_fname = op.join(meg_subject_dir, config.base_fname.format(**locals()))

    print("Output: ", epochs_fname)
    epochs.save(epochs_fname, overwrite=True)
    # epochs.save(epochs_fname)

    if config.autoreject:
        epochs.load_data()

        # Running AutoReject "global" (https://autoreject.github.io) -> just get the thresholds
        from autoreject import get_rejection_threshold
        reject = get_rejection_threshold(epochs, ch_types=['mag', 'grad', 'eeg'])
        epochsARglob = epochs.copy().drop_bad(reject=reject)
        print('  Writing "AR global" cleaned epochs to disk')
        if epoch_on_first_element:
            extension = subject + '_1st_element_ARglob_epo'
        else:
            extension = subject + '_ARglob_epo'
        epochs_fname = op.join(meg_subject_dir, config.base_fname.format(**locals()))
        print("Output: ", epochs_fname)
        epochsARglob.save(epochs_fname, overwrite=True)
        # Save autoreject thresholds
        pickle.dump(reject, open(epochs_fname[:-4] + '_ARglob_thresholds.obj', 'wb'))

        # Running AutoReject "local" (https://autoreject.github.io)
        ar = AutoReject()
        epochsAR, reject_log = ar.fit_transform(epochs, return_log=True)
        print('  Writing "AR local" cleaned epochs to disk')
        if epoch_on_first_element:
            extension = subject + '_1st_element_clean_epo'
        else:
            extension = subject + '_clean_epo'
        epochs_fname = op.join(meg_subject_dir, config.base_fname.format(**locals()))
        print("Output: ", epochs_fname)
        epochsAR.save(epochs_fname, overwrite=True)
        # Save autoreject reject_log
        pickle.dump(reject_log, open(epochs_fname[:-4] + '_reject_local_log.obj', 'wb'))
        # To read, would be: reject_log = pickle.load(open(epochs_fname[:-4]+'_reject_log.obj', 'rb'))




# ______________________________________________________________________________________________________________________
def sliding_window(epoch,sliding_window_size=25, sliding_window_step=1,
                                             sliding_window_min_size=None):

    """
    This function outputs an epoch object that has been built from a sliding window on the data
    :param epoch:
    :param delta_t: sliding window in number of data points
    :return:
    """
    epoch2 = epoch.copy()

    from ABseq_func import SVM_funcs

    xformer = SVM_funcs.SlidingWindow(window_size=sliding_window_size, step=sliding_window_step,
                                         min_window_size=sliding_window_min_size)

    n_time_points = epoch._data.shape[2]
    window_start = np.array(range(0, n_time_points - sliding_window_size + 1, sliding_window_step))
    window_end = window_start + sliding_window_size

    window_end[-1] = min(window_end[-1], n_time_points)  # make sure that the last window doesn't exceed the input size

    intermediate_times = [int((window_start[i] + window_end[i]) / 2) for i in range(len(window_start))]
    times = epoch.times[intermediate_times]

    epoch2 = mne.EpochsArray(xformer.fit_transform(epoch._data),epoch.info)
    epoch2._set_times(times)
    epoch2.metadata = epoch.metadata

    return epoch2