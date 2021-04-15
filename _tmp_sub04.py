import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths


import os.path as op
import mne
import numpy as np
from warnings import warn
import config
import pandas as pd
from ABseq_func import epoching_funcs
import pickle
from autoreject import AutoReject


subject = "sub04-rf_190499"
config.ch_types = ['eeg']
config.autoreject = True
run_epochs(subject, epoch_on_first_element=False, baseline=True,l_freq= 1,suffix = '_eeg_lfreq_1Hz')
run_epochs(subject, epoch_on_first_element=False, baseline=True,l_freq= 0.1,suffix = '_eeg_lfreq_01Hz')


def run_epochs(subject, epoch_on_first_element, baseline=True, l_freq=None, h_freq= None, suffix  = '_eeg_1Hz' ):

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
        raw.filter(l_freq=1,h_freq=None)
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
    metadata = epoching_funcs.convert_csv_info_to_metadata(run_info_subject_dir)
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
        extension = subject + '_1st_element_epo' + suffix
    else:
        extension = subject + '_epo' + suffix
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
            extension = subject + '_1st_element_ARglob_epo' +suffix
        else:
            extension = subject + '_ARglob_epo'+suffix
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
            extension = subject + '_1st_element_clean_epo' +suffix
        else:
            extension = subject + '_clean_epo'+suffix
        epochs_fname = op.join(meg_subject_dir, config.base_fname.format(**locals()))
        print("Output: ", epochs_fname)
        epochsAR.save(epochs_fname, overwrite=True)
        # Save autoreject reject_log
        pickle.dump(reject_log, open(epochs_fname[:-4] + '_reject_local_log.obj', 'wb'))
        # To read, would be: reject_log = pickle.load(open(epochs_fname[:-4]+'_reject_log.obj', 'rb'))

