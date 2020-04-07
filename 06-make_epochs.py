"""
====================
06. Construct epochs
====================

The epochs are constructed by using the events created in script 03. MNE
supports hierarchical events that allows selection to different groups more
easily (see config.event_id). Automatic rejection is applied to the epochs (or not).
Finally the epochs are saved to disk.
To save space, the epoch data can be decimated.
"""

import os.path as op
import mne
import pandas as pd
import config
import numpy as np
import pickle

from mne.parallel import parallel_func
from ABseq_func import epoching_funcs
from autoreject import AutoReject

# make less parallel runs to limit memory usage
# N_JOBS = max(config.N_JOBS // 4, 1)
N_JOBS = config.N_JOBS

config.subjects_list = ['sub03-mr_190273']

###############################################################################
# Now we define a function to extract epochs for one subject
def run_epochs(subject,epoch_on_first_element):

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
        # RESAMPLING EACH RUN BEFORE CONCAT
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

    meg = False
    if 'meg' in config.ch_types:
        meg = True
    elif 'grad' in config.ch_types:
        meg = 'grad'
    elif 'mag' in config.ch_types:
        meg = 'mag'
    eeg = 'eeg' in config.ch_types
    picks = mne.pick_types(raw.info, meg=meg, eeg=eeg, stim=True, eog=True, exclude=())

    # # ---------------------------------------------------------------------------------------------------------------- #
    # # DO THIS BEFORE CONCAT??
    # # Resampling the raw data while keeping events from original raw data, to avoid potential loss of
    # # events when downsampling: https://www.nmr.mgh.harvard.edu/mne/dev/auto_examples/preprocessing/plot_resample.html
    # # Find events
    events = mne.find_events(raw, stim_channel=config.stim_channel,
                             consecutive=True,
                             min_duration=config.min_event_duration,
                             shortest_event=config.shortest_event)
    #
    # print('  Downsampling raw data')
    # raw, events = raw.resample(config.resample_sfreq, npad='auto', events=events)
    # if len(events) != len(runs)*46*16:
    #     raise Exception('We expected %i events but we got %i'%(len(runs)*46*16,len(events)))
    # # ---------------------------------------------------------------------------------------------------------------- #

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
        config.tmin = -0.500
        config.tmax = 0.25*17
        config.baseline = (config.tmin, 0)
        for k in range(len(events)):
            events[k, 2] = k % 16+1
        epochs = mne.Epochs(raw, events, {'sequence_starts': 1}, config.tmin, config.tmax,
                            proj=True, picks=picks, baseline=config.baseline,
                            preload=False, decim=config.decim,
                            reject=None)
        epochs.metadata = metadata_pandas[metadata_pandas['StimPosition'] == 1.0]
    else:
        config.tmin = -0.100
        config.tmax = 0.750
        config.baseline = (config.tmin, 0)
        epochs = mne.Epochs(raw, events, None, config.tmin, config.tmax,
                            proj=True, picks=picks, baseline=config.baseline,
                            preload=False, decim=config.decim,
                            reject=None)
        # Add metadata to epochs
        epochs.metadata = metadata_pandas

    # Save epochs (before AutoReject)
    print('  Writing epochs to disk')
    if epoch_on_first_element:
        extension = subject+'_1st_element_epo'
    else:
        extension = subject+'_epo'
    epochs_fname = op.join(meg_subject_dir, config.base_fname.format(**locals()))

    print("Output: ", epochs_fname)
    # epochs.save(epochs_fname, overwrite=True)
    epochs.save(epochs_fname)

    if config.autoreject:
        # Running AutoReject (https://autoreject.github.io)
        epochs.load_data()
        ar = AutoReject()
        epochs = ar.fit_transform(epochs)
        reject_log = ar.get_reject_log(epochs)

        # Save epochs (after AutoReject)
        print('  Writing cleaned epochs to disk')

        if epoch_on_first_element:
            extension = subject + '_1st_element_clean_epo'
        else:
            extension = subject + '_clean_epo'
        epochs_fname = op.join(meg_subject_dir, config.base_fname.format(**locals()))
        print("Output: ", epochs_fname)
        # epochs.save(epochs_fname, overwrite=True)
        epochs.save(epochs_fname)

        # Save autoreject reject_log
        pickle.dump(reject_log, open(epochs_fname[:-4]+'_reject_log.obj', 'wb'))
        # To read, would be: reject_log = pickle.load(open(epochs_fname[:-4]+'_reject_log.obj', 'rb'))

# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(epoching_funcs.run_epochs, n_jobs=N_JOBS)

epoch_on_first_element = True
parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)

epoch_on_first_element = False
parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)
