import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import io, EvokedArray
from mne.datasets import sample
from mne.decoding import EMS, compute_ems
from sklearn.model_selection import StratifiedKFold

from scipy.ndimage.filters import gaussian_filter1d

import config
from mne.parallel import parallel_func

# make less parallel runs to limit memory usage
N_JOBS = max(config.N_JOBS // 4, 1)

Fosca_linux = False
if Fosca_linux:
    config.meg_dir = '/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/'

# subject = 'pa_190002'

def EMS_filter(subject):

    print("Processing subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)
    fig_path = op.join(config.study_path, 'Figures', 'EMS') + op.sep
    sensor_types = ['eeg','grad','mag']
    # Load epoch data
    extension = subject + '_epo'
    fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Input: ", fname_in)
    epochs = mne.read_epochs(fname_in, preload=True)

    score_sensor_type = {sensor:  [] for sensor in sensor_types}

    for sensor_type in sensor_types:
        if sensor_type=='eeg':
            epochs_sensor = epochs.copy().pick_types(eeg=True, meg = False)
        else:
            epochs_sensor = epochs.copy().pick_types(meg=sensor_type,eeg = False)

        score_all = []

        for seqID in range(1, 8):

            epochs_OI = epochs_sensor['SequenceID == "' + str(seqID) + '"'].copy()
            epochs_OI.events[:, 2] = 1*(epochs_OI.metadata['ViolationOrNot'].values>0)
            event_ids = {'non_viol': 0, 'viol': 1}
            epochs_OI.event_id = event_ids

            tmp = epochs_OI['ViolationOrNot == "1"']  # Deviant trials
            devpos = np.unique(tmp.metadata.StimPosition)  # Position of deviants
            # Keep only positions where there can be deviants
            epochs_OI = epochs_OI['StimPosition == "' + str(devpos[0]) +
                                  '" or StimPosition == "' + str(devpos[1]) +
                                  '" or StimPosition == "' + str(devpos[2]) +
                                  '" or StimPosition == "' + str(devpos[3]) + '"']

            epochs_OI_noviol = epochs_OI["ViolationInSequence == 0"]
            epochs_OI_viol = epochs_OI["ViolationInSequence > 0 and ViolationOrNot ==1"]

            epochs_OI = mne.concatenate_epochs([epochs_OI_noviol,epochs_OI_viol])
            # epochs_OI.equalize_event_counts(event_ids)

            # From https://martinos.org/mne/stable/auto_examples/decoding/plot_ems_filtering.html
            X = epochs_OI.get_data()
            y = np.asarray([int(ll) for ll in epochs_OI.metadata['ViolationOrNot'].values])

            n_epochs, n_channels, n_times = X.shape

            # Initialize EMS transformer
            ems = EMS()

            # Initialize the variables of interest
            X_transform = np.zeros((n_epochs, n_times))  # Data after EMS transformation
            filters = list()  # Spatial filters at each time point

            # In the original paper, the cross-validation is a leave-one-out. However,
            # we recommend using a Stratified KFold, because leave-one-out tends
            # to overfit and cannot be used to estimate the variance of the
            # prediction within a given fold.

            for train, test in StratifiedKFold(n_splits=5).split(X, y):
                # In the original paper, the z-scoring is applied outside the CV.
                # However, we recommend to apply this preprocessing inside the CV.
                # Note that such scaling should be done separately for each channels if the
                # data contains multiple channel types.

                X_scaled = X / np.std(X[train])

                # Fit and store the spatial filters
                ems.fit(X_scaled[train], y[train])

                # Store filters for future plotting
                filters.append(ems.filters_)

                # Generate the transformed data
                X_transform[test] = ems.transform(X_scaled[test])

            # Average the spatial filters across folds
            filters = np.mean(filters, axis=0)

            # Plot individual trials
            plt.figure()
            fig_name = 'individual_trials.png'
            plt.title('single trial surrogates')
            plt.imshow(X_transform[y.argsort()], origin='lower', aspect='auto',
                       extent=[epochs_OI.times[0], epochs_OI.times[-1], 1, len(X_transform)],
                       cmap='RdBu_r')
            plt.xlabel('Time (ms)')
            plt.ylabel('Trials (reordered by condition)')
            plt.savefig(fig_path + str(seqID)+ sensor_type + fig_name)
            plt.close('all')

            # Plot average response
            plt.figure()
            plt.title('Score EMS')
            mappings = [(key, value) for key, value in event_ids.items()]
            ems_all =[]
            for key, value in mappings:
                ems_ave = X_transform[y == value]
                # plt.plot(epochs_OI.times, ems_ave.mean(0), label=key)
                ems_all.append(ems_ave)
            fig_name = 'Score_EMS.png'
            score = ems_all[0].mean(0)-ems_all[1].mean(0)
            score_all.append(score)
            plt.plot(epochs_OI.times,score)
            plt.xlabel('Time (ms)')
            plt.ylabel('a.u.')
            plt.legend(loc='best')
            # plt.show()
            plt.savefig(fig_path + str(seqID)+ sensor_type + fig_name)
            plt.close('all')

            # Visualize spatial filters across time
            fig_name = 'topo_EMS.png'
            evoked = EvokedArray(filters, epochs_OI.info, tmin=epochs_OI.tmin)
            times = np.arange(-0.100, 0.700, 0.100)
            evoked.plot_topomap(times, time_unit='s', scalings=1, average=0.030)
            plt.savefig(fig_path + str(seqID) + sensor_type + fig_name)
            plt.close('all')

        score_sensor_type[sensor_type] = score_all


    sensor = 'eeg'
    plt.figure()
    for ll in range(7):
        # Original
        # plt.plot(epochs_OI.times, score_sensor_type[sensor][ll], label='SeqID' + str(ll+1))
        # Or gaussian filtered
        ysmoothed = gaussian_filter1d(score_sensor_type[sensor][ll], sigma=2)
        plt.plot(epochs_OI.times, ysmoothed, label='SeqID' + str(ll+1))
    plt.xlabel('Time (ms)')
    plt.ylabel('a.u.')
    plt.legend(loc='best')
    fig_name = 'Score_EMS_all_EEG.png'
    plt.savefig(fig_path + fig_name)
    plt.close('all')

    sensor = 'mag'
    plt.figure()
    for ll in range(7):
        # Original
        # plt.plot(epochs_OI.times, score_sensor_type[sensor][ll], label='SeqID' + str(ll+1))
        # Or gaussian filtered
        ysmoothed = gaussian_filter1d(score_sensor_type[sensor][ll], sigma=2)
        plt.plot(epochs_OI.times, ysmoothed, label='SeqID' + str(ll+1))
    plt.xlabel('Time (ms)')
    plt.ylabel('a.u.')
    plt.legend(loc='best')
    fig_name = 'Score_EMS_all_MAG.png'
    plt.savefig(fig_path + fig_name)
    plt.close('all')

    sensor = 'grad'
    plt.figure()
    for ll in range(7):
        # Original
        # plt.plot(epochs_OI.times, score_sensor_type[sensor][ll], label='SeqID' + str(ll+1))
        # Or gaussian filtered
        ysmoothed = gaussian_filter1d(score_sensor_type[sensor][ll], sigma=2)
        plt.plot(epochs_OI.times, ysmoothed, label='SeqID' + str(ll+1))
    plt.xlabel('Time (ms)')
    plt.ylabel('a.u.')
    plt.legend(loc='best')
    fig_name = 'Score_EMS_all_GRAD.png'
    plt.savefig(fig_path + fig_name)
    plt.close('all')

# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(EMS_filter, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)