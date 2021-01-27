from ABseq_func import *
import mne
import config
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import time
import pandas as pd
import warnings

# /!\ DO NOT SHOW SettingWithCopyWarning /!\
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Exclude some subjects
# config.exclude_subjects.append('sub10-gp_190568')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

# =========================================================== #
# Options
# =========================================================== #
cleaned = True  # epochs cleaned with autoreject or not, only when using original epochs (resid_epochs=False)
resid_epochs = False  # use epochs created by regressing out surprise effects, instead of original epochs
detrend_epochs = False
baseline_epochs = False  # apply baseline to the epochs
lowpass_epochs = False  # option to filter epochs with  30Hz lowpass filter
# NOTE: set to False because lowpass/detrend/baseline is now performed at the evoked level
preliminary = False
n_jobs = 6

# =========================================================== #
# Output folder
# =========================================================== #
results_path = op.join(config.result_path, 'peak_data')
utils.create_folder(results_path)

# =========================================================== #
# Time windows and ROI
# =========================================================== #
# Time windows of interest
n1m_timewin = [.070 - 0.030, .070 + 0.030]
p2m_timewin = [.140 - 0.030, .140 + 0.030]
n1_timewin = [.080 - 0.030, .080 + 0.030]
p2_timewin = [.130 - 0.030, .130 + 0.030]

# NEW VERSION
n1m_timewin = [.030, .110]
p2m_timewin = [.110, .220]
n1_timewin = n1m_timewin
p2_timewin = p2m_timewin
print(n1m_timewin)
print(p2m_timewin)

# ROIs
mag_roi = ['MEG0131', 'MEG0141', 'MEG0211', 'MEG0221', 'MEG0231', 'MEG0241', 'MEG1811', 'MEG0411', 'MEG0441', 'MEG2221', 'MEG2411', 'MEG2421', 'MEG2431', 'MEG2441', 'MEG2521', 'MEG2611', 'MEG2621', 'MEG2631', 'MEG2641',
           'MEG1121', 'MEG1131', 'MEG1311', 'MEG1321', 'MEG1331', 'MEG1341', 'MEG1431', 'MEG1441', 'MEG1511', 'MEG1521', 'MEG1531', 'MEG1541', 'MEG1611', 'MEG1621', 'MEG1631', 'MEG1641', 'MEG1721']
grad_roi = ['MEG0132', 'MEG0133', 'MEG0143', 'MEG0142', 'MEG0213', 'MEG0212', 'MEG0222', 'MEG0223', 'MEG0232', 'MEG0233', 'MEG0243', 'MEG0242', 'MEG0413', 'MEG0412', 'MEG1813', 'MEG0443', 'MEG1812', 'MEG0442', 'MEG2223',
            'MEG2222', 'MEG2412', 'MEG2413', 'MEG2423', 'MEG2422', 'MEG2433', 'MEG2432', 'MEG2442', 'MEG2443', 'MEG2522', 'MEG2523', 'MEG2612', 'MEG2613', 'MEG2623', 'MEG2622', 'MEG2633', 'MEG2632', 'MEG2642', 'MEG2643',
            'MEG1123', 'MEG1122', 'MEG1133', 'MEG1132', 'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322', 'MEG1333', 'MEG1332', 'MEG1342', 'MEG1343', 'MEG1433', 'MEG1432', 'MEG1442', 'MEG1443', 'MEG1512', 'MEG1513', 'MEG1522',
            'MEG1523', 'MEG1533', 'MEG1532', 'MEG1543', 'MEG1542', 'MEG1613', 'MEG1612', 'MEG1622', 'MEG1623', 'MEG1632', 'MEG1633', 'MEG1643',
            'MEG1642', 'MEG1722', 'MEG1723']
eeg_roi = ['EEG005', 'EEG006', 'EEG009', 'EEG010', 'EEG011', 'EEG012', 'EEG013', 'EEG014', 'EEG015', 'EEG019', 'EEG020', 'EEG021', 'EEG022', 'EEG028', 'EEG029', 'EEG030', 'EEG031', 'EEG032']

if preliminary:
    # =========================================================== #
    # Explore activation to standard items to help define time windows and ROIs
    # =========================================================== #
    plt.close('all')
    # Group evoked standard items
    item_ev, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_all', filter_not=None)
    item_ev = [item_ev['items_standard_all-'][ii][0] for ii in range(len(item_ev['items_standard_all-']))]  # correct the list of lists issue
    item_ev_mean = mne.grand_average(item_ev)
    # Group evoked violation items
    item_ev_viol, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_all', filter_not=None)
    item_ev_viol = [item_ev_viol['items_viol_all-'][ii][0] for ii in range(len(item_ev_viol['items_viol_all-']))]  # correct the list of lists issue
    item_ev_viol_mean = mne.grand_average(item_ev_viol)
    # Find timepoints of interest
    item_ev_mean.crop(tmin=0, tmax=0.300).plot_joint(topomap_args=dict(show_names=False))

    fig, ax = plt.subplots(figsize=(9,4))
    item_ev_mean.crop(tmin=0, tmax=0.300).plot(picks=mag_roi, spatial_colors=True, time_unit='ms', axes=ax)
    ax.set(xticks=np.arange(0, 301, 10))
    ax.grid(True)
    fig.savefig('tmpfig/grad_chans_standitems.png', bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(figsize=(9,4))
    item_ev_viol_mean.crop(tmin=0, tmax=0.300).plot(picks=mag_roi, spatial_colors=True, time_unit='ms', axes=ax)
    ax.set(xticks=np.arange(0, 301, 10))
    ax.grid(True)
    fig.savefig('tmpfig/grad_chans_violitems.png', bbox_inches='tight', dpi=300)


    # Select sensors of interest
    tmp = item_ev_mean.plot_sensors(kind='select', ch_type='mag')  # use ctrl + mouse to draw roi, then look selection in "tmp"
    tmp = item_ev_mean.plot_sensors(kind='select', ch_type='grad')  # use ctrl + mouse to draw roi, then look selection in "tmp"
    tmp = item_ev_mean.plot_sensors(kind='select', ch_type='eeg')  # use ctrl + mouse to draw roi, then look selection in "tmp"
    timepoint = .136  # meg .072 .136 // eeg .076 .136
    item_ev_mean.plot_topomap(ch_type='eeg', times=timepoint, show_names=False)
    mag_roi = ['MEG0131', 'MEG0141', 'MEG0211', 'MEG0221', 'MEG0231', 'MEG0241', 'MEG1811', 'MEG0411', 'MEG0441', 'MEG2221', 'MEG2411', 'MEG2421', 'MEG2431', 'MEG2441', 'MEG2521', 'MEG2611', 'MEG2621', 'MEG2631',
               'MEG2641',
               'MEG1121', 'MEG1131', 'MEG1311', 'MEG1321', 'MEG1331', 'MEG1341', 'MEG1431', 'MEG1441', 'MEG1511', 'MEG1521', 'MEG1531', 'MEG1541', 'MEG1611', 'MEG1621', 'MEG1631', 'MEG1641', 'MEG1721']
    grad_roi = ['MEG0132', 'MEG0133', 'MEG0143', 'MEG0142', 'MEG0213', 'MEG0212', 'MEG0222', 'MEG0223', 'MEG0232', 'MEG0233', 'MEG0243', 'MEG0242', 'MEG0413', 'MEG0412', 'MEG1813', 'MEG0443', 'MEG1812', 'MEG0442',
                'MEG2223',
                'MEG2222', 'MEG2412', 'MEG2413', 'MEG2423', 'MEG2422', 'MEG2433', 'MEG2432', 'MEG2442', 'MEG2443', 'MEG2522', 'MEG2523', 'MEG2612', 'MEG2613', 'MEG2623', 'MEG2622', 'MEG2633', 'MEG2632', 'MEG2642',
                'MEG2643',
                'MEG1123', 'MEG1122', 'MEG1133', 'MEG1132', 'MEG1312', 'MEG1313', 'MEG1323', 'MEG1322', 'MEG1333', 'MEG1332', 'MEG1342', 'MEG1343', 'MEG1433', 'MEG1432', 'MEG1442', 'MEG1443', 'MEG1512', 'MEG1513',
                'MEG1522',
                'MEG1523', 'MEG1533', 'MEG1532', 'MEG1543', 'MEG1542', 'MEG1613', 'MEG1612', 'MEG1622', 'MEG1623', 'MEG1632', 'MEG1633', 'MEG1643',
                'MEG1642', 'MEG1722', 'MEG1723']
    eeg_roi = ['EEG005', 'EEG006', 'EEG009', 'EEG010', 'EEG011', 'EEG012', 'EEG013', 'EEG014', 'EEG015', 'EEG019', 'EEG020', 'EEG021', 'EEG022', 'EEG028', 'EEG029', 'EEG030', 'EEG031', 'EEG032']

    # Show sensors of interest on data
    # - mag
    picks = mne.pick_channels(item_ev_mean.ch_names, mag_roi)
    mask = np.zeros((len(item_ev_mean.ch_names), 1), dtype=bool)
    mask[picks, :] = True
    timepoint = .072
    fig = item_ev_mean.copy().crop(tmin=timepoint, tmax=timepoint).plot_topomap(ch_type='mag', times=timepoint, show_names=False, mask=mask,
                                                                   mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=8))
    timepoint = .136
    fig = item_ev_mean.copy().crop(tmin=timepoint, tmax=timepoint).plot_topomap(ch_type='mag', times=timepoint, show_names=False, mask=mask,
                                                                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=8))
    fig.savefig('tmpfig/mag_chans.png', bbox_inches='tight', dpi=300)
    # - grad
    picks = mne.pick_channels(item_ev_mean.ch_names, grad_roi)
    mask = np.zeros((len(item_ev_mean.ch_names), 1), dtype=bool)
    mask[picks, :] = True
    timepoint = .072
    fig = item_ev_mean.copy().crop(tmin=timepoint, tmax=timepoint).plot_topomap(ch_type='grad', times=timepoint, show_names=False, mask=mask,
                                                                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=8))
    timepoint = .136
    fig = item_ev_mean.copy().crop(tmin=timepoint, tmax=timepoint).plot_topomap(ch_type='grad', times=timepoint, show_names=False, mask=mask,
                                                                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=8))
    fig.savefig('tmpfig/grad_chans.png', bbox_inches='tight', dpi=300)
    # - eeg
    picks = mne.pick_channels(item_ev_mean.ch_names, eeg_roi)
    mask = np.zeros((len(item_ev_mean.ch_names), 1), dtype=bool)
    mask[picks, :] = True
    timepoint = .072
    fig = item_ev_mean.copy().crop(tmin=timepoint, tmax=timepoint).plot_topomap(ch_type='eeg', times=timepoint, show_names=False, mask=mask,
                                                                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=8))
    timepoint = .136
    fig = item_ev_mean.copy().crop(tmin=timepoint, tmax=timepoint).plot_topomap(ch_type='eeg', times=timepoint, show_names=False, mask=mask,
                                                                          mask_params=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=8))
    fig.savefig('tmpfig/eeg_chans.png', bbox_inches='tight', dpi=300)

# =========================================================== #
# Data extraction
# =========================================================== #
for subject in config.subjects_list:
    # subject = config.subjects_list[3]
    print('\n########## peak data extraction for subject ' + subject + " ##########")
    start_time = time.time()

    # Load epochs data
    print('      Loading epochs...')
    if resid_epochs:
        epochs = epoching_funcs.load_resid_epochs_items(subject)
    else:
        if cleaned:
            epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
        else:
            epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
            epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
    # epo1 = epochs[4000].copy()  # for sanity check
    if lowpass_epochs:
        print('      Low pass filtering...')
        epochs = epochs.filter(l_freq=None, h_freq=30, n_jobs=n_jobs)  # default parameters (maybe should filter raw data instead of epochs...)
    # epo2 = epochs[4000].copy()  # for sanity check
    if detrend_epochs:
        print('      Detrending...')
        # epochs = epochs.detrend() # does not work...
        epochs._data[:] = scipy.signal.detrend(epochs.get_data(), axis=-1, type='linear')
    # epo3 = epochs[4000].copy()  # for sanity check
    if baseline_epochs:
        print('      Baseline correction...')
        epochs = epochs.apply_baseline(baseline=(-0.050, 0))
    # epo4 = epochs[4000].copy()  # for sanity check

    # # sanity checks
    # epo1.crop(tmin=.030, tmax=.180).average(picks=mag_roi).plot_joint()
    # epo2.crop(tmin=.030, tmax=.180).average(picks=mag_roi).plot_joint()
    # epo3.crop(tmin=.030, tmax=.180).average(picks=mag_roi).plot_joint()
    # epo4.crop(tmin=.030, tmax=.180).average(picks=mag_roi).plot_joint()
    # channel, latency, value = epo1.average(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
    # channel, latency, value = epo2.average(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
    # channel, latency, value = epo3.average(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
    # channel, latency, value = epo4.average(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)

    # Add surprise100 in metadata
    metadata = epoching_funcs.update_metadata(subject, clean=True, new_field_name=None, new_field_values=None)
    metadata["surprise100"] = metadata["surprise_100.00000"]  # "rename" the variable
    metadata = metadata[metadata.columns.drop(list(metadata.filter(regex='surprise_')))]  # remove all other surprise versions
    # add a balanced_standard yes/no column to metadata (position matched with deviants)
    metadata = epoching_funcs.metadata_balance_epochs_violation_positions(metadata)
    epochs.metadata = metadata
    print("      Preparing data took %.01f minutes ---" % ((time.time() - start_time) / 60))

    # Extract peak info for each epoch, in each time-win of interest, for each sensor_roi
    old_version = False
    if old_version:
        print('      Extracting peak data for ' + str(len(epochs)) + ' epochs...')
        epochs = epochs.crop(tmin=.000, tmax=.230)  # improve computing time?
        results_data = epochs.metadata.copy()
        results_data.index = range(0, len(results_data))  # we want df index corresponding to epochs
        results_data['Subject'] = subject
        results_data['n1_mag_val'] = np.nan
        results_data['n1_mag_lat'] = np.nan
        results_data['p2_mag_val'] = np.nan
        results_data['p2_mag_lat'] = np.nan
        results_data['n1_grad_val'] = np.nan
        results_data['n1_grad_lat'] = np.nan
        results_data['p2_grad_val'] = np.nan
        results_data['p2_grad_lat'] = np.nan
        results_data['n1_eeg_val'] = np.nan
        results_data['n1_eeg_lat'] = np.nan
        results_data['p2_eeg_val'] = np.nan
        results_data['p2_eeg_lat'] = np.nan
        for nepo in range(len(epochs)):
            channel, latency, value = epochs[nepo].average().pick(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
            results_data.loc[nepo, 'n1_mag_val'] = value
            results_data.loc[nepo, 'n1_mag_lat'] = latency
            channel, latency, value = epochs[nepo].average().pick(picks=mag_roi).get_peak(tmin=p2m_timewin[0], tmax=p2m_timewin[1], return_amplitude=True)
            results_data.loc[nepo, 'p2_mag_val'] = value
            results_data.loc[nepo, 'p2_mag_lat'] = latency
            channel, latency, value = epochs[nepo].average().pick(picks=grad_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
            results_data.loc[nepo, 'n1_grad_val'] = value
            results_data.loc[nepo, 'n1_grad_lat'] = latency
            channel, latency, value = epochs[nepo].average().pick(picks=grad_roi).get_peak(tmin=p2m_timewin[0], tmax=p2m_timewin[1], return_amplitude=True)
            results_data.loc[nepo, 'p2_grad_val'] = value
            results_data.loc[nepo, 'p2_grad_lat'] = latency
            channel, latency, value = epochs[nepo].average().pick(picks=eeg_roi).get_peak(tmin=n1_timewin[0], tmax=n1_timewin[1], return_amplitude=True)
            results_data.loc[nepo, 'n1_eeg_val'] = value
            results_data.loc[nepo, 'n1_eeg_lat'] = latency
            channel, latency, value = epochs[nepo].average().pick(picks=eeg_roi).get_peak(tmin=p2_timewin[0], tmax=p2_timewin[1], return_amplitude=True)
            results_data.loc[nepo, 'p2_eeg_val'] = value
            results_data.loc[nepo, 'p2_eeg_lat'] = latency
        del epochs
        output_file = op.join(results_path, subject + '_peakdata.csv')
        results_data.to_csv(output_file, index=False)
        print('      ========> ' + output_file + " saved !")
        print("      --- Took %.01f minutes ---" % ((time.time() - start_time) / 60))
    else:
        #############################"
        nn = 0
        print('      N created evoked =', sep=' ', end=' ', flush=True)
        for ttype in ['habituation', 'standard', 'violation']:
            for seqID in range(1, 8):
                for stimPos in range(1, 17):
                    if ttype == 'habituation':
                        selec = epochs['TrialNumber <= 10 and SequenceID == ' + str(seqID) + ' and StimPosition == ' + str(stimPos)].copy()
                    elif ttype == 'standard':  # For standards, taking only items from trials with no violation
                        selec = epochs['TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationInSequence == 0 and StimPosition == ' + str(stimPos)].copy()
                    elif ttype == 'violation':
                        selec = epochs['TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationOrNot == 1 and StimPosition == ' + str(stimPos)].copy()
                    if len(selec) > 0:
                        # create metadata info for this subset of epochs
                        selecMetadata = pd.DataFrame(selec.metadata.mean()).transpose()  # "average" metadata for this subset of epochs
                        selecMetadata['nave'] = len(selec)
                        selecMetadata['trialtype'] = ttype
                        # create evoked
                        ev = selec.average().filter(l_freq=None, h_freq=30, n_jobs=n_jobs).detrend().apply_baseline((-0.050, 0))
                        # ev = selec.average().detrend().apply_baseline((-0.050, 0))
                        # extract peak data
                        channel, latency, value = ev.copy().pick(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True, ch_type='mag')
                        selecMetadata['n1_mag_val'] = value
                        selecMetadata['n1_mag_lat'] = latency
                        channel, latency, value = ev.copy().pick(picks=mag_roi).get_peak(tmin=p2m_timewin[0], tmax=p2m_timewin[1], return_amplitude=True, ch_type='mag')
                        selecMetadata['p2_mag_val'] = value
                        selecMetadata['p2_mag_lat'] = latency
                        channel, latency, value = ev.copy().pick(picks=grad_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True, ch_type='grad', merge_grads=False)
                        selecMetadata['n1_grad_val'] = value
                        selecMetadata['n1_grad_lat'] = latency
                        channel, latency, value = ev.copy().pick(picks=grad_roi).get_peak(tmin=p2m_timewin[0], tmax=p2m_timewin[1], return_amplitude=True, ch_type='grad', merge_grads=False)
                        selecMetadata['p2_grad_val'] = value
                        selecMetadata['p2_grad_lat'] = latency
                        channel, latency, value = ev.copy().pick(picks=eeg_roi).get_peak(tmin=n1_timewin[0], tmax=n1_timewin[1], return_amplitude=True, ch_type='eeg')
                        selecMetadata['n1_eeg_val'] = value
                        selecMetadata['n1_eeg_lat'] = latency
                        channel, latency, value = ev.copy().pick(picks=eeg_roi).get_peak(tmin=p2_timewin[0], tmax=p2_timewin[1], return_amplitude=True, ch_type='eeg')
                        selecMetadata['p2_eeg_val'] = value
                        selecMetadata['p2_eeg_lat'] = latency
                        # add to the results table
                        if nn == 0:
                            results_data = selecMetadata
                        else:
                            results_data = results_data.append(selecMetadata)
                        nn += 1
                        if nn % 10 == 0:
                            print(nn, sep=' ', end=' ', flush=True)
        print("\n      %.01f epochs per evoked on average (median = %.01f) ---" % (np.mean(results_data.nave), np.median(results_data.nave)))
        del epochs
        results_data['Subject'] = subject
        output_file = op.join(results_path, subject + '_peakdata.csv')
        results_data.to_csv(output_file, index=False)
        print('      ========> ' + output_file + " saved !")
        print("      --- Took %.01f minutes ---" % ((time.time() - start_time) / 60))

# =========================================================== #
# Merge group data
# =========================================================== #
group_data = []
for subject in config.subjects_list[0:20]:
    print(subject)
    data = pd.read_csv(op.join(results_path, subject + '_peakdata.csv'))
    group_data.append(data)
group_data_merged = pd.concat(group_data)
output_file = op.join(results_path, 'allsubjects_peakdata.csv')
group_data_merged.to_csv(output_file, index=False)
print('      ========> ' + output_file + " saved !")

# =========================================================== #
# TESTS EXPLORE DATA
# =========================================================== #
plt.close('all')
selec = epochs['TrialNumber <= 10 and SequenceID == 3 and ViolationInSequence == 0 and StimPosition == 11'].copy()
ev = selec.copy().average().filter(l_freq=None, h_freq=30, n_jobs=n_jobs).detrend().apply_baseline((-0.050, 0))
channel, latency, value = ev.copy().pick(picks=mag_roi).get_peak(tmin=p2m_timewin[0], tmax=p2m_timewin[1], return_amplitude=True)
channel, latency, value = ev.copy().pick(picks=grad_roi).get_peak(tmin=p2m_timewin[0], tmax=p2m_timewin[1], return_amplitude=True, ch_type='grad', merge_grads=True)

print(channel + ' ' + str(latency*1000))
fig, ax = plt.subplots()
ev.plot(picks=mag_roi, spatial_colors=True, time_unit='ms', axes=ax)
# ax.set(xticks=np.arange(0, 301, 5))
ax.grid(True)

nepo = 8428
channel, latency, value = epochs[nepo].average(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
print(latency)
epochs[nepo].average(picks=mag_roi).plot(ylim=dict(mag=[-1600, 1600]))
epochs[nepo].copy().average(picks=mag_roi).detrend().plot(ylim=dict(mag=[-1600, 1600]))
epochs[nepo].copy().average(picks=mag_roi).filter(l_freq=None, h_freq=20, n_jobs=n_jobs).plot(ylim=dict(mag=[-1600, 1600]))
epochs[nepo].copy().average(picks=mag_roi).filter(l_freq=None, h_freq=20, n_jobs=n_jobs).detrend().plot(ylim=dict(mag=[-1600, 1600]))
epochs[500].copy().plot_psd(picks=mag_roi)
epochs[500].copy().filter(l_freq=None, h_freq=20, n_jobs=n_jobs).plot_psd(picks=mag_roi)

#
# nepo = 3958
# channel, latency, value = epochs[nepo].average(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
# print(latency)
# epochs[nepo].average(picks=mag_roi).plot(ylim=dict(mag=[-1600, 1600]))
#
# nepo = 4670
# channel, latency, value = epochs[nepo].average(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
# print(latency)
# epochs[nepo].average(picks=mag_roi).plot(ylim=dict(mag=[-1600, 1600]))
#
# nepo = 8006
# channel, latency, value = epochs[nepo].average(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
# print(latency)
# epochs[nepo].average(picks=mag_roi).plot(ylim=dict(mag=[-1600, 1600]))
#
# nepo = 5501
# channel, latency, value = epochs[nepo].average(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True)
# print(latency)
# epochs[nepo].average(picks=mag_roi).plot(ylim=dict(mag=[-1600, 1600]))
