import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import time
from mne.decoding import EMS
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage.filters import gaussian_filter1d
import config


Fosca_linux = False
if Fosca_linux:
    config.meg_dir = '/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/'

subject = 'pa_190002'


# ============== load the epochs ============================

print("Processing subject: %s" % subject)
meg_subject_dir = op.join(config.meg_dir, subject)
fig_path = op.join(config.study_path, 'Figures', 'EMS') + op.sep
sensor_types = ['eeg', 'grad', 'mag']
# Load epoch data
extension = subject + '_epo'
fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
print("Input: ", fname_in)
epochs = mne.read_epochs(fname_in, preload=True)

# ============== balance violations and standards by position for each sequence ============================

epochs_balanced_allseq = []

for seqID in range(1, 8):

    epochs_seq = epochs['SequenceID == "' + str(seqID) + '"'].copy()
    tmp = epochs_seq['ViolationOrNot == "1"']  # Deviant trials
    devpos = np.unique(tmp.metadata.StimPosition)  # Position of deviants
    # Keep only positions where there can be deviants
    epochs_seq = epochs_seq['StimPosition == "' + str(devpos[0]) +
                            '" or StimPosition == "' + str(devpos[1]) +
                            '" or StimPosition == "' + str(devpos[2]) +
                            '" or StimPosition == "' + str(devpos[3]) + '"']

    epochs_seq_noviol = epochs_seq["ViolationInSequence == 0"]
    epochs_seq_viol = epochs_seq["ViolationInSequence > 0 and ViolationOrNot ==1"]

    epochs_balanced_allseq.append([epochs_seq_noviol, epochs_seq_viol])
    print('We appended the balanced epochs for SeqID%i' % seqID)

epochs_balanced = mne.concatenate_epochs(list(np.hstack(epochs_balanced_allseq)))

# ======= epochs for specific sensor types =======================

epochs_balanced_mag = epochs_balanced.copy().pick_types(meg='mag')
epochs_balanced_grad = epochs_balanced.copy().pick_types(meg='grad')
epochs_balanced_eeg = epochs_balanced.copy().pick_types(eeg=True, meg=False)


# ============== create the temporary labels that will allow us to create the training and testing sets in a nicely balanced way =======================

metadata_epochs = epochs_balanced.metadata
y_tmp = [int(metadata_epochs['SequenceID'].values[i]*1000 + metadata_epochs['StimPosition'].values[i]*10 + metadata_epochs['ViolationOrNot'].values[i]) for i in range(len(epochs_balanced))]
y_violornot = np.asarray(epochs_balanced.metadata['ViolationOrNot'].values)

# ==========================================

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

    EMS_results[senso] = {'EMS': All_EMS, 'train_ind': training_inds, 'test_ind': testing_inds, 'epochs': epochs_all[l]}

# ======= si on sauvegarde ça, c'est cool on a tout ==============
np.save(op.join(meg_subject_dir, 'EMS_results.npy'), EMS_results)

EMS_results = np.load(op.join(meg_subject_dir, 'EMS_results.npy')).item()



sens = 'grad'
EMS_sens = EMS_results[sens]['EMS']
epochs_sens = EMS_results[sens]['epochs']

n_epochs, n_channels, n_times = epochs_sens.get_data().shape
X_transform = np.zeros((n_epochs, n_times))
for fold_number in range(4):
    test_indices = EMS_results[sens]['test_ind'][fold_number]
    epochs_sens_test = epochs_sens[test_indices]
    X = epochs_sens_test.get_data()
    X_scaled = X / np.std(X)
    # Generate the transformed data
    X_transform[test_indices] = EMS_sens[fold_number].transform(X_scaled)


# Generalization for training time time_train and testing time time_test
GAT = np.zeros((4,n_times, n_times))
for fold_number in range(4):
    test_indices = EMS_results[sens]['test_ind'][fold_number]
    epochs_sens_test = epochs_sens[test_indices]
    inds_seq_noviol = np.where(epochs_sens_test.metadata['ViolationOrNot'].values == 0)[0]
    inds_seq_viol = np.where(epochs_sens_test.metadata['ViolationOrNot'].values == 1)[0]
    X = epochs_sens_test.get_data()
    X_scaled = X / np.std(X)
    for time_train in range(n_times):
        for time_test in range(n_times):
            GAT_each_epoch = np.dot(EMS_sens[fold_number].filters_[:, time_train], X_scaled[:, :, time_test].T)
            # Generate the transformed data
            GAT[fold_number,time_train, time_test] = np.mean(GAT_each_epoch[inds_seq_noviol]) - np.mean(GAT_each_epoch[inds_seq_viol])
GAT_avg = np.mean(GAT, axis=0)

minT = epochs_sens_test.times[1]*1000
maxT = epochs_sens_test.times[-1]*1000
plt.figure()
plt.imshow(GAT_avg.T, origin='lower', extent=[minT,maxT,minT,maxT])
plt.title('Average EMS signal - ' + sens)
plt.xlabel('Training time (ms)')  # Correct ?
plt.ylabel('Testing time (ms)')  # Correct ?
plt.colorbar()
plt.show()
plt.savefig(fig_path + 'GAT_all_sequences_' + sens)

plt.figure()
plt.title('Average EMS signal - ' + sens)
plt.axhline(0, linestyle='-', color='black', linewidth=1)
plt.plot(epochs_sens.times, X_transform[y_violornot == 0].mean(0), label="standard", linestyle='--', linewidth=0.5)
plt.plot(epochs_sens.times, X_transform[y_violornot == 1].mean(0), label="deviant", linestyle='--', linewidth=0.5)
plt.plot(epochs_sens.times, X_transform[y_violornot == 0].mean(0) - X_transform[y_violornot == 1].mean(0), label="standard vs deviant")
plt.xlabel('Time (ms)')
plt.ylabel('a.u.')
plt.legend(loc='best')
plt.show()
plt.savefig(fig_path + 'Average_EMS_all_sequences_' + sens)
plt.close('all')












# Generalization for training time time_train and testing time time_test // one seq // one sensor type
SeqID = 4
GAT = np.zeros((4,n_times, n_times))
for fold_number in range(4):
    test_indices = EMS_results[sens]['test_ind'][fold_number]
    epochs_sens_test = epochs_sens[test_indices]
    # inds_seq_noviol = np.where(epochs_sens_test.metadata['ViolationOrNot'].values == 0)[0]
    # inds_seq_viol = np.where(epochs_sens_test.metadata['ViolationOrNot'].values == 1)[0]
    inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == SeqID) & (epochs_sens_test.metadata['ViolationOrNot'].values == 0))[0]
    inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == SeqID) & (epochs_sens_test.metadata['ViolationOrNot'].values == 1))[0]
    X = epochs_sens_test.get_data()
    X_scaled = X / np.std(X)
    for time_train in range(n_times):
        for time_test in range(n_times):
            GAT_each_epoch = np.dot(EMS_sens[fold_number].filters_[:, time_train], X_scaled[:, :, time_test].T)
            # Generate the transformed data
            GAT[fold_number,time_train, time_test] = np.mean(GAT_each_epoch[inds_seq_noviol]) - np.mean(GAT_each_epoch[inds_seq_viol])
GAT_avg = np.mean(GAT, axis=0)

minT = epochs_sens_test.times[1]*1000
maxT = epochs_sens_test.times[-1]*1000
plt.figure()
plt.imshow(GAT_avg, origin='lower', extent=[minT, maxT, minT, maxT])
plt.title('Average EMS signal - SeqID_' + str(SeqID) + ' - ' + sens)
plt.xlabel('Training time (ms)')  # Correct ?
plt.ylabel('Testing time (ms)')  # Correct ?
plt.colorbar()
plt.savefig(fig_path + 'GAT_SeqID_' + str(SeqID) + '_' + sens)

plt.figure()
plt.title('Average EMS signal - SeqID_' + str(SeqID) + ' - ' + sens)
plt.axhline(0, linestyle='-', color='black', linewidth=1)
plt.plot(epochs_sens.times, X_transform[inds_seq_viol].mean(0), label="standard", linestyle='--', linewidth=0.5)
plt.plot(epochs_sens.times, X_transform[inds_seq_noviol].mean(0), label="deviant", linestyle='--', linewidth=0.5)
plt.plot(epochs_sens.times, X_transform[inds_seq_noviol].mean(0) - X_transform[inds_seq_viol].mean(0), label="standard vs deviant")
plt.xlabel('Time (ms)')
plt.ylabel('a.u.')
plt.legend(loc='best')
plt.show()
plt.savefig(fig_path + 'Average_EMS_SeqID_' + str(SeqID) + '_' + sens)
plt.close('all')














# ===================== plots ==========================
sens = 'grad'
EMS_sens = EMS_results[sens]['EMS']
epochs_sens = EMS_results[sens]['epochs']

plt.close('all')
plt.figure()
plt.title('single trial surrogates')
plt.imshow(X_transform[y_violornot.argsort()], origin='lower', aspect='auto',
           extent=[epochs_sens.times[0], epochs_sens.times[-1], 1, len(X_transform)],
           cmap='RdBu_r', vmin=-20, vmax=20)
plt.colorbar()
plt.xlabel('Time (ms)')
plt.ylabel('Trials (reordered by condition)')
plt.show()


# =====  Collect and plot scores "viol - noviol" for each sequenceID =====
sens = 'mag'
EMS_sens = EMS_results[sens]['EMS']
epochs_sens = EMS_results[sens]['epochs']

plt.close('all')
for ll in range(7):
    inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == ll+1) & (
                epochs_sens_test.metadata['ViolationOrNot'].values == 0))[0]
    inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == ll+1) & (
                epochs_sens_test.metadata['ViolationOrNot'].values == 1))[0]
    plt.figure()
    plt.title(sens + ', seqID_' + str(ll+1))
    plt.axhline(0, linestyle='-', color='black', linewidth=1)
    plt.plot(epochs_sens.times, X_transform[inds_seq_viol].mean(0), label="deviant", linestyle='--', linewidth=0.5)
    plt.plot(epochs_sens.times, X_transform[inds_seq_noviol].mean(0), label="standard", linestyle='--', linewidth=0.5)
    ysmoothed = gaussian_filter1d(X_transform[inds_seq_noviol].mean(0) - X_transform[inds_seq_viol].mean(0), sigma=2)
    plt.plot(epochs_sens.times, ysmoothed,label="standard vs deviant")
    plt.xlabel('Time (ms)')
    plt.ylabel('a.u.')
    plt.legend(loc='best')
    plt.show()

# all sequences together
score_all = []
for seqID in range(1, 8):
    inds_seq_noviol = np.where((epochs_sens_test.metadata['SequenceID'].values == seqID) & (
                epochs_sens_test.metadata['ViolationOrNot'].values == 0))[0]
    inds_seq_viol = np.where((epochs_sens_test.metadata['SequenceID'].values == seqID) & (
                epochs_sens_test.metadata['ViolationOrNot'].values == 1))[0]
    score = X_transform[inds_seq_noviol].mean(0) - X_transform[inds_seq_viol].mean(0)
    score_all.append(score)
plt.figure()
plt.title(sens)
plt.axhline(0, linestyle='-', color='black', linewidth=1)
for ll in range(7):
    # Original
    # plt.plot(epochs_OI.times, score_sensor_type[sensor][ll], label='SeqID' + str(ll+1))
    # Or gaussian filtered
    ysmoothed = gaussian_filter1d(score_all[ll], sigma=2)
    plt.plot(epochs_balanced.times, ysmoothed, label='SeqID' + str(ll+1))
plt.xlabel('Time (ms)')
plt.ylabel('a.u.')
plt.legend(loc='best')
plt.show()
plt.savefig(fig_path + 'EMS_from_all_ALL_sequences_' + sens)
# plt.close('all')










# %%
# now we want to test the spatial filters for a given time over all the 16 items of a sequence.
# We still have to use some folds things because we cannot apply the filter trained on a violation of a sequence to the 16 items of that sequence

# 1 - load the epochs on the first element
# 2 - across the different folds, extract the indices of the sequence that will allow us to apply the spatial filter to that sequence

# load the epochs on the first element
meg_subject_dir = op.join(config.meg_dir, subject)
fig_path = op.join(config.study_path, 'Figures', 'EMS') + op.sep
extension = subject + '_1st_element_epo'
fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
print("Input: ", fname_in)
epochs_1st_element = mne.read_epochs(fname_in, preload=True)

# epochs_1st = {'mag': epochs_balanced.copy().pick_types(meg='mag'),'grad':epochs_balanced.copy().pick_types(meg='grad'), 'eeg':epochs_balanced.copy().pick_types(eeg=True, meg=False)}
epochs_1st = {'mag': epochs_1st_element.copy().pick_types(meg='mag'),'grad':epochs_1st_element.copy().pick_types(meg='grad'), 'eeg':epochs_1st_element.copy().pick_types(eeg=True, meg=False)}

# load the EMS things
EMS_results = np.load(op.join(meg_subject_dir, 'EMS_results.npy')).item()
sens = 'grad'
EMS_sens = EMS_results[sens]['EMS']
epochs_sens = EMS_results[sens]['epochs']
epochs_1st_sens = epochs_1st[sens]

# Generalization for training time time_train and testing time time_test

# data_frame_meta = []
data_frame_meta = pd.DataFrame([])

elapsed = 0
for fold_number in range(4):
    start = time.time()
    print('Fold ' + str(fold_number + 1) + ' on 4')
    print('Elapsed since last fold: ' + str(elapsed))
    test_indices = EMS_results[sens]['test_ind'][fold_number]
    epochs_sens_test = epochs_sens[test_indices] ###

    for m in test_indices:
        # we extract the fields that will allow us to identify properly the corresponding epochs_1st_sens
        # seqID_m = epochs_sens_test[m].metadata['SequenceID']
        # run_m = epochs_sens_test[m].metadata['RunNumber']
        # trial_number_m = epochs_sens_test[m].metadata['TrialNumber'] # this is the number of the trial, that will allow to determine which sequence within the run of 46 is the one that was left appart
        seqID_m = epochs_sens[m].metadata['SequenceID'].values[0]
        run_m = epochs_sens[m].metadata['RunNumber'].values[0]
        trial_number_m = epochs_sens[m].metadata['TrialNumber'].values[0] # this is the number of the trial, that will allow to determine which sequence within the run of 46 is the one that was left appart
        epochs_1st_sens_m = epochs_1st_sens['SequenceID == "%i" and RunNumber == %i and TrialNumber == %i' % (seqID_m, run_m, trial_number_m)]

        # print('Fold ' + str(fold_number + 1) + ' on 4; test_index ' + str(m))
        # apply the spatial filter to the whole data of the epoch
        data_1st_el_m = epochs_1st_sens_m.get_data()
        # for point_of_interest in range(data_1st_el_m.shape[-1]):
        for point_of_interest in range(EMS_sens[fold_number].filters_.shape[-1]):
            epochs_1st_sens_m_filtered_data = np.dot(EMS_sens[fold_number].filters_[:, point_of_interest], data_1st_el_m.T)
            metadata_m = []
            metadata_m = epochs_1st_sens_m.metadata
            # metadata_m['Projection_on_EMS'] = epochs_1st_sens_m_filtered_data
            metadata_m['Projection_on_EMS'] = [epochs_1st_sens_m_filtered_data]
            metadata_m['EMS_filter_datapoint'] = int(point_of_interest)
            # data_frame_meta.append(metadata_m)
            data_frame_meta = data_frame_meta.append(metadata_m)

    end = time.time()
    elapsed = end - start

# all_data_1st = pd.concatenate(data_frame_meta)
# all_data_1st = pd.concat(data_frame_meta)
all_data_1st = data_frame_meta

# ========= save
np.save(op.join(meg_subject_dir, 'EMS_results_v2.npy'), all_data_1st)

all_data_1st = np.load(op.join(meg_subject_dir, 'EMS_results_v2.npy')) #.item()

# ========= ensuite, pour regarder par sequence et position de violation pour une topographie de filtre correspondant a un data point donne ==========

seqID = 2
# point_of_interest = 60
viol_pos = 9

plt.close('all')
for point_of_interest in np.arange(0, 225,50):
    # metadata_16items_seq_viol = all_data_1st['SequenceID == %s and EMS_filter_datapoint == %i and Violation_position_1234 == %i'%(seqID, point_of_interest,viol_pos)]
    metadata_16items_seq_viol = all_data_1st[(all_data_1st.SequenceID == seqID) &
                                             (all_data_1st.EMS_filter_datapoint == point_of_interest) &
                                             (all_data_1st.ViolationInSequence == viol_pos)]
    # ======== ca devrait faire la moyenne sur les 4 sequences correspondant a ce SeqID et a la position de la violation.
    # data_16items_seq_viol = np.mean(metadata_16items_seq_viol.metadata["Projection_on_EMS"].values, axis=0)
    data_16items_seq_viol = np.mean(metadata_16items_seq_viol["Projection_on_EMS"].values, axis=0)
    # ====== il ne reste plus qu a plotter tout ce beau monde =========


    plt.plot(epochs_1st_sens.times,data_16items_seq_viol, label = str(point_of_interest*4 -100))
plt.title('SequenceID_' + str(seqID) + ', violation position = ' + str(viol_pos) + ' (i.e. ' + str((viol_pos-1)*250) +  ' ms) - N=' + str(metadata_16items_seq_viol.shape[0]))
plt.legend()
plt.show()








###
sens = 'grad'
EMS_sens = EMS_results[sens]['EMS']
epochs_sens = EMS_results[sens]['epochs']
epochs_1st_sens = epochs_1st[sens]

# Generalization for training time time_train and testing time time_test

# data_frame_meta = []
data_frame_meta = pd.DataFrame([])
times = [x / 1000 for x in range(100, 700, 50)]
elapsed = 0
# data_for_epoch_object = np.zeros((epochs_1st_sens.get_data().shape[0]*len(times), epochs_1st_sens.get_data().shape[2]))
data_for_epoch_object = np.zeros((epochs_sens.get_data().shape[0]*len(times), epochs_1st_sens.get_data().shape[2]))
counter = 0
for fold_number in range(4):

    print('Fold ' + str(fold_number + 1) + ' on 4...')
    start = time.time()
    test_indices = EMS_results[sens]['test_ind'][fold_number]
    epochs_sens_test = epochs_sens[test_indices] ###
    points = epochs_sens_test.time_as_index(times)
    for m in test_indices:

        seqID_m = epochs_sens[m].metadata['SequenceID'].values[0]
        run_m = epochs_sens[m].metadata['RunNumber'].values[0]
        trial_number_m = epochs_sens[m].metadata['TrialNumber'].values[0] # this is the number of the trial, that will allow to determine which sequence within the run of 46 is the one that was left appart
        epochs_1st_sens_m = epochs_1st_sens['SequenceID == "%i" and RunNumber == %i and TrialNumber == %i' % (seqID_m, run_m, trial_number_m)]

        # print('Fold ' + str(fold_number + 1) + ' on 4; test_index ' + str(m))
        # apply the spatial filter to the whole data of the epoch
        data_1st_el_m = epochs_1st_sens_m.get_data()
        for mm, point_of_interest in enumerate(points):
            # print(mm)
            epochs_1st_sens_m_filtered_data = np.dot(EMS_sens[fold_number].filters_[:, point_of_interest], data_1st_el_m.T)
            data_for_epoch_object[counter, :] = np.squeeze(epochs_1st_sens_m_filtered_data)
            metadata_m = []
            metadata_m = epochs_1st_sens_m.metadata
            metadata_m['EMS_filter_datapoint'] = int(point_of_interest)
            metadata_m['EMS_filter_time'] = times[mm]
            data_frame_meta = data_frame_meta.append(metadata_m)
            counter += 1
    end = time.time()
    elapsed = end - start
    print('... lasted: ' + str(elapsed) + ' s')

epochs_1st_data = np.asarray(epochs_1st_sens_m_filtered_data)

dat = data_for_epoch_object
dat = np.expand_dims(dat, axis=1)

# info = epochs_1st_sens.info.copy()
# info['ch_names'] = ['EMS']
# info['chs'] = info['chs'][1]
# info['chs']['ch_name'] = 'EMS'
info = mne.create_info(['EMS'], epochs_1st_sens.info['sfreq']) # but we may lose some info, such as times (with baseline)
epochs_to_save = mne.EpochsArray(dat, info,tmin=-0.5)
#epochs_to_save.times = epochs_1st_element.times.copy() # does not work ??
epochs_to_save.metadata = data_frame_meta
epochs_to_save.save(op.join(meg_subject_dir, 'Filter_on_16_items-epo.fif'), overwrite=True)


# data_frame_meta.to_csv('/Users/fosca/Downloads/Filter_on_16_items.csv')
# data_frame_meta.to_csv(op.join(meg_subject_dir, 'Filter_on_16_items.csv'))

# data_frame_meta.to_csv(op.join(meg_subject_dir, 'Filter_on_16_items.csv'))
# data_frame_meta = pd.read_csv(op.join(meg_subject_dir, 'Filter_on_16_items.csv'))

meg_subject_dir = op.join(config.meg_dir, subject)
epochs = mne.read_epochs(op.join(meg_subject_dir, 'Filter_on_16_items-epo.fif'), preload=True)



# Test plot
seqID = 2
viol_pos = 14
time_filter = 0.25
epochs_subset = epochs['SequenceID == "' + str(seqID)
                       + '" and EMS_filter_time == "' + str(time_filter)
                       + '" and ViolationInSequence == "' + str(viol_pos) + '"']
epochs_subset.average(picks='EMS').plot(picks='EMS')





# Plot in a loop
plt.close('all')
times = [x / 1000 for x in range(100, 700, 50)]
subtimes = range(0, len(times), 2)
times = [times[i] for i in subtimes]

times = [0.25, 0.5]


seqID = 2

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
fig.suptitle('EMS (MAG) - SequenceID_' + str(seqID), fontsize=12)
ax = axes.ravel()[::1]
epochs_seq_subset = epochs['SequenceID == "' + str(seqID) + '"']
violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])[1:]
n = 0
for viol_pos in violpos_list:
    for point_of_interest in times:
        epochs_subset = epochs['SequenceID == "' + str(seqID)
                               + '" and EMS_filter_time == "' + str(point_of_interest)
                               + '" and ViolationInSequence == "' + str(viol_pos) + '"']


        y = np.squeeze(epochs_subset.savgol_filter(2).average(picks='EMS').data)
        ax[n].plot(epochs_1st_element.times, y, label=str(point_of_interest))
        ax[n].legend(loc='upper left', fontsize=10)
        ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
        ax[n].axvline(0.25*(viol_pos-1), linestyle='-', color='black', linewidth=2)
        # ax[n].set_ylabel('Violation position = ' + str(viol_pos) + ' (i.e. ' + str((viol_pos - 1) * 250) + ' ms) - N=' + str(len(epochs_subset.events)))
    n += 1

    # plt.title('SequenceID_' + str(seqID) + ', violation position = ' + str(viol_pos) + ' (i.e. ' + str((viol_pos-1)*250) +  ' ms) - N=' + str(len(epochs_subset.events)))
fig.show()

        ax[n].axvline(0, linestyle='-', color='black', linewidth=2)
        ax[n].axvline(-100, linestyle='--', color='black', linewidth=1)
        ax[n].set_yticklabels([])
        ax[n].set_ylabel('SequenceID_' + str(x + 1))
        ax[n].legend(loc='upper left', fontsize=10)
        ax[n].set_xlim(-250, 750)