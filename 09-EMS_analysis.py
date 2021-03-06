from mne.parallel import parallel_func
from ABseq_func import *
import config
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.signal import savgol_filter


# make less parallel runs to limit memory usage
# N_JOBS = max(config.N_JOBS // 4, 1)
N_JOBS = 2  # config.N_JOBS
#
config.subjects_list = ['sub16-ma_190185']


def EMS_analysis(subject):
    # creating the EMS results dictionnary
    EMS_funcs.generate_EMS_all_sequences(subject)
    EMS_funcs.GAT_EMS(subject)
    EMS_funcs.GAT_EMS_4pos(subject)
    EMS_funcs.apply_EMS_filter_16_items_epochs(subject)

# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(EMS_analysis, n_jobs=N_JOBS)

parallel(run_func(subject) for subject in config.subjects_list)



# ______________________________________________________________________________________

# ======================== WHAT FOLLOWS IS TO PLOT THE EMS RESULTS =====================
# ______________________________________________________________________________________



config.subjects_list = ['sub01-pa_190002', 'sub02-ch_180036', 'sub03-mr_190273', 'sub04-rf_190499', 'sub05-cr_170417', 'sub06-kc_160388',
                        'sub07-jm_100109', 'sub08-cc_150418', 'sub09-ag_170045', 'sub10-gp_190568', 'sub11-fr_190151', 'sub12-lg_170436',
                        'sub13-lq_180242', 'sub14-js_180232', 'sub15-ev_070110',                    'sub17-mt_170249', 'sub18-eo_190576',
                        'sub19-mg_190180']


## =========================== EMS Generalization Across Time FIGURES =========================== ##

# ===== LOAD (& PLOT) INDIVIDUAL DATA ===== #
GAT_sens_seq_all = {sens: [] for sens in ['eeg', 'mag', 'grad']}
# for sens in ['eeg','mag','grad']:
#     GAT_sens_seq_all[sens]{'average_all_sequences': []}
for subject in config.subjects_list:
    EMS_path = op.join(config.EMS_path, subject)
    GAT_results = np.load(op.join(EMS_path, 'GAT_results.npy'), allow_pickle=True).item()
    print(op.join(EMS_path, 'GAT_results.npy'))
    times = GAT_results['times']
    GAT_results = GAT_results['GAT']
    sub_fig_path = op.join(config.fig_path, 'EMS', 'GAT', subject)
    utils.create_folder(sub_fig_path)
    for sens in ['eeg', 'mag', 'grad']:
        if not GAT_sens_seq_all[sens]:  # initialize the keys and empty lists only the first time
            GAT_sens_seq_all[sens] = {'SeqID_%i' % i: [] for i in range(1, 8)}
            GAT_sens_seq_all[sens]['average_all_sequences'] = []
        for key in ['SeqID_%i' % i for i in range(1, 8)]:
            GAT_sens_seq_all[sens][key].append(GAT_results[sens][key])

            # ================ Plot & save each subject / each sequence figures ???
            # EMS_funcs.plot_GAT_EMS(GAT_results[sens][key], times, sens=sens, save_path=sub_fig_path, figname='GAT_' + key + '_'); plt.close('all')

        # ================ Plot & save each subject / average of all sequences figures ???
        GAT_sens_seq_all[sens]['average_all_sequences'].append(GAT_results[sens]['average_all_sequences'])
        EMS_funcs.plot_GAT_EMS(GAT_results[sens]['average_all_sequences'], times, sens=sens, save_path=sub_fig_path, figname='GAT_all_seq'+'_'); plt.close('all')

# ===== GROUP AVG FIGURES ===== #
plt.close('all')
for sens in ['eeg', 'mag', 'grad']:
    GAT_avg_sens = GAT_sens_seq_all[sens]
    for seqID in range(1, 8):
        GAT_avg_sens_seq = GAT_avg_sens['SeqID_%i'%seqID]
        GAT_avg_sens_seq_groupavg = np.mean(GAT_avg_sens_seq,axis=0)
        EMS_funcs.plot_GAT_EMS(GAT_avg_sens_seq_groupavg, times, sens=sens, save_path=op.join(config.fig_path, 'EMS', 'GAT'), figname='GAT_'+str(seqID)+'_')
        plt.close('all')
    GAT_avg_sens_allseq_groupavg = np.mean(GAT_avg_sens['average_all_sequences'], axis=0)
    EMS_funcs.plot_GAT_EMS(GAT_avg_sens_allseq_groupavg, times, sens=sens, save_path=op.join(config.fig_path, 'EMS', 'GAT'), figname='GAT_all_seq'+'_')

## =========================== FULL SEQUENCE PROJECTION FIGURES =========================== ##

# ===== LOAD DATA ===== #
epochs_16_items_mag_test = []; epochs_16_items_grad_test = []; epochs_16_items_eeg_test = []
epochs_16_items_mag_habituation = []; epochs_16_items_grad_habituation = []; epochs_16_items_eeg_habituation = []
epochs_16_items_mag_test_window = []; epochs_16_items_grad_test_window = []; epochs_16_items_eeg_test_window = []
epochs_16_items_mag_habituation_window = []; epochs_16_items_grad_habituation_window = []; epochs_16_items_eeg_habituation_window = []
for subject in config.subjects_list:
    # epochs_16_items_mag_test.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag_filter_on_16_items_test-epo.fif')))
    # epochs_16_items_grad_test.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad_filter_on_16_items_test-epo.fif')))
    # epochs_16_items_eeg_test.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg_filter_on_16_items_test-epo.fif')))
    # epochs_16_items_mag_habituation.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag_filter_on_16_items_habituation-epo.fif')))
    # epochs_16_items_grad_habituation.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad_filter_on_16_items_habituation-epo.fif')))
    # epochs_16_items_eeg_habituation.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg_filter_on_16_items_habituation-epo.fif')))
    epochs_16_items_mag_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag_filter_on_16_items_test_window-epo.fif')))
    epochs_16_items_grad_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad_filter_on_16_items_test_window-epo.fif')))
    epochs_16_items_eeg_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg_filter_on_16_items_test_window-epo.fif')))
    epochs_16_items_mag_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag_filter_on_16_items_habituation_window-epo.fif')))
    epochs_16_items_grad_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad_filter_on_16_items_habituation_window-epo.fif')))
    epochs_16_items_eeg_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg_filter_on_16_items_habituation_window-epo.fif')))

# ===== FIGURES ===== #
save_folder = op.join(config.fig_path, 'EMS', 'Full_sequence_projection')
utils.create_folder(save_folder)


# Figure with multiple EMS projected (different times)
EMS_filter_times = [x / 1000 for x in range(100, 601, 100)]
NUM_COLORS = len(EMS_filter_times)
cm = plt.get_cmap('viridis')
colorslist = ([cm(1. * i / (NUM_COLORS-1)) for i in range(NUM_COLORS)])
epochs_list = {}
for sens in ['mag', 'grad', 'eeg']:
    if sens == 'mag':
        epochs_list['hab'] = epochs_16_items_mag_habituation
        epochs_list['test'] = epochs_16_items_mag_test
    elif sens == 'grad':
        epochs_list['hab'] = epochs_16_items_grad_habituation
        epochs_list['test'] = epochs_16_items_grad_test
    elif sens == 'eeg':
        epochs_list['hab'] = epochs_16_items_eeg_habituation
        epochs_list['test'] = epochs_16_items_eeg_test
    for seq_ID in range(1, 8):
        # "curve" figure  # UNCOMPLETE, TO DO WITH HAB TRIALS AND epochs_list NOW AS A DICT
        # EMS_filter_times = [x / 1000 for x in range(100, 601, 100)]
        # EMS_funcs.plot_EMS_projection_for_seqID(epochs_list, sensor_type=sens, seqID=seq_ID, EMS_filter_times=EMS_filter_times, color_mean=colorslist,
        #                                         save_path=op.join(save_folder,'Seq%i_%s_%i_%i_test.png' % (seq_ID, sens, min(EMS_filter_times) * 1000, max(EMS_filter_times) * 1000)))
        # "heatmap" figure
        EMS_filter_times = [x / 1000 for x in range(0, 701, 100)]
        EMS_funcs.plot_EMS_projection_for_seqID_heatmap(epochs_list, sensor_type=sens, seqID=seq_ID, EMS_filter_times=EMS_filter_times,
                                                        save_path=op.join(save_folder, 'hm_Seq%i_%s_%i_%i.png' % (seq_ID, sens, min(EMS_filter_times)*1000, max(EMS_filter_times)*1000)))

# Figure with only one EMS projected (average window)
epochs_list = {}
for sens in ['mag', 'grad', 'eeg']:
    if sens == 'mag':
        epochs_list['hab'] = epochs_16_items_mag_habituation_window
        epochs_list['test'] = epochs_16_items_mag_test_window
    elif sens == 'grad':
        epochs_list['hab'] = epochs_16_items_grad_habituation_window
        epochs_list['test'] = epochs_16_items_grad_test_window
    elif sens == 'eeg':
        epochs_list['hab'] = epochs_16_items_eeg_habituation_window
        epochs_list['test'] = epochs_16_items_eeg_test_window
    win_tmin = epochs_list['test'][0][0].metadata.EMS_filter_tmin_window[0]*1000
    win_tmax = epochs_list['test'][0][0].metadata.EMS_filter_tmax_window[0]*1000
    # for seq_ID in range(1, 8):
    #     # "curve" figure
    #     EMS_funcs.plot_EMS_projection_for_seqID_window(epochs_list, sensor_type=sens, seqID=seq_ID,
    #                                                    save_path=op.join(save_folder, 'Seq%i_%s_window_%i_%ims.png' % (seq_ID, sens, win_tmin, win_tmax)))
    EMS_funcs.plot_EMS_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type=sens, save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims.png' % ( sens, win_tmin, win_tmax)))

# ===== OTHER FIGURES ===== #
# Create & plot average of EMS filters topos for the given times (average 4 folds in each subject, average all subjects)
for sens in ['mag', 'grad', 'eeg']:
    times = [x / 1000 for x in range(0, 701, 100)]
    all_meanfilt_as_evo = []
    for subject in config.subjects_list:
        EMS_results_path = op.join(config.EMS_path, subject)
        EMS_results = np.load(op.join(EMS_results_path, 'EMS_results.npy'), allow_pickle=True).item()
        points = EMS_results[sens]['epochs'].time_as_index(times)
        EMS_sens = EMS_results[sens]['EMS']
        # average filters from the 4 folds
        meanfilt = np.average([EMS_sens[0].filters_, EMS_sens[1].filters_,  EMS_sens[2].filters_,  EMS_sens[3].filters_], axis=0)
        # filter for each timepoint of interest
        dat = []
        for mm, point_of_interest in enumerate(points):
            dat.append(meanfilt[:, point_of_interest])
        dat = np.stack(dat)
        # create "fake" evo object with times corresponding to the times of the different filters
        meanfilt_as_evo = mne.EvokedArray(dat.T, EMS_results[sens]['epochs'].info)
        meanfilt_as_evo.times = np.asarray(times)
        all_meanfilt_as_evo.append(meanfilt_as_evo)
    avg = mne.grand_average(all_meanfilt_as_evo)
    avg.plot_topomap(times=times)
    save_path=op.join(config.fig_path, 'EMS', 'Full_sequence_projection', 'avg_%s_filters.png'%sens)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close('all')

### ========================================================= ###
### PLOT GROUP AVG (item) EMS
### ========================================================= ###

for sens in ['mag', 'grad', 'eeg']:  # WILL LOAD THE +20Go OF DATA 3 TIMES !! (not optimal !!)
    # initialize group data dicts
    all_avg_stand = []
    all_avg_dev = []
    all_avg_diff = []
    all_avg_stand_seq = dict()
    all_avg_dev_seq = dict()
    all_avg_diff_seq = dict()
    for SeqID in range(1, 8):
        all_avg_stand_seq['SeqID_' + str(SeqID)] = []
        all_avg_dev_seq['SeqID_' + str(SeqID)] = []
        all_avg_diff_seq['SeqID_' + str(SeqID)] = []
    # load data & fill group data dict // LONG
    for subject in config.subjects_list:
        EMS_results_path = op.join(config.EMS_path, subject)
        EMS_results = np.load(op.join(EMS_results_path, 'EMS_results.npy'), allow_pickle=True).item()
        EMS_sens = EMS_results[sens]['EMS']
        epochs_sens = EMS_results[sens]['epochs']
        y_violornot = np.asarray(epochs_sens.metadata['ViolationOrNot'].values)
        y_seqID = np.asarray(epochs_sens.metadata['SequenceID'].values)
        n_epochs, n_channels, n_times = epochs_sens.get_data().shape
        X_transform_all_folds = []
        for fold_number in range(4):
            X_transform = np.zeros((n_epochs, n_times))
            test_indices = EMS_results[sens]['test_ind'][fold_number]
            epochs_sens_test = epochs_sens[test_indices]
            X = epochs_sens_test.get_data()
            X_scaled = X / np.std(X)
            # Generate the transformed data
            X_transform[test_indices] = EMS_sens[fold_number].transform(X_scaled)
            X_transform_all_folds.append(X_transform)
        # subject average
        avg_X_transform_all_folds = np.average(X_transform_all_folds, axis=0)
        # append to group data arrays
        all_avg_stand.append(avg_X_transform_all_folds[y_violornot == 0].mean(0))
        all_avg_dev.append(avg_X_transform_all_folds[y_violornot == 1].mean(0))
        all_avg_diff.append(avg_X_transform_all_folds[y_violornot == 0].mean(0) - avg_X_transform_all_folds[y_violornot == 1].mean(0))
        for SeqID in range(1, 8):
            all_avg_stand_seq['SeqID_'+str(SeqID)].append(avg_X_transform_all_folds[(y_violornot == 0) & (y_seqID == SeqID)].mean(0))
            all_avg_dev_seq['SeqID_'+str(SeqID)].append(avg_X_transform_all_folds[(y_violornot == 1) & (y_seqID == SeqID)].mean(0))
            all_avg_diff_seq['SeqID_'+str(SeqID)].append(avg_X_transform_all_folds[(y_violornot == 0) & (y_seqID == SeqID)].mean(0) - avg_X_transform_all_folds[(y_violornot == 1) & (y_seqID == SeqID)].mean(0))
    group_avg_all_stand = np.average(all_avg_stand, axis=0)
    group_avg_all_dev = np.average(all_avg_dev, axis=0)
    group_avg_all_diff = np.average(all_avg_diff, axis=0)

    # QUICK FIG ALL SEQUENCES
    plt.close('all')
    plt.figure(figsize=(10, 3))
    plt.plot(epochs_sens.times*1000, group_avg_all_stand, label='stand')
    plt.plot(epochs_sens.times*1000, group_avg_all_dev, label='dev')
    plt.plot(epochs_sens.times*1000, group_avg_all_diff, label='diff')
    plt.xlim(-100, 700)
    plt.xticks(np.arange(-100, 700, step=25),rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(op.join(config.fig_path, 'EMS', 'AllsequencesAVG_%s.png' % sens), dpi=300)

    # FIGURE FOR EACH SEQUENCE (with Stand/Dev/Diff)
    for SeqID in range(1, 8):
        group_avg_stand = np.average(all_avg_stand_seq['SeqID_' + str(SeqID)], axis=0)
        group_avg_dev = np.average(all_avg_dev_seq['SeqID_' + str(SeqID)], axis=0)
        group_avg_diff = np.average(all_avg_diff_seq['SeqID_' + str(SeqID)], axis=0)
        plt.figure()
        plt.title('Average EMS signal - ' + sens + '_ seqID_' + str(SeqID))
        plt.axhline(0, linestyle='-', color='black', linewidth=1)
        plt.plot(epochs_sens.times, group_avg_stand, label="Standard", linestyle='--', linewidth=0.5)
        plt.plot(epochs_sens.times, group_avg_dev, label="Deviant", linestyle='--', linewidth=0.5)
        plt.plot(epochs_sens.times, group_avg_diff, label="Standard vs Deviant")
        plt.xlabel('Time (ms)')
        plt.ylabel('a.u.')
        plt.legend(loc='best')
        plt.savefig(op.join(config.fig_path, 'EMS', 'Sequence_%i_%s.png' %(SeqID, sens)))
        plt.close('all')

    # ALL SEQUENCES IN THE SAME FIGURE (MEAN WITH CI) (just Diff)
    times = (epochs_sens.times)*1000
    filter = True
    NUM_COLORS = 7
    cm = plt.get_cmap('viridis')
    colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    plt.axhline(0, linestyle='-', color='black', linewidth=1)
    plt.title('Average EMS signal [dev-stand difference] - ' + sens)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    for SeqID in range(1, 8):
        color_mean = colorslist[SeqID - 1]
        mean = np.average(all_avg_diff_seq['SeqID_' + str(SeqID)], axis=0)
        ub = mean + sem(all_avg_diff_seq['SeqID_' + str(SeqID)], axis=0)
        lb = mean - sem(all_avg_diff_seq['SeqID_' + str(SeqID)], axis=0)
        if filter == True:
            mean = savgol_filter(mean, 11, 3)
            ub = savgol_filter(ub, 11, 3)
            lb = savgol_filter(lb, 11, 3)
        plt.fill_between(times, ub, lb, color=color_mean, alpha=.2)
        plt.plot(times, mean, color=color_mean, linewidth=1.5, label='SeqID_' + str(SeqID))
    plt.legend(loc='best', fontsize=9)
    ax.set_xlim(-100, 700)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('a.u.')
    plt.savefig(op.join(config.fig_path, 'EMS', 'All_sequences_diff_%s.png' % sens), dpi=300)
    ax.set_xlim(0, 250)
    fig.set_figwidth(5)
    plt.savefig(op.join(config.fig_path, 'EMS', 'All_sequences_diff_%s_crop.png' % sens), dpi=300)

    # ALL SEQUENCES IN THE SAME FIGURE (MEAN WITH CI) (just stand)
    times = (epochs_sens.times)*1000
    filter = True
    NUM_COLORS = 7
    cm = plt.get_cmap('viridis')
    colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    plt.axhline(0, linestyle='-', color='black', linewidth=1)
    plt.title('Average EMS signal [standard] - ' + sens)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    for SeqID in range(1, 8):
        color_mean = colorslist[SeqID - 1]
        mean = np.average(all_avg_stand_seq['SeqID_' + str(SeqID)], axis=0)
        ub = mean + sem(all_avg_stand_seq['SeqID_' + str(SeqID)], axis=0)
        lb = mean - sem(all_avg_stand_seq['SeqID_' + str(SeqID)], axis=0)
        if filter == True:
            mean = savgol_filter(mean, 11, 3)
            ub = savgol_filter(ub, 11, 3)
            lb = savgol_filter(lb, 11, 3)
        plt.fill_between(times, ub, lb, color=color_mean, alpha=.2)
        plt.plot(times, mean, color=color_mean, linewidth=1.5, label='SeqID_' + str(SeqID))
    plt.legend(loc='best', fontsize=9)
    ax.set_xlim(-100, 700)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('a.u.')
    plt.savefig(op.join(config.fig_path, 'EMS', 'All_sequences_stand_%s.png' % sens), dpi=300)
    ax.set_xlim(0, 250)
    fig.set_figwidth(5)
    plt.savefig(op.join(config.fig_path, 'EMS', 'All_sequences_stand_%s_crop.png' % sens), dpi=300)

    # ALL SEQUENCES IN THE SAME FIGURE (MEAN WITH CI) (just dev)
    times = (epochs_sens.times)*1000
    filter = True
    NUM_COLORS = 7
    cm = plt.get_cmap('viridis')
    colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    plt.axhline(0, linestyle='-', color='black', linewidth=1)
    plt.title('Average EMS signal [deviant] - ' + sens)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    for SeqID in range(1, 8):
        color_mean = colorslist[SeqID - 1]
        mean = np.average(all_avg_dev_seq['SeqID_' + str(SeqID)], axis=0)
        ub = mean + sem(all_avg_dev_seq['SeqID_' + str(SeqID)], axis=0)
        lb = mean - sem(all_avg_dev_seq['SeqID_' + str(SeqID)], axis=0)
        if filter == True:
            mean = savgol_filter(mean, 11, 3)
            ub = savgol_filter(ub, 11, 3)
            lb = savgol_filter(lb, 11, 3)
        plt.fill_between(times, ub, lb, color=color_mean, alpha=.2)
        plt.plot(times, mean, color=color_mean, linewidth=1.5, label='SeqID_' + str(SeqID))
    plt.legend(loc='best', fontsize=9)
    # ax.set_yticklabels([])
    # ax.set_yticks([])
    ax.set_xlim(-100, 700)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('a.u.')
    plt.savefig(op.join(config.fig_path, 'EMS', 'All_sequences_dev_%s.png' % sens), dpi=300)
    ax.set_xlim(0, 250)
    fig.set_figwidth(5)
    plt.savefig(op.join(config.fig_path, 'EMS', 'All_sequences_dev_%s_crop.png' % sens), dpi=300)
