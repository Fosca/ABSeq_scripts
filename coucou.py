import os.path as op
import config
import numpy as np
import matplotlib.pyplot as plt
from ABseq_func import *
import mne
import os.path as op
config.subjects_list = ['sub01-pa_190002', 'sub02-ch_180036', 'sub05-cr_170417', 'sub06-kc_160388',
                        'sub07-jm_100109', 'sub09-ag_170045', 'sub10-gp_190568', 'sub11-fr_190151', 'sub12-lg_170436',
                        'sub13-lq_180242', 'sub14-js_180232', 'sub15-ev_070110','sub17-mt_170249', 'sub18-eo_190576',
                        'sub19-mg_190180']


# ===== LOAD DATA ===== #
epochs_16_items_mag_test = []; epochs_16_items_grad_test = []; epochs_16_items_eeg_test = []
epochs_16_items_mag_habituation = []; epochs_16_items_grad_habituation = []; epochs_16_items_eeg_habituation = []
epochs_16_items_mag_test_window = []; epochs_16_items_grad_test_window = []; epochs_16_items_eeg_test_window = []
epochs_16_items_mag_habituation_window = []; epochs_16_items_grad_habituation_window = []; epochs_16_items_eeg_habituation_window = []
for subject in config.subjects_list:
    epochs_16_items_mag_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag_SVM_on_16_items_test_window-epo.fif')))
    epochs_16_items_grad_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad_SVM_on_16_items_test_window-epo.fif')))
    epochs_16_items_eeg_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg_SVM_on_16_items_test_window-epo.fif')))
    epochs_16_items_mag_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag_filter_on_16_items_habituation_window-epo.fif')))
    epochs_16_items_grad_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad_filter_on_16_items_habituation_window-epo.fif')))
    epochs_16_items_eeg_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg_filter_on_16_items_habituation_window-epo.fif')))

# ===== FIGURES ===== #
save_folder = op.join(config.fig_path, 'SVM', 'Full_sequence_projection')
utils.create_folder(save_folder)

# Figure with multiple EMS projected (different times)
SVM_filter_times = [x / 1000 for x in range(100, 601, 100)]

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
    win_tmin = epochs_list['test'][0][0].metadata.SVM_filter_tmin_window[0]*1000
    win_tmax = epochs_list['test'][0][0].metadata.SVM_filter_tmax_window[0]*1000
    # for seq_ID in range(1, 8):
    #     # "curve" figure
    #     EMS_funcs.plot_EMS_projection_for_seqID_window(epochs_list, sensor_type=sens, seqID=seq_ID,
    #                                                    save_path=op.join(save_folder, 'Seq%i_%s_window_%i_%ims.png' % (seq_ID, sens, win_tmin, win_tmax)))
    fig = SVM_funcs.plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type=sens, save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims.png' % ( sens, win_tmin, win_tmax)))
