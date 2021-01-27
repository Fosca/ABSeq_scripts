import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import os.path as op
import config
import numpy as np
import matplotlib.pyplot as plt
from ABseq_func import *
import mne
import os.path as op
from importlib import reload
from mne.parallel import parallel_func
from scipy.signal import savgol_filter



# -------------

def plot_all_subjects_results_SVM(analysis_name,subjects_list,fig_name,plot_per_sequence=False,plot_individual_subjects=False,score_field='GAT',folder_name = 'GAT',sensors = ['eeg', 'mag', 'grad','all_chans'],vmin=-0.1,vmax=.1):

    GAT_sens_all = {sens: [] for sens in sensors}

    for sens in sensors:
        print("==== running the plotting for sensor type %s"%sens)
        count = 0
        for subject in subjects_list:
            SVM_path = op.join(config.SVM_path, subject)
            GAT_path = op.join(SVM_path,analysis_name+'.npy')
            if op.exists(GAT_path):
                count +=1

                GAT_results = np.load(GAT_path, allow_pickle=True).item()
                print(op.join(SVM_path, analysis_name+'.npy'))
                times = GAT_results['times']
                GAT_results = GAT_results[score_field]
                fig_path = op.join(config.fig_path, 'SVM', folder_name)
                sub_fig_path = op.join(fig_path,subject)
                utils.create_folder(sub_fig_path)
                if plot_per_sequence:
                    if not GAT_sens_all[sens]:  # initialize the keys and empty lists only the first time
                        GAT_sens_all[sens] = {'SeqID_%i' % i: [] for i in range(1, 8)}
                        GAT_sens_all[sens]['average_all_sequences'] = []
                    for key in ['SeqID_%i' % i for i in range(1, 8)]:
                        GAT_sens_all[sens][key].append(GAT_results[sens][key])
                        # ================ Plot & save each subject / each sequence figures ???
                        if plot_individual_subjects:
                                SVM_funcs.plot_GAT_SVM(GAT_results[sens][key], times, sens=sens, save_path=sub_fig_path, figname=fig_name+key); plt.close('all')
                    # ================ Plot & save each subject / average of all sequences figures ???
                    GAT_sens_all[sens]['average_all_sequences'].append(GAT_results[sens]['average_all_sequences'])
                    SVM_funcs.plot_GAT_SVM(GAT_results[sens]['average_all_sequences'], times, sens=sens, save_path=sub_fig_path, figname=fig_name+'_all_seq',vmin=vmin,vmax=vmax)
                    plt.close('all')
                else:
                    GAT_sens_all[sens].append(GAT_results)
                    if plot_individual_subjects:
                        print('plotting for subject:%s'%subject)
                        print(sub_fig_path)
                        print("the shape of the GAT result is ")
                        print(GAT_results.shape)
                        SVM_funcs.plot_GAT_SVM(GAT_results, times, sens=sens, save_path=sub_fig_path,
                                               figname=fig_name,vmin=vmin,vmax=vmax)
                        plt.close('all')
        # return GAT_sens_all

        print("plotting in %s"%config.fig_path)
        if plot_per_sequence:
            for key in ['SeqID_%i' % i for i in range(1, 8)]:
                SVM_funcs.plot_GAT_SVM(np.nanmean(GAT_sens_all[sens][key],axis=0), times, sens=sens, save_path=fig_path,
                                       figname=fig_name+key,vmin=vmin, vmax=vmax)
                plt.close('all')
            SVM_funcs.plot_GAT_SVM(np.nanmean(GAT_sens_all[sens]['average_all_sequences'],axis=0), times, sens=sens,
                                   save_path=fig_path, figname=fig_name + '_all_seq' + '_',
                                   vmin=vmin, vmax=vmax)
            plt.close('all')
        else:
            SVM_funcs.plot_GAT_SVM(-np.mean(GAT_sens_all[sens],axis=0), times, sens=sens, save_path=fig_path, figname=fig_name,vmin=vmin,vmax=vmax)

    print("============ THE AVERAGE GAT WAS COMPUTED OVER %i PARTICIPANTS ========"%count)

    # ===== GROUP AVG FIGURES ===== #
    plt.close('all')
    for sens in ['eeg', 'mag', 'grad', 'all_chans']:
        GAT_avg_sens = GAT_sens_all[sens]
        for seqID in range(1, 8):
            GAT_avg_sens_seq = GAT_avg_sens['SeqID_%i' % seqID]
            GAT_avg_sens_seq_groupavg = np.mean(GAT_avg_sens_seq, axis=0)
            SVM_funcs.plot_GAT_SVM(GAT_avg_sens_seq_groupavg, times, sens=sens,
                                   save_path=op.join(config.fig_path, 'SVM', 'GAT'),
                                   figname=suf + 'GAT_' + str(seqID) + score_suff + '_')
            plt.close('all')
        GAT_avg_sens_allseq_groupavg = np.mean(GAT_avg_sens['average_all_sequences'], axis=0)
        SVM_funcs.plot_GAT_SVM(GAT_avg_sens_allseq_groupavg, times, sens=sens,
                               save_path=op.join(config.fig_path, 'SVM', 'GAT'),
                               figname=suf + 'GAT_all_seq' + score_suff + '_')

    return GAT_sens_all, times
# ___________________________________________________________________________
# ======= plot the GAT for all the sequences apart and together =============
# ___________________________________________________________________________


GAT_sens_all, times = plot_all_subjects_results_SVM('SW_train_test_different_blocksGAT_results_score',config.subjects_list,'SW_train_test_different_blocksGAT_results_score',plot_per_sequence=True,vmin=-0.1,vmax=0.1)

# ___________________________________________________________________________
# ======= plot the GAT for the different features =============
# ___________________________________________________________________________
vmin = [0.45,0.20,0.4]
vmax = [0.55,0.3,0.6]

for ii,name in enumerate(['StimID_score_dict','RepeatAlter_score_dict','WithinChunkPosition_score_dict']):
    anal_name = 'feature_decoding/'+name
    plot_all_subjects_results_SVM(anal_name,config.subjects_list,name,score_field='score',plot_per_sequence=False,plot_individual_subjects=True,sensors = ['all_chans'],vmin=vmin[ii],vmax=vmax[ii])




# ___________________________________________________________________________
# ======= plot the decoder predictions for the 16 item sequences ============
# ___________________________________________________________________________

# ===== LOAD DATA ===== #

suf = 'train_test_different_blocks'

epochs_16_items_mag_test_window = []; epochs_16_items_grad_test_window = []; epochs_16_items_eeg_test_window = [];epochs_16_items_all_chans_test_window = [];
epochs_16_items_mag_habituation_window = []; epochs_16_items_grad_habituation_window = []; epochs_16_items_eeg_habituation_window = [];epochs_16_items_all_chans_habituation_window = [];
for subject in config.subjects_list:
    if op.exists(op.join(config.meg_dir, subject, 'mag_SVM_on_16_items_test_window-epo.fif')):
        epochs_16_items_mag_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag'+suf+'_SVM_on_16_items_test_window-epo.fif')))
        epochs_16_items_grad_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad'+suf+'_SVM_on_16_items_test_window-epo.fif')))
        epochs_16_items_eeg_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg'+suf+'_SVM_on_16_items_test_window-epo.fif')))
        epochs_16_items_all_chans_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'all_chans'+suf+'_SVM_on_16_items_test_window-epo.fif')))
        epochs_16_items_mag_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag'+suf+'_SVM_on_16_items_habituation_window-epo.fif')))
        epochs_16_items_grad_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad'+suf+'_SVM_on_16_items_habituation_window-epo.fif')))
        epochs_16_items_eeg_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg'+suf+'_SVM_on_16_items_habituation_window-epo.fif')))
        epochs_16_items_all_chans_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'all_chans'+suf+'_SVM_on_16_items_habituation_window-epo.fif')))


# ===== FIGURES ===== #
save_folder = op.join(config.fig_path, 'SVM', 'Full_sequence_projection')
utils.create_folder(save_folder)

# Figure with only one EMS projected (average window)
epochs_list = {}
for sens in ['all_chans','mag', 'grad', 'eeg']:
    if sens == 'mag':
        epochs_list['hab'] = epochs_16_items_mag_habituation_window
        epochs_list['test'] = epochs_16_items_mag_test_window
    elif sens == 'grad':
        epochs_list['hab'] = epochs_16_items_grad_habituation_window
        epochs_list['test'] = epochs_16_items_grad_test_window
    elif sens == 'eeg':
        epochs_list['hab'] = epochs_16_items_eeg_habituation_window
        epochs_list['test'] = epochs_16_items_eeg_test_window
    elif sens == 'all_chans':
        epochs_list['hab'] = epochs_16_items_all_chans_habituation_window
        epochs_list['test'] = epochs_16_items_all_chans_test_window

    win_tmin = epochs_list['test'][0][0].metadata.SVM_filter_tmin_window[0]*1000
    win_tmax = epochs_list['test'][0][0].metadata.SVM_filter_tmax_window[0]*1000

    SVM_funcs.plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type=sens, save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims.png' % ( sens, win_tmin, win_tmax)),vmin=-1.0,vmax=1.0)

# ___________________________________________________________________________
# ======= plot the GAT diagonal for each of the 7 sequences in the nice viridis colors ============
# ___________________________________________________________________________

GAT_sens_all, times = plot_all_subjects_results_SVM('SW_train_test_different_blocksGAT_results_score',
                                             config.subjects_list,
                                             'SW_train_test_different_blocksGAT_results_score',
                                             plot_per_sequence=True, vmin=-0.1, vmax=0.1)

times = times*1000
# ALL SEQUENCES IN THE SAME FIGURE (MEAN WITH CI) (just Diff)
sensors = ['all_chans','eeg','mag','grad']
for sens in sensors:
    filter = True
    NUM_COLORS = 7
    cm = plt.get_cmap('viridis')
    colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    plt.axhline(0, linestyle='-', color='black', linewidth=1)
    perform_seq = GAT_sens_all[sens]

    plt.title('Decoder performance - ' + sens)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    for SeqID in range(1, 8):
        color_mean = colorslist[SeqID - 1]
        mean = np.diagonal(-np.nanmean(perform_seq['SeqID_' + str(SeqID)], axis=0))
        ub = np.diagonal(mean + np.nanstd(perform_seq['SeqID_' + str(SeqID)], axis=0)/(np.sqrt(15)))
        lb = np.diagonal(mean - np.nanstd(perform_seq['SeqID_' + str(SeqID)], axis=0)/(np.sqrt(15)))
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
    plt.savefig(op.join(config.fig_path, 'SVM', 'All_sequences_diff_%s.png' % sens), dpi=300)
    ax.set_xlim(0, 250)
    fig.set_figwidth(5)
    plt.savefig(op.join(config.fig_path, 'SVM', 'All_sequences_diff_%s_crop.png' % sens), dpi=300)