import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
import os.path as op
import config
import numpy as np
import matplotlib.pyplot as plt
import ABseq_func.SVM_funcs
from ABseq_func import *
from ABseq_func import SVM_funcs
import mne
import os.path as op
from importlib import reload
from mne.parallel import parallel_func
import pandas as pd
from scipy.signal import savgol_filter
from jr.plot import pretty_gat

# ________________The list of participants (4,8 previously excluded and 16 has a problem)_______________________________

config.subjects_list = ['sub01-pa_190002',
 'sub02-ch_180036',
 'sub03-mr_190273',
 'sub05-cr_170417',
 'sub06-kc_160388',
 'sub07-jm_100109',
 'sub09-ag_170045',
 'sub10-gp_190568',
 'sub11-fr_190151',
 'sub12-lg_170436',
 'sub13-lq_180242',
 'sub14-js_180232',
 'sub15-ev_070110',
 'sub17-mt_170249',
 'sub18-eo_190576',
 'sub19-mg_190180']

# ___________________________________________________________________________
# ============== GAT decoding Standard // Deviant ===========================
# ___________________________________________________________________________

def results_SVM_standard_deviant(fname,subjects_list):

    results = {sens: [] for sens in ['eeg', 'mag', 'grad', 'all_chans']}

    for sens in ['eeg', 'mag', 'grad', 'all_chans']:
        results[sens] = {'SeqID_%i' % i: [] for i in range(1, 8)}
        for subject in subjects_list:

            load_path = config.result_path+'/SVM/'+subject+'/'+fname
            data = np.load(load_path, allow_pickle=True).item()
            # Load the results
            data_GAT_sens = data['GAT'][sens]
            times = data['times']

            for seqID in range(1,8):
                results[sens]["SeqID_"+str(seqID)].append(data_GAT_sens["SeqID_"+str(seqID)])

    return results, times

results, times = results_SVM_standard_deviant('SW_train_different_blocksGAT_results.npy',config.subjects_list)
results_sepseq, times_sepseq = results_SVM_standard_deviant('SW_train_different_blocks_and_sequencesGAT_results.npy',config.subjects_list,clim=None,tail=1)


def plot_GAT(results,times,save_folder,compute_significance=None,suffix='SW_train_different_blocks',chance = 0.5,clim=None,tail=1):
    if compute_significance is not None:
        tmin_sig = compute_significance[0]
        tmax_sig = compute_significance[1]
        times_sig = np.where(np.logical_and(times <= tmax_sig, times > tmin_sig))[0]
        sig_all = np.ones(results[0].shape)
        GAT_all_for_sig = results[:, times_sig, :]
        GAT_all_for_sig = GAT_all_for_sig[:, :, times_sig]
        sig = stats_funcs.stats(GAT_all_for_sig-chance, tail=tail)
        sig_all = SVM_funcs.replace_submatrix(sig_all, times_sig, times_sig, sig)

        # -------- plot the gat --------
    pretty_gat(np.mean(results,axis=0),times=times,sig=sig_all<0.05,chance = chance,clim=clim)
    plt.gcf().savefig(config.fig_path+save_folder+'/'+suffix+'.png')
    plt.gcf().savefig(config.fig_path+save_folder+'/'+suffix+'.svg')
    plt.close('all')



def plot_results_GAT_chans_seqID(results,times,save_folder,compute_significance=None,suffix='SW_train_different_blocks',chance = 0.5):
=======
results_sepseq, times_sepseq = results_SVM_standard_deviant('SW_train_different_blocks_and_sequencesGAT_results.npy',config.subjects_list)


results, times = results_SVM_standard_deviant('SW_train_different_blocksGAT_results.npy',config.subjects_list)

def plot_results_GAT(results,times,save_folder,compute_significance=None,suffix='SW_train_different_blocks'):
>>>>>>> 0d7e392ce569ff1de9b58e10ab7f498f2417169a

    for chans in results.keys():
        res_chan = results[chans]
        for seqID in res_chan.keys():
            res_chan_seq = np.asarray(res_chan[seqID])
            sig_all = None
            # ---- compute significance ----
            if compute_significance is not None:
                tmin_sig = compute_significance[0]
                tmax_sig = compute_significance[1]
                times_sig = np.where(np.logical_and(times <= tmax_sig, times > tmin_sig))[0]
                sig_all = np.ones(res_chan_seq[0].shape)
                GAT_all_for_sig = res_chan_seq[:, times_sig, :]
                GAT_all_for_sig = GAT_all_for_sig[:, :, times_sig]
                sig = stats_funcs.stats(GAT_all_for_sig-chance, tail=1)
                sig_all = SVM_funcs.replace_submatrix(sig_all, times_sig, times_sig, sig)

            # -------- plot the gat --------
            pretty_gat(np.mean(res_chan_seq,axis=0),times=times,sig=sig_all<0.05,chance = 0.5)
            plt.gcf().savefig(config.fig_path+save_folder+'/'+chans+'_'+seqID+suffix+'.png')
            plt.gcf().savefig(config.fig_path+save_folder+'/'+chans+'_'+seqID+suffix+'.svg')
            plt.close('all')


plot_results_GAT_chans_seqID(results,times,'/SVM/GAT',compute_significance=[0,0.6],suffix='SW_train_different_blocks')
plot_results_GAT_chans_seqID(results_sepseq,times_sepseq,'/SVM/GAT',compute_significance=[0,0.6],suffix='SW_train_different_blocks_different_seq')



GAT_sens_all, times = plot_all_subjects_results_SVM('SW_train_test_different_blocksGAT_results_score',config.subjects_list,
                                                    'SW_train_test_different_blocksGAT_results_score',plot_per_sequence=True,
                                                    vmin=-0.1,vmax=0.1,analysis_type='perSeq',compute_significance = [0,0.6])

# subjects_list = ['sub02-ch_180036', 'sub05-cr_170417', 'sub06-kc_160388',
#                   'sub09-ag_170045', 'sub10-gp_190568', 'sub11-fr_190151', 'sub12-lg_170436',
#                  'sub13-lq_180242', 'sub14-js_180232', 'sub15-ev_070110', 'sub16-ma_190185', 'sub17-mt_170249', 'sub18-eo_190576']


analysis_name,subjects_list,fig_name,plot_per_sequence=False,plot_individual_subjects=False,score_field='GAT',folder_name = 'GAT'

# __________Linear regression of the GATs as a function of complexity____________________________________________
SVM_funcs.check_missing_GAT_data(config.subjects_list)

coeff_complexity_sepseq = []
coeff_complexity = []
coeff_constant = []
coeff_constant_sepseq = []
for subject in config.subjects_list:
    print(subject)
    comp, const, times = SVM_funcs.SVM_GAT_linear_reg_sequence_complexity(subject,suffix='SW_train_different_blocksGAT_results.npy')
    comp_sepseq, const_sepseq, times = SVM_funcs.SVM_GAT_linear_reg_sequence_complexity(subject,suffix='SW_train_different_blocks_and_sequencesGAT_results.npy')
    coeff_complexity.append(comp)
    coeff_complexity_sepseq.append(comp_sepseq)
    coeff_constant.append(const)
    coeff_constant_sepseq.append(const_sepseq)


plot_GAT(np.asarray(coeff_complexity),times,save_folder='/SVM/GAT/',suffix = 'regression_complexity_SW_train_different_blocks',compute_significance=[0,0.6],chance = 0,tail=-1,clim=[-0.005,0.005])
plot_GAT(np.asarray(coeff_constant),times,save_folder='/SVM/GAT/',suffix = 'regression_const_SW_train_different_blocks',compute_significance=[0,0.6],chance = 0,tail=-1)
plot_GAT(np.asarray(coeff_complexity_sepseq),times,save_folder='/SVM/GAT/',suffix = 'regression_complexity_SW_train_different_blocks_and_sequences',compute_significance=[0,0.6],chance = 0,tail=-1,clim=[-0.005,0.005])
plot_GAT(np.asarray(coeff_constant_sepseq),times,save_folder='/SVM/GAT/',suffix = 'regression_const_SW_train_different_blocks_and_sequences',compute_significance=[0,0.6],chance=0,tail=-1)


# ___________________________________________________________________________
# ============== GAT for the different features ===========================
# ___________________________________________________________________________

anal_name = 'feature_decoding/' + "full_data_" + "ordinal_code_quads_tested_others"
SVM_funcs.plot_gat_simple(anal_name, config.subjects_list, "full_data_" + "ordinal_code_quads_tested_others.npy", chance=0.25, score_field='score',
                          vmin=None, vmax=None, compute_significance=[0., 0.6])


vmin = [0.45,0.45,0.20,0.45,0.20,0.20]
vmax = [0.55,0.55,0.3,0.55,0.3,0.3]

for residual_analysis in [True]:
    if residual_analysis:
        suffix = 'resid_cv_'
    else:
        suffix = 'full_data_'
    chance = [0.5,0.5,0.25,0.5,0.25,0.25]

    for ii,name in enumerate(['RepeatAlter_score_dict','ChunkEnd_score_dict','Number_Open_Chunks_score_dict','ChunkBeg_score_dict','WithinChunkPosition_score_dict']):
        anal_name = 'feature_decoding/'+suffix+name
        SVM_funcs.plot_gat_simple(anal_name,config.subjects_list,suffix+name,chance = chance[ii],score_field='score',vmin=None,vmax=None,compute_significance=[0.,0.6])


# ___________________________________________________________________________
# ======= plot the decoder predictions for the 16 item sequences ============
# ___________________________________________________________________________

# ===== LOAD DATA ===== #

suf = 'SW_train_test_different_blocks'
# suf = 'train_test_different_blocks'

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
vminvmax = {'all_chans':2,'mag':1,'grad':1,'eeg':1.5}

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

    SVM_funcs.plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_list, compute_reg_complexity = True,window_CBPT_violation = 0.7,sensor_type=sens, save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims.png' % ( sens, win_tmin, win_tmax)),vmin=-vminvmax[sens],vmax=vminvmax[sens])


# ___________________________________________________________________________
# ======= plot the GAT diagonal for each of the 7 sequences in the nice viridis colors ============
# ___________________________________________________________________________

GAT_sens_all, times = SVM_funcs.plot_all_subjects_results_SVM('SW_train_test_different_blocksGAT_results_score',
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

# ___________________________________________________________________________________________________________
# ======= plot the Ordinal code when training on quads and testing on the 16 items of the others ============
# ___________________________________________________________________________________________________________

fname = 'ordinal_code_quads_tested_others.npy'
# ---- load ----
avg_subj = {'%i'%i:[] for i in [1,2,3,5,6,7]}
for subject in config.subjects_list:
    data_path = config.result_path + '/SVM/ordinal_code_16items/' + subject + '/' + fname
    print("loading data from %s"%subject)
    ord_16 = np.load(data_path, allow_pickle=True).item()
    for seq in [1,2,3,5,6,7]:
        avg_subj['%i'%seq].append(np.mean(np.mean(ord_16['SeqID_%i'%seq]['projection'],axis=0),axis=0))
    times = ord_16['SeqID_%i'%seq]['times']

#----- plot -----
predictions = np.asarray(avg_subj['5'])


def plot_ordinal_code_sequences(data_ord16):

    labelsize = 6
    fontsize = 6
    linewidth = 0.7
    linewidth_zero = 1
    linewidth_other = 0.5
    n_cat = 4

    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True, sharey=False, constrained_layout=True)
    fig.suptitle("Projection on the decision axis", fontsize=12)
    ax = axes.ravel()[::1]
    ax[0].set_title('Repeat', loc='left', weight='bold')
    ax[1].set_title('Alternate', loc='left', weight='bold')
    ax[2].set_title('Pairs', loc='left', weight='bold')
    ax[3].set_title('Pairs+Alt', loc='left', weight='bold')
    ax[4].set_title('Shrinking', loc='left', weight='bold')
    ax[5].set_title('Complex', loc='left', weight='bold')

    for n, seq in enumerate([1,2,3,5,6,7]):

        predictions = np.asarray(avg_subj['%i'%seq])
        mean_plot = np.mean(predictions, axis=0)
        sem_plot = np.std(predictions, axis=0) / np.sqrt(predictions.shape[0])

        # ============== And now, let's plot ============================
        for k in range(n_cat):
            ax[n].plot(times, mean_plot[:, k], linewidth=linewidth)
            # ax[n] = plt.gca()
            # ax.set_ylim(ylim)
            ax[n].fill_between(times, mean_plot[:, k] - sem_plot[:, k], mean_plot[:, k] + sem_plot[:, k], alpha=0.6)

        ax[n].axvline(0, 0, 200, color='k', linewidth=linewidth_zero)
        ax[n].set_xticks([np.round(0.250 * xx, 2) for xx in range(17)])
        for ti in [0.250 * xx for xx in range(16)]:
            ax[n].axvline(ti, 0, 200, color='k', linewidth=linewidth_other)
        ax[n].xaxis.set_ticks_position('bottom')
        ax[n].tick_params(axis='both', which='major', labelsize=labelsize)
        # ax[n].set_ylabel('Probability ordinal position', fontsize=fontsize)

    ax[5].set_xlabel('Testing Time (s)', fontsize=fontsize)

    plt.show()