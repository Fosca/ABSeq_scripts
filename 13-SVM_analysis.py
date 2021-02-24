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

load_path = "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/results/SVM/sub02-ch_180036/SW_train_different_blocksGAT_results.npy"

data = np.load(load_path,allow_pickle=True).item()

data_GAT = data['GAT']
times = data['times']

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
results_sepseq, times_sepseq = results_SVM_standard_deviant('SW_train_different_blocks_and_sequencesGAT_results.npy',config.subjects_list)

def plot_results_GAT(results,times,save_folder,compute_significance=None,suffix='SW_train_different_blocks'):

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
                sig = stats_funcs.stats(GAT_all_for_sig-0.5, tail=1)
                sig_all = SVM_funcs.replace_submatrix(sig_all, times_sig, times_sig, sig)

            # -------- plot the gat --------
            pretty_gat(np.mean(res_chan_seq,axis=0),times=times,sig=sig_all<0.05,chance = 0.5)
            plt.gcf().savefig(config.fig_path+save_folder+'/'+chans+'_'+seqID+suffix+'.png')
            plt.gcf().savefig(config.fig_path+save_folder+'/'+chans+'_'+seqID+suffix+'.svg')
            plt.close('all')


plot_results_GAT(results,times,'/SVM/GAT',compute_significance=[0,0.6],suffix='SW_train_different_blocks')



GAT_sens_all, times = plot_all_subjects_results_SVM('SW_train_test_different_blocksGAT_results_score',config.subjects_list,
                                                    'SW_train_test_different_blocksGAT_results_score',plot_per_sequence=True,
                                                    vmin=-0.1,vmax=0.1,analysis_type='perSeq',compute_significance = [0,0.6])

# subjects_list = ['sub02-ch_180036', 'sub05-cr_170417', 'sub06-kc_160388',
#                   'sub09-ag_170045', 'sub10-gp_190568', 'sub11-fr_190151', 'sub12-lg_170436',
#                  'sub13-lq_180242', 'sub14-js_180232', 'sub15-ev_070110', 'sub16-ma_190185', 'sub17-mt_170249', 'sub18-eo_190576']


analysis_name,subjects_list,fig_name,plot_per_sequence=False,plot_individual_subjects=False,score_field='GAT',folder_name = 'GAT'

# __________Linear regression of the GATs as a function of complexity____________________________________________
SVM_funcs.check_missing_GAT_data(config.subjects_list)

coeff_complexity = []
coeff_constant = []
for subject in config.subjects_list:
    print(subject)
    comp, const, times = SVM_funcs.SVM_GAT_linear_reg_sequence_complexity(subject)
    coeff_complexity.append(comp)
    coeff_constant.append(const)

fig_const = ABseq_func.SVM_funcs.plot_GAT_SVM(np.mean(coeff_constant,axis=0), times, sens='all', save_path=config.fig_path+'/SVM/GAT/', figname='regression_const', vmin=-0.1, vmax=0.1)
fig_complexity = ABseq_func.SVM_funcs.plot_GAT_SVM(np.mean(coeff_complexity,axis=0), times, sens='all', save_path=config.fig_path+'/SVM/GAT/', figname='regression_complexity', vmin=-0.1, vmax=0.1)
plt.show()

# ___________________________________________________________________________
# ============== GAT for the different features ===========================
# ___________________________________________________________________________


vmin = [0.45,0.45,0.20,0.45,0.20,0.20]
vmax = [0.55,0.55,0.3,0.55,0.3,0.3]

for residual_analysis in [False,True]:
    if residual_analysis:
        suffix = 'resid_'
    else:
        suffix = 'full_data_'
    chance = [0.5,0.5,0.25,0.5,0.25,0.25]
    for ii,name in enumerate(['ChunkBeg_score_dict','ChunkEnd_score_dict','Number_Open_Chunks_score_dict','RepeatAlter_score_dict','WithinChunkPosition_score_dict']):
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

compute_regression_complexity = True;sensor_type=sens; save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims.png' % ( sens, win_tmin, win_tmax)); vmin=-vminvmax[sens];vmax=vminvmax[sens]

# ==== run regressions as a function of complexity on magnetometer data ====
suf = 'SW_train_test_different_blocks'
constant_hab, complexity_hab = SVM_funcs.compute_regression_complexity('mag' + suf + '_SVM_on_16_items_habituation_window-epo.fif')
constant_test, complexity_test = SVM_funcs.compute_regression_complexity('mag' + suf + '_SVM_on_16_items_test_window-epo.fif')
p_hab = stats_funcs.stats(complexity_hab,tail = 1)
p_test = stats_funcs.stats(complexity_test,tail = 1)


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