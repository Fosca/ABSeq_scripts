"""
==========================================================
Testing the Standard vs Deviant decoder on the full 16 item sequences
==========================================================
# DESCRIPTION OF THE ANALYSIS
# 1 - Which trials : no cleaning or with autoreject Global ?
# 2 - Sliding window size 100 ms every XX ms ?
# 3 - Excluded participants (with no cleaning)?
# 4 - We average the predictions (distance ?) for the decoders trained on the window 140 - 180 ms.
# 5 - We test them on the trials complementary to the training trials (in the test set) and on the
# habituation trials.
"""

# ---- import the packages -------
import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
from ABseq_func import *
import matplotlib.pyplot as plt
import config
import numpy as np
import os.path as op
import mne
import matplotlib

import matplotlib.ticker as ticker
from jr.plot import pretty_gat

# ---------- plot the gats for all the sequences ------------
# compute the average and find a time window of maximal decoding in order to apply it on the 16 item sequences

results, times = results_SVM_standard_deviant('SW_train_different_blocks_cleanedGAT_results.npy',config.subjects_list)
plot_results_GAT_chans_seqID(results,times,'/SVM/standard_vs_deviant/GAT/',compute_significance=[0,0.6],suffix='_cleaned_SW',clim=[0.37,0.63])

# -------------- plot the average output of the projection -------

suf = {'mag':'SW_train_different_blocks_cleaned_131_210ms','grad':'SW_train_different_blocks_cleaned_131_210ms_131_210ms'}
# suf = {'mag':'SW_train_different_blocks_cleaned_210_410ms','grad':'SW_train_different_blocks_cleaned_210_410ms_210_410ms'}
sensors = ['mag','grad']
# sensors = ['mag', 'grad']
epochs_16 = {sens : {'hab':[],'test':[]} for sens in sensors}

for subject in config.subjects_list:
    for sens in sensors:
        epochs_16[sens]['test'].append(mne.read_epochs(op.join(config.meg_dir, subject, sens+suf[sens]+'_SVM_on_16_items_test_window-epo.fif')))
        epochs_16[sens]['hab'].append(mne.read_epochs(op.join(config.meg_dir, subject, sens+suf[sens]+'_SVM_on_16_items_habituation_window-epo.fif')))

for subject in config.subjects_list:
    print(mne.read_epochs(op.join(config.meg_dir, subject, sens+suf[sens]+'_SVM_on_16_items_habituation_window-epo.fif')))

# ===== FIGURES ===== #
save_folder = op.join(config.fig_path, 'SVM', 'Full_sequence_projection')
utils.create_folder(save_folder)

# Figure with only one EMS projected (average window)
epochs_list = {}
vminvmax = {'all_chans':2,'mag':1,'grad':1}
for sens in sensors:
    win_tmin = epochs_16[sens]['test'][0][0].metadata.SVM_filter_tmin_window[0]*1000
    win_tmax = epochs_16[sens]['test'][0][0].metadata.SVM_filter_tmax_window[0]*1000

    plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_16[sens], compute_reg_complexity = False,
                                                                  window_CBPT_violation = 0.7,sensor_type=sens,
                                                                  save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims_tvals.svg' % ( sens, win_tmin, win_tmax)),
                                                                  vmin=-vminvmax[sens],vmax=vminvmax[sens],plot_betas=False)


# ______________________________________________________________________________________________________________________
#  PLOT SEPARATELY FOR THE 7 SEQUENCES THE HABITUATION AND THE STANDARDS IN 3 DIFFERENT MANNERS : FULL SEQUENCE (16 ITEMS)
#  COLLAPSED ON 1 EPOCH ITEM, OR COLLAPSED OVER ALL THE FULL SEQUENCE PRESENTATION BUT SEPARATED BY SEQUENCE PRESENTATION
# (RELEVANT FOR HABITUATION ONLY)
# ______________________________________________________________________________________________________________________

for anal_type in ['full_sequence','1item','collapse']:
    plot_16_item_projection(epochs_16,analysis_type = anal_type)


# ______________________________________________________________________________________________________________________
#  SVM PLOTTING FUNCTIONS - SVM PLOTTING FUNCTIONS - SVM PLOTTING FUNCTIONS - SVM PLOTTING FUNCTIONS - SVM PLOTTING FUNCTIONS
# ______________________________________________________________________________________________________________________
def results_SVM_standard_deviant(fname,subjects_list):
    """
    Function to load the results from the decoding of standard VS deviant
    """

    results = {sens: [] for sens in config.ch_types}
    times = []
    for sens in config.ch_types:
        results[sens] = {'SeqID_%i' % i: [] for i in range(1, 8)}
        results[sens]["average_all_sequences"] = []
        for subject in subjects_list:
            print("running the loop for subject %s \n"%subject)
            load_path = config.result_path+'/SVM/'+subject+'/'+fname
            data = np.load(load_path, allow_pickle=True).item()
            # Load the results
            data_GAT_sens = data['GAT'][sens]
            times = data['times']
            for seqID in range(1,8):
                results[sens]["SeqID_"+str(seqID)].append(data_GAT_sens["SeqID_"+str(seqID)])
            results[sens]["average_all_sequences"].append(data_GAT_sens["average_all_sequences"])

    return results, times

# ______________________________________________________________________________________________________________________
def plot_results_GAT_chans_seqID(results,times,save_folder,compute_significance=None,suffix='SW_train_different_blocks',chance = 0.5,clim=None):

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
            ax = pretty_gat(np.nanmean(res_chan_seq,axis=0),times=times,sig=sig_all<0.05,chance = 0.5,clim=clim)
            plt.gcf().savefig(config.fig_path+save_folder+'/'+chans+'_'+seqID+suffix+'.png')
            plt.gcf().savefig(config.fig_path+save_folder+'/'+chans+'_'+seqID+suffix+'.svg')
            plt.close('all')

# ______________________________________________________________________________________________________________________
def plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type, save_path=None, vmin=-1, vmax=1,compute_reg_complexity = False, window_CBPT_violation = None,plot_betas=True):

    color_viol = ['lightgreen','mediumseagreen','mediumslateblue','darkviolet']


    import matplotlib.colors as mcolors

    colors = [(0, 0, 0, c) for c in np.linspace(0, 1, 2)]
    cmapsig = mcolors.LinearSegmentedColormap.from_list('significance_cmpa', colors, N=5)

    # window info, just for figure title
    win_tmin = epochs_list['test'][0][0].metadata.SVM_filter_tmin_window[0] * 1000
    win_tmax = epochs_list['test'][0][0].metadata.SVM_filter_tmax_window[0] * 1000
    n_plots = 7
    if compute_reg_complexity:
        n_plots = 8
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 12), sharex=True, sharey=False, constrained_layout=True)
    fig.suptitle('SVM %s - window %d-%dms; N subjects = %d' % (
        sensor_type, win_tmin, win_tmax, len(epochs_list['test'])), fontsize=12)
    ax = axes.ravel()[::1]
    ax[0].set_title('Repeat', loc='left', weight='bold')
    ax[1].set_title('Alternate', loc='left', weight='bold')
    ax[2].set_title('Pairs', loc='left', weight='bold')
    ax[3].set_title('Quadruplets', loc='left', weight='bold')
    ax[4].set_title('Pairs+Alt', loc='left', weight='bold')
    ax[5].set_title('Shrinking', loc='left', weight='bold')
    ax[6].set_title('Complex', loc='left', weight='bold')

    seqtxtXY = ['xxxxxxxxxxxxxxxx',
                'xYxYxYxYxYxYxYxY',
                'xxYYxxYYxxYYxxYY',
                'xxxxYYYYxxxxYYYY',
                'xxYYxYxYxxYYxYxY',
                'xxxxYYYYxxYYxYxY',
                'xYxxxYYYYxYYxxxY']

    if compute_reg_complexity:
        if plot_betas:
            ax[7].set_title('Beta_complexity', loc='left', weight='bold')
        else:
            ax[7].set_title('t-values-betas', loc='left', weight='bold')
        seqtxtXY.append('')

    print("vmin = %0.02f, vmax = %0.02f" % (vmin, vmax))

    n = 0

    violation_significance = {i:[] for i in range(1, 8)}
    epochs_data_hab_allseq = []
    epochs_data_test_allseq = []

    for seqID in range(1, 8):
        print("=== running for sequence %i ==="%seqID)
        #  this provides us with the position of the violations and the times
        epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
        times = epochs_seq_subset.times
        times = times + 0.3
        violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])
        violation_significance[seqID] = {'times':times,'window_significance':window_CBPT_violation}

        #  ----------- habituation trials -----------
        epochs_data_hab_seq = []
        y_list_epochs_hab = []
        data_nanmean = []
        where_sig = []
        for epochs in epochs_list['hab']:
            epochs_subset = epochs['SequenceID == "' + str(seqID) + '"']
            avg_epo = np.nanmean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0)
            y_list_epochs_hab.append(avg_epo)
            epochs_data_hab_seq.append(avg_epo)
        epochs_data_hab_allseq.append(epochs_data_hab_seq)
        nanmean_hab = np.nanmean(y_list_epochs_hab, axis=0)
        data_nanmean.append(nanmean_hab)
        where_sig.append(np.zeros(nanmean_hab.shape))
        where_sig.append(np.zeros(nanmean_hab.shape))

        #  ----------- test trials -----------
        epochs_data_test_seq = []

        for viol_pos in violpos_list:
            y_list = []
            contrast_viol_pos = []
            for epochs in epochs_list['test']:
                epochs_subset = epochs[
                    'SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                avg_epo = np.nanmean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0)
                y_list.append(avg_epo)
                if viol_pos==0:
                    avg_epo_standard = np.nanmean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0)
                    epochs_data_test_seq.append(avg_epo_standard)
                if viol_pos !=0 and window_CBPT_violation is not None:
                    epochs_standard = epochs[
                        'SequenceID == "' + str(seqID) + '" and ViolationInSequence == 0']
                    avg_epo_standard = np.nanmean(np.squeeze(epochs_standard.savgol_filter(20).get_data()), axis=0)
                    contrast_viol_pos.append(avg_epo - avg_epo_standard)

            # --------------- CBPT to test for significance ---------------
            if window_CBPT_violation is not None and viol_pos !=0:
                time_start_viol = 0.250 * (viol_pos - 1)
                time_stop_viol = time_start_viol + window_CBPT_violation
                inds_stats = np.where(np.logical_and(times>time_start_viol,times<=time_stop_viol))
                contrast_viol_pos = np.asarray(contrast_viol_pos)
                p_vals = np.asarray([1]*contrast_viol_pos.shape[1])
                p_values = stats_funcs.stats(contrast_viol_pos[:,inds_stats[0]],tail=1)
                p_vals[inds_stats[0]] = p_values
                violation_significance[seqID][int(viol_pos)] = p_vals
                y_list_alpha = 1*(p_vals<0.05)
                where_sig.append(y_list_alpha)

            nanmean_y = np.nanmean(y_list, axis=0)
            data_nanmean.append(nanmean_y)
        epochs_data_test_allseq.append(epochs_data_test_seq)
        where_sig = np.asarray(where_sig)

        width = 75
        # Add vertical lines, and "xY"
        for xx in range(16):
            ax[n].axvline(250 * xx,ymin=0,ymax= width, linestyle='--', color='black', linewidth=0.8)
            txt = seqtxtXY[n][xx]
            ax[n].text(250 * (xx + 1) - 125, width * 6 + (width / 3), txt, horizontalalignment='center', fontsize=16)

        # return data_nanmean
        ax[n].spines["top"].set_visible(False)
        ax[n].spines["right"].set_visible(False)
        ax[n].spines["bottom"].set_visible(False)
        ax[n].spines["left"].set_visible(False)

        im = ax[n].imshow(data_nanmean, extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap='RdBu_r',
                          vmin=vmin, vmax=vmax)
        # add colorbar
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.50, aspect=10, pad=.005)
        cb.ax.yaxis.set_offset_position('left')
        cb.set_label('a. u.')
        # if window_CBPT_violation:
        #     masked = np.ma.masked_where(where_sig == 0, where_sig)
        #     im = ax[n].imshow(masked, extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap=cmapsig,
        #                       vmin=vmin, vmax=vmax,alpha=0.7)
        ax[n].set_yticks(np.arange(width / 2, 6 * width, width))
        ax[n].set_yticklabels(['Violation (pos. %d)' % violpos_list[4], 'Violation (pos. %d)' % violpos_list[3],
                               'Violation (pos. %d)' % violpos_list[2], 'Violation (pos. %d)' % violpos_list[1],
                               'Standard', 'Habituation'])
        ax[n].axvline(0, linestyle='-', color='black', linewidth=2)

        # add deviant marks
        for k in range(4):
            viol_pos = violpos_list[k + 1]
            x = 250 * (viol_pos - 1)
            y1 = (4 - k) * width
            y2 = (4 - 1 - k) * width
            ax[n].plot([x, x], [y1, y2], linestyle='-', color='black', linewidth=6)
            ax[n].plot([x, x], [y1, y2], linestyle='-', color=color_viol[k], linewidth=3)

            find_where_sig = np.where(where_sig[k+2,:]==1)[0]
            if len(find_where_sig)!=0:
                ax[n].plot([1000 * times[find_where_sig[0]], 1000 * times[find_where_sig[-1]]], [-(k+1)*width/3, -(k+1)*width/3], linestyle='-', color=color_viol[k], linewidth=3)
        n += 1

    if compute_reg_complexity:
        epochs_data_hab_allseq = np.asarray(epochs_data_hab_allseq)
        epochs_data_test_allseq = np.asarray(epochs_data_test_allseq)
        coeff_const_hab, coeff_complexity_hab, t_const_hab, t_complexity_hab = SVM_funcs.compute_regression_complexity(epochs_data_hab_allseq)
        coeff_const_test, coeff_complexity_test, t_const_test, t_complexity_test = SVM_funcs.compute_regression_complexity(epochs_data_test_allseq)

        for xx in range(16):
            ax[7].axvline(250 * xx, linestyle='--', color='black', linewidth=1)

        # return data_nanmean
        if plot_betas:
            im = ax[7].imshow(np.asarray([np.nanmean(coeff_complexity_hab,axis=0),np.nanmean(coeff_complexity_test,axis=0)]), extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap='RdBu_r',
                              vmin=-0.5, vmax=0.5)
        else:
            im = ax[7].imshow(np.asarray([t_complexity_hab,t_complexity_test]), extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap='RdBu_r',
                              vmin=-6, vmax=6)

        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.50, aspect=10, pad=.005)
        cb.ax.yaxis.set_offset_position('left')
        width = width*3
        ax[7].set_yticks(np.arange(width / 2, 2 * width, width))
        ax[7].set_yticklabels(['Standard', 'Habituation'])
        ax[7].axvline(0, linestyle='-', color='black', linewidth=2)

    axes.ravel()[-1].set_xlabel('Time (ms)')

    figure = plt.gcf()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close('all')

    return figure

# ______________________________________________________________________________________________________________________
def reshaped_the_data(epochs_list, sens,analysis_type = 'full_sequence'):
    """
    Function that plots on the left the habituation trials and on the right the standard trials grouped by sequence.
    param epochs_list: Contains a dictionnary with the epochs organized by 1 - sensor type 2 - 'hab' or 'test'
    param sensor_type :
    param analysis_type : 'full_sequence' if we want to keep the whole 16 items. '1item' if we want to average across all
    the 16 items over 1 item window, 'collapse' averages throughout the 16 items but separates each of the 10 sequence presentations
    during the habituation
    param save_path : where we save the picture
    param vmin, vmax : the ylimits
    param compute_reg_complexity : set it to true if you want to compute the correlation with complexity and append it to the figure
    """
    # epochs_list = epochs_16
    # sens = 'mag'
    # ============ access the habituation and the standard data =========

    print("============================================================================================================")
    print("     RESHAPING THE DATA FOR THE ANALYSIS %s      "%analysis_type)
    print("============================================================================================================")

    pseudoepo_habituation = epochs_list[sens]['hab']
    pseudoepo_test = epochs_list[sens]['test']
    times = pseudoepo_habituation[0].times
    times = times + 0.3

    n_subjects = len(pseudoepo_habituation)
    n_times = len(times)
    n_hab = 10
    data_hab_seq = {'SeqID_%i' % i: np.ones((n_hab,n_subjects,n_times)) for i in range(1, 8)}
    data_stand_seq = {'SeqID_%i' % i: np.ones((n_subjects,n_times)) for i in range(1, 8)}
    for seq in range(1, 8):
        for trial_number in range(1,n_hab+1):
            for subj_n in range(n_subjects):
                data_hab_seq['SeqID_%i' % seq][trial_number-1,subj_n,:]  = np.squeeze(np.nanmean(pseudoepo_habituation[subj_n]["SequenceID == %i and TrialNumber == %i"%(seq,trial_number)].get_data(),axis=0))
    for seq in range(1, 8):
        for subj_n in range(n_subjects):
            data_stand_seq['SeqID_%i' % seq][subj_n, :] = np.squeeze(np.nanmean(
                pseudoepo_test[subj_n]["SequenceID == %i and ViolationInSequence == 0" % seq].get_data(), axis=0))

    # ============ put the data in the right shape according to the analysis type ===========
    data_hab_seq_reshaped = {'SeqID_%i' % i: [] for i in range(1, 8)}
    data_stand_seq_reshaped = {'SeqID_%i' % i: [] for i in range(1, 8)}
    if analysis_type == '1item' or analysis_type == 'collapse':
        for seq in range(1, 8):
            for k in range(16):
                inds_to_append_for_epochs = np.logical_and(times >=0.250*k,times <0.25*(k+1)+0.001)
                data_stand_to_append = data_stand_seq['SeqID_%i' % seq][:,inds_to_append_for_epochs]
                data_hab_to_append = data_hab_seq['SeqID_%i' % seq][:,:,inds_to_append_for_epochs]
                data_stand_seq_reshaped['SeqID_%i' % seq].append(data_stand_to_append)
                data_hab_seq_reshaped['SeqID_%i' % seq].append(data_hab_to_append)
            data_stand_seq_reshaped['SeqID_%i' % seq] = np.asarray(data_stand_seq_reshaped['SeqID_%i' % seq])
            data_hab_seq_reshaped['SeqID_%i' % seq] = np.asarray(data_hab_seq_reshaped['SeqID_%i' % seq])
            data_stand_seq_reshaped['SeqID_%i' % seq] = np.nanmean(data_stand_seq_reshaped['SeqID_%i' % seq],axis=0)
            data_hab_seq_reshaped['SeqID_%i' % seq] = np.nanmean(np.asarray(data_hab_seq_reshaped['SeqID_%i' % seq]),axis=0)

            if analysis_type == 'collapse':
                data_stand_seq_reshaped['SeqID_%i' % seq] = np.nanmean(data_stand_seq_reshaped['SeqID_%i' % seq], axis=-1)
                data_hab_seq_reshaped['SeqID_%i' % seq] = np.transpose(np.nanmean(np.asarray(data_hab_seq_reshaped['SeqID_%i' % seq]), axis=-1))
            else:
                data_hab_seq_reshaped['SeqID_%i' % seq] = np.nanmean(np.asarray(data_hab_seq_reshaped['SeqID_%i' % seq]), axis=-0)
    else:
        # This is the case full sequence
        data_hab_seq_reshaped = {'SeqID_%i' % i: [] for i in range(1, 8)}
        for seq in data_hab_seq_reshaped.keys():
            data_hab_seq_reshaped[seq] = np.nanmean(data_hab_seq[seq],axis=0)
        data_stand_seq_reshaped = data_stand_seq

    if analysis_type == 'collapse':
        times = np.asarray(range(1, 11)) * 0.001
    elif analysis_type == '1item':
        times = times[np.logical_and(times >= 0, times < 0.251)]

    return data_hab_seq_reshaped, data_stand_seq_reshaped, times

def compute_correlation_complexity(data):
    from scipy import stats
    C = [4, 6, 6, 6, 12, 15, 28]
    n_subjects, n_times = data['SeqID_1'].shape
    data_corr = np.asarray([data[key] for key in data.keys()])
    pearson_r = []
    spearman_rho = []
    for nn in range(n_subjects):
        # ---- for 1 subject, diagonal of the GAT for all the 7 sequences through time ---
        dd = data_corr[:,nn,:]
        r = []
        rho = []
        for t in range(n_times):
            r_t, _ = stats.pearsonr(dd[:, t], C)
            rho_t, _ = stats.spearmanr(dd[:, t], C)
            r.append(r_t)
            rho.append(rho_t)
        pearson_r.append(r)
        spearman_rho.append(rho)
    pearson_r = np.asarray(pearson_r)
    spearman_rho = np.asarray(spearman_rho)
    sig_pearson = stats_funcs.stats(pearson_r)
    sig_spearman = stats_funcs.stats(spearman_rho)

    return {'pearson':{'corr':pearson_r,'sig':sig_pearson}, 'spearman':{'corr':spearman_rho,'sig':sig_spearman}}


def plot_16_item_projection(epochs_list,analysis_type = 'full_sequence', compute_reg_complexity = True):
    sensors = ['mag','grad']

    for sens in sensors:
        data_hab, data_stand, times = reshaped_the_data(epochs_list, sens,analysis_type = analysis_type)
        data = {'habituation':data_hab,'standard':data_stand}
        if analysis_type=='collapse':
            data = {'habituation': data_hab}
        times = times*1000
        NUM_COLORS = 7
        cm = plt.get_cmap('viridis')
        colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
        figsize_dict = {'full_sequence':(10 * 0.8, 7 * 0.8),'1item':(10 * 0.8, 7 * 0.8),'collapse':(10 * 0.8, 5)}
        vmin_dict = {'full_sequence':-0.7,'1item':-0.45,'collapse':-0.6}
        vmax_dict = {'full_sequence':0.75,'1item':0.65,'collapse':1}


        for key in data.keys():
            # ---- now, if analysis type is 'full_sequence' or '1item', we want to visualize it in different ways -----
            results = compute_correlation_complexity(data[key])
            plt.close('all')
            # ---- set figure's parameters, plot layout ----
            fig, ax = plt.subplots(1, 1, figsize=figsize_dict[analysis_type])
            plt.axvline(0, linestyle='-', color='black', linewidth=2)
            plt.axhline(0.5, linestyle='-', color='black', linewidth=1)
            # ---- put some vertical bars every 16 item ----
            if analysis_type == 'full_sequence':
                for xx in range(17):
                    plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            elif analysis_type == 'collapse':
                for xx in range(10):
                    plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            ax.set_xlim(np.min(times), np.max(times))
            for ii, SeqID in enumerate(range(1, 8)):
                plotiplot(data[key]['SeqID_%i'%SeqID], times,colorslist[SeqID - 1],vmin_dict[analysis_type],vmax_dict[analysis_type])

            if analysis_type != 'collapse':
                plt.gca().set_xlabel('Time (ms)', fontsize=12)
            else:
                plt.gca().set_xlabel('Presentation number in habituation block', fontsize=12)
            plt.gca().set_ylabel('Decoder predictions', fontsize=14)

            if compute_reg_complexity:
                sig = results['pearson']['sig']
                times_sig = times[sig < 0.05]
                plt.plot(times_sig, [(vmin_dict[analysis_type]+0.05)] * len(times_sig), linestyle='None', marker='o',markersize=2, color='k', linewidth=1)

            save_path = config.fig_path + '/SVM/hab_stand_7seq/'
            plt.gcf().savefig(save_path + key +'_%s_%s.svg' % (analysis_type,sens))
            plt.gcf().savefig(save_path + key +'_%s_%s.png' % (analysis_type,sens), dpi=300)
            plt.close('all')


            if compute_reg_complexity:
                fig, ax = plt.subplots(1, 1, figsize=(10 * 0.8, 0.3))
                # ---- determine the significant time-windows ----
                mean = np.mean(results['pearson']['corr'], axis=0)
                # extent = [min(times), max(times), 0,0.03]
                plt.imshow(mean[np.newaxis, :], aspect="auto", cmap="RdBu_r")
                plt.gca().set_yticks([])
                plt.colorbar()
                plt.gcf().savefig(save_path + key + '_%s_%s_corrComp.svg' % (analysis_type, sens))
                plt.gcf().savefig(save_path + key + '_%s_%s_corrComp.png' % (analysis_type, sens), dpi=300)
                plt.close('all')




def plotiplot(score_timeseries,times,color,vmin,vmax):

    plt.gcf()
    n_subj = score_timeseries.shape[0]
    nanmean_data = np.nanmean(score_timeseries, axis=0)
    ub = (nanmean_data + np.nanstd(score_timeseries, axis=0) / (np.sqrt(n_subj)))
    lb = (nanmean_data - np.nanstd(score_timeseries, axis=0) / (np.sqrt(n_subj)))

    plt.fill_between(times, ub, lb, alpha=.2,color=color)
    plt.plot(times, nanmean_data, linewidth=1.5,color=color)
    plt.ylim([vmin,vmax])



