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
from ABseq_func import *
import matplotlib.pyplot as plt
import config
import numpy as np
from scipy.signal import savgol_filter
import os.path as op
from scipy import stats
import mne
import matplotlib.ticker as ticker


# ---------- plot the gats for all the sequences ------------
# compute the average and find a time window of maximal decoding in order to apply it on the 16 item sequences



# --------------


suf = 'SW_train_test_different_blocks'
sensors = ['all_chans','mag', 'grad']
epochs_16 = {sens : {'hab':[],'test':[]} for sens in sensors}

for subject in config.subjects_list:
    if op.exists(op.join(config.meg_dir, subject, 'mag_SVM_on_16_items_test_window-epo.fif')):
        for sens in sensors:
            epochs_16[sens]['test'].append(mne.read_epochs(op.join(config.meg_dir, subject, sens+suf+'_SVM_on_16_items_test_window-epo.fif')))
            epochs_16[sens]['hab'].append(mne.read_epochs(op.join(config.meg_dir, subject, sens+suf+'_SVM_on_16_items_habituation_window-epo.fif')))

# ===== FIGURES ===== #
save_folder = op.join(config.fig_path, 'SVM', 'Full_sequence_projection')
utils.create_folder(save_folder)

# Figure with only one EMS projected (average window)
epochs_list = {}
vminvmax = {'all_chans':2,'mag':1,'grad':1}
for sens in sensors:
    win_tmin = epochs_16[sens]['test'][0][0].metadata.SVM_filter_tmin_window[0]*1000
    win_tmax = epochs_16[sens]['test'][0][0].metadata.SVM_filter_tmax_window[0]*1000

    plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_16[sens], compute_reg_complexity = True,
                                                                  window_CBPT_violation = 0.7,sensor_type=sens,
                                                                  save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims_tvals.png' % ( sens, win_tmin, win_tmax)),
                                                                  vmin=-vminvmax[sens],vmax=vminvmax[sens],plot_betas=False)






# ______________________________________________________________________________________________________________________
def plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type, save_path=None, vmin=-1, vmax=1,compute_reg_complexity = False, window_CBPT_violation = None,plot_betas=True):
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
        #  this provides us with the position of the violations and the times
        epochs_seq_subset = epochs_list['test'][0]['SequenceID == "' + str(seqID) + '"']
        times = epochs_seq_subset.times
        times = times + 0.3
        violpos_list = np.unique(epochs_seq_subset.metadata['ViolationInSequence'])
        violation_significance[seqID] = {'times':times,'window_significance':window_CBPT_violation}

        #  ----------- habituation trials -----------
        epochs_data_hab_seq = []
        y_list_epochs_hab = []
        data_mean = []
        mean_alpha = []
        for epochs in epochs_list['hab']:
            epochs_subset = epochs['SequenceID == "' + str(seqID) + '"']
            avg_epo = np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0)
            y_list_epochs_hab.append(avg_epo)
            epochs_data_hab_seq.append(avg_epo)
        epochs_data_hab_allseq.append(epochs_data_hab_seq)
        mean_hab = np.mean(y_list_epochs_hab, axis=0)
        data_mean.append(mean_hab)
        mean_alpha.append(np.zeros(mean_hab.shape))

        #  ----------- test trials -----------
        epochs_data_test_seq = []

        for viol_pos in violpos_list:
            y_list = []
            y_list_alpha = []
            contrast_viol_pos = []
            for epochs in epochs_list['test']:
                epochs_subset = epochs[
                    'SequenceID == "' + str(seqID) + '" and ViolationInSequence == "' + str(viol_pos) + '"']
                avg_epo = np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0)
                y_list.append(avg_epo)
                if viol_pos==0:
                    avg_epo_standard = np.mean(np.squeeze(epochs_subset.savgol_filter(20).get_data()), axis=0)
                    epochs_data_test_seq.append(avg_epo_standard)
                    y_list_alpha.append(np.zeros(avg_epo_standard.shape))
                if viol_pos !=0 and window_CBPT_violation is not None:
                    epochs_standard = epochs[
                        'SequenceID == "' + str(seqID) + '" and ViolationInSequence == 0']
                    avg_epo_standard = np.mean(np.squeeze(epochs_standard.savgol_filter(20).get_data()), axis=0)
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
                y_list_alpha.append(1*(p_vals<0.05))

            mean = np.mean(y_list, axis=0)
            mean_alpha_seq = np.mean(y_list_alpha, axis=0)
            data_mean.append(mean)
            mean_alpha.append(mean_alpha_seq)
        epochs_data_test_allseq.append(epochs_data_test_seq)

        width = 75
        # Add vertical lines, and "xY"
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
            txt = seqtxtXY[n][xx]
            ax[n].text(250 * (xx + 1) - 125, width * 6 + (width / 3), txt, horizontalalignment='center', fontsize=16)

        # return data_mean
        im = ax[n].imshow(data_mean, extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap='RdBu_r',
                          vmin=vmin, vmax=vmax)
        # add colorbar
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.50, aspect=10, pad=.005)
        cb.ax.yaxis.set_offset_position('left')
        cb.set_label('a. u.')
        if window_CBPT_violation:
            masked = np.ma.masked_where(mean_alpha == 0, mean_alpha)
            im = ax[n].imshow(masked, extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap=cmapsig,
                              vmin=vmin, vmax=vmax,alpha=0.7)
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
            ax[n].plot([x, x], [y1, y2], linestyle='-', color='yellow', linewidth=3)

        n += 1

    if compute_reg_complexity:
        epochs_data_hab_allseq = np.asarray(epochs_data_hab_allseq)
        epochs_data_test_allseq = np.asarray(epochs_data_test_allseq)
        coeff_const_hab, coeff_complexity_hab, t_const_hab, t_complexity_hab = SVM_funcs.compute_regression_complexity(epochs_data_hab_allseq)
        coeff_const_test, coeff_complexity_test, t_const_test, t_complexity_test = SVM_funcs.compute_regression_complexity(epochs_data_test_allseq)

        for xx in range(16):
            ax[7].axvline(250 * xx, linestyle='--', color='black', linewidth=1)

        # return data_mean
        if plot_betas:
            im = ax[7].imshow(np.asarray([np.mean(coeff_complexity_hab,axis=0),np.mean(coeff_complexity_test,axis=0)]), extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap='RdBu_r',
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

