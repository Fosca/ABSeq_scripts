from ABseq_func import *
import mne
import config
import scipy
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from scipy.stats.stats import pearsonr
import pickle
import os.path as op
from scipy.stats import sem
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================================== #
# Set up analysis
# ============================================================================================== #
# Exclude some subjects
# config.exclude_subjects.append('sub10-gp_190568')
# config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
# config.subjects_list.sort()

# Parameters
epochs_type = 'items'  # 'items' or 'fullseq'
detrend_epochs = False
lowpass_epochs = False  # option to filter epochs with  30Hz lowpass filter
baseline_epochs = False  # apply baseline to the epochs (if wasn't already done)
complexity_values = [config.complexity[ii] for ii in range(1,8)]

# Output folder
results_path = op.join(config.result_path, 'Corr_GFPxComplexity', epochs_type)
if detrend_epochs:
    results_path = op.join(results_path, 'with_detrend')
else:
    results_path = op.join(results_path, 'no_detrend')
if lowpass_epochs:
    results_path = op.join(results_path, 'with_lowpass')
else:
    results_path = op.join(results_path, 'no_lowpass')
if baseline_epochs:
    results_path = op.join(results_path, 'with_baseline')
else:
    results_path = op.join(results_path, 'no_baseline')
utils.create_folder(results_path)
print(results_path)

# ============================================================================================== #
# ======= Extract correlation with complexity over time for 3 trial types X 3 ch_types x N subjects
# ============================================================================================== #
# Empty dictionaries to fill
correlation_data = {}
gfp_data = {}
for ttype in ['habituation', 'standard', 'violation', 'viol_minus_stand']:
    correlation_data[ttype] = {}
    gfp_data[ttype] = {}
    for ch_type in config.ch_types:
        correlation_data[ttype][ch_type] = []
        gfp_data[ttype][ch_type] = {}
        for seqID in range(1, 8):
            gfp_data[ttype][ch_type][seqID] = []

# Extract the data: subjects loop
for subject in config.subjects_list:
    print('-- Subject ' + subject)

    # -- LOAD THE EPOCHS -- #
    if epochs_type == 'items':
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=True, AR_type='global')
    elif epochs_type == 'fullseq':
        epochs = epoching_funcs.load_epochs_full_sequence(subject, cleaned=True, AR_type='global')

    for ttype in ['habituation', 'standard', 'violation', 'viol_minus_stand']:
        print('  ---- Trial type ' + str(ttype))
        gfp_eeg_seq = []
        gfp_mag_seq = []
        gfp_grad_seq = []
        for seqID in range(1, 8):
            print('    ------ Sequence ' + str(seqID))
            if ttype == 'viol_minus_stand':
                epochs_bal = epoching_funcs.balance_epochs_violation_positions(epochs.copy(), 'sequence')
                epochs_subset = epochs_bal['SequenceID == ' + str(seqID)].copy()
            elif ttype == 'habituation':
                epochs_subset = epochs['TrialNumber <= 10 and SequenceID == ' + str(seqID)].copy()
            elif ttype == 'standard':  # For standards, taking only items from trials with no violation
                epochs_subset = epochs['TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationInSequence == 0'].copy()
            elif ttype == 'violation':
                if epochs_type == 'items':
                    epochs_subset = epochs['TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationOrNot == 1'].copy()
                elif epochs_type == 'fullseq':
                    epochs_subset = epochs['TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationInSequence > 0'].copy()

            # Linear detrend
            if detrend_epochs:
                epochs_subset._data[:] = scipy.signal.detrend(epochs_subset.get_data(), axis=-1, type='linear')

            # Filter
            if lowpass_epochs:
                epochs_subset = epochs_subset.filter(l_freq=None, h_freq=30, n_jobs=4)

            # Baseline
            if baseline_epochs:
                if epochs_type == 'items':
                    epochs_subset = epochs_subset.apply_baseline((-0.050, 0))
                elif epochs_type == 'fullseq':
                    epochs_subset = epochs_subset.apply_baseline((-0.200, 0))

            # Average epochs
            if ttype == 'viol_minus_stand':
                # Here we compute a contrast between evoked
                e1 = epochs_subset['ViolationOrNot == 0'].copy().average()
                e2 = epochs_subset['ViolationOrNot == 1'].copy().average()
                ev = mne.combine_evoked([e1, e2], weights=[-1, 1])
                # ev = mne.combine_evoked([e2, -e1], 'equal')  # not equivalent ?! different nave and scaling
            else:
                ev = epochs_subset.average()

            # clear
            epochs_subset = []

            # Compute GFP
            for ch_type in config.ch_types:
                if ch_type == 'eeg':
                    gfp = np.sum(ev.copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
                    gfp_eeg_seq.append(gfp)
                elif ch_type == 'mag':
                    gfp = np.sum(ev.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
                    gfp_mag_seq.append(gfp)
                elif ch_type == 'grad':
                    gfp = np.sum(ev.copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
                    gfp_grad_seq.append(gfp)
                # Store gfp each seq
                gfp_data[ttype][ch_type][seqID].append(gfp)

        # Arrays of GFP for the 7 sequences
        if 'eeg' in config.ch_types:
            gfp_eeg_seq = np.array(gfp_eeg_seq)
        gfp_mag_seq = np.array(gfp_mag_seq)
        gfp_grad_seq = np.array(gfp_grad_seq)

        # Compute correlation over time
        for ch_type in config.ch_types:
            timeR = []
            for timepoint in range(len(ev.times)):
                if ch_type == 'eeg':
                    r, pval = pearsonr(gfp_eeg_seq[:, timepoint], complexity_values)
                elif ch_type == 'mag':
                    r, pval = pearsonr(gfp_mag_seq[:, timepoint], complexity_values)
                elif ch_type == 'grad':
                    r, pval = pearsonr(gfp_grad_seq[:, timepoint], complexity_values)
                timeR.append(r)
            # Store correlation curve
            correlation_data[ttype][ch_type].append(timeR)
    # clear
    epochs = []

    # plt.close('all')
    # plt.figure()
    # for ttype in ['habituation', 'standard', 'violation']:
    #     plt.plot(ev.times, correlation_data[ttype]['grad'][0], label=ttype)
    # plt.legend()

# Save data
correlation_data['times'] = ev.times
with open(op.join(results_path, 'correlation_data.pickle'), 'wb') as f:
    pickle.dump(correlation_data, f, pickle.HIGHEST_PROTOCOL)
gfp_data['times'] = ev.times
with open(op.join(results_path, 'gfp_each_seq_data.pickle'), 'wb') as f:
    pickle.dump(gfp_data, f, pickle.HIGHEST_PROTOCOL)

# ============================================================================================== #
# ======= Plot correlation results
# ============================================================================================== #
with open(op.join(results_path, 'correlation_data.pickle'), 'rb') as f:
    correlation_data = pickle.load(f)
# subs_to_exclude_idx = []
# for ii in range(len(config.exclude_subjects)):
#     subs_to_exclude_idx.append(config.subjects_list.index(config.exclude_subjects[ii]))

times = correlation_data['times']*1000
if epochs_type == 'items':
    figsize = (5, 8)
elif epochs_type == 'fullseq':
    figsize = (15, 8)

ch_colors = dict(eeg='green', grad='red', mag='blue')
for ttype in ['habituation', 'standard', 'violation', 'viol_minus_stand']:
    # Open figure
    plt.close('all')
    fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=False, constrained_layout=True)  #, sharex=False, sharey=True, constrained_layout=True)
    fig.suptitle('Corr_GFPxComplexity: ' + ttype + ' trials', fontsize=12)

    for iplot, ch_type in enumerate(['mag', 'grad']):

        # Data to plot
        data = correlation_data[ttype][ch_type]
        mean = np.mean(data, axis=0)
        ub = mean + sem(data, axis=0)
        lb = mean - sem(data, axis=0)

        # T test
        t, pval = scipy.stats.ttest_1samp(data, popmean=0, alternative='two-sided')
        signif = (pval < 0.001)*1

        # Cluster perm T test
        t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(np.array(data), n_permutations=1000, threshold=None, out_type='mask')  # If threshold is None, t-threshold equivalent to p < 0.05 (if t-statistic))
        good_cluster_inds = np.where(cluster_pv < 0.05)[0]

        ax[iplot].set_title(ch_type, loc='left', weight='normal', fontsize=10)
        # Add vertical lines
        for xx in range(16):
            ax[iplot].axvline(250 * xx, linestyle='--', color='grey', linewidth=1)
        # Add horizontal lines
        ax[iplot].axhline(0, linestyle='-', color='black', linewidth=1)

        # Remove spines
        for key in ('top', 'right', 'bottom'):
            ax[iplot].spines[key].set(visible=False)

        # # Plot markers for significant effect: T test
        # for xx in range(len(signif)):
        #     if signif[xx] == 1:
        #         print(xx)
        #         ax[iplot].plot(times[xx], -0.8, marker='s', color='black', markersize=8)

        # Plot markers for significant effect: cluster perm t test
        if len(good_cluster_inds) > 0:
            for i_clu, clu_idx in enumerate(good_cluster_inds):
                clu_times = times[clusters[clu_idx]]
                # ax.plot(clu_times, np.ones(len(clu_times))*-0.8, linestyle='-', color='grey', linewidth=10)  # line ?
                ax[iplot].fill_betweenx((-1, 1), clu_times[0], clu_times[-1], color='grey', alpha=.2, linewidth=0.0)  # bar ?

        # Plot data
        ax[iplot].fill_between(times, ub, lb, color=ch_colors[ch_type], alpha=.2)
        ax[iplot].plot(times, mean, color=ch_colors[ch_type], linewidth=1.5, label='Pearson r')

        # Various...
        ax[iplot].set_ylabel('Pearson r')
        # ax.set_xticks([], [])
        ax[iplot].set_xlabel('Time (ms)')
        if epochs_type == 'items':
            # ax[iplot].set_xticks(range(-100, 600, 100), [])
            ax[iplot].set_xticks(range(-100, 600, 100))
            ax[iplot].set_xlim([-50, 600])
        elif epochs_type == 'fullseq':
            # ax[iplot].set_xticks(range(-200, 4250, 250), [])
            ax[iplot].set_xticks(range(-200, 4250, 250))
            ax[iplot].set_xlim([-200, 4250])
        ax[iplot].set_ylim([-0.8, 0.8])
    # fig.tight_layout(pad=3.0)
    # Save figure
    fig_name = op.join(results_path, 'Corr_with_complexity_ ' + epochs_type + '_' + ttype + '.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)


# ============================================================================================== #
# ======= Plot gfp V1
# ============================================================================================== #
with open(op.join(results_path, 'gfp_each_seq_data.pickle'), 'rb') as f:
    gfp_data = pickle.load(f)
datatype = 'items'
figsize = (3, 9)
times = gfp_data['times']*1000
ch_colors = dict(eeg='green', grad='red', mag='blue')

for ttype in ['habituation', 'standard', 'violation', 'viol_minus_stand']:
    for ch_type in config.ch_types:

        # Create figure
        fig, ax = plt.subplots(7, 1, figsize=figsize, sharex=False, sharey=True, constrained_layout=True)
        fig.suptitle(ttype+'_'+ch_type, fontsize=12)

        # Plot
        for nseq in range(7):
            seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(nseq+1)
            mean = np.mean(gfp_data[ttype][ch_type][nseq+1], axis=0)
            ub = mean + sem(gfp_data[ttype][ch_type][nseq+1], axis=0)
            lb = mean - sem(gfp_data[ttype][ch_type][nseq+1], axis=0)

            ax[nseq].set_title(seqname, loc='left', weight='bold', fontsize=12)
            ax[nseq].fill_between(times, ub, lb, color='black', alpha=.2)
            ax[nseq].plot(times, mean, color=ch_colors[ch_type], linewidth=1.5, label=seqname)
            ax[nseq].axvline(0, linestyle='-', color='black', linewidth=2)
            # ax[nseq].axhline(0, linestyle='-', color='black', linewidth=1)
            ax[nseq].set_xlim([min(times), max(times)])
            # Add vertical lines
            for xx in range(16):
                ax[nseq].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
            # Remove spines
            for key in ('top', 'right'):
                ax[nseq].spines[key].set(visible=False)
            ax[nseq].set_ylabel('GFP')
            # ax[nseq].set_xticks([], [])
            fmt = ticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            ax[nseq].get_yaxis().set_major_formatter(fmt)
            if datatype == 'items':
                ax[nseq].get_yaxis().get_offset_text().set_position((-0.22, 0))  # move 'x10^x', does not work with y
            elif datatype == 'fullseq':
                ax[nseq].get_yaxis().get_offset_text().set_position((-0.07, 0))  # move 'x10^x', does not work with y
        if datatype == 'items':
            tmp = np.arange(0, 800, 200)
            ax[nseq].set_xticks(tmp)
            ax[nseq].set_ylim(ymin=0)
        elif datatype == 'fullseq':
            ax[nseq].set_xticks(range(-500, 4500, 500), [])
            # Add "xY" using the same yval for all
            ylim = ax[nseq].get_ylim()
            yval = ylim[1] - ylim[1] * 0.1
            for nseq in range(7):
                seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(nseq + 1)
                for xx in range(16):
                    ax[nseq].text(250 * (xx + 1) - 125, yval, seqtxtXY[xx], horizontalalignment='center', fontsize=12)
        ax[nseq].set_xlabel('Time (ms)')

        # Save
        fig_name = op.join(results_path, ttype+'_'+ch_type + '_GFP_eachseq.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)

# ============================================================================================== #
# ======= Plot gfp V2
# ============================================================================================== #
with open(op.join(results_path, 'gfp_each_seq_data.pickle'), 'rb') as f:
    gfp_data = pickle.load(f)

times = gfp_data['times']
NUM_COLORS = 7
cm = plt.get_cmap('viridis')
colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])

for ttype in ['habituation', 'standard', 'violation', 'viol_minus_stand']:
    for ch_type in config.ch_types:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        plt.axvline(0, linestyle='-', color='black', linewidth=2)
        for xx in range(3):
            plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        # fig.suptitle('GFP (%s)' % ch_type + ', N subjects=' + str(len(evoked_list[next(iter(evoked_list))])), fontsize=12)
        for seqID in range(1,8):
            seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqID)
            GFP_funcs.plot_GFP_with_sem(gfp_data[ttype][ch_type][seqID], times * 1000, color_mean=colorslist[seqID-1], label=seqname, filter=False)
        plt.legend(loc='upper right', fontsize=9)
        ax.set_xlim([-50, 600])
        ax.set_ylim(ymin=0)
        tmp = np.arange(0, 800, 200)
        ax.set_xticks(tmp)
        # Remove spines
        for key in ('top', 'right'):
            ax.spines[key].set(visible=False)
        ax.set_ylabel('GFP')
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax.get_yaxis().set_major_formatter(fmt)
        ax.legend(fontsize=6)

        fig_name = op.join(results_path, ttype+'_'+ch_type + '_GFP_allseq.png')
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close('all')
