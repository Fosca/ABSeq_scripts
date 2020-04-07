import config
import os
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.signal import savgol_filter
mne.set_log_level(verbose='WARNING')

def plot_gfp_items_standard(epochs_items, subject, h_freq=20):

    # Figures folder
    fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP_Standard_Items', subject)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Loop over the 3 ch_types
    ch_types = ['eeg', 'grad', 'mag']
    for ch_type in ch_types:
        print(ch_type)

        plt.close('all')

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if ch_type == 'eeg':
            fig.suptitle('GFP (EEG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_standard_items_EEG.png'
        elif ch_type == 'mag':
            fig.suptitle('GFP (MAG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_standard_items_MAG.png'
        elif ch_type == 'grad':
            fig.suptitle('GFP (GRAD)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_standard_items_GRAD.png'

        # ax = axes.ravel()[::1]

        plt.axvline(0, linestyle='-', color='black', linewidth=2)
        for xx in range(3):
            plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        for x in range(7):
            standards = epochs_items['SequenceID == "' + str(x + 1) + '" and ViolationInSequence == "0"'].average().savgol_filter(h_freq)
            times = standards.times * 1000
            if ch_type == 'eeg':
                gfp = np.sum(standards.copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
            elif ch_type == 'mag':
                gfp = np.sum(standards.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
            elif ch_type == 'grad':
                gfp = np.sum(standards.copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
            gfp = mne.baseline.rescale(gfp, times, baseline=(-100, 0))
            plt.plot(times, gfp, label=('SeqID_' + str(x + 1) + ', N=' + str(standards.nave)), linewidth=2)
        plt.legend(loc='upper right', fontsize=9)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xlim(-100, 750)
        ax.set_xlabel('Time [ms]')

        # Save
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

def plot_gfp_items_deviant(epochs_items, subject, h_freq=20):

    # Figures folder
    fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP_Deviant_Items', subject)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Loop over the 3 ch_types
    ch_types = ['eeg', 'grad', 'mag']
    for ch_type in ch_types:
        print(ch_type)

        plt.close('all')

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if ch_type == 'eeg':
            fig.suptitle('GFP (EEG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_deviant_items_EEG.png'
        elif ch_type == 'mag':
            fig.suptitle('GFP (MAG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_deviant_items_MAG.png'
        elif ch_type == 'grad':
            fig.suptitle('GFP (GRAD)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_deviant_items_GRAD.png'

        # ax = axes.ravel()[::1]

        plt.axvline(0, linestyle='-', color='black', linewidth=2)
        for xx in range(3):
            plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        for x in range(7):
            deviants = epochs_items['SequenceID == "' + str(x + 1) + '" and ViolationOrNot == "1"'].average().savgol_filter(h_freq)
            times = deviants.times * 1000
            if ch_type == 'eeg':
                gfp = np.sum(deviants.copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
            elif ch_type == 'mag':
                gfp = np.sum(deviants.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
            elif ch_type == 'grad':
                gfp = np.sum(deviants.copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
            gfp = mne.baseline.rescale(gfp, times, baseline=(-100, 0))
            plt.plot(times, gfp, label=('SeqID_' + str(x + 1) + ', N=' + str(deviants.nave)), linewidth=2)
        plt.legend(loc='upper right', fontsize=9)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xlim(-100, 750)
        ax.set_xlabel('Time [ms]')

        # Save
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

def plot_gfp_full_sequence_standard(epochs_full_sequence, subject, h_freq=20):

    # Figures folder
    fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP_Standard_FullSequence', subject)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Loop over the 3 ch_types
    ch_types = ['eeg', 'grad', 'mag']
    for ch_type in ch_types:
        print(ch_type)

        plt.close('all')
        fig, axes = plt.subplots(7, 1, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        if ch_type == 'eeg':
            fig.suptitle('GFP (EEG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_7sequences_standard_full_trials_EEG.png'
        elif ch_type == 'mag':
            fig.suptitle('GFP (MAG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_7sequences_standard_full_trials_MAG.png'
        elif ch_type == 'grad':
            fig.suptitle('GFP (GRAD)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_7sequences_standard_full_trials_GRAD.png'
        ax = axes.ravel()[::1]
        for x in range(7):
            standards = epochs_full_sequence['SequenceID == "' + str(x + 1) + '" and ViolationInSequence == "0"'].average().savgol_filter(h_freq)
            times = standards.times * 1000
            ax[x].axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(16):
                ax[x].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            if ch_type == 'eeg':
                gfp = np.sum(standards.copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(-500, 0))
                ax[x].plot(times, gfp, label=('Standard, N=' + str(standards.nave)), linewidth=2.5, color='green')
            elif ch_type == 'mag':
                gfp = np.sum(standards.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(-500, 0))
                ax[x].plot(times, gfp, label=('Standard, N=' + str(standards.nave)), linewidth=2.5, color='darkorange')
            elif ch_type == 'grad':
                gfp = np.sum(standards.copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(-500, 0))
                ax[x].plot(times, gfp, label=('Standard, N=' + str(standards.nave)), linewidth=2.5, color='purple')
            ax[x].legend(loc='upper left', fontsize=10)
            ax[x].set_yticklabels([])
            ax[x].set_ylabel('SequenceID_' + str(x + 1))
            ax[x].set_xlim(-500, 4250)
        axes.ravel()[-1].set_xlabel('Time [ms]')
        # Save
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

def plot_gfp_full_sequence_deviants_4pos(epochs_full_sequence, subject, h_freq=20):

    # Figures folder
    fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP_Deviants4pos_FullSequence', subject)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Loop over the 3 ch_types
    ch_types = ['eeg', 'grad', 'mag']
    for ch_type in ch_types:
        print(ch_type)

        plt.close('all')
        fig, axes = plt.subplots(7, 1, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
        if ch_type == 'eeg':
            fig.suptitle('GFP (EEG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_7sequences_deviant_4positions_full_trials_EEG.png'
        elif ch_type == 'mag':
            fig.suptitle('GFP (MAG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_7sequences_deviant_4positions_full_trials_MAG.png'
        elif ch_type == 'grad':
            fig.suptitle('GFP (GRAD)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_7sequences_deviant_4positions_full_trials_GRAD.png'
        ax = axes.ravel()[::1]
        for x in range(7):
            # Select only the trials with a violation (for one sequence)
            seqEpochs = epochs_full_sequence['SequenceID == "' + str(x + 1) + '" and ViolationInSequence > 0'].copy()
            # Create 'Evoked' object that will contain the evoked response for each of the 4 deviant positions (of one sequence)
            data4pos = []
            all_devpos = np.unique(seqEpochs.metadata.ViolationInSequence)  # Position of deviants
            for devpos in range(len(all_devpos)):
                data4pos.append(
                    seqEpochs['ViolationInSequence == "' + str(all_devpos[devpos]) + '"'].average().savgol_filter(h_freq))
            # -------
            times = seqEpochs.times * 1000
            ax[x].axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(16):
                ax[x].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            if ch_type == 'eeg':
                for devpos in range(len(all_devpos)):
                    gfp = np.sum(data4pos[devpos].copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
                    gfp = mne.baseline.rescale(gfp, times, baseline=(-500, 0))
                    ax[x].plot(times, gfp, label=('Deviant position=' + str(all_devpos[devpos]) + ', N=' + str(data4pos[devpos].nave)), linewidth=2)
            elif ch_type == 'mag':
                for devpos in range(len(all_devpos)):
                    gfp = np.sum(data4pos[devpos].copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
                    gfp = mne.baseline.rescale(gfp, times, baseline=(-500, 0))
                    ax[x].plot(times, gfp, label=('Deviant position=' + str(all_devpos[devpos]) + ', N=' + str(data4pos[devpos].nave)), linewidth=2)
            elif ch_type == 'grad':
                for devpos in range(len(all_devpos)):
                    gfp = np.sum(data4pos[devpos].copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
                    gfp = mne.baseline.rescale(gfp, times, baseline=(-500, 0))
                    ax[x].plot(times, gfp, label=('Deviant position=' + str(all_devpos[devpos]) + ', N=' + str(data4pos[devpos].nave)), linewidth=2)
            ax[x].legend(loc='upper left', fontsize=10)
            ax[x].set_yticklabels([])
            ax[x].set_ylabel('SequenceID_' + str(x + 1))
            ax[x].set_xlim(-500, 4250)
        axes.ravel()[-1].set_xlabel('Time [ms]')
        # Save
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

def gfp_evoked(evoked_list,baseline=None):

    times = evoked_list[0][0].times
    gfp_eeg_all = []
    gfp_mag_all = []
    gfp_grad_all = []

    for evoked in evoked_list:

        gfp_eeg = np.sum(evoked[0].copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
        gfp_grad = np.sum(evoked[0].copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
        gfp_mag = np.sum(evoked[0].copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)

        if baseline is not None:
            for gfp in [gfp_eeg, gfp_grad, gfp_mag]:
                gfp = mne.baseline.rescale(gfp, times, baseline=(baseline, 0))

        gfp_eeg_all.append(gfp_eeg)
        gfp_grad_all.append(gfp_grad)
        gfp_mag_all.append(gfp_mag)

    gfp_evoked = {'eeg':gfp_eeg_all,
                  'grad':gfp_grad_all,
                  'mag':gfp_mag_all}

    return gfp_evoked,times

def plot_GFP_with_sem(GFP_all_subjects,times, color_mean=None, label=None, filter=False):

    mean = np.mean(GFP_all_subjects, axis=0)
    ub = mean + sem(GFP_all_subjects, axis=0)
    lb = mean - sem(GFP_all_subjects, axis=0)

    if filter==True:
        mean = savgol_filter(mean, 11, 3)
        ub = savgol_filter(ub, 11, 3)
        lb = savgol_filter(lb, 11, 3)

    plt.fill_between(times, ub, lb, color=color_mean, alpha=.2)
    plt.plot(times, mean, color=color_mean, linewidth=1.5, label=label)

