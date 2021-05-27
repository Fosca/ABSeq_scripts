# All the functions related to the computation and plotting of the GFP
import config
import os
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import sem
from scipy.signal import savgol_filter
import copy
import matplotlib.ticker as ticker
from ABseq_func import GFP_funcs
from ABseq_func import epoching_funcs

mne.set_log_level(verbose='WARNING')


# ______________________________________________________________________________________________________________________
def gfp_evoked(evoked_list, baseline=None,times=None):
    """
    Compute the global field power from the list of the evoked activities evoked_list.
    :param evoked_list: List of evoked activities that may correspond to the different participants.
    :param baseline: Set it to tmin < 0 if you want to baseline the evoked activity from tmin to 0
    :return: Dictionnary that contains the gfp for all the types of sensors, times array
    """
    if times is None:
        times = evoked_list[0][0].times

    gfp_eeg_all = []
    gfp_mag_all = []
    gfp_grad_all = []

    for evoked in evoked_list:


        gfp_grad = np.sum(evoked[0].copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
        gfp_mag = np.sum(evoked[0].copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
        if baseline is not None:
            for gfp in [gfp_grad, gfp_mag]:
                gfp = mne.baseline.rescale(gfp, times, baseline=(baseline, 0))
        gfp_grad_all.append(gfp_grad)
        gfp_mag_all.append(gfp_mag)
        gfp_evoked = {'grad': gfp_grad_all,
                      'mag': gfp_mag_all}

        if config.noEEG == False:
            gfp_eeg = np.sum(evoked[0].copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
            if baseline is not None:
                gfp_eeg = mne.baseline.rescale(gfp_eeg, times, baseline=(baseline, 0))
            gfp_eeg_all.append(gfp_eeg)
            gfp_evoked = {'eeg': gfp_eeg_all,
                          'grad': gfp_grad_all,
                          'mag': gfp_mag_all}

    return gfp_evoked, times


# ______________________________________________________________________________________________________________________
def plot_GFP_with_sem(GFP_all_subjects, times, color_mean=None, label=None, filter=False):
    """
    Plots the mean GFP_all_subjects with the sem of GFP_all_subjects in shaded areas
    :param GFP_all_subjects: Could be the output of gfp_evoked
    :param times: From the output of gfp_evoked
    :param color_mean: Color for the mean of the GFP
    :param label:
    :param filter: If you want to lowpass filter the data to smooth it, e.g. for visualisation purposes.
    :return: None
    """

    mean = np.mean(GFP_all_subjects, axis=0)
    ub = mean + sem(GFP_all_subjects, axis=0)
    lb = mean - sem(GFP_all_subjects, axis=0)

    if filter == True:
        mean = savgol_filter(mean, 9, 3)
        ub = savgol_filter(ub, 9, 3)
        lb = savgol_filter(lb, 9, 3)

    plt.fill_between(times, ub, lb, color=color_mean, alpha=.2)
    plt.plot(times, mean, color=color_mean, linewidth=1.5, label=label)


# ______________________________________________________________________________________________________________________
def plot_gfp_items_standard_or_deviants(epochs_items, subject, h_freq=30, standard_or_deviant='standard',ch_types = ['eeg', 'grad', 'mag'],list_sequences=range(1,8)):
    """
    For the subject 'subject', this function gets the corresponding epochs on each item and computes the GFP.
    :param epochs_items:
    :param subject: NIP of the participant
    :param h_freq: Low pass filter to filter the epochs and have a smoother GFP
    :return: None : this function plots and saves, that's it.
    """

    # # Crop the epochs_items data if config.tcrop is not None
    # if config.tcrop is not None:
    #     epochs_items.crop(tmax=config.tcrop)

    # Create the figure folder if it does not exist yet
    if standard_or_deviant == "standard":
        suffix = 'standard'
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP_Standard_Items', subject)
    else:
        suffix = 'deviant'
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP_Deviant_Items', subject)

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Loop over the 3 ch_types and plot the GFP

    for ch_type in ch_types:
        print("Plotting the GFP for the 7 sequences for channel type %s .\n" % ch_type)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if ch_type == 'eeg':
            fig.suptitle('GFP (EEG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_' + suffix + '_items_EEG.png'
        elif ch_type == 'mag':
            fig.suptitle('GFP (MAG)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_' + suffix + '_items_MAG.png'
        elif ch_type == 'grad':
            fig.suptitle('GFP (GRAD)', fontsize=12)
            fig_name = fig_path + op.sep + 'GFP_' + suffix + '_items_GRAD.png'

        plt.axvline(0, linestyle='-', color='black', linewidth=2)
        for xx in range(3):
            plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        for x in list_sequences:
            if standard_or_deviant == "standard":
                epochs_of_interest = epochs_items['SequenceID == "' + str(x) + '" and ViolationInSequence == 0']
                epochs_of_interest = epochs_of_interest.average()
                epochs_of_interest = epochs_of_interest.filter(h_freq=h_freq, l_freq=None)
            else:
                epochs_of_interest = epochs_items['SequenceID == "' + str(x) + '" and ViolationOrNot == 1']
                epochs_of_interest = epochs_of_interest.average()
                epochs_of_interest = epochs_of_interest.filter(h_freq=h_freq, l_freq=None)
            times = epochs_of_interest.times * 1000
            if ch_type == 'eeg':
                gfp = np.sum(epochs_of_interest.copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
            elif ch_type == 'mag':
                gfp = np.sum(epochs_of_interest.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
            elif ch_type == 'grad':
                gfp = np.sum(epochs_of_interest.copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
            gfp = mne.baseline.rescale(gfp, times, baseline=(-100, 0))
            plt.plot(times, gfp, label=('SeqID_' + str(x) + ', N=' + str(epochs_of_interest.nave)), linewidth=2)
        plt.legend(loc='upper right', fontsize=9)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xlim(times[0], times[-1])
        ax.set_xlabel('Time [ms]')

        # Save
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)
        plt.close('all')


# ______________________________________________________________________________________________________________________
def plot_gfp_full_sequence_standard(epochs_full_sequence, subject, h_freq=30,ch_types = ['eeg', 'grad', 'mag'],list_sequences=range(1,8)):
    """
    For each participant "subject", this function computes the GFP on the 16 item long sequence epoch
    :param epochs_full_sequence: 16 item long sequence epoch
    :param subject: The NIP of the subject
    :param h_freq: Low pass filter to filter the epochs and have a smoother GFP.
    :return:
    """

    # Figures folder
    fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP_Standard_FullSequence', subject)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Loop over the 3 ch_types
    for ch_type in ch_types:
        print(ch_type)

        plt.close('all')
        fig, axes = plt.subplots(len(list_sequences), 1, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
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
        for x, seq_num in enumerate(list_sequences):
            standards = epochs_full_sequence['SequenceID == ' + str(seq_num) + ' and ViolationInSequence == 0'].average()
            standards = standards.filter(h_freq=h_freq, l_freq=None)
            times = standards.times * 1000
            ax[x].axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(16):
                ax[x].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            if ch_type == 'eeg':
                gfp = np.sum(standards.copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(-200, 0))
                ax[x].plot(times, gfp, label=('Standard, N=' + str(standards.nave)), linewidth=2.5, color='green')
            elif ch_type == 'mag':
                gfp = np.sum(standards.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(-200, 0))
                ax[x].plot(times, gfp, label=('Standard, N=' + str(standards.nave)), linewidth=2.5, color='darkorange')
            elif ch_type == 'grad':
                gfp = np.sum(standards.copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
                gfp = mne.baseline.rescale(gfp, times, baseline=(-200, 0))
                ax[x].plot(times, gfp, label=('Standard, N=' + str(standards.nave)), linewidth=2.5, color='purple')
            ax[x].legend(loc='upper left', fontsize=10)
            ax[x].set_yticklabels([])
            ax[x].set_ylabel('SequenceID_' + str(seq_num))
            ax[x].set_xlim(-200, 4250)
        axes.ravel()[-1].set_xlabel('Time [ms]')
        # Save
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)


# ______________________________________________________________________________________________________________________
def plot_gfp_full_sequence_deviants_4pos(epochs_items, subject, h_freq=30,ch_types = ['eeg', 'grad', 'mag'],list_sequences=range(1,8)):
    """
    For each participant "subject", this function plots the GFP only for the violations.
    :param epochs_items:
    :param subject: NIP of the participant
    :param l_freq: Low pass filter to filter the epochs and have a smoother GFP
    :return: None
    """

    # Figures folder
    fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP_Deviants4pos_FullSequence', subject)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Loop over the 3 ch_types
    for ch_type in ch_types:
        print(ch_type)

        plt.close('all')
        fig, axes = plt.subplots(len(list_sequences), 1, figsize=(16, 12), sharex=True, sharey=True, constrained_layout=True)
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
        for x, num_seq in enumerate(list_sequences):
            # Select only the trials with a violation (for one sequence)
            seqEpochs = epochs_items['SequenceID == ' + str(num_seq) + ' and ViolationInSequence > 0'].copy()
            # Create 'Evoked' object that will contain the evoked response for each of the 4 deviant positions (of one sequence)
            data4pos = []
            all_devpos = np.unique(seqEpochs.metadata.ViolationInSequence)  # Position of deviants
            for devpos in range(len(all_devpos)):
                data4pos.append(
                    seqEpochs['ViolationInSequence == ' + str(all_devpos[devpos])].average().filter(h_freq=h_freq, l_freq=None))

            # -------
            times = seqEpochs.times * 1000
            ax[x].axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(16):
                ax[x].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            if ch_type == 'eeg':
                for devpos in range(len(all_devpos)):
                    gfp = np.sum(data4pos[devpos].copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
                    gfp = mne.baseline.rescale(gfp, times, baseline=(-200, 0))
                    ax[x].plot(times, gfp, label=('Deviant position=' + str(all_devpos[devpos]) + ', N=' + str(data4pos[devpos].nave)), linewidth=2)
            elif ch_type == 'mag':
                for devpos in range(len(all_devpos)):
                    gfp = np.sum(data4pos[devpos].copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
                    gfp = mne.baseline.rescale(gfp, times, baseline=(-200, 0))
                    ax[x].plot(times, gfp, label=('Deviant position=' + str(all_devpos[devpos]) + ', N=' + str(data4pos[devpos].nave)), linewidth=2)
            elif ch_type == 'grad':
                for devpos in range(len(all_devpos)):
                    gfp = np.sum(data4pos[devpos].copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
                    gfp = mne.baseline.rescale(gfp, times, baseline=(-200, 0))
                    ax[x].plot(times, gfp, label=('Deviant position=' + str(all_devpos[devpos]) + ', N=' + str(data4pos[devpos].nave)), linewidth=2)
            ax[x].legend(loc='upper left', fontsize=10)
            ax[x].set_yticklabels([])
            ax[x].set_ylabel('SequenceID_' + str(num_seq))
            ax[x].set_xlim(-200, 4250)
        axes.ravel()[-1].set_xlabel('Time [ms]')
        # Save
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)


# ______________________________________________________________________________________________________________________
def plot_GFP_timecourse_7seq(evoked_dict, ch_type='eeg', filter=True):
    """
    Function to plot the mean and the sem of the GFP for all the sequences across the participants.
    :param evoked_dict:
    :param ch_type:
    :param filter:
    :return:
    """

    evoked_dict_copy = copy.deepcopy(evoked_dict)

    # Additional parameters
    units = dict(eeg='uV', grad='fT/cm', mag='fT')
    ch_colors = dict(eeg='green', grad='red', mag='blue')

    # Create group average GFP per sequence
    allseq_mean = []
    allseq_ub = []
    allseq_lb = []
    for nseq in range(7):
        cond = list(evoked_dict_copy.keys())[nseq]
        data = copy.deepcopy(evoked_dict)
        data = data[cond]
        gfp_cond, times = GFP_funcs.gfp_evoked(data)
        mean = np.mean(gfp_cond[ch_type], axis=0)
        ub = mean + sem(gfp_cond[ch_type], axis=0)
        lb = mean - sem(gfp_cond[ch_type], axis=0)
        if filter:
            mean = savgol_filter(mean, 11, 3)
            ub = savgol_filter(ub, 11, 3)
            lb = savgol_filter(lb, 11, 3)
        allseq_mean.append(mean)
        allseq_ub.append(ub)
        allseq_lb.append(lb)
    times = times * 1000

    if times[-1] > 3000:
        datatype = 'fullseq'
        figsize = (9, 9)
    else:
        datatype = 'items'
        figsize = (3, 9)

    # Create figure
    fig, ax = plt.subplots(7, 1, figsize=figsize, sharex=False, sharey=True, constrained_layout=True)
    fig.suptitle(ch_type, fontsize=12)

    # Plot
    for nseq in range(7):
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(nseq + 1)
        mean = allseq_mean[nseq]
        ub = allseq_ub[nseq]
        lb = allseq_lb[nseq]
        ax[nseq].set_title(seqname, loc='left', weight='bold', fontsize=12)
        ax[nseq].fill_between(times, ub, lb, color='black', alpha=.2)
        ax[nseq].plot(times, mean, color=ch_colors[ch_type], linewidth=1.5, label=seqname)
        ax[nseq].axvline(0, linestyle='-', color='black', linewidth=2)
        ax[nseq].set_xlim([min(times), max(times)])
        # Add vertical lines
        for xx in range(16):
            ax[nseq].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
        # Remove spines
        for key in ('top', 'right', 'bottom'):
            ax[nseq].spines[key].set(visible=False)
        ax[nseq].set_ylabel('GFP (' + units[ch_type] + ')')
        ax[nseq].set_xticks([], [])
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax[nseq].get_yaxis().set_major_formatter(fmt)
        if datatype == 'items':
            ax[nseq].get_yaxis().get_offset_text().set_position((-0.22, 0))  # move 'x10^x', does not work with y
        elif datatype == 'fullseq':
            ax[nseq].get_yaxis().get_offset_text().set_position((-0.07, 0))  # move 'x10^x', does not work with y
    ax[nseq].set_xlabel('Time (ms)')
    if datatype == 'items':
        ax[nseq].set_xticks(range(0, 800, 200), [])
    elif datatype == 'fullseq':
        ax[nseq].set_xticks(range(-500, 4500, 500), [])
        # Add "xY" using the same yval for all
        ylim = ax[nseq].get_ylim()
        yval = ylim[1] - ylim[1] * 0.1
        for nseq in range(7):
            seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(nseq + 1)
            for xx in range(16):
                ax[nseq].text(250 * (xx + 1) - 125, yval, seqtxtXY[xx], horizontalalignment='center', fontsize=12)

    return fig
