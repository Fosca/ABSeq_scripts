import config
import os
import os.path as op
import mne
import matplotlib.pyplot as plt
import numpy as np
from ABseq_func import *
from ABseq_func import utils   # why do we need this now ?? (error otherwise)
from scipy.signal import savgol_filter
from scipy.stats import sem
import matplotlib.ticker as ticker

def plot_butterfly_items(epochs_items, subject, ylim_eeg=10, ylim_mag=300, ylim_grad=100, times="peaks",
                         violation_or_not=1):
    # Figures folder
    if violation_or_not:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyViolation_Items', subject)
    else:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyStandard_Items', subject)
        if ylim_eeg == 10:  # i.e., if default value
            ylim_eeg = 3;
            ylim_mag = 150;
            ylim_grad = 40

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Make evoked - deviants OR standards for each sequence
    evokeds_per_seq = []
    for x in range(7):
        evokeds_per_seq.append(epochs_items['SequenceID == "' + str(x + 1) +
                                            '" and ViolationOrNot == "%i"' % violation_or_not].average())

    # Butterfly plots for violations (one graph per sequence) - in EEG/MAG/GRAD
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
    ts_args = dict(gfp=True, time_unit='s', ylim=ylim)
    topomap_args = dict(time_unit='s')

    for x in range(7):
        # EEG
        fig = evokeds_per_seq[x].plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
                                            topomap_args=topomap_args, picks='eeg', times=times, show=False)
        fig_name = fig_path + op.sep + ('EEG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

        # MAG
        fig = evokeds_per_seq[x].plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
                                            topomap_args=topomap_args, picks='mag', times=times, show=False)
        fig_name = fig_path + op.sep + ('MAG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

        # #GRAD
        fig = evokeds_per_seq[x].plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
                                            topomap_args=topomap_args, picks='grad', times=times, show=False)
        fig_name = fig_path + op.sep + ('GRAD_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)




def plot_butterfly_items_allsubj(evoked, ylim_eeg=10, ylim_mag=300, ylim_grad=100, times="peaks",
                         violation_or_not=1):

    # Figures folder
    if violation_or_not:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyViolation_Items', 'GROUP')
    else:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyStandard_Items', 'GROUP')
        if ylim_eeg == 10:  # i.e., if default value
            ylim_eeg = 3;
            ylim_mag = 150;
            ylim_grad = 40

    # Butterfly plots for violations (one graph per sequence) - in EEG/MAG/GRAD
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
    ts_args = dict(gfp=True, time_unit='s', ylim=ylim)
    topomap_args = dict(time_unit='s')

    for x, seq in enumerate(evoked.keys()):
        evokeds_seq = evoked[seq]
        grand_avg_seq = mne.grand_average([evokeds_seq[i][0] for i in range(len(evokeds_seq))])

        fig = grand_avg_seq.plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
                                            topomap_args=topomap_args, picks='eeg', times=times, show=False)
        fig_name = fig_path + op.sep + ('EEG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

        # MAG
        fig = grand_avg_seq.plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
                                            topomap_args=topomap_args, picks='mag', times=times, show=False)
        fig_name = fig_path + op.sep + ('MAG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

        # #GRAD
        fig = grand_avg_seq.plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
                                            topomap_args=topomap_args, picks='grad', times=times, show=False)
        fig_name = fig_path + op.sep + ('GRAD_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)



def plot_butterfly_first_item(epochs_first_item, subject, ylim_eeg=10, ylim_mag=300, ylim_grad=100, times="peaks"):
    # Figures folder
    fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyStandard_FullSequence', subject)
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)

    # Make evoked - not-violated - for each sequence
    evokeds_per_seq = []
    for x in range(7):
        evokeds_per_seq.append(epochs_first_item['SequenceID == "' + str(x + 1) +
                                                 '" and ViolationInSequence == "0"'].average())

    # Butterfly plots for violations (one graph per sequence) - in EEG/MAG/GRAD
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
    ts_args = dict(gfp=True, time_unit='s', ylim=ylim)
    topomap_args = dict(time_unit='s')

    for x in range(7):
        # EEG
        fig = evokeds_per_seq[x].plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
                                            topomap_args=topomap_args, picks='eeg', times=times, show=False)
        fig_name = fig_path + op.sep + ('EEG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

        # MAG
        fig = evokeds_per_seq[x].plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
                                            topomap_args=topomap_args, picks='mag', times=times, show=False)
        fig_name = fig_path + op.sep + ('MAG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)

        # #GRAD
        fig = evokeds_per_seq[x].plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
                                            topomap_args=topomap_args, picks='grad', times=times, show=False)
        fig_name = fig_path + op.sep + ('GRAD_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name)
        plt.close(fig)


def create_evoked(subject, cleaned=True):
    """
    This function creates several types of evoked (epochs averages, "-ave.fif") for a given subject:
    - Full 16-items sequences [full_seq]:
        - Habituation sequences [_habituation]: all sequencesIDs pooled together [_all] + 1 for each sequenceID [_seqN]
        - TestStandard sequences [_teststandard] (non violated, within test blocks): all sequencesIDs pooled together [_all]  + 1 for each sequenceID [_seqN]
        - Standard sequences [_standard] (all non violated, whether within habituation or test blocks): all sequencesIDs pooled together [_all] + 1 for each sequenceID [_seqN]
        - Violated sequences [_viol]: one for each sequenceID & position (7 x 4) [_seqN_posN]
    - Sequence individual items [items]:
        - Habituation items [_habituation]: all sequencesIDs pooled together [_all] + 1 for each sequenceID [_seqN]
        - TestStandard items [_teststandard] (non violated, within test blocks): all sequencesIDs pooled together [_all]  + 1 for each sequenceID [_seqN]
        - Standard items [_standard] (all non violated, whether within habituation or test blocks): all sequencesIDs pooled together [_all] + 1 for each sequenceID [_seqN]
        - Standard items balanced [_standard_balanced] (non violated, whether within habituation or test blocks, matched in position with violations): all sequencesIDs pooled together [_all] + 1 for each sequenceID [_seqN]
        - Violated sequences [_viol]: one for each sequenceID [_seqN] and one for each sequenceID & position (7 x 4) [_seqN_posN]
        - ?? Standard items violpositions [_standard] (non violated items from violated sequences with matched postions): for each sequenceID & position (7 x 4) [_seqN_posN] ??
    """
    # create folder for evoked
    path_evo = op.join(config.meg_dir, subject, 'evoked')
    if cleaned:
        path_evo = path_evo+'_cleaned'
    utils.create_folder(path_evo)

    epochs_full_sequence = epoching_funcs.load_epochs_full_sequence(subject, cleaned=cleaned)

    # =======================================================
    # ========== evoked on full 16-items seq ================
    # =======================================================

    epochs_full_sequence['ViolationInSequence == "0"'].average().save(op.join(path_evo, 'full_seq_standard_all-ave.fif'))
    epochs_full_sequence['ViolationInSequence == "0" and TrialNumber > 11'].average().save(op.join(path_evo, 'full_seq_teststandard_all-ave.fif'))
    epochs_full_sequence['ViolationInSequence == "0" and TrialNumber < 11'].average().save(op.join(path_evo, 'full_seq_habituation_all-ave.fif'))

    for k in range(1, 8):
        epochs_full_sequence['SequenceID == "%i" and ViolationInSequence == "0"' % k].average().save(op.join(path_evo, 'full_seq_standard_seq%i-ave.fif' % k))
        epochs_full_sequence['SequenceID == "%i" and ViolationInSequence == "0" and TrialNumber > 11' % k].average().save(op.join(path_evo, 'full_seq_teststandard_seq%i-ave.fif' % k))
        epochs_full_sequence['SequenceID == "%i" and ViolationInSequence == "0" and TrialNumber < 11' % k].average().save(op.join(path_evo, 'full_seq_habituation_seq%i-ave.fif' % k))
        # determine the position of the deviants
        tmp = epochs_full_sequence['SequenceID == "%i" and ViolationInSequence > 0' % k]
        devpos = np.unique(tmp.metadata.ViolationInSequence)
        for pos_viol in devpos:
            epochs_full_sequence['SequenceID == "%i" and  ViolationInSequence == "%i"' % (k, int(pos_viol))].average().save(op.join(path_evo, 'full_seq_viol_seq%i_pos%i-ave.fif' % (k, int(pos_viol))))
    del epochs_full_sequence
    # evoked on individual items

    # =======================================================
    # ========== evoked on each item separately =============
    # =======================================================

    epochs_items = epoching_funcs.load_epochs_items(subject)
    epochs_items['ViolationInSequence == "0"'].average().save(op.join(path_evo, 'items_standard_all-ave.fif'))
    epochs_items['ViolationInSequence == "0" and TrialNumber > 11'].average().save(op.join(path_evo, 'items_teststandard_all-ave.fif'))
    epochs_items['ViolationInSequence == "0" and TrialNumber < 11'].average().save(op.join(path_evo, 'items_habituation_all-ave.fif'))
    epochs_items['ViolationOrNot == "1"'].average().save(op.join(path_evo, 'items_viol_all-ave.fif'))
    epochs_balanced = epoching_funcs.balance_epochs_violation_positions(epochs_items)
    epochs_balanced['ViolationInSequence == "0"'].average().save(op.join(path_evo, 'items_standard_balanced_all-ave.fif'))

    for k in range(1, 8):
        epochs_items['SequenceID == "%i" and ViolationInSequence == "0"' % k].average().save(op.join(path_evo, 'items_standard_seq%i-ave.fif' % k))
        epochs_items['SequenceID == "%i" and ViolationInSequence == "0" and TrialNumber > 11' % k].average().save(op.join(path_evo, 'items_teststandard_seq%i-ave.fif' % k))
        epochs_items['SequenceID == "%i" and ViolationInSequence == "0" and TrialNumber < 11' % k].average().save(op.join(path_evo, 'items_habituation_seq%i-ave.fif' % k))
        epochs_items['SequenceID == "%i" and ViolationOrNot == "1"' % k].average().save(op.join(path_evo, 'items_viol_seq%i-ave.fif' % k))
        epochs_balanced['SequenceID == "%i" and ViolationOrNot == "0"' % k].average().save(op.join(path_evo, 'items_standard_balanced_seq%i-ave.fif' % k))
        # determine the position of the deviants
        tmp = epochs_items['SequenceID == "%i" and ViolationInSequence > 0' % k]
        devpos = np.unique(tmp.metadata.ViolationInSequence)
        for pos_viol in devpos:
            epochs_items['SequenceID == "%i" and  ViolationInSequence == "%i" and ViolationOrNot == "1"' % (k, pos_viol)].average().save(op.join(path_evo, 'items_viol_seq%i_pos%i-ave.fif' % (k, int(pos_viol))))
            epochs_items['SequenceID == "%i" and  ViolationInSequence == "%i" and ViolationOrNot == "0"' % (k, pos_viol)].average().save(op.join(path_evo, 'items_standard_seq%i_pos%i-ave.fif' % (k, int(pos_viol))))

    del epochs_items


def create_evoked_for_regression_factors(regressor_names, subject, cleaned=True):

    # This will create one evoked fif for each (unique) level of each regressor
    # of regressor_names, e.g., one evoked for ChunkSize=1, one for ChunkSize=2, etc.

    # create folder for evoked
    path_evo = op.join(config.meg_dir, subject, 'evoked')
    if cleaned:
        path_evo = path_evo+'_cleaned'
    utils.create_folder(path_evo)

    # ========================================= #
    # ========== evoked for items ============= #
    # ========================================= #

    # Load data & update metadata (in case new things were added)
    if cleaned:
        epochs_items = epoching_funcs.load_epochs_items(subject, cleaned=True)
        epochs_items = epoching_funcs.update_metadata_rejected(subject, epochs_items)
    else:
        epochs_items = epoching_funcs.load_epochs_items(subject, cleaned=False)
        epochs_items = epoching_funcs.update_metadata_rejected(subject, epochs_items)

    # ====== remove some items (excluded from the linear model) ====== #
    print('We remove the first sequence item for which the surprise is not well computed and for which there is no RepeatAlter')
    epochs_items = epochs_items["StimPosition > 1"]
    # print('We remove items from trials with violation')
    # epochs_items = epochs_items["ViolationInSequence == 0"]

    # ===== create evoked for each level of each regressor_name ===== #
    for regressor_name in regressor_names:
        levels = np.unique(epochs_items.metadata[regressor_name])
        for x, level in enumerate(levels):
            epochs_items[regressor_name + ' == "' + str(level) + '"'].average().save(op.join(path_evo, regressor_name + '_' + str(level) + '-ave.fif'))

    del epochs_items


def load_evoked(subject='all', filter_name='', filter_not=None,root_path=None, cleaned = True):
    """
    Cette fonction charge tous les evoques ayant un nom qui commence par filter_name et n'ayant pas filter_not dans leur nom.
    Elle cree un dictionnaire ayant pour champs les differentes conditions

    :param subject: 'all' si on veut charger les evoques de tous les participants de config.subject_list sinon mettre le nom du participant d'intereet
    :param filter_name:
    :param filter_not:
    :return:
    """
    import glob
    evoked_dict = {}
    if subject == 'all':
        for subj in config.subjects_list:
            if cleaned:
                path_evo = op.join(config.meg_dir, subj, 'evoked_cleaned')
            else:
                path_evo = op.join(config.meg_dir, subj, 'evoked')
            if root_path is not None:
                path_evo = op.join(root_path, subj)
            evoked_names = sorted(glob.glob(path_evo + op.sep + filter_name + '*'))
            file_names = []
            full_names = []
            for names in evoked_names:
                path, file = op.split(names)
                if filter_name in names:
                    if filter_not is not None:
                        if filter_not not in names:
                            file_names.append(file)
                            full_names.append(names)
                    else:
                        file_names.append(file)
                        full_names.append(names)

            for k in range(len(file_names)):
                print(file_names)
                if file_names[k][:-7] in evoked_dict.keys():
                    evoked_dict[file_names[k][:-7]].append(mne.read_evokeds(full_names[k]))
                else:
                    evoked_dict[file_names[k][:-7]] = [mne.read_evokeds(full_names[k])]
    else:
        if cleaned:
            path_evo = op.join(config.meg_dir, subject, 'evoked_cleaned')
        else:
            path_evo = op.join(config.meg_dir, subject, 'evoked')
        if root_path is not None:
            path_evo = op.join(root_path, subject)
        evoked_names = glob.glob(path_evo + op.sep + filter_name + '*')
        file_names = []
        full_names = []
        for names in evoked_names:
            path, file = op.split(names)
            if filter_name in names:
                if filter_not is not None:
                    if filter_not not in names:
                        file_names.append(file)
                        full_names.append(names)
                else:
                    file_names.append(file)
                    full_names.append(names)

        evoked_dict = {file_names[k][:-7]: mne.read_evokeds(full_names[k]) for k in range(len(file_names))}

    return evoked_dict


def plot_GFP_compare_evoked(evoked):
    """
    Fonction pour explorer les evoques de chaque participant individuellement
    :param evoked:
    :return:
    """
    NUM_COLORS = len(evoked.keys())
    cm = plt.get_cmap('viridis')
    colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
    mne.viz.plot_compare_evokeds(evoked, colors=colorslist)


def average_evoked(evoked_dict):
    evoked_dict_average = {key: [] for key in evoked_dict.keys()}

    for key in evoked_dict.keys():

        evoked_dict_average[key] = evoked_dict[key][0][0]

        avg_data = []
        for k in range(len(evoked_dict[key])):
            avg_data.append(evoked_dict[key][k][0]._data)

        avg_data = np.mean(avg_data, axis=0)

        evoked_dict_average[key].data = avg_data

    return evoked_dict_average


def plot_evoked_with_sem_7seq(evoked_dict, ch_inds, label=None, filter=True):

    NUM_COLORS = 7
    cm = plt.get_cmap('viridis')
    color_mean = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    # fig.suptitle('', fontsize=12)
    # fig_name_save = save_path + op.sep + '' + fig_name + '_' + ch_type + '.png'

    for nseq in range(7):
        cond = list(evoked_dict.keys())[nseq]
        data = evoked_dict[cond].copy()
        times = data[0][0].times*1000

        group_data_seq = []
        for nn in range(len(data)):
            sub_data = data[nn][0]
            sub_data = np.array(sub_data.pick_types(meg='mag', eeg=False)._data)
            group_data_seq.append(sub_data[ch_inds].mean(axis=0))

        mean = np.mean(group_data_seq, axis=0)
        ub = mean + sem(group_data_seq, axis=0)
        lb = mean - sem(group_data_seq, axis=0)

        if filter==True:
            mean = savgol_filter(mean, 11, 3)
            ub = savgol_filter(ub, 11, 3)
            lb = savgol_filter(lb, 11, 3)

        plt.fill_between(times, ub, lb, color=color_mean[nseq], alpha=.2)
        plt.plot(times, mean, color=color_mean[nseq], linewidth=1.5, label=cond)
    plt.legend(loc='upper right', fontsize=9)
    ax.set_xlim([-100, 750])
    # fig.savefig(fig_name_save, bbox_inches='tight', dpi=300)
    # plt.close('all')


def plot_evoked_with_sem_1cond(data, cond, ch_type, ch_inds, color=None, filter=True, axis=None):

    times = data[0][0].times * 1000

    group_data_seq = []
    for nn in range(len(data)):
        sub_data = data[nn][0].copy()
        if ch_type == 'eeg':
            sub_data = np.array(sub_data.pick_types(meg=False, eeg=True)._data)
        elif ch_type == 'mag':
            sub_data = np.array(sub_data.pick_types(meg='mag', eeg=False)._data)
        elif ch_type == 'grad':
            sub_data = np.array(sub_data.pick_types(meg='grad', eeg=False)._data)
        group_data_seq.append(sub_data[ch_inds].mean(axis=0))

    mean = np.mean(group_data_seq, axis=0)
    ub = mean + sem(group_data_seq, axis=0)
    lb = mean - sem(group_data_seq, axis=0)

    if filter == True:
        mean = savgol_filter(mean, 9, 3)
        ub = savgol_filter(ub, 9, 3)
        lb = savgol_filter(lb, 9, 3)

    if axis is not None:
        axis.fill_between(times, ub, lb, color=color, alpha=.2)
        axis.plot(times, mean, color=color, linewidth=1.5, label=cond)
    else:
        plt.fill_between(times, ub, lb, color=color, alpha=.2)
        plt.plot(times, mean, color=color, linewidth=1.5, label=cond)


def allsequences_heatmap_figure(data_to_plot, times, cmap_style='bilateral', fig_title='', file_name=None):

    """
    :param data_to_plot: dictionary with keys: 'hab', 'teststand', 'violpos1', 'violpos2', 'violpos3', 'violpos4'
                         each contains keys 'seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7'
                         each contains a data vector
    :param times: x values
    :param cmap_style: 'unilateral' scale uses min-max (& viridis), 'bilateral' uses -absolutemax & +abolutemax (& blue/white/red)
    :param fig_title:
    :param save_path:
    :return: figure
    """
    # Additional parameters
    cmap_rescale_ratio = 0.2  # 'saturate' the colormap, min/max will be reduced with this ratio

    # Create figure
    fig, axes = plt.subplots(7, 1, figsize=(12, 12), sharex=True, sharey=False, constrained_layout=True)
    fig.suptitle(fig_title, fontsize=12)
    ax = axes.ravel()[::1]

    # Get max absolute value of the whole dataset to use it as vmin/vmax
    maxlist = []
    minlist = []
    maxabslist = []
    for key1 in data_to_plot.keys():
        for key2 in data_to_plot[key1].keys():
            maxlist.append(max(data_to_plot[key1][key2]))
            minlist.append(min(data_to_plot[key1][key2]))
            maxabslist.append(max(data_to_plot[key1][key2], key=abs))

    # colormap
    if cmap_style == 'unilateral':
        cmap = 'viridis'
        vmin = min(minlist)
        vmax = max(maxlist) - max(maxlist) * cmap_rescale_ratio
    else:
        cmap = 'RdBu_r'
        vmin = -max(maxabslist) + max(maxabslist) * cmap_rescale_ratio
        vmax = max(maxabslist) - max(maxabslist) * cmap_rescale_ratio

    n = 0
    for seqID in range(1, 8):

        # Sequence info
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqID)

        # Subfig title
        ax[seqID-1].set_title(seqname, loc='left', weight='bold')

        # Data
        y_list = []
        y_list.append(data_to_plot['hab']['seq' + str(seqID)])
        y_list.append(data_to_plot['teststand']['seq' + str(seqID)])
        y_list.append(data_to_plot['violpos1']['seq' + str(seqID)])
        y_list.append(data_to_plot['violpos2']['seq' + str(seqID)])
        y_list.append(data_to_plot['violpos3']['seq' + str(seqID)])
        y_list.append(data_to_plot['violpos4']['seq' + str(seqID)])

        width = 75
        # Add vertical lines, and "xY"
        for xx in range(16):
            ax[n].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
            ax[n].text(250 * (xx + 1) - 125, width * 6 + (width / 3), seqtxtXY[xx], horizontalalignment='center', fontsize=16)
        im = ax[n].imshow(y_list, extent=[min(times) * 1000, max(times) * 1000, 0, 6 * width], cmap=cmap, vmin=vmin, vmax=vmax)
        # ax[n].set_xlim(-500, 4250)
        # ax[n].legend(loc='upper left', fontsize=10)
        ax[n].set_yticks(np.arange(width / 2, 6 * width, width))
        fig.canvas.draw()
        ax[n].set_yticklabels(['Violation (pos. %d)' % violation_positions[3], 'Violation (pos. %d)' % violation_positions[2],
                               'Violation (pos. %d)' % violation_positions[1], 'Violation (pos. %d)' % violation_positions[0],
                               'Standard', 'Habituation'])
        ax[n].axvline(0, linestyle='-', color='black', linewidth=2)

        # add deviant marks
        for k in range(4):
            viol_pos = violation_positions[k]
            x = 250 * (viol_pos - 1)
            y1 = (4 - k) * width
            y2 = (4 - 1 - k) * width
            ax[n].plot([x, x], [y1, y2], linestyle='-', color='black', linewidth=6)
            ax[n].plot([x, x], [y1, y2], linestyle='-', color='yellow', linewidth=3)
        # add colorbar
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        # cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.5, pad=.2, aspect=10)
        cb = fig.colorbar(im, ax=ax[n], location='right', format=fmt, shrink=.50, aspect=10, pad=.005)
        cb.ax.yaxis.set_offset_position('left')
        cb.set_label('a. u.')
        n += 1
    axes.ravel()[-1].set_xlabel('Time (ms)')

    figure = plt.gcf()
    if file_name is not None:
        print('Saving '+file_name)
        figure.savefig(file_name, bbox_inches='tight', dpi=300)
        plt.close('all')

    return figure
