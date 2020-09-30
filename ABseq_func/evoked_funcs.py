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
import copy

def plot_butterfly_items(epochs_items, subject, ylim_eeg=10, ylim_mag=300, ylim_grad=100, times="peaks", violation_or_not=1):
    # Figures folder
    if violation_or_not:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyViolation_Items', subject)
    else:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyStandard_Items', subject)
        if ylim_eeg == 10:  # i.e., if default value
            ylim_eeg = 3
            ylim_mag = 150
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


def plot_butterfly_items_allsubj(evoked, times="peaks", violation_or_not=1, residevoked=False):

    # Figures folder
    if violation_or_not:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyViolation_Items', 'GROUP')
        ylim_eeg = 4
        ylim_mag = 200
        ylim_grad = 70
    else:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyStandard_Items', 'GROUP')
        ylim_eeg = 1.5
        ylim_mag = 70
        ylim_grad = 20
    if residevoked:
        fig_path = fig_path + op.sep + 'residevoked'
    utils.create_folder(fig_path)

    # Butterfly plots for violations (one graph per sequence) - in EEG/MAG/GRAD
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
    ts_args = dict(gfp=True, time_unit='ms', ylim=ylim)
    topomap_args = dict(time_unit='ms', ylim=ylim)
    topotimes = np.arange(0.070, 0.501, 0.050)
    topo_avg = 0.020


    for x, seq in enumerate(evoked.keys()):
        # Sequence info
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(x+1)

        evokeds_seq = evoked[seq]
        grand_avg_seq = mne.grand_average([evokeds_seq[i][0] for i in range(len(evokeds_seq))])

        # EEG
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        grand_avg_seq.plot(axes=axes, picks='eeg', gfp=False, spatial_colors=True, show=True, window_title=seqname, time_unit='ms', ylim=ylim)
        axes.texts = []
        axes.set_title('EEG: '+seqname)
        # Remove spines
        for key in ('top', 'right'):
            axes.spines[key].set(visible=False)
        # Correct the small topomap
        tmp = fig.get_axes()
        smallfigax = tmp[1]
        smallfigax.set_aspect('equal')
        fig_name = fig_path + op.sep + ('EEG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # topomaps
        fig = grand_avg_seq.plot_topomap(topotimes, ch_type='eeg', time_unit='ms', title='EEG: '+seqname, vmin=-ylim_eeg*0.6, vmax=ylim_eeg*0.6, average=topo_avg)
        fig_name = fig_path + op.sep + ('EEG_SequenceID_' + str(x + 1) + '_TOPOS.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # MAG
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        grand_avg_seq.plot(axes=axes, picks='mag', gfp=False, spatial_colors=True, show=True, window_title=seqname, time_unit='ms', ylim=ylim)
        axes.texts = []
        axes.set_title('MAG: ' + seqname)
        # Remove spines
        for key in ('top', 'right'):
            axes.spines[key].set(visible=False)
        # Correct the small topomap
        tmp = fig.get_axes()
        smallfigax = tmp[1]
        smallfigax.set_aspect('equal')
        # Save
        fig_name = fig_path + op.sep + ('MAG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # topomaps
        fig = grand_avg_seq.plot_topomap(topotimes, ch_type='mag', time_unit='ms', title='MAG: '+seqname, vmin=-ylim_mag*0.6, vmax=ylim_mag*0.6, average=topo_avg)
        fig_name = fig_path + op.sep + ('MAG_SequenceID_' + str(x + 1) + '_TOPOS.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # GRAD
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        grand_avg_seq.plot(axes=axes, picks='grad', gfp=False, spatial_colors=True, show=True, window_title=seqname, time_unit='ms', ylim=ylim)
        axes.texts = []
        axes.set_title('GRAD: ' + seqname)
        # Remove spines
        for key in ('top', 'right'):
            axes.spines[key].set(visible=False)
        # Correct the small topomap
        tmp = fig.get_axes()
        smallfigax = tmp[1]
        smallfigax.set_aspect('equal')
        # Save
        fig_name = fig_path + op.sep + ('GRAD_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # topomaps
        fig = grand_avg_seq.plot_topomap(topotimes, ch_type='grad', time_unit='ms', title='GRAD: '+seqname, vmin=-ylim_grad*0.6, vmax=ylim_grad*0.6, average=topo_avg)
        fig_name = fig_path + op.sep + ('GRAD_SequenceID_' + str(x + 1) + '_TOPOS.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Previous version
        # # EEG
        # fig = grand_avg_seq.plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
        #                                     topomap_args=topomap_args, picks='eeg', times=times, show=False)
        # fig_name = fig_path + op.sep + ('EEG_SequenceID_' + str(x + 1) + '.png')
        # print('Saving ' + fig_name)
        # plt.savefig(fig_name)
        # plt.close(fig)
        #
        # # MAG
        # fig = grand_avg_seq.plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
        #                                     topomap_args=topomap_args, picks='mag', times=times, show=False)
        # fig_name = fig_path + op.sep + ('MAG_SequenceID_' + str(x + 1) + '.png')
        # print('Saving ' + fig_name)
        # plt.savefig(fig_name)
        # plt.close(fig)
        #
        # # #GRAD
        # fig = grand_avg_seq.plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1),
        #                                     topomap_args=topomap_args, picks='grad', times=times, show=False)
        # fig_name = fig_path + op.sep + ('GRAD_SequenceID_' + str(x + 1) + '.png')
        # print('Saving ' + fig_name)
        # plt.savefig(fig_name)
        # plt.close(fig)


def plot_butterfly_items_allsubj_allseq(evoked, times="peaks", violation_or_not=1, residevoked=False):

    # Grand average
    tmp = list(evoked.keys())
    evoked_seq = evoked[tmp[0]]
    grand_avg_seq = mne.grand_average([evoked_seq[i][0] for i in range(len(evoked_seq))])

    # Figures folder & Ylim
    if violation_or_not:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyViolation_Items', 'GROUP')
        ylim_eeg = 2.0
        ylim_mag = 130
        ylim_grad = 40
    else:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyStandard_Items', 'GROUP')
        ylim_eeg = 1.5
        ylim_mag = 70
        ylim_grad = 20
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
    if residevoked:
        fig_path = fig_path + op.sep + 'residevoked'
    utils.create_folder(fig_path)

    for ch_type in ['eeg', 'mag', 'grad']:
        if ch_type == 'eeg':
            ymax = ylim_eeg
            ymin = -ymax
        elif ch_type == 'mag':
            ymax = ylim_mag
            ymin = -ymax
        elif ch_type == 'grad':
            ymax = ylim_grad
            ymin = -ymax
        time_peak1 = grand_avg_seq.get_peak(ch_type=ch_type, tmin=0.000, tmax=0.100, mode='abs')[1]
        time_peak2 = grand_avg_seq.get_peak(ch_type=ch_type, tmin=0.100, tmax=0.200, mode='abs')[1]
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        grand_avg_seq.plot(axes=axes, picks=ch_type, gfp=False, spatial_colors=True, show=True, time_unit='ms', ylim=ylim)
        axes.texts = []
        axes.set_title(ch_type)
        peak_line = axes.axvline(x=time_peak1*1000, ymin=0.90, ymax=1, color='#707070', ls='-', linewidth=3)
        peak_line = axes.axvline(x=time_peak2*1000, ymin=0.90, ymax=1, color='#707070', ls='-', linewidth=3)
        # Remove spines
        for key in ('top', 'right'):
            axes.spines[key].set(visible=False)
        # Correct the small topomap
        tmp = fig.get_axes()
        smallfigax = tmp[1]
        smallfigax.set_aspect('equal')
        fig_name = fig_path + op.sep + (ch_type + '_AllSeq_.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # topomaps
        fig = grand_avg_seq.plot_topomap([time_peak1, time_peak2], ch_type=ch_type, time_unit='ms', title=ch_type+ '_AllSeq', vmin=ymin*0.6, vmax=ymax*0.6, average=0.010)
        fig_name = fig_path + op.sep + (ch_type+'_AllSeq_TOPOS.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)


def plot_butterfly_fullseq_allsubj(evoked, ylim_eeg=10, ylim_mag=300, ylim_grad=100, times="peaks", violation_or_not=0):

    # Figures folder & lims
    if violation_or_not:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyViolation_FullSequence', 'GROUP')
    else:
        fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'ButterflyStandard_FullSequence', 'GROUP')
        if ylim_eeg == 10:  # i.e., if default value
            ylim_eeg = 3
            ylim_mag = 150
            ylim_grad = 40
    utils.create_folder(fig_path)

    # Butterfly plots for violations (one graph per sequence) - in EEG/MAG/GRAD
    ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])

    for x, seq in enumerate(evoked.keys()):
        # Sequence info
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(x+1)

        evokeds_seq = evoked[seq]
        grand_avg_seq = mne.grand_average([evokeds_seq[i][0] for i in range(len(evokeds_seq))])

        # EEG
        # fig = grand_avg_seq.plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1), topomap_args=topomap_args, picks='eeg', times=times, show=False)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 2))
        grand_avg_seq.plot(axes=axes, picks='eeg', gfp=False, spatial_colors=True, show=True, window_title=seqname, time_unit='ms', ylim=ylim)
        axes.texts = []
        axes.set_title('EEG: '+seqname)
        # Remove spines
        for key in ('top', 'right'):
            axes.spines[key].set(visible=False)
        # Correct the small topomap
        tmp = fig.get_axes()
        smallfigax = tmp[1]
        smallfigax.set_aspect('equal')
        fig_name = fig_path + op.sep + ('EEG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # MAG
        # fig = grand_avg_seq.plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1), topomap_args=topomap_args, picks='mag', times=times, show=False)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 2))
        grand_avg_seq.plot(axes=axes, picks='mag', gfp=False, spatial_colors=True, show=True, window_title=seqname, time_unit='ms', ylim=ylim)
        axes.texts = []
        axes.set_title('MAG: ' + seqname)
        # Remove spines
        for key in ('top', 'right'):
            axes.spines[key].set(visible=False)
        # Correct the small topomap
        tmp = fig.get_axes()
        smallfigax = tmp[1]
        smallfigax.set_aspect('equal')
        # Save
        fig_name = fig_path + op.sep + ('MAG_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # #GRAD
        # fig = grand_avg_seq.plot_joint(ts_args=ts_args, title='SequenceID_' + str(x + 1), topomap_args=topomap_args, picks='grad', times=times, show=False)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 2))
        grand_avg_seq.plot(axes=axes, picks='grad', gfp=False, spatial_colors=True, show=True, window_title=seqname, time_unit='ms', ylim=ylim)
        axes.texts = []
        axes.set_title('GRAD: ' + seqname)
        # Remove spines
        for key in ('top', 'right'):
            axes.spines[key].set(visible=False)
        # Correct the small topomap
        tmp = fig.get_axes()
        smallfigax = tmp[1]
        smallfigax.set_aspect('equal')
        # Save
        fig_name = fig_path + op.sep + ('GRAD_SequenceID_' + str(x + 1) + '.png')
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
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


def create_evoked_resid(subject, resid_epochs_type='reg_repeataltern_surpriseOmegainfinity'):
    """
    IDENTICAL TO create_evoked BUT TAKES RESIDUALS OF A REGRESSION INSTEAD OF ORIGINAL EPOCHS

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
    path_evo = op.join(config.meg_dir, subject, 'evoked_resid')
    utils.create_folder(path_evo)


    # # =======================================================
    # # ========== evoked on full 16-items seq ================
    # # =======================================================
    # epochs_full_sequence = epoching_funcs.load_epochs_full_sequence(subject, cleaned=cleaned)

    # epochs_full_sequence['ViolationInSequence == "0"'].average().save(op.join(path_evo, 'full_seq_standard_all-ave.fif'))
    # epochs_full_sequence['ViolationInSequence == "0" and TrialNumber > 11'].average().save(op.join(path_evo, 'full_seq_teststandard_all-ave.fif'))
    # epochs_full_sequence['ViolationInSequence == "0" and TrialNumber < 11'].average().save(op.join(path_evo, 'full_seq_habituation_all-ave.fif'))
    #
    # for k in range(1, 8):
    #     epochs_full_sequence['SequenceID == "%i" and ViolationInSequence == "0"' % k].average().save(op.join(path_evo, 'full_seq_standard_seq%i-ave.fif' % k))
    #     epochs_full_sequence['SequenceID == "%i" and ViolationInSequence == "0" and TrialNumber > 11' % k].average().save(op.join(path_evo, 'full_seq_teststandard_seq%i-ave.fif' % k))
    #     epochs_full_sequence['SequenceID == "%i" and ViolationInSequence == "0" and TrialNumber < 11' % k].average().save(op.join(path_evo, 'full_seq_habituation_seq%i-ave.fif' % k))
    #     # determine the position of the deviants
    #     tmp = epochs_full_sequence['SequenceID == "%i" and ViolationInSequence > 0' % k]
    #     devpos = np.unique(tmp.metadata.ViolationInSequence)
    #     for pos_viol in devpos:
    #         epochs_full_sequence['SequenceID == "%i" and  ViolationInSequence == "%i"' % (k, int(pos_viol))].average().save(op.join(path_evo, 'full_seq_viol_seq%i_pos%i-ave.fif' % (k, int(pos_viol))))
    # del epochs_full_sequence
    # # evoked on individual items

    # =======================================================
    # ========== evoked on each item separately =============
    # =======================================================
    resid_path = op.join(config.result_path, 'linear_models', resid_epochs_type, subject)
    fname_in = op.join(resid_path, 'residuals-epo.fif')
    epochs_items = mne.read_epochs(fname_in, preload=True)

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
        evoked_names = glob.glob(path_evo + op.sep + filter_name + '*.fif')
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

    return evoked_dict, path_evo


def load_evoked_resid(subject='all', filter_name='', filter_not=None,root_path=None):
    """
    IDENTICAL TO load_evoked BUT TAKES EVOKED FROM RESIDUALS OF A REGRESSION INSTEAD OF ORIGINAL EVOKED

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
            path_evo = op.join(config.meg_dir, subj, 'evoked_resid')
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
        path_evo = op.join(config.meg_dir, subject, 'evoked_resid')
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


def plot_evoked_with_sem_7seq(evoked_dict, ch_inds, ch_type='eeg', label=None, filter=True):

    evoked_dict_copy = copy.deepcopy(evoked_dict)

    NUM_COLORS = 7
    cm = plt.get_cmap('viridis')
    color_mean = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    # fig.suptitle('', fontsize=12)
    # fig_name_save = save_path + op.sep + '' + fig_name + '_' + ch_type + '.png'

    for nseq in range(7):
        cond = list(evoked_dict_copy.keys())[nseq]
        data = copy.deepcopy(evoked_dict)
        data = data[cond]
        times = data[0][0].times*1000

        group_data_seq = []
        for nn in range(len(data)):
            sub_data = data.copy()[nn][0]
            if ch_type == 'eeg':
                tmp = copy.deepcopy(sub_data)
                sub_data_ch = np.array(tmp.pick_types(meg=False, eeg=True)._data)
            else:
                tmp = copy.deepcopy(sub_data)
                sub_data_ch = np.array(tmp.pick_types(meg=ch_type, eeg=False)._data)
            if np.size(ch_inds) > 1:
                group_data_seq.append(sub_data_ch[ch_inds].mean(axis=0))
            else:
                group_data_seq.append(sub_data_ch[ch_inds])
        mean = np.mean(group_data_seq, axis=0)
        ub = mean + sem(group_data_seq, axis=0)
        lb = mean - sem(group_data_seq, axis=0)

        if filter == True:
            mean = savgol_filter(mean, 9, 3)
            ub = savgol_filter(ub, 9, 3)
            lb = savgol_filter(lb, 9, 3)

        # Sequence info
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(nseq+1)

        plt.fill_between(times, ub, lb, color=color_mean[nseq], alpha=.2)
        plt.plot(times, mean, color=color_mean[nseq], linewidth=1.5, label=seqname)
    plt.legend(loc='upper right', fontsize=9)
    # Remove spines
    for key in ('top', 'right'):
        ax.spines[key].set(visible=False)
    ax.set_xlim([-100, 750])
    # fig.savefig(fig_name_save, bbox_inches='tight', dpi=300)
    # plt.close('all')


def plot_evoked_with_sem_7seq_fullseq(evoked_dict, ch_inds, ch_type='eeg', label=None, filter=True):

    evoked_dict_copy = copy.deepcopy(evoked_dict)

    NUM_COLORS = 7
    cm = plt.get_cmap('viridis')
    color_mean = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])

    fig, ax = plt.subplots(1, 1, figsize=(18, 4))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    for xx in range(16):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    # fig.suptitle('', fontsize=12)
    # fig_name_save = save_path + op.sep + '' + fig_name + '_' + ch_type + '.png'

    for nseq in range(7):
        cond = list(evoked_dict_copy.keys())[nseq]
        data = copy.deepcopy(evoked_dict)
        data = data[cond]
        times = data[0][0].times*1000

        group_data_seq = []
        for nn in range(len(data)):
            sub_data = data.copy()[nn][0]
            if ch_type == 'eeg':
                tmp = copy.deepcopy(sub_data)
                sub_data_ch = np.array(tmp.pick_types(meg=False, eeg=True)._data)
            else:
                tmp = copy.deepcopy(sub_data)
                sub_data_ch = np.array(tmp.pick_types(meg=ch_type, eeg=False)._data)
            if np.size(ch_inds) > 1:
                group_data_seq.append(sub_data_ch[ch_inds].mean(axis=0))
            else:
                group_data_seq.append(sub_data_ch[ch_inds])
        mean = np.mean(group_data_seq, axis=0)
        ub = mean + sem(group_data_seq, axis=0)
        lb = mean - sem(group_data_seq, axis=0)

        if filter == True:
            mean = savgol_filter(mean, 9, 3)
            ub = savgol_filter(ub, 9, 3)
            lb = savgol_filter(lb, 9, 3)

        # Sequence info
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(nseq+1)

        plt.fill_between(times, ub, lb, color=color_mean[nseq], alpha=.2)
        plt.plot(times, mean, color=color_mean[nseq], linewidth=1.5, label=seqname)
    plt.legend(loc='best', fontsize=9)
    # Remove spines
    for key in ('top', 'right'):
        ax.spines[key].set(visible=False)
    ax.set_xlim([min(times), max(times)])
    # fig.savefig(fig_name_save, bbox_inches='tight', dpi=300)
    # plt.close('all')


def plot_evoked_heatmap_7seq_fullseq(evoked_dict, ch_inds, ch_type='eeg', cmap_style='bilateral', filter=True):

    evoked_dict_copy = copy.deepcopy(evoked_dict)

    # Additional parameters
    cmap_rescale_ratio = 0.2  # 'saturate' the colormap, min/max will be reduced with this ratio
    units = dict(eeg='uV', grad='fT/cm', mag='fT')

    # Create group average (of ch_inds) per sequence
    allseq_mean = []
    for nseq in range(7):
        cond = list(evoked_dict_copy.keys())[nseq]
        data = copy.deepcopy(evoked_dict)
        data = data[cond]
        times = data[0][0].times*1000

        group_data_seq = []
        for nn in range(len(data)):
            sub_data = data.copy()[nn][0]
            if ch_type == 'eeg':
                tmp = copy.deepcopy(sub_data)
                sub_data_ch = np.array(tmp.pick_types(meg=False, eeg=True)._data)
            else:
                tmp = copy.deepcopy(sub_data)
                sub_data_ch = np.array(tmp.pick_types(meg=ch_type, eeg=False)._data)
            if np.size(ch_inds) > 1:
                group_data_seq.append(sub_data_ch[ch_inds].mean(axis=0))
            else:
                group_data_seq.append(sub_data_ch[ch_inds])
        mean = np.mean(group_data_seq, axis=0)
        if filter == True:
            mean = savgol_filter(mean, 9, 3)
        allseq_mean.append(mean)

    # Get max absolute value of the whole dataset to use it as vmin/vmax
    maxlist = [max(allseq_mean[ii]) for ii in range(len(allseq_mean))]
    minlist = [min(allseq_mean[ii]) for ii in range(len(allseq_mean))]
    maxabslist = [abs(max(allseq_mean[ii], key=abs)) for ii in range(len(allseq_mean))]

    # colormap
    if cmap_style == 'unilateral':
        cmap = 'viridis'
        vmin = min(minlist)
        vmax = max(maxlist) - max(maxlist) * cmap_rescale_ratio
    else:
        cmap = 'RdBu_r'
        vmin = -max(maxabslist) + max(maxabslist) * cmap_rescale_ratio
        vmax = max(maxabslist) - max(maxabslist) * cmap_rescale_ratio

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(18, 4))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    for xx in range(16):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax.set_xlabel('Time (ms)')
    # fig.suptitle('', fontsize=12)

    # Plot im
    width = 75
    im = ax.imshow(allseq_mean, extent=[min(times), max(times), 0, len(allseq_mean) * width], cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    ax.set_yticks(np.arange(width / 2, len(allseq_mean) * width, width))

    # Sequence info
    seqnames = []
    for seqID in range(7, 0, -1):
        seqname, _, _ = epoching_funcs.get_seqInfo(seqID)
        seqnames.append(seqname)
    ax.set_yticklabels(seqnames)

    # add colorbar
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cb = fig.colorbar(im, ax=ax, format=fmt, shrink=.40, aspect=10, pad=.012)
    cb.ax.yaxis.set_offset_position('left')
    cb.set_label(units[ch_type])


def plot_evoked_timecourse_7seq_fullseq(evoked_dict, ch_inds, ch_type='eeg', filter=True):

    evoked_dict_copy = copy.deepcopy(evoked_dict)

    # Additional parameters
    units = dict(eeg='uV', grad='fT/cm', mag='fT')
    scalings = dict(mag=1e-12, grad=4e-11, eeg=20e-6)
    ch_colors = dict(eeg='green', grad='red', mag='blue')

    # Create group average (of ch_inds) per sequence
    allseq_mean = []
    allseq_ub = []
    allseq_lb = []
    for nseq in range(7):
        cond = list(evoked_dict_copy.keys())[nseq]
        data = copy.deepcopy(evoked_dict)
        data = data[cond]
        times = data[0][0].times*1000

        group_data_seq = []
        for nn in range(len(data)):
            sub_data = data.copy()[nn][0]
            if ch_type == 'eeg':
                tmp = copy.deepcopy(sub_data)
                sub_data_ch = np.array(tmp.pick_types(meg=False, eeg=True)._data)
            else:
                tmp = copy.deepcopy(sub_data)
                sub_data_ch = np.array(tmp.pick_types(meg=ch_type, eeg=False)._data)
            if np.size(ch_inds) > 1:
                group_data_seq.append(sub_data_ch[ch_inds].mean(axis=0))
            else:
                group_data_seq.append(sub_data_ch[ch_inds])
        mean = np.mean(group_data_seq, axis=0)
        ub = mean + sem(group_data_seq, axis=0)
        lb = mean - sem(group_data_seq, axis=0)

        if filter == True:
            mean = savgol_filter(mean, 9, 3)
            ub = savgol_filter(ub, 9, 3)
            lb = savgol_filter(lb, 9, 3)
        allseq_mean.append(mean)
        allseq_ub.append(ub)
        allseq_lb.append(lb)

    # Create figure
    fig, ax = plt.subplots(7, 1, figsize=(9, 9), sharex=False, sharey=True, constrained_layout=True)
    fig.suptitle(ch_type, fontsize=12)

    # Plot
    for nseq in range(7):
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(nseq + 1)
        mean = allseq_mean[nseq]
        ub = allseq_ub[nseq]
        lb = allseq_lb[nseq]
        ax[nseq].set_title(seqname, loc='left', weight='bold', fontsize=12)
        ax[nseq].fill_between(times, ub/scalings[ch_type], lb/scalings[ch_type], color='black', alpha=.2)
        ax[nseq].plot(times, mean/scalings[ch_type], color=ch_colors[ch_type], linewidth=1.5, label=seqname)
        ax[nseq].axvline(0, linestyle='-', color='black', linewidth=2)
        ax[nseq].set_xlim([min(times), max(times)])
        # Add vertical lines
        for xx in range(16):
            ax[nseq].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
        # Remove spines
        for key in ('top', 'right', 'bottom'):
            ax[nseq].spines[key].set(visible=False)
        ax[nseq].set_ylabel(units[ch_type])
        ax[nseq].set_xticks([], [])
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        ax[nseq].get_yaxis().set_major_formatter(fmt)
        ax[nseq].get_yaxis().get_offset_text().set_position((-0.07, 0))  # move 'x10-x', does not work with y
    ax[nseq].set_xticks(range(-500, 4500, 500), [])
    ax[nseq].set_xlabel('Time (ms)')
    # Add "xY" using the same yval for all
    ylim = ax[nseq].get_ylim()
    yval = ylim[1] - ylim[1]*0.1
    for nseq in range(7):
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(nseq + 1)
        for xx in range(16):
            ax[nseq].text(250 * (xx + 1) - 125, yval, seqtxtXY[xx], horizontalalignment='center', fontsize=12)


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
        if np.size(ch_inds) > 1:
            group_data_seq.append(sub_data[ch_inds].mean(axis=0))
        else:
            group_data_seq.append(sub_data[ch_inds])

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
            maxabslist.append(abs(max(data_to_plot[key1][key2], key=abs)))

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
