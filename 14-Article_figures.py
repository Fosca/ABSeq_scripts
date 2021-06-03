import config
from ABseq_func import *
from importlib import reload
import os.path as op
import matplotlib.pyplot as plt
import mne
import numpy as np


# Exclude some subjects
config.exclude_subjects.append('sub10-gp_190568')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()


# =============================================================================== #
# ================================= Load data =================================== #
# =============================================================================== #
# -- items: Nsubjects X 7 sequences
evoked_standard_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_seq', filter_not='pos')
evoked_viol_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_seq', filter_not='pos')
evoked_balanced_standard_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_balanced_seq', filter_not='pos')
# -- full sequence: Nsubjects X 7 sequences
evoked_full_seq_standard_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq', filter_not='pos')
# -- items: Nsubjects X 1 sequence (all seq average)
evoked_all_standard, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_all', filter_not=None)
evoked_all_viol, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_all', filter_not=None)
evoked_all_standard_balanced, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_balanced_all', filter_not=None)

# =============================================================================== #
# ===================== Butterfly full sequences / each seq ===================== #
# =============================================================================== #

evoked_funcs.plot_butterfly_fullseq_allsubj(evoked_full_seq_standard_seq, violation_or_not=0)

# =============================================================================== #
# ========================= Butterfly items / each seq ========================== #
# =============================================================================== #
# -- Standards
evoked_funcs.plot_butterfly_items_allsubj(evoked_standard_seq, violation_or_not=0)
# -- Deviants
evoked_funcs.plot_butterfly_items_allsubj(evoked_viol_seq, violation_or_not=1)

# =============================================================================== #
# ====================== Butterfly items / all seq average ====================== #
# =============================================================================== #
# -- Standards
evoked_funcs.plot_butterfly_items_allsubj_allseq(evoked_all_standard, violation_or_not=0)
# -- Deviants
evoked_funcs.plot_butterfly_items_allsubj_allseq(evoked_all_viol, violation_or_not=1)

# =============================================================================== #
# ============================== One sensor plots =============================== #
# =============================================================================== #
fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'Evoked_sensor', 'GROUP')
utils.create_folder(fig_path)

# Grand average items AllSeq
evoked_all_stand = evoked_all_standard['items_standard_all-']
grand_avg_stand = mne.grand_average([evoked_all_stand[i][0] for i in range(len(evoked_all_stand))])
evoked_all_viol = evoked_all_viol['items_viol_all-']
grand_avg_viol = mne.grand_average([evoked_all_viol[i][0] for i in range(len(evoked_all_viol))])


# fig = grand_avg_stand.plot_topomap(times='auto', cmap='plasma')
# fig_name = fig_path + op.sep + 'test.png'
# print('Saving ' + fig_name)
# fig.savefig(fig_name, bbox_inches='tight', dpi=300)
# plt.close(fig)

for ch_type in ['eeg', 'grad', 'mag']:

    # Get peak sensor from grand average all_seq
    peak1 = grand_avg_stand.get_peak(ch_type=ch_type, tmin=0.000, tmax=0.100, mode='abs')
    peak2 = grand_avg_stand.get_peak(ch_type=ch_type, tmin=0.100, tmax=0.200, mode='abs')
    peak3 = grand_avg_viol.get_peak(ch_type=ch_type, tmin=0.000, tmax=0.100, mode='abs')
    peak4 = grand_avg_viol.get_peak(ch_type=ch_type, tmin=0.100, tmax=0.200, mode='abs')

    for peak in [peak1, peak2, peak3, peak4]:

        if ch_type == 'eeg':
            grand_avg_stand_ch = grand_avg_stand.copy().pick_types(eeg=True, meg=False)
            grand_avg_viol_ch = grand_avg_viol.copy().pick_types(eeg=True, meg=False)
        else:
            grand_avg_stand_ch = grand_avg_stand.copy().pick_types(eeg=False, meg=ch_type)
            grand_avg_viol_ch = grand_avg_viol.copy().pick_types(eeg=False, meg=ch_type)

        # Peak sensor index (within the current ch_type only)
        ch_ind = grand_avg_stand_ch.ch_names.index(peak[0])

        # # Find sensor with largest variance??
        # tmp = np.asarray(grand_avg_seq_ch._data)
        # ch_ind = np.argmax(np.var(tmp, axis=1), axis=0)

        # Plot peak sensor on topomap STAND
        mask = np.zeros((len(grand_avg_stand_ch.ch_names), len(grand_avg_stand_ch.times)), dtype=bool)
        mask[ch_ind, ...] = True
        fig = grand_avg_stand_ch.plot_topomap(times=peak[1], mask=mask, mask_params=dict(markerfacecolor='yellow', markersize=12))
        fig_name = fig_path + op.sep + ch_type + '_topoStand_' + peak[0] + '.png'
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # Plot peak sensor on topomap VIOL
        mask = np.zeros((len(grand_avg_viol_ch.ch_names), len(grand_avg_viol_ch.times)), dtype=bool)
        mask[ch_ind, ...] = True
        fig = grand_avg_viol_ch.plot_topomap(times=peak[1], mask=mask, mask_params=dict(markerfacecolor='yellow', markersize=12))
        fig_name = fig_path + op.sep + ch_type + '_topoViol_' + peak[0] + '.png'
        print('Saving ' + fig_name)
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # EACH 7 SEQUENCES, STAND & VIOL
        # stand item
        evoked_funcs.plot_evoked_with_sem_7seq(evoked_standard_seq, ch_ind, ch_type=ch_type, label=None)
        fig_name = fig_path + op.sep + ch_type + '_eachSeqTimecourse_stand_' + peak[0] + '.png'
        print('Saving ' + fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # viol item
        evoked_funcs.plot_evoked_with_sem_7seq(evoked_viol_seq, ch_ind, ch_type=ch_type, label=None)
        fig_name = fig_path + op.sep + ch_type + '_eachSeqTimecourse_viol_' + peak[0] + '.png'
        print('Saving ' + fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # stand full seq
        evoked_funcs.plot_evoked_with_sem_7seq_fullseq(evoked_full_seq_standard_seq, ch_ind, ch_type=ch_type, label=None)
        fig_name = fig_path + op.sep + ch_type + '_eachSeqTimecourse_FULLSEQ_stand_' + peak[0] + '.png'
        print('Saving ' + fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # stand full seq heatmap
        evoked_funcs.plot_evoked_heatmap_7seq_fullseq(evoked_full_seq_standard_seq, ch_ind, ch_type=ch_type)
        fig_name = fig_path + op.sep + ch_type + '_eachSeqTimecourse_FULLSEQ_stand_heatmap_' + peak[0] + '.png'
        print('Saving ' + fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)
        # stand full seq timecourses
        evoked_funcs.plot_evoked_timecourse_7seq_fullseq(evoked_full_seq_standard_seq, ch_ind, ch_type=ch_type)
        fig_name = fig_path + op.sep + ch_type + '_eachSeqTimecourse_FULLSEQ_stand_timecourses_' + peak[0] + '.png'
        print('Saving ' + fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)

        # ONE FIGURE, ONE SEQUENCE, STAND vs VIOL -- ITEMS
        plt.close('all')
        fig, axes = plt.subplots(7, 1, figsize=(5, 14), sharex=True, sharey=True, constrained_layout=True)
        ax = axes.ravel()[::1]
        for seqID in range(7):
            condS = 'items_standard_seq' + str(seqID+1) + '-'
            condV = 'items_viol_seq' + str(seqID+1) + '-'
            condS2 = 'items_standard_balanced_seq' + str(seqID+1) + '-'
            # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax[seqID].axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(3):
                ax[seqID].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            ax[seqID].set_xlabel('Time (ms)')
            seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqID + 1)
            ax[seqID].set_title(seqname, loc='right', weight='bold')
            data = evoked_viol_seq[condV].copy()
            evoked_funcs.plot_evoked_with_sem_1cond(data, condV, ch_type, ch_ind, color='r', filter=True, axis=ax[seqID])
            data = evoked_balanced_standard_seq[condS2].copy()
            evoked_funcs.plot_evoked_with_sem_1cond(data, condS2, ch_type, ch_ind, color='b', filter=True, axis=ax[seqID])
            # plt.legend(loc='upper right', fontsize=9)
            ax[seqID].set_xlim([-100, 750])
        info = peak[0]
        # info = 'complexity_cluster'
        fig_name = fig_path + op.sep + ch_type + '_VIOLvsBALSTAND_' + peak[0] + '.png'
        fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close('all')

# =============================================================================== #
# ================================== GFP plots =================================== #
# =============================================================================== #
fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP', 'GROUP')
utils.create_folder(fig_path)

# EACH 7 SEQUENCES, ITEMS, STAND
for ch_type in config.ch_types:
    fig = GFP_funcs.plot_GFP_timecourse_7seq(evoked_standard_seq, ch_type=ch_type)
    fig_name = op.join(fig_path, ch_type + '_eachSeqGFP_ITEMS_stand_timecourses.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close(fig)

# EACH 7 SEQUENCES, ITEMS, DEVIANTS
for ch_type in ['eeg', 'grad', 'mag']:
    fig = GFP_funcs.plot_GFP_timecourse_7seq(evoked_viol_seq, ch_type=ch_type)
    fig_name = op.join(fig_path, ch_type + '_eachSeqGFP_ITEMS_deviant_timecourses.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close(fig)

# EACH 7 SEQUENCES, FULL SEQUENCES, STAND
for ch_type in config.ch_types:
    fig = GFP_funcs.plot_GFP_timecourse_7seq(evoked_full_seq_standard_seq, ch_type=ch_type)
    fig_name = op.join(fig_path, ch_type + '_eachSeqGFP_FULLSEQ_stand_timecourses.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close(fig)

###################################################################################
# ------------------------------------------------------------------------------- #
###################################################################################

# =============================================================================== #
# ============================== Load data Resid ================================ #
# =============================================================================== #
# -- items: Nsubjects X 7 sequences
evoked_standard_seq_resid = evoked_funcs.load_evoked_resid(subject='all', filter_name='items_standard_seq', filter_not='pos')
evoked_viol_seq_resid = evoked_funcs.load_evoked_resid(subject='all', filter_name='items_viol_seq', filter_not='pos')
evoked_balanced_standard_seq_resid = evoked_funcs.load_evoked_resid(subject='all', filter_name='items_standard_balanced_seq', filter_not='pos')
# -- full sequence: Nsubjects X 7 sequences
# evoked_full_seq_standard_seq_resid = evoked_funcs.load_evoked_resid(subject='all', filter_name='full_seq_standard_seq', filter_not='pos')
# -- items: Nsubjects X 1 sequence (all seq average)
evoked_all_standard_resid = evoked_funcs.load_evoked_resid(subject='all', filter_name='items_standard_all', filter_not=None)
evoked_all_viol_resid = evoked_funcs.load_evoked_resid(subject='all', filter_name='items_viol_all', filter_not=None)
evoked_all_standard_balanced_resid = evoked_funcs.load_evoked_resid(subject='all', filter_name='items_standard_balanced_all', filter_not=None)

# =============================================================================== #
# ===================== Butterfly full sequences / each seq ===================== #
# =============================================================================== #
# evoked_funcs.plot_butterfly_fullseq_allsubj(evoked_full_seq_standard_seq, violation_or_not=0, residevoked=True)

# =============================================================================== #
# ========================= Butterfly items / each seq ========================== #
# =============================================================================== #
# -- Standards
evoked_funcs.plot_butterfly_items_allsubj(evoked_standard_seq_resid, violation_or_not=0, residevoked=True)
# -- Deviants
evoked_funcs.plot_butterfly_items_allsubj(evoked_viol_seq_resid, violation_or_not=1, residevoked=True)

# =============================================================================== #
# ====================== Butterfly items / all seq average ====================== #
# =============================================================================== #
# -- Standards
evoked_funcs.plot_butterfly_items_allsubj_allseq(evoked_all_standard_resid, violation_or_not=0, residevoked=True)
# -- Deviants
evoked_funcs.plot_butterfly_items_allsubj_allseq(evoked_all_viol_resid, violation_or_not=1, residevoked=True)

# =============================================================================== #
# ================================== GFP plots =================================== #
# =============================================================================== #
fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GFP', 'GROUP', 'residevoked')
utils.create_folder(fig_path)

# EACH 7 SEQUENCES, ITEMS, STAND
for ch_type in ['eeg', 'grad', 'mag']:
    fig = GFP_funcs.plot_GFP_timecourse_7seq(evoked_standard_seq_resid, ch_type=ch_type)
    fig_name = op.join(fig_path, ch_type + '_eachSeqGFP_ITEMS_stand_timecourses.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close(fig)

# EACH 7 SEQUENCES, ITEMS, DEVIANTS
for ch_type in ['eeg', 'grad', 'mag']:
    fig = GFP_funcs.plot_GFP_timecourse_7seq(evoked_viol_seq_resid, ch_type=ch_type)
    fig_name = op.join(fig_path, ch_type + '_eachSeqGFP_ITEMS_deviant_timecourses.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close(fig)

# # EACH 7 SEQUENCES, FULL SEQUENCES, STAND
# for ch_type in ['eeg', 'grad', 'mag']:
#     fig = GFP_funcs.plot_GFP_timecourse_7seq(evoked_full_seq_standard_seq_resid, ch_type=ch_type)
#     fig_name = op.join(fig_path, ch_type + '_eachSeqGFP_FULLSEQ_stand_timecourses.png')
#     print('Saving ' + fig_name)
#     plt.savefig(fig_name, bbox_inches='tight', dpi=300)
#     plt.close(fig)


# reload(evoked_funcs)
# =============================================================================== #
# ================================== Sources plots - TO DO !! =============================== #
# =============================================================================== #

fig_path = op.join(config.fig_path, 'Sources_evoked')
utils.create_folder(fig_path)

# items_standard_all
name = 'items_standard_all'
ev_data = evoked_all_standard['items_standard_all-']
all_stcs = []
for nsub, subject in enumerate(config.subjects_list):
    stc = source_estimation_funcs.normalized_sources_from_evoked(subject, ev_data[nsub][0])
    all_stcs.append(stc)

# Group mean stc
n_subjects = len(all_stcs)
mean_stc = all_stcs[0].copy()  # get copy of first instance
for sub in range(1, n_subjects):
    mean_stc._data += all_stcs[sub].data
mean_stc._data /= n_subjects
mean_ev = mne.grand_average(ev_data)  # not working... list of list issue??
mean_ev = mne.grand_average([ev_data[ii][0] for ii in range(len(ev_data))])  # correct the list of lists issue ?


# Timecourse source figure
output_file = op.join(fig_path, 'Sources_' + name + '_timecourse.png')
times_to_plot = [.0, .075, .150, .225, .300, .375, .450, .575]
times_to_plot = [.060, .130]
win_size = .030
stc = mean_stc
maxval = np.max(stc._data)
colorlims = [maxval * .30, maxval * .40, maxval * .80]
# plot and screenshot for each timewindow
stc_screenshots = []
for t in times_to_plot:
    twin_min = t
    twin_max = t + win_size
    stc_timewin = stc.copy()
    stc_timewin.crop(tmin=twin_min, tmax=twin_max)
    stc_timewin = stc_timewin.mean()
    brain = stc_timewin.plot(views=['lat'], surface='pial', hemi='split', size=(1400, 600), subject='fsaverage', clim=dict(kind='value', lims=colorlims),
                             subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'), background='w', smoothing_steps=5,
                             colormap='hot', colorbar=False, time_viewer=False)
    screenshot = brain.screenshot()
    brain.close()
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    plt.close('all')
    stc_screenshots.append(cropped_screenshot)
# main figure
fig, axes = plt.subplots(len(times_to_plot), 1, figsize=(len(times_to_plot) * 1.1, 4))
fig.suptitle(name, fontsize=8, fontweight='bold')
for idx in range(len(times_to_plot)):
    axes[idx].imshow(stc_screenshots[idx])
    axes[idx].axis('off')
    twin_min = times_to_plot[idx]
    twin_max = times_to_plot[idx] + win_size
    axes[idx].set_title('[%d - %d ms]' % (twin_min * 1000, twin_max * 1000), fontsize=5)
# tweak margins and spacing
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.01, top=0.9, wspace=1, hspace=0.6)
fig.savefig(output_file, bbox_inches='tight', dpi=600)
print('========> ' + output_file + " saved !")
plt.close(fig)
