import mne
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *
import numpy as np
import pickle
from scipy.stats import sem

# Exclude some subjects
config.exclude_subjects.append('sub08-cc_150418')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

subjects_list = config.subjects_list
n_subjects = len(subjects_list)

fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
results_path = op.join(config.result_path, 'Sources_ROI')
utils.create_folder(results_path)

label_names = ['parsopercularis-lh', 'parsopercularis-rh', 'parsorbitalis-lh', 'parsorbitalis-rh', 'parstriangularis-rh', 'parstriangularis-lh',
               'bankssts-rh', 'bankssts-lh', 'superiortemporal-rh', 'superiortemporal-lh', 'middletemporal-rh', 'middletemporal-lh',
               'superiorparietal-lh', 'superiorparietal-rh', 'inferiorparietal-lh', 'inferiorparietal-rh', 'supramarginal-rh', 'supramarginal-lh']

########################### Extract ROI timecourse data ###########################
baseline = True
# create empty nested dict to store everything
group_all_labels_data = dict()
for label_name in label_names:
    group_all_labels_data[label_name] = dict()
    group_all_labels_data[label_name]['Habituation'] = dict()
    group_all_labels_data[label_name]['Standard'] = dict()
    group_all_labels_data[label_name]['Deviant'] = dict()
    group_all_labels_data[label_name]['Habituation']['allseq'] = []
    group_all_labels_data[label_name]['Standard']['allseq'] = []
    group_all_labels_data[label_name]['Deviant']['allseq'] = []
    for seqID in range(1,8):
        group_all_labels_data[label_name]['Habituation']['seq'+str(seqID)] = []
        group_all_labels_data[label_name]['Standard']['seq'+str(seqID)] = []
        group_all_labels_data[label_name]['Deviant']['seq'+str(seqID)] = []
    group_all_labels_data[label_name]['StandVSDev'] = dict()
    group_all_labels_data[label_name]['StandVSDev']['allseq'] = dict()
    group_all_labels_data[label_name]['StandVSDev']['allseq']['cond1'] = []
    group_all_labels_data[label_name]['StandVSDev']['allseq']['cond2'] = []
    for seqID in range(1,8):
        group_all_labels_data[label_name]['StandVSDev']['seq'+str(seqID)] = dict()
        group_all_labels_data[label_name]['StandVSDev']['seq'+str(seqID)]['cond1'] = []
        group_all_labels_data[label_name]['StandVSDev']['seq'+str(seqID)]['cond2'] = []

# subject loop
for subject in subjects_list:

    # # METHOD 1: USE ALREADY CREATED EVOKED
    # # Load evoked and sources for the 2 conditions
    # evoked1, stc1 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name='items_standard_all', evoked_filter_not=None, evoked_path='evoked_cleaned', apply_baseline=True,
    #                                                                  lowpass_evoked=True, morph_sources=False, fake_nave=True)
    # evoked2, stc2 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name='items_viol_all', evoked_filter_not=None, evoked_path='evoked_cleaned', apply_baseline=True,
    #                                                                  lowpass_evoked=True, morph_sources=False, fake_nave=True)

    # METHOD 2: LOAD EPOCHS, (BALANCE,) COMPUTE EVOKED AND SOURCES
    epochs_items = epoching_funcs.load_epochs_items(subject)
    epochs_items = epochs_items.pick_types(meg=True, eeg=False, eog=False)  ## Exclude EEG (was done when computing inverse
    if baseline:
        epochs_items = epochs_items.apply_baseline(baseline=(-0.050, 0))
    src = mne.read_source_spaces(op.join(config.meg_dir, subject, subject + '_oct6-inv.fif'))

    # ==== HABITUATION DATA
    # -- Compute sources of evoked
    ev = epochs_items['TrialNumber < 11'].average()
    stcs = dict()
    stcs['allseq'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    for seqID in range(1, 8):
        ev = epochs_items['TrialNumber < 11 & SequenceID == ' + str(seqID)].average()
        stcs['seq'+str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    # -- Store ROI timecourses
    for label_name in label_names:
        anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
        label_data = stcs['allseq'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['Habituation']['allseq'].append(label_data)
        for seqID in range(1, 8):
            label_data = stcs['seq'+str(seqID)].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['Habituation']['seq'+str(seqID)].append(label_data)

    # ==== STANDARDS DATA
    # -- Compute sources of evoked
    ev = epochs_items['TrialNumber > 10 & ViolationInSequence == 0'].average()
    stcs = dict()
    stcs['allseq'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    for seqID in range(1, 8):
        ev = epochs_items['TrialNumber > 10 & ViolationInSequence == 0 & SequenceID == ' + str(seqID)].average()
        stcs['seq'+str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    # -- Store ROI timecourses
    for label_name in label_names:
        anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
        label_data = stcs['allseq'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['Standard']['allseq'].append(label_data)
        for seqID in range(1, 8):
            label_data = stcs['seq'+str(seqID)].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['Standard']['seq'+str(seqID)].append(label_data)

    # ==== DEVIANTS DATA
    # -- Compute sources of evoked
    ev = epochs_items['ViolationOrNot == 1'].average()
    stcs = dict()
    stcs['allseq'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    for seqID in range(1, 8):
        ev = epochs_items['ViolationOrNot == 1 & SequenceID == ' + str(seqID)].average()
        stcs['seq'+str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    # -- Store ROI timecourses
    for label_name in label_names:
        anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
        label_data = stcs['allseq'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['Deviant']['allseq'].append(label_data)
        for seqID in range(1, 8):
            label_data = stcs['seq'+str(seqID)].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['Deviant']['seq'+str(seqID)].append(label_data)

    # ==== STAND VS DEV (balanced - but across seqIDs not for each seqID)
    # -- Compute sources of evoked
    ep_bal = epoching_funcs.balance_epochs_violation_positions(epochs_items, 'sequence')
    ev1 = ep_bal['ViolationOrNot == 0'].average()
    ev2 = ep_bal['ViolationOrNot == 1'].average()
    stcs = dict()
    stcs['allseq'] = dict()
    stcs['allseq']['cond1'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev1, morph_sources=False)
    stcs['allseq']['cond2'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev2, morph_sources=False)

    for seqID in range(1, 8):
        ev1 = ep_bal['ViolationOrNot == 0 & SequenceID == ' + str(seqID)].average()
        ev2 = ep_bal['ViolationOrNot == 1 & SequenceID == ' + str(seqID)].average()
        stcs['seq'+str(seqID)] = dict()
        stcs['seq'+str(seqID)]['cond1'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev1, morph_sources=False)
        stcs['seq'+str(seqID)]['cond2'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev2, morph_sources=False)
    # -- Store ROI timecourses
    for label_name in label_names:
        anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
        label_data = stcs['allseq']['cond1'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['StandVSDev']['allseq']['cond1'].append(label_data)
        label_data = stcs['allseq']['cond2'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['StandVSDev']['allseq']['cond2'].append(label_data)
        for seqID in range(1, 8):
            label_data = stcs['seq'+str(seqID)]['cond1'].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['StandVSDev']['seq'+str(seqID)]['cond1'].append(label_data)
            label_data = stcs['seq'+str(seqID)]['cond2'].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['StandVSDev']['seq'+str(seqID)]['cond2'].append(label_data)

# Save all subjects data to a file
with open(op.join(results_path, 'ROI_data.pickle'), 'wb') as f:
    pickle.dump(group_all_labels_data, f, pickle.HIGHEST_PROTOCOL)

# =======================================================================
# ============== Plot group means StandVSDev allseq
times = (1e3 * ev.times)
analysis_name = 'StandVSDev'
plt.close('all')
for label_name in label_names:
    data = group_all_labels_data[label_name]['StandVSDev']['allseq']['cond1']
    mean1 = np.mean(data, axis=0)
    ub1 = mean1 + sem(data, axis=0)
    lb1 = mean1 - sem(data, axis=0)
    data = group_all_labels_data[label_name]['StandVSDev']['allseq']['cond2']
    mean2 = np.mean(data, axis=0)
    ub2 = mean2 + sem(data, axis=0)
    lb2 = mean2 - sem(data, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=False)
    plt.axvline(0, linestyle='-', color='k', linewidth=2)
    ax.fill_between(times, ub1, lb1, color='b', alpha=.2)
    ax.plot(times, mean1, color='b', linewidth=1.5, label='Standard')
    ax.fill_between(times, ub2, lb2, color='r', alpha=.2)
    ax.plot(times, mean2, color='r', linewidth=1.5, label='Deviant')
    ax.set_xlabel('Time (ms)')
    ax.set_xlim([-50, 600])
    for key in ('top', 'right'):  # Remove spines
        ax.spines[key].set(visible=False)
    plt.legend()
    plt.title(analysis_name + ': ' + label_name, fontsize=14, weight='bold', color='k')
    fig_name = op.join(results_path, analysis_name + '_allseq_' + label_name + '.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close('all')

# =======================================================================
# ============== Plot group means StandVSDev reach seqID
times = (1e3 * ev.times)
analysis_name = 'StandVSDev'
plt.close('all')
for label_name in label_names:
    fig, ax = plt.subplots(7, 1, figsize=(6, 12), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(analysis_name + ': ' + label_name, fontsize=12)

    for seqID in range(7):
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqID + 1)
        data = group_all_labels_data[label_name]['StandVSDev']['seq'+str(seqID+1)]['cond1']
        mean1 = np.mean(data, axis=0)
        ub1 = mean1 + sem(data, axis=0)
        lb1 = mean1 - sem(data, axis=0)
        data = group_all_labels_data[label_name]['StandVSDev']['seq'+str(seqID+1)]['cond2']
        mean2 = np.mean(data, axis=0)
        ub2 = mean2 + sem(data, axis=0)
        lb2 = mean2 - sem(data, axis=0)
        ax[seqID].axvline(0, linestyle='-', color='k', linewidth=2)
        ax[seqID].fill_between(times, ub1, lb1, color='b', alpha=.2)
        ax[seqID].plot(times, mean1, color='b', linewidth=1.5, label='Standard')
        ax[seqID].fill_between(times, ub2, lb2, color='r', alpha=.2)
        ax[seqID].plot(times, mean2, color='r', linewidth=1.5, label='Deviant')
        ax[seqID].set_xlabel('Time (ms)')
        ax[seqID].set_xlim([-50, 600])
        for key in ('top', 'right'):  # Remove spines
            ax[seqID].spines[key].set(visible=False)
        ax[seqID].set_title(seqname, loc='left', weight='bold', fontsize=12)
    ax[seqID].set_xlabel('Time (ms)')
    fig_name = op.join(results_path, analysis_name + '_eachSeq_' + label_name + '.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close('all')

    #     mean = allseq_mean[seqID]
    #     ub = allseq_ub[seqID]
    #     lb = allseq_lb[seqID]
    #     ax[seqID].set_title(seqname, loc='left', weight='bold', fontsize=12)
    #     ax[seqID].fill_between(times, ub / scalings[ch_type], lb / scalings[ch_type], color='black', alpha=.2)
    #     ax[seqID].plot(times, mean / scalings[ch_type], color=ch_colors[ch_type], linewidth=1.5, label=seqname)
    #     ax[seqID].axvline(0, linestyle='-', color='black', linewidth=2)
    #     ax[seqID].set_xlim([min(times), max(times)])
    #     # Add vertical lines
    #     for xx in range(2):
    #         ax[seqID].axvline(250 * xx, linestyle='--', color='black', linewidth=1)
    #     # Remove spines
    #     for key in ('top', 'right', 'bottom'):
    #         ax[seqID].spines[key].set(visible=False)
    #     ax[seqID].set_ylabel(units[ch_type])
    #     ax[seqID].set_xticks([], [])
    #     fmt = ticker.ScalarFormatter(useMathText=True)
    #     fmt.set_powerlimits((0, 0))
    #     ax[seqID].get_yaxis().set_major_formatter(fmt)
    #     ax[seqID].get_yaxis().get_offset_text().set_position((-0.07, 0))  # move 'x10-x', does not work with y
    # ax[seqID].set_xticks(range(-500, 4500, 500), [])
    # ax[seqID].set_xlabel('Time (ms)')

# =======================================================================
# ============== Plot group means Habituation/Standard/Deviants each seq
times = (1e3 * ev.times)
for analysis_name in ['Habituation', 'Standard', 'Deviant']:
    plt.close('all')
    for label_name in label_names:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=False)
        plt.axvline(0, linestyle='-', color='k', linewidth=2)
        NUM_COLORS = 7
        cm = plt.get_cmap('viridis')
        colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])

        for seqID in range(1,8):
            data = group_all_labels_data[label_name][analysis_name]['seq'+str(seqID)]
            mean = np.mean(data, axis=0)
            ub = mean + sem(data, axis=0)
            lb = mean - sem(data, axis=0)
            ax.fill_between(times, ub, lb, color=colorslist[seqID-1], alpha=.2)
            ax.plot(times, mean, color=colorslist[seqID-1], linewidth=1.5, label='SeqID'+str(seqID))

        ax.set_xlabel('Time (ms)')
        ax.set_xlim([-50, 600])
        for key in ('top', 'right'):  # Remove spines
            ax.spines[key].set(visible=False)
        plt.legend(loc='best', fontsize=8)
        plt.title(analysis_name + ': ' + label_name, fontsize=14, weight='bold', color='k')
        fig_name = op.join(results_path, analysis_name + '_' + label_name + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close('all')
