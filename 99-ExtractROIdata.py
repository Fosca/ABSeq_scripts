import mne
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *
import numpy as np
import pickle
from scipy.stats import sem
from scipy.stats.stats import pearsonr

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
    for seqID in range(1, 8):
        group_all_labels_data[label_name]['Habituation']['seq' + str(seqID)] = []
        group_all_labels_data[label_name]['Standard']['seq' + str(seqID)] = []
        group_all_labels_data[label_name]['Deviant']['seq' + str(seqID)] = []
    group_all_labels_data[label_name]['StandVSDev'] = dict()
    group_all_labels_data[label_name]['StandVSDev']['allseq'] = dict()
    group_all_labels_data[label_name]['StandVSDev']['allseq']['cond1'] = []
    group_all_labels_data[label_name]['StandVSDev']['allseq']['cond2'] = []
    for seqID in range(1, 8):
        group_all_labels_data[label_name]['StandVSDev']['seq' + str(seqID)] = dict()
        group_all_labels_data[label_name]['StandVSDev']['seq' + str(seqID)]['cond1'] = []
        group_all_labels_data[label_name]['StandVSDev']['seq' + str(seqID)]['cond2'] = []

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
        stcs['seq' + str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    # -- Store ROI timecourses
    for label_name in label_names:
        anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
        label_data = stcs['allseq'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['Habituation']['allseq'].append(label_data)
        for seqID in range(1, 8):
            label_data = stcs['seq' + str(seqID)].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['Habituation']['seq' + str(seqID)].append(label_data)

    # ==== STANDARDS DATA
    # -- Compute sources of evoked
    ev = epochs_items['TrialNumber > 10 & ViolationInSequence == 0'].average()
    stcs = dict()
    stcs['allseq'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    for seqID in range(1, 8):
        ev = epochs_items['TrialNumber > 10 & ViolationInSequence == 0 & SequenceID == ' + str(seqID)].average()
        stcs['seq' + str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    # -- Store ROI timecourses
    for label_name in label_names:
        anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
        label_data = stcs['allseq'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['Standard']['allseq'].append(label_data)
        for seqID in range(1, 8):
            label_data = stcs['seq' + str(seqID)].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['Standard']['seq' + str(seqID)].append(label_data)

    # ==== DEVIANTS DATA
    # -- Compute sources of evoked
    ev = epochs_items['ViolationOrNot == 1'].average()
    stcs = dict()
    stcs['allseq'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    for seqID in range(1, 8):
        ev = epochs_items['ViolationOrNot == 1 & SequenceID == ' + str(seqID)].average()
        stcs['seq' + str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=False)
    # -- Store ROI timecourses
    for label_name in label_names:
        anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
        label_data = stcs['allseq'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['Deviant']['allseq'].append(label_data)
        for seqID in range(1, 8):
            label_data = stcs['seq' + str(seqID)].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['Deviant']['seq' + str(seqID)].append(label_data)

    # ==== STAND VS DEV (balanced)
    # -- Compute sources of evoked
    ep_bal = epoching_funcs.balance_epochs_violation_positions(epochs_items, 'local')
    ev1 = ep_bal['ViolationOrNot == 0'].average()
    ev2 = ep_bal['ViolationOrNot == 1'].average()
    stcs = dict()
    stcs['allseq'] = dict()
    stcs['allseq']['cond1'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev1, morph_sources=False)
    stcs['allseq']['cond2'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev2, morph_sources=False)

    for seqID in range(1, 8):
        ev1 = ep_bal['ViolationOrNot == 0 & SequenceID == ' + str(seqID)].average()
        ev2 = ep_bal['ViolationOrNot == 1 & SequenceID == ' + str(seqID)].average()
        stcs['seq' + str(seqID)] = dict()
        stcs['seq' + str(seqID)]['cond1'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev1, morph_sources=False)
        stcs['seq' + str(seqID)]['cond2'] = source_estimation_funcs.compute_sources_from_evoked(subject, ev2, morph_sources=False)
    # -- Store ROI timecourses
    for label_name in label_names:
        anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
        label_data = stcs['allseq']['cond1'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['StandVSDev']['allseq']['cond1'].append(label_data)
        label_data = stcs['allseq']['cond2'].extract_label_time_course(anat_label, src, mode='mean')[0]
        group_all_labels_data[label_name]['StandVSDev']['allseq']['cond2'].append(label_data)
        for seqID in range(1, 8):
            label_data = stcs['seq' + str(seqID)]['cond1'].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['StandVSDev']['seq' + str(seqID)]['cond1'].append(label_data)
            label_data = stcs['seq' + str(seqID)]['cond2'].extract_label_time_course(anat_label, src, mode='mean')[0]
            group_all_labels_data[label_name]['StandVSDev']['seq' + str(seqID)]['cond2'].append(label_data)

# Save all subjects data to a file
with open(op.join(results_path, 'ROI_data.pickle'), 'wb') as f:
    pickle.dump(group_all_labels_data, f, pickle.HIGHEST_PROTOCOL)

# =======================================================================
# ============== Plot group means StandVSDev allseq
# (Re)Load extracted ROI timecourses
with open(op.join(results_path, 'ROI_data.pickle'), 'rb') as f:
    group_all_labels_data = pickle.load(f)

# Load one evoked to get times
ev, _ = evoked_funcs.load_evoked(subject=subjects_list[0], filter_name='items_standard_all')
times = (1e3 * ev['items_standard_all-'][0].times)

complexity_values = [4, 6, 6, 6, 12, 14, 28]

# =======================================================================
# ============== Plot group means StandVSDev allseq
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
# ============== Plot group means StandVSDev each seqID
analysis_name = 'StandVSDev'
plt.close('all')
for label_name in label_names:
    fig, ax = plt.subplots(7, 1, figsize=(6, 12), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(analysis_name + ': ' + label_name, fontsize=12)

    for seqID in range(7):
        seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqID + 1)
        data = group_all_labels_data[label_name]['StandVSDev']['seq' + str(seqID + 1)]['cond1']
        mean1 = np.mean(data, axis=0)
        ub1 = mean1 + sem(data, axis=0)
        lb1 = mean1 - sem(data, axis=0)
        data = group_all_labels_data[label_name]['StandVSDev']['seq' + str(seqID + 1)]['cond2']
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
for analysis_name in ['Habituation', 'Standard', 'Deviant']:
    plt.close('all')
    for label_name in label_names:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 4), constrained_layout=True, gridspec_kw={'height_ratios': [10, 1], 'width_ratios': [2, 1]}, sharex=False)  # sharex was not working as expected...
        plt.suptitle(analysis_name + ': ' + label_name, fontsize=14, weight='bold', color='k')
        NUM_COLORS = 7
        cmap = 'viridis'
        cmap2 = 'RdBu'
        cm = plt.get_cmap(cmap)
        colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])

        # Main plot
        allseq_means = []
        ax[0][0].axvline(0, linestyle='-', color='k', linewidth=2)
        for seqID in range(1, 8):
            data = group_all_labels_data[label_name][analysis_name]['seq' + str(seqID)]
            mean = np.mean(data, axis=0)
            ub = mean + sem(data, axis=0)
            lb = mean - sem(data, axis=0)
            ax[0][0].fill_between(times, ub, lb, color=colorslist[seqID - 1], alpha=.2)
            ax[0][0].plot(times, mean, color=colorslist[seqID - 1], linewidth=1.5, label='SeqID' + str(seqID))
            allseq_means.append(mean)
        ax[0][0].set_xlabel('Time (ms)')
        ax[0][0].set_xlim([-50, 600])

        for key in ('top', 'right'):  # Remove spines
            ax[0][0].spines[key].set(visible=False)
        ax[0][0].legend(loc='best', fontsize=8)

        # Compute & plot correlation over time
        allseq_means_array = np.array(allseq_means)
        timeR = []
        for timepoint in range(len(times)):
            r, pval = pearsonr(allseq_means_array[:, timepoint], complexity_values)
            timeR.append(r)
        a = np.array(timeR)
        a = np.expand_dims(a, axis=0)
        ax[1][0].imshow(a, extent=[min(times), max(times), 0, 10], cmap=cmap2, vmin=-1, vmax=1, interpolation='none')
        ax[1][0].axis('off')

        # ROI on Brain ?
        anat_label = mne.read_labels_from_annot('fsaverage', parc='aparc', subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'), regexp=label_name)[0]
        Brain = mne.viz.get_brain_class()
        brain = Brain('fsaverage', label_name[-2:], 'inflated', subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'), background='white', size=(800, 600))
        brain.add_label(anat_label, borders=False, color='r', alpha=.8)
        screenshot = brain.screenshot()
        brain.close()
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        ax[0][1].imshow(cropped_screenshot)
        ax[0][1].axis('off')
        ax[1][1].axis('off')

        # Save figure
        fig_name = op.join(results_path, analysis_name + '_' + label_name + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close('all')
