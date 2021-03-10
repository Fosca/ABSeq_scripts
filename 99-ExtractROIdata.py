import mne
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *
from importlib import reload
import numpy as np
from scipy import stats
import pickle
from mne.stats import (spatio_temporal_cluster_1samp_test, summarize_clusters_stc)
from scipy.signal import savgol_filter
from scipy.stats import sem
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Exclude some subjects
# config.exclude_subjects.append('sub10-gp_190568')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()


analysis_name = 'Viol_vs_Stand'  # Viol_vs_Stand
subjects_list = config.subjects_list
# subjects_list = [config.subjects_list[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16]]  # only subjects with available sources data
fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')

results_path = op.join(config.result_path, 'TMP')
utils.create_folder(results_path)

path = op.join(results_path, 'ROI')
utils.create_folder(path)
subjects_list = config.subjects_list

# for analysis_name in [analysis_main_name + '_seqID1']:
print(analysis_name)
n_subjects = len(subjects_list)

label_names = ['parsopercularis-lh', 'parsopercularis-rh', 'bankssts-rh', 'bankssts-lh', 'superiorparietal-lh', 'superiorparietal-rh', 'inferiorparietal-lh', 'inferiorparietal-rh', ]

# =======================================================================
# ============== Extract ROI data
# create empty dict to store everything
group_all_labels_data = dict()
for label_name in label_names:
    group_all_labels_data[label_name] = dict()
    group_all_labels_data[label_name]['cond1'] = []
    group_all_labels_data[label_name]['cond2'] = []
# subject loop
for subject in subjects_list:
    # Load evoked and sources for the 2 conditions
    evoked1, stc1 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name='items_standard_all', evoked_filter_not=None, evoked_path='evoked_cleaned', apply_baseline=True,
                                                                     lowpass_evoked=True, morph_sources=False, fake_nave=True)
    evoked2, stc2 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name='items_viol_all', evoked_filter_not=None, evoked_path='evoked_cleaned', apply_baseline=True,
                                                                     lowpass_evoked=True, morph_sources=False, fake_nave=True)
    src = mne.read_source_spaces(op.join(config.meg_dir, subject, subject + '_oct6-inv.fif'))
    for label_name in label_names:
        anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
        label_data_cond1 = stc1.extract_label_time_course(anat_label, src, mode='mean')[0]
        label_data_cond2 = stc2.extract_label_time_course(anat_label, src, mode='mean')[0]
        # label_data_cond1 = stc1.extract_label_time_course(anat_label, src, mode='pca_flip')[0]
        # label_data_cond1 *= np.sign(label_data_cond1[np.argmax(np.abs(label_data_cond1))])  # flip the pca so that the max power is positive ??
        # label_data_cond2 = stc2.extract_label_time_course(anat_label, src, mode='pca_flip')[0]
        # label_data_cond2 *= np.sign(label_data_cond2[np.argmax(np.abs(label_data_cond2))])  # flip the pca so that the max power is positive ??
        group_all_labels_data[label_name]['cond1'].append(label_data_cond1)
        group_all_labels_data[label_name]['cond2'].append(label_data_cond2)

# =======================================================================
# ============== Plot group means
times = (1e3 * stc1.times)
filter = False
plt.close('all')
for label_name in label_names:
    data = group_all_labels_data[label_name]['cond1']
    mean1 = np.mean(data, axis=0)
    ub1 = mean1 + sem(data, axis=0)
    lb1 = mean1 - sem(data, axis=0)
    data = group_all_labels_data[label_name]['cond2']
    mean2 = np.mean(data, axis=0)
    ub2 = mean2 + sem(data, axis=0)
    lb2 = mean2 - sem(data, axis=0)

    if filter == True:
        mean1 = savgol_filter(mean1, 13, 3)
        ub1 = savgol_filter(ub1, 13, 3)
        lb1 = savgol_filter(lb1, 13, 3)
        mean2 = savgol_filter(mean2, 13, 3)
        ub2 = savgol_filter(ub2, 13, 3)
        lb2 = savgol_filter(lb2, 13, 3)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=False)
    plt.axvline(0, linestyle='-', color='k', linewidth=2)
    ax.fill_between(times, ub1, lb1, color='r', alpha=.2)
    ax.plot(times, mean1, color='r', linewidth=1.5, label='Deviant')
    ax.fill_between(times, ub2, lb2, color='b', alpha=.2)
    ax.plot(times, mean2, color='b', linewidth=1.5, label='Standard')
    ax.set_xlabel('Time (ms)')
    ax.set_xlim([-100, 600])
    for key in ('top', 'right'):  # Remove spines
        ax.spines[key].set(visible=False)
    plt.legend()
    plt.title(analysis_name + ': ' + label_name, fontsize=14, weight='bold', color='k')
    fig_name = op.join(path, analysis_name + '_' + label_name + '.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name, dpi=300)
    plt.close('all')
