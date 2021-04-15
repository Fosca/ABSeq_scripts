import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")

import mne
import config
import os.path as op
from ABseq_func import *
import pickle
from ABseq_func import stc_funcs

# Exclude some subjects
config.exclude_subjects.append('sub08-cc_150418')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

# ======================================================================================================
# ======================== load and save the concatenated object =======================================
# ======================================================================================================

def load_correl(subject,condition_name):

    results_path = op.join(config.result_path, 'Correlation_complexity/')
    with open(op.join(results_path, subject + '_stc_correl_'+condition_name+'.pickle'), 'rb') as f:
        stc = pickle.load(f)

    return stc

correl_hab = []
correl_standard = []
correl_deviant = []
correl_standard_minus_deviant = []
for subject in config.subjects_list:
    correl_standard_minus_deviant_subj = load_correl(subject, condition_name="stc_standard_minus_deviant_habituation")
    correl_hab_subj = load_correl(subject, condition_name="complexity_habituation")
    correl_standard_subj = load_correl(subject, condition_name="standard_habituation")
    correl_deviant_subj = load_correl(subject, condition_name="deviant_habituation")

    correl_hab.append(correl_hab_subj)
    correl_standard.append(correl_standard)
    correl_deviant.append(correl_deviant)
    correl_standard_minus_deviant.append(correl_standard_minus_deviant)

# Save all subjects data to a file
with open(op.join(op.join(config.result_path, 'Correlation_complexity/'), 'correlation_complexity_habituation.pickle'), 'wb') as f:
    pickle.dump(correl_hab, f, pickle.HIGHEST_PROTOCOL)

with open(op.join(op.join(config.result_path, 'Correlation_complexity/'), 'correlation_complexity_standard.pickle'), 'wb') as f:
    pickle.dump(correl_standard, f, pickle.HIGHEST_PROTOCOL)

with open(op.join(op.join(config.result_path, 'Correlation_complexity/'), 'correlation_complexity_deviant.pickle'), 'wb') as f:
    pickle.dump(correl_deviant, f, pickle.HIGHEST_PROTOCOL)

with open(op.join(op.join(config.result_path, 'Correlation_complexity/'), 'correlation_complexity_standard_minus_deviant.pickle'), 'wb') as f:
    pickle.dump(correl_standard_minus_deviant, f, pickle.HIGHEST_PROTOCOL)


# ======================================================================================================
# ======================== load and save the concatenated object =======================================
# ======================================================================================================

def tvalues_from_stc(stcs_morphed_fsaverage):

    import numpy as np
    import scipy.stats

    n_vertices, n_times = stcs_morphed_fsaverage[0].shape
    full_data = np.zeros((len(stcs_morphed_fsaverage),n_vertices,n_times))
    tval_map = np.zeros((n_vertices,n_times))
    for sub in range(len(stcs_morphed_fsaverage)):
        full_data[sub,:,:] = stcs_morphed_fsaverage[sub]._data

    for vert in range(n_vertices):
        print("vertex number %i"%vert)
        for tim in range(n_times):
            tval_map[vert,tim], _ = scipy.stats.ttest_1samp(full_data[:,vert,tim],popmean=0)

    stc_tvalue = stcs_morphed_fsaverage[0].copy()
    stc_tvalue._data = tval_map


whichs = ['correlation_complexity_habituation','correlation_complexity_standard','correlation_complexity_deviant','correlation_complexity_standard_minus_deviant']

for which in whichs:

    # Load correlation STCs
    with open(op.join(config.result_path, 'Correlation_complexity', which+'.pickle'), 'rb') as f:
        stcs = pickle.load(f)
    n_subjects = len(stcs)


    # Morth to fsaverage
    stcs_fsaverage = []
    for sub in range(n_subjects):
        morph = mne.compute_source_morph(stcs[sub], subject_from=stcs[sub].subject, subject_to='fsaverage', subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'))
        stcs_fsaverage.append(morph.apply(stcs[sub]))

    # ----- compute also the t-values across participants ---
    tval_map = tvalues_from_stc(stcs)

    # Group average
    mean_stc = stcs_fsaverage[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        mean_stc._data += stcs_fsaverage[sub].data
    mean_stc._data /= n_subjects

    # Timecourse source figure
    output_file = op.join(config.result_path, 'Correlation_complexity',  which+'_timecourse.png')
    output_file_tval = op.join(config.result_path, 'Correlation_complexity',  which+'tvalues_timecourse.png')
    times_to_plot = [.0, .050, .100, .150, .200, .250, .300, .350, .400]
    win_size = .050
    source_estimation_funcs.timecourse_source_figure(mean_stc, which, times_to_plot, win_size, output_file)
    source_estimation_funcs.timecourse_source_figure(tval_map, which, times_to_plot, win_size, output_file_tval)
