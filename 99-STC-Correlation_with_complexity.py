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



results_path = op.join(config.result_path, 'Correlation_complexity')
utils.create_folder(results_path)

correl_hab = []
correl_standard = []
correl_deviant = []
correl_standard_minus_deviant = []

for subject in subjects_list:
    correl_standard_minus_deviant_subj = compute_correlation_with_complexity(subject, condition_name="standard_minus_deviant", baseline=True, morph_sources=False)

    correl_hab_subj = compute_correlation_with_complexity(subject, condition_name="habituation", baseline=True, morph_sources=False)
    correl_standard_subj = compute_correlation_with_complexity(subject, condition_name="standard", baseline=True, morph_sources=False)
    correl_deviant_subj = compute_correlation_with_complexity(subject, condition_name="deviant", baseline=True, morph_sources=False)

    correl_hab.append(correl_hab_subj)
    correl_standard.append(correl_standard)
    correl_deviant.append(correl_deviant)
    correl_standard_minus_deviant.append(correl_standard_minus_deviant)

# Save all subjects data to a file
with open(op.join(results_path, 'correlation_complexity_habituation.pickle'), 'wb') as f:
    pickle.dump(correl_hab, f, pickle.HIGHEST_PROTOCOL)

with open(op.join(results_path, 'correlation_complexity_standard.pickle'), 'wb') as f:
    pickle.dump(correl_standard, f, pickle.HIGHEST_PROTOCOL)

with open(op.join(results_path, 'correlation_complexity_deviant.pickle'), 'wb') as f:
    pickle.dump(correl_deviant, f, pickle.HIGHEST_PROTOCOL)

with open(op.join(results_path, 'correlation_complexity_standard_minus_deviant.pickle'), 'wb') as f:
    pickle.dump(correl_standard_minus_deviant, f, pickle.HIGHEST_PROTOCOL)
