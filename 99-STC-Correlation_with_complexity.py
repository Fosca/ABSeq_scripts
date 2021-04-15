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
