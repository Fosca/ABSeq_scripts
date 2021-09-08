import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
from autoreject import AutoReject
import pickle
import numpy as np
import mne

path_log_1 = '/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/sub04-rf_190499/sub04-rf_190499_clean_epo_eeg_1Hz_reject_local_log.obj'
log_1 = pickle.load(open(path_log_1,'rb'))

path_log_01 = '/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/sub04-rf_190499/sub04-rf_190499_clean_epo_eeg_lfreq_01Hz_reject_local_log.obj'
log_01 = pickle.load(open(path_log_01,'rb'))

path_epoch = '/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/sub04-rf_190499/sub04-rf_190499_epo.fif'
epochs = mne.read_epochs(path_epoch)

path_epoch_clean_1 = "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/sub04-rf_190499/sub04-rf_190499_clean_epo_eeg_1Hz.fif"
epochs_clean_1 = mne.read_epochs(path_epoch_clean_1)

path_epoch_clean_01 = "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/sub04-rf_190499/sub04-rf_190499_clean_epo_eeg_lfreq_01Hz.fif"
epochs_clean_01 = mne.read_epochs(path_epoch_clean_01)

epochs_rejected_ar_1 = epochs[log_1.bad_epochs]
epochs_rejected_ar_01 = epochs[log_01.bad_epochs]

epochs_rejected_ar_1.average().plot_joint()
epochs_rejected_ar_01.average().plot_joint()

epochs_clean.average().plot_joint()

###########################
path_log = '/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/sub12-lg_170436/noEEG/'
import glob, os
os.chdir(path_log)
for file in glob.glob("sub12-lg_170436_ARglob_epo*_ARglob_thresholds.obj"):
    print(file)
    reject_thresholds = pickle.load(open(file, 'rb'))
    print(reject_thresholds)
###########################

import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import csv
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import config
import os.path as op
import mne
import glob
import warnings
from autoreject import AutoReject
from autoreject import get_rejection_threshold
import pickle
import config
from mne.parallel import parallel_func
from ABseq_func import epoching_funcs
from ABseq_func import autoreject_funcs
from ABseq_func import source_estimation_funcs
import csv
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import config
import os.path as op
import mne
import glob
import warnings
from autoreject import AutoReject
import pickle


# subject = config.subjects_list[11]
subject = 'sub08-cc_150418'
meg_subject_dir = op.join(config.meg_dir, subject)
epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)



# run autoreject "global" -> just get the thresholds
reject = get_rejection_threshold(epochs, ch_types=['mag', 'grad', 'eeg'])
epochs1 = epochs.copy().drop_bad(reject=reject)
fname = op.join(meg_subject_dir, 'epochs_globalAR-epo.fif')
print("Saving: ", fname)
epochs1.save(fname, overwrite=True)


# run autoreject "local"
ar = AutoReject()
epochs2, reject_log = ar.fit_transform(epochs, return_log=True)
fname = op.join(meg_subject_dir, 'epochs_localAR-epo.fif')
print("Saving: ", fname)
epochs2.save(fname, overwrite=True)
# Save autoreject reject_log
pickle.dump(reject_log, open(fname[:-4] + '_reject_log.obj', 'wb'))


######################
fname = op.join(meg_subject_dir, 'epochs_globalAR-epo.fif')
epochs1 = mne.read_epochs(fname, preload=True)
epochs1
epochs1['ViolationOrNot == 1'].copy().average().plot_joint()

fname = op.join(meg_subject_dir, 'epochs_localAR-epo.fif')
epochs2 = mne.read_epochs(fname, preload=True)
epochs2['ViolationOrNot == 1'].copy().average().plot_joint()

arlog_name = op.join(meg_subject_dir, 'epochs_localAR_reject_log.obj')
reject_log = pickle.load(open(arlog_name, 'rb'))
Nrej = sum(reject_log.bad_epochs == True)
Nepochs = 16 * 46 * 7 * 2
print('%s, items epochs: %d/%d rejected bad epochs items = %.2f%%' % (subject, Nrej, Nepochs, Nrej / Nepochs * 100))
autoreject_funcs.reject_log_plot(reject_log, subject, save_path=meg_subject_dir, fig_name='AutoReject')


########################################################
########################################################
# Exclude some subjects
config.exclude_subjects.append('sub08-cc_150418')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()


for subject in config.subjects_list:
    source_estimation_funcs.compute_noise_cov(subject)

for subject in config.subjects_list:
    source_estimation_funcs.inverse_operator(subject)


########################################################
########################################################
import config
import matplotlib.pyplot as plt
from ABseq_func import *

which = 'correlation_complexity_habituation'

# Load correlation STCs
with open(op.join(config.result_path, 'Correlation_complexity', which+'.pickle'), 'rb') as f:
    stcs = pickle.load(f)
n_subjects = len(stcs)

# Morth to fsaverage
stcs_fsaverage = []
for sub in range(n_subjects):
    morph = mne.compute_source_morph(stcs[sub], subject_from=stcs[sub].subject, subject_to='fsaverage', subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'))
    stcs_fsaverage.append(morph.apply(stcs[sub]))

# Group average
mean_stc = stcs_fsaverage[0].copy()  # get copy of first instance
for sub in range(1, n_subjects):
    mean_stc._data += stcs_fsaverage[sub].data
mean_stc._data /= n_subjects

# Timecourse source figure
output_file = op.join(config.result_path, 'Correlation_complexity',  which+'_timecourse.png')
times_to_plot = [.0, .050, .100, .150, .200, .250, .300, .350, .400]
win_size = .050
source_estimation_funcs.timecourse_source_figure(mean_stc, which, times_to_plot, win_size, output_file)



######################
for subject in config.subjects_list:
    if config.noEEG:
        path_evo = op.join(config.meg_dir, subject, 'noEEG', 'evoked_resid')
    else:
        path_evo = op.join(config.meg_dir, subject, 'evoked_resid')
    utils.create_folder(path_evo)


########################################################
# Email Fosca 11/08/2021
########################################################

import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
from ABseq_func import regression_funcs
import config

# Exclude some subjects
config.exclude_subjects.append('sub16-ma_190185')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

# ==============================

filter_names = ['Hab', 'Stand', 'Viol']
for filter_name in filter_names:
    # Regression of structure regressors on surprise-regression residuals - group analysis
    # reg_names = ['Complexity','WithinChunkPosition','RepeatAlter','ChunkBeginning', 'ChunkEnd', 'ChunkNumber', 'ChunkDepth']
    # regression_funcs.merge_individual_regression_results(reg_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name)
    # regression_funcs.regression_group_analysis(reg_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, remap_grads=True, Do3Dplot=True)

    reg_names = ['RepeatAlter']
    regression_funcs.merge_individual_regression_results(reg_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name)
    regression_funcs.regression_group_analysis(reg_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, remap_grads=True, Do3Dplot=True)


########################################################
########################################################
