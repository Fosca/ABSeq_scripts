from __future__ import division
from mne.stats import linear_regression, fdr_correction, bonferroni_correction
from mne.viz import plot_compare_evokeds
import os.path as op
import pandas as pd
import mne
from mne.parallel import parallel_func
import numpy as np
import config
from matplotlib import pyplot as plt
from ABseq_func import *
from scipy.io import loadmat
import pandas as pd
import pickle


def glm_surprise_complexity():
    # Load data

    for subject in config.subjects_list:

        epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)

        # add surprise_dynamic in metadata (excluding removed items/trials)
        print('We merge the dynamical model of surprise with the metadata')
        run_info_subject_dir = op.join(config.run_info_dir, subject)
        surprise = loadmat(op.join(run_info_subject_dir, 'surprise.mat'))
        surprise = list(surprise['Surprise'])
        badidx = np.where(epochs.drop_log)
        badidx = badidx[0]
        [surprise.pop(i) for i in badidx[::-1]]
        surprise = np.asarray(surprise)
        epochs.metadata['surprise_dynamic'] = surprise

        # ====== let's add in the metadata a term of violation_or_not X complexity ==========
        print('We remove the first sequence item for which the surprise is not well computed')
        epochs.metadata['violation_X_complexity'] = np.asarray([epochs.metadata['violation_or_not'][i]*epochs.metadata['Complexity'][i] for i in range(len(epochs.metadata))])
        # ====== remove the first item of each sequence in the linear model ==========
        print('We remove the first sequence item for which the surprise is not well computed')
        epochs = epochs["StimPosition > 1"]



        # Linear model (all items)
        df = epochs.metadata
        epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
        names = ["Intercept", "Complexity", "surprise_dynamic","violation_or_not","violation_X_complexity"]
        res = linear_regression(epochs, epochs.metadata[names], names=names)

        # Save regression results
        out_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic', subject)
        utils.create_folder(out_path)
        res['Intercept'].beta.save(op.join(out_path, 'beta_intercept-ave.fif'))
        res['Complexity'].beta.save(op.join(out_path, 'beta_Complexity-ave.fif'))
        res['surprise_dynamic'].beta.save(op.join(out_path, 'beta_surprise_dynamic-ave.fif'))
        res['violation_or_not'].beta.save(op.join(out_path, 'beta_violation_or_not-ave.fif'))
        res['violation_X_complexity'].beta.save(op.join(out_path, 'beta_violation_X_complexity-ave.fif'))


    intercept_evo = evoked_funcs.load_evoked('all',filter_name = 'beta_intercept',root_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))
    complexity_evo = evoked_funcs.load_evoked('all',filter_name = 'beta_Complexity',root_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))
    surprise_evo = evoked_funcs.load_evoked('all',filter_name = 'beta_surprise',root_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))
    violation_or_not_evo = evoked_funcs.load_evoked('all',filter_name = 'beta_violation_or_not',root_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))
    violation_X_complexity_evo = evoked_funcs.load_evoked('all',filter_name = 'beta_violation_X_complexity',root_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))


    intercept_epo = mne.EpochsArray(np.asarray([intercept_evo['beta_intercept-'][i][0].data for i in range(len(intercept_evo['beta_intercept-']))]) ,intercept_evo['beta_intercept-'][0][0].info,tmin=-0.1)
    complexity_epo = mne.EpochsArray(np.asarray([complexity_evo['beta_Complexity-'][i][0].data for i in range(len(complexity_evo['beta_Complexity-']))]) ,complexity_evo['beta_Complexity-'][0][0].info,tmin=-0.1)
    surprise_epo = mne.EpochsArray(np.asarray([surprise_evo['beta_surprise_dynamic-'][i][0].data for i in range(len(surprise_evo['beta_surprise_dynamic-']))]) ,surprise_evo['beta_surprise_dynamic-'][0][0].info,tmin=-0.1)
    violation_or_not_evo_epo = mne.EpochsArray(np.asarray([violation_or_not_evo['violation_or_not-'][i][0].data for i in range(len(surprise_evo['beta_surprise_dynamic-']))]) ,surprise_evo['beta_violation_or_not-'][0][0].info,tmin=-0.1)
    violation_X_complexity_epo = mne.EpochsArray(np.asarray([violation_X_complexity_evo['violation_X_complexity-'][i][0].data for i in range(len(surprise_evo['beta_surprise_dynamic-']))]) ,surprise_evo['beta_violation_X_complexity-'][0][0].info,tmin=-0.1)

    out_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic', 'group')
    utils.create_folder(out_path)
    intercept_epo.save(op.join(out_path, 'intercept_epo.fif'))
    complexity_epo.save(op.join(out_path, 'complexity_epo.fif'))
    surprise_epo.save(op.join(out_path, 'surprise_epo.fif'))
    violation_or_not_evo_epo.save(op.join(out_path, 'violation_or_not_epo.fif'))
    violation_X_complexity_epo.save(op.join(out_path, 'violation_X_complexity_epo.fif'))