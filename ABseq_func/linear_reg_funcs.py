from __future__ import division
import mne
from ABseq_func import *
import config
from mne.stats import linear_regression, fdr_correction, bonferroni_correction
from mne.viz import plot_compare_evokeds
from mne.parallel import parallel_func
import os.path as op
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import pickle
from importlib import reload
from scipy.stats import sem
from sklearn.preprocessing import scale

def run_linear_regression(subject, cleaned=True):

    # Load data
    if cleaned:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
        epochs = epoching_funcs.update_metadata(subject, epochs)
    else:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
        epochs = epoching_funcs.update_metadata(subject, epochs)

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

    # ====== remove the first item of each sequence in the linear model ==========
    print('We remove the first sequence item for which the surprise is not well computed')
    epochs = epochs["StimPosition > 1"]

    # ====== normalization ?
    epochs.metadata['surprise_dynamic'] = scale(epochs.metadata['surprise_dynamic'])
    epochs.metadata['Complexity'] = scale(epochs.metadata['Complexity'])
    epochs.metadata['ViolationOrNot'] = scale(epochs.metadata['ViolationOrNot'])

    # ====== let's add in the metadata a term of violation_or_not X complexity ==========
    print('We remove the first sequence item for which the surprise is not well computed')
    # epochs.metadata['violation_X_complexity'] = np.asarray([epochs.metadata['ViolationOrNot'][i]*epochs.metadata['Complexity'][i] for i in range(len(epochs.metadata))])  # does not work, replaced by the next line (correct?)
    epochs.metadata['violation_X_complexity'] = scale(epochs.metadata['ViolationOrNot']*epochs.metadata['Complexity'])

    # Linear model (all items)
    df = epochs.metadata
    epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
    names = ["Intercept", "Complexity", "surprise_dynamic", "ViolationOrNot", "violation_X_complexity"]
    res = linear_regression(epochs, epochs.metadata[names], names=names)

    # Save regression results
    out_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic', subject)
    utils.create_folder(out_path)
    res['Intercept'].beta.save(op.join(out_path, 'beta_intercept-ave.fif'))
    res['Complexity'].beta.save(op.join(out_path, 'beta_Complexity-ave.fif'))
    res['surprise_dynamic'].beta.save(op.join(out_path, 'beta_surprise_dynamic-ave.fif'))
    res['ViolationOrNot'].beta.save(op.join(out_path, 'beta_violation_or_not-ave.fif'))
    res['violation_X_complexity'].beta.save(op.join(out_path, 'beta_violation_X_complexity-ave.fif'))


def run_linear_regression_v2(analysis_name, regressor_names, subject, cleaned=True):

    # Load data & update metadata (in case new things were added)
    if cleaned:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
        epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
    else:
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
        epochs = epoching_funcs.update_metadata_rejected(subject, epochs)

    # # ====== remove some items from the linear model ==========
    print('We remove the first sequence item for which the surprise is not well computed and for which there is no RepeatAlter')
    epochs = epochs["StimPosition > 1"]
    print('We remove items from trials with violation')
    epochs = epochs["ViolationInSequence == 0"]

    # ====== regressors
    names = regressor_names

    # ====== normalization ?
    for name in names:
        epochs.metadata[name] = scale(epochs.metadata[name])

    # ====== Linear model (all items)
    df = epochs.metadata
    epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
    names = ["Intercept"] + names
    res = linear_regression(epochs, epochs.metadata[names], names=names)

    # Save regression results
    out_path = op.join(config.result_path, 'linear_models', analysis_name, subject)
    utils.create_folder(out_path)
    for name in names:
        res[name].beta.save(op.join(out_path, name + '.fif'))


    # # TO REVIEW RESULTS...
    # import matplotlib.pyplot as plt
    # path = op.join(config.result_path, 'linear_models', 'test', subject)
    # Intercept = mne.read_evokeds(op.join(path, 'Intercept.fif'))
    # StimPosition = mne.read_evokeds(op.join(path, 'StimPosition.fif'))
    # RepeatAlter = mne.read_evokeds(op.join(path, 'RepeatAlter.fif'))
    # OpenedChunks = mne.read_evokeds(op.join(path, 'OpenedChunks.fif'))
    # Intercept[0].plot_joint()
    # StimPosition[0].plot_joint()
    # RepeatAlter[0].plot_joint()
    # OpenedChunks[0].plot_joint()
    # plt.close('all')


