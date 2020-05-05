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
    # print('We remove items from trials with violation')
    # epochs = epochs["ViolationInSequence == 0"]

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


def run_linear_reg_surprise_repeat_alt(subject):

    TP_funcs.append_surprise_to_metadata_clean(subject)

    # ====== load the data , remove the first item for which the surprise is not computed ==========
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
    metadata = epoching_funcs.update_metadata(subject, clean=True, new_field_name=None, new_field_values=None)

    # ============ build the repeatAlter and the surprise 299 for n+1 ==================
    metadata_notclean = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None)
    RepeatAlternp1_notclean = metadata_notclean["RepeatAlter"].values[1:].tolist()
    RepeatAlternp1_notclean.append(np.nan)
    Surprisenp1_notclean = metadata_notclean["surprise_299"].values[1:].tolist()
    Surprisenp1_notclean.append(np.nan)
    good_idx = np.where([len(epochs.drop_log[i])==0 for i in range(len(epochs.drop_log))])[0]
    RepeatAlternp1 = np.asarray(RepeatAlternp1_notclean)[good_idx]
    Surprisenp1= np.asarray(Surprisenp1_notclean)[good_idx]
    # ======================================================================================


    metadata = metadata.assign(Intercept=1)  # Add an intercept for later
    metadata = metadata.assign(RepeatAlternp1=RepeatAlternp1)
    metadata = metadata.assign(Surprisenp1=Surprisenp1)  # Add an intercept for later

    epochs.metadata = metadata
    epochs.pick_types(meg=True, eeg=True)


    np.unique(metadata[np.isnan(epochs.metadata['RepeatAlter'])]['StimPosition'].values)
    np.unique(metadata[np.isnan(epochs.metadata['surprise_299'])]['StimPosition'].values)
    np.unique(metadata[np.isnan(metadata['RepeatAlternp1'])]['StimPosition'].values)
    np.unique(metadata[np.isnan(metadata['Surprisenp1'])]['StimPosition'].values)


    epochs = epochs[np.where(1 - np.isnan(epochs.metadata["surprise_299"].values))[0]]
    epochs = epochs[np.where(1 - np.isnan(epochs.metadata["RepeatAlternp1"].values))[0]]

    # =============== define the regressors =================
    # Repetition and alternation for n (not defined for the 1st item of the 16)
    # Repetition and alternation for n+1 (not defined for the last item of the 16)
    # Omega infinity for n (not defined for the 1st item of the 16)
    # Omega infinity for n+1 (not defined for the last item of the 16)

    names = ["Intercept", "surprise_299","Surprisenp1","RepeatAlter","RepeatAlternp1"]
    for name in names:
        print(name)
        print(np.unique(epochs.metadata[name].values))


    lin_reg = linear_regression(epochs, epochs.metadata[names], names=names)


    # Save surprise regression results
    out_path = op.join(config.result_path, 'linear_models', 'reg_repeataltern_surpriseOmegainfinity', subject)
    utils.create_folder(out_path)
    lin_reg['Intercept'].beta.save(op.join(out_path, 'beta_intercept-ave.fif'))
    lin_reg['surprise_299'].beta.save(op.join(out_path, 'beta_surpriseN-ave.fif'))
    lin_reg['Surprisenp1'].beta.save(op.join(out_path, 'beta_surpriseNp1-ave.fif'))
    lin_reg['RepeatAlternp1'].beta.save(op.join(out_path, 'beta_RepeatAlternp1-ave.fif'))
    lin_reg['RepeatAlter'].beta.save(op.join(out_path, 'beta_RepeatAlter-ave.fif'))


    # save the residuals epoch in the same folder

    residuals = epochs.get_data()-lin_reg['Intercept'].beta.data
    for nn in ["surprise_299","Surprisenp1","RepeatAlter","RepeatAlternp1"]:
        residuals = residuals - np.asarray([epochs.metadata[nn].values[i]*lin_reg[nn].beta._data for i in range(len(epochs))])

    residual_epochs = epochs.copy()
    residual_epochs._data = residuals

    # save the residuals epoch in the same folder
    residual_epochs.save(out_path+op.sep+'residuals-epo.fif', overwrite=True)


    return True