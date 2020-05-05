from __future__ import division
from mne.stats import linear_regression, fdr_correction, bonferroni_correction, permutation_cluster_1samp_test
from mne.viz import plot_compare_evokeds
import os.path as op
import pandas as pd
import mne
from mne.viz import plot_topomap
from mne.parallel import parallel_func
import numpy as np
import config
from matplotlib import pyplot as plt
from ABseq_func import *
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import pandas as pd
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable


for subject in [config.subjects_list[0]]:

    # =========== correction of the metadata with the surprise for the clean epochs ============
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


