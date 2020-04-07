import initialization_paths
import numpy as np
import csv
import os
import scipy.io as sio
import pandas as pd
import config
import os.path as op
import mne
import glob
import warnings
from ABseq_func import epoching_funcs,utils
import pickle
import MarkovModel_Python
from MarkovModel_Python import IdealObserver as IO
import matplotlib.pyplot as plt
from mne.stats import linear_regression

# ======================================================================================================================
def remove_pause(seq, post, Nitem):
    """
    Remove the pause (i.e. the items whose value is >= Nitem) from the posterior
    dictionary returned by the ideal observer.
    """
    ind = seq < Nitem
    new_post = {}
    for key in post.keys():
        if type(post[key]) is dict:
            new_post[key] = {}
            for sub_key in post[key].keys():
                new_post[key][sub_key] = post[key][sub_key][ind]
        else:
            new_post[key] = post[key][ind]
    return new_post


# ======================================================================================================================
def from_epochs_to_surprise(subject, list_omegas, clean = False,order = 1):


    metadata = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=None, new_field_values=None)
    # extract the sequences from the metadata, add the pauses as code '2' and concatenate them
    runs = np.unique(metadata['RunNumber'])

    for ome in list_omegas:
        print("========== DETERMINING THE SURPRISE FOR LEAKY WINDOW OF %i ITEMS ========="%ome)
        options = {'Decay': ome, 'prior_weight': 1}
        field_name = 'surprise_%i'%ome
        surprise = []
        for run in runs:
            metadata_run = metadata[metadata['RunNumber']==run]
            seq_with_pause = list(metadata_run['StimID'])
            for pos_insert in range(45*16,0,-16):
                seq_with_pause.insert(pos_insert,2)
            post_with_pause = IO.IdealObserver(np.asarray(seq_with_pause), 'fixed', Nitem=2,
                                               order=order, options=options)
            post_omega_run = remove_pause(np.asarray(seq_with_pause), post_with_pause, 2)
            surprise.append(post_omega_run['surprise'])

        surprise = np.concatenate(surprise)
        metadata_updated = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=field_name, new_field_values=surprise)

    return metadata_updated

# ======================================================================================================================
def correlate_surprise_regressors(subject, list_omegas, clean = False,plot_figure=False):

    metadata = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=None, new_field_values=None)

    var_names = ['surprise_%i'%k for k in list_omegas ]
    meta_surprise = metadata[var_names]
    correlation_matrix = np.asarray(meta_surprise.corr())

    if plot_figure:
        fig = plt.imshow(correlation_matrix)
        plt.xticks(range(len(list_omegas)),list_omegas)
        plt.yticks(range(len(list_omegas)),list_omegas)
        plt.title("Correlation of the Surprise for the different window sizes")
        plt.colorbar()

    return correlation_matrix

# ======================================================================================================================
def run_linear_regression_surprises(subject,omega_list,clean=False,decim = None):

    epochs = epoching_funcs.load_epochs_items(subject, cleaned=clean)
    epochs.filter(None,30)
    if decim is not None:
        epochs.decimate(decim)
    metadata = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=None, new_field_values=None)
    epochs.metadata = metadata
    df = epochs.metadata
    epochs.metadata = df.assign(Intercept=1)

    r2_surprise = {omega:[] for omega in omega_list}
    r2_surprise['times'] = epochs.times

    for omega in omega_list:
        print("==== running the regression for omega %i ======="%omega)
        surprise_name = "surprise_%i" % omega
        names = ["Intercept", surprise_name]
        epochs_for_reg = epochs[np.where(1-np.isnan(epochs.metadata[surprise_name].values))[0]]
        r2_surprise[omega] = linear_regression_from_sklearn(epochs_for_reg, names)

    # ===== save all the regression results =========

    out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
    utils.create_folder(out_path)
    np.save(op.join(out_path,'r2_surprise.npy'),r2_surprise)

    return True

# ======================================================================================================================
def linear_regression_from_sklearn(epochs_for_reg,names):
    from sklearn.linear_model import LinearRegression

    y = epochs_for_reg.get_data()
    x = np.asarray(epochs_for_reg.metadata[names])

    r2_all = []
    for time in range(y.shape[2]):
        reg = LinearRegression().fit(x, y[:,:,time])
        r2 = reg.score(x, y[:,:,0])
        r2_all.append([r2])

    return np.concatenate(r2_all)


# ======================================================================================================================

def plot_r2_surprise(subjects_list):

    mat_2_plot = []

    for subject in subjects_list:
        out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
        surprise_dict = np.load(op.join(out_path,'r2_surprise.npy'),allow_pickle=True).item()
        times = surprise_dict['times']
        del surprise_dict['times']
        surprise_df = pd.DataFrame.from_dict(surprise_dict)
        mat_2_plot.append(np.asarray(surprise_df))

    mat_2_plot = np.asarray(mat_2_plot)
    for_plot = np.asarray(np.mean(mat_2_plot, axis=0)).T
    plt.imshow(for_plot)
    plt.xticks(range(len(times)),times)
    plt.yticks(range(len(surprise_dict.keys())),list(surprise_dict.keys()))
    plt.title("r2 Brain Signal = f(Surprise_omega)")
    plt.colorbar()

    return mat_2_plot




