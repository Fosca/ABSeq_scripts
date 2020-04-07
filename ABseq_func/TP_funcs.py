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
def run_linear_regression_surprises(subject,omega_list,clean=False,decim = None,prefix=''):

    epochs = epoching_funcs.load_epochs_items(subject, cleaned=clean)
    epochs.pick_types(meg=True, eeg=True)
    epochs.crop(tmin=0,tmax=0.3)
    # epochs.filter(None,30)

    if decim is not None:
        epochs.decimate(decim)

    metadata = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=None, new_field_values=None)
    epochs.metadata = metadata
    df = epochs.metadata
    epochs.metadata = df.assign(Intercept=1)
    r2_surprise = {omega:[] for omega in omega_list}
    r2_surprise['times'] = epochs.times
    epochs_for_reg = epochs[np.where(1 - np.isnan(epochs.metadata["surprise_1"].values))[0]]
    epochs_for_reg_normalized = normalize_data(epochs_for_reg)

    for omega in omega_list:
        print("==== running the regression for omega %i ======="%omega)
        surprise_name = "surprise_%i" % omega
        r2_surprise[omega] = linear_regression_from_sklearn(epochs_for_reg_normalized, surprise_name)

    # ===== save all the regression results =========

    out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
    utils.create_folder(out_path)
    fname = prefix +'r2_surprise.npy'
    np.save(op.join(out_path,fname),r2_surprise)

    return True


def normalize_data(epochs_to_normalize):

    from sklearn.preprocessing import StandardScaler
    y = epochs_to_normalize.get_data()
    scalers = {}
    for i in range(y.shape[1]):
        scalers[i] = StandardScaler()
        y[:, i, :] = scalers[i].fit_transform(y[:, i, :])

    epochs_to_normalize._data = y

    return epochs_to_normalize


# ======================================================================================================================
def linear_regression_from_sklearn(epochs_for_reg_normalized,surprise_name):
    """
    This function does a very simple linear regression. We store the predicted y_preds, the regression coefficients
    and the r2_score per sensor and per time
    :param epochs_for_reg: the epochs data on which we run the regression
    :param names: the regression variables
    :return:
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    y = epochs_for_reg_normalized.get_data()
    x = np.asarray(epochs_for_reg_normalized.metadata[surprise_name])

    results = {'regcoef_%s'%surprise_name:[]}
    results['regcoeff_intercept'] = []
    results['r2_score'] = []


    for time in range(y.shape[2]):
        reg = LinearRegression().fit(x[:,np.newaxis], y[:,:,time])

        results['regcoeff_intercept'].append(reg.intercept_)
        results['regcoef_%s'%surprise_name].append(reg.coef_)
        # results['r2_score'].append(reg.score(x[:,np.newaxis], y[:,:,time]))
        y_pred = reg.predict(x[:,np.newaxis])
        results['r2_score'].append(r2_score(y_pred, y[:,:,time]))
    # y_pred = reg.predict(x[:,np.newaxis])
    # y_pred_main = np.matmul(reg.intercept_[:,np.newaxis],np.ones(x[:,np.newaxis].T.shape)) + np.matmul(reg.coef_,x[:,np.newaxis].T)
    for key in results.keys():
        results[key] = np.asarray(results[key])

    results['times'] = epochs_for_reg_normalized.times

    return results


# ======================================================================================================================

def plot_r2_surprise(subjects_list,clean=False,tmin=None,tmax = None, vmin = 0,vmax=0.0002):

    times_of_interest = [-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    omegas_of_interest = [1,10,20,30,40,50,100,200,299]

    mat_2_plot = []

    for subject in subjects_list:
        out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
        fname = 'r2_surprise.npy'
        if clean:
            fname = 'clean_r2_surprise.npy'
        surprise_dict = np.load(op.join(out_path,fname),allow_pickle=True).item()
        times = surprise_dict['times']
        del surprise_dict['times']
        surprise_df = pd.DataFrame.from_dict(surprise_dict)
        mat_2_plot.append(np.asarray(surprise_df))

    mat_2_plot = np.asarray(mat_2_plot)
    for_plot = np.asarray(np.mean(mat_2_plot, axis=0)).T

    if tmax is not None:
        inds_tmin = np.where(times==tmin)[0][0]
        inds_tmax = np.where(times==tmax)[0][0]
        for_plot = for_plot[:,inds_tmin:inds_tmax]
        interval_true_false = np.logical_and(np.asarray(times_of_interest)<=(tmax),(np.asarray(times_of_interest)>=(tmin)))
        times_of_interest = list(np.asarray(times_of_interest)[np.where(interval_true_false)[0]])

    omegas_computed = np.asarray(list(surprise_dict.keys()))

    inds_t = [np.where(times == times_of_interest[k])[0][0] for k in range(len(times_of_interest))]
    inds_o = [np.where(omegas_computed == omegas_of_interest[k])[0][0] for k in range(len(omegas_of_interest))]

    plt.figure(figsize=(10,20))
    plt.imshow(for_plot,vmin=vmin,vmax = vmax)
    plt.xticks(inds_t,times_of_interest)
    plt.yticks(inds_o,omegas_of_interest)
    plt.title("r2 Brain Signal = f(Surprise_omega)")
    plt.colorbar()

    return mat_2_plot




