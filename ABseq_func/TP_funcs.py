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
def run_linear_regression_surprises(subject,omega_list,clean=False,decim = None,prefix='',Ridge=False):

    epochs = epoching_funcs.load_epochs_items(subject, cleaned=clean)
    epochs.pick_types(meg=True, eeg=True)
    # epochs.crop(tmin=0,tmax=0.3)
    epochs.filter(None,30)

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

    out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
    utils.create_folder(out_path)
    if not Ridge:
        for omega in omega_list:
            print("==== running the regression for omega %i ======="%omega)
            surprise_name = "surprise_%i" % omega
            r2_surprise[omega] = linear_regression_from_sklearn(epochs_for_reg_normalized, surprise_name)
        # ===== save all the regression results =========
        fname = prefix +'r2_surprise.npy'
        np.save(op.join(out_path,fname),r2_surprise)

    else:
        surprise_names = ["surprise_%i" % omega for omega in omega_list]
        results_ridge = multi_ridge_regression_allIO(epochs_for_reg_normalized,surprise_names)
        fname = prefix +'Ridge_surprise.npy'
        np.save(op.join(out_path,fname),results_ridge)

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
    results['score_per_channel']=[]

    for time in range(y.shape[2]):
        reg = LinearRegression().fit(x[:,np.newaxis], y[:,:,time])

        results['regcoeff_intercept'].append(reg.intercept_)
        results['regcoef_%s'%surprise_name].append(reg.coef_)
        results['r2_score'].append(reg.score(x[:,np.newaxis], y[:,:,time]))
        y_pred = reg.predict(x[:,np.newaxis])
        r2 = [r2_score(y_pred[:,k],y[:,k,time]) for k in range(y.shape[1])]
        results['score_per_channel'].append(r2)

        # y_pred = reg.predict(x[:,np.newaxis])
        # results['r2_score'].append(r2_score(y_pred, y[:,:,time]))
    # y_pred = reg.predict(x[:,np.newaxis])
    # y_pred_main = np.matmul(reg.intercept_[:,np.newaxis],np.ones(x[:,np.newaxis].T.shape)) + np.matmul(reg.coef_,x[:,np.newaxis].T)
    for key in results.keys():
        results[key] = np.asarray(results[key])


    results['times'] = epochs_for_reg_normalized.times

    return results


# ======================================================================================================================

def plot_r2_surprise(subjects_list,tmin=None,tmax = None, vmin = None,vmax=None,fname = 'r2_surprise.npy',omegas_of_interest = range(1,300),times_of_interest = [-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]):

    mat_2_plot = []
    for subject in subjects_list:
        out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
        surprise_dict = np.load(op.join(out_path,fname),allow_pickle=True).item()
        times = surprise_dict['times']
        mat_subj = []
        for omega in omegas_of_interest:
            surprise_omega = surprise_dict[omega]
            surprise_df = pd.DataFrame.from_dict(surprise_omega['r2_score'])
            mat_subj.append(np.asarray(surprise_df))
        del surprise_dict['times']
        mat_2_plot.append(mat_subj)

    mat_2_plot = np.asarray(mat_2_plot)
    for_plot = np.squeeze(np.asarray(np.mean(mat_2_plot, axis=0)))

    if tmax is not None:
        inds_tmin = np.where(times==tmin)[0][0]
        inds_tmax = np.where(times==tmax)[0][0]
        for_plot = for_plot[:,inds_tmin:inds_tmax]
        interval_true_false = np.logical_and(np.asarray(times_of_interest)<=(tmax),(np.asarray(times_of_interest)>=(tmin)))
        times_of_interest = list(np.asarray(times_of_interest)[np.where(interval_true_false)[0]])

    omegas_computed = np.asarray(list(surprise_dict.keys()))
    inds_t = [np.where(times == times_of_interest[k])[0][0] for k in range(len(times_of_interest))]

    plt.figure(figsize=(10,20))
    plt.imshow(for_plot,vmin=vmin,vmax = vmax,origin='lower')
    plt.xticks(inds_t,times_of_interest)
    if len(omegas_of_interest)>20:
        step = int(np.ceil(3*np.log(len(omegas_of_interest))))
        plt.yticks(for_plot.shape[::step],omegas_of_interest[::step])
    else:
        plt.yticks(range(for_plot.shape[0]), omegas_of_interest)
    plt.title("r2 Brain Signal = f(Surprise_omega)")
    plt.colorbar()

    return mat_2_plot, times





# ======================================================================================================================

def multi_ridge_regression_allIO(epochs_for_reg_normalized,surprise_names):
    """
    This function does a very simple linear regression. We store the predicted y_preds, the regression coefficients
    and the r2_score per sensor and per time
    :param epochs_for_reg: the epochs data on which we run the regression
    :param names: the regression variables
    :return:
    """
    from sklearn.linear_model import Ridge

    y = epochs_for_reg_normalized.get_data()
    x = np.asarray(epochs_for_reg_normalized.metadata[surprise_names])

    results = {'regcoef_%s'%surprise_name:[] for surprise_name in surprise_names}
    results['regcoeff_intercept'] = []
    results['score'] = []


    for time in range(y.shape[2]):
        reg = Ridge().fit(x, y[:,:,time])
        results['regcoeff_intercept'].append(reg.intercept_)
        for k in range(len(surprise_names)):
            results['regcoef_%s'%surprise_names[k]].append(reg.coef_[:,k])
        results['score'].append(reg.score(x, y[:,:,time]))

    for key in results.keys():
        results[key] = np.asarray(results[key])

    results['times'] = epochs_for_reg_normalized.times

    return results


# ======================================================================================================================
def compute_and_save_argmax_omega(omega_list=range(1, 300)):
    """
    This function averages over the participants and computes the value of omega that explains the most variance as a function of time
    :param omega_list:
    :return:
    """
    r2_omegas, times = plot_r2_surprise(config.subjects_list, fname='r2_surprise.npy', omegas_of_interest=omega_list)
    r2_max = []
    omega_argmax = []
    r2_omegas_mean = np.squeeze(np.mean(r2_omegas, axis=0))
    for time in range(r2_omegas_mean.shape[1]):
        for_max = r2_omegas_mean[:, time]
        max_val = np.max(for_max)
        r2_max.append(max_val)
        omega_argmax.append(omega_list[np.where(for_max == max_val)[0][0]])

    omega_argmax = np.asarray(omega_argmax)
    save_name = op.join(config.result_path, 'TP_effects', 'surprise_omegas', 'argmax_omega.npy')
    np.save(save_name, omega_argmax)

    return omega_argmax, times


# ======================================================================================================================
def regress_out_optimal_omega(subject,clean=True):
    """
    This function computes the regression for each time step and each optimal omega, i.e. that explains the most the variance
    :param epochs_for_reg: the epochs data on which we run the regression
    :param names: the regression variables
    :return:
    """

    save_name = op.join(config.result_path, 'TP_effects', 'surprise_omegas','argmax_omega.npy')
    omega_argmax = np.load(save_name)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    epochs = epoching_funcs.load_epochs_items(subject,cleaned=clean)
    metadata = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=None, new_field_values=None)
    epochs.metadata = metadata
    epochs.pick_types(meg=True, eeg=True)
    epochs = epochs[np.where(1 - np.isnan(epochs.metadata["surprise_1"].values))[0]]
    y = epochs.get_data()

    results = {}
    results['regcoeff_intercept'] = []
    results['regcoef_'] = []
    results['score'] = []
    results['omega'] = []
    results['residual'] = []
    results['times'] = epochs.times
    results['predictions'] = []
    results['score_per_channel'] = []

    for time in range(y.shape[2]):
        print("======= time step %i =========="%time)
        surprise_name = "surprise_%i"%omega_argmax[time]
        x = np.asarray(epochs.metadata[surprise_name])
        reg = LinearRegression().fit(x[:,np.newaxis], y[:,:,time])
        results['regcoeff_intercept'].append(reg.intercept_)
        results['regcoef_'].append(reg.coef_)
        results['omega'].append(omega_argmax[time])
        y_pred = reg.predict(x[:,np.newaxis])
        r2 = [r2_score(y_pred[:,k],y[:,k,time]) for k in range(y.shape[1])]
        results['score_per_channel'].append(r2)
        results['score'].append(reg.score(x[:,np.newaxis], y[:,:,time]))
        results['predictions'].append(np.matmul(reg.coef_,x[:,np.newaxis].T))
        y_residual_time = y[:,:,time] - np.matmul(reg.coef_,x[:,np.newaxis].T).T
        results['residual'].append(y_residual_time)

    for key in results.keys():
        results[key] = np.asarray(results[key])

    epochs_residual = epochs.copy()
    epochs_reg_coeff_surprise = epochs.copy()

    epochs_residual._data = np.transpose(results['residual'],(1,2,0))
    save_name = op.join(config.meg_dir,subject,subject+'_residuals_surprise-epo.fif')
    epochs_residual.save(save_name)

    # ============= it is the topomap of this that is going to tell us the contribution of the topography of the variance ====
    epochs_reg_coeff_surprise._data = np.transpose(results['predictions'],(1,2,0))
    save_name = op.join(config.meg_dir,subject,subject+'_regcoeff_surprise-epo.fif')
    epochs_reg_coeff_surprise.save(save_name)

    res_fname = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject,'residuals_results.npy')
    np.save(res_fname,results)


    return results


