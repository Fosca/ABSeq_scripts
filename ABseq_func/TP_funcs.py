# This module contains the functions related to the computation of transition probabilities, surprise etc.
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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
def from_epochs_to_surprise(subject, list_omegas,order = 1):

    """
    :param subject:
    :param list_omegas:
    :param order:
    :return:
    """

    print("================== The surprise is computed on the non-cleaned epochs and metadata  !!!! ==========")

    clean = False
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

# =====================================================================================================================

def append_surprise_to_metadata_clean(subject):
    """
    Load the metadata that contains the surprise for the non-clean epochs, removes the bad epochs from the metadata
    and this becomes the metadata for the clean epochs
    :param subject:
    :return:
    """

    meg_subject_dir = op.join(config.meg_dir, subject)
    if config.noEEG:
        meg_subject_dir = op.join(meg_subject_dir, 'noEEG')


    metadata_path = os.path.join(meg_subject_dir, 'metadata_item_clean.pkl')

    metadata = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None, recompute=False)
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
    good_idx = [len(epochs.drop_log[i])==0 for i in range(len(epochs.drop_log))]
    metadata_final = metadata[good_idx]

    with open(metadata_path,'wb') as fid:
        pickle.dump(metadata_final,fid)

    return True



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
def run_linear_regression_surprises(subject,omega_list,clean=False,decim = None,prefix='',Ridge=False,hfilter=20):

    epochs = epoching_funcs.load_epochs_items(subject, cleaned=clean)
    epochs.pick_types(meg=True, eeg=True)
    if hfilter is not None:
        epochs.filter(None,hfilter)

    if decim is not None:
        epochs.decimate(decim)

    metadata = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=None, new_field_values=None)
    epochs.metadata = metadata
    df = epochs.metadata
    epochs.metadata = df.assign(Intercept=1)
    r2_surprise = {omega:[] for omega in omega_list}
    r2_surprise['times'] = epochs.times
    epochs_for_reg = epochs[np.where(1 - np.isnan(epochs.metadata["surprise_1"].values))[0]]
    epochs_for_reg = epochs_for_reg["SequenceID != 1"]
    epochs_for_reg_normalized = normalize_data(epochs_for_reg)

    out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
    utils.create_folder(out_path)
    if not Ridge:
        for omega in omega_list:
            print("==== running the regression for omega %i ======="%omega)
            surprise_name = "surprise_%.005f" % omega
            r2_surprise[omega] = linear_regression_from_sklearn(epochs_for_reg_normalized, surprise_name)
        # ===== save all the regression results =========
        fname = prefix +'results_surprise.npy'
        np.save(op.join(out_path,fname),r2_surprise)

    else:
        surprise_names = ["surprise_%i" % omega for omega in omega_list]
        results_ridge = multi_ridge_regression_allIO(epochs_for_reg_normalized,surprise_names)
        fname = prefix +'results_Ridge_surprise.npy'
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
    the r2_score per sensor and per time and the mean squared error
    :param epochs_for_reg: the epochs data on which we run the regression
    :param names: the regression variables
    :return:
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

    y = epochs_for_reg_normalized.get_data()
    x = np.asarray(epochs_for_reg_normalized.metadata[surprise_name])

    results = {'regcoef_%s'%surprise_name:[]}
    results['regcoeff_intercept'] = []
    results['r2_score'] = []
    results['score_per_channel']= []
    results['mean_squared_error'] = []
    results['mean_squared_error_per_channel'] = []
    results['n_trials'] = y.shape[0]

    for time in range(y.shape[2]):
        reg = LinearRegression().fit(x[:,np.newaxis], y[:,:,time])
        results['regcoeff_intercept'].append(reg.intercept_)
        results['regcoef_%s'%surprise_name].append(reg.coef_)
        results['r2_score'].append(reg.score(x[:,np.newaxis], y[:,:,time]))
        y_pred = reg.predict(x[:,np.newaxis])
        r2 = [r2_score(y_pred[:,k],y[:,k,time]) for k in range(y.shape[1])]
        results['score_per_channel'].append(r2)
        mse_p_chan = [mean_squared_error(y_pred[:,k],y[:,k,time]) for k in range(y.shape[1])]
        results['mean_squared_error_per_channel'].append(mse_p_chan)
        results['mean_squared_error'].append(mean_squared_error(y_pred,y[:,:,time]))

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
def compute_posterior_probability(subject,fname = 'results_surprise.npy',omega_list=range(1,300)):
    """
    This function returns a dictionnary that contains, per time point, the posterior probability of each model.
    The interesting thing about this is that for each time point, the posterior sums to 1, so it normalizes the stuff and
    we should have a nicer plot.

    :param subject_ID:
    :param fname:
    :return:
    """

    posterior = {'per_channel': [],
                 'all_channels': [], 'times': [],'n_trials':[]}

    out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
    surprise_dict = np.load(op.join(out_path, fname), allow_pickle=True).item()
    posterior['times'] = surprise_dict['times']
    n = surprise_dict[1]['n_trials']
    posterior['n_trials'] = n

    for omega in omega_list:
        print("==== running the computation for omega %i =======" % omega)
        surprise = surprise_dict[omega]
        mse = surprise['mean_squared_error']
        mse_per_channel = surprise['mean_squared_error_per_channel']
        p_omega = []
        p_omega_chan = []
        for tt in range(mse.shape[0]):
            BIC_tt = n * np.log(mse[tt])
            p_tt = np.exp(-BIC_tt/2)
            p_tt_chan = [np.exp(n * np.log(mse_per_channel[tt,k])) for k in range(mse_per_channel.shape[1])]
            p_omega.append(p_tt)
            p_omega_chan.append(p_tt_chan)

        posterior['all_channels'].append(p_omega)
        posterior['per_channel'].append(p_omega_chan)


    posterior['all_channels'] = np.asarray(posterior['all_channels'])
    posterior['per_channel'] = np.asarray(posterior['per_channel'])

    # ========== and now we normalize across the values of omega =============

    for t in range(posterior['all_channels'].shape[1]):
        norm = np.sum(posterior['all_channels'][:,t])
        posterior['all_channels'][:, t] = posterior['all_channels'][:,t]/norm
        for k in range(posterior['per_channel'].shape[2]):
            norm_chan = np.sum(posterior['per_channel'][:,t,k])
            posterior['per_channel'][:,t,k] = posterior['per_channel'][:,t,k] / norm_chan

    np.save(op.join(config.result_path, 'TP_effects', 'surprise_omegas',subject,'posterior.npy'),posterior)


# ======================================================================================================================
def compute_optimal_omega_per_channel(subjects_list, fname='posterior.npy', omega_list=range(1, 300)):
    """
    We plot the posterior built from the mse of the model across all the sensors.

    :param subjects_list:
    :param fname:
    :return:
    """
    post_per_channels = []
    omegas = []
    times = []
    subject_number = []

    for ii, subject in enumerate(subjects_list):
        out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
        posterior = np.load(op.join(out_path, fname), allow_pickle=True).item()
        the_times = posterior['times']
        post_per_channels_per_subject = []
        # loop per time point on the mse to compute the posterior
        for omega in omega_list:
            post_per_channels_per_subject.append(posterior['per_channel'][omega])
            times.append(the_times)
            omegas.append([omega]*len(the_times))
            subject_number.append([ii]*len(the_times))
        post_per_channels.append(post_per_channels_per_subject)

    omega_max = []
    for ii, subject in enumerate(subjects_list):
        post_per_subj = np.asarray(post_per_channels[ii])
        omegas_max_subj = np.zeros((post_per_subj.shape[1],post_per_subj.shape[2]))
        for t in range(post_per_subj.shape[1]):
            for chan in range(post_per_subj.shape[2]):
                for_max = post_per_subj[:,t,chan]
                inds_max = np.where(for_max==np.max(for_max))[0]
                omegas_max_subj[t,chan] = omega_list[inds_max[0]]
        omega_max.append(omegas_max_subj)

    omega_max = np.asarray(omega_max)
    # ======== find the maximal value across omegas ================
    diction = {'omega_arg_max':omega_max,'omega':omega_list,'time':the_times}

    # =========== save omega optimal ===============
    out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', 'omega_optimal_per_channels.npy')
    np.save(out_path,diction)

    return diction



# ======================================================================================================================
def for_plot_posterior_probability(subjects_list, fname='posterior.npy', omega_list=range(1, 300)):
    """
    We plot the posterior built from the mse of the model across all the sensors.

    :param subjects_list:
    :param fname:
    :return:
    """
    post = []
    omegas = []
    times = []
    subject_number = []

    for ii, subject in enumerate(subjects_list):
        out_path = op.join(config.result_path, 'TP_effects', 'surprise_omegas', subject)
        posterior = np.load(op.join(out_path, fname), allow_pickle=True).item()
        the_times = posterior['times']
        post_subj = []
        # loop per time point on the mse to compute the posterior
        for omega in omega_list:
            post_subj.append(posterior['all_channels'][omega])
            times.append(the_times)
            omegas.append([omega]*len(the_times))
            subject_number.append([ii]*len(the_times))
        post.append(post_subj)

    post = np.asarray(post)
    diction = {'posterior':post,'omega':omegas,'time':times,'subject':subject_number}
    # dataFrame_posterior = pd.DataFrame.from_dict(diction)


    return diction


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


    # =============== LOOK AT THE DATA ACROSS SUBJECTS ===============
    r2_max = []
    r2_omegas_all_subj = np.squeeze(r2_omegas)
    omega_argmax_subj = np.zeros((19,r2_omegas_all_subj.shape[-1]))
    timet = range(r2_omegas_all_subj.shape[-1])
    for subj in range(19):
        for time in timet:
            omega_subjects = r2_omegas_all_subj[subj,:, time]
            for_max = omega_subjects
            max_val = np.max(for_max)
            r2_max.append(max_val)
            omega_argmax_subj[subj,time] = omega_list[np.where(for_max == max_val)[0][0]]

    mean = np.mean(omega_argmax_subj,axis=0)
    sem = np.std(omega_argmax_subj,axis=0)/np.sqrt(19)

    fig, ax = plt.subplots(1, 1)
    ax.fill_between(times, mean-sem,mean + sem, alpha=0.5)
    plt.plot(times,mean)
    plt.xticks([-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
    plt.xlabel('Times in seconds')
    plt.ylabel('Omegas')


    # ================ AVERAGE ACROSS PARTICIPANTS ==================
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


# ======================================================================================================================
def regress_out_optimal_omega_per_channel(subject, clean=True):

    # =========== load the optimal parameters =========
    load_optimal = op.join(config.result_path, 'TP_effects', 'surprise_omegas', 'omega_optimal_per_channels.npy')
    optimal_omega = np.load(load_optimal,allow_pickle=True).item()
    optimal_omega = optimal_omega['omega_arg_max']
    optimal_omega = np.mean(optimal_omega,axis=0)

    # =========== load the data on which to perform the regression =========
    epochs = epoching_funcs.load_epochs_items(subject,cleaned=clean)
    metadata = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=None, new_field_values=None)
    epochs.metadata = metadata
    epochs.pick_types(meg=True, eeg=True)
    epochs = epochs[np.where(1 - np.isnan(epochs.metadata["surprise_1"].values))[0]]
    y = epochs.get_data()

    # =========== we initialize the output =========
    n_trials, n_channels, n_times = y.shape
    residual_model_no_constant = np.zeros((n_trials, n_channels, n_times))
    residual_model_constant = np.zeros((n_trials, n_channels, n_times))
    residual_constant = np.zeros((n_trials, n_channels, n_times))
    residual_surprise = np.zeros((n_trials, n_channels, n_times))

    # ======== we run the regression for each time point and each channel ===============
    for time in range(y.shape[2]):
        for channel in range(y.shape[1]):
            print("----- running the regression for time %i and channel %i -----"%(time,channel))
            surprise_name = "surprise_%i" % int(np.round(optimal_omega[time,channel],0))
            x = np.asarray(epochs.metadata[surprise_name])
            x = x[:, np.newaxis]
            # ========== regression with constant ==============
            reg_with_constant = LinearRegression().fit(x, y[:, channel, time])
            # ========== regression without constant ==============
            reg_without_constant = LinearRegression(fit_intercept=False).fit(x, y[:, channel, time])

            residual_model_constant[:,channel,time] = y[:,channel,time] - reg_with_constant.predict(x)
            residual_constant[:,channel,time] = y[:,channel,time] - np.squeeze(reg_with_constant.intercept_*x)
            residual_surprise[:,channel,time] = y[:,channel,time] - np.squeeze(reg_with_constant.coef_*x)
            residual_model_no_constant[:,channel,time] = y[:,channel,time] - reg_without_constant.predict(x)

    # ============================================================================================================

    epo_residual_model_constant = epochs.copy()
    epo_residual_constant = epochs.copy()
    epo_residual_surprise = epochs.copy()
    epo_residual_model_no_constant = epochs.copy()

    epo_residual_model_constant._data = residual_model_constant
    epo_residual_constant._data = residual_constant
    epo_residual_surprise._data = residual_surprise
    epo_residual_model_no_constant._data = residual_model_no_constant


    save_name = op.join(config.meg_dir,subject,subject+'residual_model_constant-epo.fif')
    epo_residual_model_constant.save(save_name)

    save_name = op.join(config.meg_dir,subject,subject+'residual_constant-epo.fif')
    epo_residual_constant.save(save_name)

    save_name = op.join(config.meg_dir,subject,subject+'residual_surprise-epo.fif')
    epo_residual_surprise.save(save_name)

    save_name = op.join(config.meg_dir,subject,subject+'residual_model_no_constant-epo.fif')
    epo_residual_model_no_constant.save(save_name)




# ======================================================================================================================
def regress_surprise_in_cluster(subject_list, cluster_info, omega_list=range(1, 300), clean=True):
    """
    This function regresses the data within a cluster as a function of the surprise for all the omegas specified in omega_list.
    4 different types of regressions are considered,
    1 - 'original_data' : for each channel and time-point
    2 - 'average_time' : averaging the data across time
    3 - 'average_channels' : averaging the data across channels
    4 - 'average_channels_and_times' : averaging the data across channels and time

    :param subject_list:
    :param cluster_info: a dictionnary containing  the keys
    'sig_times': the significant times
    'channels_cluster' : the channels that are significant
    'ch_type': the type of channel
    :param omega_list: The list of omegas for which we compute the regressions.
    :return: dataFrame containing the results of all the regressions
    """

    sig_times = cluster_info['sig_times']
    sig_channels = cluster_info['channels_cluster']
    ch_type = cluster_info['ch_type']

    results = {subject: {} for subject in subject_list}

    for subject in subject_list:
        results[subject] = {'average_channels_and_times': {}, 'average_channels': {}, 'average_times': {},
                            'original_data': {}}

        epochs = epoching_funcs.load_epochs_items(subject, cleaned=clean)
        metadata = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=None, new_field_values=None)
        epochs.metadata = metadata

        if ch_type in ['grad', 'mag']:
            epochs.pick_types(meg=ch_type, eeg=False)
        elif ch_type in ['eeg']:
            epochs.pick_types(meg=False, eeg=True)
        else:
            print('Invalid ch_type')

        epochs = epochs[np.where(1 - np.isnan(epochs.metadata["surprise_1"].values))[0]]
        # ========= select the significant times and channels ======
        epochs.crop(tmin=np.min(sig_times), tmax=np.max(sig_times))
        epochs.pick(sig_channels)  # not sure this is working actually
        epochs_avg_time = epochs.copy()
        epochs_avg_time._data = np.transpose(np.mean(epochs_avg_time._data, axis=2)[:, np.newaxis], (0, 2, 1))
        epochs_avg_channels = epochs.copy()
        epochs_avg_channels._data = np.mean(epochs_avg_channels._data, axis=1)[:, np.newaxis]
        epochs_avg_channels_times = epochs.copy()
        epochs_avg_channels_times._data = np.mean(epochs_avg_time._data, axis=1)[:, np.newaxis]

        # ============== And now the regressions =============================================================
        for key in results[subject].keys():
            results[subject][key] = {omega: {} for omega in omega_list}

        for omega in omega_list:
            print("==== running the regression for omega %i =======" % omega)
            surprise_name = "surprise_%i" % omega
            results[subject]['original_data'][omega] = linear_regression_from_sklearn(epochs, surprise_name)
            results[subject]['average_times'][omega] = linear_regression_from_sklearn(epochs_avg_time,
                                                                                               surprise_name)
            results[subject]['average_channels'][omega] = linear_regression_from_sklearn(epochs_avg_channels,
                                                                                                  surprise_name)
            results[subject]['average_channels_and_times'][omega] =  linear_regression_from_sklearn(
                epochs_avg_channels_times, surprise_name)

    return results

