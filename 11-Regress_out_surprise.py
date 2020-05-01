from ABseq_func import TP_funcs, cluster_funcs, epoching_funcs
import config
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




path_optimal_omega = "/Volumes/COUCOU_CFC/ABSeq/results/TP_effects/surprise_omegas/omega_optimal_per_channels.npy"

# _______ optimal omega for each channel _____

omega_optimal = np.load(path_optimal_omega,allow_pickle=True).item()
omega_argmax = omega_optimal['omega_arg_max']
# fig = sns.heatmap(np.mean(omega_argmax,axis=0))
times_of_interest = [-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
inds_t = np.hstack([np.where(omega_optimal['time']==tim)[0] for tim in times_of_interest])
plt.figure(figsize=(10, 20))
plt.imshow(np.mean(omega_argmax,axis=0), origin='lower')
plt.yticks(inds_t, times_of_interest)
plt.colorbar()
plt.title("Optimal Omega f(time,channel)")
plt.ylabel("Time in sec")
plt.xlabel("Channel index")



# ___________ plot like plot joint the optimal omega _______________________
data_to_plot = np.mean(omega_argmax,axis=0)
epoch = epoching_funcs.load_epochs_items(config.subjects_list[0])
average = epoch.average()
average._data = data_to_plot.T
average.plot_joint()

# __________ average across channels _________
plt.plot(omega_optimal['time'],np.mean(data_to_plot,axis=1))
plt.xticks(times_of_interest)
plt.xlabel('Time')
plt.ylabel('Optimal Omega')
plt.title('Optimal Omega averaged over channels')



df_posterior = TP_funcs.for_plot_posterior_probability(config.subjects_list,omega_list=range(1,299))
df_post = np.mean(df_posterior['posterior'],axis=0)


times_of_interest = [-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
inds_t = np.hstack([np.where(df_posterior['time'][0]==tim)[0] for tim in times_of_interest])
plt.figure(figsize=(10, 20))
plt.imshow(df_post, vmin=0.001, vmax=0.006, origin='lower')
plt.xticks(inds_t, times_of_interest)
plt.colorbar()
plt.xlabel("Time in sec")
plt.ylabel("Omega")
plt.title("Posterior")


def compute_explained_variance(subject,clean=True,fname='residual_blabla-epo.fif'):

    import sklearn
    from sklearn.metrics import r2_score


    epochs = epoching_funcs.load_epochs_items(subject,cleaned=clean)
    metadata = epoching_funcs.update_metadata(subject, clean=clean, new_field_name=None, new_field_values=None)
    epochs.metadata = metadata
    epochs.pick_types(meg=True, eeg=True)
    epochs = epochs[np.where(1 - np.isnan(epochs.metadata["surprise_1"].values))[0]]
    y_true = epochs.get_data()


    epo_res = mne.read_epochs(op.join(config.meg_dir, subject,fname))
    # res = y_true - y_pred => y_pred = y_true - res

    y_pred = y_true - epo_res.get_data()


    R2 = r2_score(y_true,y_pred)

