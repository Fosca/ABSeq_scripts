import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")

from ABseq_func import TP_funcs, cluster_funcs, epoching_funcs
import config
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mne
from jr.plot import pretty_decod

config.exclude_subjects.append('sub10-gp_190568')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

scores_regression_surprise = []
for subj in config.subjects_list:
    base_path = config.result_path+'/linear_models/reg_repeataltern_surpriseOmegainfinity/'+subj+'/scores_linear_reg_CV.npy'
    score_subj = np.load(base_path,allow_pickle=True)
    scores_regression_surprise.append(score_subj)

scores_regression_surprise = np.mean(np.asarray(scores_regression_surprise),axis=1)
times = [-0.048, -0.044, -0.04 , -0.036, -0.032, -0.028, -0.024, -0.02 ,
       -0.016, -0.012, -0.008, -0.004,  0.   ,  0.004,  0.008,  0.012,
        0.016,  0.02 ,  0.024,  0.028,  0.032,  0.036,  0.04 ,  0.044,
        0.048,  0.052,  0.056,  0.06 ,  0.064,  0.068,  0.072,  0.076,
        0.08 ,  0.084,  0.088,  0.092,  0.096,  0.1  ,  0.104,  0.108,
        0.112,  0.116,  0.12 ,  0.124,  0.128,  0.132,  0.136,  0.14 ,
        0.144,  0.148,  0.152,  0.156,  0.16 ,  0.164,  0.168,  0.172,
        0.176,  0.18 ,  0.184,  0.188,  0.192,  0.196,  0.2  ,  0.204,
        0.208,  0.212,  0.216,  0.22 ,  0.224,  0.228,  0.232,  0.236,
        0.24 ,  0.244,  0.248,  0.252,  0.256,  0.26 ,  0.264,  0.268,
        0.272,  0.276,  0.28 ,  0.284,  0.288,  0.292,  0.296,  0.3  ,
        0.304,  0.308,  0.312,  0.316,  0.32 ,  0.324,  0.328,  0.332,
        0.336,  0.34 ,  0.344,  0.348,  0.352,  0.356,  0.36 ,  0.364,
        0.368,  0.372,  0.376,  0.38 ,  0.384,  0.388,  0.392,  0.396,
        0.4  ,  0.404,  0.408,  0.412,  0.416,  0.42 ,  0.424,  0.428,
        0.432,  0.436,  0.44 ,  0.444,  0.448,  0.452,  0.456,  0.46 ,
        0.464,  0.468,  0.472,  0.476,  0.48 ,  0.484,  0.488,  0.492,
        0.496,  0.5  ,  0.504,  0.508,  0.512,  0.516,  0.52 ,  0.524,
        0.528,  0.532,  0.536,  0.54 ,  0.544,  0.548,  0.552,  0.556,
        0.56 ,  0.564,  0.568,  0.572,  0.576,  0.58 ,  0.584,  0.588,
        0.592,  0.596,  0.6  ]

pretty_decod(scores_regression_surprise,times)
plt.plot(times,scores_regression_surprise.T)
plt.title('r2 score CV 4 surprise')
plt.xlabel('time (s)')
plt.xticks([0,0.25,0.5])
plt.axvline([0])
plt.axvline([0.25])
plt.axvline([0.5])
plt.show()

betas_dict = {'intercept':[],'RepeatAlter':[],'RepeatAlternp1':[],'surpriseN':[],'surpriseNp1':[]}
mean_evoked = {'intercept':[],'RepeatAlter':[],'RepeatAlternp1':[],'surpriseN':[],'surpriseNp1':[]}
for key in betas_dict:
    for subj in config.subjects_list:
        base_path = config.result_path + '/linear_models/reg_repeataltern_surpriseOmegainfinity/' + subj +'/_cvbeta_'+key+'-ave.fif'
        evo = mne.read_evokeds(base_path)
        betas_dict[key].append(evo)

for key in mean_evoked.keys():
    list_evo = [betas_dict[key][i][0] for i in range(len(betas_dict[key]))]
    mean_evoked[key] = mne.combine_evoked(list_evo,weights='equal')
    fig_path = config.result_path + '/linear_models/reg_repeataltern_surpriseOmegainfinity/figures/Signals/cv_betas'
    [mag_fig,eeg_fig,grad_fig] = mean_evoked[key].plot_joint()
    mag_fig.savefig(fig_path+'_mag'+key+'.png')
    eeg_fig.savefig(fig_path+'_eeg'+key+'.png')
    grad_fig.savefig(fig_path+'_grad'+key+'.png')


betas_dict = {'Intercept':[],'Complexity':[]}
mean_evoked = {'Intercept':[],'Complexity':[]}
for key in betas_dict:
    for subj in config.subjects_list:
        base_path = config.result_path + 'linear_models/StandComplexity/TP_corrected_data/Signals/With_baseline_correction/data/' + subj +'/_cv'+key+'.fif'
        evo = mne.read_evokeds(base_path)
        betas_dict[key].append(evo)

for key in mean_evoked.keys():
    print(key)
    list_evo = [betas_dict[key][i][0] for i in range(len(betas_dict[key]))]
    mean_evoked[key] = mne.combine_evoked(list_evo,weights='equal')
    fig_path = config.result_path + 'linear_models/StandComplexity/TP_corrected_data/Signals/With_baseline_correction/figures/Signals/cv_betas'
    [mag_fig,eeg_fig,grad_fig] = mean_evoked[key].plot_joint()
    mag_fig.savefig(fig_path+'_mag'+key+'.png')
    eeg_fig.savefig(fig_path+'_eeg'+key+'.png')
    grad_fig.savefig(fig_path+'_grad'+key+'.png')



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

