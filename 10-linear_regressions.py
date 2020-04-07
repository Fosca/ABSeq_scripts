from __future__ import division
import mne
import config
from ABseq_func import *
from mne.stats import linear_regression, fdr_correction, bonferroni_correction, permutation_cluster_1samp_test
from mne.viz import plot_compare_evokeds
from mne.viz import plot_topomap
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
from scipy import stats
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========== LOAD INDIVIDUAL REGRESSION RESULTS AND SAVE THEM AS GROUP FIF FILES =========== #
# Load data from all subjects
intercept_evo = evoked_funcs.load_evoked('all', filter_name='beta_intercept', root_path=op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))
complexity_evo = evoked_funcs.load_evoked('all', filter_name='beta_Complexity', root_path=op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))
surprise_evo = evoked_funcs.load_evoked('all', filter_name='beta_surprise', root_path=op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))
violation_or_not_evo = evoked_funcs.load_evoked('all', filter_name='beta_violation_or_not', root_path=op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))
violation_X_complexity_evo = evoked_funcs.load_evoked('all', filter_name='beta_violation_X_complexity', root_path=op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic'))

# Store as epo objects
intercept_epo = mne.EpochsArray(np.asarray([intercept_evo['beta_intercept-'][i][0].data for i in range(len(intercept_evo['beta_intercept-']))]), intercept_evo['beta_intercept-'][0][0].info, tmin=-0.1)
complexity_epo = mne.EpochsArray(np.asarray([complexity_evo['beta_Complexity-'][i][0].data for i in range(len(complexity_evo['beta_Complexity-']))]), complexity_evo['beta_Complexity-'][0][0].info, tmin=-0.1)
surprise_epo = mne.EpochsArray(np.asarray([surprise_evo['beta_surprise_dynamic-'][i][0].data for i in range(len(surprise_evo['beta_surprise_dynamic-']))]), surprise_evo['beta_surprise_dynamic-'][0][0].info, tmin=-0.1)
violation_or_not_epo = mne.EpochsArray(np.asarray([violation_or_not_evo['beta_violation_or_not-'][i][0].data for i in range(len(violation_or_not_evo['beta_violation_or_not-']))]), violation_or_not_evo['beta_violation_or_not-'][0][0].info, tmin=-0.1)
violation_X_complexity_epo = mne.EpochsArray(np.asarray([violation_X_complexity_evo['beta_violation_X_complexity-'][i][0].data for i in range(len(violation_X_complexity_evo['beta_violation_X_complexity-']))]), violation_X_complexity_evo['beta_violation_X_complexity-'][0][0].info, tmin=-0.1)

out_path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic', 'group')
utils.create_folder(out_path)
intercept_epo.save(op.join(out_path, 'intercept_epo.fif'), overwrite=True)
complexity_epo.save(op.join(out_path, 'complexity_epo.fif'), overwrite=True)
surprise_epo.save(op.join(out_path, 'surprise_epo.fif'), overwrite=True)
violation_or_not_epo.save(op.join(out_path, 'violation_or_not_epo.fif'), overwrite=True)
violation_X_complexity_epo.save(op.join(out_path, 'violation_X_complexity_epo.fif'), overwrite=True)

# ======================= RELOAD GROUP REGRESSION RESULTS ====================== #
path = op.join(config.result_path, 'linear_models', 'complexity&surprisedynamic', 'group')
intercept_epo = mne.read_epochs(op.join(path, 'intercept_epo.fif'))
complexity_epo = mne.read_epochs(op.join(path, 'complexity_epo.fif'))
surprise_epo = mne.read_epochs(op.join(path, 'surprise_epo.fif'))
violation_or_not_epo = mne.read_epochs(op.join(path, 'violation_or_not_epo.fif'))
violation_X_complexity_epo = mne.read_epochs(op.join(path, 'violation_X_complexity_epo.fif'))

betas = {'intercept':intercept_epo,'complexity':complexity_epo,'surprise':surprise_epo,'violation':violation_or_not_epo,'Violation_X_complexity':violation_X_complexity_epo}


intercept_epo.average().plot_joint()

# ======================= PLOT THE BUTTERFLY OF THE REGRESSORS ====================== #


fig_path = op.join(config.fig_path, 'Linear_regressions', 'surprise_complexity_violation_interaction')
utils.create_folder(fig_path)
ylim_eeg = 0.3
ylim_mag = 20
ylim_grad = 4

# Butterfly plots for violations (one graph per sequence) - in EEG/MAG/GRAD
ylim = dict(eeg=[-ylim_eeg, ylim_eeg], mag=[-ylim_mag, ylim_mag], grad=[-ylim_grad, ylim_grad])
ts_args = dict(gfp=True, time_unit='s', ylim=ylim)
topomap_args = dict(time_unit='s')
times = 'peaks'

for x, regressor_name in enumerate(betas.keys()):
    evokeds = betas[regressor_name].average()

    fig = evokeds.plot_joint(ts_args=ts_args, title='EEG_' + regressor_name ,
                                   topomap_args=topomap_args, picks='eeg', times=times, show=False)

    fig_name = fig_path + op.sep + ('EEG_' + regressor_name + '.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name)
    plt.close(fig)

    # MAG
    fig = evokeds.plot_joint(ts_args=ts_args, title='MAG_' + regressor_name,
                                   topomap_args=topomap_args, picks='mag', times=times, show=False)
    fig_name = fig_path + op.sep + ('MAG_' + regressor_name+'.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name)
    plt.close(fig)

    # #GRAD
    fig = evokeds.plot_joint(ts_args=ts_args, title='GRAD_' + regressor_name,
                                   topomap_args=topomap_args, picks='grad', times=times, show=False)
    fig_name = fig_path + op.sep + ('GRAD_' + regressor_name+ '.png')
    print('Saving ' + fig_name)
    plt.savefig(fig_name)
    plt.close(fig)

# ======================= RUN CLUSTER STATISTICS ====================== #
ch_type = 'mag'
data_condition = violation_or_not_epo.copy()  # !!!!!!
fname = 'violation_or_not_epo'                # !!!!!!
# data_condition.apply_baseline(baseline=(-0.100, 0.0))  # baseline ?? (probably not at this step - betas)
data_condition.crop(tmin=0.0, tmax=0.600)  # crop

connectivity = mne.channels.find_ch_connectivity(data_condition.info, ch_type=ch_type)[0]
data = np.array([data_condition.pick_types(meg=ch_type, eeg=False)[c].get_data() for c in range(len(data_condition))])

data = np.transpose(np.squeeze(data), (0, 2, 1))  # transpose for clustering
cluster_stats = permutation_cluster_1samp_test(data, threshold=None, n_jobs=6, verbose=True, tail=0, n_permutations=1000,
                                               connectivity=connectivity, out_type='indices', check_disjoint=True, step_down_p=0.05)
T_obs, clusters, p_values, _ = cluster_stats
good_cluster_inds = np.where(p_values < 0.05)[0]
print("Good clusters: %s" % good_cluster_inds)

# PLOT CLUSTERS
plt.close('all')
# set_matplotlib_defaults()
times = data_condition.times * 1e3
colors = 'r', 'steelblue'
linestyles = '-', '--'
pos = mne.find_layout(data_condition.info).pos
T_obs_max = 5.
T_obs_min = -T_obs_max
# loop over significant clusters
for i_clu, clu_idx in enumerate(good_cluster_inds):

    # unpack cluster information, get unique indices
    time_inds, space_inds = np.squeeze(clusters[clu_idx])
    ch_inds = np.unique(space_inds)
    time_inds = np.unique(time_inds)

    # get topography for T0 stat
    T_obs_map = T_obs[time_inds, ...].mean(axis=0)

    # get signals at significant sensors
    signals = data[..., ch_inds].mean(axis=-1)
    sig_times = times[time_inds]

    # create spatial mask
    mask = np.zeros((T_obs_map.shape[0], 1), dtype=bool)
    mask[ch_inds, :] = True

    # initialize figure
    fig, ax_topo = plt.subplots(1, 1, figsize=(7, 2.))

    # plot average test statistic and mark significant sensors
    image, _ = plot_topomap(T_obs_map, pos, mask=mask, axes=ax_topo,
                            vmin=T_obs_min, vmax=T_obs_max,
                            show=False)

    # advanced matplotlib for showing image with figure and colorbar
    # in one plot
    divider = make_axes_locatable(ax_topo)

    # add axes for colorbar
    ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(image, cax=ax_colorbar, format='%0.1f')
    ax_topo.set_xlabel('Averaged t-map\n({:0.1f} - {:0.1f} ms)'.format(
        *sig_times[[0, -1]]
    ))
    # ax_topo.annotate(chr(65 + 2 * i_clu), (0.1, 1.1), **annot_kwargs)

    # add new axis for time courses and plot time courses
    ax_signals = divider.append_axes('right', size='300%', pad=1.2)
    for signal, name, col, ls in zip(signals, [fname], colors,
                                     linestyles):
        ax_signals.plot(times, signal * 1e6, color=col,
                        linestyle=ls, label=name)

    # add information
    ax_signals.axvline(0, color='k', linestyle=':', label='stimulus onset')
    ax_signals.set_xlim([times[0], times[-1]])
    ax_signals.set_xlabel('Time [ms]')
    ax_signals.set_ylabel('Amplitude [uV]')

    # plot significant time range
    ymin, ymax = ax_signals.get_ylim()
    ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                             color='orange', alpha=0.3)
    ax_signals.legend(loc='lower right')
    title = 'Cluster #{0} (p < {1:0.3f})'.format(i_clu + 1, p_values[clu_idx])
    ax_signals.set(ylim=[ymin, ymax], title=title)
    # ax_signals.annotate(chr(65 + 2 * i_clu + 1), (-0.125, 1.1), **annot_kwargs)

    # clean up viz
    fig.tight_layout(pad=0.5, w_pad=0)
    fig.subplots_adjust(bottom=.05)
    # plt.savefig(op.join('..', 'figures',
    #                     'spatiotemporal_stats_cluster_highpass-%sHz-%02d.pdf'
    #                     % (l_freq, i_clu)))
    plt.show()

# =========================================================== #
# ==========  cluster data plot
# =========================================================== #

# ------------------ LOAD THE EVOKED OF INTEREST ------------ #
# one key per sequence ID
evoked_standard_seq = evoked_funcs.load_evoked(subject='all', filter_name='standard_seq', filter_not='pos')  #
evoked_viol_seq = evoked_funcs.load_evoked(subject='all', filter_name='viol_seq', filter_not='pos')  #
evoked_balanced_standard_seq = evoked_funcs.load_evoked(subject='all', filter_name='balanced_standard_seq', filter_not='pos')  #

# all sequences pooled together
evoked_all_standard = evoked_funcs.load_evoked(subject='all', filter_name='all_standard', filter_not=None)
evoked_all_viol = evoked_funcs.load_evoked(subject='all', filter_name='all_viol', filter_not=None)
evoked_all_standard_balanced = evoked_funcs.load_evoked(subject='all', filter_name='balanced', filter_not=None)

# ----------------- SELECT CHANNELS OF INTEREST ------------ #
clu_idx = good_cluster_inds[0]  # index of the cluster (among significant ones)
time_inds, space_inds = np.squeeze(clusters[clu_idx])
ch_inds = np.unique(space_inds)  # list of channels we want to average and plot (!? after pick_types or from all ?!)

# # OR ONE CHANNEL ?
# ch_name = 'MEG1131'
# ch_inds = [data_condition.ch_names.index(ch_name)]  # MEG1621 // MEG1131

# ==== Explore data... (plot difference all stand vs. all viol)
list_evoked_stand = evoked_all_standard['all_standard-']
list_evoked_stand = [list_evoked_stand[i][0] for i in range(len(list_evoked_stand))]
grand_average_stand = mne.grand_average(list_evoked_stand)
grand_average_stand.copy().crop(tmin=0.0, tmax=0.300).plot_joint()
list_evoked_viol = evoked_all_viol['all_viol-']
list_evoked_viol = [list_evoked_viol[i][0] for i in range(len(list_evoked_viol))]
grand_average_viol = mne.grand_average(list_evoked_viol)
# grand_average_viol.plot_joint()
evoked_contrast = mne.combine_evoked([grand_average_stand, grand_average_viol], weights=[1, -1])
evoked_contrast.plot_joint()
data_condition.plot_sensors(kind='select', ch_groups='position')

# Make empty (fake) topomap just to show the cluster/channels of interest
fdata = T_obs_map
mask = np.zeros((fdata.shape[0], 1), dtype=bool)
mask[ch_inds, :] = True
fig, ax_topo = plt.subplots(1, 1, figsize=(5, 5))
image, _ = plot_topomap(fdata, pos, mask=mask, axes=ax_topo, vmin=0, vmax=0, cmap='Greys', contours=0, mask_params=dict(markerfacecolor='r', markersize=6))
# Topomap with data, to show the cluster/channels of interest
# dat = grand_average_stand.copy().pick_types(meg='mag')
dat = evoked_contrast.copy().pick_types(meg='mag')
mask = np.zeros((len(dat.ch_names), len(dat.times)), dtype=bool)
mask[ch_inds, ...] = True
dat.plot_topomap(times=0.168, mask=mask, mask_params=dict(markerfacecolor='yellow', markersize=16))


# --------------------------    PLOTS            ------------ #
# TWO FIGURES, EACH 7 SEQUENCES, STAND & VIOL
plt.close('all')
evoked_funcs.plot_evoked_with_sem_7seq(evoked_standard_seq, ch_inds, label=None)
evoked_funcs.plot_evoked_with_sem_7seq(evoked_viol_seq, ch_inds, label=None)

# ONE FIGURE, AVERAGE ALL SEQUENCES, STAND vs VIOL
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plt.axvline(0, linestyle='-', color='black', linewidth=2)
for xx in range(3):
    plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
ax.set_xlabel('Time (ms)')
cond = 'all_standard-'
data = evoked_all_standard[cond].copy()
evoked_funcs.plot_evoked_with_sem_1cond(data, cond, ch_inds, color='b', filter=True)
grd_avg1 = mne.grand_average([data[i][0] for i in range(len(data))])
cond = 'all_viol-'
data = evoked_all_viol[cond].copy()
evoked_funcs.plot_evoked_with_sem_1cond(data, cond, ch_inds, color='r', filter=True)
grd_avg2 = mne.grand_average([data[i][0] for i in range(len(data))])
cond = 'balanced_standard-'
data = evoked_all_standard_balanced[cond].copy()
evoked_funcs.plot_evoked_with_sem_1cond(data, cond, ch_inds, color='dodgerblue', filter=True)
plt.legend(loc='upper right', fontsize=9)
ax.set_xlim([-100, 750])
# fig.savefig(fig_name_save, bbox_inches='tight', dpi=300)
# plt.close('all')

evoked_contrast = mne.combine_evoked([grd_avg2, grd_avg1], weights=[1, -1])
evoked_contrast.plot_joint()
evoked_contrast.plot(picks=ch_name)

# ONE FIGURE, ONE SEQUENCE, STAND vs VIOL -- VERSION2
# ch_inds = [evoked_all_standard['all_standard-'][0][0].ch_names.index('MEG1621')]
plt.close('all')
fig, axes = plt.subplots(7, 1, figsize=(5, 14), sharex=True, sharey=True, constrained_layout=True)
ax = axes.ravel()[::1]
for seqID in range(7):
    condS = 'standard_seq' + str(seqID+1) + '-'
    condV = 'viol_seq' + str(seqID+1) + '-'
    condS2 = 'balanced_standard_seq' + str(seqID+1) + '-'
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax[seqID].axvline(0, linestyle='-', color='black', linewidth=2)
    for xx in range(3):
        ax[seqID].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax[seqID].set_xlabel('Time (ms)')
    ax[seqID].set_title('Sequence ' + str(seqID+1), loc='right', weight='bold')
    # data = evoked_standard_seq[condS].copy()
    # evoked_funcs.plot_evoked_with_sem_1cond(data, condS, ch_inds, color='b', filter=True, axis=ax[seqID])
    data = evoked_viol_seq[condV].copy()
    evoked_funcs.plot_evoked_with_sem_1cond(data, condV, ch_inds, color='r', filter=True, axis=ax[seqID])
    data = evoked_balanced_standard_seq[condS2].copy()
    evoked_funcs.plot_evoked_with_sem_1cond(data, condS2, ch_inds, color='b', filter=True, axis=ax[seqID])
    # plt.legend(loc='upper right', fontsize=9)
    ax[seqID].set_xlim([-100, 750])
info = ch_name
# info = 'complexity_cluster'
fig.savefig(op.join(config.fig_path, 'stand_vs_seq_ch_' + ch_type + '_' + info), bbox_inches='tight', dpi=300)


def previous_scripting():

    #  =================================================================================================================
    # let's see if the response amplitude is modulated by SURPRISE + COMPLEXITY in the standard sequences (no violations)
    #  =================================================================================================================

    # Data: all items from not-violated sequences
    epo_noviol = epochs['ViolationInSequence == "0"'].copy()

    # # Plot surprise GFP
    # name = "surprise_dynamic"
    # df = epo_noviol.metadata
    # evokeds_no_viol = {str(val): epo_noviol[name + " == " + str(val)].average().savgol_filter(20) for val in np.unique(epo_noviol.metadata['surprise_dynamic'].values)}
    # colors_rgb = [[val/3,0,1-val/3] for val in np.unique(epo_noviol.metadata['surprise_dynamic'].values)]
    # magnifiques_couleurs = ['palegreen','mediumturquoise','darkslategray','blue','navy','rebeccapurple','mediumorchid','pink','crimson','red','brown']
    # mne.viz.plot_compare_evokeds(evokeds_no_viol,colors=colors_rgb,show_legend = 'upper right')

    # # Plot Complexity GFP
    # name = "Complexity"
    # df = epo_noviol.metadata
    # colors = {'4.0':[255/255,128/255,0], '6.0':[204/255,204/255,0], '12.0':[100/255,102/255,0], '14.0':[0,102/255,102/255], '23.0':[51/255,0,102/255]}
    # evokeds_no_viol = {val: epo_noviol[name + " == " + val].average().savgol_filter(20) for val in colors}
    # mne.viz.plot_compare_evokeds(evokeds_no_viol, colors=colors, split_legend=True,
    #                      cmap=(name + " Percentile", "viridis"))

    # Linear model
    df = epo_noviol.metadata
    epo_noviol.metadata = df.assign(Intercept=1)  # Add an intercept for later
    names = ["Intercept", "Complexity", "surprise_dynamic"]
    res = linear_regression(epo_noviol, epo_noviol.metadata[names], names=names)

    # Butterfly plot for each regressor
    plt.close('all')
    for cond in names:
        res[cond].beta.plot_joint(title=cond, ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))

    # Plots for suprise effect
    reject_H0, fdr_pvals = fdr_correction(res["surprise_dynamic"].p_val.data)
    # reject_H0, bonf_pvals = bonferroni_correction(res["Surprise"].p_val.data)
    evoked1 = res["surprise_dynamic"].beta
    evoked1.plot_image(mask=reject_H0, time_unit='s', picks='eeg')
    evoked1.plot_image(mask=reject_H0, time_unit='s', picks='mag')
    evoked1.plot_image(mask=reject_H0, time_unit='s', picks='grad')
    times = np.arange(-0.100, 0.700, 0.005)
    evoked1.animate_topomap(ch_type='mag', times=times, frame_rate=10, time_unit='s')

    # Plots for complexity effect
    reject_H0, fdr_pvals = fdr_correction(res["Complexity"].p_val.data)
    reject_H0, bonf_pvals = bonferroni_correction(res["Complexity"].p_val.data)
    evoked2 = res["Complexity"].beta
    evoked2.plot_image(mask=reject_H0, time_unit='s', picks='eeg')
    evoked2.plot_image(mask=reject_H0, time_unit='s', picks='mag')
    evoked2.plot_image(mask=reject_H0, time_unit='s', picks='grad')
    times = np.arange(-0.100, 0.700, 0.005)
    evoked2.animate_topomap(ch_type='mag', times=times, frame_rate=10, time_unit='s')

    chans = evoked2.ch_names
    signif = []
    for nchan in range(reject_H0.shape[0]):
        signif.append(sum(reject_H0[nchan,:]))
    chans = np.asarray(chans)
    signifchans = chans[np.asarray(signif)>0]
    print(signifchans)

    mne.viz.plot_compare_evokeds(evoked1, picks=list(signifchans)[0:21])
    mne.viz.plot_compare_evokeds(evoked2, picks=list(signifchans)[0:21])
    mne.viz.plot_compare_evokeds(dict(Complexity=evoked2, Surprise=evoked1), picks=list(signifchans)[0:19], show_legend='upper left')
    plt.close('all')

    signifchans2 = sum(reject_H0[:,:])>0
    signifchans2 = [i for i, x in enumerate(signifchans2) if x]
    epo_noviol._channel_type_idx['mag']


    evoked.plot( time_unit='s', picks=['MEG1431'])

    #  =================================================================================================================
    # what about the violation trials
    #  =================================================================================================================

    # Data: violatation items
    epo_viol = epochs['ViolationOrNot == "1"'].copy()

    # Plot surprise GFP
    name = "Surprise"
    df = epo_viol.metadata
    evokeds_viol = {str(val): epo_viol[name + " == " + str(val)].average().savgol_filter(20) for val in np.unique(epo_viol.metadata['Surprise'].values)}
    colors_rgb = [[val/3,0,1-val/3] for val in np.unique(epo_viol.metadata['Surprise'].values)]
    magnifiques_couleurs = ['palegreen','mediumturquoise','darkslategray','blue','navy','rebeccapurple','mediumorchid','pink','crimson','red','brown']
    # mne.viz.plot_compare_evokeds(evokeds_viol, colors=colors_rgb, show_legend = 'upper right')
    mne.viz.plot_compare_evokeds(evokeds_viol, show_legend = 'upper right')

    # Plot Complexity GFP
    name = "Complexity"
    df = epo_viol.metadata
    colors = {'4.0':[255/255,128/255,0], '6.0':[204/255,204/255,0], '12.0':[100/255,102/255,0], '14.0':[0,102/255,102/255], '23.0':[51/255,0,102/255]}
    evokeds_viol = {val: epo_viol[name + " == " + val].average().savgol_filter(20) for val in colors}
    mne.viz.plot_compare_evokeds(evokeds_viol, colors=colors, split_legend=True,
                         cmap=(name + " Percentile", "viridis"))

    # Linear model
    epo_viol.metadata = df.assign(Intercept=1)  # Add an intercept for later
    names = ["Intercept", "Complexity", "Surprise"]
    names = ["Intercept", "Complexity"]
    res = linear_regression(epo_viol, epo_viol.metadata[names], names=names)

    # Butterfly plot for each regressor
    plt.close('all')
    for cond in names:
        res[cond].beta.plot_joint(title=cond, ts_args=dict(time_unit='s'),topomap_args=dict(time_unit='s'))

    # Plots for suprise effect
    reject_H0, fdr_pvals = fdr_correction(res["Surprise"].p_val.data)
    # reject_H0, bonf_pvals = bonferroni_correction(res["Surprise"].p_val.data)
    evoked1 = res["Surprise"].beta
    evoked1.plot_image(mask=reject_H0, time_unit='s', picks='eeg')
    evoked1.plot_image(mask=reject_H0, time_unit='s', picks='mag')
    evoked1.plot_image(mask=reject_H0, time_unit='s', picks='grad')
    # times = np.arange(-0.250, 0.750, 0.005)
    # evoked1.animate_topomap(ch_type='mag', times=times, frame_rate=10, time_unit='s')

    # Plots for complexity effect
    reject_H0, fdr_pvals = fdr_correction(res["Complexity"].p_val.data)
    # reject_H0, bonf_pvals = bonferroni_correction(res["Complexity"].p_val.data)
    evoked2 = res["Complexity"].beta
    evoked2.plot_image(mask=reject_H0, time_unit='s', picks='eeg')
    evoked2.plot_image(mask=reject_H0, time_unit='s', picks='mag')
    evoked2.plot_image(mask=reject_H0, time_unit='s', picks='grad')
    # times = np.arange(-0.250, 0.750, 0.005)
    # evoked2.animate_topomap(ch_type='mag', times=times, frame_rate=10, time_unit='s')

    plt.close('all')
    fig = evoked2.plot_image(mask=reject_H0, time_unit='s', picks=evoked2.ch_names[0:50], show_names='all')
    fig.set_size_inches(20, 10)
    fig = evoked2.plot_image(mask=reject_H0, time_unit='s', picks=evoked2.ch_names[51:100], show_names='all')
    fig.set_size_inches(20, 10)
    fig = evoked2.plot_image(mask=reject_H0, time_unit='s', picks=evoked2.ch_names[101:150], show_names='all')
    fig.set_size_inches(20, 10)
    fig = evoked2.plot_image(mask=reject_H0, time_unit='s', picks=evoked2.ch_names[151:200], show_names='all')
    fig.set_size_inches(20, 10)

    chans = evoked2.ch_names
    signif = []
    for nchan in range(reject_H0.shape[0]):
        signif.append(sum(reject_H0[nchan,:]))
    chans = np.asarray(chans)
    signifchans = chans[np.asarray(signif)>0]
    print(signifchans)

    chans_of_interest = list(signifchans)[0:77]
    chans_of_interest = ['MEG1621', 'MEG0223']
    mne.viz.plot_compare_evokeds(evoked1, picks=chans_of_interest)
    mne.viz.plot_compare_evokeds(evoked2, picks=chans_of_interest)






