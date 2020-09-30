from __future__ import division
from mne.stats import linear_regression, permutation_cluster_1samp_test
import os.path as op
import mne
from mne.viz import plot_topomap
import numpy as np
import config
from matplotlib import pyplot as plt
from ABseq_func import *
from scipy.io import loadmat
from sklearn.preprocessing import scale
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================
# ======== 1// regress the surprise ==========
# ============================================

for subject in config.subjects_list:

    # ====== load the data and remove the first item of each sequence in the linear model ==========
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
    epochs = epochs["StimPosition > 1"]
    # ====== load the surprise =============
    run_info_subject_dir = op.join(config.run_info_dir, subject)
    surprise = loadmat(op.join(run_info_subject_dir, 'surprise.mat'))
    surprise = list(surprise['Surprise'])
    badidx = np.where(epochs.drop_log)
    badidx = badidx[0]
    [surprise.pop(i) for i in badidx[::-1]]
    surprise = np.asarray(surprise)

    epochs.metadata['surprise_dynamic'] = scale(surprise)

    df = epochs.metadata
    epochs.metadata = df.assign(Intercept=1)  # Add an intercept for later
    names = ["Intercept", "surprise_dynamic"]
    lin_reg = linear_regression(epochs, epochs.metadata[names], names=names)

    # Save surprise regression results
    out_path = op.join(config.result_path, 'linear_models', 'surprisedynamic', subject)
    utils.create_folder(out_path)
    lin_reg['Intercept'].beta.save(op.join(out_path, 'beta_intercept-ave.fif'))
    lin_reg['surprise_dynamic'].beta.save(op.join(out_path, 'beta_surprise_dynamic-ave.fif'))

    # ======== now we compute the residuals and save them =============
    # the betas are of size: n_channels X n_times

    residuals = epochs.get_data()-lin_reg['Intercept'].beta.data - np.asarray([epochs.metadata['surprise_dynamic'].values[i]*lin_reg['surprise_dynamic'].beta._data for i in range(len(epochs))])
    residual_epochs = epochs.copy()
    residual_epochs._data = residuals
    # epochs.average().plot_joint()
    # residual_epochs.average().plot_joint()

    # save the residuals epoch in the same folder
    residual_epochs.save(out_path+op.sep+'residuals-epo.fif', overwrite=True)

    del epochs, residual_epochs

# =============================================================================
# ======== 2// regress the on the residuals the factors of interest ===========
# =============================================================================


for subject in config.subjects_list:

    in_path = op.join(config.result_path, 'linear_models', 'surprisedynamic', subject)
    residual_epochs = mne.read_epochs(in_path+op.sep+'residuals-epo.fif')

    residual_epochs.metadata['violation_X_complexity'] = scale(residual_epochs.metadata['ViolationOrNot']*residual_epochs.metadata['Complexity'])
    residual_epochs.metadata['Complexity'] = scale(residual_epochs.metadata['Complexity'])
    residual_epochs.metadata['ViolationOrNot'] = scale(residual_epochs.metadata['ViolationOrNot'])

    names = [ "Complexity","ViolationOrNot","violation_X_complexity"]
    res = linear_regression(residual_epochs, residual_epochs.metadata[names], names=names)

    # Save regression results
    out_path = op.join(config.result_path, 'linear_models', 'residual_regression_complexity', subject)
    utils.create_folder(out_path)
    res['Complexity'].beta.save(op.join(out_path, 'beta_Complexity-ave.fif'))
    res['ViolationOrNot'].beta.save(op.join(out_path, 'beta_ViolationOrNot-ave.fif'))
    res['violation_X_complexity'].beta.save(op.join(out_path, 'beta_violation_X_complexity-ave.fif'))

    # ====== now look at the influence of hierarchical structure ============

    n_open = []
    n_closed = []
    for k in range(len(residual_epochs)):
        n_open_k, n_closed_k = hierarch_struct(residual_epochs[k].metadata['SequenceID'],residual_epochs[k].metadata['StimPosition'])
        n_open.append(n_open_k)
        n_closed.append(n_closed_k)

    residual_epochs.metadata['n_open_nodes'] = n_open
    residual_epochs.metadata['n_closed_nodes'] = n_closed

    # we consider only the standards
    residual_epochs = residual_epochs["ViolationInSequence == 0"]
    # maybe we should exclude the first sequences of habituation ?


def hierarch_struct(seqID,position_in_seq):
        if seqID == 1:
            n_open_nodes = [1]*16

        elif seqID == 2:
            n_open_nodes = [1]*16

        elif seqID == 3:

        elif seqID == 4:

        elif seqID == 5:

        elif seqID == 6:

        elif seqID == 7:

        else:
            print('This sequence was not recognized!!!! ')

        n_nodes_closing = np.diff(n_open_nodes+[1])


        return n_open_nodes[position_in_seq], n_nodes_closing[position_in_seq]


# =================================================================
# ======== 3// group averages of the betas + statistics ===========
# =================================================================

complexity_evo = evoked_funcs.load_evoked('all', filter_name='beta_Complexity', root_path=op.join(config.result_path, 'linear_models', 'residual_regression_complexity'))
violation_or_not_evo = evoked_funcs.load_evoked('all', filter_name='beta_ViolationOrNot', root_path=op.join(config.result_path, 'linear_models', 'residual_regression_complexity'))
violation_X_complexity_evo = evoked_funcs.load_evoked('all', filter_name='beta_violation_X_complexity', root_path=op.join(config.result_path, 'linear_models', 'residual_regression_complexity'))

complexity_epo = mne.EpochsArray(np.asarray([complexity_evo['beta_Complexity-'][i][0].data for i in range(len(complexity_evo['beta_Complexity-']))]), complexity_evo['beta_Complexity-'][0][0].info, tmin=-0.1)
violation_or_not_epo = mne.EpochsArray(np.asarray([violation_or_not_evo['beta_ViolationOrNot-'][i][0].data for i in range(len(violation_or_not_evo['beta_ViolationOrNot-']))]), violation_or_not_evo['beta_ViolationOrNot-'][0][0].info, tmin=-0.1)
violation_X_complexity_epo = mne.EpochsArray(np.asarray([violation_X_complexity_evo['beta_violation_X_complexity-'][i][0].data for i in range(len(violation_X_complexity_evo['beta_violation_X_complexity-']))]), violation_X_complexity_evo['beta_violation_X_complexity-'][0][0].info, tmin=-0.1)

out_path = op.join(config.result_path, 'linear_models', 'residual_regression_complexity', 'group')
utils.create_folder(out_path)
complexity_epo.save(op.join(out_path, 'complexity_epo.fif'), overwrite=True)
violation_or_not_epo.save(op.join(out_path, 'violation_or_not_epo.fif'), overwrite=True)
violation_X_complexity_epo.save(op.join(out_path, 'violation_X_complexity_epo.fif'), overwrite=True)

# ======================= RELOAD GROUP REGRESSION RESULTS ====================== #
path = op.join(config.result_path, 'linear_models', 'residual_regression_complexity', 'group')
complexity_epo = mne.read_epochs(op.join(path, 'complexity_epo.fif'))
violation_or_not_epo = mne.read_epochs(op.join(path, 'violation_or_not_epo.fif'))
violation_X_complexity_epo = mne.read_epochs(op.join(path, 'violation_X_complexity_epo.fif'))

betas = {'complexity':complexity_epo,'violation':violation_or_not_epo,'Violation_X_complexity':violation_X_complexity_epo}

# ======================= RUN CLUSTER STATISTICS ====================== #
ch_type = 'mag'
data_condition = complexity_epo.copy()  # !!!!!!
fname = 'complexity_epo'                # !!!!!!
# data_condition.apply_baseline(baseline=(-0.100, 0.0))  # baseline ?? (probably not at this step - betas)
data_condition.crop(tmin=0.0, tmax=0.600)  # crop

connectivity = mne.channels.find_ch_connectivity(data_condition.info, ch_type=ch_type)[0]
data = np.array([data_condition.pick_types(meg=ch_type, eeg=False)[c].get_data() for c in range(len(data_condition))])

data = np.transpose(np.squeeze(data), (0, 2, 1))  # transpose for clustering
cluster_stats = permutation_cluster_1samp_test(data, threshold=None, n_jobs=6, verbose=True, tail=0, n_permutations=5000,
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















