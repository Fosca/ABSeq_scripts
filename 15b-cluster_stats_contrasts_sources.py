import mne
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *
from importlib import reload
import numpy as np
from scipy import stats
import pickle
from mne.stats import (spatio_temporal_cluster_1samp_test, summarize_clusters_stc)
from scipy.signal import savgol_filter
from scipy.stats import sem
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Exclude some subjects
# config.exclude_subjects.append('sub10-gp_190568')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()


analysis_main_name = 'Viol_vs_Stand'  # Viol_vs_Stand
subjects_list = config.subjects_list
# subjects_list = [config.subjects_list[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16]]  # only subjects with available sources data
fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')


# =========================================================== #
# Options
# =========================================================== #
Do3Dplot = True
DoROIplots = True
RunStats = True
use_baseline = True  # option to apply a baseline to the evokeds before running the contrast/extracting the sources
lowpass_evoked = True  # option to filter evoked with  30Hz lowpass filter
resid_epochs = False  # use epochs/evoked created by regressing out surprise effects, instead of original epochs
if resid_epochs:
    resid_epochs_type = 'reg_repeataltern_surpriseOmegainfinity'  # 'residual_surprise'  'residual_model_constant' 'reg_repeataltern_surpriseOmegainfinity'

# =========================================================== #
# Output folder
# =========================================================== #
if resid_epochs:
    results_path = op.join(config.result_path, 'Paired_contrasts', analysis_main_name, 'TP_corrected_data', 'Sources')
else:
    results_path = op.join(config.result_path, 'Paired_contrasts', analysis_main_name, 'Original_data', 'Sources')
if use_baseline:
    results_path = results_path + op.sep + 'With_baseline_correction'
utils.create_folder(results_path)

# ASSUMES THAT THERE IS FIF FILES FOR EACH SUBJECT IN "results_path" (FOR evoked_cond1 AND evoked_cond2, created in "15-cluster_stats_contrasts.py")

print('####### ' + str(len(subjects_list)) + ' subjects included #######')
# =============================================================================================
# ========= MAIN ANALYSIS: SOURCES FROM EVOKED CONTRASTS EACH SUBJECT, GROUP CLUSTER PERMUTATION TEST COND1 vs COND2
for analysis_name in [analysis_main_name + '_allseq', analysis_main_name + '_seqID1', analysis_main_name + '_seqID2', analysis_main_name + '_seqID3', analysis_main_name + '_seqID4', analysis_main_name + '_seqID5', analysis_main_name + '_seqID6', analysis_main_name + '_seqID7']:
    print(analysis_name)
    n_subjects = len(subjects_list)

    # =======================================================================
    # ============== Create average evoked & sources for cond1, cond2 & delta

    # ====== Load evoked, compute sources, for each subject
    group_evokeds_cond1 = []
    group_evokeds_cond2 = []
    group_evokeds_delta = []
    group_morphed_stcs_cond1 = []
    group_morphed_stcs_cond2 = []
    group_morphed_stcs_delta = []
    for subject in subjects_list:

        # Load evoked and sources for the 2 conditions
        if resid_epochs:
            evoked1, stc1 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name=analysis_name + '_cond1', evoked_filter_not=None, evoked_path='evoked_resid', apply_baseline=use_baseline,
                                                                             lowpass_evoked=lowpass_evoked)
            evoked2, stc2 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name=analysis_name + '_cond2', evoked_filter_not=None, evoked_path='evoked_resid', apply_baseline=use_baseline,
                                                                             lowpass_evoked=lowpass_evoked)
        else:
            evoked1, stc1 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name=analysis_name + '_cond1', evoked_filter_not=None, evoked_path='evoked_cleaned', apply_baseline=use_baseline,
                                                                             lowpass_evoked=lowpass_evoked)
            evoked2, stc2 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name=analysis_name + '_cond2', evoked_filter_not=None, evoked_path='evoked_cleaned', apply_baseline=use_baseline,
                                                                             lowpass_evoked=lowpass_evoked)

        # Compute delta evoked, with sources
        delta_evoked = mne.combine_evoked([evoked1, evoked2], weights=[1, -1])
        delta_stc = source_estimation_funcs.compute_sources_from_evoked(subject, delta_evoked)

        # Store subject data in group list
        group_evokeds_cond1.append(evoked1)
        group_evokeds_cond2.append(evoked2)
        group_evokeds_delta.append(delta_evoked)
        group_morphed_stcs_cond1.append(stc1)
        group_morphed_stcs_cond2.append(stc2)
        group_morphed_stcs_delta.append(delta_stc)

    # ====== Evoked: create group averages: cond1, cond2, delta
    mean_ev_cond1 = mne.grand_average(group_evokeds_cond1)
    mean_ev_cond2 = mne.grand_average(group_evokeds_cond2)
    mean_ev_delta = mne.grand_average(group_evokeds_delta)

    # ====== Sources: create group averages: cond1, cond2, delta
    # mean_stc_cond1
    mean_stc_cond1 = group_morphed_stcs_cond1[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        mean_stc_cond1._data += group_morphed_stcs_cond1[sub].data
    mean_stc_cond1._data /= n_subjects
    # mean_stc_cond2
    mean_stc_cond2 = group_morphed_stcs_cond2[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        mean_stc_cond2._data += group_morphed_stcs_cond2[sub].data
    mean_stc_cond2._data /= n_subjects
    # mean_stc_delta
    mean_stc_delta = group_morphed_stcs_delta[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        mean_stc_delta._data += group_morphed_stcs_delta[sub].data
    mean_stc_delta._data /= n_subjects

    # ====== Create sources figures
    if Do3Dplot:
        fig_path = op.join(results_path, 'Group_averages')
        utils.create_folder(fig_path)
        figure_title = analysis_name + ' cond1'
        output_file = op.join(fig_path, 'Sources_' + analysis_name + '_cond1.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc_cond1, mean_ev_cond1, output_file, figure_title, timepoint='max', ch_type='grad', colormap='hot', colorlims='auto')
        output_file = op.join(fig_path, 'Sources_' + analysis_name + '_cond1_at80ms.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc_cond1, mean_ev_cond1, output_file, figure_title, timepoint=0.080, ch_type='grad', colormap='viridis', colorlims=[1, 2, 3])
        output_file = op.join(fig_path, 'Sources_' + analysis_name + '_cond1_at170ms.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc_cond1, mean_ev_cond1, output_file, figure_title, timepoint=0.170, ch_type='grad', colormap='viridis', colorlims=[1, 2, 3])

        figure_title = analysis_name + ' cond2'
        output_file = op.join(fig_path, 'Sources_' + analysis_name + '_cond2.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc_cond2, mean_ev_cond2, output_file, figure_title, timepoint='max', ch_type='grad', colormap='hot', colorlims='auto')
        output_file = op.join(fig_path, 'Sources_' + analysis_name + '_cond2_at80ms.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc_cond2, mean_ev_cond2, output_file, figure_title, timepoint=0.080, ch_type='grad', colormap='viridis', colorlims=[1, 2, 3])
        output_file = op.join(fig_path, 'Sources_' + analysis_name + '_cond2_at170ms.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc_cond2, mean_ev_cond2, output_file, figure_title, timepoint=0.170, ch_type='grad', colormap='viridis', colorlims=[1, 2, 3])

        figure_title = analysis_name + ' delta'
        output_file = op.join(fig_path, 'Sources_' + analysis_name + '_delta.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc_delta, mean_ev_delta, output_file, figure_title, timepoint='max', ch_type='grad', colormap='hot', colorlims='auto')
        output_file = op.join(fig_path, 'Sources_' + analysis_name + '_delta_at80ms.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc_delta, mean_ev_delta, output_file, figure_title, timepoint=0.080, ch_type='grad', colormap='viridis', colorlims=[1, 2, 3])
        output_file = op.join(fig_path, 'Sources_' + analysis_name + '_delta_at170ms.png')
        source_estimation_funcs.sources_evoked_figure(mean_stc_delta, mean_ev_delta, output_file, figure_title, timepoint=0.170, ch_type='grad', colormap='viridis', colorlims=[1, 2, 3])

    # =======================================================================
    # ======  Statistics in sources spaces V2, cond1 vs cond2
    # Adapted from https://mne.tools/stable/auto_tutorials/stats-source-space/plot_stats_cluster_spatio_temporal.html?highlight=visualize%20sources
    if RunStats:
        # Create data structure for the contrast
        time_window = [0.000, 0.500]  # to crop data on the time dimension
        n_vertices, n_times = group_morphed_stcs_cond1[0].crop(tmin=time_window[0], tmax=time_window[1]).data.shape
        X = np.zeros((n_subjects, n_vertices, n_times, 2))
        for nsub in range(n_subjects):
            X[nsub, :, :, 0] = group_morphed_stcs_cond1[nsub].crop(tmin=time_window[0], tmax=time_window[1]).data
            X[nsub, :, :, 1] = group_morphed_stcs_cond2[nsub].crop(tmin=time_window[0], tmax=time_window[1]).data
        X = np.abs(X)  # only magnitude
        X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast, cond1 - cond2
        X = np.transpose(X, [0, 2, 1])  # X : array, shape (n_observations, n_times, n_vertices)

        # Connectivity
        print('Computing connectivity')  # need to use connectivity instead of adjacency in old mne version (0.19)
        src = mne.read_source_spaces(op.join(fsMRI_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
        connectivity = mne.spatial_src_connectivity(src)
        fsaverage_vertices = [s['vertno'] for s in src]

        # Clustering
        print('Clustering')
        # p_threshold = 0.05
        # t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
        T_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_1samp_test(X, connectivity=connectivity, n_jobs=6, n_permutations=500, threshold=None, buffer_size=None,
                                                                                         verbose=True)  # with connectivity instead of adjacency

        # Save or plot clusters (if any significant one)
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
        if len(good_cluster_inds) > 0:
            print(analysis_name + ': ' + str(len(good_cluster_inds)) + ' good clusters')
            print('Saving clusters to ' + op.join(results_path, analysis_name + '_sources_clusters.pickle'))
            with open(op.join(results_path, analysis_name + '_sources_clusters.pickle'), 'wb') as f:
                pickle.dump(clu, f, pickle.HIGHEST_PROTOCOL)
            # if Do3Dplot:
            #     print('Visualizing clusters.')
            #     #    Now let's build a convenient representation of each cluster, where each cluster becomes a "time point" in the SourceEstimate
            #     stc_all_clusters = summarize_clusters_stc(clu, p_thresh=0.05, tstep=tstep, vertices=fsaverage_vertices, subject='fsaverage')
            #     #    Let's actually plot the first "time point" in the SourceEstimate, which shows all the clusters, weighted by duration. blue blobs are for condition A < condition B, red for A > B
            #     brain = stc_all_clusters.plot(hemi='split', views=['lat', 'med'], subjects_dir=fsMRI_dir, time_label='Temporal extent (ms)', size=(800, 800),
            #                                   smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 100]), time_viewer=False)  # , show_traces=True)
            #     brain.save_image(op.join(results_path, 'Sources_clusters_' + analysis_name + '.png'))
        else:
            print('No significant clusters')

# =============================================================================================
# ========= RELOAD AND PLOT CLUSTERS
if Do3Dplot:
    src = mne.read_source_spaces(op.join(fsMRI_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
    fsaverage_vertices = [s['vertno'] for s in src]
    tstep = 4 / 1000
    timevec = np.arange(0.0, 0.504, tstep)
    for analysis_name in [analysis_main_name + '_allseq', analysis_main_name + '_seqID1', analysis_main_name + '_seqID2', analysis_main_name + '_seqID3', analysis_main_name + '_seqID4', analysis_main_name + '_seqID5',
                          analysis_main_name + '_seqID6', analysis_main_name + '_seqID7']:
        # for analysis_name in [analysis_main_name + '_seqID4', analysis_main_name + '_seqID5', analysis_main_name + '_seqID6', analysis_main_name + '_seqID7']:

        if op.isfile(op.join(results_path, analysis_name + '_sources_clusters.pickle')) is False:
            print('No cluster file found for ' + analysis_name + ' (not significant?)')
        else:
            print('Loading sources clusters for ' + analysis_name + '...')
            with open(op.join(results_path, analysis_name + '_sources_clusters.pickle'), 'rb') as f:
                clu = pickle.load(f)

            # Plot all clusters as a "cluster duration map"
            stc_all_clusters = summarize_clusters_stc(clu, p_thresh=0.05, tstep=tstep, vertices=fsaverage_vertices, subject='fsaverage')
            brain = stc_all_clusters.plot(surface='inflated', hemi='split', views=['lat', 'med'], subjects_dir=fsMRI_dir, time_label='Temporal extent (ms)', size=(800, 800),
                                          smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 50]), time_viewer=False)  # , show_traces=True)
            print(op.join(results_path, 'Sources_' + analysis_name + '_allclusters.png'))
            brain.save_image(op.join(results_path, 'Sources_' + analysis_name + '_allclusters.png'))
            brain.close()

            # Plot each cluster peak
            t_obs, clusters, clu_pvals, _ = clu
            n_times, n_vertices = t_obs.shape
            good_cluster_inds = np.where(clu_pvals < 0.05)[0]
            print(str(len(good_cluster_inds)) + ' good clusters')
            for ii, iclu in enumerate(good_cluster_inds):
                print(ii)
                if ii <= 6:  # plot only the first 6 clusters !!
                    data = np.zeros((n_vertices, n_times))
                    v_inds = clusters[iclu][1]
                    t_inds = clusters[iclu][0]
                    data[v_inds, t_inds] = t_obs[t_inds, v_inds]
                    cluster_stc = mne.SourceEstimate(data, vertices=fsaverage_vertices, tmin=0.0, tstep=tstep, subject='fsaverage')
                    # timeval = cluster_stc.get_peak()[1]
                    # datamin = round(np.min(abs(data)[data != 0]), 2)-0.2
                    # datamax = round(np.max(abs(data)[data != 0]), 2)-0.5
                    # brain = cluster_stc.plot(hemi='split', views=['lat', 'med'], subjects_dir=fsMRI_dir, size=(800, 800), smoothing_steps=5,
                    #                          clim=dict(kind='value', pos_lims=[datamin, datamin+(datamax-datamin)/2, datamax]), initial_time=timeval, time_viewer=True)  # , show_traces=True)
                    ctmin = timevec[t_inds[0]]
                    ctmax = timevec[t_inds[-1]]
                    winlength = len(range(t_inds[0], t_inds[-1]))
                    stc_timewin = cluster_stc.copy()
                    stc_timewin.crop(tmin=ctmin, tmax=ctmax)  # keep only times of cluster
                    stc_timewin = stc_timewin.mean()  # average T values in the cluster window
                    dat = stc_timewin._data
                    datamin = round(np.min(abs(dat)[dat != 0]), 2)
                    datamax = round(np.max(abs(dat)[dat != 0]), 2)
                    brain = stc_timewin.plot(hemi='split', views=['lat', 'med'], subjects_dir=fsMRI_dir, size=(800, 800), smoothing_steps=5,
                                             clim=dict(kind='value', pos_lims=[datamin, datamin + (datamax - datamin) / 2, datamax]), time_viewer=False)  # , show_traces=True)
                    fname = op.join(results_path, 'Sources_' + analysis_name + '_cluster_' + str(ii + 1) + '_[' + str('%d' % (ctmin * 1000)) + '-' + str('%d' % (ctmax * 1000)) + 'ms].png')
                    print('Saving ' + fname)
                    brain.save_image(fname)
                    brain.close()
                    # NEGAT CLUSTER ???

# =============================================================================================
# ========= TEST EXTRACT ROI DATA
if DoROIplots:
    path = op.join(results_path, 'ROI')
    utils.create_folder(path)
    subjects_list = config.subjects_list
    for analysis_name in [analysis_main_name + '_allseq', analysis_main_name + '_seqID1', analysis_main_name + '_seqID2', analysis_main_name + '_seqID3', analysis_main_name + '_seqID4', analysis_main_name + '_seqID5',
                          analysis_main_name + '_seqID6', analysis_main_name + '_seqID7']:
        # for analysis_name in [analysis_main_name + '_seqID1']:
        print(analysis_name)
        n_subjects = len(subjects_list)

        label_names = ['parsopercularis-lh', 'parsopercularis-rh', 'bankssts-rh', 'bankssts-lh', 'superiorparietal-lh', 'superiorparietal-rh', 'inferiorparietal-lh', 'inferiorparietal-rh', ]

        # =======================================================================
        # ============== Extract ROI data
        # create empty dict to store everything
        group_all_labels_data = dict()
        for label_name in label_names:
            group_all_labels_data[label_name] = dict()
            group_all_labels_data[label_name]['cond1'] = []
            group_all_labels_data[label_name]['cond2'] = []
        # subject loop
        for subject in subjects_list:
            # Load evoked and sources for the 2 conditions
            if resid_epochs:
                evoked1, stc1 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name=analysis_name + '_cond1', evoked_filter_not=None, evoked_path='evoked_resid', apply_baseline=use_baseline,
                                                                                 lowpass_evoked=lowpass_evoked, morph_sources=False)
                evoked2, stc2 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name=analysis_name + '_cond2', evoked_filter_not=None, evoked_path='evoked_resid', apply_baseline=use_baseline,
                                                                                 lowpass_evoked=lowpass_evoked, morph_sources=False)
            else:
                evoked1, stc1 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name=analysis_name + '_cond1', evoked_filter_not=None, evoked_path='evoked_cleaned', apply_baseline=use_baseline,
                                                                                 lowpass_evoked=lowpass_evoked, morph_sources=False, fake_nave=False)
                evoked2, stc2 = source_estimation_funcs.load_evoked_with_sources(subject, evoked_filter_name=analysis_name + '_cond2', evoked_filter_not=None, evoked_path='evoked_cleaned', apply_baseline=use_baseline,
                                                                                 lowpass_evoked=lowpass_evoked, morph_sources=False, fake_nave=False)
            src = mne.read_source_spaces(op.join(config.meg_dir, subject, subject + '_oct6-inv.fif'))
            for label_name in label_names:
                anat_label = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=fsMRI_dir, regexp=label_name)[0]
                label_data_cond1 = stc1.extract_label_time_course(anat_label, src, mode='mean')[0]
                label_data_cond2 = stc2.extract_label_time_course(anat_label, src, mode='mean')[0]
                # label_data_cond1 = stc1.extract_label_time_course(anat_label, src, mode='pca_flip')[0]
                # label_data_cond1 *= np.sign(label_data_cond1[np.argmax(np.abs(label_data_cond1))])  # flip the pca so that the max power is positive ??
                # label_data_cond2 = stc2.extract_label_time_course(anat_label, src, mode='pca_flip')[0]
                # label_data_cond2 *= np.sign(label_data_cond2[np.argmax(np.abs(label_data_cond2))])  # flip the pca so that the max power is positive ??
                group_all_labels_data[label_name]['cond1'].append(label_data_cond1)
                group_all_labels_data[label_name]['cond2'].append(label_data_cond2)

        # =======================================================================
        # ============== Plot group means
        times = (1e3 * stc1.times)
        filter = False
        plt.close('all')
        for label_name in label_names:
            data = group_all_labels_data[label_name]['cond1']
            mean1 = np.mean(data, axis=0)
            ub1 = mean1 + sem(data, axis=0)
            lb1 = mean1 - sem(data, axis=0)
            data = group_all_labels_data[label_name]['cond2']
            mean2 = np.mean(data, axis=0)
            ub2 = mean2 + sem(data, axis=0)
            lb2 = mean2 - sem(data, axis=0)

            if filter == True:
                mean1 = savgol_filter(mean1, 13, 3)
                ub1 = savgol_filter(ub1, 13, 3)
                lb1 = savgol_filter(lb1, 13, 3)
                mean2 = savgol_filter(mean2, 13, 3)
                ub2 = savgol_filter(ub2, 13, 3)
                lb2 = savgol_filter(lb2, 13, 3)

            fig, ax = plt.subplots(1, 1, figsize=(10, 4), constrained_layout=False)
            plt.axvline(0, linestyle='-', color='k', linewidth=2)
            ax.fill_between(times, ub1, lb1, color='r', alpha=.2)
            ax.plot(times, mean1, color='r', linewidth=1.5, label='Deviant')
            ax.fill_between(times, ub2, lb2, color='b', alpha=.2)
            ax.plot(times, mean2, color='b', linewidth=1.5, label='Standard')
            ax.set_xlabel('Time (ms)')
            ax.set_xlim([-100, 600])
            for key in ('top', 'right'):  # Remove spines
                ax.spines[key].set(visible=False)
            plt.legend()
            plt.title(analysis_name + ': ' + label_name, fontsize=14, weight='bold', color='k')
            fig_name = op.join(path, analysis_name + '_' + label_name + '.png')
            print('Saving ' + fig_name)
            plt.savefig(fig_name, dpi=300)
            plt.close('all')
