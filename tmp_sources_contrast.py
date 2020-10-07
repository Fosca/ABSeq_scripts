import mne
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *
from importlib import reload
import numpy as np
from scipy import stats
import pickle
from nilearn import plotting
from mne.stats import (spatio_temporal_cluster_1samp_test, summarize_clusters_stc)

fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
results_path = op.join(config.result_path, 'Contrasts_tests', 'Viol_vs_Stand')
utils.create_folder(results_path)
subjects_list = [config.subjects_list[i] for i in [0, 1, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16]]

# =======================  STEP1: =======================  #
# ======================= Subject-level
# Load evokeds for condition 1 & condition 2, compute delta, compute sources for cond1, cond2 & delta, morph the 3 to fsaverage
for analysis_name in ['Viol_vs_Stand_allseq', 'Viol_vs_Stand_seqID1', 'Viol_vs_Stand_seqID2', 'Viol_vs_Stand_seqID3',
                      'Viol_vs_Stand_seqID4', 'Viol_vs_Stand_seqID5', 'Viol_vs_Stand_seqID6', 'Viol_vs_Stand_seqID7']:
    print(analysis_name)
    morphed_stcs_cond1 = []
    morphed_stcs_cond2 = []
    morphed_stcs_delta = []
    for subject in subjects_list:
        # Load evoked for the 2 conditions
        evoked1, path_evo = evoked_funcs.load_evoked(subject=subject, filter_name=analysis_name + '_cond1',
                                                     filter_not=None, cleaned=True)
        evoked2, path_evo = evoked_funcs.load_evoked(subject=subject, filter_name=analysis_name + '_cond2',
                                                     filter_not=None, cleaned=True)
        e1 = next(iter(evoked1.values()))[0]
        e2 = next(iter(evoked2.values()))[0]

        # # Baseline evoked ??!
        # e1.apply_baseline((None, 0))
        # e2.apply_baseline((None, 0))

        # Compute & save contrast
        delta_evoked = mne.combine_evoked([e1, e2], weights=[1, -1])
        delta_evoked.save(op.join(path_evo, analysis_name + '_delta-ave.fif'))

        # Estimate sources from evoked/constrast
        source_estimation_funcs.source_estimates(subject, evoked_filter_name=analysis_name + '_cond1',
                                                 evoked_filter_not=None)
        source_estimation_funcs.source_estimates(subject, evoked_filter_name=analysis_name + '_cond2',
                                                 evoked_filter_not=None)
        source_estimation_funcs.source_estimates(subject, evoked_filter_name=analysis_name + '_delta',
                                                 evoked_filter_not=None)

        # Morph to fsaverage (save & store)
        stc_fsaverage = source_estimation_funcs.source_morph(subject, source_evoked_name=analysis_name + '_cond1')
        morphed_stcs_cond1.append(stc_fsaverage)
        stc_fsaverage = source_estimation_funcs.source_morph(subject, source_evoked_name=analysis_name + '_cond2')
        morphed_stcs_cond2.append(stc_fsaverage)
        stc_fsaverage = source_estimation_funcs.source_morph(subject, source_evoked_name=analysis_name + '_delta')
        morphed_stcs_delta.append(stc_fsaverage)

        # Sanity check
        m = np.round(np.max(stc_fsaverage.data), 0)
        if m > 100:
            raise ValueError('/!\ Probable issue with sources for subject ' + subject + ': max value = ' + str(m))

    # Save all subjects morphed sources to files
    with open(op.join(results_path, 'allsubs_' + analysis_name + '_cond1_sources.pickle'), 'wb') as f:
        pickle.dump(morphed_stcs_cond1, f, pickle.HIGHEST_PROTOCOL)
    with open(op.join(results_path, 'allsubs_' + analysis_name + '_cond2_sources.pickle'), 'wb') as f:
        pickle.dump(morphed_stcs_cond2, f, pickle.HIGHEST_PROTOCOL)
    with open(op.join(results_path, 'allsubs_' + analysis_name + '_delta_sources.pickle'), 'wb') as f:
        pickle.dump(morphed_stcs_delta, f, pickle.HIGHEST_PROTOCOL)

# =======================  STEP2: =======================  #
# ======================= Group level
# Group averages and plots

for analysis_name in ['Viol_vs_Stand_allseq', 'Viol_vs_Stand_seqID1', 'Viol_vs_Stand_seqID2', 'Viol_vs_Stand_seqID3',
                      'Viol_vs_Stand_seqID4', 'Viol_vs_Stand_seqID5', 'Viol_vs_Stand_seqID6', 'Viol_vs_Stand_seqID7']:

    analysis_name = 'Viol_vs_Stand_allseq'
    # (Re)Load  morphed sources
    with open(op.join(results_path, 'allsubs_' + analysis_name + '_cond1_sources.pickle'), 'rb') as f:
        morphed_stcs_cond1 = pickle.load(f)
    with open(op.join(results_path, 'allsubs_' + analysis_name + '_cond2_sources.pickle'), 'rb') as f:
        morphed_stcs_cond2 = pickle.load(f)
    with open(op.join(results_path, 'allsubs_' + analysis_name + '_delta_sources.pickle'), 'rb') as f:
        morphed_stcs_delta = pickle.load(f)

    # Mean morphed_stcs_delta
    n_subjects = len(morphed_stcs_delta)
    mean_morphed_stcs_delta = morphed_stcs_delta[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        print('maxval sub' + str(sub) + ' = ' + str(np.round(np.max(morphed_stcs_delta[sub].data), 0)))
        mean_morphed_stcs_delta._data += morphed_stcs_delta[sub].data
    mean_morphed_stcs_delta._data /= n_subjects

    # Mean morphed_stcs_cond1
    n_subjects = len(morphed_stcs_cond1)
    mean_morphed_stcs_cond1 = morphed_stcs_cond1[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        print('maxval sub' + str(sub) + ' = ' + str(np.round(np.max(morphed_stcs_cond1[sub].data), 0)))
        mean_morphed_stcs_cond1._data += morphed_stcs_cond1[sub].data
    mean_morphed_stcs_cond1._data /= n_subjects

    # Mean morphed_stcs_cond2
    n_subjects = len(morphed_stcs_cond2)
    mean_morphed_stcs_cond2 = morphed_stcs_cond2[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        print('maxval sub' + str(sub) + ' = ' + str(np.round(np.max(morphed_stcs_cond2[sub].data), 0)))
        mean_morphed_stcs_cond2._data += morphed_stcs_cond2[sub].data
    mean_morphed_stcs_cond2._data /= n_subjects

    #
    stc = mean_morphed_stcs_cond2
    vertno_max, time_max = stc.get_peak()
    surfer_kwargs = dict(surface='inflated', hemi='both', subjects_dir=fsMRI_dir, subject='fsaverage',
                        clim=dict(kind='value', lims=[2, 6, 10]), views='lat',
                        initial_time=time_max, time_unit='s', size=(1300, 900), smoothing_steps=5,
                        time_viewer=False, show_traces=False,  alpha=1)
    matplotlib.use('Qt5Agg')
    from mayavi import mlab
    mlab.init_notebook('x3d')
    brain = stc.plot(**surfer_kwargs)

    brain = stc.plot(views=['lat'], surface='inflated', hemi='both', size=(1300, 900), subject='fsaverage', clim=dict(kind='value', lims=[2, 6, 10]),subjects_dir=fsMRI_dir, initial_time=time_max, smoothing_steps=5, show_traces=False)
    brain = stc.plot(**surfer_kwargs)
    brain = stc.plot(views=['lat'], surface='inflated', hemi='both', size=(1300, 900), subject='fsaverage', clim=dict(kind='value', lims=[2, 6, 10]),subjects_dir=fsMRI_dir, initial_time=170, smoothing_steps=5)

    plt.figure()
    brain = stc.plot(views=['lat', 'med'], surface='inflated', hemi='split', size=(1600, 900), subject='fsaverage',
                     clim=dict(kind='value', lims=[2, 6, 10]),
                     subjects_dir=fsMRI_dir, initial_time=time_max, smoothing_steps=5, time_viewer=True)
    mlab.savefig('test.png')

    colormap = 'viridis'
    clim = dict(kind='value', lims=[2, 4, 8])
    # Plot the STC, get the brain image, crop it:
    brain = stc.plot(views='lat', hemi='split', size=(2000, 900), subject='fsaverage',
                     subjects_dir=fsMRI_dir, initial_time=time_max, background='w',
                     colorbar=False, clim=clim, colormap=colormap,
                     time_viewer=False, show_traces=False)
    screenshot = brain.screenshot()
    brain.close()
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
    # before/after results
    from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
    fig = plt.figure(figsize=(4, 4))
    axes = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.5)
    for ax, image, title in zip(axes, [screenshot, cropped_screenshot],
                                ['Before', 'After']):
        ax.imshow(image)
        ax.set_title('{} cropping'.format(title))
    # Tweak the figure style
    plt.rcParams.update({
        'ytick.labelsize': 'small',
        'xtick.labelsize': 'small',
        'axes.labelsize': 'small',
        'axes.titlesize': 'medium',
        'grid.color': '0.75',
        'grid.linestyle': ':',
    })





    # brain with peak rh
    vertno_max, time_max = stc.get_peak(hemi='rh')
    surfer_kwargs = dict(
        surface='inflated', hemi='split', subjects_dir=fsMRI_dir, subject=subject,
        clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
        initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
    brain = stc.plot(**surfer_kwargs)
    brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6, alpha=0.5)
    brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title', font_size=14)

    vertno_max, time_max = stc.get_peak(hemi='rh')
    surfer_kwargs = dict(
        hemi='rh', subjects_dir=fsMRI_dir,
        clim=dict(kind='value', lims=[2, 6, 10]), views='lateral',
        initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
    brain = stc.plot(**surfer_kwargs)
    brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
                   scale_factor=0.6, alpha=0.5)
    brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
                   font_size=14)

# =======================  STEP2: =======================  #
# ======================= Group level
# Statistics in sources spaces V2
# Adapted from https://mne.tools/stable/auto_tutorials/stats-source-space/plot_stats_cluster_spatio_temporal.html?highlight=visualize%20sources
time_window = [0.000, 0.500]  # to crop data on the time dimension
n_subjects = len(morphed_stcs_cond1)
n_vertices, n_times = morphed_stcs_cond1[0].crop(tmin=time_window[0], tmax=time_window[1]).data.shape
tstep = morphed_stcs_cond1[0].tstep * 1000  # convert to milliseconds
X = np.zeros((n_subjects, n_vertices, n_times, 2))
for nsub in range(n_subjects):
    X[nsub, :, :, 0] = morphed_stcs_cond1[sub].crop(tmin=time_window[0], tmax=time_window[1]).data
    X[nsub, :, :, 1] = morphed_stcs_cond2[sub].crop(tmin=time_window[0], tmax=time_window[1]).data
X = np.abs(X)  # only magnitude
X = X[:, :, :, 0] - X[:, :, :, 1]  # make paired contrast
print('Computing adjacency.')
src = mne.read_source_spaces(op.join(fsMRI_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
adjacency = mne.spatial_src_adjacency(src)
fsaverage_vertices = [s['vertno'] for s in src]
X = np.transpose(X, [0, 2, 1])  # Note that X needs to be a multi-dimensional array of shape samples (subjects) x time x space, so we permute dimensions

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
p_threshold = 0.05
t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
print('Clustering.')
T_obs, clusters, cluster_p_values, H0 = clu = \
    spatio_temporal_cluster_1samp_test(X, adjacency=adjacency, n_jobs=6, n_permutations=1000,
                                       threshold=t_threshold, buffer_size=None,
                                       verbose=True)
#    Now select the clusters that are sig. at p < 0.05 (note that this value is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

print('Visualizing clusters.')
#    Now let's build a convenient representation of each cluster, where each cluster becomes a "time point" in the SourceEstimate
stc_all_cluster_vis = summarize_clusters_stc(clu, tstep=tstep, vertices=fsaverage_vertices, subject='fsaverage')
with open(op.join(results_path, 'allsubs_' + analysis_name + '_TMP.pickle'), 'wb') as f:
    pickle.dump(stc_all_cluster_vis, f, pickle.HIGHEST_PROTOCOL)
with open(op.join(results_path, 'allsubs_' + analysis_name + '_TMP.pickle'), 'rb') as f:
    stc_all_cluster_vis = pickle.load(f)

#    Let's actually plot the first "time point" in the SourceEstimate, which
#    shows all the clusters, weighted by duration
# subjects_dir = op.join(data_path, 'subjects')
# blue blobs are for condition A < condition B, red for A > B
brain = stc_all_cluster_vis.plot(
    hemi='both', views='lateral', subjects_dir=fsMRI_dir,
    time_label='temporal extent (ms)', size=(800, 800),
    smoothing_steps=5, clim=dict(kind='value', pos_lims=[0, 1, 40]),
    time_viewer=True, show_traces=True)
# brain.save_image('clusters.png')

# pip install pyvista
# pip install pyvistaqt


# ======== Plots
plt.close('all')

stc = mean_morphed_stcs_cond2
stc = mean_morphed_stcs_delta
stc = morphed_stcs_cond2[0]
# vertex activity
fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[::500, :].T)
ax.set(xlabel='time (ms)', ylabel='dSPM value')

# brain
vertno_max, time_max = stc.get_peak()
brain = stc.plot(views=['lat'], surface='inflated', hemi='both', size=(1300, 900), subject='fsaverage',
                 clim=dict(kind='value', lims=[2, 6, 10]),
                 subjects_dir=fsMRI_dir, initial_time=time_max, smoothing_steps=5)

brain = stc.plot(views=['lat', 'med'], surface='inflated', hemi='split', size=(1300, 900), subject='fsaverage',
                 clim=dict(kind='value', lims=[2, 6, 10]),
                 subjects_dir=fsMRI_dir, initial_time=time_max, smoothing_steps=5, time_viewer=True)

# brain with peak rh
vertno_max, time_max = stc.get_peak(hemi='rh')
surfer_kwargs = dict(
    surface='inflated', hemi='split', subjects_dir=fsMRI_dir, subject=subject,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue', scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title', font_size=14)

vertno_max, time_max = stc.get_peak(hemi='rh')
surfer_kwargs = dict(
    hemi='rh', subjects_dir=fsMRI_dir,
    clim=dict(kind='value', lims=[2, 6, 10]), views='lateral',
    initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
brain = stc.plot(**surfer_kwargs)
brain.add_foci(vertno_max, coords_as_verts=True, hemi='rh', color='blue',
               scale_factor=0.6, alpha=0.5)
brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
               font_size=14)

# Load sources data one subject
stc1 = mne.read_source_estimate(
    op.join(config.meg_dir, subject, 'evoked_cleaned', analysis_name + '_allseq_delta_dSPM_inverse'))
stc2 = mne.read_source_estimate(
    op.join(config.meg_dir, subject, 'evoked_cleaned', analysis_name + '_allseq_delta_dSPM_inverse'))
stcdelta = mne.read_source_estimate(
    op.join(config.meg_dir, subject, 'evoked_cleaned', analysis_name + '_allseq_delta_dSPM_inverse'))

# ============ Statistics in sources spaces
# # adapted from https://github.com/ualsbombe/omission_frontiers
# time_window = [0.050, 0.250]
# input_data = dict(cond1=morphed_stcs_cond1,
#                   cond2=morphed_stcs_cond2)
# info_data = morphed_stcs_cond1
# n_subjects = len(info_data)
# n_sources, n_samples = info_data[0].data.shape
#
# ## get data in the right format
# statistics_data_1 = np.zeros((n_subjects, n_sources, n_samples))
# statistics_data_2 = np.zeros((n_subjects, n_sources, n_samples))
#
# for subject_index in range(n_subjects):
#     statistics_data_1[subject_index, :, :] = input_data['cond1'][subject_index].data
#     statistics_data_2[subject_index, :, :] = input_data['cond2'][subject_index].data
#     print('processing data from subject: ' + str(subject_index))
#
# ## crop data on the time dimension
# times = info_data[0].times
# time_indices = np.logical_and(times >= time_window[0], times <= time_window[1])
# statistics_data_1 = statistics_data_1[:, :, time_indices]
# statistics_data_2 = statistics_data_2[:, :, time_indices]
#
# ## set up cluster analysis
# n_permutations = 5
# p_threshold = 0.05
# t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, n_subjects - 1)
# # seed = 7  ## my lucky number
# statistics_list = [statistics_data_1, statistics_data_2]
#
# T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(statistics_list,
#                                                                            n_permutations=n_permutations,
#                                                                            threshold=t_threshold,
#                                                                            n_jobs=4,
#                                                                            buffer_size=10000)
# # mne.stats.permutation_cluster_test(statistics_list,
# #                                    n_permutations=n_permutations,
# #                                    threshold=t_threshold,
# #                                    seed=seed,
# #                                    n_jobs=1)
