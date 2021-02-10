"""
Multi-subject joint source localization with multi-task lasso
=============================================================

The aim of this tutorial is to show how to leverage functional similarity
across subjects to improve source localization with multi-task Lasso.
Multi-task Lasso assumes that the exact same sources are active for
all subjects at all times. This example illustrates this on the
the high frequency SEF MEG dataset of (Nurminen et al., 2017) which provides
MEG and MRI data for two subjects.
"""

# Author: Hicham Janati (hicham.janati@inria.fr)
#
# License: BSD (3-clause)

import sys

sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
import mne
import os
import os.path as op
import initialization_paths
from mne.parallel import parallel_func
from mne.datasets import hf_sef
from matplotlib import pyplot as plt
import config
from ABseq_func import source_estimation_funcs, utils
from importlib import reload
from groupmne import compute_group_inverse, prepare_fwds, compute_fwd
import numpy as np

##########################################################
# Download and process MEG data
# -----------------------------
#
# For this example, we use the HF somatosensory dataset [2].
# We need the raw data to estimate the noise covariance
# since only average MEG data (and MRI) are provided in "evoked".
# The data will be downloaded in the same location

data_path = config.study_path
meg_path = data_path + "/MEG/"

config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

subjects = config.subjects_list#[:2]

data_path = op.expanduser(data_path)
subjects_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
# subjects_dir = data_path + "/subjects/"
os.environ['SUBJECTS_DIR'] = subjects_dir
# fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')

# raw_name_s = [meg_path + s for s in ["subject_a/sef_right_raw.fif",
#               "subject_b/hf_sef_15min_raw.fif"]]

#
# def process_meg(raw_name):
#     """Extract epochs from a raw fif file.
#
#     Parameters
#     ----------
#     raw_name: str.
#         path to the raw fif file.
#
#     Returns
#     -------
#     epochs: Epochs instance
#
#     """
#     raw = mne.io.read_raw_fif(raw_name)
#     events = mne.find_events(raw)
#
#     event_id = dict(hf=1)  # event trigger and conditions
#     tmin = -0.05  # start of each epoch (50ms before the trigger)
#     tmax = 0.3  # end of each epoch (300ms after the trigger)
#     baseline = (None, 0)  # means from the first instant to t = 0
#     epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
#                         baseline=baseline)
#     return epochs
#
#
# epochs_s = [process_meg(raw_name) for raw_name in raw_name_s]
# evokeds = [ep.average() for ep in epochs_s]
#
# # compute noise covariance (takes a few minutes)
# noise_covs = []
# for subj, ep in zip(["a", "b"], epochs_s):
#     cov_fname = meg_path + f"subject_{subj}/sef-cov.fif"
#     cov = mne.compute_covariance(ep[:100], tmin=None, tmax=0.)
#     noise_covs.append(cov)

noise_covs = []
for subject in subjects:
    cov = source_estimation_funcs.compute_noise_cov(subject)
    noise_covs.append(cov)

# f, axes = plt.subplots(1, 2, sharey=True)
# for ax, ev, nc, ll in zip(axes.ravel(), evokeds, noise_covs, ["a", "b"]):
#     picks = mne.pick_types(ev.info, meg="grad")
#     ev.plot(picks=picks, axes=ax, noise_cov=nc, show=False)
#     ax.set_title("Subject %s" % ll, fontsize=15)
# plt.show()
#
# del epochs_s

#########################################################
# Source and forward modeling
# ---------------------------
# To guarantee an alignment across subjects, we start by
# computing the source space of `fsaverage`

resolution = 6
spacing = "oct%d" % resolution

fsaverage_fname = op.join(subjects_dir, "fsaverage")
if not op.exists(fsaverage_fname):
    mne.datasets.fetch_fsaverage(subjects_dir)
src_ref = mne.setup_source_space(subject="fsaverage",
                                 spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 add_dist=False)

######################################################
# Compute forward models with a reference source space
# ----------------------------------------------------
# the function `compute_fwd` morphs the source space src_ref to the
# surface of each subject by mapping the sulci and gyri patterns
# and computes their forward operators. Next we prepare the forward operators
# to be aligned across subjects

trans_fname_s = [meg_path + "%s/%s-trans.fif" % (s, s) for s in subjects]
bem_fname_s = [meg_path + "%s/%s-5120-5120-5120-bem-sol.fif" % (s, s) for s in subjects]
infos = utils.load_info_subjects(subjects)

####################################################################
# Before computing the forward operators, we make sure the coordinate
# transformation of the trans file provides a reasonable alignement between the
# different coordinate systems MEG <-> HEAD

# # for raw_fname, trans, subject in zip(raw_name_s, trans_fname_s, subjects):
# for info, trans, subject in zip(infos, trans_fname_s, subjects):
#     # raw = mne.io.read_raw_fif(raw_fname)
#     # fig = mne.viz.plot_alignment(raw.info, trans=trans, subject=subject,
#     fig = mne.viz.plot_alignment(info, trans=trans, subject=subject,
#                                  subjects_dir=subjects_dir,
#                                  surfaces='head-dense',
#                                  show_axes=True, dig=True, eeg=[],
#                                  meg='sensors',
#                                  coord_frame='meg')

# n_jobs = 20
n_jobs = 20
parallel, run_func, _ = parallel_func(compute_fwd, n_jobs=n_jobs)

fwds_ = parallel(run_func(s, src_ref, info, trans, bem, mindist=3)
                 for s, info, trans, bem in zip(subjects, infos,
                                                trans_fname_s, bem_fname_s))

fwds = prepare_fwds(fwds_, src_ref, copy=False)

##################################################
# Solve the inverse problems with Multi-task Lasso
# ------------------------------------------------

# The Multi-task Lasso assumes the source locations are the same across
# subjects for all instants i.e if a source is zero for one subject, it will
# be zero for all subjects. "alpha" is a hyperparameter that controls this
# structured sparsity prior. it must be set as a positive number between 0
# and 1. With alpha = 1, all the sources are 0.

# We restric the time points around 20ms in order to reconstruct the sources of
# the N20 response.

from ABseq_func import evoked_funcs

evokeds = []
for subject in subjects:
    ev, _ = evoked_funcs.load_evoked(subject=subject, filter_name='items_standard_all', filter_not=None)
    for key in ev.keys():
        evokeds.append(ev[key][0])

# evokeds = [ev.crop(0.0, 0.25) for ev in evokeds]
evokeds = [ev.filter(l_freq=None, h_freq=30).apply_baseline((-0.050, 0)).crop(0.0, 0.25) for ev in evokeds]



for alpha in [0.3, 0.5, 0.8]:
    print('Computing group inverse with alpha=0.%i'% int(alpha * 10))
    stcs = compute_group_inverse(fwds, evokeds, noise_covs,
                                 method='multitasklasso',
                                 spatiotemporal=True,
                                 alpha=alpha)

    save_lasso_res_path = config.result_path + '/groupmne/lasso/'
    utils.create_folder(save_lasso_res_path)
    np.save(save_lasso_res_path + '/stcs_alpha0%i.npy' % int(alpha * 10), stcs)
    # sstcs = np.load(save_lasso_res_path+'/stcs_alpha0%i.npy'%int(alpha*10),allow_pickle=True)

# ############################################
# # Visualize stcs
# alpha = 0.3
# stcs = np.load(save_lasso_res_path+'/stcs_alpha0%i.npy'%int(alpha*10), allow_pickle=True)
# data = np.average([s.data for s in stcs], axis=0)
# stc = mne.SourceEstimate(data, stcs[0].vertices, stcs[0].tmin, stcs[0].tstep, stcs[0].subject)
# stc.plot(subjects_dir=subjects_dir, backend='auto', hemi='rh')

# ############################################
# # Let's visualize the N20 response. The stimulus was applied on the right
# # hand, thus we only show the left hemisphere. The activation is exactly in
# # the primary somatosensory cortex. We highlight the borders of the post
# # central gyrus.
#
#
# t = 0.02
# plot_kwargs = dict(
#     hemi='lh', subjects_dir=subjects_dir, views="lateral",
#     initial_time=t, time_unit='s', size=(800, 800),
#     smoothing_steps=5)
#
# t_idx = stcs[0].time_as_index(t)
#
# for stc, subject in zip(stcs, subjects):
#     g_post_central = mne.read_labels_from_annot(subject, "aparc.a2009s",
#                                                 subjects_dir=subjects_dir,
#                                                 regexp="G_postcentral-lh")[0]
#     n_sources = [stc.vertices[0].size, stc.vertices[1].size]
#     m = abs(stc.data[:n_sources[0], t_idx]).max()
#     plot_kwargs["clim"] = dict(kind='value', pos_lims=[0., 0.2 * m, m])
#     brain = stc.plot(**plot_kwargs)
#     brain.add_text(0.1, 0.9, "multi-subject-grouplasso (%s)" % subject,
#                    "title")
#     brain.add_label(g_post_central, borders=True, color="green")
#
# #####################################
# # Group MNE leads to better accuracy
# # ----------------------------------
# # To evaluate the effect of the joint inverse solution, we compute the
# # individual solutions independently for each subject
#
#
# for subject, fwd, evoked, cov in zip(subjects, fwds_, evokeds, noise_covs):
#     fwd_ = prepare_fwds([fwd], src_ref)
#     stc = compute_group_inverse(fwd_, [evoked], [cov],
#                                 method='multitasklasso',
#                                 spatiotemporal=True,
#                                 alpha=0.8)[0]
#     stc.subject = subject
#     g_post_central = mne.read_labels_from_annot(subject, "aparc.a2009s",
#                                                 subjects_dir=subjects_dir,
#                                                 regexp="G_postcentral-lh")[0]
#     n_sources = [stc.vertices[0].size, stc.vertices[1].size]
#     m = abs(stc.data[:n_sources[0], t_idx]).max()
#     plot_kwargs["clim"] = dict(kind='value', pos_lims=[0., 0.2 * m, m])
#     brain = stc.plot(**plot_kwargs)
#     brain.add_text(0.1, 0.9, "single-subject-grouplasso (%s)" % subject,
#                    "title")
#     brain.add_label(g_post_central, borders=True, color="green")

###########################################
# References
# ----------
# [1] Michael Lim, Justin M. Ales, Benoit R. Cottereau, Trevor Hastie,
# Anthony M. Norcia. Sparse EEG/MEG source estimation via a group lasso,
# PLOS ONE, 2017
#
# [2] Jussi Nurminen, Hilla Paananen, & Jyrki Mäkelä. (2017). High frequency
# somatosensory MEG: evoked responses, FreeSurfer reconstruction [Data set].
# Zenodo. http://doi.org/10.5281/zenodo.889235
