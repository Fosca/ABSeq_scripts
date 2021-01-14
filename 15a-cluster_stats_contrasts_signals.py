"""
This performs basic cluster-corrected tests between 2 conditions
It first computes the signal difference between the average of the trials for conditions 1 and the average of the trials
for condition 2, in each subject ("first level") (saved in results folder with pickle)
Then it performs a permutations-based one-sample test using theses Nsubjects signal differences ("second level")

The pair of conditions to compare is defined using metadata filters below
(e.g., ['(StimPosition==1)', '(StimPosition==2)'])

/!/ Contrasts are currently computed independently for each seqID  /!/
"""

import os.path as op
import config
from ABseq_func import *
import mne
import numpy as np
from matplotlib import pyplot as plt
from importlib import reload
import pickle

# Exclude some subjects
config.exclude_subjects.append('sub10-gp_190568')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

# =========================================================== #
# Options
# =========================================================== #
cleaned = True  # epochs cleaned with autoreject or not, only when using original epochs (resid_epochs=False)
resid_epochs = False  # use epochs created by regressing out surprise effects, instead of original epochs
use_balanced_epochs = True  # option to use only standard epochs with positions matched with the positions of deviants
use_baseline = True  # option to apply a baseline to the evokeds before running the contrast
lowpass_epochs = True  # option to filter epochs with  30Hz lowpass filter
if resid_epochs:
    resid_epochs_type = 'reg_repeataltern_surpriseOmegainfinity'  # 'residual_surprise'  'residual_model_constant' 'reg_repeataltern_surpriseOmegainfinity'
    # /!\ if 'reg_repeataltern_surpriseOmegainfinity', epochs wil be loaded from '/results/linear_models' instead of '/data/MEG/'
DoFirstLevel = True  # To compute the contrasts (delta 2 conditions) and evoked for each subject
DoSecondLevel = True  # Run the group level statistics
# analyses_to_do = ['OddEven', 'PairsOpen', 'PairsClose', 'QuadOpen', 'QuadClose',  'QuadOpenBis', 'QuadCloseBis',
#                   'ChunkBeginning', 'ChunkBeginningBis', 'RepeatAlter']
analyses_to_do = ['Viol_vs_Stand']

# =========================================================== #
# Define metadatafilters for different analyses
# note: 'and SequenceID == x' is added later in the seqID loop
# =========================================================== #
filters = dict()  # contains analysis name & associated contrast (metadatafilters for cond1, for cond2)
filters['OddEven'] = [
    '(ViolationInSequence == 0 and (StimPosition==2 | StimPosition==4 | StimPosition==6 | StimPosition==8 | StimPosition==10 | StimPosition==12 | StimPosition==14))',
    '(ViolationInSequence == 0 and (StimPosition==3 | StimPosition==5 | StimPosition==7 | StimPosition==9 | StimPosition==11 | StimPosition==13 | StimPosition==15))']
filters['PairsOpen'] = ['(ViolationInSequence == 0 and (StimPosition==5 | StimPosition==9 | StimPosition==13))',
                        '(ViolationInSequence == 0 and (StimPosition==7 | StimPosition==11 | StimPosition==15))']
filters['PairsClose'] = ['(ViolationInSequence == 0 and (StimPosition==4 | StimPosition==8 | StimPosition==12))',
                         '(ViolationInSequence == 0 and (StimPosition==2 | StimPosition==6 | StimPosition==10))']
filters['QuadOpen'] = ['(ViolationInSequence == 0 and (StimPosition==5))',
                       '(ViolationInSequence == 0 and (StimPosition==9))']
filters['QuadClose'] = ['(ViolationInSequence == 0 and (StimPosition==8))',
                        '(ViolationInSequence == 0 and (StimPosition==12))']
filters['QuadOpenBis'] = ['(ViolationInSequence == 0 and (StimPosition==5 | StimPosition==13))',
                          '(ViolationInSequence == 0 and (StimPosition==9))']
filters['QuadCloseBis'] = ['(ViolationInSequence == 0 and (StimPosition==8))',
                           '(ViolationInSequence == 0 and (StimPosition==12 | StimPosition==4))']
filters['ChunkBeginning'] = ['(ViolationInSequence == 0 and (ChunkBeginning==0))',
                             '(ViolationInSequence == 0 and (ChunkBeginning==1))']
filters['ChunkBeginningBis'] = ['(ViolationInSequence == 0 and (ChunkBeginning==0 and StimPosition!=1 and StimPosition!=16))',
                                '(ViolationInSequence == 0 and (ChunkBeginning==1 and StimPosition!=1 and StimPosition!=16))']
filters['RepeatAlter'] = ['(ViolationInSequence == 0 and (RepeatAlter==0))',
                          '(ViolationInSequence == 0 and (RepeatAlter==1))']
filters['Viol_OddEven'] = [
    '(ViolationOrNot == 1 and (StimPosition==2 | StimPosition==4 | StimPosition==6 | StimPosition==8 | StimPosition==10 | StimPosition==12 | StimPosition==14))',
    '(ViolationOrNot == 1 and (StimPosition==3 | StimPosition==5 | StimPosition==7 | StimPosition==9 | StimPosition==11 | StimPosition==13 | StimPosition==15))']
filters['Viol_PairsOpen'] = ['(ViolationOrNot == 1 and (StimPosition==5 | StimPosition==9 | StimPosition==13))',
                             '(ViolationOrNot == 1 and (StimPosition==7 | StimPosition==11 | StimPosition==15))']
filters['Viol_PairsClose'] = ['(ViolationOrNot == 1 and (StimPosition==4 | StimPosition==8 | StimPosition==12))',
                              '(ViolationOrNot == 1 and (StimPosition==2 | StimPosition==6 | StimPosition==10))']
filters['Viol_QuadOpen'] = ['(ViolationOrNot == 1 and (StimPosition==5))',
                            '(ViolationOrNot == 1 and (StimPosition==9))']
filters['Viol_QuadClose'] = ['(ViolationOrNot == 1 and (StimPosition==8))',
                             '(ViolationOrNot == 1 and (StimPosition==12))']
filters['Viol_QuadOpenBis'] = ['(ViolationOrNot == 1 and (StimPosition==5 | StimPosition==13))',
                               '(ViolationOrNot == 1 and (StimPosition==9))']
filters['Viol_QuadCloseBis'] = ['(ViolationOrNot == 1 and (StimPosition==8))',
                                '(ViolationOrNot == 1 and (StimPosition==12 | StimPosition==4))']
filters['Viol_ChunkBeginning'] = ['(ViolationOrNot == 1 and (ChunkBeginning==0))',
                                  '(ViolationOrNot == 1 and (ChunkBeginning==1))']
filters['Viol_ChunkBeginningBis'] = ['(ViolationOrNot == 1 and (ChunkBeginning==0 and StimPosition!=1 and StimPosition!=16))',
                                     '(ViolationOrNot == 1 and (ChunkBeginning==1 and StimPosition!=1 and StimPosition!=16))']
filters['Viol_RepeatAlter'] = ['(ViolationOrNot == 1 and (RepeatAlter==0))',
                               '(ViolationOrNot == 1 and (RepeatAlter==1))']
filters['Viol_vs_Stand'] = ['(ViolationOrNot == 1)',
                            '(ViolationOrNot == 0)']

print('cleaned = ' + str(cleaned))
print('resid_epochs = ' + str(resid_epochs))
print('use_balanced_epochs = ' + str(use_balanced_epochs))
print('use_baseline = ' + str(use_baseline))
print('lowpass_epochs = ' + str(lowpass_epochs))
print('DoFirstLevel = ' + str(DoFirstLevel))
print('DoSecondLevel = ' + str(DoSecondLevel))

if DoFirstLevel:
    # =========================================================== #
    # Set contrast analysis with corresponding metadata filters
    # =========================================================== #
    for analysis_name in analyses_to_do:
        cond_filters = filters[analysis_name]
        print('\n#=====================================================================#\n        Analysis: '
              + analysis_name + '\n' + cond_filters[0])
        print('vs.\n' + cond_filters[1] + '\n#=====================================================================#\n')
        if resid_epochs:
            results_path = op.join(config.result_path, 'Paired_contrasts', analysis_name, 'TP_corrected_data', 'Signals')
        else:
            results_path = op.join(config.result_path, 'Paired_contrasts', analysis_name, 'Original_data', 'Signals')
        if use_baseline:
            results_path = results_path + op.sep + 'With_baseline_correction'
        utils.create_folder(results_path)

        # =========================================================== #
        # Compute the contrast for each seqID at the subject's level
        # =========================================================== #
        allsubs_contrast_data = dict()
        allsubs_contrast_data['Seq1'] = []
        allsubs_contrast_data['Seq2'] = []
        allsubs_contrast_data['Seq3'] = []
        allsubs_contrast_data['Seq4'] = []
        allsubs_contrast_data['Seq5'] = []
        allsubs_contrast_data['Seq6'] = []
        allsubs_contrast_data['Seq7'] = []
        allsubs_contrast_data['allseq'] = []
        for nsub, subject in enumerate(config.subjects_list):
            # Load data & update metadata (in case new things were added)
            if resid_epochs and resid_epochs_type == 'reg_repeataltern_surpriseOmegainfinity':
                print("Processing subject: %s" % subject)
                resid_path = op.join(config.result_path, 'linear_models', 'reg_repeataltern_surpriseOmegainfinity', subject)
                fname_in = op.join(resid_path, 'residuals-epo.fif')
                print("Input: ", fname_in)
                epochs = mne.read_epochs(fname_in, preload=True)
            # elif resid_epochs:
            #     epochs = epoching_funcs.load_resid_epochs_items(subject, type=resid_epochs_type)
            else:
                if cleaned:
                    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
                    # epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
                else:
                    epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
                    epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
            if use_balanced_epochs:
                epochs = epoching_funcs.balance_epochs_violation_positions(epochs, balance_violation_standards=False)
            if lowpass_epochs:
                print('Low pass filtering...')
                epochs = epochs.filter(l_freq=None, h_freq=30)  # default parameters (maybe should filter raw data instead of epochs...)

            # ====== Generate list of evoked objects from conditions names
            # --- First allseq combined
            # Compute evoked
            cond_filters_seq = [cond_filters[0], cond_filters[1]]
            if use_baseline:
                print('Baseline correction...')
                evokeds = [epochs[name].average().apply_baseline(baseline=(-0.050, 0.000)) for name in cond_filters_seq]
            else:
                evokeds = [epochs[name].average() for name in cond_filters_seq]

            # Save evoked per condition (used later for plots)
            path_evo = op.join(config.meg_dir, subject, 'evoked')
            if cleaned:
                path_evo = op.join(config.meg_dir, subject, 'evoked_cleaned')
            if resid_epochs:
                path_evo = op.join(config.meg_dir, subject, 'evoked_resid')
            print('Saving: ' + op.join(path_evo, analysis_name + '_allseq_cond' + str(1) + '-ave.fif'))
            evokeds[0].save(op.join(path_evo, analysis_name + '_allseq_cond' + str(1) + '-ave.fif'))
            print('Saving: ' + op.join(path_evo, analysis_name + '_allseq_cond' + str(2) + '-ave.fif'))
            evokeds[1].save(op.join(path_evo, analysis_name + '_allseq_cond' + str(2) + '-ave.fif'))
            # Store contrast for the subject & seqID (cond1 - cond2)
            allsubs_contrast_data['allseq'].append(mne.combine_evoked([evokeds[0], -evokeds[1]], 'equal'))

            # --- Then each seq
            for seqID in range(1, 8):
                # Add seqID to the condition filter
                cond_filters_seq = [cond_filters[0] + ' and SequenceID == "' + str(seqID) + '"',
                                    cond_filters[1] + ' and SequenceID == "' + str(seqID) + '"']
                # Compute evoked
                if use_baseline:
                    print('Baseline correction...')
                    evokeds = [epochs[name].average().apply_baseline(baseline=(-0.050, 0.000)) for name in cond_filters_seq]
                else:
                    evokeds = [epochs[name].average() for name in cond_filters_seq]
                # Save evoked per condition (used later for plots)
                path_evo = op.join(config.meg_dir, subject, 'evoked')
                if cleaned:
                    path_evo = op.join(config.meg_dir, subject, 'evoked_cleaned')
                if resid_epochs:
                    path_evo = op.join(config.meg_dir, subject, 'evoked_resid')
                print('Saving: ' + op.join(path_evo, analysis_name + '_seqID' + str(seqID) + '_cond' + str(1) + '-ave.fif'))
                evokeds[0].save(op.join(path_evo, analysis_name + '_seqID' + str(seqID) + '_cond' + str(1) + '-ave.fif'))
                print('Saving: ' + op.join(path_evo, analysis_name + '_seqID' + str(seqID) + '_cond' + str(2) + '-ave.fif'))
                evokeds[1].save(op.join(path_evo, analysis_name + '_seqID' + str(seqID) + '_cond' + str(2) + '-ave.fif'))
                # Store contrast for the subject & seqID (cond1 - cond2)
                allsubs_contrast_data['Seq' + str(seqID)].append(mne.combine_evoked([evokeds[0], -evokeds[1]], 'equal'))

        # Save all subjects contrasts to a file
        with open(op.join(results_path, 'allsubs_contrast_data.pickle'), 'wb') as f:
            pickle.dump(allsubs_contrast_data, f, pickle.HIGHEST_PROTOCOL)

if DoSecondLevel:
    for analysis_name in analyses_to_do:
        # =========================================================== #
        # Load 1st-level contrast data & set folders
        # =========================================================== #
        cond_filters = filters[analysis_name]
        print('\n#=====================================================================#\n        Analysis: '
              + analysis_name + '\n' + cond_filters[0])
        print('vs.\n' + cond_filters[1] + '\n#=====================================================================#\n')
        if resid_epochs:
            results_path = op.join(config.result_path, 'Paired_contrasts', analysis_name, 'TP_corrected_data', 'Signals')
        else:
            results_path = op.join(config.result_path, 'Paired_contrasts', analysis_name, 'Original_data', 'Signals')
        if use_baseline:
            results_path = results_path + op.sep + 'With_baseline_correction'
        utils.create_folder(results_path)

        # Load dictionary of 1st level contrasts
        with open(op.join(results_path, 'allsubs_contrast_data.pickle'), 'rb') as f:
            allsubs_contrast_data = pickle.load(f)

        # =========================================================== #
        # Group stats
        # =========================================================== #
        nperm = 5000  # number of permutations
        threshold = None  # If threshold is None, t-threshold equivalent to p < 0.05 (if t-statistic)
        p_threshold = 0.05
        tmin = 0.000  # timewindow to test (crop data)
        tmax = 0.500  # timewindow to test (crop data)

        # ==== For each seqID === #
        for seqID in range(1, 8):

            # Combine allsubjects data as one 'epochs array' object
            allsubs_contrast_data_seq = allsubs_contrast_data['Seq' + str(seqID)].copy()
            dataEpArray = mne.EpochsArray(np.asarray([allsubs_contrast_data_seq[i].data for i in range(len(allsubs_contrast_data_seq))]),
                                          allsubs_contrast_data['Seq' + str(seqID)][0].info, tmin=-0.1)

            # RUN THE STATS
            dataEpArray.crop(tmin=tmin, tmax=tmax)  # crop
            for ch_type in ['eeg', 'grad', 'mag']:
                data_stat = dataEpArray.copy()  # to avoid modifying data (e.g. when picking ch_type)
                print('\n\nseqID' + str(seqID) + ', ch_type ' + ch_type)
                cluster_stats, data_array_chtype, _ = stats_funcs.run_cluster_permutation_test_1samp(data_stat, ch_type=ch_type, nperm=nperm, threshold=threshold, n_jobs=6, tail=0)
                cluster_info = stats_funcs.extract_info_cluster(cluster_stats, p_threshold, data_stat, data_array_chtype, ch_type)

                # Significant clusters
                T_obs, clusters, p_values, _ = cluster_stats
                good_cluster_inds = np.where(p_values < p_threshold)[0]
                print("Good clusters: %s" % good_cluster_inds)

                # PLOT CLUSTERS
                if len(good_cluster_inds) > 0:
                    figname_initial = results_path + op.sep + analysis_name + '_seq' + str(seqID) + '_stats_' + ch_type
                    stats_funcs.plot_clusters(cluster_info, ch_type, T_obs_max=5., fname=analysis_name, figname_initial=figname_initial, filter_smooth=False)

                # =========================================================== #
                # ==========  cluster evoked data plot
                # =========================================================== #

                if len(good_cluster_inds) > 0:
                    # ------------------ LOAD THE EVOKED FOR THE CURRENT CONDITION ------------ #
                    if resid_epochs:
                        evoked_reg, path_evo = evoked_funcs.load_evoked(subject='all', filter_name=analysis_name + '_seqID' + str(seqID), filter_not='delta', cleaned=True, evoked_resid=True)
                    else:
                        evoked_reg, path_evo = evoked_funcs.load_evoked(subject='all', filter_name=analysis_name + '_seqID' + str(seqID), filter_not='delta', cleaned=True, evoked_resid=False)

                    # # ---------------------------------------------------------- #
                    # # TMP SANITY CHECK: COND1/COND2/DELTA
                    # cinfo = cluster_info[0]
                    # ch_inds = cinfo['channels_cluster']
                    # k = list(evoked_reg.keys())
                    # k1 = list(evoked_reg[k[0]])
                    # k2 = list(evoked_reg[k[1]])
                    # group_data_seq = []
                    # for nn in range(len(k1)):
                    #     sub_data = k1[nn][0].copy()
                    #     if ch_type == 'eeg':
                    #         sub_data = np.array(sub_data.pick_types(meg=False, eeg=True)._data)
                    #     elif ch_type == 'mag':
                    #         sub_data = np.array(sub_data.pick_types(meg='mag', eeg=False)._data)
                    #     elif ch_type == 'grad':
                    #         sub_data = np.array(sub_data.pick_types(meg='grad', eeg=False)._data)
                    #     group_data_seq.append(sub_data[ch_inds].mean(axis=0))
                    # meank1 = np.mean(group_data_seq, axis=0)
                    # # meank1 = savgol_filter(meank1, 11, 3)
                    # group_data_seq = []
                    # for nn in range(len(k2)):
                    #     sub_data = k2[nn][0].copy()
                    #     if ch_type == 'eeg':
                    #         sub_data = np.array(sub_data.pick_types(meg=False, eeg=True)._data)
                    #     elif ch_type == 'mag':
                    #         sub_data = np.array(sub_data.pick_types(meg='mag', eeg=False)._data)
                    #     elif ch_type == 'grad':
                    #         sub_data = np.array(sub_data.pick_types(meg='grad', eeg=False)._data)
                    #     group_data_seq.append(sub_data[ch_inds].mean(axis=0))
                    # meank2 = np.mean(group_data_seq, axis=0)
                    # # meank2 = savgol_filter(meank2, 11, 3)
                    # fig, ax_topo = plt.subplots(1, 1, figsize=(7, 2.))
                    # plt.plot(k1[0][0].times, meank1)
                    # plt.plot(k1[0][0].times, meank2)
                    # plt.savefig('tmpfig1')
                    # fig, ax_topo = plt.subplots(1, 1, figsize=(7, 2.))
                    # plt.plot(k1[0][0].times, meank1-meank2)
                    # plt.xlim([0.0, 0.50])
                    # plt.savefig('tmpfig2')
                    # plt.close('all')
                    # # ---------------------------------------------------------- #

                    for i_clu, clu_idx in enumerate(good_cluster_inds):
                        cinfo = cluster_info[i_clu]
                        # ----------------- PLOT ----------------- #
                        stats_funcs.plot_clusters_evo(evoked_reg, cinfo, ch_type, i_clu, analysis_name=analysis_name + '_seq' + str(seqID), filter_smooth=False)
                        fig_name = results_path + op.sep + analysis_name + '_seq' + str(seqID) + '_stats_' + ch_type + '_clust_' + str(i_clu + 1) + '_evo.png'
                        print('Saving ' + fig_name)
                        plt.savefig(fig_name, dpi=300)
                        plt.close('all')

        # ==== Then for allseq combined  === #
        # Combine allsubjects data as one 'epochs array' object
        allsubs_contrast_data_seq = allsubs_contrast_data['allseq'].copy()
        dataEpArray = mne.EpochsArray(np.asarray([allsubs_contrast_data_seq[i].data for i in range(len(allsubs_contrast_data_seq))]), allsubs_contrast_data['allseq'][0].info, tmin=-0.1)

        # RUN THE STATS
        dataEpArray.crop(tmin=tmin, tmax=tmax)  # crop
        ch_types = ['eeg', 'grad', 'mag']
        for ch_type in ch_types:
            data_stat = dataEpArray.copy()  # to avoid modifying data (e.g. when picking ch_type)
            print('\n\nallseq' + ', ch_type ' + ch_type)
            cluster_stats, data_array_chtype, _ = stats_funcs.run_cluster_permutation_test_1samp(data_stat, ch_type=ch_type, nperm=nperm, threshold=threshold, n_jobs=6, tail=0)
            cluster_info = stats_funcs.extract_info_cluster(cluster_stats, p_threshold, data_stat, data_array_chtype, ch_type)

            # Significant clusters
            T_obs, clusters, p_values, _ = cluster_stats
            good_cluster_inds = np.where(p_values < p_threshold)[0]
            print("Good clusters: %s" % good_cluster_inds)

            # PLOT CLUSTERS
            if len(good_cluster_inds) > 0:
                figname_initial = results_path + op.sep + analysis_name + '_allseq_stats_' + ch_type
                stats_funcs.plot_clusters(cluster_info, ch_type, T_obs_max=5., fname=analysis_name, figname_initial=figname_initial, filter_smooth=False)

            # =========================================================== #
            # ==========  cluster evoked data plot
            # =========================================================== #

            if len(good_cluster_inds) > 0:
                # ------------------ LOAD THE EVOKED FOR THE CURRENT CONDITION ------------ #
                if resid_epochs:
                    evoked_reg, _ = evoked_funcs.load_evoked(subject='all', filter_name=analysis_name + '_allseq', filter_not='delta', cleaned=True, evoked_resid=True)
                else:
                    evoked_reg, _ = evoked_funcs.load_evoked(subject='all', filter_name=analysis_name + '_allseq', filter_not='delta', cleaned=True, evoked_resid=False)

                for i_clu, clu_idx in enumerate(good_cluster_inds):
                    cinfo = cluster_info[i_clu]
                    # ----------------- PLOT ----------------- #
                    stats_funcs.plot_clusters_evo(evoked_reg, cinfo, ch_type, i_clu, analysis_name=analysis_name + '_allseq', filter_smooth=False)
                    fig_name = results_path + op.sep + analysis_name + '_allseq' + '_stats_' + ch_type + '_clust_' + str(i_clu + 1) + '_evo.png'
                    print('Saving ' + fig_name)
                    plt.savefig(fig_name, dpi=300)
                    plt.close('all')
