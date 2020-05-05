import os.path as op
import config
from ABseq_func import *
import mne
import numpy as np
from matplotlib import pyplot as plt
from importlib import reload
import pickle

"""
This performs basic cluster-corrected tests between 2 conditions
It first computes the signal difference between the average of the trials for conditions 1 and the average of the trials 
for condition 2, in each subject ("first level") (saved in results folder with pickle)
Then it performs a permutations-based one-sample test using theses Nsubjects signal differences ("second level") 

The pair of conditions to compare is defined using metadata filters below
(e.g., ['(StimPosition==1)', '(StimPosition==2)'])

/!/ Contrasts are currently computed independently for each seqID  /!/
"""

# =========================================================== #
# Options
# =========================================================== #
cleaned = True  # epochs cleaned with autoreject or not, only when using original epochs (resid_epochs=False)
resid_epochs = False  # use epochs created by regressing out surprise effects, instead of original epochs
resid_epochs_type = 'reg_repeataltern_surpriseOmegainfinity'  # 'residual_surprise'  'residual_model_constant' 'reg_repeataltern_surpriseOmegainfinity'
# /!\ if 'reg_repeataltern_surpriseOmegainfinity', epochs wil be loaded from '/results/linear_models' instead of '/data/MEG/'
DoFirstLevel = False  # To compute the contrasts (delta 2 conditions) and evoked for each subject
DoSecondLevel = True  # Run the group level statistics
analyses_to_do = ['OddEven', 'PairsOpen', 'PairsClose', 'QuadOpen', 'QuadClose',  'QuadOpenBis', 'QuadCloseBis',
                  'ChunkBeginning', 'ChunkBeginningBis', 'RepeatAlter']
analyses_to_do = ['OddEven']


# =========================================================== #
# Define metadatafilters for different analyses
# note: 'and SequenceID == x' is added later in the seqID loop
# =========================================================== #
filters = dict()  # contains analysis name & associated contrast (metadatafilters for cond1, for cond2)
filters['OddEven'] = ['(ViolationInSequence == 0 and (StimPosition==2 | StimPosition==4 | StimPosition==6 | StimPosition==8 | StimPosition==10 | StimPosition==12 | StimPosition==14))',
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
            result_path = op.join(config.result_path, 'Contrasts_tests', resid_epochs_type, analysis_name)
        else:
            result_path = op.join(config.result_path, 'Contrasts_tests', analysis_name)
        utils.create_folder(result_path)

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
            if resid_epochs and resid_epochs_type=='reg_repeataltern_surpriseOmegainfinity':
                print("Processing subject: %s" % subject)
                resid_path = op.join(config.result_path, 'linear_models', 'reg_repeataltern_surpriseOmegainfinity', subject)
                fname_in = op.join(resid_path, 'residuals-epo.fif')
                print("Input: ", fname_in)
                epochs = mne.read_epochs(fname_in, preload=True)
            elif resid_epochs:
                epochs = epoching_funcs.load_resid_epochs_items(subject, type=resid_epochs_type)
            else:
                if cleaned:
                    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
                    epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
                else:
                    epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
                    epochs = epoching_funcs.update_metadata_rejected(subject, epochs)

            # Generate list of evoked objects from conditions names

            # First allseq combined
            # Compute evoked
            cond_filters_seq = [cond_filters[0], cond_filters[1]]
            evokeds = [epochs[name].average() for name in cond_filters_seq]

            # Save evoked per condition (used later for plots)
            path_evo = op.join(config.meg_dir, subject, 'evoked')
            if cleaned:
                path_evo = path_evo + '_cleaned'
            evokeds[0].save(op.join(path_evo, analysis_name + '_allseq_cond' + str(1) + '-ave.fif'))
            evokeds[1].save(op.join(path_evo, analysis_name + '_allseq_cond' + str(2) + '-ave.fif'))

            # Store contrast for the subject & seqID (cond1 - cond2)
            allsubs_contrast_data['allseq'].append(mne.combine_evoked([evokeds[0], -evokeds[1]], 'equal'))

            # Then each seq
            for seqID in range(1, 8):
                # Add seqID to the condition filter
                cond_filters_seq = [cond_filters[0] + ' and SequenceID == "' + str(seqID) + '"',
                                    cond_filters[1] + ' and SequenceID == "' + str(seqID) + '"']
                # Compute evoked
                evokeds = [epochs[name].average() for name in cond_filters_seq]

                # Save evoked per condition (used later for plots)
                path_evo = op.join(config.meg_dir, subject, 'evoked')
                if cleaned:
                    path_evo = path_evo + '_cleaned'
                evokeds[0].save(op.join(path_evo, analysis_name + '_seqID' + str(seqID) + '_cond' + str(1) + '-ave.fif'))
                evokeds[1].save(op.join(path_evo, analysis_name + '_seqID' + str(seqID) + '_cond' + str(2) + '-ave.fif'))

                # Store contrast for the subject & seqID (cond1 - cond2)
                allsubs_contrast_data['Seq' + str(seqID)].append(mne.combine_evoked([evokeds[0], -evokeds[1]], 'equal'))

        # Save all subjects contrasts to a file
        with open(op.join(result_path, 'allsubs_contrast_data.pickle'), 'wb') as f:
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
            fig_path = op.join(config.fig_path, 'Contrasts_tests', resid_epochs_type, analysis_name)
            result_path = op.join(config.result_path, 'Contrasts_tests', resid_epochs_type, analysis_name)
        else:
            fig_path = op.join(config.fig_path, 'Contrasts_tests', analysis_name)
            result_path = op.join(config.result_path, 'Contrasts_tests', analysis_name)
        utils.create_folder(fig_path)

        # Load dictionary of 1st level contrasts
        with open(op.join(result_path, 'allsubs_contrast_data.pickle'), 'rb') as f:
            allsubs_contrast_data = pickle.load(f)

        # =========================================================== #
        # Group stats
        # =========================================================== #
        nperm = 500  # number of permutations
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

            ch_types = ['eeg', 'grad', 'mag']
            for ch_type in ch_types:
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
                    figname_initial = fig_path + op.sep + analysis_name + '_seq' + str(seqID) + '_stats_' + ch_type
                    # stats_funcs.plot_clusters(cluster_stats, p_threshold, data_stat, data_array_chtype, ch_type, T_obs_max=5., fname=analysis_name, figname_initial=figname_initial)
                    stats_funcs.plot_clusters(cluster_info, ch_type, T_obs_max=5., fname=analysis_name, figname_initial=figname_initial)


                # =========================================================== #
                # ==========  cluster evoked data plot
                # =========================================================== #

                if len(good_cluster_inds) > 0:
                    # ------------------ LOAD THE EVOKED FOR THE CURRENT CONDITION ------------ #
                    evoked_reg = evoked_funcs.load_evoked(subject='all', filter_name=analysis_name + '_seqID' + str(seqID), filter_not=None, cleaned=True)


                    # TMP SANITY CHECK: COND1/COND2/DELTA
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
                    # meank1 = savgol_filter(meank1, 11, 3)
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
                    # meank2 = savgol_filter(meank2, 11, 3)
                    # plt.figure()
                    # plt.plot(k1[0][0].times, meank1)
                    # plt.plot(k1[0][0].times, meank2)
                    # plt.figure()
                    # plt.plot(k1[0][0].times, meank1-meank2)
                    # plt.xlim([0.0, 0.50])




                    for i_clu, clu_idx in enumerate(good_cluster_inds):

                        cinfo = cluster_info[i_clu]

                        # # ----------------- SELECT CHANNELS OF INTEREST ----------------- #
                        # time_inds, space_inds = np.squeeze(clusters[clu_idx])
                        # ch_inds = np.unique(space_inds)  # list of channels we want to average and plot (!should be from one ch_type!)

                        # ----------------- PLOT ----------------- #
                        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                        # fig.suptitle(ch_type, fontsize=12, weight='bold')
                        plt.axvline(0, linestyle='-', color='black', linewidth=2)
                        for xx in range(3):
                            plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
                        ax.set_xlabel('Time (ms)')
                        condnames = list(evoked_reg.keys())
                        if len(condnames) == 2:
                            colorslist = ['r', 'b']
                        else:
                            NUM_COLORS = len(condnames)
                            cm = plt.get_cmap('jet')
                            colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
                        for ncond, condname in enumerate(condnames):
                            data = evoked_reg[condname].copy()
                            evoked_funcs.plot_evoked_with_sem_1cond(data, condname, ch_type, cinfo['channels_cluster'], color=colorslist[ncond], filter=True, axis=None)
                        ymin, ymax = ax.get_ylim()
                        ax.fill_betweenx((ymin, ymax), cinfo['sig_times'][0], cinfo['sig_times'][-1], color='orange', alpha=0.2)
                        # plt.legend(loc='upper right', fontsize=9)
                        ax.set_xlim([-100, 750])
                        ax.set_ylim([ymin, ymax])
                        plt.title(ch_type + '_' + analysis_name + '_seq' + str(seqID) + '_clust_' + str(i_clu+1), fontsize=10, weight='bold')
                        fig.tight_layout(pad=0.5, w_pad=0)
                        fig_name = fig_path + op.sep + analysis_name + '_seq' + str(seqID) + '_stats_' + ch_type + '_clust_' + str(i_clu+1) + '_evo.png'
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
                figname_initial = fig_path + op.sep + analysis_name + '_allseq_stats_' + ch_type
                # stats_funcs.plot_clusters(cluster_stats, p_threshold, data_stat, data_array_chtype, ch_type, T_obs_max=5., fname=analysis_name, figname_initial=figname_initial)
                stats_funcs.plot_clusters(cluster_info, ch_type, T_obs_max=5., fname=analysis_name, figname_initial=figname_initial)


            # =========================================================== #
            # ==========  cluster evoked data plot
            # =========================================================== #

            if len(good_cluster_inds) > 0:
                # ------------------ LOAD THE EVOKED FOR THE CURRENT CONDITION ------------ #
                evoked_reg = evoked_funcs.load_evoked(subject='all', filter_name=analysis_name + '_allseq', filter_not=None, cleaned=True)

                for i_clu, clu_idx in enumerate(good_cluster_inds):

                    cinfo = cluster_info[i_clu]

                    # # ----------------- SELECT CHANNELS OF INTEREST ----------------- #
                    # time_inds, space_inds = np.squeeze(clusters[clu_idx])
                    # ch_inds = np.unique(space_inds)  # list of channels we want to average and plot (!should be from one ch_type!)

                    # ----------------- PLOT ----------------- #
                    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
                    # fig.suptitle(ch_type, fontsize=12, weight='bold')
                    plt.axvline(0, linestyle='-', color='black', linewidth=2)
                    for xx in range(3):
                        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
                    ax.set_xlabel('Time (ms)')
                    condnames = list(evoked_reg.keys())
                    if len(condnames) == 2:
                        colorslist = ['r', 'b']
                    else:
                        NUM_COLORS = len(condnames)
                        cm = plt.get_cmap('jet')
                        colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
                    for ncond, condname in enumerate(condnames):
                        data = evoked_reg[condname].copy()
                        evoked_funcs.plot_evoked_with_sem_1cond(data, condname, ch_type, cinfo['channels_cluster'], color=colorslist[ncond], filter=True, axis=None)
                    ymin, ymax = ax.get_ylim()
                    ax.fill_betweenx((ymin, ymax), cinfo['sig_times'][0], cinfo['sig_times'][-1], color='orange', alpha=0.2)
                    # plt.legend(loc='upper right', fontsize=9)
                    ax.set_xlim([-100, 750])
                    ax.set_ylim([ymin, ymax])
                    plt.title(ch_type + '_' + analysis_name + '_allseq' + '_clust_' + str(i_clu+1), fontsize=10, weight='bold')
                    fig.tight_layout(pad=0.5, w_pad=0)
                    fig_name = fig_path + op.sep + analysis_name + '_allseq_stats_' + ch_type + '_clust_' + str(i_clu+1) + '_evo.png'
                    print('Saving ' + fig_name)
                    plt.savefig(fig_name, dpi=300)
                    plt.close('all')
