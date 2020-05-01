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

/!/ Analysis is currently performed with all non-violated sequences (from hab & test blocks) /!/
/!/ Contrasts are currently computed independently for each seqID  /!/
"""

# =========================================================== #
# Options
# =========================================================== #
cleaned = True  # epochs cleaned with autoreject or not, only when using original epochs
resid_epochs = False  # use epochs created by regressing out surprise effects, instead of original epochs
resid_epochs_type = 'residual_model_constant'
DoFirstLevel = True  # To compute the contrasts (delta 2 conditions) and evoked for each subject
DoSecondLevel = True  # Run the group level statistics
analyses_to_do = ['OddEven', 'PairsOpen', 'PairsClose', 'QuadOpen', 'QuadClose',  'QuadOpenBis', 'QuadCloseBis',
                  'ChunkBeginning', 'ChunkBeginningBis']

# =========================================================== #
# Define metadatafilters for different analyses
# note: 'and SequenceID == x' is added later in the seqID loop
# =========================================================== #
filters = dict()  # contains analysis name & associated contrast (metadatafilters for cond1, for cond2)
filters['OddEven'] = [
    '(StimPosition==2 | StimPosition==4 | StimPosition==6 | StimPosition==8 | StimPosition==10 | StimPosition==12 | StimPosition==14)',
    '(StimPosition==3 | StimPosition==5 | StimPosition==7 | StimPosition==9 | StimPosition==11 | StimPosition==13 | StimPosition==15)']
filters['PairsOpen'] = [
    '(StimPosition==5 | StimPosition==9 | StimPosition==13)',
    '(StimPosition==7 | StimPosition==11 | StimPosition==15)']
filters['PairsClose'] = [
    '(StimPosition==4 | StimPosition==8 | StimPosition==12)',
    '(StimPosition==2 | StimPosition==6 | StimPosition==10)']
filters['QuadOpen'] = [
    '(StimPosition==5)',
    '(StimPosition==9)']
filters['QuadClose'] = [
    '(StimPosition==8)',
    '(StimPosition==12)']
filters['QuadOpenBis'] = [
    '(StimPosition==5 | StimPosition==13)',
    '(StimPosition==9)']
filters['QuadCloseBis'] = [
    '(StimPosition==8)',
    '(StimPosition==12 | StimPosition==4)']
filters['ChunkBeginning'] = [
    '(ChunkBeginning==0)',
    '(ChunkBeginning==1)']
filters['ChunkBeginningBis'] = [
    '(ChunkBeginning==0 and StimPosition!=1 and StimPosition!=16)',
    '(ChunkBeginning==1 and StimPosition!=1 and StimPosition!=16)']


if DoFirstLevel:
    # =========================================================== #
    # Set contrast analysis with corresponding metadata filters
    # =========================================================== #
    for analysis_name in analyses_to_do:
        cond_filters = filters[analysis_name]
        print('\n#=====================================================================#\n        Analysis: ' + analysis_name + '\n' + cond_filters[0])
        print('vs.\n' + cond_filters[1] + '\n#=====================================================================#\n')
        if resid_epochs:
            fig_path = op.join(config.fig_path, 'Contrasts_tests', 'residSurprise', analysis_name)
            result_path = op.join(config.result_path, 'Contrasts_tests', 'residSurprise', analysis_name)
        else:
            fig_path = op.join(config.fig_path, 'Contrasts_tests', analysis_name)
            result_path = op.join(config.result_path, 'Contrasts_tests', 'residSurprise', analysis_name)
        utils.create_folder(fig_path)
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
        for nsub, subject in enumerate(config.subjects_list):

            # Load data & update metadata (in case new things were added)
            if resid_epochs:
                epochs = epoching_funcs.load_resid_epochs_items(subject, type=resid_epochs_type)
            else:
                if cleaned:
                    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
                    epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
                else:
                    epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
                    epochs = epoching_funcs.update_metadata_rejected(subject, epochs)

            # ====== remove sequences with deviants from the comparison ========== #
            print('We remove items from trials with violation')
            epochs = epochs["ViolationInSequence == 0"]

            # Generate list of evoked objects from conditions names
            for seqID in range(1, 8):
                # Add seqID to the condition filter
                cond_filters_seq = [cond_filters[0] + ' and SequenceID == "' + str(seqID) + '"',
                                    cond_filters[1] + ' and SequenceID == "' + str(seqID) + '"']
                # Computed evoked
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
        # Set contrast analysis with corresponding metadata filters
        # =========================================================== #
        for analysis_name in analyses_to_do:
            cond_filters = filters[analysis_name]
            print(
                '\n#=====================================================================#\n        Analysis: ' + analysis_name + '\n' +
                cond_filters[0])
            print('vs.\n' + cond_filters[
                1] + '\n#=====================================================================#\n')
            if resid_epochs:
                fig_path = op.join(config.fig_path, 'Contrasts_tests', 'residSurprise', analysis_name)
                result_path = op.join(config.result_path, 'Contrasts_tests', 'residSurprise', analysis_name)
            else:
                fig_path = op.join(config.fig_path, 'Contrasts_tests', analysis_name)
                result_path = op.join(config.result_path, 'Contrasts_tests', 'residSurprise', analysis_name)

        # =========================================================== #
        # Group stats
        # =========================================================== #
        # Load dictionary of 1st level contrasts
        with open(op.join(result_path, 'allsubs_contrast_data.pickle'), 'rb') as f:
            allsubs_contrast_data = pickle.load(f)

        for seqID in range(1, 8):

            # Combine allsubjects data as one 'epochs array' object
            allsubs_contrast_data_seq = allsubs_contrast_data['Seq' + str(seqID)].copy()
            dataEpArray = mne.EpochsArray(np.asarray([allsubs_contrast_data_seq[i].data for i in range(len(allsubs_contrast_data_seq))]),
                                            allsubs_contrast_data['Seq' + str(seqID)][0].info, tmin=-0.1)

            # RUN THE STATS
            nperm = 5000  # number of permutations
            threshold = None  # If threshold is None, t-threshold equivalent to p < 0.05 (if t-statistic)
            p_threshold = 0.01
            tmin = 0.000  # timewindow to test (crop data)
            tmax = 0.500  # timewindow to test (crop data)
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
                    stats_funcs.plot_clusters(cluster_stats, p_threshold, data_stat, data_array_chtype, ch_type, T_obs_max=5., fname=analysis_name, figname_initial=figname_initial)

                # =========================================================== #
                # ==========  cluster evoked data plot
                # =========================================================== #

                if len(good_cluster_inds) > 0:
                    # ------------------ LOAD THE EVOKED FOR THE CURRENT CONDITION ------------ #
                    evoked_reg = evoked_funcs.load_evoked(subject='all', filter_name=analysis_name + '_seqID' + str(seqID), filter_not=None, cleaned=True)

                    for i_clu, clu_idx in enumerate(good_cluster_inds):

                        # ----------------- SELECT CHANNELS OF INTEREST ----------------- #
                        time_inds, space_inds = np.squeeze(clusters[clu_idx])
                        ch_inds = np.unique(space_inds)  # list of channels we want to average and plot (!should be from one ch_type!)

                        # ----------------- PLOT ----------------- #
                        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                        fig.suptitle(ch_type, fontsize=12, weight='bold')
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
                            evoked_funcs.plot_evoked_with_sem_1cond(data, condname, ch_type, ch_inds, color=colorslist[ncond], filter=True, axis=None)
                        plt.legend(loc='upper right', fontsize=9)
                        ax.set_xlim([-100, 750])
                        plt.title(ch_type + '_' + analysis_name + '_seq' + str(seqID) + '_clust_' + str(i_clu+1))
                        fig_name = fig_path + op.sep + analysis_name + '_seq' + str(seqID) + '_stats_' + ch_type + '_clust_' + str(i_clu+1) + '_evo.png'
                        print('Saving ' + fig_name)
                        plt.savefig(fig_name, dpi=300)
                        plt.close('all')
