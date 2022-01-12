"""
==========================================================
  PLOT GFP FOR DIFFERENT CATEGORIES OF TRIALS
  + CORRELATION WITH COMPLEXITY
==========================================================

1 - Compute (or load) GFP for each subject and 4 categories of trials (items):
habituation, standard, violation, violation_minus_standard (epochs balanced with parameter 'sequence')

2 - Plot data:
    - line plot 7 sequences
    - correlation with complexity: heatmap
For habituation epochs, deviant_minus_standard epochs
"""

# ---- import the packages -------
import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import os.path as op
from ABseq_func import *
import config
import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import pickle
from ABseq_func import article_plotting_funcs

#  ============== ============== ============== ============== ============== ============== ============== ============
#                1 - LOAD OR EXTRACT THE DATA: GFP FOR EACH SUBJECT AND TRIAL TYPE
#  ============== ============== ============== ============== ============== ============== ============== ============

recompute_GFP = False
results_path = op.join(config.result_path, 'Corr_GFPxComplexity', 'items')
utils.create_folder(results_path)

if recompute_GFP:
    # Extract GFP for different categories of trials
    remap_grads = True  # combine mag et grads, cf https://mne.tools/stable/auto_examples/preprocessing/virtual_evoked.html
    baseline_epochs = True  # apply baseline to the epochs (if wasn't already done)

    ch_types = config.ch_types
    if remap_grads:
        ch_types = ['mag']

    # Empty dictionaries to fill
    gfp_data = {}
    for ttype in ['habituation', 'standard', 'violation', 'viol_minus_stand']:
        gfp_data[ttype] = {}
        for ch_type in ch_types:
            gfp_data[ttype][ch_type] = {}
            for seqID in range(1, 8):
                gfp_data[ttype][ch_type][seqID] = []

    # Extract the data: subjects loop
    for subject in config.subjects_list:
        print('-- Subject ' + subject)

        # -- LOAD THE EPOCHS -- #
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=True, AR_type='global')

        # ====== REMAP GRAD TO MAGS !! ====== #
        if remap_grads:
            print('Remapping grads to mags')
            epochs = epochs.as_type('mag')
            print(str(len(epochs.ch_names)) + ' remaining channels!')

        for ttype in ['habituation', 'standard', 'violation', 'viol_minus_stand']:
            print('  ---- Trial type ' + str(ttype))
            gfp_eeg_seq = []
            gfp_mag_seq = []
            gfp_grad_seq = []
            for seqID in range(1, 8):
                print('    ------ Sequence ' + str(seqID))
                if ttype == 'viol_minus_stand':
                    epochs_bal = epoching_funcs.balance_epochs_violation_positions(epochs.copy(), 'sequence')
                    epochs_subset = epochs_bal['SequenceID == ' + str(seqID)].copy()
                elif ttype == 'habituation':
                    epochs_subset = epochs['TrialNumber <= 10 and SequenceID == ' + str(seqID)].copy()
                elif ttype == 'standard':  # For standards, taking only items from trials with no violation
                    epochs_subset = epochs['TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationInSequence == 0'].copy()
                elif ttype == 'violation':
                    epochs_subset = epochs['TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationOrNot == 1'].copy()

                # Baseline
                if baseline_epochs:
                    epochs_subset = epochs_subset.apply_baseline((-0.050, 0))

                # Average epochs
                if ttype == 'viol_minus_stand':
                    # Here we compute a contrast between evoked
                    e1 = epochs_subset['ViolationOrNot == 0'].copy().average()
                    e2 = epochs_subset['ViolationOrNot == 1'].copy().average()
                    ev = mne.combine_evoked([e1, e2], weights=[-1, 1])
                    # ev = mne.combine_evoked([e2, -e1], 'equal')  # not equivalent ?! different nave and scaling
                else:
                    ev = epochs_subset.average()
                # clear
                epochs_subset = []

                # Compute GFP
                for ch_type in ch_types:
                    if ch_type == 'eeg':
                        gfp = np.sum(ev.copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
                        gfp_eeg_seq.append(gfp)
                    elif ch_type == 'mag':
                        gfp = np.sum(ev.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
                        gfp_mag_seq.append(gfp)
                    elif ch_type == 'grad':
                        gfp = np.sum(ev.copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
                        gfp_grad_seq.append(gfp)
                    # Store gfp each seq
                    gfp_data[ttype][ch_type][seqID].append(gfp)

            # Arrays of GFP for the 7 sequences
            if 'eeg' in ch_types:
                gfp_eeg_seq = np.array(gfp_eeg_seq)
            gfp_mag_seq = np.array(gfp_mag_seq)
            gfp_grad_seq = np.array(gfp_grad_seq)
        # clear
        epochs = []
    # Keep "times" in the dict
    gfp_data['times'] = ev.times

    # Save the data
    with open(op.join(results_path, 'gfp_each_seq_data.pickle'), 'wb') as f:
        pickle.dump(gfp_data, f, pickle.HIGHEST_PROTOCOL)
else:
    with open(op.join(results_path, 'gfp_each_seq_data.pickle'), 'rb') as f:
        gfp_data = pickle.load(f)

#  ============== ============== ============== ============== ============== ============== ============== ============
#                2 - PLOTS
#  ============== ============== ============== ============== ============== ============== ============== ============
from importlib import reload

reload(article_plotting_funcs)

#  ============== HABITUATION PLOTS ============== #
data_7seq = np.dstack(gfp_data['habituation']['mag'].values())
data_7seq = np.transpose(data_7seq, (2, 0, 1))
# Data line plot 7seq
article_plotting_funcs.plot_7seq_timecourses(data_7seq, gfp_data['times'] * 1000, save_fig_path='GFP/', fig_name='GFPxComplexity_Habituation', suffix='',
                                             pos_horizontal_bar=0.47, plot_pearson_corrComplexity=True, chance=None, xlims=[-50, 350], ymin=0, ylabel='GFP')
# Correlation with complexity heatmap
pearsonr = article_plotting_funcs.compute_corr_comp(data_7seq)
article_plotting_funcs.heatmap_avg_subj(pearsonr, gfp_data['times'] * 1000, xlims=[-50, 350], ylims=[-0.5, 0.5], fig_name=op.join(config.fig_path, 'GFP', 'GFPxComplexity_Habituation_heatmap_complexity_pearsonr.png'), figsize=(10, 0.5))
article_plotting_funcs.heatmap_avg_subj(pearsonr, gfp_data['times'] * 1000, xlims=[-50, 350], ylims=[-0.5, 0.5], fig_name=op.join(config.fig_path, 'GFP', 'GFPxComplexity_Habituation_heatmap_complexity_pearsonr.svg'), figsize=(10, 0.5))

#  ============== STAND PLOTS ============== #
data_7seq = np.dstack(gfp_data['standard']['mag'].values())
data_7seq = np.transpose(data_7seq, (2, 0, 1))
# Data line plot 7seq
article_plotting_funcs.plot_7seq_timecourses(data_7seq, gfp_data['times'] * 1000, save_fig_path='GFP/', fig_name='GFPxComplexity_Standard', suffix='',
                                             pos_horizontal_bar=0.47, plot_pearson_corrComplexity=True, chance=None, xlims=[-50, 350], ymin=0, ylabel='GFP')
# Correlation with complexity heatmap
pearsonr = article_plotting_funcs.compute_corr_comp(data_7seq)
article_plotting_funcs.heatmap_avg_subj(pearsonr, gfp_data['times'] * 1000, xlims=[-50, 350], ylims=[-0.5, 0.5], fig_name=op.join(config.fig_path, 'GFP', 'GFPxComplexity_Standard_heatmap_complexity_pearsonr.png'), figsize=(10, 0.5))
article_plotting_funcs.heatmap_avg_subj(pearsonr, gfp_data['times'] * 1000, xlims=[-50, 350], ylims=[-0.5, 0.5], fig_name=op.join(config.fig_path, 'GFP', 'GFPxComplexity_Standard_heatmap_complexity_pearsonr.svg'), figsize=(10, 0.5))

#  ============== DEV minus STAND PLOTS ============== #
data_7seq = np.dstack(gfp_data['viol_minus_stand']['mag'].values())
data_7seq = np.transpose(data_7seq, (2, 0, 1))
# Data line plot 7seq
article_plotting_funcs.plot_7seq_timecourses(data_7seq, gfp_data['times'] * 1000, save_fig_path='GFP/', fig_name='GFPxComplexity_viol_minus_stand', suffix='',
                                             pos_horizontal_bar=0.47, plot_pearson_corrComplexity=True, chance=None, xlims=[-50, 600], ymin=0, ylabel='GFP')
# Correlation with complexity heatmap
pearsonr = article_plotting_funcs.compute_corr_comp(data_7seq)
article_plotting_funcs.heatmap_avg_subj(pearsonr, gfp_data['times'] * 1000, xlims=[-50, 600], ylims=[-0.5, 0.5], fig_name=op.join(config.fig_path, 'GFP', 'GFPxComplexity_viol_minus_stand_heatmap_complexity_pearsonr.png'), figsize=(10, 1))
article_plotting_funcs.heatmap_avg_subj(pearsonr, gfp_data['times'] * 1000, xlims=[-50, 600], ylims=[-0.5, 0.5], fig_name=op.join(config.fig_path, 'GFP', 'GFPxComplexity_viol_minus_stand_heatmap_complexity_pearsonr.svg'), figsize=(10, 0.5))

article_plotting_funcs.heatmap_avg_subj(pearsonr, gfp_data['times'] * 1000, xlims=[-50, 600], ylims=[-0.5, 0.5], fig_name=op.join(config.fig_path, 'GFP', 'justforcolorbar.png'), figsize=(10, 1))
