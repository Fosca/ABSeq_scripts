import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")

import mne
import config
import os.path as op
from ABseq_func import *
import numpy as np
import pickle
from scipy.stats import sem, pearsonr




def extract_stc_per_sequence(subject,condition_name="habituation",baseline=True,morph_sources=False):

    epochs_items = epoching_funcs.load_epochs_items(subject)
    epochs_items = epochs_items.pick_types(meg=True, eeg=False, eog=False)  ## Exclude EEG (was done when computing inverse
    if baseline:
        epochs_items = epochs_items.apply_baseline(baseline=(-0.050, 0))

    if condition_name == 'habituation':
        # ==== HABITUATION DATA
        # -- Compute sources of evoked
        ev = epochs_items['TrialNumber < 11'].average()
        stcs = dict()
        for seqID in range(1, 8):
            ev = epochs_items['TrialNumber < 11 & SequenceID == ' + str(seqID)].average()
            stcs['seq'+str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=morph_sources)

    elif condition_name == 'standard':
        # ==== STANDARDS DATA
        # -- Compute sources of evoked
        ev = epochs_items['TrialNumber > 10 & ViolationInSequence == 0'].average()
        stcs = dict()
        for seqID in range(1, 8):
            ev = epochs_items['TrialNumber > 10 & ViolationInSequence == 0 & SequenceID == ' + str(seqID)].average()
            stcs['seq'+str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=morph_sources)

    elif condition_name == 'deviant':
    # -- Compute sources of evoked
        ev = epochs_items['ViolationOrNot == 1'].average()
        stcs = dict()
        for seqID in range(1, 8):
            ev = epochs_items['ViolationOrNot == 1 & SequenceID == ' + str(seqID)].average()
            stcs['seq'+str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject, ev, morph_sources=morph_sources)

    elif condition_name=='standard_minus_deviant':
        # ==== STAND VS DEV (balanced - but across seqIDs not for each seqID)
        # -- Compute sources of evoked
        ep_bal = epoching_funcs.balance_epochs_violation_positions(epochs_items, 'local_position')

        stcs = dict()
        for seqID in range(1, 8):
            ev1 = ep_bal['ViolationOrNot == 0 and SequenceID == ' + str(seqID)].average()
            ev2 = ep_bal['ViolationOrNot == 1 and SequenceID == ' + str(seqID)].average()
            cont = mne.combine_evoked([ev1,-ev2],weights='equal')
            stcs['seq'+str(seqID)] = source_estimation_funcs.compute_sources_from_evoked(subject,cont, morph_sources=morph_sources)

    else:
        ValueError("The condition name was not recognized")
        stcs = {}

    return stcs



def compute_correlation_with_complexity(subject,condition_name="habituation",baseline=True,morph_sources=False):

    stcs = extract_stc_per_sequence(subject,condition_name=condition_name,baseline=baseline,morph_sources=morph_sources)
    complexity = config.complexity
    complexity = np.asarray(list(complexity.values()))

    sources = list(stcs.values())
    sources = np.asarray([sour._data for sour in sources])

    correl = np.zeros((sources.shape[1],sources.shape[2]))
    for ii, time in enumerate(range(sources.shape[2])):
        print("Running for step %i out of %i"%(ii,sources.shape[2]))
        for vertex in range(sources.shape[1]):
            correl[vertex,time],p = pearsonr(sources[:,vertex,time],complexity)
    stc_correl = stcs['seq1'].copy()
    stc_correl._data = correl

    return stc_correl


def compute_correlation_comp_all_conditions(subject,baseline=True, morph_sources=False):

    stc_hab = compute_correlation_with_complexity(subject, condition_name="habituation", baseline=baseline, morph_sources=morph_sources)
    stc_standard = compute_correlation_with_complexity(subject, condition_name="standard", baseline=baseline, morph_sources=morph_sources)
    stc_deviant = compute_correlation_with_complexity(subject, condition_name="deviant", baseline=baseline, morph_sources=morph_sources)
    stc_standard_minus_deviant = compute_correlation_with_complexity(subject, condition_name="standard_minus_deviant", baseline=baseline, morph_sources=morph_sources)

    # save the stcs per subject

    results_path = op.join(config.result_path, 'Correlation_complexity/')
    utils.create_folder(results_path)

    with open(op.join(results_path, subject + '_stc_correl_complexity.pickle'), 'wb') as f:
        pickle.dump(stc_hab, f, pickle.HIGHEST_PROTOCOL)

    with open(op.join(results_path, subject + '_stc_correl_standard.pickle'), 'wb') as f:
        pickle.dump(stc_standard, f, pickle.HIGHEST_PROTOCOL)

    with open(op.join(results_path, subject + '_stc_correl_deviant.pickle'), 'wb') as f:
        pickle.dump(stc_deviant, f, pickle.HIGHEST_PROTOCOL)

    with open(op.join(results_path, subject + '_stc_correl_stc_standard_minus_deviant.pickle'), 'wb') as f:
        pickle.dump(stc_standard_minus_deviant, f, pickle.HIGHEST_PROTOCOL)