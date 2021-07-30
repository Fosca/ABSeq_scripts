import os.path as op
import os
import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths

from ABseq_func import TP_funcs, epoching_funcs
import config
import numpy as np


def update_metadata_epochs_and_save_epochs(subject):
    """
    This function updates the metadata fields for the epochs such that they contain all the useful information for
    the complexity and surprise regressions.
    """

    # update the metadata for the non-clean epochs by adding the surprise computed for an observer that has 100 items in memory.
    metadata_notclean = TP_funcs.from_epochs_to_surprise(subject, [100])
    epochs_notclean, fname = epoching_funcs.load_epochs_items(subject, cleaned=False,return_fname=True)

    # load the metadata for the non-cleaned epochs, remove the bad ones, and this becomes the metadata for the cleaned epochs
    metadata_clean = TP_funcs.append_surprise_to_metadata_clean(subject)
    epochs_clean, fname_clean = epoching_funcs.load_epochs_items(subject, cleaned=True,return_fname=True)

    # ============ build the repeatAlter and the surprise 100 for n+1 ==================

    # 1 - update the full epochs (not_clean) metadata with the new fields
    RepeatAlternp1_notclean = metadata_notclean["RepeatAlter"].values[1:].tolist()
    RepeatAlternp1_notclean.append(np.nan)
    Surprisenp1_notclean = metadata_notclean["surprise_100"].values[1:].tolist()
    Surprisenp1_notclean.append(np.nan)

    metadata_notclean.assign(Intercept=1)
    metadata_notclean.assign(RepeatAlternp1=RepeatAlternp1_notclean)
    metadata_notclean.assign(Surprisenp1=Surprisenp1_notclean)
    epochs_notclean.metadata = metadata_notclean
    epochs_notclean.save(fname,overwrite = True)

    # 2 - subselect only the good epochs indices to filter the metadata
    good_idx = np.where([len(epochs_clean.drop_log[i]) == 0 for i in range(len(epochs_clean.drop_log))])[0]
    RepeatAlternp1 = np.asarray(RepeatAlternp1_notclean)[good_idx]
    Surprisenp1 = np.asarray(Surprisenp1_notclean)[good_idx]

    metadata_clean = metadata_clean.assign(Intercept=1)  # Add an intercept for later
    metadata_clean = metadata_clean.assign(RepeatAlternp1=RepeatAlternp1)
    metadata_clean = metadata_clean.assign(Surprisenp1=Surprisenp1)  # Add an intercept for later

    epochs_clean.metadata = metadata_clean
    epochs_clean.save(fname_clean,overwrite = True)

    return True


def filter_good_epochs_for_regression_analysis(subject,clean=True,fields_of_interest = ['surprise_100','RepeatAlternp1']):
    """
    This function removes the epochs that have Nans in the fields of interest specified in the list
    """
    epochs = epoching_funcs.load_epochs_items(subject,cleaned=clean)
    if fields_of_interest is not None:
        for field in fields_of_interest:
            epochs = epochs[np.where(1 - np.isnan(epochs.metadata[field].values))[0]]
            print("--- removing the epochs that have Nan values for field %s ----\n"%field)

    if config.noEEG:
        epochs = epochs.pick_types(meg=True, eeg=False)
    else:
        epochs = epochs.pick_types(meg=True, eeg=True)

    return epochs


def filter_string_for_metadata():
    """
    function that generates a dictionnary for conveniant selection of type of epochs
    """

    filters = dict()
    filters['Stand'] = 'TrialNumber > 10 and ViolationInSequence == 0 and StimPosition > 1'
    filters['Viol'] = 'ViolationOrNot == 1'
    filters['StandMultiStructure'] = 'ViolationInSequence == 0 and StimPosition > 1'
    filters['Hab'] = 'TrialNumber <= 10 and StimPosition > 1'

    for keyname in filters.keys():
        filters[keyname+'_excluseRA'] = filters[keyname] + + ' and SequenceID >= 3'

    return filters