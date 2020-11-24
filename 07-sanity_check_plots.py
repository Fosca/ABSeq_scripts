import config
from mne.parallel import parallel_func
from ABseq_func import *
import mne
import os.path as op


def make_figures(subject):
    # ----------------------------------------------------------------------------------------------------------- #
    # PLOTS - Items epochs
    # ----------------------------------------------------------------------------------------------------------- #

    # -- LOAD THE EPOCHS -- #
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)

    # -- PLOTS -- #
    evoked_funcs.plot_butterfly_items(epochs, subject, violation_or_not=1, apply_baseline=True)
    evoked_funcs.plot_butterfly_items(epochs, subject, violation_or_not=0, apply_baseline=True)
    GFP_funcs.plot_gfp_items_standard_or_deviants(epochs, subject, h_freq=30, standard_or_deviant='standard')
    GFP_funcs.plot_gfp_items_standard_or_deviants(epochs, subject, h_freq=30, standard_or_deviant='deviant')

    # ----------------------------------------------------------------------------------------------------------- #
    # PLOTS - Full-sequence epochs
    # ----------------------------------------------------------------------------------------------------------- #

    # -- LOAD THE EPOCHS -- #
    epochs = epoching_funcs.load_epochs_full_sequence(subject, cleaned=True)

    # -- PLOTS -- #
    evoked_funcs.plot_butterfly_first_item(epochs, subject, apply_baseline=True)
    GFP_funcs.plot_gfp_full_sequence_standard(epochs, subject, h_freq=20)
    GFP_funcs.plot_gfp_full_sequence_deviants_4pos(epochs, subject, h_freq=20)

    # # ----------------------------------------------------------------------------------------------------------- #
    # # TEST BASELINE
    # # ----------------------------------------------------------------------------------------------------------- #
    # evoked_all_1item_stand = []
    # evoked_all_1item_dev = []
    # for i, subj in enumerate(config.subjects_list):
    #     epochs = epoching_funcs.load_epochs_items(subj, cleaned=True)
    #
    #     epochs_standard = epochs['ViolationInSequence == "0"']
    #     epochs_deviant = epochs['ViolationInSequence > 0']
    #
    #     evoked_all_1item_stand.append(epochs_standard.apply_baseline(baseline=(-0.1, 0)).average())
    #     evoked_all_1item_dev.append(epochs_deviant.apply_baseline(baseline=(-0.1, 0)).average())
    #
    # evoked_all_1item_stand_grdavg = mne.grand_average(evoked_all_1item_stand)
    # evoked_all_1item_dev_grdavg = mne.grand_average(evoked_all_1item_dev)
    # evoked_all_1item_stand_grdavg.plot_joint()
    # evoked_all_1item_dev_grdavg.plot_joint()


parallel, run_func, _ = parallel_func(make_figures, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
