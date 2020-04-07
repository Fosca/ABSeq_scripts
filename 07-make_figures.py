import config
from mne.parallel import parallel_func
from ABseq_func import *
import os.path as op

# config.subjects_list = ['sub18-eo_190576', 'sub19-mg_190180']

def make_figures(subject):

    # ----------------------------------------------------------------------------------------------------------- #
    # PLOTS - Items epochs
    # ----------------------------------------------------------------------------------------------------------- #

    # -- LOAD THE EPOCHS -- #
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)

    # -- PLOTS -- #
    evoked_funcs.plot_butterfly_items(epochs, subject, violation_or_not=1)
    evoked_funcs.plot_butterfly_items(epochs, subject, violation_or_not=0)
    GFP_funcs.plot_gfp_items_standard(epochs, subject, h_freq=20)
    GFP_funcs.plot_gfp_items_deviant(epochs, subject, h_freq=20)

    # ----------------------------------------------------------------------------------------------------------- #
    # PLOTS - Full-sequence epochs
    # ----------------------------------------------------------------------------------------------------------- #

    # -- LOAD THE EPOCHS -- #
    epochs = epoching_funcs.load_epochs_full_sequence(subject, cleaned=True)

    # -- PLOTS -- #
    evoked_funcs.plot_butterfly_first_item(epochs, subject)
    GFP_funcs.plot_gfp_full_sequence_standard(epochs, subject, h_freq=10)
    GFP_funcs.plot_gfp_full_sequence_deviants_4pos(epochs, subject, h_freq=10)

    # ----------------------------------------------------------------------------------------------------------- #
    # TEST BASELINE
    # ----------------------------------------------------------------------------------------------------------- #
    evoked_all_1item_stand = []
    evoked_all_1item_dev = []
    for i, subj in enumerate(config.subjects_list):
        epochs = epoching_funcs.load_epochs_items(subj, cleaned=True)

        epochs_standard = epochs['ViolationInSequence == "0"']
        epochs_deviant = epochs['ViolationInSequence > 0']

        evoked_all_1item_stand.append(epochs_standard.apply_baseline(baseline=(-0.1, 0)).average())
        evoked_all_1item_dev.append(epochs_deviant.apply_baseline(baseline=(-0.1, 0)).average())

    evoked_all_1item_stand_grdavg = mne.grand_average(evoked_all_1item_stand)
    evoked_all_1item_dev_grdavg = mne.grand_average(evoked_all_1item_dev)
    evoked_all_1item_stand_grdavg.plot_joint()
    evoked_all_1item_dev_grdavg.plot_joint()



parallel, run_func, _ = parallel_func(make_figures, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
