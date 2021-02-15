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
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=True, AR_type='local')

    # -- PLOTS -- #
    evoked_funcs.plot_butterfly_items(epochs, subject, violation_or_not=1, apply_baseline=True)  # items epoch are not already baseline corrected
    evoked_funcs.plot_butterfly_items(epochs, subject, violation_or_not=0, apply_baseline=True)
    GFP_funcs.plot_gfp_items_standard_or_deviants(epochs, subject, h_freq=None, standard_or_deviant='standard')
    GFP_funcs.plot_gfp_items_standard_or_deviants(epochs, subject, h_freq=None, standard_or_deviant='deviant')

    # ----------------------------------------------------------------------------------------------------------- #
    # PLOTS - Full-sequence epochs
    # ----------------------------------------------------------------------------------------------------------- #

    # -- LOAD THE EPOCHS -- #
    epochs = epoching_funcs.load_epochs_full_sequence(subject, cleaned=True, AR_type='local')

    # -- PLOTS -- #
    evoked_funcs.plot_butterfly_first_item(epochs, subject, apply_baseline=False)  # fullseq epoch are already baseline corrected
    GFP_funcs.plot_gfp_full_sequence_standard(epochs, subject, h_freq=None)
    GFP_funcs.plot_gfp_full_sequence_deviants_4pos(epochs, subject, h_freq=None)

parallel, run_func, _ = parallel_func(make_figures, n_jobs=config.N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
