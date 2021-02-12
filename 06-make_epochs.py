"""
====================
06. Construct epochs
====================

The epochs are constructed by using the events created in script 03. MNE
supports hierarchical events that allows selection to different groups more
easily (see config.event_id). Automatic rejection is applied to the epochs (or not).
Finally the epochs are saved to disk.
To save space, the epoch data can be decimated.
"""

import config
from mne.parallel import parallel_func
from ABseq_func import epoching_funcs
from ABseq_func import autoreject_funcs

# make less parallel runs to limit memory usage
N_JOBS = max(config.N_JOBS // 3, 1)
# N_JOBS = config.N_JOBS

# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(epoching_funcs.run_epochs, n_jobs=N_JOBS)

epoch_on_first_element = True
parallel(run_func(subject, epoch_on_first_element, baseline=True) for subject in config.subjects_list)

epoch_on_first_element = False
parallel(run_func(subject, epoch_on_first_element, baseline=False) for subject in config.subjects_list)


# # ______________________________________________________________________________________
#
# # ====== RUN THE AUTOREJECT ALGORITHM TO INTERPOLATE OR REMOVE THE BAD EPOCHS ==========  /// NOW INCLUDED IN epoching_funcs.run_epochs
# # ______________________________________________________________________________________
#
#
# # AutoReject function parallel
# parallel, run_func, _ = parallel_func(autoreject_funcs.run_autoreject, n_jobs=N_JOBS)
#
# # Run the AutoReject function on "full_sequence" epochs
# epoch_on_first_element = True
# parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)
#
# # Run the AutoReject function on "items" epochs
# epoch_on_first_element = False
# parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)
#
# # AutoReject plot function parallel
# parallel, run_func, _ = parallel_func(autoreject_funcs.ar_log_summary, n_jobs=config.N_JOBS)
# epoch_on_first_element = False
# parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)
# epoch_on_first_element = True
# parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)

