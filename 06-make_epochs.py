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

import os.path as op
import mne
import pandas as pd
import config
import numpy as np
import pickle

from mne.parallel import parallel_func
from ABseq_func import epoching_funcs
from autoreject import AutoReject

# make less parallel runs to limit memory usage
# N_JOBS = max(config.N_JOBS // 4, 1)
N_JOBS = config.N_JOBS

# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(epoching_funcs.run_epochs, n_jobs=N_JOBS)

epoch_on_first_element = True
parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)

epoch_on_first_element = False
parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)
