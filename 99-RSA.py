import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
from initialization_paths import initialization_paths

from ABseq_func import rsa_funcs
import config
import umne
import umne.src
from umne.scr import umne

import numpy as np

# ______________________________________________________________________________________________________________________
# compute the dissimilarity matrix from the behavioral data
subject = config.subjects_list[0]
rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'spearmanr', baseline=None,
                                               which_analysis='SeqID_StimPos')
# ______________________________________________________________________________________________________________________