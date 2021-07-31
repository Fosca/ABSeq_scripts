from ABseq_func import rsa_funcs
import config
import umne.scr.umne as umne
import numpy as np

# ______________________________________________________________________________________________________________________
# compute the dissimilarity matrix from the behavioral data
subject = config.subjects_list[0]
rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'spearmanr', baseline=None,
                                               which_analysis='primitives_and_sequences',
                                               factors_or_interest=('primitive', 'position_pair', 'block_type'))
# ______________________________________________________________________________________________________________________