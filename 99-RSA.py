import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/umne/')
from initialization_paths import initialization_paths

from ABseq_func import rsa_funcs
import config
import numpy as np
from src import umne

# ______________________________________________________________________________________________________________________
# compute the dissimilarity matrix from the behavioral data
for subject in config.subjects_list:
    rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'spearmanr', baseline=None,
                                                   which_analysis='')
# ______________________________________________________________________________________________________________________

# ====== now run the RSA regression analysis ==========

dis = rsa_funcs.dissimilarity
reg_dis = umne.rsa.load_and_regress_dissimilarity(
    config.result_path+'/rsa/dissim/SeqID_StimPosSequenceID_StimPosition_no_baseline/*',
    [dis.stim_ID,dis.Complexity],
    included_cells_getter=None)
