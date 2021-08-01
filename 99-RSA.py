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
dissim_pearson = rsa_funcs.load_and_avg_dissimilarity_matrices(config.result_path + "rsa/dissim/SequenceID_StimPosition_no_baseline/spearmanr*")



umne.rsaplot.video_dissim(cc, reordering='block_type_primitive_pair', which_labels='primitive', tmin=-0.4, tmax=1,
                          save_name='/Users/fosca/Desktop/videos/11prim_seq_prim')


# ====== now run the RSA regression analysis ==========

dis = rsa_funcs.dissimilarity
reg_dis = umne.rsa.load_and_regress_dissimilarity(
    config.result_path+'/rsa/dissim/SeqID_StimPosSequenceID_StimPosition_no_baseline/*',
    [dis.stim_ID,dis.Complexity],
    included_cells_getter=None)
