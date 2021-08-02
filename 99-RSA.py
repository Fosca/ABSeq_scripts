import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/umne/')
from initialization_paths import initialization_paths
from ABseq_func import rsa_funcs, epoching_funcs, utils
import config
import numpy as np
from src import umne
from src.umne import rsaplot
import pickle
import matplotlib.pyplot as plt
# ______________________________________________________________________________________________________________________
# compute the dissimilarity matrix from the behavioral data
# for subject in config.subjects_list:
#     rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'spearmanr', baseline=None,
#                                                    which_analysis='')
# ______________________________________________________________________________________________________________________

dissim_metric = rsa_funcs.load_and_avg_dissimilarity_matrices(config.result_path + "rsa/dissim/SequenceID_StimPosition_no_baseline/"+analysis_type+"*")
data = dissim_metric.data

utils.create_folder(config.result_path + "rsa/dissim/SequenceID_StimPosition_no_baseline/"+analysis_type+"/")
times = dissim_metric.times
for tt in range(dissim_metric.n_timepoints):
     print(tt)
     diss = dissim_metric.copy()
     diss.data = diss.data[tt]
     rsa_funcs.plot_dissimilarity(diss,vmin = np.mean(dissim_metric.data)-2*np.std(dissim_metric.data),vmax = np.mean(dissim_metric.data)+2*np.std(dissim_metric.data))
     plt.gcf().set_title('Time %s'%str(times[tt]))
     plt.gcf().savefig(config.result_path + "rsa/dissim/SequenceID_StimPosition_no_baseline/"+analysis_type+"/"+str(times[tt])+".png")
     plt.close("all")

# ====== now run the RSA regression analysis ==========

dis = rsa_funcs.dissimilarity
reg_dis = umne.rsa.load_and_regress_dissimilarity(
    config.result_path+'/rsa/dissim/SequenceID_StimPosition_no_baseline/spearman*',
    [dis.stim_ID,dis.Complexity],
    included_cells_getter=None,filename_subj_id_pattern='.*_(\\w+).*.dmat')


filename_mask = config.result_path+'/rsa/dissim/SequenceID_StimPosition_no_baseline/spearman*'
gen_predictor_funcs = [dis.stim_ID,dis.Complexity]
filename_subj_id_pattern='.*_(\\w+).*.dmat'
cell_filter=None
zscore_predictors=True
included_cells_getter=None
excluded_cells_getter=None