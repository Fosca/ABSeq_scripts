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

for metric_type in ["spearmanr","euclidean","mahalanobis"]:

    dissim_metric = rsa_funcs.load_and_avg_dissimilarity_matrices(config.result_path + "rsa/dissim/SequenceID_StimPosition_no_baseline/"+metric_type+"*")
    data = dissim_metric.data

    utils.create_folder(config.result_path + "rsa/dissim/SequenceID_StimPosition_no_baseline/"+metric_type+"/")
    times = dissim_metric.times
    for tt in range(dissim_metric.n_timepoints):
         print(tt)
         diss = dissim_metric.copy()
         diss.data = diss.data[tt]
         rsa_funcs.plot_dissimilarity(diss,vmin = np.mean(dissim_metric.data)-2*np.std(dissim_metric.data),vmax = np.mean(dissim_metric.data)+2*np.std(dissim_metric.data))
         plt.gcf().set_title('Time %s'%str(times[tt]))
         plt.gcf().savefig(config.result_path + "rsa/dissim/SequenceID_StimPosition_no_baseline/"+metric_type+"/"+str(times[tt])+".png")
         plt.close("all")

# ====== now run the RSA regression analysis ==========

dis = rsa_funcs.dissimilarity
reg_dis = umne.rsa.load_and_regress_dissimilarity(
    config.result_path+'/rsa/dissim/SequenceID_StimPosition_no_baseline/spearman*',
    [dis.stim_ID,dis.Complexity],
    included_cells_getter=None,filename_subj_id_pattern='.*_(\\w+).*.dmat')

# ====== Determine which regressors are too correlated =====
md = dissim_metric.md1
diss_matrix = []

dissim_block_type = rsa_funcs.gen_predicted_dissimilarity(dis.stim_ID,md = md)
dissim_prim = rsa_funcs.gen_predicted_dissimilarity(dis.Complexity,md = md)
dissim_prim_diff_blocks = rsa_funcs.gen_predicted_dissimilarity(dis.SequenceID,md = md)
dissim_samesecond = rsa_funcs.gen_predicted_dissimilarity(dis.OrdinalPos,md = md)
dissim_samesecond = rsa_funcs.gen_predicted_dissimilarity(dis.repeatalter,md = md)
dissim_rotorsym = rsa_funcs.gen_predicted_dissimilarity(dis.ChunkBeg,md = md)
dissim_rotorsym_diff_blocks = rsa_funcs.gen_predicted_dissimilarity(dis.ChunkEnd,md = md)
dissim_distance = rsa_funcs.gen_predicted_dissimilarity(dis.ChunkNumber,md = md)
dissim_samefirst = rsa_funcs.gen_predicted_dissimilarity(dis.ChunkDepth,md = md)
dissim_samesecond = rsa_funcs.gen_predicted_dissimilarity(dis.NOpenChunks,md = md)
dissim_samesecond = rsa_funcs.gen_predicted_dissimilarity(dis.NClosedChunks,md = md)



dissim_matrix = [dissim_block_type,dissim_prim,dissim_prim_diff_blocks,dissim_rotorsym,dissim_rotorsym_diff_blocks,dissim_distance,dissim_samefirst,dissim_samesecond]
correlation_matrix = np.zeros((8,8))

for k in range(8):
    for l in range(8):
        r = np.corrcoef([np.reshape(dissim_matrix[k].data, dissim_matrix[k].data.size),
                         np.reshape(dissim_matrix[l].data, dissim_matrix[l].data.size)])
        correlation_matrix[k,l]=r[0,1]

plt.imshow(correlation_matrix, cmap=cm.viridis)
plt.colorbar()
plt.title('Correlation across predictors')
plt.xticks(range(8),names,rotation=30)
plt.yticks(range(8),names,rotation=30)

fig = plt.gcf()
fig.savefig('correlation_regressors.png')


