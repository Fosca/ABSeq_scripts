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
cm = plt.get_cmap('viridis')

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


dis = rsa_funcs.dissimilarity
dissim_mat = np.load("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/results/rsa/dissim/SequenceID_StimPosition_no_baseline/spearmanr_sub01-pa_190002.dmat",allow_pickle=True)
md = dissim_mat.md1
diss_matrix = dict()


diss_matrix['stim_ID'] = rsa_funcs.gen_predicted_dissimilarity(dis.stim_ID,md = md)
diss_matrix['Complexity'] = rsa_funcs.gen_predicted_dissimilarity(dis.Complexity,md = md)
diss_matrix['SequenceID'] = rsa_funcs.gen_predicted_dissimilarity(dis.SequenceID,md = md)
diss_matrix['OrdinalPos'] = rsa_funcs.gen_predicted_dissimilarity(dis.OrdinalPos,md = md)
diss_matrix['repeatalter'] = rsa_funcs.gen_predicted_dissimilarity(dis.repeatalter,md = md)
diss_matrix['ChunkBeg'] = rsa_funcs.gen_predicted_dissimilarity(dis.ChunkBeg,md = md)
diss_matrix['ChunkEnd'] = rsa_funcs.gen_predicted_dissimilarity(dis.ChunkEnd,md = md)
diss_matrix['ChunkNumber'] = rsa_funcs.gen_predicted_dissimilarity(dis.ChunkNumber,md = md)
diss_matrix['ChunkDepth'] = rsa_funcs.gen_predicted_dissimilarity(dis.ChunkDepth,md = md)
diss_matrix['NOpenChunks'] = rsa_funcs.gen_predicted_dissimilarity(dis.NOpenChunks,md = md)
diss_matrix['NClosedChunks'] = rsa_funcs.gen_predicted_dissimilarity(dis.NClosedChunks,md = md)

correlation_matrix = np.zeros((len(diss_matrix.keys()),len(diss_matrix.keys())))

for k, key1 in enumerate(diss_matrix.keys()):
    for l, key2 in enumerate(diss_matrix.keys()):
        r = np.corrcoef([np.reshape(diss_matrix[key1].data, diss_matrix[key1].data.size),
                         np.reshape(diss_matrix[key2].data, diss_matrix[key2].data.size)])
        correlation_matrix[k,l]=r[0,1]

plt.imshow(correlation_matrix, cmap=cm.viridis)
plt.colorbar()
plt.title('Correlation across predictors')
plt.xticks(range(len(diss_matrix.keys())),diss_matrix.keys(),rotation=30)
plt.yticks(range(len(diss_matrix.keys())),diss_matrix.keys(),rotation=30)

fig = plt.gcf()
fig.savefig(config.result_path+'/rsa/dissim/correlation_regressors.png')


