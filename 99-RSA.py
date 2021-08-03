import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/umne/')
from initialization_paths import initialization_paths
from ABseq_func import rsa_funcs, epoching_funcs, utils,cluster_funcs
import config
import numpy as np
from src import umne
import matplotlib.pyplot as plt
cm = plt.get_cmap('viridis')

# # ______________________________________________________________________________________________________________________
# compute the dissimilarity matrix from the behavioral data
# for subject in config.subjects_list:
#     rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'spearmanr', baseline=None,
#                                                    which_analysis='')
# ______________________________________________________________________________________________________________________
analysis_name = "SequenceID_StimPosition_Complexity_RepeatAlter_ChunkBeginning_ChunkEnd_OpenedChunks_ChunkDepth_ChunkNumber_WithinChunkPosition_ClosedChunks_no_baseline"

for metric_type in ["spearmanr","euclidean","mahalanobis"]:

    dissim_metric = rsa_funcs.load_and_avg_dissimilarity_matrices(config.result_path + "rsa/dissim/"+analysis_name+"/"+metric_type+"*.dmat")
    data = dissim_metric.data

    utils.create_folder(config.result_path + "rsa/dissim/"+analysis_name+"/"+metric_type+"/")
    times = dissim_metric.times
    for tt in range(dissim_metric.n_timepoints):
         print(tt)
         diss = dissim_metric.copy()
         diss.data = diss.data[tt]
         rsa_funcs.plot_dissimilarity(diss,vmin = np.mean(dissim_metric.data)-2*np.std(dissim_metric.data),vmax = np.mean(dissim_metric.data)+2*np.std(dissim_metric.data))
         plt.suptitle('Time %s'%str(times[tt]))
         plt.gcf().savefig(config.result_path + "rsa/dissim/"+analysis_name+"/"+metric_type+"/"+str(times[tt])+".png")
         plt.close("all")

# ====== now run the RSA regression analysis ==========

    dis = rsa_funcs.dissimilarity
    reg_dis = umne.rsa.load_and_regress_dissimilarity(
        config.result_path+"/rsa/dissim/"+analysis_name+"/"+metric_type+"*",
        [dis.Complexity,dis.OrdinalPos,dis.repeatalter,dis.ChunkBeg,dis.ChunkEnd,dis.ChunkNumber,dis.ChunkDepth,dis.NOpenChunks,dis.NClosedChunks],
        included_cells_getter=None,filename_subj_id_pattern='.*_(\\w+).*.dmat')

    path_save_reg = config.result_path+'/rsa/dissim/'+analysis_name+'/regression_results/'
    utils.create_folder(path_save_reg)
    np.save(path_save_reg+metric_type+'_reg.npy',reg_dis)
    # reg_dis = np.load(path_save_reg+'spearman_reg.npy',allow_pickle=True)
    # ---- plot the regression coefficients separately ------
    names = ('Complexity','SequenceID','OrdinalPos','repeatalter','ChunkBeg', 'ChunkEnd', 'ChunkNumber', 'ChunkDepth','NOpenChunks','NClosedChunks')
    for ii, name in enumerate(names):
        fig = umne.rsa.plot_regression_results(reg_dis[0][:, :, ii, np.newaxis], times)
        fig.savefig(path_save_reg+name + '.png')
        plt.close('all')


# ========= WHICH PREDICTORS TO CHOOSE FOR THE REGRESSION ======

dis = rsa_funcs.dissimilarity
dissim_mat = np.load("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/results/rsa/dissim/"+analysis_name+"/spearmanr_sub01-pa_190002.dmat",allow_pickle=True)
md = dissim_mat.md1
diss_matrix = dict()

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

#  --- Visualize the predictor matrices ---
def viz_predictor_mats(dis_pred,md, max_val=1):
    umne.rsa.plot_dissimilarity(rsa_funcs.gen_predicted_dissimilarity(dis_pred, md),
                                get_label=lambda md: md['SequenceID'],max_value=max_val)

viz_predictor_mats(dis.Complexity,md,max_val=1)

# --- Determine which regressors are too correlated ---







correlation_matrix = np.zeros((len(diss_matrix.keys()),len(diss_matrix.keys())))

for k, key1 in enumerate(diss_matrix.keys()):
    for l, key2 in enumerate(diss_matrix.keys()):
        r = np.corrcoef([np.reshape(diss_matrix[key1].data, diss_matrix[key1].data.size),
                         np.reshape(diss_matrix[key2].data, diss_matrix[key2].data.size)])
        correlation_matrix[k,l]=r[0,1]

cm = plt.get_cmap('viridis')
plt.imshow(correlation_matrix)
plt.colorbar()
plt.title('Correlation across predictors')
plt.xticks(range(len(diss_matrix.keys())),diss_matrix.keys(),rotation=30)
plt.yticks(range(len(diss_matrix.keys())),diss_matrix.keys(),rotation=30)

fig = plt.gcf()
fig.savefig(config.result_path+'/rsa/dissim/correlation/correlation_regressors.png')

plt.close('all')
plt.imshow(1*(correlation_matrix<-0.5))
plt.colorbar()
plt.title('Correlation across predictors')
plt.xticks(range(len(diss_matrix.keys())),diss_matrix.keys(),rotation=30)
plt.yticks(range(len(diss_matrix.keys())),diss_matrix.keys(),rotation=30)

fig = plt.gcf()
fig.savefig(config.result_path+'/rsa/dissim/correlation_regressors_below_-05.png')

