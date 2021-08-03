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


# ========= WHICH PREDICTORS TO CHOOSE FOR THE REGRESSION ======
diss_matrix, md, dis = rsa_funcs.Predictor_dissimilarity_matrix_and_md(analysis_name)

#  --- Visualize the predictor matrices ---
def viz_predictor_mats(dis_pred,md, max_val=None):
    dis_pred_field = rsa_funcs.gen_predicted_dissimilarity(dis_pred, md)
    if max_val is None:
        max_val = np.max(dis_pred_field.data)
    umne.rsa.plot_dissimilarity(dis_pred_field,max_value=max_val,
                                get_label=lambda md: md['SequenceID'])
    # tick_filter=lambda md: md['StimPosition'] == 1
save_regressors_path = config.result_path+"/rsa/dissim/"+analysis_name+'/regressors_matrix/'
utils.create_folder(save_regressors_path)

for key in diss_matrix.keys():
    viz_predictor_mats(eval('dis.'+key), md)
    plt.gcf().savefig(save_regressors_path+key+'.png')
    plt.close('all')

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
fig.savefig(config.result_path+'/rsa/dissim/'+analysis_name+'/correlations/correlation_regressors.png')

# ______________________________________________________________________________________________________________________

for metric_type in ["spearmanr","euclidean","mahalanobis"]:
    print("==== Running the analysis for the metric %s ===="%metric_type)
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
for metric_type in ["spearmanr", "euclidean", "mahalanobis"]:

    diss_matrix, md, dis, times = rsa_funcs.Predictor_dissimilarity_matrix_and_md(analysis_name)
    # dis = rsa_funcs.dissimilarity
    # reg_dis = umne.rsa.load_and_regress_dissimilarity(
    #     config.result_path+"/rsa/dissim/"+analysis_name+"/"+metric_type+"*",
    #     [dis.Complexity,dis.OrdinalPos,dis.repeatalter,dis.ChunkBeg,dis.ChunkEnd,dis.ChunkNumber,dis.ChunkDepth,dis.NOpenChunks,dis.NClosedChunks],
    #     included_cells_getter=None,filename_subj_id_pattern='.*_(\\w+).*.dmat')

    path_save_reg = config.result_path+'/rsa/dissim/'+analysis_name+'/regression_results/'
    # utils.create_folder(path_save_reg)
    # np.save(path_save_reg+metric_type+'_reg.npy',reg_dis)
    reg_dis = np.load(path_save_reg+metric_type+'_reg.npy',allow_pickle=True)
    # ---- plot the regression coefficients separately ------
    names = ('Complexity','SequenceID','OrdinalPos','repeatalter','ChunkBeg', 'ChunkEnd', 'ChunkNumber', 'ChunkDepth','NOpenChunks','NClosedChunks')
    for ii, name in enumerate(names):
        plot_path = path_save_reg + '/plots/'
        utils.create_folder(plot_path)
        fig = umne.rsa.plot_regression_results(reg_dis[0][:, :, ii, np.newaxis], times)
        fig.savefig(plot_path+metric_type+'_'+name + '.png')
        plt.close('all')

