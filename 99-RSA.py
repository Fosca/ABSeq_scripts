import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/umne/')
from initialization_paths import initialization_paths
from ABseq_func import rsa_funcs, epoching_funcs, utils, cluster_funcs
import config
import numpy as np
from src import umne
import matplotlib.pyplot as plt
cm = plt.get_cmap('viridis')
from ABseq_func import rsa_funcs
dis = rsa_funcs.dissimilarity
# # ______________________________________________________________________________________________________________________
# compute the dissimilarity matrix from the behavioral data
# for subject in config.subjects_list:
#      rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'spearmanr', baseline=None,
#                                                    which_analysis='')
# ______________________________________________________________________________________________________________________
# _____________________________________________________________________________________________________________________
def viz_predictor_mats(dis_pred, md,md2, max_val=None):
    dis_pred_field = rsa_funcs.gen_predicted_dissimilarity(dis_pred, md,md2=md2)
    if max_val is None:
        max_val = np.max(dis_pred_field.data)
    umne.rsa.plot_dissimilarity(dis_pred_field, max_value=max_val,
                                get_label=lambda md: md['SequenceID'])

# _____________________________________________________________________________________________________________________
def plot_regression_results(regression_results, times, figure_id=1, alpha=0.15, legend=None, reset=True,
                            save_as=None):
    """
    Plot the RSA regression results
    :param regression_results: #Subjects x #TimePoints x #Predictors matrix
    :param times: x axis values
    :param figure_id:
    :param colors: Array, one color for each predictor
    :param alpha: For the shaded part that represents 1 standard error
    :param pred_mult_factor: Multiple each predictor by this factor (array, size = # of predictors)
    :param subject_x_factor: The x values of each subject are stretched by this factor.
    :param legend: Legend names (array, size = # of predcitors)
    :param reset: If True (default), the figure is cleared before plotting
    """
    import math
    color_map = plt.get_cmap('twilight')

    n_subj, n_tp, n_pred = regression_results.shape
    inds = np.linspace(0, color_map.N, n_pred + 1)
    colors = [color_map.colors[int(i) - 1] for i in inds]

    if legend is not None:
        assert len(legend) == n_pred, "There are {} legend entries but {} predictors".format(len(legend), n_pred)
    if reset:
        plt.close(figure_id)
    fig = plt.figure(figure_id, figsize=(20, 12))
    ax = plt.gca()
    plt.plot(times, [0] * len(times), color='black', label='_nolegend_')
    for ipred in range(n_pred):
        curr_predictor_rr = regression_results[:, :, ipred]
        coeffs = np.mean(curr_predictor_rr, axis=0)
        coeffs_se = np.std(curr_predictor_rr, axis=0) / math.sqrt(n_subj)
        # -- plot
        plt.fill_between(times, coeffs - coeffs_se, coeffs + coeffs_se, color=colors[ipred], alpha=alpha,
                         label='_nolegend_')
        plt.plot(times, coeffs, color=colors[ipred])
    if legend is not None:
        plt.legend(legend, loc="upper left")
    if save_as is not None:
        fig = plt.gcf()
        fig.savefig(save_as)
    return fig

# _____________________________________________________________________________________________________________________
# _____________________________________________________________________________________________________________________
# _____________________________________________________________________________________________________________________

analysis_name = "SequenceID_StimPosition_Complexity_RepeatAlter_ChunkBeginning_ChunkEnd_OpenedChunks_ChunkDepth_ChunkNumber_WithinChunkPosition_ClosedChunks_no_baseline"
analysis_name = "StimID_SequenceID_StimPosition_Complexity_RepeatAlter_ChunkBeginning_ChunkEnd_OpenedChunks_ChunkDepth_ChunkNumber_WithinChunkPosition_ClosedChunks_no_baseline"
analysis_name = "_no_baseline_all_dataStimID_SequenceID_StimPosition_Complexity_RepeatAlter_ChunkBeginning_ChunkEnd_OpenedChunks_ChunkDepth_ChunkNumber_WithinChunkPosition_ClosedChunks_no_baseline"

# ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======
#                                            LOOKING AT PREDICTORS
# ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======

diss_matrix, md,md2, dis, times = rsa_funcs.Predictor_dissimilarity_matrix_and_md(analysis_name)

#  --- Visualize the predictor matrices ---

    # tick_filter=lambda md: md['StimPosition'] == 1
save_regressors_path = config.result_path+"/rsa/dissim/"+analysis_name+'/regressors_matrix/'
utils.create_folder(save_regressors_path)

for key in diss_matrix.keys():
    viz_predictor_mats(eval('dis.'+key), md,md2=md2)
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

# ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======
#                             VISUALIZING THE DISSIMILARITY MATRIX DATA
# ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======

for metric_type in ["correlation"]:
    print("==== Running the analysis for the metric %s ===="%metric_type)
    dissim_metric = rsa_funcs.load_and_avg_dissimilarity_matrices(config.result_path + "rsa/dissim/"+analysis_name+"/"+metric_type+"*.dmat")
    dissim_metric = rsa_funcs.reorder_matrix(dissim_metric, fields=(
    'SequenceID', 'StimPosition', 'Complexity', 'RepeatAlter', 'ChunkBeginning', 'ChunkEnd', 'OpenedChunks',
    'ChunkDepth', 'ChunkNumber', 'WithinChunkPosition', 'ClosedChunks'))
    data = dissim_metric.data
    utils.create_folder(config.result_path + "rsa/dissim/"+analysis_name+"/"+metric_type+"/")
    times = dissim_metric.times
    for tt in range(dissim_metric.n_timepoints):
         print(tt)
         diss = dissim_metric.copy()
         diss.data = diss.data[tt]
         rsa_funcs.plot_dissimilarity(diss,vmin = np.mean(dissim_metric.data)-2*np.std(dissim_metric.data),vmax = np.mean(dissim_metric.data)+2*np.std(dissim_metric.data))
         plt.suptitle('Time %s'%str(int(times[tt]*1000)))
         plt.gcf().savefig(config.result_path + "rsa/dissim/"+analysis_name+"/"+metric_type+"/"+str(int(times[tt]*1000))+".png")
         plt.close("all")

# ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======
#                             RUNNING THE REGRESSION ON THE DISSIMILARITY MATRICES FROM THE DATA
# ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======  ======
# %%% the regressors we consider %%%%
reg_dict = {'stimID':dis.stim_ID ,'SequenceID':dis.SequenceID,'Complexity':dis.Complexity,'OrdinalPos':dis.OrdinalPos,'repeatalter':dis.repeatalter,
            'ChunkBeg':dis.ChunkBeg, 'ChunkEnd':dis.ChunkEnd, 'ChunkNumber':dis.ChunkNumber, 'ChunkDepth':dis.ChunkDepth,
            'NOpenChunks':dis.NOpenChunks}

# reg_dict = {'StimID':dis.stim_ID,'SeqID':dis.SequenceID}
suffix = ''
# metrics = ["euclidean","spearmanr"]
metrics = ["correlation"]
# 1 - 1 - 1 -  PERFORM THE REGRESSION WITH ALL THE REGRESSORS TOGETHER
for metric_type in metrics:
    diss_matrix, md, md2, dis, times = rsa_funcs.Predictor_dissimilarity_matrix_and_md(analysis_name)
    dis = rsa_funcs.dissimilarity
    reg_dis = umne.rsa.load_and_regress_dissimilarity(
        config.result_path+"/rsa/dissim/"+analysis_name+"/"+metric_type+"*",
        [reg_dict[key] for key in reg_dict.keys()],
        included_cells_getter=None,filename_subj_id_pattern='.*_(\\w+).*.dmat')
    path_save_reg = config.result_path+'/rsa/dissim/'+analysis_name+'/regression_results/'
    plot_path = path_save_reg + '/plots/'
    utils.create_folder(path_save_reg)
    utils.create_folder(plot_path)
    np.save(path_save_reg+metric_type+suffix+'_reg.npy',reg_dis)

# AND PLOT
for metric_type in metrics:

    path_save_reg = config.result_path+'/rsa/dissim/'+analysis_name+'/regression_results/'
    plot_path = path_save_reg + '/plots/'
    reg_dis = np.load(path_save_reg+metric_type+suffix+'_reg.npy',allow_pickle=True)
    # ---- plot the regression coefficients separately ------
    fig = plot_regression_results(reg_dis[0][:, :,:-1], times,legend=reg_dict.keys())
    fig.savefig(plot_path+metric_type+suffix+'_all.png')
    for ii, name in enumerate(reg_dict.keys()):
        plt.close('all')
        fig = umne.rsa.plot_regression_results(reg_dis[0][:, :, ii, np.newaxis], times,show_significance=True, significance_time_window=[-0.4,1])
        plt.ylim([-0.06,0.1])
        fig.savefig(plot_path+metric_type+suffix+'_'+name + '.png')


# 2 - 2 - 2 -  PERFORM THE REGRESSION FOR EACH REGRESSOR SEPARATELY

reg_dict = {'maths':dis.Maths_network ,'language':dis.Language_network}

for metric_type in ["correlation"]:
    # diss_matrix, md, md2, dis, times = rsa_funcs.Predictor_dissimilarity_matrix_and_md(analysis_name)
    dis = rsa_funcs.dissimilarity
    for ii , name in enumerate(reg_dict.keys()):
        reg_dis = umne.rsa.load_and_regress_dissimilarity(
            config.result_path+"/rsa/dissim/"+analysis_name+"/"+metric_type+"*",
            [reg_dict[name]],
            included_cells_getter=None,filename_subj_id_pattern='.*_(\\w+).*.dmat')

        path_save_reg = config.result_path+'/rsa/dissim/'+analysis_name+'/regression_results/'
        utils.create_folder(path_save_reg)
        np.save(path_save_reg+metric_type+name+'_reg.npy',reg_dis)

# AND PLOT
for metric_type in ["correlation"]:
    for ii , name in enumerate(reg_dict.keys()):
        path_save_reg = config.result_path+'/rsa/dissim/'+analysis_name+'/regression_results/'
        reg_dis = np.load(path_save_reg + metric_type+name + '_reg.npy', allow_pickle=True)
        plot_path = path_save_reg + '/plots/'
        plt.close('all')
        fig = umne.rsa.plot_regression_results(reg_dis[0][:, :,0, np.newaxis], times,show_significance=True, significance_time_window=[0,0.6])
        fig.savefig(plot_path+metric_type+'_'+name + '_alone.png')


# AND PLOT
for metric_type in ["euclidean","spearmanr"]:
    for ii , name in enumerate(reg_dict.keys()):
        path_save_reg = config.result_path+'/rsa/dissim/'+analysis_name+'/regression_results/'
        reg_dis = np.load(path_save_reg + metric_type+name + '_reg.npy', allow_pickle=True)
        plot_path = path_save_reg + '/plots/'
        plt.close('all')
        fig = umne.rsa.plot_regression_results(reg_dis[0][:, :,0, np.newaxis], times,show_significance=True, significance_time_window=[0,0.6])
        fig.savefig(plot_path+metric_type+'_'+name + '_alone.png')


