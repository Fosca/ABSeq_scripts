"""
==========================================================
Decoding of the Standard VS Deviants with SVM per sequence
==========================================================
# DESCRIPTION OF THE ANALYSIS
# 1 - Which trials : no cleaning or with autoreject Global ?
# 2 - Sliding window size 100 ms every XX ms ?
# 3 - Excluded participants (with no cleaning)?
# 4 - If we use Savgol filter to plot the data, say we do so

"""
# ---- import the packages -------
import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths

from ABseq_func import article_plotting_funcs

import matplotlib.pyplot as plt
import config
import os.path as op
import numpy as np
import matplotlib
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

#  ============== ============== ============== ============== ============== ============== ============== ============
#                         1 -  SET the params and the subjects
#  ============== ============== ============== ============== ============== ============== ============== ============

subjects_list = config.subjects_list
sensors = ['mag','grad']
n_subjects = len(config.subjects_list)
name = "SW_train_different_blocks_cleanedGAT_results"


for suffix in ["_hab","_stand","_viol",""]:
    filename = name+suffix
    #  ============== ============== ============== ============== ============== ============== ============== ============
    #                         2 -  LOAD THE DATA AND RESHAPE IT
    #  ============== ============== ============== ============== ============== ============== ============== ============
    results = {sens: {'SeqID_%i' % i: [] for i in range(1, 8)} for sens in sensors}
    significance = {sens: {'SeqID_%i' % i: [] for i in range(1, 8)} for sens in sensors}
    avg_res = {sens: [] for sens in sensors}
    for sens in sensors:
        n_subj = 0
        for subject in subjects_list:
            GAT_path = op.join(config.SVM_path, subject, filename + '.npy')
            print("---- loading data for subject %s ----- "%subject)
            if op.exists(GAT_path):
                GAT_results = np.load(GAT_path, allow_pickle=True).item()
                times = 1000*GAT_results['times']
                GAT_results = GAT_results['GAT']
                for key in ['SeqID_%i' % i for i in range(1, 8)]:
                    results[sens][key].append(GAT_results[sens][key])
                n_subj +=1
            else:
                print("Missing data for %s "%GAT_path)

    reshaped_data = {sens : np.zeros((7,n_subj,len(times))) for sens in sensors}
    plt.close('all')

    #  ============== ============== ============== ============== ============== ============== ============== ============
    #                         3 -  Plot and compute the pearson correlation
    #  ============== ============== ============== ============== ============== ============== ============== ============

    for sens in sensors:
        print(sens)
        perform_seq = results[sens]
        for ii,SeqID in enumerate(range(1, 8)):
            perform_seqID = np.asarray(perform_seq['SeqID_' + str(SeqID)])
            diago_seq = np.diagonal(perform_seqID,axis1=1,axis2=2)
            reshaped_data[sens][ii,:,:] = diago_seq
        article_plotting_funcs.plot_7seq_timecourses(reshaped_data[sens],times, save_fig_path='SVM/standard_vs_deviant/',fig_name='All_sequences_standard_VS_deviant_cleaned_', suffix= suffix+sens,
                                  pos_horizontal_bar = 0.47,plot_pearson_corrComplexity=True,chance=0.5)

        pearson = article_plotting_funcs.compute_corr_comp(reshaped_data[sens])
        article_plotting_funcs.heatmap_avg_subj(pearson, times, xlims=None, ylims=[-.5, .5], filter=False, fig_name=config.fig_path+'/SVM/standard_vs_deviant/heatmap_complexity_pearson_'+suffix+sens+'.svg',
                         figsize=(10 * 0.8, 1))
