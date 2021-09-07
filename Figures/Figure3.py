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

from ABseq_func import *
import matplotlib
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

import matplotlib.pyplot as plt
import config
import numpy as np
from scipy.signal import savgol_filter
import os.path as op
from scipy import stats
from jr.plot import pretty_decod
import numpy as np


filename = "SW_train_different_blocks_cleanedGAT_results"
suffix = "_viol"
filename = filename+suffix


# ---- remind how the data was computed ---
# SVM decoder on standard VS deviants where standards are the ones that happen at the same location
# Training on one block testing on the other (nice because there is invertion of A and B). When there is only one block
# because of data acquisition problems, we split into 2 that block.
# Training on all the sequences together.
# Testing on each sequence separately.


#  ============== ============== ============== ============== ============== ============== ============== ============
#                                       PLOTTING FUNCTION
#  ============== ============== ============== ============== ============== ============== ============== ============

def petit_plot_heatmap_mode(diago_score,times,filter=True,pos_heatmap = 0.38,fig_name=''):

    fig, ax = plt.subplots(1, 1, figsize=(10*0.8, 1))
    # ---- determine the significant time-windows ----
    mean = np.mean(diago_score, axis=0)
    if filter == True:
        mean = savgol_filter(mean, 11, 3)
    extent = [min(times), max(times), pos_heatmap, pos_heatmap+0.03]
    plt.imshow(mean[np.newaxis, :], aspect="auto", cmap="viridis", extent=extent)
    plt.gca().set_yticks([])
    plt.colorbar()
    if fig_name is not None:
        plt.gcf().savefig(fig_name)


def petit_plot(diago_score,times,filter=False,fig_name='',color='b',chance = 0.5, pos_sig = None,plot_shaded_vertical = False):
    """
    Petite fonction qui plot la diagonale du GAT et qui calcule les temps auquels c'est significatif.
    Si pos_sig = None, alors ça plotte en gras les moments où c'est significatif et si pos_sig = 0.5 - XX alors ça plot en une ligne en dessous des courbes
    les moments où c'est significatif.
    """
    plt.gcf()
    # ---- determine the significant time-windows ----
    if chance is not None:
        sig = stats_funcs.stats(diago_score[:, times > 0] - chance)
        # ---- determine the significant times ----
        times_sig = times[times > 0]
        times_sig = times_sig[sig<0.05]
    n_subj = diago_score.shape[0]
    mean = np.mean(diago_score, axis=0)
    ub = (mean + np.std(diago_score, axis=0) / (np.sqrt(n_subj)))
    lb = (mean - np.std(diago_score, axis=0) / (np.sqrt(n_subj)))

    if filter == True:
        mean = savgol_filter(mean, 11, 3)
        ub = savgol_filter(ub, 11, 3)
        lb = savgol_filter(lb, 11, 3)

    if plot_shaded_vertical and len(times_sig)!=0:
        ylims = plt.gca().get_ylim()
        plt.gca().fill_between([times_sig[0],times_sig[-1]],ylims[1], ylims[0], color='black', alpha=.1)
        return True

    if chance is not None:
        sig_mean = mean[times>0]
        sig_mean = sig_mean[sig<0.05]
    plt.fill_between(times, ub, lb, alpha=.2,color=color)
    plt.plot(times, mean, linewidth=1.5,color=color)
    plt.xlabel('Time (ms)')
    plt.ylabel('Performance')
    if chance is not None and pos_sig is None:
        plt.plot(times_sig,sig_mean,linewidth=3,color=color)
    elif chance is not None and pos_sig is not None:
        plt.plot(times_sig,[pos_sig]*len(times_sig), linestyle='-', color=color, linewidth=2)
    if fig_name is not None:
        plt.gcf().savefig(fig_name)


#  ============== ============== ============== ============== ============== ============== ============== ============
#                1 - SET THE PLOTTING PARAMS
#  ============== ============== ============== ============== ============== ============== ============== ============

# ---- set figure parameters ----

filter = True
NUM_COLORS = 7
cm = plt.get_cmap('viridis')
colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
plt.close('all')

subjects_list = config.subjects_list
sensors = ['mag','grad']
n_subjects = len(config.subjects_list)

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
            # print(np.mean(GAT_results[sens]["average_all_sequences"]))
            for key in ['SeqID_%i' % i for i in range(1, 8)]:
                results[sens][key].append(GAT_results[sens][key])
            # avg_res[sens].append(GAT_results[sens]["average_all_sequences"])
            n_subj +=1
        else:
            print("Missing data for %s "%GAT_path)

reshaped_data = {sens : np.zeros((7,n_subj,len(times))) for sens in sensors}

#  ============== ============== ============== ============== ============== ============== ============== ============
#                3 - Correlating performance WITH COMPLEXITY
#  ============== ============== ============== ============== ============== ============== ============== ============

# ------ PART 2 OF THE FIGURE: CORRELATION WITH COMPLEXITY -------
# compute per subject the correlation with complexity

C = [4,6,6,6,12,15,28]
N = [1,2,4,8,8,16,16]

complexity_dict = {1:4,2:6,3:6,4:6,5:12,6:15,7:28}
complexity = list(complexity_dict.values())
pearson_r = {sens : [] for sens in sensors}
spearman_rho = {sens : [] for sens in sensors}

for sens in sensors:
    data_sens = reshaped_data[sens]
    for nn in range(n_subjects):
        # ---- for 1 subject, diagonal of the GAT for all the 7 sequences through time ---
        dd = data_sens[:,nn,:]
        r = []
        rho = []
        # Pearson correlation
        for t in range(len(times)):
            r_t, _ = stats.pearsonr(dd[:,t],complexity)
            rho_t, _ = stats.spearmanr(dd[:,t],complexity)
            r.append(r_t)
            rho.append(rho_t)
        pearson_r[sens].append(r)
        spearman_rho[sens].append(rho)
    pearson_r[sens] = np.asarray(pearson_r[sens])
    spearman_rho[sens] = np.asarray(spearman_rho[sens])

for sens in sensors:
    plt.close('all')
    petit_plot_heatmap_mode(pearson_r[sens],times,fig_name=config.fig_path+'/SVM/standard_vs_deviant/heatmap_complexity_pearson_%s'+suffix+'.png')
    petit_plot_heatmap_mode(pearson_r[sens],times,fig_name=config.fig_path+'/SVM/standard_vs_deviant/heatmap_complexity_pearson_%s'+suffix+'.svg')
    plt.close('all')
    petit_plot_heatmap_mode(spearman_rho[sens],times,fig_name=config.fig_path+'/SVM/standard_vs_deviant/heatmap_complexity_spearman_%s'%sens+suffix+'.png')
    petit_plot_heatmap_mode(spearman_rho[sens],times,fig_name=config.fig_path+'/SVM/standard_vs_deviant/heatmap_complexity_spearman_%s'%sens+suffix+'.svg')
    plt.close('all')
    petit_plot(pearson_r[sens],times,chance=0,fig_name=config.fig_path+'/SVM/standard_vs_deviant/corr_complexity_pearson_%s'%sens+suffix+'.png',plot_shaded_vertical=True)
    petit_plot(pearson_r[sens],times,chance=0,fig_name=config.fig_path+'/SVM/standard_vs_deviant/corr_complexity_pearson_%s'%sens+suffix+'.svg')
    plt.close('all')
    petit_plot(spearman_rho[sens],times,chance=0,fig_name=config.fig_path+'/SVM/standard_vs_deviant/corr_complexity_spearman_%s'%sens+suffix+'.png')
    petit_plot(spearman_rho[sens],times,chance=0,fig_name=config.fig_path+'/SVM/standard_vs_deviant/corr_complexity_spearman_%s'%sens+suffix+'.svg')

# ----- Then we compute the t-test to determine the statistical significance across subjects ----
# ----- on obtient la carte de à quel point c'est statistiquement significatif en fonction du temps ---
t_r = {sens : [] for sens in sensors}
t_rho = {sens : [] for sens in sensors}
for sens in sensors:
    for t in range(len(times)):
        corr_comp_pearson = pearson_r[sens]
        corr_comp_spearman = spearman_rho[sens]
        t_pearson, p_pearson = stats.ttest_1samp(corr_comp_pearson[:,t],popmean=0)
        t_spear, p_spear = stats.ttest_1samp(corr_comp_spearman[:,t],popmean=0)
        t_r[sens].append(t_pearson)
        t_rho[sens].append(t_spear)


#  ============== ============== ============== ============== ============== ============== ============== ============
#                         4 - plot the GAT diagonal for each of the 7 sequences
#  ============== ============== ============== ============== ============== ============== ============== ============

plt.close('all')
for sens in sensors:
    # ---- set figure's parameters, plot layout ----
    fig, ax = plt.subplots(1, 1, figsize=(10*0.8, 7*0.8))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    plt.axhline(0.5, linestyle='-', color='black', linewidth=1)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax.set_xlim(np.min(times),np.max(times))
    perform_seq = results[sens]
    for ii,SeqID in enumerate(range(1, 8)):
        perform_seqID = np.asarray(perform_seq['SeqID_' + str(SeqID)])
        diago_seq = np.diagonal(perform_seqID,axis1=1,axis2=2)
        reshaped_data[sens][ii,:,:] = diago_seq
        petit_plot(diago_seq, times, filter=True, color= colorslist[SeqID - 1],pos_sig=0.47-0.005*ii) #
    petit_plot(pearson_r[sens],times,chance=0,plot_shaded_vertical=True)
    plt.gca().set_xlabel('Time (ms)',fontsize=14)
    plt.gca().set_ylabel('Performance',fontsize=14)
    # plt.show()
    plt.gcf().savefig(op.join(config.fig_path, 'SVM/standard_vs_deviant/', 'All_sequences_standard_VS_deviant_cleaned_%s' % sens+suffix+'.svg'))
    plt.gcf().savefig(op.join(config.fig_path, 'SVM/standard_vs_deviant/', 'All_sequences_standard_VS_deviant_cleaned_%s' % sens+suffix+'.png'), dpi=300)
    plt.close('all')


