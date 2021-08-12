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
from ABseq_func import *
import matplotlib.pyplot as plt
import config
import numpy as np
from scipy.signal import savgol_filter
import os.path as op
from scipy import stats
from jr.plot import pretty_decod
import numpy as np

# ---- remind how the data was computed ---
# SVM decoder on standard VS deviants where standards are the ones that happen at the same location
# Training on one block testing on the other (nice because there is invertion of A and B). When there is only one block
# because of data acquisition problems, we split into 2 that block.
# Training on all the sequences together.
# Testing on each sequence separately.

# ---- set figure parameters ----

filter = True
NUM_COLORS = 7
cm = plt.get_cmap('viridis')
colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
plt.close('all')

filename = "SW_train_different_blocks_cleanedGAT_results"
subjects_list = config.subjects_list
sensors = ['mag','grad']
n_subjects = len(config.subjects_list)

# ------ load the data ------

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
            print(np.mean(GAT_results[sens]["average_all_sequences"]))
            for key in ['SeqID_%i' % i for i in range(1, 8)]:
                results[sens][key].append(GAT_results[sens][key])
            avg_res[sens].append(GAT_results[sens]["average_all_sequences"])
            n_subj +=1
        else:
            print("Missing data for %s "%GAT_path)

#  -----  plot the average diagonal over all sequences (and participants). Shaded bars = sem ------
def petit_plot(diago_score,times,filter=True,fig_name='',color='b',label='Average',chance = 0.5):

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

    if chance is not None:
        sig_mean = mean[times>0]
        sig_mean = sig_mean[sig<0.05]

    plt.fill_between(times, ub, lb, alpha=.2,color=color)
    plt.plot(times, mean, linewidth=1.5, label=label,color=color)
    if chance is not None:
        plt.plot(times_sig,sig_mean,linewidth=3,color=color)
    if fig_name is not None:
        plt.gcf().savefig(fig_name)

for sens in ['mag','grad']:
    plt.close('all')
    diago_score = np.asarray(avg_res[sens])
    diago_score = np.diagonal(diago_score,axis1=1,axis2=2)
    petit_plot(diago_score, times, filter=True, fig_name=config.fig_path+"/SVM/standard_vs_deviant/average_diagonal_cleaned"+sens+".svg")

# ------ PART 1 OF THE FIGURE -------
# ----- plot the GAT diagonal for each of the 7 sequences -----------
reshaped_data = {sens : np.zeros((7,n_subj,len(times))) for sens in sensors}
plt.close('all')
for sens in sensors:
    # ---- set figure's parameters, plot layout ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.axvline(0, linestyle='-', color='black', linewidth=2)
    plt.axhline(0.5, linestyle='-', color='black', linewidth=1)
    for xx in range(3):
        plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
    ax.set_xlim(np.min(times),np.max(times))
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('a.u.')
    plt.title('Decoder performance - ' + sens)
    # -----------------------------------------------
    perform_seq = results[sens]
    for ii,SeqID in enumerate(range(1, 8)):
        perform_seqID = np.asarray(perform_seq['SeqID_' + str(SeqID)])
        diago_seq = np.diagonal(perform_seqID,axis1=1,axis2=2)
        reshaped_data[sens][ii,:,:] = diago_seq
        petit_plot(diago_seq, times, filter=True, fig_name=None, color= colorslist[SeqID - 1],label='SeqID_' + str(SeqID))
    plt.show()
    plt.legend(loc='best', fontsize=9)
    plt.gcf().savefig(op.join(config.fig_path, 'SVM/standard_vs_deviant/', 'All_sequences_standard_VS_deviant_cleaned_%s.png' % sens), dpi=300)

# Commentaire : on dirait qu'il y a quelque chose d'étrange dans le calcul de ce qui est significatif. Pas grand chose ne sort alors qu'on aurait pu penser que oui ---

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
    petit_plot(pearson_r[sens],times,chance=0,fig_name=config.fig_path+'/SVM/standard_vs_deviant/corr_complexity_pearson_%s.png'%sens)
    petit_plot(pearson_r[sens],times,chance=0,fig_name=config.fig_path+'/SVM/standard_vs_deviant/corr_complexity_pearson_%s.svg'%sens)
    plt.close('all')
    petit_plot(spearman_rho[sens],times,chance=0,fig_name=config.fig_path+'/SVM/standard_vs_deviant/corr_complexity_spearman_%s.png'%sens)
    petit_plot(spearman_rho[sens],times,chance=0,fig_name=config.fig_path+'/SVM/standard_vs_deviant/corr_complexity_spearman_%s.svg'%sens)

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


t_values = {'Pearson':t_r,'Spearman':t_rho}
for name in t_values.keys():
    for sens in t_values[name].keys():
        pretty_decod(t_values[name][sens],times,chance=0)

        plt.gca().set_xlabel('Time [ms]')
        plt.gca().set_ylabel('T values')
        plt.gca().set_title('%s correlations - %s'%(name,sens))
        plt.gcf().savefig(op.join(config.fig_path, 'SVM/standard_vs_deviant/', 'tvalues_%s_correlation_%s.png'%(name,sens)))
        plt.gcf().savefig(op.join(config.fig_path, 'SVM/standard_vs_deviant/', 'tvalues_%s_correlation_%s.svg'%(name,sens)))
        plt.close('all')

extent=[0,100,0,1]
toplot = np.mean(pearson_r['mag'],axis=0)
plt.imshow(toplot[np.newaxis,:], aspect = "auto", cmap="viridis", extent=extent)
plt.gca().set_yticks([])
plt.colorbar()
plt.show()

sig = stats_funcs.stats(corr_comp_pearson['grad'])

# ----------- Temporal cluster based permutation test pour les différentes courbes
#  ---------------------------------------------------  (afficher en plus épais les parties significatives)



