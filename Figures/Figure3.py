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

for sens in ['mag','grad']:
    plt.close('all')
    diago_score = np.asarray(avg_res[sens])
    diago_score = np.diagonal(diago_score,axis1=1,axis2=2)
    n_subj = 19
    import numpy as np
    mean = np.mean(diago_score, axis=0)
    ub = (mean + np.std(diago_score, axis=0) / (np.sqrt(n_subj)))
    lb = (mean - np.std(diago_score, axis=0) / (np.sqrt(n_subj)))
    if filter == True:
        mean = savgol_filter(mean, 11, 3)
        ub = savgol_filter(ub, 11, 3)
        lb = savgol_filter(lb, 11, 3)
    plt.fill_between(times, ub, lb, alpha=.2)
    plt.plot(times, mean, linewidth=1.5, label='Average')
    plt.gcf().savefig("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/figures/SVM/GAT/average_diagonal_cleaned"+sens+".svg")
    plt.gcf().savefig("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/figures/SVM/GAT/average_diagonal_cleaned"+sens+".png")

# ------ PART 1 OF THE FIGURE -------
# ----- plot the GAT diagonal for each of the 7 sequences -----------
reshaped_data = {sens : np.zeros((7,len(times),n_subj)) for sens in sensors}
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
        # Why is there a '-' sign ?
        data_seq_sens = np.diagonal(perform_seqID)
        reshaped_data[sens][ii,:,:] = data_seq_sens

        sig = stats_funcs.stats(data_seq_sens.T[:,times>0]-0.5)

        significance[sens]['SeqID_' + str(SeqID)] = sig
        # sig_times= times[sig<0.05]
        # sig_data= data_seq_sens[sig<0.05]

        color_mean = colorslist[SeqID - 1]
        # --- check out why I have to take the oposite ---
        mean = np.diagonal(np.mean(perform_seq['SeqID_' + str(SeqID)], axis=0))
        ub = np.diagonal(mean + np.std(perform_seq['SeqID_' + str(SeqID)], axis=0)/(np.sqrt(n_subj)))
        lb = np.diagonal(mean - np.std(perform_seq['SeqID_' + str(SeqID)], axis=0)/(np.sqrt(n_subj)))
        if filter == True:
            mean = savgol_filter(mean, 11, 3)
            ub = savgol_filter(ub, 11, 3)
            lb = savgol_filter(lb, 11, 3)
        plt.fill_between(times, ub, lb, color=color_mean, alpha=.2)
        plt.plot(times, mean, color=color_mean, linewidth=1.5, label='SeqID_' + str(SeqID))
        # plt.plot(sig_times, sig_data, color=color_mean, linewidth=3)

    plt.legend(loc='best', fontsize=9)
    plt.savefig(op.join(config.fig_path, 'SVM', 'All_sequences_standard_VS_deviant_cleaned_%s.png' % sens), dpi=300)

# Commentaire : on dirait qu'il y a quelque chose d'étrange dans le calcul de ce qui est significatif. Pas grand chose ne sort alors qu'on aurait pu penser que oui ---

# ------ PART 2 OF THE FIGURE: CORRELATION WITH COMPLEXITY -------
# compute per subject the correlation with complexity

correlation_complexity_pearson = {sens : [] for sens in sensors}
correlation_complexity_spearman = {sens : [] for sens in sensors}
complexity_dict = {1:4,2:6,3:6,4:6,5:12,6:15,7:28}
complexity = list(complexity_dict.values())
pearson_r = {sens : [] for sens in sensors}
spearman_rho = {sens : [] for sens in sensors}

for sens in sensors:
    data_sens = reshaped_data[sens]
    for nn in range(n_subjects):
        # ---- for 1 subject, diagonal of the GAT for all the 7 sequences through time ---
        dd = data_sens[:,:,nn]
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
    correlation_complexity_pearson[sens] = pearson_r[sens]
    correlation_complexity_spearman[sens] = spearman_rho[sens]

# ----- Then we compute the t-test to determine the statistical significance across subjects ----
# ----- on obtient la carte de à quel point c'est statistiquement significatif en fonction du temps ---
t_r = {sens : [] for sens in sensors}
t_rho = {sens : [] for sens in sensors}
for sens in sensors:
    for t in range(len(times)):
        corr_comp_pearson = correlation_complexity_pearson[sens]
        corr_comp_spearman = correlation_complexity_spearman[sens]
        t_pearson, p_pearson = stats.ttest_1samp(corr_comp_pearson[:,t],popmean=0)
        t_spear, p_spear = stats.ttest_1samp(corr_comp_spearman[:,t],popmean=0)
        t_r[sens].append(t_pearson)
        t_rho[sens].append(t_spear)

sig = stats_funcs.stats(correlation_complexity_pearson['grad'])

# ======= plot the t-values =====
plt.figure()
plt.plot(times, t_r['grad'])
plt.gca().set_xlabel('Time [ms]')
plt.gca().set_ylabel('T values')
plt.gca().set_title('Pearson correlations - grad')
plt.show()
plt.gcf().savefig(op.join(config.fig_path, 'SVM', 'tvalues_pearson_correlation_grad.png'))

plt.figure()
plt.plot(times, t_r['mag'])
plt.gca().set_xlabel('Time [ms]')
plt.gca().set_ylabel('T values')
plt.gca().set_title('Pearson correlations - mag')
plt.show()
plt.gcf().savefig(op.join(config.fig_path, 'SVM', 'tvalues_pearson_correlation_mag.png'))


plt.figure()
plt.plot(times, t_rho['mag'])
plt.gca().set_xlabel('Time [ms]')
plt.gca().set_ylabel('T values')
plt.gca().set_title('Spearman correlations - mag')
plt.show()
plt.gcf().savefig(op.join(config.fig_path, 'SVM', 'tvalues_spearman_correlation_mag.png'))


plt.figure()
plt.plot(times, t_rho['grad'])
plt.gca().set_xlabel('Time [ms]')
plt.gca().set_ylabel('T values')
plt.gca().set_title('Spearman correlations - grad')
plt.show()
plt.gcf().savefig(op.join(config.fig_path, 'SVM', 'tvalues_spearman_correlation_grad.png'))


# ----------- Temporal cluster based permutation test pour les différentes courbes
#  ---------------------------------------------------  (afficher en plus épais les parties significatives)



