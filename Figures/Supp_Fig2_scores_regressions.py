import mne
from sklearn.metrics import explained_variance_score, max_error,mean_absolute_error,mean_squared_error, median_absolute_error, r2_score
import numpy as np
import os.path as op
import matplotlib.pyplot as plt

def load_and_plot_scores_regressions(sub_path = "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1",filter_name='Stand',scores_name = 'scores--remapped_gtmbaselined_clean.npy'):

    from jr.plot import pretty_decod

    results_path = op.join(config.result_path, 'linear_models', filter_name,sub_path)

    scores_all = []
    for subject in config.subjects_list:
        subject_path = op.join(results_path,subject)
        score = np.load(op.join(subject_path,scores_name))
        scores_all.append(score)

    scores_all = np.asarray(scores_all)

    pretty_decod(np.mean(scores_all,1),np.linspace(-0.048,0.65,scores_all.shape[-1]),chance = 0)
    plt.title("Score "+ filter_name )
    plt.show()

load_and_plot_scores_regressions(filter_name='Hab')
load_and_plot_scores_regressions(filter_name='Stand')
load_and_plot_scores_regressions(filter_name='Viol')


epo = mne.read_epochs('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/results/linear_models/Viol/Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/epochs_allsubjects-epo.fif')
intercept = mne.read_epochs('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/results/linear_models/Viol/Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/intercept_allsubjects-epo.fif')
explained = mne.read_epochs('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/results/linear_models/Viol/Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/explained_signal_allsubjects-epo.fif')


def plot_score(score_metric,epo_ref = epo,epo2 = intercept):
    score_acc = []
    for chan in range(epo2._data.shape[1]):
        score_acc_sub = []
        for times in range(epo2._data.shape[2]):
            data_intercept = list(epo2._data[:,chan,times]*10**15)
            data_epoch = list(epo_ref._data[:,chan,times]*10**15)
            score = score_metric(data_intercept,data_epoch)
            score_acc_sub.append(score)
        score_acc.append(score_acc_sub)

    score_acc = np.asarray(score_acc)
    epo_score = mne.EpochsArray(score_acc[np.newaxis,:,:],info=epo_ref.info)
    epo_score.average().plot_joint()

scores =  [explained_variance_score, max_error,mean_absolute_error,mean_squared_error, median_absolute_error, r2_score]
scores =  [median_absolute_error, r2_score]
for score_metric in scores :
    plot_score(score_metric)

for score_metric in scores:
    plot_score(score_metric,epo2 = explained)