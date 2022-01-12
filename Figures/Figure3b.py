import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import os.path as op
from ABseq_func import regression_funcs
import config
import mne
import numpy as np
# =========================================================== #
# Linear regression group analysis (2nd level)
# =========================================================== #
import matplotlib.pyplot as plt  # avoids the script getting stuck when plotting sources ?!

def load_epochs_explained_signal_and_residuals(regressors_names,filter_name='Hab',suffix='--remapped_gtmbaselined_clean-epo.fif',compute=True):

    results_path = op.join(config.result_path, 'linear_models')
    epochs_all = []
    explained_signal_all = []
    residuals_all = []
    intercept_all = []

    to_append_to_results_path = ''
    for name in regressors_names:
        to_append_to_results_path += '_' + name
    results_path = op.join(results_path,filter_name, to_append_to_results_path[1:])

    if compute:
        for subject in config.subjects_list:
            print(subject)
            subj_path = op.join(results_path,subject)
            epochs = mne.read_epochs(op.join(subj_path,'epochs'+suffix))
            intercept = mne.read_epochs(op.join(subj_path,'intercept'+suffix))
            explained_signal = mne.read_epochs(op.join(subj_path,'explained_signal'+suffix))
            residuals = mne.read_epochs(op.join(subj_path,'residuals'+suffix))
            epochs_all.append(epochs.average()._data)
            explained_signal_all.append(explained_signal.average()._data)
            residuals_all.append(residuals.average()._data)
            intercept_all.append(intercept.average()._data)

        epo = mne.EpochsArray(np.asarray(epochs_all),tmin=epochs.tmin, info = epochs.info)
        expl = mne.EpochsArray(np.asarray(explained_signal_all),tmin=epochs.tmin, info = epochs.info)
        resid = mne.EpochsArray(np.asarray(residuals_all),tmin=epochs.tmin, info = epochs.info)
        interc =mne.EpochsArray(np.asarray(intercept_all),tmin=epochs.tmin, info = epochs.info)
        epo.save(op.join(results_path,'epochs_allsubjects-epo.fif'),overwrite=True)
        expl.save(op.join(results_path,'explained_signal_allsubjects-epo.fif'),overwrite=True)
        resid.save(op.join(results_path,'residuals_allsubjects-epo.fif'),overwrite=True)
        interc.save(op.join(results_path,'intercept_allsubjects-epo.fif'),overwrite=True)
    else:
        epo = mne.read_epochs(op.join(results_path,'epochs_allsubjects-epo.fif'))
        expl = mne.read_epochs(op.join(results_path,'explained_signal_allsubjects-epo.fif'))
        resid = mne.read_epochs(op.join(results_path,'residuals_allsubjects-epo.fif'))
        interc = mne.read_epochs(op.join(results_path,'intercept_allsubjects-epo.fif'))

    print("==== NOW PLOTTING ===")

    fig = epo.average().plot_joint()
    plt.gcf().savefig(op.join(results_path,'group','figures','epochs_allsubjects.svg'))
    plt.close(fig)

    fig = expl.average().plot_joint()
    plt.gcf().savefig(op.join(results_path,'group','figures','explained_signal_allsubjects.svg'))
    plt.close(fig)

    fig = resid.average().plot_joint()
    plt.savefig(op.join(results_path,'group','figures','residuals_allsubjects.svg'))
    plt.close(fig)

    fig = interc.average().plot_joint()
    plt.savefig(op.join(results_path,'group','figures','intercept_allsubjects.svg'))
    plt.close(fig)



filter_names = ['Hab', 'Stand', 'Viol']
for filter_name in filter_names:
    print("--- runing the analysis for "+filter_name +" -----")
    regressors_names = ['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter',
                                                      'RepeatAlternp1']
    load_epochs_explained_signal_and_residuals(regressors_names, filter_name='Hab',
                                               suffix='--remapped_gtmbaselined_clean-epo.fif',compute=False)


filter_names = ['Hab', 'Stand', 'Viol']
for filter_name in filter_names:
    print("--------------------1---------------------")
    # Regression of complexity on data remapped on magnetometers - group analysis
    regressors_names = ['Intercept', 'Complexity']
    regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined')
    regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined', Do3Dplot=False)

    # Regression of complexity on original data - group analysis

    print("--------------------2---------------------")
    regressors_names = ['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter',
                                                      'RepeatAlternp1']

    regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined')
    regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined', Do3Dplot=False)

    print("--------------------3---------------------")
    regressors_names = ['Complexity']
    regression_funcs.merge_individual_regression_results(regressors_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, suffix='--clean')
    regression_funcs.regression_group_analysis(regressors_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, suffix='--clean', Do3Dplot=False,ch_types=['mag'])

    # regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name, suffix='--remapped_mtgclean')
    # regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_mtgclean', Do3Dplot=False)










