"""
===============================================================================
 LINEAR REGRESSION AS A FUNCTION OF SURPRISE FROM STATISTICAL TRANSITION PROBAS
 LINEAR REGRESSION OF THE RESIDUALS AS A FUNCTION OF COMPLEXITY
===============================================================================

1 - In the 3 trial types ('Hab','Stand','Viol'), plot the evoked responses for 1 - Epochs, 2 - Intercept-explained signal 3 - Surprise from transitions explained signal 4 - Residuals

2 - Linear regression of the residuals as a function of complexity. Select a cluster obtained from the Cluster based permutation test in the sensor space. The same channels should be selected across the 3 conditions.
Illustrate the topomap of the cluster and show the averaged signals over the selected channels for the 7 different sequences.
"""

import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
from ABseq_func import regression_funcs, article_plotting_funcs

"""
filter_names = ['Hab', 'Stand', 'Viol']
for filter_name in filter_names:
    article_plotting_funcs.load_epochs_explained_signal_and_residuals_and_plot(['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter',
                        'RepeatAlternp1'], filter_name=filter_name,suffix='--remapped_gtmbaselined_clean-epo.fif', compute=False,format='.png')
    article_plotting_funcs.load_epochs_explained_signal_and_residuals_and_plot(['Complexity'], filter_name=filter_name,suffix='--clean-epo.fif',
                                                                               compute=False,to_append_to_results_path="_from_Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1--Complexity",format='.png')

"""

#filter_names = ['Hab', 'Stand', 'Viol']
filter_names = ['Hab']
for filter_name in filter_names:
    regressors_names = ['Intercept','Complexity','surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1']
    regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined')
    regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined', Do3Dplot=False)
    regressors_names = ['Complexity']
    regression_funcs.merge_individual_regression_results(regressors_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, suffix='--clean')
    regression_funcs.regression_group_analysis(regressors_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, suffix='--clean', Do3Dplot=False,ch_types=['mag'])
    #


epo = epoching_funcs.load_epochs_items(config.subjects_list[3],cleaned=True)

metadata = epo.metadata
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

predictors = ['Complexity','surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1']
correlation_matrix = np.zeros((len(predictors),len(predictors)))

for i1, pred1 in enumerate(predictors):
    for i2, pred2 in enumerate(predictors):
        val1 = metadata[pred1].values
        val2 = metadata[pred2].values
        r = ma.corrcoef(ma.masked_invalid(val1), ma.masked_invalid(val2))
        correlation_matrix[i1,i2] = np.abs(r[0,1])


cm = plt.get_cmap('viridis')
plt.imshow(correlation_matrix)
plt.colorbar()
plt.title('Correlation across predictors')
plt.xticks(range(len(predictors)),predictors,rotation=30)
plt.yticks(range(len(predictors)),predictors,rotation=30)
fig = plt.gcf()
fig.show()

import mne
path_res = "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/results/linear_models/Hab/from_Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1--Complexity/group/Complexity--clean_epo.fif"
path_mutli = "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/results/linear_models/Hab/Intercept_Complexity_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/group/Complexity--remapped_gtmbaselined_epo.fif"

epo_res = mne.read_epochs(path_res)
epo_mutli = mne.read_epochs(path_mutli)

epo_diff = epo_res.copy()
epo_diff._data = epo_res._data - epo_mutli._data

epo_diff.average().plot_joint()
epo_res.average().plot_joint()
epo_mutli.average().plot_joint()
