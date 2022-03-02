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

filter_names = ['Hab', 'Stand', 'Viol']
for filter_name in filter_names:
    # regressors_names = ['Intercept','Complexity','surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1']
    # regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined')
    # regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined', Do3Dplot=False)
    regressors_names = ['Complexity']
    regression_funcs.merge_individual_regression_results(regressors_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, suffix='--clean')
    regression_funcs.regression_group_analysis(regressors_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, suffix='--clean', Do3Dplot=False,ch_types=['mag'])
    #
