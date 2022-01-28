"""
==========================================================
  PLOT LINEAR REGRESSION AS A FUNCTION OF COMPLEXITY
==========================================================

1 - Plot the regression coefficient for the complexity for Hab, Standard and Viol types of trials. Plot the projections on the sources

2 - Select a cluster obtained from the Cluster based permutation test in the sensor space. The same channels should be selected across the 3 conditions.
Illustrate the topomap of the cluster and show the averaged signals over the selected channels for the 7 different sequences.
"""

import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
from ABseq_func import regression_funcs
import os.path as op

# If you want to plot the projection of the betas on the source space, Samuel should run this code on his computer with Do3Dplot=True.
filter_names = ['Hab', 'Stand', 'Viol']
filter_names = ['Viol']
for filter_name in filter_names:
    print("--------------------1---------------------")
    # Regression of complexity on data remapped on magnetometers - group analysis
    regressors_names = ['Intercept', 'Complexity']
    # regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined')
    regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined', Do3Dplot=False)

