from ABseq_func import regression_funcs
import config


# =========================================================== #
# Linear regression group analysis (2nd level)
# =========================================================== #
import matplotlib.pyplot as plt  # avoids the script getting stuck when plotting sources ?!

filter_names = ['Hab', 'Stand', 'Viol']
for filter_name in filter_names:

    # Regression of complexity on data remapped on magnetometers - group analysis
    regressors_names = ['Intercept', 'Complexity']
    regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined')
    regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined', Do3Dplot=False)

    # Regression of complexity on original data - group analysis

    regressors_names = ['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter',
                                                      'RepeatAlternp1']

    regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined')
    regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_gtmbaselined', Do3Dplot=False)


    regressors_names = ['Complexity']
    regression_funcs.merge_individual_regression_results(regressors_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, suffix='--remapped_gtmbaselined')
    regression_funcs.regression_group_analysis(regressors_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, suffix='--remapped_gtmbaselined', Do3Dplot=False)

    # regression_funcs.merge_individual_regression_results(regressors_names, "", filter_name, suffix='--remapped_mtgclean')
    # regression_funcs.regression_group_analysis(regressors_names, "", filter_name, suffix='--remapped_mtgclean', Do3Dplot=False)
