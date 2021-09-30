import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
from ABseq_func import regression_funcs
import config

# Exclude some subjects
config.exclude_subjects.append('sub16-ma_190185')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

# =========================================================== #
# Update epochs metadata
# =========================================================== #

# --- update the metadata fields for the epochs (clean and dirty) and save them again ----
for subject in config.subjects_list:
    regression_funcs.update_metadata_epochs_and_save_epochs(subject)

# =========================================================== #
# Individual subjects regressions (1st level)
# =========================================================== #

filter_names = ['Hab', 'Stand', 'Viol']
for subject in config.subjects_list:
    for filter_name in filter_names:
        # Regression of complexity on original data - each participant
        regression_funcs.compute_regression(subject, ['Intercept', 'Complexity'], "", filter_name, apply_baseline=True)
        # Regression of surprise (& more) on original data - each participant
        regression_funcs.compute_regression(subject, ['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1'], "", filter_name, apply_baseline=True)
        # Regression of complexity on surprise-regression residuals - each participant
        regression_funcs.compute_regression(subject, ['Complexity'], "/Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/" + subject + "/residuals--remapped_baselined_clean-epo.fif",
                                            filter_name, apply_baseline=False)

# =========================================================== #
# Group analyses (2nd level)
# =========================================================== #
import matplotlib.pyplot as plt  # avoids the script getting stuck when plotting sources ?!

filter_names = ['Hab', 'Stand', 'Viol']
for filter_name in filter_names:

    # Regression of complexity on original data - group analysis
    reg_names = ['Intercept', 'Complexity']
    regression_funcs.merge_individual_regression_results(reg_names, "", filter_name)
    regression_funcs.regression_group_analysis(reg_names, "", filter_name, remap_grads=True, Do3Dplot=True)

    # Regression of structure regressors on surprise-regression residuals - group analysis
    reg_names = ['Complexity','WithinChunkPosition','ChunkBeginning', 'ChunkEnd', 'ChunkNumber', 'ChunkDepth']
    regression_funcs.merge_individual_regression_results(reg_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name)
    regression_funcs.regression_group_analysis(reg_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, remap_grads=True, Do3Dplot=False)


    # Regression of surprise (& more) on original data - group analysis
    reg_names = ['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter', 'RepeatAlternp1']
    regression_funcs.merge_individual_regression_results(reg_names, "", filter_name)
    regression_funcs.regression_group_analysis(reg_names, "", filter_name, remap_grads=True, Do3Dplot=True)

    # Regression of complexity on surprise-regression residuals - group analysis
    reg_names = ['Complexity']
    regression_funcs.merge_individual_regression_results(reg_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name)
    regression_funcs.regression_group_analysis(reg_names, "Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1", filter_name, remap_grads=True, Do3Dplot=True)
