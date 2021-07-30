import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
from ABseq_func import regression_funcs
import config
subject = config.subjects_list[0]


epochs_fname = ''
filter_name = 'Hab'
cleaned = True
remap_grads = False
lowpass_epochs = False
apply_baseline = True
suffix = ''
regressors_names = ['Intercept','surprise_100','Surprisenp1','RepeatAlter','RepeatAlternp1']
linear_reg_path = config.result_path+'/linear_models/'


# --- update the metadata fields for the epochs (clean and dirty) and save them again ----
regression_funcs.update_metadata_epochs_and_save_epochs(subject)
# - prepare the epochs (removing the ones that have nans for the fields of interest) and define the results path and suffix ---
epochs, results_path, suffix = regression_funcs.prepare_epochs_for_regression(subject,cleaned,epochs_fname,regressors_names,filter_name,remap_grads,lowpass_epochs,apply_baseline,suffix,linear_reg_path)
# --- run the regression with 4 folds ----
betas, scores = regression_funcs.run_regression_CV(epochs,regressors_names)
#  save the outputs of the regression : score, betas and residuals
regression_funcs.save_regression_outputs(subject, results_path, regressors_names, betas, scores)


