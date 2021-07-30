import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
from ABseq_func import regression_funcs
import config
subject = config.subjects_list[0]

filter_names = ['Hab']

# --- update the metadata fields for the epochs (clean and dirty) and save them again ----
regression_funcs.update_metadata_epochs_and_save_epochs(subject)

for filter_name in filter_names:
    regression_funcs.compute_regression(subject,['Intercept','surprise_100','Surprisenp1','RepeatAlter','RepeatAlternp1'],"",filter_name)
    regression_funcs.compute_regression(subject,['Complexity'],"/Hab/Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/"+subject+"-residuals-baselined_clean-epo.fif",filter_name)




