from ABseq_func import *
import config
from mne.parallel import parallel_func

# make less parallel runs to limit memory usage
# N_JOBS = max(config.N_JOBS // 4, 1)
N_JOBS = 2  # config.N_JOBS
#
config.subjects_list = ['sub16-ma_190185']


def EMS_analysis(subject):
    # creating the EMS results dictionnary
    EMS_funcs.generate_EMS_all_sequences(subject)
    EMS_funcs.GAT_EMS(subject)
    EMS_funcs.GAT_EMS_4pos(subject)
    EMS_funcs.apply_EMS_filter_16_items_epochs(subject)


# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(EMS_analysis, n_jobs=N_JOBS)

parallel(run_func(subject) for subject in config.subjects_list)
