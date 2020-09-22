from ABseq_func import *
import config
from mne.parallel import parallel_func
import numpy as np
# make less parallel runs to limit memory usage
# N_JOBS = max(config.N_JOBS // 4, 1)
N_JOBS = 2  # config.N_JOBS
#
config.subjects_list = ['sub16-ma_190185']


def SVM_analysis(subject):
    # creating the SVM results dictionnary
    # SVM_funcs.generate_SVM_all_sequences(subject)
    # SVM_funcs.GAT_SVM(subject)
    # SVM_funcs.GAT_SVM_4pos(subject)
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject)


def SVM_features(subject):
    list_features = ['Identity','RepeatAlter','ChunkNumber','WithinChunkPosition','WithinChunkPositionReverse',
                     'ChunkDepth','OpenedChunks','ChunkSize','ChunkBeginning','ChunkEnd']
    for feature_name in list_features:
        score, times = SVM_funcs.SVM_decode_feature(subject, feature_name, load_residuals_regression=False)
        save_path = config.SVM_path+subject + '/feature_decoding/'
        utils.create_folder(save_path)
        save_name = save_path+feature_name+'_score_dict.npy'
        np.save(save_name,{'score':score,'times':times})


# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(SVM_analysis, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
