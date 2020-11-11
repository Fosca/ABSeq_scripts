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
SVM_analysis('sub16-ma_190185')

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
parallel, run_func, _ = parallel_func(SVM_features, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)


## --------------------  Plot the SVM results    -----------------------

import os.path as op
import config
import numpy as np
import matplotlib.pyplot as plt
subject = config.subjects_list[0]
saving_directory = op.join(config.SVM_path, subject)

SVM_res = op.join(saving_directory,'GAT_results_4pos.npy')

res = np.load(SVM_res,allow_pickle=True).item()


import os.path as op
import config
import numpy as np
import matplotlib.pyplot as plt
from ABseq_func import *

config.subjects_list = ['sub01-pa_190002', 'sub02-ch_180036', 'sub05-cr_170417', 'sub06-kc_160388',
                        'sub07-jm_100109', 'sub09-ag_170045', 'sub10-gp_190568', 'sub11-fr_190151', 'sub12-lg_170436',
                        'sub13-lq_180242', 'sub14-js_180232', 'sub15-ev_070110', 'sub16-ma_190185','sub17-mt_170249', 'sub18-eo_190576',
                        'sub19-mg_190180']



# ===== LOAD (& PLOT) INDIVIDUAL DATA ===== #
GAT_sens_seq_all = {sens: [] for sens in ['eeg', 'mag', 'grad','all_chans']}
# for sens in ['eeg','mag','grad']:
#     GAT_sens_seq_all[sens]{'average_all_sequences': []}
for subject in config.subjects_list:
    SVM_path = op.join(config.SVM_path, subject)
    GAT_results = np.load(op.join(SVM_path, 'GAT_results.npy'), allow_pickle=True).item()
    print(op.join(SVM_path, 'GAT_results.npy'))
    times = GAT_results['times']
    GAT_results = GAT_results['GAT']
    sub_fig_path = op.join(config.fig_path, 'SVM', 'GAT', subject)
    utils.create_folder(sub_fig_path)
    for sens in ['eeg', 'mag', 'grad','all_chans']:
        if not GAT_sens_seq_all[sens]:  # initialize the keys and empty lists only the first time
            GAT_sens_seq_all[sens] = {'SeqID_%i' % i: [] for i in range(1, 8)}
            GAT_sens_seq_all[sens]['average_all_sequences'] = []
        for key in ['SeqID_%i' % i for i in range(1, 8)]:
            GAT_sens_seq_all[sens][key].append(GAT_results[sens][key])

            # ================ Plot & save each subject / each sequence figures ???
            #SVM_funcs.plot_GAT_SVM(GAT_results[sens][key], times, sens=sens, save_path=sub_fig_path, figname='GAT_' + key + '_'); plt.close('all')

        # ================ Plot & save each subject / average of all sequences figures ???
        GAT_sens_seq_all[sens]['average_all_sequences'].append(GAT_results[sens]['average_all_sequences'])
        SVM_funcs.plot_GAT_SVM(GAT_results[sens]['average_all_sequences'], times, sens=sens, save_path=sub_fig_path, figname='GAT_all_seq'+'_'); plt.close('all')

# ===== GROUP AVG FIGURES ===== #
plt.close('all')
for sens in ['eeg', 'mag', 'grad','all_chans']:
    GAT_avg_sens = GAT_sens_seq_all[sens]
    for seqID in range(1, 8):
        GAT_avg_sens_seq = GAT_avg_sens['SeqID_%i'%seqID]
        GAT_avg_sens_seq_groupavg = np.mean(GAT_avg_sens_seq,axis=0)
        SVM_funcs.plot_GAT_SVM(GAT_avg_sens_seq_groupavg, times, sens=sens, save_path=op.join(config.fig_path, 'SVM', 'GAT'), figname='GAT_'+str(seqID)+'_')
        plt.close('all')
    GAT_avg_sens_allseq_groupavg = np.mean(GAT_avg_sens['average_all_sequences'], axis=0)
    SVM_funcs.plot_GAT_SVM(GAT_avg_sens_allseq_groupavg, times, sens=sens, save_path=op.join(config.fig_path, 'SVM', 'GAT'), figname='GAT_all_seq'+'_')
