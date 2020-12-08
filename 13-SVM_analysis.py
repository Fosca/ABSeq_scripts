import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import os.path as op
import config
import numpy as np
import matplotlib.pyplot as plt
from ABseq_func import *
import mne
import os.path as op
from importlib import reload
from mne.parallel import parallel_func



# -------------

def plot_all_subjects_results_SVM(analysis_name,subjects_list,fig_name,plot_per_sequence=False,plot_individual_subjects=False,score_field='GAT',folder_name = 'GAT',sensors = ['eeg', 'mag', 'grad','all_chans'],vmin=-0.1,vmax=.1):

    GAT_sens_all = {sens: [] for sens in sensors}

    for sens in sensors:
        print("==== running the plotting for sensor type %s"%sens)
        count = 0
        for subject in subjects_list:
            SVM_path = op.join(config.SVM_path, subject)
            GAT_path = op.join(SVM_path,analysis_name+'.npy')
            if op.exists(GAT_path):
                count +=1

                GAT_results = np.load(GAT_path, allow_pickle=True).item()
                print(op.join(SVM_path, analysis_name+'.npy'))
                times = GAT_results['times']
                GAT_results = GAT_results[score_field]
                fig_path = op.join(config.fig_path, 'SVM', folder_name)
                sub_fig_path = op.join(fig_path,subject)
                utils.create_folder(sub_fig_path)
                if plot_per_sequence:
                    if not GAT_sens_all[sens]:  # initialize the keys and empty lists only the first time
                        GAT_sens_all[sens] = {'SeqID_%i' % i: [] for i in range(1, 8)}
                        GAT_sens_all[sens]['average_all_sequences'] = []
                    for key in ['SeqID_%i' % i for i in range(1, 8)]:
                        GAT_sens_all[sens][key].append(GAT_results[sens][key])
                        # ================ Plot & save each subject / each sequence figures ???
                        if plot_individual_subjects:
                                SVM_funcs.plot_GAT_SVM(GAT_results[sens][key], times, sens=sens, save_path=sub_fig_path, figname=fig_name+key); plt.close('all')
                    # ================ Plot & save each subject / average of all sequences figures ???
                    GAT_sens_all[sens]['average_all_sequences'].append(GAT_results[sens]['average_all_sequences'])
                    SVM_funcs.plot_GAT_SVM(GAT_results[sens]['average_all_sequences'], times, sens=sens, save_path=sub_fig_path, figname=fig_name+'_all_seq',vmin=vmin,vmax=vmax)
                    plt.close('all')
                else:
                    GAT_sens_all[sens].append(GAT_results)
                    if plot_individual_subjects:
                        print('plotting for subject:%s'%subject)
                        print(sub_fig_path)
                        print("the shape of the GAT result is ")
                        print(GAT_results.shape)
                        SVM_funcs.plot_GAT_SVM(GAT_results, times, sens=sens, save_path=sub_fig_path,
                                               figname=fig_name,vmin=vmin,vmax=vmax)
                        plt.close('all')
        # return GAT_sens_all

        print("plotting in %s"%config.fig_path)
        if plot_per_sequence:
            for key in ['SeqID_%i' % i for i in range(1, 8)]:
                SVM_funcs.plot_GAT_SVM(np.nanmean(GAT_sens_all[sens][key],axis=0), times, sens=sens, save_path=fig_path,
                                       figname=fig_name+key,vmin=vmin, vmax=vmax)
                plt.close('all')
            SVM_funcs.plot_GAT_SVM(np.nanmean(GAT_sens_all[sens]['average_all_sequences'],axis=0), times, sens=sens,
                                   save_path=fig_path, figname=fig_name + '_all_seq' + '_',
                                   vmin=vmin, vmax=vmax)
            plt.close('all')
        else:
            SVM_funcs.plot_GAT_SVM(-np.mean(GAT_sens_all[sens],axis=0), times, sens=sens, save_path=fig_path, figname=fig_name,vmin=vmin,vmax=vmax)

    print("============ THE AVERAGE GAT WAS COMPUTED OVER %i PARTICIPANTS ========"%count)



# config.subjects_list = ['sub01-pa_190002', 'sub02-ch_180036', 'sub05-cr_170417', 'sub06-kc_160388',
#                         'sub07-jm_100109', 'sub09-ag_170045', 'sub10-gp_190568', 'sub11-fr_190151', 'sub12-lg_170436',
#                         'sub13-lq_180242', 'sub14-js_180232', 'sub15-ev_070110','sub17-mt_170249', 'sub18-eo_190576',
#                         'sub19-mg_190180']

# ___________________________________________________________________________
# ======= plot the GAT for all the sequences apart and together =============
# ___________________________________________________________________________


plot_all_subjects_results_SVM('SW_train_test_different_blocksGAT_results_score',config.subjects_list,'SW_train_test_different_blocksGAT_results_score',plot_per_sequence=True,vmin=-0.1,vmax=0.1)

# ___________________________________________________________________________
# ======= plot the GAT for the different features =============
# ___________________________________________________________________________
vmin = [0.45,0.20,0.4]
vmax = [0.55,0.3,0.6]

for ii,name in enumerate(['StimID_score_dict','RepeatAlter_score_dict','WithinChunkPosition_score_dict']):
    anal_name = 'feature_decoding/'+name
    plot_all_subjects_results_SVM(anal_name,config.subjects_list,name,score_field='score',plot_per_sequence=False,plot_individual_subjects=True,sensors = ['all_chans'],vmin=vmin[ii],vmax=vmax[ii])



# ===== GROUP AVG FIGURES ===== #
plt.close('all')
for sens in ['eeg', 'mag', 'grad','all_chans']:
    GAT_avg_sens = GAT_sens_seq_all[sens]
    for seqID in range(1, 8):
        GAT_avg_sens_seq = GAT_avg_sens['SeqID_%i'%seqID]
        GAT_avg_sens_seq_groupavg = np.mean(GAT_avg_sens_seq,axis=0)
        SVM_funcs.plot_GAT_SVM(GAT_avg_sens_seq_groupavg, times, sens=sens, save_path=op.join(config.fig_path, 'SVM', 'GAT'), figname=suf+'GAT_'+str(seqID)+score_suff+'_')
        plt.close('all')
    GAT_avg_sens_allseq_groupavg = np.mean(GAT_avg_sens['average_all_sequences'], axis=0)
    SVM_funcs.plot_GAT_SVM(GAT_avg_sens_allseq_groupavg, times, sens=sens, save_path=op.join(config.fig_path, 'SVM', 'GAT'), figname=suf+'GAT_all_seq'+score_suff+'_')

# ___________________________________________________________________________
# ======= plot the decoder predictions for the 16 item sequences ============
# ___________________________________________________________________________

# ===== LOAD DATA ===== #

suf = 'train_test_different_blocks'

epochs_16_items_mag_test_window = []; epochs_16_items_grad_test_window = []; epochs_16_items_eeg_test_window = [];epochs_16_items_all_chans_test_window = [];
epochs_16_items_mag_habituation_window = []; epochs_16_items_grad_habituation_window = []; epochs_16_items_eeg_habituation_window = [];epochs_16_items_all_chans_habituation_window = [];
for subject in config.subjects_list:
    if op.exists(op.join(config.meg_dir, subject, 'mag_SVM_on_16_items_test_window-epo.fif')):
        epochs_16_items_mag_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag'+suf+'_SVM_on_16_items_test_window-epo.fif')))
        epochs_16_items_grad_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad'+suf+'_SVM_on_16_items_test_window-epo.fif')))
        epochs_16_items_eeg_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg'+suf+'_SVM_on_16_items_test_window-epo.fif')))
        epochs_16_items_all_chans_test_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'all_chans'+suf+'_SVM_on_16_items_test_window-epo.fif')))
        epochs_16_items_mag_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'mag'+suf+'_SVM_on_16_items_habituation_window-epo.fif')))
        epochs_16_items_grad_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'grad'+suf+'_SVM_on_16_items_habituation_window-epo.fif')))
        epochs_16_items_eeg_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'eeg'+suf+'_SVM_on_16_items_habituation_window-epo.fif')))
        epochs_16_items_all_chans_habituation_window.append(mne.read_epochs(op.join(config.meg_dir, subject, 'all_chans'+suf+'_SVM_on_16_items_habituation_window-epo.fif')))


# ===== FIGURES ===== #
save_folder = op.join(config.fig_path, 'SVM', 'Full_sequence_projection')
utils.create_folder(save_folder)

# Figure with only one EMS projected (average window)
epochs_list = {}
for sens in ['all_chans','mag', 'grad', 'eeg']:
    if sens == 'mag':
        epochs_list['hab'] = epochs_16_items_mag_habituation_window
        epochs_list['test'] = epochs_16_items_mag_test_window
    elif sens == 'grad':
        epochs_list['hab'] = epochs_16_items_grad_habituation_window
        epochs_list['test'] = epochs_16_items_grad_test_window
    elif sens == 'eeg':
        epochs_list['hab'] = epochs_16_items_eeg_habituation_window
        epochs_list['test'] = epochs_16_items_eeg_test_window
    elif sens == 'all_chans':
        epochs_list['hab'] = epochs_16_items_all_chans_habituation_window
        epochs_list['test'] = epochs_16_items_all_chans_test_window

    win_tmin = epochs_list['test'][0][0].metadata.SVM_filter_tmin_window[0]*1000
    win_tmax = epochs_list['test'][0][0].metadata.SVM_filter_tmax_window[0]*1000

    SVM_funcs.plot_SVM_projection_for_seqID_window_allseq_heatmap(epochs_list, sensor_type=sens, save_path=op.join(save_folder, 'AllSeq_%s_window_%i_%ims.png' % ( sens, win_tmin, win_tmax)),vmin=-1.5,vmax=1.5)

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
    list_features = ['Identity','RepeatAlter','ChunkNumber','WithinChunkPosition']
    for feature_name in list_features:
        score, times = SVM_funcs.SVM_decode_feature(subject, feature_name, load_residuals_regression=False)
        save_path = config.SVM_path+subject + '/feature_decoding/'
        utils.create_folder(save_path)
        save_name = save_path+feature_name+'_score_dict.npy'
        np.save(save_name,{'score':score,'times':times})



# Here we use fewer N_JOBS to prevent potential memory problems
parallel, run_func, _ = parallel_func(SVM_features, n_jobs=N_JOBS)
parallel(run_func(subject) for subject in config.subjects_list)
>>>>>>> Stashed changes
