# This module contains all the functions that allow the computations to run on the cluster.
from __future__ import division
import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
from initialization_paths import initialization_paths
from ABseq_func import TP_funcs, SVM_funcs, utils, epoching_funcs
import config
import mne
import numpy as np
import pickle
from autoreject import AutoReject

def create_qsub(function_name, folder_name, suffix_name, sublist_subjects=None, queue='Unicog_long'):
    import subprocess
    import os, sys, glob

    # ==========================================================================================
    # ============= ============= create the jobs ============================= ============= ==
    # ==========================================================================================

    ########################################################################
    # List of parameters to be parallelized
    ListSubject = config.subjects_list
    if sublist_subjects is not None:
        ListSubject = sublist_subjects

    ########################################################################
    # Initialize job files and names

    List_python_files = []

    wkdir = config.cluster_path
    base_path = config.scripts_path
    initbody = 'import sys \n'
    initbody = initbody + "sys.path.append(" + "'" + base_path + "')\n"
    initbody = initbody + 'from ABseq_func import cluster_funcs\n'

    # Write actual job files
    python_file, Listfile, ListJobName = [], [], []

    for s, subject in enumerate(ListSubject):
        print(subject)

        additionnal_parameters = ''

        body = initbody + "cluster_funcs.%s('%s')" % (function_name, subject)

        jobname = suffix_name + '_' + subject

        ListJobName.append(jobname)

        # Write jobs in a dedicated folder
        path_jobs = wkdir + '/generated_jobs/' + folder_name + '/'
        utils.create_folder(path_jobs)
        name_file = path_jobs + jobname + '.py'
        Listfile.append(name_file)

        with open(name_file, 'w') as python_file:
            python_file.write(body)

    # ============== Loop over your jobs ===========

    jobs_path = config.cluster_path + "/generated_jobs/"
    results_path = config.cluster_path + "/results_qsub/"
    utils.create_folder(results_path + folder_name)
    list_scripts = sorted(glob.glob(jobs_path + folder_name + "/*.py"))

    # Loop over your jobs

    for i in list_scripts:
        # Customize your options here
        file_name = os.path.split(i)
        job_name = "%s" % file_name[1]

        walltime = "48:00:00"  # "24:00:00"
        if 'short' in queue:
            walltime = "2:00:00"  # "24:00:00"

        processors = "nodes=1:ppn=1"
        command = "python %s" % i
        standard_output = "/std_%s" % file_name[1]
        error_output = "/err_%s" % file_name[1]
        name_file = "/qsub_cmd_%s" % file_name[1]

        job_string = """#!/bin/bash
        #PBS -N %s
        #PBS -q %s 
        #PBS -l walltime=%s
        #PBS -l %s
        #PBS -o %s
        #PBS -e %s 
        cd %s
        %s""" % (job_name, queue, walltime, processors, results_path + folder_name + standard_output,
                 results_path + folder_name + error_output, results_path, command)

        # job_file = jobs_path + folder_name + '/' + name_file
        job_file = jobs_path + name_file
        fichier = open(job_file, "w")
        fichier.write(job_string)
        fichier.close()

        # Send job_string to qsub
        cmd = "qsub %s" % (job_file)
        subprocess.call(cmd, shell=True)


def epoch_items(subject):
    epoching_funcs.run_epochs(subject, epoch_on_first_element=False, baseline=None)


def epoch_full_trial(subject):
    epoching_funcs.run_epochs(subject, epoch_on_first_element=True, baseline=True)


def EMS(subject):
    # EMS_funcs.generate_EMS_all_sequences(subject)
    # EMS_funcs.GAT_EMS(subject)
    # EMS_funcs.GAT_EMS_4pos(subject)
    # EMS_funcs.apply_EMS_filter_16_items_epochs(subject)
    # EMS_funcs.apply_EMS_filter_16_items_epochs_habituation(subject)
    EMS_funcs.apply_EMS_filter_16_items_epochs(subject, times=[0.140, 0.180], window=True)
    EMS_funcs.apply_EMS_filter_16_items_epochs_habituation(subject, times=[0.140, 0.180], window=True)



def autoreject_marmouset(subject):

    root_path = '/neurospin/unicog/protocols/ABSeq_marmousets/'
    neural_data_path = root_path+'neural_data/'

    subject = 'Nr'
    epoch_name = '/epoch_items'
    tmin = -0.099

    # ======== rebuild the epoch object and run autoreject ========
    epoch_data = np.load(neural_data_path + subject + epoch_name+'_data.npy')
    info = np.load(neural_data_path + subject + epoch_name+ '_info.npy',allow_pickle=True).item()
    metadata = np.load(neural_data_path + subject + epoch_name +'_metadata.pkl',allow_pickle=True)
    epochs = mne.EpochsArray(epoch_data,info=info,tmin=tmin)
    epochs.metadata = metadata
    epochs.load_data()

    # ======== ======== ======== ======== ======== ======== ========
    ar = AutoReject()
    epochs, reject_log = ar.fit_transform(epochs, return_log=True)
    epochs_clean_fname = neural_data_path + subject + epoch_name+'_clean.fif'
    print("Output: ", epochs_clean_fname)
    epochs.save(epochs_clean_fname, overwrite=True)
    # Save autoreject reject_log
    pickle.dump(reject_log, open(epochs_clean_fname[:-4] + '_reject_log.obj', 'wb'))
    np.save(neural_data_path + subject + epoch_name+'_data_clean.npy',epochs.get_data())
    epochs.metadata.to_pickle(neural_data_path + subject + epoch_name +'_metadata_clean.pkl')
    np.save(neural_data_path + subject + epoch_name+ '_info_clean.npy',epochs.info)

# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------------------- DECODING FUNCTIONS FOR THE CLUSTER ------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def SVM_generate_different_sequences(subject):
    SVM_funcs.generate_SVM_separate_sequences(subject, load_residuals_regression=True,sliding_window=True)
    SVM_funcs.GAT_SVM_trained_separate_sequences(subject, load_residuals_regression=True,sliding_window=True)


def SVM_generate_all_sequences(subject):
    SVM_funcs.generate_SVM_all_sequences(subject, load_residuals_regression=True,sliding_window=True)
def GAT_SVM_all_seq(subject):
    SVM_funcs.GAT_SVM_trained_all_sequences(subject, load_residuals_regression=True,sliding_window=True)

def GAT_SVM_separate_seq(subject):
    SVM_funcs.GAT_SVM_trained_separate_sequences(subject, load_residuals_regression=True,sliding_window=True)

def SVM_generate_different_sequences(subject):
    # SVM_funcs.generate_SVM_separate_sequences(subject, load_residuals_regression=True,sliding_window=True)
    SVM_funcs.GAT_SVM_trained_separate_sequences(subject, load_residuals_regression=True,sliding_window=True)

def SVM_GAT_all_sequences(subject):
    SVM_funcs.GAT_SVM(subject, load_residuals_regression=True,sliding_window=True)

def SVM_full_sequences_16items(subject):
    # ----- We test on the 16 items sequences. We average the predictions of the decoders between 140 and 180 ms -----
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject, times=[0.140, 0.180], window=True, sliding_window=True)
    SVM_funcs.apply_SVM_filter_16_items_epochs_habituation(subject, times=[0.140, 0.180], window=True, sliding_window=True)

def SVM_features_repeatalter(subject,load_residuals_regression=True):
    if load_residuals_regression:
        resid_suffix = 'resid_cv_'
    else:
        resid_suffix='full_data_'
    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'RepeatAlter',load_residuals_regression=load_residuals_regression,list_sequences=[3,4,5,6,7],crop = [-0.1,0.4],cross_val_func=None)
    save_name = config.SVM_path + subject + '/feature_decoding/' + resid_suffix + 'RepeatAlter' + '_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})

def SVM_features_stimID(subject,load_residuals_regression=True):
    if load_residuals_regression:
        resid_suffix = 'resid_cv_'
    else:
        resid_suffix='full_data_'
    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'StimID',load_residuals_regression=load_residuals_regression,crop = [-0.1,0.4],cross_val_func=None,eeg=False,decim=4)
    save_name = config.SVM_path + subject + '/feature_decoding/' + resid_suffix + 'StimID' + '_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})

def SVM_features_stimID_eeg(subject,load_residuals_regression=True):
    if load_residuals_regression:
        resid_suffix = 'resid_cv_'
    else:
        resid_suffix='full_data_'
    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'StimID', load_residuals_regression=load_residuals_regression,crop = [-0.1,0.4],cross_val_func=None,meg=False)
    save_name = config.SVM_path + subject + '/feature_decoding/' + resid_suffix + 'StimID' + '_EEGONLY_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})


def SVM_quad_ordpos(subject):

    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'WithinChunkPosition',load_residuals_regression=True, list_sequences=[4],crop = [-0.1,0.4],cross_val_func=None,filter_from_metadata="StimPosition > 2 and StimPosition < 15")
    save_name = config.SVM_path + subject + '/feature_decoding/' + 'resid_'+ 'WithinChunkPosition' + '_quads_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})
    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'WithinChunkPosition',load_residuals_regression=False, list_sequences=[4],crop = [-0.1,0.4],cross_val_func=None,filter_from_metadata="StimPosition > 2 and StimPosition < 15")
    save_name = config.SVM_path + subject + '/feature_decoding/' + 'full_data_'+ 'WithinChunkPosition' + '_quads_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})

def SVM_features_withinchunk(subject,load_residuals_regression=True):
    if load_residuals_regression:
        resid_suffix = 'resid_'
    else:
        resid_suffix='full_data_'
    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'WithinChunkPosition',load_residuals_regression=load_residuals_regression, list_sequences=[4,5,6],crop = [-0.1,0.4],cross_val_func=None)
    save_name = config.SVM_path + subject + '/feature_decoding/' + resid_suffix + 'WithinChunkPosition' + '_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})

def SVM_features_withinchunk_train_quads_test_others(subject,load_residuals_regression=True):
    if load_residuals_regression:
        resid_suffix = 'resid_cv_'
    else:
        resid_suffix='full_data_'
    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'WithinChunkPosition',load_residuals_regression=load_residuals_regression,crop = [-0.1,0.4],cross_val_func=SVM_funcs.train_quads_test_others,balance_features=False,filter_from_metadata="StimPosition > 2 and StimPosition < 15")
    save_name = config.SVM_path + subject + '/feature_decoding/' + resid_suffix + 'WithinChunkPosition_train_Quads_test_others' + '_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})

def SVM_features_withinchunkrev(subject,load_residuals_regression=True):
    if load_residuals_regression:
        resid_suffix = 'resid_cv_'
    else:
        resid_suffix='full_data_'
    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'WithinChunkPositionReverse',load_residuals_regression=load_residuals_regression,list_sequences=[4,5,6],crop = [-0.1,0.4],cross_val_func=None)
    save_name = config.SVM_path + subject + '/feature_decoding/' + resid_suffix + 'WithinChunkPositionReverse' + '_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})

# ----- quelles séquences ? ----
def SVM_features_number_ofOpenedChunks(subject,load_residuals_regression=True):
    if load_residuals_regression:
        resid_suffix = 'resid_cv_'
    else:
        resid_suffix='full_data_'
    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'OpenedChunks',SVM_dec=SVM_funcs.regression_decoder(),load_residuals_regression=load_residuals_regression,list_sequences=[3,4,5,6,7],crop = [-0.1,0.4],cross_val_func=None,balance_features=False,distance=False)
    save_name = config.SVM_path + subject + '/feature_decoding/' + resid_suffix + 'Number_Open_Chunks' + '_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})

# ----- quelles séquences pour chunk opening ? ----
def SVM_features_chunkBeg(subject,load_residuals_regression=True):
    if load_residuals_regression:
        resid_suffix = 'resid_cv_'
    else:
        resid_suffix='full_data_'
    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'ChunkBeginning',load_residuals_regression=load_residuals_regression,list_sequences=[3,4,5,6,7],crop = [-0.1,0.4],cross_val_func=None)
    save_name = config.SVM_path + subject + '/feature_decoding/' + resid_suffix + 'ChunkBeg' + '_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})

# ----- quelles séquences pour chunk closing ? ----
def SVM_features_chunkEnd(subject,load_residuals_regression=True):
    if load_residuals_regression:
        resid_suffix = 'resid_cv_'
    else:
        resid_suffix='full_data_'

    score, distance, times = SVM_funcs.SVM_decode_feature(subject, 'ChunkEnd',load_residuals_regression=load_residuals_regression,list_sequences=[3,4,5,6,7], crop = [-0.1,0.4],cross_val_func=None)
    save_name = config.SVM_path + subject + '/feature_decoding/' + resid_suffix + 'ChunkEnd' + '_score_dict.npy'
    np.save(save_name, {'score': score, 'times': times, 'distance':distance})

def ord_code_16items(subject,load_residuals_regression=False):
    SVM_funcs.SVM_ordinal_code_train_quads_test_others(subject, load_residuals_regression=load_residuals_regression)

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


def compute_evoked(subject):
    evoked_funcs.create_evoked(subject, cleaned=False)
    evoked_funcs.create_evoked(subject, cleaned=True)


def linear_reg(subject):
    from ABseq_func import linear_reg_funcs  # spent hours on the issue "linear_reg_funcs is not defined", although all other similar functions worked with no issues. This was the solution.
    linear_reg_funcs.run_linear_regression(subject, cleaned=True)


def surprise_omegas_analysis(subject):
    import numpy as np
    from ABseq_func import TP_funcs
    list_omegas = np.logspace(-1,2,50)

    TP_funcs.from_epochs_to_surprise(subject, list_omegas)
    TP_funcs.append_surprise_to_metadata_clean(subject)
    from importlib import reload
    reload(TP_funcs)
    # TP_funcs.run_linear_regression_surprises(subject, list_omegas, clean=True, decim=50,hfilter=None)
    TP_funcs.run_linear_regression_surprises(subject, list_omegas, clean=True, decim=None, hfilter=10)

    # ----------- then we have to compute the optimal omega for each time and channel -------------
    # TP_funcs.regress_out_optimal_omega(subject, clean=True)
    # TP_funcs.compute_posterior_probability(subject)
    # TP_funcs.regress_out_optimal_omega_per_channel(subject)


def simplified_linear_regression(subject):
    from ABseq_func import linear_reg_funcs
    linear_reg_funcs.run_linear_reg_surprise_repeat_alt(subject)


def simplified_with_complexity(subject):
    from ABseq_func import linear_reg_funcs
    linear_reg_funcs.run_linear_reg_surprise_repeat_alt(subject, with_complexity=True)
