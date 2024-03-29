# This module contains all the functions that allow the computations to run on the cluster.
from __future__ import division
import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/umne/')
from initialization_paths import initialization_paths
from ABseq_func import TP_funcs, SVM_funcs, utils, epoching_funcs, rsa_funcs
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
    SVM_funcs.generate_SVM_separate_sequences(subject, load_residuals_regression=False,sliding_window=True)
    SVM_funcs.GAT_SVM_trained_separate_sequences(subject, load_residuals_regression=False,sliding_window=True)


# 1 - torun
def SVM_generate_all_sequences(subject):
    # ---- I modified the following functions so they run on the cleaned epochs ------
    # --- generate the decoders on the different folds (split per run) ------
    SVM_funcs.generate_SVM_all_sequences(subject, load_residuals_regression=False,sliding_window=True,cleaned=True)
    # ----
    SVM_funcs.GAT_SVM_trained_all_sequences(subject, load_residuals_regression=False,sliding_window=True,cleaned=True)
    # apply to the 16 items sequences
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject, times=[0.131, 0.210], sliding_window=True,cleaned=True)
    SVM_funcs.apply_SVM_filter_16_items_epochs_habituation(subject, times=[0.131, 0.210], sliding_window=True,cleaned=True)
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject, times=[0.211, 0.410],  sliding_window=True,cleaned=True)
    SVM_funcs.apply_SVM_filter_16_items_epochs_habituation(subject, times=[0.211, 0.410], sliding_window=True,cleaned=True)



def GAT_SVM_separate_seq(subject):
    SVM_funcs.GAT_SVM_trained_separate_sequences(subject, load_residuals_regression=True,sliding_window=True)

def SVM_generate_different_sequences(subject):
    # SVM_funcs.generate_SVM_separate_sequences(subject, load_residuals_regression=True,sliding_window=True)
    SVM_funcs.GAT_SVM_trained_separate_sequences(subject, load_residuals_regression=True,sliding_window=True)

def SVM_GAT_all_sequences(subject):
    SVM_funcs.GAT_SVM(subject, load_residuals_regression=True,sliding_window=True)

def SVM_full_sequences_16items1(subject):
    # subject = config.subjects_list[0]
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject, times=[0.130, 0.210], window=True, sliding_window=True,cleaned=True)
def SVM_full_sequences_16items2(subject):
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject, times=[0.210, 0.410], window=True, sliding_window=True,cleaned=True)
def SVM_full_sequences_16items3(subject):
    SVM_funcs.apply_SVM_filter_16_items_epochs_habituation(subject, times=[0.130, 0.210], window=True, sliding_window=True,cleaned=True)
def SVM_full_sequences_16items4(subject):
    SVM_funcs.apply_SVM_filter_16_items_epochs_habituation(subject, times=[0.210, 0.410], window=True, sliding_window=True,cleaned=True)


def SVM_full_sequences_16itemsX(subject):
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject, times=[0.120, 0.190], window=True, sliding_window=True,cleaned=False)
def SVM_full_sequences_16itemsY(subject):
    SVM_funcs.apply_SVM_filter_16_items_epochs_habituation(subject, times=[0.120, 0.190], window=True, sliding_window=True,cleaned=False)


def epoching_ARglobal(subject):
    epoching_funcs.run_epochs(subject,epoch_on_first_element=False,baseline=False)
    epoching_funcs.run_epochs(subject,epoch_on_first_element=True,baseline=False)

# ======================================================================================================================
# =====================================  FEATURES DECODING =============================================================
# ======================================================================================================================
# ---- stimulus ID ------
def SVM_features_stimID(subject,load_residuals_regression=True,cross_validation = None):
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'StimID', load_residuals_regression=load_residuals_regression,
                                 list_sequences=[3,4,5,6,7], cross_val_func=cross_validation)
# ---- repetition or alternation ------
def SVM_features_repeatalter(subject,load_residuals_regression=False,cross_validation = None):
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'RepeatAlter', load_residuals_regression=load_residuals_regression,
                                 list_sequences=[3,4,5,6,7], cross_val_func=cross_validation)
# ---- ordinal position ------
def SVM_features_withinchunk(subject, load_residuals_regression=False, cross_validation=None):
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'WithinChunkPosition',
                                           load_residuals_regression=load_residuals_regression,
                                           list_sequences=[4, 5, 6], cross_val_func=cross_validation,nvalues_feature=4)

# ----- ordinal position focus on quads ----
def SVM_quad_ordpos(subject,cleaned=True):
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'WithinChunkPosition', load_residuals_regression=False,
                                 list_sequences=[4], cross_val_func=None,filter_from_metadata="StimPosition > 2 and StimPosition < 15",nvalues_feature=4,clean=cleaned)


# ----- ordinal position focus on quads ----
# def SVM_features_withinchunk_train_quads_test_others(subject,load_residuals_regression=True):
#
#     SVM_funcs.SVM_feature_decoding_wrapper(subject, 'WithinChunkPosition', load_residuals_regression=load_residuals_regression,
#                                            filter_from_metadata="StimPosition > 2 and StimPosition < 15",cross_val_func=SVM_funcs.train_quads_test_others,nvalues_feature=4)

# ----- quelles séquences ? ----
def SVM_features_number_ofOpenedChunks(subject,load_residuals_regression=False,cleaned=True):
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'OpenedChunks',SVM_dec =SVM_funcs.regression_decoder(),balance_features=False,distance=False,  load_residuals_regression=load_residuals_regression,
                                           cross_val_func=None,list_sequences=[3,4,5,6,7])

# ----- quelles séquences pour chunk opening ? ----
def SVM_features_chunkBeg(subject,load_residuals_regression=False,cleaned=True):
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'ChunkBeginning',load_residuals_regression=load_residuals_regression,
                                           cross_val_func=None,list_sequences=[3,4,5,6,7])


# ----- quelles séquences pour chunk closing ? ----
def SVM_features_chunkEnd(subject,load_residuals_regression=False,cleaned=True):

    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'ChunkEnd',load_residuals_regression=load_residuals_regression,
                                           cross_val_func=None,list_sequences=[3,4,5,6,7])


# ----- quelles séquences pour chunk closing ? ----
def SVM_features_sequence_structure1(subject,load_residuals_regression=False,cleaned=True):
    cross_val_func = SVM_funcs.leave_one_sequence_out
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'ChunkBeginning',load_residuals_regression=load_residuals_regression,
                                           cross_val_func=cross_val_func,list_sequences=[3,4,5,6])

def SVM_features_sequence_structure2(subject, load_residuals_regression=False, cleaned=True):
    cross_val_func = SVM_funcs.leave_one_sequence_out
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'ChunkEnd',load_residuals_regression=load_residuals_regression,
                                           cross_val_func=cross_val_func,list_sequences=[3,4,5,6])

def SVM_features_sequence_structure3(subject, load_residuals_regression=False, cleaned=True):
    cross_val_func = SVM_funcs.leave_one_sequence_out
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'WithinChunkPosition',
                                           load_residuals_regression=load_residuals_regression,
                                           list_sequences=[4, 5, 6], cross_val_func=cross_val_func,nvalues_feature=4)

def SVM_features_sequence_structure4(subject, load_residuals_regression=False, cleaned=True):
    cross_val_func = SVM_funcs.leave_one_sequence_out
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'RepeatAlter', load_residuals_regression=load_residuals_regression,
                                 list_sequences=[3,4,5,6], cross_val_func=cross_val_func)

def SVM_features_sequence_structure5(subject, load_residuals_regression=False, cleaned=True):
    cross_val_func = SVM_funcs.leave_one_sequence_out
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'OpenedChunks',load_residuals_regression=load_residuals_regression,
                                           cross_val_func=cross_val_func,list_sequences=[3,4,5,6],nvalues_feature=4,SVM_dec=SVM_funcs.regression_decoder(),balance_features=False,distance=False,clean = cleaned)

def SVM_features_sequence_structure6(subject, load_residuals_regression=False, cleaned=True):
    cross_val_func = SVM_funcs.leave_one_sequence_out
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'ClosedChunks',load_residuals_regression=load_residuals_regression,
                                           cross_val_func=cross_val_func,list_sequences=[3,4,5,6],nvalues_feature=4,SVM_dec=SVM_funcs.regression_decoder(),balance_features=False,distance=False,clean = cleaned)

def SVM_features_sequence_structure7(subject, load_residuals_regression=False, cleaned=True):
    cross_val_func = SVM_funcs.leave_one_sequence_out
    SVM_funcs.SVM_feature_decoding_wrapper(subject, 'ChunkDepth',load_residuals_regression=load_residuals_regression,
                                           cross_val_func=cross_val_func,list_sequences=[3,4,5,6],nvalues_feature=4,SVM_dec=SVM_funcs.regression_decoder(),balance_features=False,distance=False,clean = cleaned)



def ord_code_16items(subject,load_residuals_regression=False):
    # SVM_funcs.SVM_ordinal_code_train_quads_test_others(subject, load_residuals_regression=load_residuals_regression)
    SVM_funcs.SVM_ordinal_code_train_test_quads(subject)

# ----------------------------------------------------------------------------------------------------------------------
#                            LINEAR REGRESSIONS
# ----------------------------------------------------------------------------------------------------------------------



def linear_reg(subject):
    from ABseq_func import regression_funcs
    config.noEEG = True
    filter_names = ['Hab','Stand','Viol']
    for filter_name in filter_names:
        regression_funcs.compute_regression(subject, ['Intercept','Complexity'], "", filter_name, remap_channels='mag_to_grad')
        # regression_funcs.compute_regression(subject, ['Intercept', 'surprise_100', 'Surprisenp1', 'RepeatAlter',
        #                                               'RepeatAlternp1'], "", filter_name, remap_channels='mag_to_grad')

        # regression_funcs.compute_regression(subject,['Intercept','surprise_100','Surprisenp1','RepeatAlter','RepeatAlternp1'],"",filter_name,remap_grads=True)
        # regression_funcs.compute_regression(subject, ['Complexity','WithinChunkPosition','ChunkBeginning', 'ChunkEnd', 'ChunkNumber', 'ChunkDepth','OpenedChunks'],"/Intercept_surprise_100_Surprisenp1_RepeatAlter_RepeatAlternp1/" + subject +"/residuals--remapped_clean-epo.fif", filter_name,
        #                                     remap_channels='mag_to_grad')

# ----------------------------------------------------------------------------------------------------------------------
#                                   RSA
# ----------------------------------------------------------------------------------------------------------------------

def compute_rsa_dissim_matrix(subject):
    """
    We compute the dissimilarity matrix by grouping the epochs by sequence and position (for standard sequences only)
    """
    rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'correlation', baseline=None, which_analysis='_no_baseline_all_data',clean=False,recompute=True)
    # rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'spearmanr', baseline=None, which_analysis='_no_baseline_all_data',clean=False,recompute=True)
    # rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'euclidean', baseline=None, which_analysis='_no_baseline_all_data',clean=False,recompute=True)
    # rsa_funcs.preprocess_and_compute_dissimilarity(subject, 'mahalanobis', baseline=None, which_analysis='')


def compute_correlation_stc_complexity(subject):
    from ABseq_func import stc_funcs
    stc_funcs.compute_correlation_comp_all_conditions(subject)