# This module contains all the functions that allow the computations to run on the cluster.
from __future__ import division
import initialization_paths
from ABseq_func import *
from ABseq_func import TP_funcs
import config
import subprocess
import MarkovModel_Python

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
        walltime = "36:00:00"  # "24:00:00"
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
    epoching_funcs.run_epochs(subject, epoch_on_first_element=True)


def EMS(subject):
    # EMS_funcs.generate_EMS_all_sequences(subject)
    # EMS_funcs.GAT_EMS(subject)
    # EMS_funcs.GAT_EMS_4pos(subject)
    # EMS_funcs.apply_EMS_filter_16_items_epochs(subject)
    # EMS_funcs.apply_EMS_filter_16_items_epochs_habituation(subject)
    EMS_funcs.apply_EMS_filter_16_items_epochs(subject, times=[0.140, 0.180], window=True)
    EMS_funcs.apply_EMS_filter_16_items_epochs_habituation(subject, times=[0.140, 0.180], window=True)


def SVM_analysis(subject):
    # creating the SVM results dictionnary
    SVM_funcs.generate_SVM_all_sequences(subject)
    SVM_funcs.GAT_SVM(subject)
    SVM_funcs.GAT_SVM_4pos(subject)
    SVM_funcs.apply_SVM_filter_16_items_epochs(subject, times=[0.140, 0.180], window=True)
    SVM_funcs.apply_SVM_filter_16_items_epochs_habituation(subject, times=[0.140, 0.180], window=True)


def SVM_1(subject):
    SVM_funcs.generate_SVM_all_sequences(subject)

def SVM_2(subject):
    SVM_funcs.GAT_SVM_4pos(subject)

def SVM_3(subject):
    SVM_funcs.GAT_SVM(subject)

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

    TP_funcs.from_epochs_to_surprise(subject,list_omegas)
    TP_funcs.append_surprise_to_metadata_clean(subject)
    from importlib import reload
    reload(TP_funcs)
    # TP_funcs.run_linear_regression_surprises(subject, list_omegas, clean=True, decim=50,hfilter=None)
    TP_funcs.run_linear_regression_surprises(subject, list_omegas, clean=True, decim=None,hfilter=10)

    # ----------- then we have to compute the optimal omega for each time and channel -------------
    # TP_funcs.regress_out_optimal_omega(subject, clean=True)
    # TP_funcs.compute_posterior_probability(subject)
    # TP_funcs.regress_out_optimal_omega_per_channel(subject)

def simplified_linear_regression(subject):
    from ABseq_func import linear_reg_funcs
    linear_reg_funcs.run_linear_reg_surprise_repeat_alt(subject)


def simplified_with_complexity(subject):
    from ABseq_func import linear_reg_funcs
    linear_reg_funcs.run_linear_reg_surprise_repeat_alt(subject,with_complexity=True)
