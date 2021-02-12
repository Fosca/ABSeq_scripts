"""
===========
Config file
===========

Configuration parameters for the study.
"""

import os
from collections import defaultdict
import numpy as np
from sys import platform

# ``plot``  : boolean
#   If True, the scripts will generate plots.
#   If running the scripts from a notebook or spyder
#   run %matplotlib qt in the command line to get the plots in extra windows

plot = False

# ``tcrop``  : float
#   When you load the epochs on the sequence items, if tcrop is not None, the epochs will be cropped with tmax = tcrop
tcrop = 0.5

###############################################################################
# DIRECTORIES
# -----------
##    Set the `study path`` where the data is stored on your system.
#
# Example
# ~~~~~~~
# >>> study_path = '../MNE-sample-data/'
# or
# >>> study_path = '/Users/sophie/repos/ExampleData/'

if os.name == 'nt':
    root_path = 'Z:' + os.path.sep
elif os.name == 'posix':
    if platform == "linux" or platform == "linux2":
        root_path = '/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/'
    elif platform == "darwin":
        root_path = '//neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/'

study_path = os.path.join(root_path, 'data') + os.path.sep
result_path = os.path.join(root_path, 'results') + os.path.sep
cluster_path = os.path.join(root_path, 'scripts', 'ABSeq_scripts', 'cluster') + os.path.sep
scripts_path = os.path.join(root_path, 'scripts', 'ABSeq_scripts') + os.path.sep

EMS_path = os.path.join(result_path, 'EMS') + os.path.sep
SVM_path = os.path.join(result_path, 'SVM') + os.path.sep
decoding_path = os.path.join(result_path, 'decoding') + os.path.sep
GFP_path = os.path.join(result_path, 'GFP') + os.path.sep
linear_models_path = os.path.join(result_path, 'linear_models') + os.path.sep

# ``subjects_dir`` : str
#   The ``subjects_dir`` contains the MRI files for all subjects.

subjects_dir = os.path.join(study_path, 'subjects')
fig_path = os.path.join(root_path, 'figures')
local_fig_path = os.path.join("/Users/fosca/Desktop/coucou/")
# ``meg_dir`` : str
#   The ``meg_dir`` contains the MEG data in subfolders
#   named my_study_path/MEG/my_subject/

meg_dir = os.path.join(study_path, 'MEG')
run_info_dir = os.path.join(study_path, 'run_info')

###############################################################################
# SUBJECTS / RUNS
# ---------------
#
# ``study_name`` : str
#   This is the name of your experiment.
#
# Example
# ~~~~~~~
# >>> study_name = 'MNE-sample'
study_name = 'ABseq'

# ``subjects_list`` : list of str
#   To define the list of participants, we use a list with all the anonymized
#   participant names. Even if you plan on analyzing a single participant, it
#   needs to be set up as a list with a single element, as in the 'example'
#   subjects_list = ['SB01']

subjects_list = ['sub01-pa_190002', 'sub02-ch_180036', 'sub03-mr_190273', 'sub04-rf_190499', 'sub05-cr_170417', 'sub06-kc_160388',
                 'sub07-jm_100109', 'sub08-cc_150418', 'sub09-ag_170045', 'sub10-gp_190568', 'sub11-fr_190151', 'sub12-lg_170436',
                 'sub13-lq_180242', 'sub14-js_180232', 'sub15-ev_070110', 'sub16-ma_190185', 'sub17-mt_170249', 'sub18-eo_190576',
                 'sub19-mg_190180']

# ``exclude_subjects`` : list of str
#   Now you can specify subjects to exclude from the group study:
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Keep track of the criteria leading you to exclude
# a participant (e.g. too many movements, missing blocks, aborted experiment,
# did not understand the instructions, etc, ...)

exclude_subjects = ['sub04-rf_190499', 'sub08-cc_150418']
# sub04 & sub08: very bad EEG data


# ``runs`` : list of str
#   Define the names of your ``runs``
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# The naming should be consistent across participants. List the number of runs
# you ideally expect to have per participant. The scripts will issue a warning
# if there are less runs than is expected. If there is only just one file,
# leave empty!


runs = ['run01', 'run02', 'run03', 'run04', 'run05', 'run06', 'run07',
        'run08', 'run09', 'run10', 'run11', 'run12', 'run13', 'run14']

runs_dict = {subject: runs for subject in subjects_list}

runs_dict['sub03-mr_190273'] = ['run01', 'run02', 'run03', 'run04', 'run05', 'run06', 'run07',
                                'run08', 'run09', 'run10', 'run11', 'run12']  # importation error from MEG machine (unreadable files)
runs_dict['sub07-jm_100109'] = ['run01', 'run02', 'run03', 'run04', 'run06', 'run07',
                                'run08', 'run09', 'run10', 'run11', 'run12', 'run13', 'run14']  # skipped a run during acquisition
# runs_dict['sub14-js_180232'] = [         'run02', 'run03', 'run04', 'run05', 'run06', 'run07',
#                                 'run08', 'run09', 'run10', 'run11', 'run12', 'run13', 'run14'] # (audio)triggers too much amplified (too many detected) /// CORRECTED


# ``ch_types``  : list of st
#    The list of channel types to consider.
#
# Example
# ~~~~~~~
# >>> ch_types = ['meg', 'eeg']  # to use MEG and EEG channels
# or
# >>> ch_types = ['meg']  # to use only MEG
# or
# >>> ch_types = ['grad']  # to use only gradiometer MEG channels

ch_types = ['meg', 'eeg']

# ``base_fname`` : str
#    This automatically generates the name for all files
#    with the variables specified above.
#    Normally you should not have to touch this

base_fname = '{extension}.fif'


###############################################################################
# BAD CHANNELS
# ------------
# needed for 01-import_and_filter.py

# ``bads`` : dict of list | dict of dict
#    Bad channels are noisy sensors that *must* to be listed
#    *before* maxfilter is applied. You can use the dict of list structure
#    of you have bad channels that are the same for all runs.
#    Use the dict(dict) if you have many runs or if noisy sensors are changing
#    across runs.
#
# Example
# ~~~~~~~
# >>> bads = defaultdict(list)
# >>> bads['sample'] = ['MEG 2443', 'EEG 053']  # 2 bads channels
# or
# >>> def default_bads():
# >>>     return dict(run01=[], run02=[])
# >>>
# >>> bads = defaultdict(default_bads)
# >>> bads['subject01'] = dict(run01=['MEG1723', 'MEG1722'], run02=['MEG1723'])
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# During the acquisition of your MEG / EEG data, systematically list and keep
# track of the noisy sensors. Here, put the number of runs you ideally expect
# to have per participant. Use the simple dict if you don't have runs or if
# the same sensors are noisy across all runs.

# bads = defaultdict(list)
# bads['SB01'] = ['MEG1723', 'MEG1722']
# bads['SB04'] = ['MEG0543', 'MEG2333']
# bads['SB06'] = ['MEG2632', 'MEG2033']

def default_bads():
    return {name: [] for name in runs}


bads = defaultdict(default_bads)

bads['sub01-pa_190002'] = dict(
    run01=['MEG0213', 'MEG1323', 'MEG2233', 'MEG1732', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG046'],
    run02=['MEG0213', 'MEG1323', 'MEG2233', 'MEG1732', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG046', 'EEG054'],
    run03=['MEG0213', 'MEG1323', 'MEG2233', 'MEG1732', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG046', 'EEG037'],
    run04=['MEG0213', 'MEG1323', 'MEG2233', 'MEG1732', 'MEG0741', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG046', 'EEG054'],
    run05=['MEG0213', 'MEG1732', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG046', 'EEG054'],
    run06=['MEG0213', 'MEG1732', 'EEG025', 'EEG041'],
    run07=['MEG0213', 'MEG1732', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG037'],
    run08=['MEG0213', 'MEG0242', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG033', 'EEG037', 'EEG027'],
    run09=['MEG0213', 'MEG1732', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG033', 'EEG037', 'EEG027'],
    run10=['MEG0213', 'MEG1732', 'MEG2132', 'EEG025', 'EEG041', 'EEG007', 'EEG015'],
    run11=['MEG0213', 'MEG1732', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG037', 'EEG033', 'EEG027'],
    run12=['MEG0213', 'MEG1732', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG037'],
    run13=['MEG0213', 'MEG1732', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG037', 'EEG033', 'EEG027'],
    run14=['MEG0213', 'MEG1732', 'MEG1332', 'MEG0242', 'EEG025', 'EEG041', 'EEG034', 'EEG026', 'EEG037', 'EEG033', 'EEG027'])

bads['sub02-ch_180036'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035', 'EEG002', 'EEG017', 'EEG018'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG0632', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035', 'EEG017', 'EEG018'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035', 'EEG002', 'EEG017', 'EEG018'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035', 'EEG017', 'EEG018'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035', 'EEG017', 'EEG018'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035', 'EEG017', 'EEG018'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG1241', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG0632', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035', 'EEG004'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG0632', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035', 'EEG004', 'EEG001'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2321', 'MEG2113', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035', 'EEG001'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2321', 'MEG1241', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'EEG045', 'EEG053', 'EEG041', 'EEG038', 'EEG035'])

bads['sub03-mr_190273'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'EEG035', 'EEG041', 'EEG007', 'EEG015'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'MEG0541', 'MEG0131', 'EEG035', 'EEG041', 'EEG007', 'EEG004'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'EEG035', 'EEG041', 'EEG007'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'MEG1831', 'EEG035', 'EEG041', 'EEG034'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'EEG041', 'EEG035', 'EEG007', 'EEG004'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'EEG041', 'EEG035', 'EEG007'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'EEG041', 'EEG035', 'EEG007', 'EEG004'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'EEG035', 'EEG041', 'EEG007', 'EEG004', 'EEG015'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'EEG035', 'EEG041', 'EEG007'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'EEG035', 'EEG041', 'EEG007'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'MEG1831', 'EEG035', 'EEG034', 'EEG041', 'EEG007'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523', 'MEG1831', 'EEG035', 'EEG034', 'EEG041', 'EEG007'])
# run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'],
# run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1523'])

bads['sub04-rf_190499'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2243', 'EEG017', 'EEG041', 'EEG024', 'EEG014', 'EEG006'],  #
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2243', 'EEG017', 'EEG041', 'EEG024', 'EEG014', 'EEG006', 'EEG032'],  #
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'EEG017', 'EEG041', 'EEG024', 'EEG018', 'EEG032'],  #
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2241', 'EEG017', 'EEG041', 'EEG024', 'EEG045', 'EEG044', 'EEG032'],  #
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1943', 'MEG1941', 'EEG017', 'EEG024', 'EEG039', 'EEG047', 'EEG037', 'EEG038', 'EEG041'],  #
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1943', 'MEG2241', 'EEG017', 'EEG024', 'EEG039', 'EEG047', 'EEG037', 'EEG041'],  #
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'EEG017', 'EEG041', 'EEG024'],  #
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1941', 'EEG017', 'EEG018', 'EEG019', 'EEG041', 'EEG022', 'EEG023', 'EEG024', 'EEG027'],  #
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1942', 'EEG017', 'EEG041', 'EEG024', 'EEG048', 'EEG024'],  #
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1941', 'EEG017', 'EEG041', 'EEG024'],  #
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1941', 'EEG017', 'EEG041', 'EEG024'],  #
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2321', 'EEG017', 'EEG041', 'EEG024', 'EEG039', 'EEG047', 'EEG045'],  #
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1942', 'MEG2311', 'EEG017', 'EEG041', 'EEG024', 'EEG039', 'EEG029'],  #
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1443', 'MEG1942', 'MEG1941', 'EEG017', 'EEG041', 'EEG024', 'EEG019'])  #

bads['sub05-cr_170417'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG025', 'EEG043', 'EEG041', 'EEG035', 'EEG017', 'EEG036', 'EEG003', 'EEG007', 'EEG001'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'MEG0522', 'EEG043', 'EEG041', 'EEG017', 'EEG036', 'EEG025', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG025', 'EEG043', 'EEG041', 'EEG035', 'EEG017', 'EEG036', 'EEG025', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG043', 'EEG041', 'EEG017', 'EEG036', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG043', 'EEG041', 'EEG036', 'EEG035', 'EEG001', 'EEG007', 'EEG017', 'EEG003', 'EEG025', 'EEG002', 'EEG003', 'EEG007'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG043', 'EEG041', 'EEG037', 'EEG044', 'EEG025', 'EEG017', 'EEG035', 'EEG036', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG043', 'EEG041', 'EEG025', 'EEG017', 'EEG035', 'EEG036', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG043', 'EEG041', 'EEG025', 'EEG017', 'EEG035', 'EEG036', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG043', 'EEG041', 'EEG035', 'EEG036', 'EEG017', 'EEG025', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'EEG043', 'EEG041', 'EEG035', 'EEG036', 'EEG017', 'EEG025', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'EEG043', 'EEG041', 'EEG035', 'EEG017', 'EEG025', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'EEG043', 'EEG041', 'EEG035', 'EEG036', 'EEG017', 'EEG025', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG043', 'EEG041', 'EEG035', 'EEG036', 'EEG017', 'EEG025', 'EEG001', 'EEG002', 'EEG003', 'EEG007'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0131', 'EEG043', 'EEG041', 'EEG035', 'EEG036', 'EEG017', 'EEG025', 'EEG001', 'EEG002', 'EEG003', 'EEG007'])

bads['sub06-kc_160388'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1831', 'MEG2221', 'MEG0313', 'MEG2313', 'EEG034', 'EEG043', 'EEG060', 'EEG041'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1831', 'MEG2221', 'MEG0313', 'MEG2313', 'EEG034', 'EEG043', 'EEG060', 'EEG041'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0132', 'EEG034', 'EEG043', 'EEG060', 'EEG041'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'EEG034', 'EEG043', 'EEG060', 'EEG041', 'EEG026', 'EEG042'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'EEG034', 'EEG043', 'EEG060', 'EEG041'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'EEG034', 'EEG043', 'EEG060', 'EEG041'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'EEG043', 'EEG060', 'EEG041'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'EEG034', 'EEG043', 'EEG060', 'EEG041'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG2113', 'EEG043', 'EEG060', 'EEG041'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'EEG034', 'EEG043', 'EEG060', 'EEG041', 'EEG042'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0111', 'MEG0321', 'EEG034', 'EEG043', 'EEG041', 'EEG060'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG0321', 'MEG0111', 'EEG034', 'EEG043', 'EEG060', 'EEG041'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG0321', 'MEG0111', 'EEG034', 'EEG043', 'EEG060', 'EEG041'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG0321', 'MEG0111', 'EEG034', 'EEG043', 'EEG060', 'EEG041', 'EEG026'])

bads['sub07-jm_100109'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026'],
    # run05=['???'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026', 'EEG034', 'EEG041'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026', 'EEG041'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026', 'EEG041'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026', 'EEG041'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026', 'EEG041'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026', 'EEG041'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2231', 'EEG043', 'EEG035', 'EEG017', 'EEG026', 'EEG041'])

bads['sub08-cc_150418'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2221', 'MEG1821', 'MEG1522', 'MEG1822', 'MEG1813', 'MEG2222', 'MEG2223', 'EEG017', 'EEG025', 'EEG053', 'EEG001'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG042', 'EEG043', 'EEG026'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG036'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG036', 'EEG042', 'EEG043'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG036'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG036'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG036'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG036', 'EEG042', 'EEG043', 'EEG026'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG026', 'EEG036', 'EEG045'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'MEG1823', 'MEG0121', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG045', 'EEG001', 'EEG036'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'MEG1823', 'MEG0121', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG045', 'EEG042', 'EEG043'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1733', 'MEG1522', 'MEG1823', 'MEG0121', 'EEG017', 'EEG025', 'EEG053', 'EEG001', 'EEG045'])
# For this participant, we had some problems when concatenating the raws for run08. The error message said that raw08._cals didn't match the other ones.
# We saw that it is the 'calibration' for the channel EOG061 that was different with respect to run09._cals.
# np.where(raw_list[7]._cals-raw_list[8]._cals)
# raw_list[7].info['ch_names'][382]
# We replaced by hand run08._cals by run09._cals and saved it.
# raw_list[7]._cals = raw_list[8]._cals
# raw_list[7].save('Z:\\data\\MEG\\sub08-cc_150418\\run08_ica_raw.fif',overwrite=True)

bads['sub09-ag_170045'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG0813', 'MEG2642', 'MEG1731', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG041'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG041'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG041'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG041'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG041'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG041'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG041'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG041'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG041'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG045', 'EEG041'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG1731', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG045'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'MEG2642', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG045'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG045', 'EEG041'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0923', 'MEG1733', 'EEG053', 'EEG035', 'EEG025', 'EEG036', 'EEG038', 'EEG024', 'EEG045', 'EEG041'])

bads['sub10-gp_190568'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG0522', 'MEG2641', 'MEG1111', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG0522', 'MEG1111', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041', 'EEG026'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG1111', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG0522', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2323', 'MEG2422', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG2242', 'MEG0522', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041', 'EEG001'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1333', 'MEG1332', 'MEG0522', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2423', 'MEG1612', 'MEG0122', 'EEG053', 'EEG025', 'EEG024', 'EEG041'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'EEG053', 'EEG025', 'EEG024', 'EEG041'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1612', 'MEG0522', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041', 'EEG026'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2412', 'MEG0522', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2012', 'MEG0522', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2423', 'MEG0522', 'EEG053', 'EEG002', 'EEG025', 'EEG024', 'EEG041', 'EEG001'])

bads['sub11-fr_190151'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'EEG025', 'EEG035', 'EEG037', 'EEG041', 'EEG008'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'EEG025', 'EEG035', 'EEG037', 'EEG041'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'EEG025', 'EEG035', 'EEG037', 'EEG041', 'EEG008'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'EEG025', 'EEG035', 'EEG037', 'EEG041'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'EEG025', 'EEG035', 'EEG037', 'EEG041'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'EEG025', 'EEG035', 'EEG037', 'EEG041'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG1831', 'MEG0121', 'MEG1841', 'EEG025', 'EEG035', 'EEG037', 'EEG041', 'EEG007', 'EEG008', 'EEG015', 'EEG016', 'EEG022', 'EEG023'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2423', 'MEG2422', 'EEG025', 'EEG035', 'EEG037', 'EEG041'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2422', 'EEG025', 'EEG035', 'EEG037', 'EEG041', 'EEG008', 'EEG015', 'EEG022'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'EEG025', 'EEG035', 'EEG037', 'EEG041'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG1841', 'MEG2421', 'MEG2422', 'EEG025', 'EEG035', 'EEG037', 'EEG041'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2421', 'MEG2422', 'EEG025', 'EEG035', 'EEG037', 'EEG041'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2422', 'EEG025', 'EEG035', 'EEG037', 'EEG041', 'EEG007', 'EEG008', 'EEG015', 'EEG016', 'EEG022', 'EEG023', 'EEG033', 'EEG034'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0522', 'MEG2422', 'MEG1931', 'EEG025', 'EEG035', 'EEG037', 'EEG041', 'EEG015', 'EEG016', 'EEG022', 'EEG007'])

bads['sub12-lg_170436'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523', 'EEG025', 'EEG036', 'EEG001', 'EEG024', 'EEG041', 'EEG053'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG2113', 'MEG1831', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG037'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG037'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG0522', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG004', 'EEG010'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG2113', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG053', 'EEG004', 'EEG010'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG037', 'EEG004', 'EEG010', 'EEG008', 'EEG009'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG004', 'EEG010', 'EEG008', 'EEG009'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG004', 'EEG010', 'EEG008', 'EEG009', 'EEG037'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG004', 'EEG010', 'EEG008', 'EEG009', 'EEG037'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG004', 'EEG010'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523', 'MEG1613', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG004', 'EEG010', 'EEG008', 'EEG009', 'EEG037'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG053', 'EEG004', 'EEG010'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1522', 'MEG1523', 'MEG1613', 'EEG025', 'EEG001', 'EEG024', 'EEG036', 'EEG041', 'EEG037', 'EEG004', 'EEG010', 'EEG008', 'EEG009'])

bads['sub13-lq_180242'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'EEG025', 'EEG001', 'EEG002', 'EEG026', 'EEG004', 'EEG007'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'EEG025', 'EEG001', 'EEG002', 'EEG026', 'EEG004', 'EEG007', 'EEG009'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG026', 'EEG004', 'EEG007'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG026', 'EEG004'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG026', 'EEG004'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG026', 'EEG004'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2221', 'MEG0141', 'MEG1221', 'EEG025', 'EEG001', 'EEG024', 'EEG002', 'EEG041', 'EEG036', 'EEG026'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0141', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG026', 'EEG004'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0141', 'MEG1521', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG024', 'EEG026', 'EEG004'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1541', 'MEG1221', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG024', 'EEG026', 'EEG004'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1221', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG024', 'EEG026', 'EEG004'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG024', 'EEG026', 'EEG004'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1221', 'MEG1442', 'MEG0613', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG024', 'EEG026', 'EEG004', 'EEG007', 'EEG009'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0121', 'MEG1442', 'EEG025', 'EEG001', 'EEG002', 'EEG041', 'EEG036', 'EEG024', 'EEG026', 'EEG004', 'EEG007'])

bads['sub14-js_180232'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1222', 'MEG2511', 'EEG003', 'EEG057', 'EEG032', 'EEG041', 'EEG040', 'EEG042', 'EEG043'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1222', 'MEG2511', 'EEG003', 'EEG057', 'EEG032', 'EEG041', 'EEG040', 'EEG042', 'EEG043'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG032', 'EEG041', 'EEG040', 'EEG042', 'EEG043', 'EEG039'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG032', 'EEG041', 'EEG040', 'EEG042', 'EEG039'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1222', 'EEG003', 'EEG057', 'EEG032', 'EEG041', 'EEG040', 'EEG042'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG032', 'EEG041', 'EEG040', 'EEG042'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG030', 'EEG031', 'EEG032', 'EEG041', 'EEG040', 'EEG042'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG032', 'EEG041', 'EEG040', 'EEG042'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG029', 'EEG031', 'EEG032', 'EEG041', 'EEG040', 'EEG042'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG030', 'EEG031', 'EEG032', 'EEG041', 'EEG040', 'EEG042'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG030', 'EEG031', 'EEG032', 'EEG041', 'EEG040', 'EEG042'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG030', 'EEG029', 'EEG032', 'EEG041', 'EEG040', 'EEG042'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG030', 'EEG029', 'EEG032', 'EEG041'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'EEG003', 'EEG057', 'EEG030', 'EEG029', 'EEG032', 'EEG041'])

bads['sub15-ev_070110'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0341', 'MEG2211', 'MEG2512', 'EEG035', 'EEG041', 'EEG037', 'EEG003', 'EEG001', 'EEG004'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2211', 'MEG2512', 'EEG035', 'EEG041', 'EEG037', 'EEG003', 'EEG001', 'EEG007', 'EEG004'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2211', 'MEG2512', 'MEG2511', 'EEG035', 'EEG041', 'EEG037', 'EEG003', 'EEG001', 'EEG007', 'EEG004'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2211', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG2512', 'MEG2642', 'EEG035', 'EEG041'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2641', 'MEG2641', 'MEG2642', 'MEG2523', 'MEG2522', 'EEG035', 'EEG041', 'EEG004'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512', 'MEG2511', 'EEG035', 'EEG041', 'EEG002'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512', 'EEG035', 'EEG041', 'EEG037', 'EEG003', 'EEG001', 'EEG004'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512', 'EEG035', 'EEG041', 'EEG042'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512', 'EEG035', 'EEG041', 'EEG022', 'EEG003', 'EEG033'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'EEG035', 'EEG041', 'EEG037', 'EEG003', 'EEG004'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512', 'EEG035', 'EEG041', 'EEG037', 'EEG003', 'EEG001', 'EEG007'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2512', 'MEG1122', 'EEG035', 'EEG041', 'EEG022', 'EEG042', 'EEG033'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2323', 'MEG0522', 'MEG1721', 'EEG035', 'EEG041', 'EEG037', 'EEG003', 'EEG007', 'EEG004'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2323', 'MEG0523', 'MEG1721', 'EEG035', 'EEG041', 'EEG037', 'EEG003', 'EEG007', 'EEG027'])

bads['sub16-ma_190185'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'EEG034', 'EEG017', 'EEG001', 'EEG025'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'EEG041', 'EEG034', 'EEG001', 'EEG035'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'EEG041', 'EEG017', 'EEG001', 'EEG034', 'EEG035'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'EEG041', 'EEG017', 'EEG001', 'EEG034'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0332', 'EEG041', 'EEG017', 'EEG001'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2241', 'MEG0541', 'MEG0821', 'EEG041', 'EEG017', 'EEG001'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0821', 'EEG041', 'EEG017', 'EEG001', 'EEG034', 'EEG053'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0813', 'MEG0321', 'EEG041', 'EEG017', 'EEG001', 'EEG025'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0813', 'EEG041', 'EEG017', 'EEG001', 'EEG053', 'EEG034'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0813', 'EEG041', 'EEG017', 'EEG001', 'EEG034'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'EEG041', 'EEG001', 'EEG025', 'EEG034'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'EEG041', 'EEG001'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'EEG041', 'EEG001', 'EEG034', 'EEG004'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'EEG041', 'EEG001', 'EEG034', 'EEG004'])

bads['sub17-mt_170249'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0513', 'EEG001', 'EEG002', 'EEG041', 'EEG035'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2642', 'EEG001', 'EEG002', 'EEG053', 'EEG024', 'EEG025', 'EEG041', 'EEG035'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2642', 'MEG0813', 'EEG001', 'EEG002', 'EEG025', 'EEG053', 'EEG024', 'EEG041', 'EEG035'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG2642', 'MEG0443', 'MEG1731', 'EEG001', 'EEG002', 'EEG025', 'EEG024', 'EEG035', 'EEG053', 'EEG041'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'EEG001', 'EEG002', 'EEG025', 'EEG024', 'EEG035', 'EEG053', 'EEG041'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'EEG001', 'EEG002', 'EEG025', 'EEG024', 'EEG035', 'EEG053', 'EEG041'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'EEG001', 'EEG002', 'EEG025', 'EEG024', 'EEG035', 'EEG053', 'EEG041'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1641', 'EEG001', 'EEG002', 'EEG025', 'EEG024', 'EEG035', 'EEG053', 'EEG041'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'EEG001', 'EEG002', 'EEG025', 'EEG024', 'EEG035', 'EEG053', 'EEG041'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'EEG001', 'EEG035', 'EEG025', 'EEG041'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'EEG001', 'EEG035', 'EEG025', 'EEG041'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'EEG001', 'EEG035', 'EEG025', 'EEG041'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'EEG001', 'EEG035', 'EEG025', 'EEG041'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'EEG001', 'EEG035', 'EEG025', 'EEG041', 'EEG053', 'EEG024'])

bads['sub18-eo_190576'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG041'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041', 'EEG022'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'MEG0113', 'MEG0143', 'MEG0111', 'MEG0141', 'EEG035', 'EEG034', 'EEG036', 'EEG043', 'EEG041'])

bads['sub19-mg_190180'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run02=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run03=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run04=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run05=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run06=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041', 'EEG024'],
    run07=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run08=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run09=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run10=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run11=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run12=['MEG0213', 'MEG0311', 'MEG2643', 'MEG1133', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run13=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'],
    run14=['MEG0213', 'MEG0311', 'MEG2643', 'EEG049', 'EEG002', 'EEG003', 'EEG029', 'EEG030', 'EEG031', 'EEG032', 'EEG049', 'EEG057', 'EEG041'], )

bads['subXXXX'] = dict(
    run01=['MEG0213', 'MEG0311', 'MEG2643'],
    run02=['MEG0213', 'MEG0311', 'MEG2643'],
    run03=['MEG0213', 'MEG0311', 'MEG2643'],
    run04=['MEG0213', 'MEG0311', 'MEG2643'],
    run05=['MEG0213', 'MEG0311', 'MEG2643'],
    run06=['MEG0213', 'MEG0311', 'MEG2643'],
    run07=['MEG0213', 'MEG0311', 'MEG2643'],
    run08=['MEG0213', 'MEG0311', 'MEG2643'],
    run09=['MEG0213', 'MEG0311', 'MEG2643'],
    run10=['MEG0213', 'MEG0311', 'MEG2643'],
    run11=['MEG0213', 'MEG0311', 'MEG2643'],
    run12=['MEG0213', 'MEG0311', 'MEG2643'],
    run13=['MEG0213', 'MEG0311', 'MEG2643'],
    run14=['MEG0213', 'MEG0311', 'MEG2643'])

###############################################################################
# DEFINE ADDITIONAL CHANNELS
# --------------------------
# needed for 01-import_and_filter.py

# ``rename_channels`` : dict rename channels
#    Here you name or replace extra channels that were recorded, for instance
#    EOG, ECG.
#
# Example
# ~~~~~~~
# Here rename EEG061 to EOG061, EEG062 to EOG062, EEG063 to ECG063:
# >>> rename_channels = {'EEG061': 'EOG061', 'EEG062': 'EOG062',
#                        'EEG063': 'ECG063'}

rename_channels = None

# ``set_channel_types``: dict
#   Here you define types of channels to pick later.
#
# Example
# ~~~~~~~
# >>> set_channel_types = {'EEG061': 'eog', 'EEG062': 'eog',
#                          'EEG063': 'ecg', 'EEG064': 'misc'}

set_channel_types = None

###############################################################################
# FREQUENCY FILTERING
# -------------------
# done in 01-import_and_filter.py

# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# It is typically better to set your filtering properties on the raw data so
# as to avoid what we call border (or edge) effects.
#
# If you use this pipeline for evoked responses, you could consider
# a low-pass filter cut-off of h_freq = 40 Hz
# and possibly a high-pass filter cut-off of l_freq = 1 Hz
# so you would preserve only the power in the 1Hz to 40 Hz band.
# Note that highpass filtering is not necessarily recommended as it can
# distort waveforms of evoked components, or simply wash out any low
# frequency that can may contain brain signal. It can also act as
# a replacement for baseline correction in Epochs. See below.
#
# If you use this pipeline for time-frequency analysis, a default filtering
# could be a high-pass filter cut-off of l_freq = 1 Hz
# a low-pass filter cut-off of h_freq = 120 Hz
# so you would preserve only the power in the 1Hz to 120 Hz band.
#
# If you need more fancy analysis, you are already likely past this kind
# of tips! :)


# ``l_freq`` : float
#   The low-frequency cut-off in the highpass filtering step.
#   Keep it None if no highpass filtering should be applied.

l_freq = 0.10

# ``h_freq`` : float
#   The high-frequency cut-off in the lowpass filtering step.
#   Keep it None if no lowpass filtering should be applied.

h_freq = 40  # NEW ! (changed by SP)

###############################################################################
# MAXFILTER PARAMETERS
# --------------------
#
# ``use_maxwell_filter`` : bool
#   Use or not maxwell filter to preprocess the data.

use_maxwell_filter = True

# There are two kinds of maxfiltering: SSS and tSSS
# [SSS = signal space separation ; tSSS = temporal signal space separation]
# (Taulu et al, 2004): http://cds.cern.ch/record/709081/files/0401166.pdf
#
# ``mf_st_duration`` : float | None
#    If not None, apply spatiotemporal SSS (tSSS) with specified buffer
#    duration (in seconds). MaxFilter's default is 10.0 seconds in v2.2.
#    Spatiotemporal SSS acts as implicitly as a high-pass filter where the
#    cut-off frequency is 1/st_dur Hz. For this (and other) reasons, longer
#    buffers are generally better as long as your system can handle the
#    higher memory usage. To ensure that each window is processed
#    identically, choose a buffer length that divides evenly into your data.
#    Any data at the trailing edge that doesn't fit evenly into a whole
#    buffer window will be lumped into the previous buffer.
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# If you are interested in low frequency activity (<0.1Hz), avoid using tSSS
# and set mf_st_duration to None
#
# If you are interested in low frequency above 0.1 Hz, you can use the
# default mf_st_duration to 10 s meaning it acts like a 0.1 Hz highpass filter.
#
# Example
# ~~~~~~~
# >>> mf_st_duration = None
# or
# >>> mf_st_duration = 10.  # to apply tSSS with 0.1Hz highpass filter.

mf_st_duration = 10.

# ``mf_head_origin`` : array-like, shape (3,) | 'auto'
#   Origin of internal and external multipolar moment space in meters.
#   If 'auto', it will be estimated from headshape points.
#   If automatic fitting fails (e.g., due to having too few digitization
#   points), consider separately calling the fitting function with different
#   options or specifying the origin manually.
#
# Example
# ~~~~~~~
# >>> mf_head_origin = 'auto'

mf_head_origin = 'auto'

# ``cross talk`` : str
#   Path to the cross talk file
#
#
# ``calibration`` : str
#   Path to the calibration file.
#
#
# These 2 files should be downloaded and made available for running
# maxwell filtering.
#
# Example
# ~~~~~~~
# >>> cal_files_path = os.path.join(study_path, 'SSS')
# >>> mf_ctc_fname = os.path.join(cal_files_path, 'ct_sparse_mgh.fif')
# >>> mf_cal_fname = os.path.join(cal_files_path, 'sss_cal_mgh.dat')
#
# Warning
# ~~~~~~~
# These 2 files are site and machine specific files that provide information
# about the environmental noise. For practical purposes, place them in your
# study folder.
#
# At NeuroSpin: ct_sparse and sss_call are on the meg_tmp server

cal_files_path = os.path.join(study_path, 'system_calibration_files')
mf_ctc_fname = os.path.join(cal_files_path, 'ct_sparse_nspn.fif')
mf_cal_fname = os.path.join(cal_files_path, 'sss_cal_nspn.dat')

# Despite all possible care to avoid movements in the MEG, the participant
# will likely slowly drift down from the Dewar or slightly shift the head
# around in the course of the recording session. Hence, to take this into
# account, we are realigning all data to a single position. For this, you need
# to define a reference run (typically the one in the middle of
# the recording session).
#
# ``mf_reference_run``  : int
#   Which run to take as the reference for adjusting the head position of all
#   runs.
#
# Example
# ~~~~~~~
# >>> mf_reference_run = 0  # to use the first run

mf_reference_run = 7

###############################################################################
# RESAMPLING
# ----------
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# If you have acquired data with a very high sampling frequency (e.g. 2 kHz)
# you will likely want to downsample to lighten up the size of the files you
# are working with (pragmatics)
# If you are interested in typical analysis (up to 120 Hz) you can typically
# resample your data down to 500 Hz without preventing reliable time-frequency
# exploration of your data
#
# ``resample_sfreq``  : float
#   Specifies at which sampling frequency the data should be resampled.
#   If None then no resampling will be done.
#
# Example
# ~~~~~~~
# >>> resample_sfreq = None  # no resampling
# or
# >>> resample_sfreq = 500  # resample to 500Hz

resample_sfreq = 250  # None

# ``decim`` : int
#   Says how much to decimate data at the epochs level.
#   It is typically an alternative to the `resample_sfreq` parameter that
#   can be used for resampling raw data. 1 means no decimation.
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Decimation requires to lowpass filtered the data to avoid aliasing.
# Note that using decimation is much faster than resampling.
#
# Example
# ~~~~~~~
# >>> decim = 1  # no decimation
# or
# >>> decim = 4  # decimate by 4 ie devide sampling frequency by 4

decim = 1

###############################################################################
# AUTOMATIC REJECTION OF ARTIFACTS
# --------------------------------
#
# Good Practice / Advice
# ~~~~~~~~~~~~~~~~~~~~~~
# Have a look at your raw data and train yourself to detect a blink, a heart
# beat and an eye movement.
# You can do a quick average of blink data and check what the amplitude looks
# like.
#
#  ``reject`` : dict | None
#    The rejection limits to make some epochs as bads.
#    This allows to remove strong transient artifacts.
#    If you want to reject and retrieve blinks later, e.g. with ICA,
#    don't specify a value for the eog channel (see examples below).
#    Make sure to include values for eeg if you have EEG data
#
# Note
# ~~~~
# These numbers tend to vary between subjects.. You might want to consider
# using the autoreject method by Jas et al. 2018.
# See https://autoreject.github.io
#
# Example
# ~~~~~~~
# >>> reject = {'grad': 4000e-13, 'mag': 4e-12, 'eog': 150e-6}
# >>> reject = {'grad': 4000e-13, 'mag': 4e-12, 'eeg': 200e-6}
# >>> reject = None

# reject = {'grad': 4000e-13, 'mag': 4e-12}
reject = {'grad': None, 'mag': None, 'eeg': None}  # None

autoreject = True

###############################################################################
# EPOCHING
# --------
#
# ``tmin``: float
#    A float in seconds that gives the start time before event of an epoch.
#
# Example
# ~~~~~~~
# >>> tmin = -0.2  # take 200ms before event onset.

# tmin = -0.100 # see 06-make_epochs.py

# ``tmax``: float
#    A float in seconds that gives the end time before event of an epoch.
#
# Example
# ~~~~~~~
# >>> tmax = 0.5  # take 500ms after event onset.

# tmax = 0.800 # see 06-make_epochs.py

# ``trigger_time_shift`` : float | None
#    If float it specifies the offset for the trigger and the stimulus
#    (in seconds). You need to measure this value for your specific
#    experiment/setup.
#
# Example
# ~~~~~~~
# >>> trigger_time_shift = 0  # don't apply any offset

trigger_time_shift = 0

# ``baseline`` : tuple
#    It specifies how to baseline the epochs; if None, no baseline is applied.
#
# Example
# ~~~~~~~
# >>> baseline = (None, 0)  # baseline between tmin and 0

# There is an event 500ms prior to the time-locking event, so we want
# to take a baseline before that
# baseline = (-0.100, 0.0)  # see 06-make_epochs.py

# ``stim_channel`` : str
#    The name of the stimulus channel, which contains the events.
#
# Example
# ~~~~~~~
# >>> stim_channel = 'STI 014'  # or 'STI101'

stim_channel = 'STI008'

# ``min_event_duration`` : float
#    The minimal duration of the events you want to extract (in seconds).
#
# Example
# ~~~~~~~
# >>> min_event_duration = 0.002  # 2 miliseconds

min_event_duration = 0.002

#  `event_id`` : dict
#    Dictionary that maps events (trigger/marker values)
#    to conditions.
#
# Example
# ~~~~~~~
# >>> event_id = {'auditory/left': 1, 'auditory/right': 2}`
# or
# >>> event_id = {'Onset': 4} with conditions = ['Onset']

# event_id = {'incoherent/1': 33, 'incoherent/2': 35,
#             'coherent/down': 37, 'coherent/up': 39}

#  `conditions`` : dict
#    List of condition names to consider. Must match the keys
#    in event_id.
#
# Example
# ~~~~~~~
# >>> conditions = ['auditory', 'visual']
# or
# >>> conditions = ['left', 'right']

# conditions = ['incoherent', 'coherent']

###############################################################################
# ARTIFACT REMOVAL
# ----------------
#
# You can choose between ICA and SSP to remove eye and heart artifacts.
# SSP: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ssp.html?highlight=ssp # noqa
# ICA: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ica.html?highlight=ica # noqa
# if you choose ICA, run scripts 5a and 6a
# if you choose SSP, run scripts 5b and 6b
#
# Currently you cannot use both.
#
# ``use_ssp`` : bool
#    If True ICA should be used or not.

use_ssp = False

# ``use_ica`` : bool
#    If True ICA should be used or not.

use_ica = True

# ``ica_decim`` : int
#    The decimation parameter to compute ICA. If 5 it means
#    that 1 every 5 sample is used by ICA solver. The higher the faster
#    it is to run but the less data you have to compute a good ICA.

ica_decim = 10


# ``default_reject_comps`` : dict
#    A dictionary that specifies the indices of the ICA components to reject
#    for each subject. For example you can use:
#    rejcomps_man['subject01'] = dict(eeg=[12], meg=[7])

def default_reject_comps():
    return dict(meg=[], eeg=[])

# VERSION 12/02/2021
rejcomps_man = defaultdict(default_reject_comps)
rejcomps_man['sub01-pa_190002'] = dict(eeg=[9, 0, 6, 19], meg=[2, 67, 7, 69])
rejcomps_man['sub02-ch_180036'] = dict(eeg=[0, 1, 15], meg=[1, 0, 26])
rejcomps_man['sub03-mr_190273'] = dict(eeg=[1, 12, 0, 10], meg=[45, 24, 2, 38])
rejcomps_man['sub04-rf_190499'] = dict(eeg=[0, 7], meg=[14, 6, 28])
rejcomps_man['sub05-cr_170417'] = dict(eeg=[0, 15], meg=[18, 0])
rejcomps_man['sub06-kc_160388'] = dict(eeg=[1, 18], meg=[16, 40, 5, 31])
rejcomps_man['sub07-jm_100109'] = dict(eeg=[1, 3], meg=[0, 5])
rejcomps_man['sub08-cc_150418'] = dict(eeg=[5, 8], meg=[19, 25, 3])
rejcomps_man['sub09-ag_170045'] = dict(eeg=[0, 3, 7], meg=[4, 10, 8, 40])
rejcomps_man['sub10-gp_190568'] = dict(eeg=[1, 6], meg=[0, 5, 1, 19])
rejcomps_man['sub11-fr_190151'] = dict(eeg=[2, 9], meg=[16, 17, 22])
rejcomps_man['sub12-lg_170436'] = dict(eeg=[1, 11], meg=[11, 0, 13])
rejcomps_man['sub13-lq_180242'] = dict(eeg=[0, 12], meg=[9, 26, 0, 11])
rejcomps_man['sub14-js_180232'] = dict(eeg=[0, 1, 2], meg=[9, 15, 0, 14])
rejcomps_man['sub15-ev_070110'] = dict(eeg=[], meg=[31, 9])
rejcomps_man['sub16-ma_190185'] = dict(eeg=[11, 0, 12], meg=[12, 15, 0, 13])
rejcomps_man['sub17-mt_170249'] = dict(eeg=[13, 1, 8], meg=[19, 36, 4])
rejcomps_man['sub18-eo_190576'] = dict(eeg=[19, 1, 18], meg=[7, 14])
rejcomps_man['sub19-mg_190180'] = dict(eeg=[0, 1], meg=[13, 26, 7])

# ``ica_ctps_ecg_threshold``: float
#    The threshold parameter passed to `find_bads_ecg` method.

ica_ctps_ecg_threshold = 0.1

###############################################################################
# DECODING
# --------
#
# ``decoding_conditions`` : list
#    List of conditions to be classified.
#
# Example
# ~~~~~~~
# >>> decoding_conditions = []  # don't do decoding
# or
# >>> decoding_conditions = [('auditory', 'visual'), ('left', 'right')]

decoding_conditions = [('incoherent', 'coherent')]

# ``decoding_metric`` : str
#    The metric to use for cross-validation. It can be 'roc_auc' or 'accuracy'
#    or any metric supported by scikit-learn.

decoding_metric = 'roc_auc'

# ``decoding_n_splits`` : int
#    The number of folds (a.k.a. splits) to use in the cross-validation.

decoding_n_splits = 5

###############################################################################
# TIME-FREQUENCY
# --------------
#
# ``time_frequency_conditions`` : list
#    The conditions to compute time-frequency decomposition on.

time_frequency_conditions = ['coherent']

###############################################################################
# SOURCE SPACE PARAMETERS
# -----------------------
#

# ``spacing`` : str
#    The spacing to use. Can be ``'ico#'`` for a recursively subdivided
#    icosahedron, ``'oct#'`` for a recursively subdivided octahedron,
#    ``'all'`` for all points, or an integer to use appoximate
#    distance-based spacing (in mm).

spacing = 'oct6'

# ``mindist`` : float
#    Exclude points closer than this distance (mm) to the bounding surface.

mindist = 5

# ``loose`` : float in [0, 1] | 'auto'
#    Value that weights the source variances of the dipole components
#    that are parallel (tangential) to the cortical surface. If loose
#    is 0 then the solution is computed with fixed orientation,
#    and fixed must be True or "auto".
#    If loose is 1, it corresponds to free orientations.
#    The default value ('auto') is set to 0.2 for surface-oriented source
#    space and set to 1.0 for volumetric, discrete, or mixed source spaces,
#    unless ``fixed is True`` in which case the value 0. is used.

loose = 0.2

# ``depth`` : None | float | dict
#    If float (default 0.8), it acts as the depth weighting exponent (``exp``)
#    to use (must be between 0 and 1). None is equivalent to 0, meaning no
#    depth weighting is performed. Can also be a `dict` containing additional
#    keyword arguments to pass to :func:`mne.forward.compute_depth_prior`
#    (see docstring for details and defaults).

depth = 0.8

# method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
#    Use minimum norm, dSPM (default), sLORETA, or eLORETA.

method = 'dSPM'

# smooth : int | None
#    Number of iterations for the smoothing of the surface data.
#    If None, smooth is automatically defined to fill the surface
#    with non-zero values. The default is spacing=None.

smooth = 10

# ``base_fname_trans`` : str
#   The path to the trans files obtained with coregistration.
#
# Example
# ~~~~~~~
# >>> base_fname_trans = '{subject}_' + study_name + '_raw-trans.fif'
# or
# >>> base_fname_trans = '{subject}-trans.fif'

base_fname_trans = '{subject}-trans.fif'

fsaverage_vertices = [np.arange(10242), np.arange(10242)]

if not os.path.isdir(study_path):
    os.mkdir(study_path)

if not os.path.isdir(subjects_dir):
    os.mkdir(subjects_dir)

###############################################################################
# ADVANCED
# --------
#
# ``l_trans_bandwidth`` : float |'auto'
#    A float that specifies the transition bandwidth of the
#    highpass filter. By default it's `'auto'` and uses default mne
#    parameters.

l_trans_bandwidth = 'auto'

#  ``h_trans_bandwidth`` : float |'auto'
#    A float that specifies the transition bandwidth of the
#    lowpass filter. By default it's `'auto'` and uses default mne
#    parameters.

h_trans_bandwidth = 'auto'

#  ``N_JOBS`` : int
#    An integer that specifies how many subjects you want to run in parallel.

N_JOBS = 20

# ``random_state`` : None | int | np.random.RandomState
#    To specify the random generator state. This allows to have
#    the results more reproducible between machines and systems.
#    Some methods like ICA need random values for initialisation.

random_state = 42

# ``shortest_event`` : int
#    Minimum number of samples an event must last. If the
#    duration is less than this an exception will be raised.

shortest_event = 1

# ``allow_maxshield``  : bool
#    To import data that was recorded with Maxshield on before running
#    maxfilter set this to True.

allow_maxshield = True

###############################################################################
# CHECKS
# --------
#
# --- --- You should not touch the next lines --- ---

if (use_maxwell_filter and
        len(set(ch_types).intersection(('meg', 'grad', 'mag'))) == 0):
    raise ValueError('Cannot use maxwell filter without MEG channels.')

if use_ssp and use_ica:
    raise ValueError('Cannot use both SSP and ICA.')
