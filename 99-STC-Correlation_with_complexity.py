import mne
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *
import numpy as np
import pickle
from scipy.stats import sem

# Exclude some subjects
config.exclude_subjects.append('sub08-cc_150418')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

subjects_list = config.subjects_list



