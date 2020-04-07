from __future__ import division
import os.path as op
from matplotlib import pyplot as plt
import config
from ABseq_func import *
import mne
import numpy as np
from mne.stats import linear_regression, fdr_correction, bonferroni_correction, permutation_cluster_1samp_test
from mne.viz import plot_topomap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ======================== REGRESSORS TO INCLUDE AND RESULTS FOLDER ========================= #
analysis_name = 'test2'
regress_path = op.join(config.result_path, 'linear_models', analysis_name)
names = ["ChunkBeginning", "ChunkEnd", "ChunkDepth", "ChunkSize", "Identity"]

# ========================= RUN LINEAR REGRESSION FOR EACH SUBJECT ========================== #
# for subject in config.subjects_list:
#     linear_reg_funcs.run_linear_regression_v2(analysis_name, names, subject, cleaned=True)

# Create evoked with each (unique) level of the regressor
for subject in config.subjects_list:
    evoked_funcs.create_evoked_for_regression_factors(names, subject, cleaned=True)

