from ABseq_func import TP_funcs, cluster_funcs, epoching_funcs
import config
import numpy as np
import matplotlib.pyplot as plt

# ============ FOR EACH TIME WINDOW, WHAT IS THE OMEGA THAT EXPLAINS THE MOST THE VARIANCE ? ===========================

matrice, times = TP_funcs.plot_r2_surprise(config.subjects_list, fname='r2_surprise.npy', omegas_of_interest=range(1,300))

# for subject in config.subjects_list:
#     cluster_funcs.surprise_omegas_analysis(subject)
    # This function saves the score of the regressions for each value of omega and each time step


omega_argmax, times = TP_funcs.compute_and_save_argmax_omega(omega_list=range(1,300))

# DETERMINE THE RESIDUALS ONCE WE REGRESS OUT THESE omegas per time point ==============================================
# for subject in config.subjects_list:
#     TP_funcs.regress_out_optimal_omega(subject, clean=True)


# ======================================================================================================================
#                       DETERMINE IF THE DATA WITHING A CLUSTER DEPENDS OF THE SURPRISE
# ======================================================================================================================

# 1. Identify the cluster times and channels
# 2. Run a regression analysis for all the participants, either on the average data across channels, across times, both
# no average at all. This is done for all the values of omega <3








