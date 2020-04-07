from ABseq_func import TP_funcs
import config
import numpy as np
import matplotlib.pyplot as plt
#
# # 0 - DETERMINE THE SET OF omegas ON WHICH TO PERFORM THE REGRESSION ===================================================
#
# # list_omegas = [2**i for i in range(1,11)]
omega_list =[1,2,3,4,6,8,10,15,20,30]
clean = True
subject = config.subjects_list[0]
# metadata_updated = TP_funcs.from_epochs_to_surprise(subject, omega_list, clean = clean,order = 1)
# omega_corr_mat = TP_funcs.correlate_surprise_regressors(subject, omega_list, clean = clean)

# 1 - DETERMINE THE BEST omega (decay param) FOR EACH TIME POINT =======================================================
# Low-pass filter at 30 Hz
# Run one regression ERP ~ 1 + surprise(omega)

# Generate the time series of R2 for each value of omega
from importlib import reload
reload(TP_funcs)
TP_funcs.run_linear_regression_surprises(subject,omega_list,clean=True,decim = None,prefix = 'essai')

path = '/Users/fosca/Documents/Fosca/Post_doc/Projects/ABSeq/results/TP_effects/surprise_omegas/sub01-pa_190002/essair2_surprise.npy'
surprise_dict = np.load(path,allow_pickle=True).item()

# 2 - Check out when and for which omegas the R2 is maximal ===========================================
TP_funcs.plot_r2_surprise([config.subjects_list[0]],omegas_of_interest=omega_list,fname = 'essair2_surprise.npy',times_of_interest=[0,0.052,0.1,0.152,0.2,0.252])


# 3 - DETERMINE THE RESIDUALS ONCE WE REGRESS OUT THESE omegas per time point ===========================================
