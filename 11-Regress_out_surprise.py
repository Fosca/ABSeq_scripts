from ABseq_func import TP_funcs
import config
import numpy as np
import matplotlib.pyplot as plt
#
# # 0 - DETERMINE THE SET OF omegas ON WHICH TO PERFORM THE REGRESSION ===================================================
#
# # list_omegas = [2**i for i in range(1,11)]
# omega_list =[1,2,3,4,6,8,10,15,20,30,60,100,200,299]
omega_list = range(1,300)
# clean = True
# subject = config.subjects_list[0]
# metadata_updated = TP_funcs.from_epochs_to_surprise(subject, omega_list, clean = clean,order = 1)
# omega_corr_mat = TP_funcs.correlate_surprise_regressors(subject, omega_list, clean = clean)

# 1 - DETERMINE THE BEST omega (decay param) FOR EACH TIME POINT =======================================================
# Low-pass filter at 30 Hz
# Run one regression ERP ~ 1 + surprise(omega)

# Generate the time series of R2 for each value of omega
from importlib import reload
reload(TP_funcs)
# TP_funcs.run_linear_regression_surprises(subject,omega_list,clean=True,decim = None,prefix = 'essai')

# TP_funcs.run_linear_Ridge_surprises([config.subjects_list[0]],omega_list,clean=True,decim = None,prefix = 'essai')

# 2 - Check out when and for which omegas the R2 is maximal ===========================================
r2_omegas, times = TP_funcs.plot_r2_surprise(config.subjects_list,fname = 'r2_surprise.npy',omegas_of_interest=omega_list)


omega_argmax, times = TP_funcs.compute_and_save_argmax_omega(omega_list=range(1, 300))

plt.plot(times,omega_argmax,color='black')
plt.xlabel('time in sec')
plt.ylabel('Argmax Omega')
plt.title('Value of Omega that maximizes the explained variance')
# ================ compute what to remove from the signal and save it as an epoch object =======================












# 3 - DETERMINE THE RESIDUALS ONCE WE REGRESS OUT THESE omegas per time point ===========================================

