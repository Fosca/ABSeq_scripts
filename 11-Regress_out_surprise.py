from ABseq_func import TP_funcs
import config
import matplotlib.pyplot as plt

# 0 - DETERMINE THE SET OF omegas ON WHICH TO PERFORM THE REGRESSION ===================================================

# list_omegas = [2**i for i in range(1,11)]
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


# 2 - Check out when and for which omegas the R2 is maximal ===========================================
TP_funcs.plot_r2_surprise(config.subjects_list,clean=True)


# 3 - DETERMINE THE RESIDUALS ONCE WE REGRESS OUT THESE omegas per time point ===========================================
