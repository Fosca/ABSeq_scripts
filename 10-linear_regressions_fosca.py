from ABseq_func import epoching_funcs, regression_funcs
import os.path as op
import os
import config
from sklearn.preprocessing import scale
from mne.stats import linear_regression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn import linear_model
import numpy as np
import mne

epochs_fname = 'inter_comp_/residual-epo.fif'
epochs_fname = ''
filter_name = None
cleaned = True
subject = config.subjects_list[0]
remap_grads = False
lowpass_epochs = False
apply_baseline = True
suffix = ''

regressors_names = ['Intercept','Complexity']

linear_reg_path = config.result_path+'/linear_models/'

def suffix_regression(epochs_fname,regressors_names):
    suffix = epochs_fname
    for reg_name in regressors_names:
        suffix += reg_name[:4]+'_'


epo_fname = linear_reg_path + epochs_fname
results_path = os.path.dirname(epo_fname)

if epochs_fname == '':
    epochs = regression_funcs.filter_good_epochs_for_regression_analysis(subject, clean=cleaned,
                                           fields_of_interest=regressors_names)
else:
    print("----- loading the data from %s ------"%epo_fname)
    epochs = mne.read_epochs(epo_fname)

# ====== normalization of regressors ====== #
for name in regressors_names:
    epochs.metadata[name] = scale(epochs.metadata[name])

# - - - - OPTIONNAL STEPS - - - -
if remap_grads:
    print('Remapping grads to mags')
    epochs = epochs.as_type('mag')
    print(str(len(epochs.ch_names)) + ' remaining channels!')
    suffix += 'remapped_'

if lowpass_epochs:
    print('Low pass filtering...')
    epochs = epochs.filter(l_freq=None,
                           h_freq=30)  # default parameters (maybe should filter raw data instead of epochs...)
    suffix += 'lowpassed_'
if apply_baseline:
    epochs = epochs.apply_baseline(baseline=(-0.050, 0))
    suffix += 'baselined_'

if cleaned:
    suffix += 'clean_'
# ====== filter epochs according to the hab, test, including repeat alternate or not etc. ====== #
before = len(epochs)
filters = regression_funcs.filter_string_for_metadata()
if filter_name is not None:
    epochs = epochs[filters[filter_name]]
print('Keeping %.1f%% of epochs' % (len(epochs) / before * 100))

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#                           Now start the things related to the regression
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# cross validate 4 folds
skf = StratifiedKFold(n_splits=4)
y_balancing = epochs.metadata["SequenceID"].values * 100 + epochs.metadata["StimPosition"].values

betas = []
scores = []
fold_number = 1

for train_index, test_index in skf.split(np.zeros(len(y_balancing)), y_balancing):

    res = linear_regression(epochs, epochs.metadata[regressors_names], names=regressors_names)
    print("======= running regression for fold %i =======" % fold_number)
    # predictor matrix
    preds_matrix_train = np.asarray(epochs[train_index].metadata[regressors_names].values)
    preds_matrix_test = np.asarray(epochs[test_index].metadata[regressors_names].values)
    betas_matrix = np.zeros((len(regressors_names), epochs.get_data().shape[1], epochs.get_data().shape[2]))
    scores_cv = np.zeros((epochs.get_data().shape[1], epochs.get_data().shape[2]))

    for tt in range(epochs.get_data().shape[2]):
        # for each time-point, we run a regression for each channel
        reg = linear_model.LinearRegression(fit_intercept=False)
        data_train = epochs[train_index].get_data()
        data_test = epochs[test_index].get_data()

        reg.fit(y=data_train[:, :, tt], X=preds_matrix_train)
        betas_matrix[:, :, tt] = reg.coef_.T
        y_preds = reg.predict(preds_matrix_test)
        scores_cv[:, tt] = r2_score(y_true=data_test[:, :, tt], y_pred=y_preds)

    betas.append(betas_matrix)
    scores.append(scores_cv)
    fold_number += 1

# MEAN ACROSS CROSS-VALIDATION FOLDS
betas = np.mean(betas, axis=0)
scores = np.mean(scores, axis=0)

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#          Now save the outputs of the regression : score, betas and residuals
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

saving_path = op.join(results_path, '/linear_models')
np.save(op.join(saving_path, 'scores'+suffix+'.npy'), scores)

for ii, name_reg in enumerate(regressors_names):
    res[name_reg].beta._data = np.asarray(betas[ii, :, :])
    res[name_reg].beta.save(op.join(saving_path, name_reg + suffix + '.fif'))


    

