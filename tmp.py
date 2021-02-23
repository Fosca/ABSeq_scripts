import csv
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import config
import os.path as op
import mne
import glob
import warnings
from autoreject import AutoReject
from autoreject import get_rejection_threshold
import pickle
import config
from mne.parallel import parallel_func
from ABseq_func import epoching_funcs
from ABseq_func import autoreject_funcs
import csv
import os
import numpy as np
import scipy.io as sio
import pandas as pd
import config
import os.path as op
import mne
import glob
import warnings
from autoreject import AutoReject
import pickle


subject = config.subjects_list[11]
meg_subject_dir = op.join(config.meg_dir, subject)
epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)



# run autoreject "global" -> just get the thresholds
reject = get_rejection_threshold(epochs, ch_types=['mag', 'grad', 'eeg'])
epochs1 = epochs.copy().drop_bad(reject=reject)
fname = op.join(meg_subject_dir, 'epochs_globalAR-epo.fif')
print("Saving: ", fname)
epochs1.save(fname, overwrite=True)


# run autoreject "local"
ar = AutoReject()
epochs2, reject_log = ar.fit_transform(epochs, return_log=True)
fname = op.join(meg_subject_dir, 'epochs_localAR-epo.fif')
print("Saving: ", fname)
epochs2.save(fname, overwrite=True)
# Save autoreject reject_log
pickle.dump(reject_log, open(fname[:-4] + '_reject_log.obj', 'wb'))


######################
fname = op.join(meg_subject_dir, 'epochs_globalAR-epo.fif')
epochs1 = mne.read_epochs(fname, preload=True)
epochs1
epochs1['ViolationOrNot == 1'].copy().average().plot_joint()

fname = op.join(meg_subject_dir, 'epochs_localAR-epo.fif')
epochs2 = mne.read_epochs(fname, preload=True)
epochs2['ViolationOrNot == 1'].copy().average().plot_joint()

arlog_name = op.join(meg_subject_dir, 'epochs_localAR_reject_log.obj')
reject_log = pickle.load(open(arlog_name, 'rb'))
Nrej = sum(reject_log.bad_epochs == True)
Nepochs = 16 * 46 * 7 * 2
print('%s, items epochs: %d/%d rejected bad epochs items = %.2f%%' % (subject, Nrej, Nepochs, Nrej / Nepochs * 100))
autoreject_funcs.reject_log_plot(reject_log, subject, save_path=meg_subject_dir, fig_name='AutoReject')
