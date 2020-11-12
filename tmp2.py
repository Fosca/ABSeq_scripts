from __future__ import division
import initialization_paths
from ABseq_func import *
from ABseq_func import TP_funcs, SVM_funcs
import config
import mne
import subprocess
import MarkovModel_Python
import numpy as np
from sklearn.model_selection import KFold
import os.path as op
import os

subject = config.subjects_list[0]


run_info_subject_dir = op.join(config.run_info_dir, subject)
meg_subject_dir = op.join(config.meg_dir, subject)
metadata_path = os.path.join(meg_subject_dir, 'metadata_item.pkl')

subject = config.subjects_list[1]
decim = 10

feature_name = 'StimID'
list_sequences = [2,3,4,5,6,7]


SVM_dec = SVM_funcs.SVM_decoder()
epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
if decim is not None:
    epochs.decimate(decim)
metadata = epoching_funcs.update_metadata(subject, clean=False, new_field_name=None, new_field_values=None)
epochs.metadata = metadata
epochs = epochs["TrialNumber>10 and ViolationOrNot ==0"]


suf = ''
if load_residuals_regression:
    epochs = epoching_funcs.load_resid_epochs_items(subject)
    suf = 'resid_'


print('-- The values of the metadata for the feature %s are : '%feature_name)
print(np.unique(epochs.metadata[feature_name].values))

if list_sequences is not None:
    # concatenate the epochs belonging to the different sequences from the list_sequences
    epochs_concat1 = []
    # count the number of epochs that contribute per sequence in order later to balance this
    n_epochs = []
    for seqID in list_sequences:
        print(seqID)
        epo = epochs["SequenceID == " + str(seqID)]
        epo.events[:, 2] = epo.metadata[feature_name].values
        epo.event_id = {'%i' % i: i for i in np.unique(epo.events[:, 2])}
        epo.equalize_event_counts(epo.event_id)
        n_epochs.append(len(epo))
        print("---- there are %i epochs that contribute from sequence %i -----"%(len(epo),seqID))
        epochs_concat1.append(epo)
    # now determine the minimum number of epochs that come from a sequence
    epochs_concat2 = []
    n_min = np.min(n_epochs)
    n_max = np.max(n_epochs)
    if n_min != n_max:
        for k in range(len(list_sequences)):
            n_epo_seq = len(epochs_concat1[k])
            inds = np.random.permutation(n_epo_seq)
            inds = inds[:n_min]
            epochs_concat2.append(mne.concatenate_epochs([epochs_concat1[k][i] for i in inds]))
    else:
        epochs_concat2 = epochs_concat1

    epochs = mne.concatenate_epochs(epochs_concat2)
else:
    epochs.events[:, 2] = epochs.metadata[feature_name].values
    epochs.event_id = {'%i'%i:i for i in np.unique(epochs.events[:, 2])}
    epochs.equalize_event_counts(epochs.event_id)

kf = KFold(n_splits=4)

y = epochs.events[:, 2]
X = epochs._data
scores = []
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    SVM_dec.fit(X_train,y_train)
    scores.append(SVM_dec.score(X_test,y_test))

score = np.mean(scores, axis=0)
times = epochs.times
