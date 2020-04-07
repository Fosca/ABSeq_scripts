/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/ABseq_func/epoching_funcs.py:286: UserWarning: 
Loading pre-autoreject epochs for subject sub03-mr_190273
  warnings.warn('\nLoading pre-autoreject epochs for subject ' + subject)
Traceback (most recent call last):
  File "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/cluster//generated_jobs/surp_Omega/surp_Omega_sub03-mr_190273.py", line 4, in <module>
    cluster_funcs.surprise_omegas_analysis('sub03-mr_190273')
  File "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/ABseq_func/cluster_funcs.py", line 131, in surprise_omegas_analysis
    TP_funcs.run_linear_regression_surprises(subject, list_omegas, clean=False, decim=None)
  File "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/ABseq_func/TP_funcs.py", line 86, in run_linear_regression_surprises
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=clean)
  File "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/ABseq_func/epoching_funcs.py", line 289, in load_epochs_items
    epochs = mne.read_epochs(fname_in, preload=True)
  File "</home/fa250062/anaconda3/lib/python3.7/site-packages/mne/externals/decorator.py:decorator-gen-184>", line 2, in read_epochs
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/utils/_logging.py", line 90, in wrapper
    return function(*args, **kwargs)
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/epochs.py", line 2491, in read_epochs
    return EpochsFIF(fname, proj, preload, verbose)
  File "</home/fa250062/anaconda3/lib/python3.7/site-packages/mne/externals/decorator.py:decorator-gen-185>", line 2, in __init__
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/utils/_logging.py", line 90, in wrapper
    return function(*args, **kwargs)
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/epochs.py", line 2562, in __init__
    _read_one_epoch_file(fid, tree, preload)
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/epochs.py", line 2441, in _read_one_epoch_file
    data = read_tag(fid, data_tag.pos).data.astype(datatype)
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/io/tag.py", line 526, in read_tag
    tag.data = _read_matrix(fid, tag, shape, rlims, matrix_coding)
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/io/tag.py", line 249, in _read_matrix
    ndim = int(np.frombuffer(fid.read(4), dtype='>i4'))
TypeError: only size-1 arrays can be converted to Python scalars
