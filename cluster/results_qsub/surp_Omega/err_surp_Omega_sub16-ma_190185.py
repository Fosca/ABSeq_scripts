/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/ABseq_func/epoching_funcs.py:286: UserWarning: 
Loading pre-autoreject epochs for subject sub16-ma_190185
  warnings.warn('\nLoading pre-autoreject epochs for subject ' + subject)
Traceback (most recent call last):
  File "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/cluster//generated_jobs/surp_Omega/surp_Omega_sub16-ma_190185.py", line 4, in <module>
    cluster_funcs.surprise_omegas_analysis('sub16-ma_190185')
  File "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/ABseq_func/cluster_funcs.py", line 131, in surprise_omegas_analysis
    TP_funcs.run_linear_regression_surprises(subject, list_omegas, clean=False, decim=None)
  File "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/ABseq_func/TP_funcs.py", line 91, in run_linear_regression_surprises
    epochs.metadata = metadata
  File "</home/fa250062/anaconda3/lib/python3.7/site-packages/mne/externals/decorator.py:decorator-gen-4>", line 2, in metadata
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/utils/_logging.py", line 90, in wrapper
    return function(*args, **kwargs)
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/utils/mixin.py", line 374, in metadata
    metadata = self._check_metadata(metadata, reset_index=True)
  File "/home/fa250062/anaconda3/lib/python3.7/site-packages/mne/utils/mixin.py", line 351, in _check_metadata
    % (len(metadata), len(self.events)))
ValueError: metadata must have the same number of rows (10304) as events (10302)
