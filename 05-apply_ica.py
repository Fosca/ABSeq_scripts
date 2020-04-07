"""
===============
05. Apply ICA
===============

This function loads the ica filter as well as the user confirmed/corrected set of components to reject (in order to remove the artefacts).
It applies the filter and saves the ICA-filtered data

"""

import os.path as op

import mne
from mne.parallel import parallel_func
from mne.preprocessing import read_ica

import config


# ==================================================
#  Now remove the ICA rejected components for eeg and for meg data
# ==================================================

def apply_ica(subject):

    if subject not in config.rejcomps_man:
        raise Exception(
            'The EOG and ECG components were not saved in config.rejcomps_man\n You have to run 03-run-ica.py and 04-indentify_EOG_ECG_components.py first. Then manually enter the components in config.rejcomps_man.')


    print("Applying the ICA for subject: %s" % subject)
    meg_subject_dir = op.join(config.meg_dir, subject)

    # ==================================================
    # define ch_types according to the recorded modalities
    # ==================================================

    ch_types = []
    if 'eeg' in config.ch_types:
        ch_types.append('eeg')
    if set(config.ch_types).intersection(('meg', 'grad', 'mag')):
        ch_types.append('meg')


    ica = dict(meg=[], eeg=[])
    ica_reject = dict(meg=[], eeg=[])


    for ch_type in ch_types:
        print(ch_type)

        # ==================================================
        # Load ICA
        # ==================================================

        fname_ica = op.join(meg_subject_dir,
                            '{0}_{1}_{2}-ica.fif'.format(subject,
                                                         config.study_name,
                                                         ch_type))
        print('Reading ICA: ' + fname_ica)
        ica[ch_type] = read_ica(fname=fname_ica)

        ica_reject[ch_type] = list(config.rejcomps_man[subject][ch_type])
        print('Using user-defined bad ICA components')



    for run in config.runs_dict[subject]:
        if config.use_maxwell_filter:
            extension = run + '_sss_raw'
        else:
            extension = run + '_filt_raw'

        # = load the ICA =

        raw_fname_in = op.join(meg_subject_dir,
                               config.base_fname.format(**locals()))

        raw_before = mne.io.read_raw_fif(raw_fname_in, preload=True)

        raw_ica_eeg = ica['eeg'].apply(raw_before, exclude=ica_reject['eeg'])
        raw_ica_meg_eeg = ica['meg'].apply(raw_ica_eeg, exclude=ica_reject['meg'])
        extension = run + '_ica_raw'
        fname_out = op.join(meg_subject_dir,
                               config.base_fname.format(**locals()))

        print('Saving cleaned runs')
        raw_ica_meg_eeg.save(fname_out, overwrite=True)



if config.use_ica:
    parallel, run_func, _ = parallel_func(apply_ica, n_jobs=config.N_JOBS)
    parallel(run_func(subject) for subject in config.subjects_list)
