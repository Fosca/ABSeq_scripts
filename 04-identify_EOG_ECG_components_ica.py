"""
===============
04- identify_EOG_ECG_components_ica.py
===============

Blinks and ECG artifacts are automatically detected and the corresponding ICA
components are removed from the data.
This relies on the ICAs computed in 03-run_ica.py
!! If you manually add components to remove (config.rejcomps_man),
make sure you did not re-run the ICA in the meantime. Otherwise (especially if
the random state was not set, or you used a different machine, the component
order might differ).

!! Inspect the .html report, confirm or correct the proposed components and mark them in config.rejcomps_man
"""

import os.path as op

import mne
from mne.parallel import parallel_func
from mne.preprocessing import read_ica
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
# from mne.report import Report
from mne import open_report

import numpy as np
import config


def automatic_identification_of_components(subject):
    # ==================================================
    # determine if we set the rejected components by hand
    # ==================================================
    if subject in config.rejcomps_man:
        print(subject)
        raise Exception(
            'The EOG and ECG components were already identified and hand written in the config.rejcomps_man\n Delete it if you want to rerun this part of the script.')
    else:
        ica = dict(meg=[], eeg=[])
        ica_reject = dict(meg=[], eeg=[])

        print("Identifying the components for subject: %s" % subject)
        meg_subject_dir = op.join(config.meg_dir, subject)

        # ==================================================
        # concatenate the four last runs to compute the correlation
        # between the ICA components determined in 03-run_ica with the epochs
        # on ECG and EOG events
        # ==================================================
        runs = config.runs_dict[subject]


        raw_list = list()
        for run in runs[-4:-1]: # load four last runs
            if config.use_maxwell_filter:
                extension = run + '_sss_raw'
            else:
                extension = run + '_filt_raw'

            raw_fname_in = op.join(meg_subject_dir,
                                   config.base_fname.format(**locals()))
            raw = mne.io.read_raw_fif(raw_fname_in, preload=True)
            raw_list.append(raw)

        print('  Concatenating runs')
        raw_ref_runs = mne.concatenate_raws(raw_list)
        del raw

        # ==================================================
        # define the channels corresponding to meg and eeg data
        # ==================================================

        picks_meg = mne.pick_types(raw_ref_runs.info, meg=True, eeg=False,
                                   eog=False, stim=False, exclude='bads')
        picks_eeg = mne.pick_types(raw_ref_runs.info, meg=False, eeg=True,
                                   eog=False, stim=False, exclude='bads')
        all_picks = {'meg': picks_meg, 'eeg': picks_eeg}

        print('Finding ICA components correlating with ECG and EOG epochs...')

        # ==================================================
        # define ch_types according to the recorded modalities
        # ==================================================

        ch_types = []
        if 'eeg' in config.ch_types:
            ch_types.append('eeg')
        if set(config.ch_types).intersection(('meg', 'grad', 'mag')):
            ch_types.append('meg')


        # ==================================================
        # main loop
        # ==================================================
        for ch_type in ch_types:
            print(ch_type)
            picks = all_picks[ch_type]

            # ==================================================
            # Load ICA
            # ==================================================

            fname_ica = op.join(meg_subject_dir,
                                '{0}_{1}_{2}-ica.fif'.format(subject,
                                                             config.study_name,
                                                             ch_type))
            print('Reading ICA: ' + fname_ica)
            ica[ch_type] = read_ica(fname=fname_ica)

            # ==================================================
            # report input and output names
            # ==================================================
            # Load previous report (.h5)
            report_fname = \
                '{0}_{1}_{2}-ica.h5'.format(subject,
                                                     config.study_name,
                                                     ch_type)
            report_fname = op.join(meg_subject_dir, report_fname)

            # set the name of the final report
            report_fname_html = \
                '{0}_{1}_{2}-ica.html'.format(subject,
                                                     config.study_name,
                                                     ch_type)
            report_fname_html = op.join(meg_subject_dir, report_fname_html)
            report= open_report(report_fname)


            # ==================================================
            # Correlation with ECG epochs
            # ==================================================

            pick_ecg = mne.pick_types(raw_ref_runs.info, meg=False, eeg=False,
                                      ecg=True, eog=False)

            # either needs an ecg channel, or avg of the mags (i.e. MEG data)
            if pick_ecg or ch_type == 'meg':

                picks_ecg = np.concatenate([picks, pick_ecg])

                # Create ecg epochs
                if ch_type == 'meg':
                    reject = {'mag': config.reject['mag'],
                              'grad': config.reject['grad']}
                elif ch_type == 'eeg':
                    reject = {'eeg': config.reject['eeg']}

                ecg_epochs = create_ecg_epochs(raw_ref_runs, picks=picks_ecg, reject=None,
                                               baseline=(None, 0), tmin=-0.5,
                                               tmax=0.5)
                ecg_average = ecg_epochs.average()

                ecg_inds, scores = \
                    ica[ch_type].find_bads_ecg(ecg_epochs, method='ctps',
                                      threshold=config.ica_ctps_ecg_threshold)
                del ecg_epochs
                params = dict(exclude=ecg_inds, show=config.plot)

                # == == == == == == == ==  plots appended to report = == == == == == == == == == == ==
                # Plot r score
                report.add_figs_to_section(ica[ch_type].plot_scores(scores,**params),
                                           captions=ch_type.upper() + ' - ECG - ' +
                                           'R scores')
                # Plot source time course
                report.add_figs_to_section(ica[ch_type].plot_sources(ecg_average,**params),
                                           captions=ch_type.upper() + ' - ECG - ' +
                                           'Sources time course')
                # Plot source time course
                report.add_figs_to_section(ica[ch_type].plot_overlay(ecg_average,**params),
                                           captions=ch_type.upper() + ' - ECG - ' +
                                           'Corrections')


            else:
                # XXX : to check when EEG only is processed
                print('no ECG channel is present. Cannot automate ICAs component '
                      'detection for EOG!')


            # ==================================================
            # Correlation with EOG epochs
            # ==================================================
            pick_eog = mne.pick_types(raw_ref_runs.info, meg=False, eeg=False,
                                      ecg=False, eog=True)

            if pick_eog.any():
                print('using EOG channel')
                picks_eog = np.concatenate([picks, pick_eog])
                # Create eog epochs
                eog_epochs = create_eog_epochs(raw_ref_runs, picks=picks_eog, reject=None,
                                               baseline=(None, 0), tmin=-0.5,
                                               tmax=0.5)

                eog_average = eog_epochs.average()
                eog_inds, scores = ica[ch_type].find_bads_eog(eog_epochs, threshold=3.0)
                del eog_epochs
                params = dict(exclude=eog_inds, show=config.plot)

                # == == == == == == == ==  plots appended to report = == == == == == == == == == == ==
                # Plot r score
                report.add_figs_to_section(ica[ch_type].plot_scores(scores, **params),
                                           captions=ch_type.upper() + ' - EOG - ' +
                                           'R scores')

                # Plot source time course
                report.add_figs_to_section(ica[ch_type].plot_sources(eog_average, **params),
                                           captions=ch_type.upper() + ' - EOG - ' +
                                           'Sources time course')

                # Plot source time course
                report.add_figs_to_section(ica[ch_type].plot_overlay(eog_average, **params),
                                           captions=ch_type.upper() + ' - EOG - ' +
                                           'Corrections')

                report.save(report_fname, overwrite=True)
                report.save(report_fname_html, overwrite=True, open_browser=False)

            else:
                print('no EOG channel is present. Cannot automate ICAs component '
                      'detection for EOG!')

            ica_reject[ch_type] = list(ecg_inds) + list(eog_inds)

            # now reject the components
            print('Rejecting from %s: %s' % (ch_type, ica_reject))

        # ==================================================
        #  Visualize the data before and after cleaning
        # ==================================================
        fig = ica[ch_type].plot_overlay(raw_ref_runs, exclude=ica_reject[ch_type], show=config.plot)
        report.add_figs_to_section(fig, captions=ch_type.upper() +
                                                 ' - ALL(epochs) - Corrections')

        report.save(report_fname_html, overwrite=True, open_browser=False)
        report.save(report_fname, overwrite=True)



if config.use_ica:
    parallel, run_func, _ = parallel_func(automatic_identification_of_components, n_jobs=config.N_JOBS)
    parallel(run_func(subject) for subject in config.subjects_list)
