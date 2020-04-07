"""
===========
03. Run ICA
===========
This fits ICA on the last 4 blocks of the experiment data filtered with 1 Hz highpass,
for this purpose only using fastICA. Separate ICAs are fitted and stored for
MEG and EEG data.
To actually remove designated ICA components from your data, you will have to
1- first run 04-identify_EOG_ECG_components_ica.py to automatically identify the components related to the EOG and ECG artefacts.
2- Inspecting the report, confirm or correct the proposed components and mark them in config.rejcomps_man
3 - Only once you did so, run 05-apply_ica.py
"""

import os.path as op
import mne
from mne.report import Report
from mne.preprocessing import ICA
from mne.parallel import parallel_func
import config

def run_ica(subject, tsss=config.mf_st_duration):
    print("Processing subject: %s" % subject)

    meg_subject_dir = op.join(config.meg_dir, subject)

    raw_list = list()
    print("  Loading raw data")
    runs = config.runs_dict[subject]

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
    raw = mne.concatenate_raws(raw_list)
    if "eeg" in config.ch_types:
        raw.set_eeg_reference(projection=True)
    del raw_list


    # don't reject based on EOG to keep blink artifacts
    # in the ICA computation.
    reject_ica = config.reject
    if reject_ica and 'eog' in reject_ica:
        reject_ica = dict(reject_ica)
        del reject_ica['eog']

    # produce high-pass filtered version of the data for ICA
    raw_ica = raw.copy().filter(l_freq=1., h_freq=None)

    print("  Running ICA...")

    picks_meg = mne.pick_types(raw.info, meg=True, eeg=False,
                               eog=False, stim=False, exclude='bads')
    picks_eeg = mne.pick_types(raw.info, meg=False, eeg=True,
                               eog=False, stim=False, exclude='bads')
    all_picks = {'meg': picks_meg, 'eeg': picks_eeg}

    n_components = {'meg': 0.999, 'eeg': 0.999}

    ch_types = []
    if 'eeg' in config.ch_types:
        ch_types.append('eeg')
    if set(config.ch_types).intersection(('meg', 'grad', 'mag')):
        ch_types.append('meg')

    for ch_type in ch_types:
        print('Running ICA for ' + ch_type)

        ica = ICA(method='fastica', random_state=config.random_state,
                  n_components=n_components[ch_type])

        picks = all_picks[ch_type]

        ica.fit(raw, picks=picks, decim=config.ica_decim)

        print('  Fit %d components (explaining at least %0.1f%% of the'
              ' variance)' % (ica.n_components_, 100 * n_components[ch_type]))

        ica_fname = \
            '{0}_{1}_{2}-ica.fif'.format(subject, config.study_name, ch_type)
        ica_fname = op.join(meg_subject_dir, ica_fname)
        ica.save(ica_fname)

        # if config.plot:

        # plot ICA components to html report
        report_fname = \
            '{0}_{1}_{2}-ica.h5'.format(subject, config.study_name,
                                          ch_type)
        report_fname = op.join(meg_subject_dir, report_fname)
        report_fname_html = \
            '{0}_{1}_{2}-ica.html'.format(subject, config.study_name,
                                          ch_type)
        report_fname = op.join(meg_subject_dir, report_fname)
        report = Report(report_fname, verbose=False)

        for idx in range(0, ica.n_components_):
            figure = ica.plot_properties(raw,
                                         picks=idx,
                                         psd_args={'fmax': 60},
                                         show=False)

            report.add_figs_to_section(figure, section=subject,
                                       captions=(ch_type.upper() +
                                                 ' - ICA Components'))

        report.save(report_fname, overwrite=True)
        report.save(report_fname_html, overwrite=True)

# make less parallel runs to limit memory usage
N_JOBS = max(config.N_JOBS // 4, 1)
print('N_JOBS=' + str(N_JOBS))

if config.use_ica:
    parallel, run_func, _ = parallel_func(run_ica, n_jobs=N_JOBS)
    parallel(run_func(subject) for subject in config.subjects_list)
else:
    print("ICA is not used. Set config.use_ica=True to use it.")
