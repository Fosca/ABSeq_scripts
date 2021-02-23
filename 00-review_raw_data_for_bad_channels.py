import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import mne
import config

def raw_diag_plots(meg_subject_dir, run_number):
    # LOAD DATA
    meg_subject_dir = op.join(config.meg_dir, subject)
    run = config.runs[run_number-1]
    extension = run + '_raw'
    # extension = run + '_ica_raw'
    raw_fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print('Loading ' + raw_fname_in)
    raw = mne.io.read_raw_fif(raw_fname_in, allow_maxshield=True, preload=True, verbose='error')
    raw.set_eeg_reference('average', projection=True)  # set EEG average reference

    # remove stim, eog, ecg...
    # picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, ecg=False)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=False)
    raw.pick_channels([raw.ch_names[pick] for pick in picks])

    # notch filter (power-line) ?
    raw.notch_filter(np.arange(50, 250, 50), n_jobs=4, fir_design='firwin')

    plt.close('all')
    # PLOT DATA
    # raw.plot(highpass=0.1, lowpass=150, start=100)
    raw.plot(highpass=0.5, start=50, duration=40, decim=8, butterfly=False, clipping='clamp')
    # PLOT PSD
    raw.plot_psd(average=False, fmax=50, picks='grad')
    raw.plot_psd(average=False, fmax=50, picks='mag')
    raw.plot_psd(average=False, fmax=50, picks='eeg')

def raws_diag_plots_7runs(meg_subject_dir):
    # LOAD DATA
    meg_subject_dir = op.join(config.meg_dir, subject)

    runs = ['run01', 'run02', 'run03', 'run04', 'run05', 'run06', 'run07']
            # 'run08', 'run09', 'run10', 'run11', 'run12', 'run13', 'run14']

    raw_list = list()
    for run in runs:
        extension = run + '_filt_raw'
        raw_fname_in = op.join(meg_subject_dir, config.base_fname.format(**locals()))
        print('Loading ' + raw_fname_in)
        raw = mne.io.read_raw_fif(raw_fname_in, preload=True, verbose=False, allow_maxshield=True)
        raw_list.append(raw)

    print('Concatenating runs')
    raw = mne.concatenate_raws(raw_list)
    del raw_list

    # remove stim, eog, ecg...
    picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=False, eog=False, ecg=False)
    raw.pick_channels([raw.ch_names[pick] for pick in picks])

    plt.close('all')
    # PLOT DATA
    # raw.plot(highpass=0.1, lowpass=150, start=100)
    raw.plot(highpass=0.5, start=50, duration=40, decim=8, butterfly=False, clipping='clamp')
    # PLOT PSD
    raw.plot_psd(average=False, fmax=50, picks='grad')
    raw.plot_psd(average=False, fmax=50, picks='mag')
    raw.plot_psd(average=False, fmax=50, picks='eeg')

##===
fname = '//canif.intra.cea.fr/acquisition/neuromag/data/abccba/empty_room/191018/syllables_raw.fif'
raw = mne.io.read_raw_fif(fname, allow_maxshield=True, preload=True)
raw.plot(highpass=0.5, start=50, duration=40, decim=8, butterfly=False, clipping='clamp')
#===

subject = 'sub19-mg_190180'
run_number = 14; raw_diag_plots(subject, run_number)

# subject = 'sub05-cr_170417'
# raws_diag_plots_7runs(subject)


# subjects_list = ['sub01-pa_190002', 'sub02-ch_180036', 'sub03-mr_190273', 'sub04-rf_190499', 'sub05-cr_170417', 'sub06-kc_160388',
#                  'sub07-jm_100109', 'sub08-cc_150418', 'sub09-ag_170045', 'sub10-gp_190568', 'sub11-fr_190151', 'sub12-lg_170436',
#                  'sub13-lq_180242', 'sub14-js_180232', 'sub15-ev_070110', 'sub16-ma_190185', 'sub17-mt_170249']