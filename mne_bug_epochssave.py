import mne

data_path = "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/data_bug_mne/bug_data_raw.fif"
tmin = -0.050
tmax = 0.600

raw = mne.io.read_raw_fif(data_path,preload=True)
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=True, eog=True, exclude=())

info_anonymous = mne.io.anonymize_info(raw.info)

events = mne.find_events(raw, stim_channel="STI008",
                         consecutive=True,
                         min_duration=0.002,
                         shortest_event=2)

epochs = mne.Epochs(raw, events, None, tmin=0, tmax=0.6,
                    proj=True, picks=picks, baseline=None,
                    preload=False, decim=1,
                    reject=None)

data_path_save = "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MEG/data_bug_mne/test-epo.fif"
epochs.save(data_path_save,overwrite=True)

epochs = mne.read_epochs(data_path_save)