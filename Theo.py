import mne
import mne.viz.

# met le path de tes donnees ici
data_path = ''
# load raw data
raw = mne.io.read_raw_fif(path, allow_maxshield=True, preload=True, verbose='error')

# find the events
stim_channel = 'STI101'
min_event_duration = 0.001
shortest_event = 2

events = mne.find_events(raw, stim_channel=stim_channel,
                         consecutive=True,
                         min_duration=min_event_duration,
                         shortest_event=shortest_event)

# chercher sur le site de mne quelle est la fonction pour visualiser les events. Probablement mne.viz.events 



