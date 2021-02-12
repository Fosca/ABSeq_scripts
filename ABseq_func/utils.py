import os
import config
import mne


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def load_info_subjects(subjects_list):

    infos = []
    for subject in subjects_list:
        evoked_path = os.path.join(config.meg_dir,subject,'evoked_cleaned','items_standard_all-ave.fif')
        ev = mne.read_evokeds(evoked_path)
        infos.append(ev[0].info)

    return infos

