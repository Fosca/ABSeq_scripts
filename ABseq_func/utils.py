import os
import config
import mne
import ABseq_func.epoching_funcs

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

def age_participants(subject_list):

    for sub in subject_list:
        print(sub)
        epochs =  ABseq_func.epoching_funcs.load_epochs_items(sub, cleaned=True)
        print(epochs.info['subject_info']['birthday'])
        print("sex %i"%epochs.info['subject_info']['sex'])


def is_there_this_file_for_subjects(path,subject_list,filename):

    print("searching for files of the type: %s" %path+'/'+'subject'+filename)

    for subject in subject_list:
        full_path = path+'/'+subject+filename
        if os.path.exists(full_path):
            print("Exists for %s"%subject)
        else:
            print("------- Does not exist for %s" % subject)



