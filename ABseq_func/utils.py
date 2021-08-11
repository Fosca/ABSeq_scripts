import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
import os
import config
import mne
import ABseq_func.epoching_funcs as epoching_funcs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as op

def delete_files(basepath,tree_subject_names,file_name,subj_in_filename=False):

    for sub in tree_subject_names:
        file = os.path.join(basepath,sub,file_name)
        if subj_in_filename:
            file = os.path.join(basepath,sub,sub+file_name)

        print(file)
        stream = os.popen('rm -r %s'%file)
        output = stream.read()
        output


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

def count_bads():
    data = config.bads
    for subject in config.subjects_list:  # subjects
        for keyR in data[subject].keys():  # runs
            megcount = 0
            eegcount = 0
            for C in range(len(data[subject][keyR])):  # channels
                if 'MEG' in data[subject][keyR][C]:
                    megcount = megcount + 1
                if 'EEG' in data[subject][keyR][C]:
                    eegcount = eegcount + 1
            print('%s;%s;MEG;%d;EEG;%d' % (subject, keyR, megcount, eegcount))


def plot_features_from_metadata(sequences = [3,4,5,6,7]):


    figures_path = config.fig_path+'/features_figs/'

    # load metadata subject 1
    epo = epoching_funcs.load_epochs_items(config.subjects_list[0],cleaned=False)
    metadata = epo.metadata

    metadata_all = []
    for seqID in sequences:
        print(seqID)
        meta_all_seq = []
        for posinSeq in range(1,17):
            meta_1 = metadata.query("SequenceID == '%i' and StimID == 1 and ViolationInSequence == 0 and StimPosition == '%i' and TrialNumber == 1 "%(seqID,posinSeq))
            meta_all_seq.append(meta_1)
        meta_all_seq = pd.concat(meta_all_seq)
        metadata_all.append(meta_all_seq)

    for feature_name in ['StimID','Complexity','GlobalEntropy','StimPosition','RepeatAlter','ChunkNumber','WithinChunkPosition','WithinChunkPositionReverse','ChunkDepth','OpenedChunks','ClosedChunks','ChunkBeginning','ChunkEnd','ChunkSize']:
        # Plot
        # Prepare colors range
        cm = plt.get_cmap('viridis')
        metadata_allseq = pd.concat(metadata_all)
        metadata_allseq_reg = metadata_allseq[feature_name]
        minvalue = np.nanmin(metadata_allseq_reg)
        maxvalue = np.nanmax(metadata_allseq_reg)
        # Open figure
        if len(sequences)==5:
            fig, ax = plt.subplots(5, 1, figsize=(8.7, 4.4), sharex=False, sharey=True,
                                   constrained_layout=True)
        else:
            fig, ax = plt.subplots(len(sequences), 1, figsize=(8.7, 6), sharex=False, sharey=True,
                                   constrained_layout=True)
        fig.suptitle(feature_name, fontsize=12)
        # Plot each sequences with circle color corresponding to regressor value
        for nseq, seqs in enumerate(sequences):

            seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqs)
            ax[nseq].set_title(seqname, loc='left', weight='bold', fontsize=12)
            metadata = metadata_all[nseq][feature_name]
            # Normalize between 0 and 1 based on possible values across sequences, in order to set the color
            metadata = (metadata - minvalue) / (maxvalue - minvalue)
            # stimID is always 1, so we use seqtxtXY instead...
            if feature_name == 'StimID':
                for ii in range(len(seqtxtXY)):
                    if seqtxtXY[ii] == 'x':
                        metadata[metadata.index[ii]] = 0
                    elif seqtxtXY[ii] == 'Y':
                        metadata[metadata.index[ii]] = 1
            for stimpos in range(0, 16):
                value = metadata[metadata.index[stimpos]]
                if ~np.isnan(value):
                    circle = plt.Circle((stimpos + 1, 0.5), 0.4, facecolor=cm(value), edgecolor='k', linewidth=1)
                else:
                    circle = plt.Circle((stimpos + 1, 0.5), 0.4, facecolor='white', edgecolor='k', linewidth=1)
                ax[nseq].add_artist(circle)
            ax[nseq].set_xlim([0, 17])
            for key in ('top', 'right', 'bottom', 'left'):
                ax[nseq].spines[key].set(visible=False)
            ax[nseq].set_xticks([], [])
            ax[nseq].set_yticks([], [])
        # Add "xY" using the same yval for all
        ylim = ax[nseq].get_ylim()
        yval = ylim[1] - ylim[1] * 0.1
        for nseq, seqs in enumerate(sequences):
            seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqs)
            print(seqname)
            for xx in range(16):
                ax[nseq].text(xx + 1, 0.5, seqtxtXY[xx], horizontalalignment='center', verticalalignment='center',
                              fontsize=12)

        suffix = ''
        if len(sequences)==5:
            suffix = '_withoutSeqID12'
        fig_name = op.join(figures_path, feature_name + '_regressor'+suffix+'.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
        plt.close(fig)


