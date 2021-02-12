import os.path as op
import mne
import pandas as pd
import config
import numpy as np
import pickle
from mne.parallel import parallel_func
from ABseq_func import *
from autoreject import AutoReject
import matplotlib.pyplot as plt


def run_autoreject(subject, epoch_on_first_element):
    N_JOBS_ar = 1  # "The number of thresholds to compute in parallel."

    print('#########################################################################################')
    print('########################## Processing subject: %s ##########################' % subject)
    print('#########################################################################################')

    if epoch_on_first_element:
        print("  Loading 'full sequences' epochs")
        epochs = epoching_funcs.load_epochs_full_sequence(subject, cleaned=False)
    else:
        print("  Loading 'items' epochs")
        epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)

    # Running AutoReject (https://autoreject.github.io)
    epochs.load_data()
    ar = AutoReject(n_jobs=N_JOBS_ar)
    epochs, reject_log = ar.fit_transform(epochs, return_log=True)

    # Save epochs (after AutoReject)
    print('  Writing cleaned epochs to disk')
    meg_subject_dir = op.join(config.meg_dir, subject)
    if epoch_on_first_element:
        extension = subject + '_1st_element_clean_epo'
    else:
        extension = subject + '_clean_epo'
    epochs_fname = op.join(meg_subject_dir, config.base_fname.format(**locals()))
    print("Output: ", epochs_fname)
    epochs.save(epochs_fname)  # , overwrite=True)

    # Save autoreject reject_log
    pickle.dump(reject_log, open(epochs_fname[:-4] + '_reject_log.obj', 'wb'))
    # To read, would be: reject_log = pickle.load(open(epochs_fname[:-4]+'_reject_log.obj', 'rb'))


def ar_log_summary(subject, epoch_on_first_element):
    # reject_log content ===>
    # bad_epochs : array-like, shape (n_epochs,)
    #     The boolean array with entries True for epochs that
    #     are marked as bad.
    # labels : array, shape (n_epochs, n_channels)
    #     It contains integers that encode if a channel in a given
    #     epoch is good (value 0), bad (1), or bad and interpolated (2).
    # ch_names : list of str
    #     The list of channels corresponding to the rows of the labels.

    meg_subject_dir = op.join(config.meg_dir, subject)
    if epoch_on_first_element:
        arlog_name = op.join(meg_subject_dir, subject + '_1st_element_clean_epo_reject_log.obj')
        save_path = op.join(config.fig_path, 'AutoReject_fullsequences_epochs', subject)
    else:
        arlog_name = op.join(meg_subject_dir, subject + '_clean_epo_reject_log.obj')
        save_path = op.join(config.fig_path, 'AutoReject_items_epochs', subject)

    reject_log = pickle.load(open(arlog_name, 'rb'))

    if epoch_on_first_element:
        Nrej = sum(reject_log.bad_epochs == True)
        Nepochs = 46 * 7 * 2
        print('%s, fullsequence epochs: %d/%d rejected bad epochs items = %.2f%%' % (subject, Nrej, Nepochs, Nrej / Nepochs * 100))
    else:
        Nrej = sum(reject_log.bad_epochs == True)
        Nepochs = 16 * 46 * 7 * 2
        print('%s, items epochs: %d/%d rejected bad epochs items = %.2f%%' % (subject, Nrej, Nepochs, Nrej / Nepochs * 100))

    # Plots
    reject_log_plot(reject_log, subject, save_path=save_path, fig_name='AutoReject')


def reject_log_plot(reject_log, subject, save_path='', fig_name=''):
    utils.create_folder(save_path)

    N_channels = 366  # since reject_log.labels.shape[1] includes STI channels (which were not processed)
    N_epochs = reject_log.labels.shape[0]

    # --- Fig1
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(reject_log.labels, cmap='Reds', interpolation='nearest')
    ch_names_ = reject_log.ch_names[7::8]
    ax = plt.gca()
    ax.grid(False)
    ax.set_xlabel('Channels')
    ax.set_ylabel('Epochs')
    plt.setp(ax, xticks=range(7, reject_log.labels.shape[1], 8), xticklabels=ch_names_)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.tight_layout(rect=[None, None, None, 1.1])
    fig_name_save = op.join(save_path, fig_name + '_fig1.png')
    fig.savefig(fig_name_save, bbox_inches='tight')
    print(fig_name_save)

    # --- Fig2
    fig = plt.figure(figsize=(18, 6))
    plt.bar(range(reject_log.labels.shape[1]), sum(reject_log.labels == 1) / N_epochs * 100)
    ch_names_ = reject_log.ch_names[7::8]
    ax = plt.gca()
    ax.grid(False)
    ax.set_xlabel('Channels')
    ax.set_ylabel('% epochs')
    meanbads = sum((sum(reject_log.labels == 1)) / N_channels / N_epochs) * 100
    Nrej = sum(reject_log.bad_epochs == True)
    plt.title('Bads (%.2f%% epochs per channel) [%d on %d epochs entirely rejected]' % (meanbads, Nrej, N_epochs))
    plt.setp(ax, xticks=range(7, reject_log.labels.shape[1], 8), xticklabels=ch_names_)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.tight_layout(rect=[None, None, None, 1.1])
    fig_name_save = op.join(save_path, fig_name + '_fig2.png')
    fig.savefig(fig_name_save, bbox_inches='tight')
    print('# >>>>>>>>>>>>> Subject %s, bad epochs per channel: %.1f%%' % (subject, meanbads))
    print(fig_name_save)

    # --- Fig3
    fig = plt.figure(figsize=(18, 6))
    plt.bar(range(reject_log.labels.shape[1]), sum(reject_log.labels == 2) / N_epochs * 100)
    ch_names_ = reject_log.ch_names[7::8]
    ax = plt.gca()
    ax.grid(False)
    ax.set_xlabel('Channels')
    ax.set_ylabel('% epochs')
    meanbads = sum((sum(reject_log.labels == 2)) / N_channels / N_epochs) * 100
    Nrej = sum(reject_log.bad_epochs == True)
    plt.title('Bads and interpolated (%.2f%% epochs per channel) [%d on %d epochs entirely rejected]' % (meanbads, Nrej, N_epochs))
    plt.setp(ax, xticks=range(7, reject_log.labels.shape[1], 8), xticklabels=ch_names_)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.tight_layout(rect=[None, None, None, 1.1])
    fig_name_save = op.join(save_path, fig_name + '_fig3.png')
    fig.savefig(fig_name_save, bbox_inches='tight')
    print('# >>>>>>>>>>>>> Subject %s, bad and interpolated epochs per channel: %.1f%%' % (subject, meanbads))
    print(fig_name_save)
    plt.close('all')
    print('\n')
