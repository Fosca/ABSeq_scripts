import config
import pickle
import matplotlib.pyplot as plt
import os.path as op
import numpy as np
from ABseq_func import *
from mne.parallel import parallel_func

config.subjects_list = ['sub08-cc_150418']

def ar_log_summary(subject, epoch_on_first_element):
    meg_subject_dir = op.join(config.meg_dir, subject)
    if epoch_on_first_element:
        arlog_name = op.join(meg_subject_dir, subject + '_1st_element_clean_epo_reject_log.obj')
        save_path = op.join(config.fig_path, 'AutoReject_fullsequences_epochs', subject)
    else:
        arlog_name = op.join(meg_subject_dir, subject + '_clean_epo_reject_log.obj')
        save_path = op.join(config.fig_path, 'AutoReject_items_epochs', subject)
    reject_log = pickle.load(open(arlog_name, 'rb'))

    # Plots
    reject_log_plot(reject_log, subject, save_path=save_path, fig_name='AutoReject')

def reject_log_plot(reject_log, subject, save_path='', fig_name=''):
    utils.create_folder(save_path)

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
    fig_name_save = op.join(save_path, fig_name+'_fig1.png')
    fig.savefig(fig_name_save, bbox_inches='tight')
    print(fig_name_save)

    # --- Fig2
    fig = plt.figure(figsize=(18, 6))
    plt.bar(range(reject_log.labels.shape[1]), sum(reject_log.labels == 1))
    ch_names_ = reject_log.ch_names[7::8]
    ax = plt.gca()
    ax.grid(False)
    ax.set_xlabel('Channels')
    ax.set_ylabel('N epochs')
    plt.title('N bads')
    plt.setp(ax, xticks=range(7, reject_log.labels.shape[1], 8), xticklabels=ch_names_)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.tight_layout(rect=[None, None, None, 1.1])
    fig_name_save = op.join(save_path, fig_name+'_fig2.png')
    fig.savefig(fig_name_save, bbox_inches='tight')
    print('# >>>>>>>>>>>>> Subject %s, bad epochs per channel: %.1f%%' % (subject, np.mean(
        sum(reject_log.labels == 1) / reject_log.labels.shape[0]) * 100))
    print(fig_name_save)

    # --- Fig3
    fig = plt.figure(figsize=(18, 6))
    plt.bar(range(reject_log.labels.shape[1]), sum(reject_log.labels == 2))
    ch_names_ = reject_log.ch_names[7::8]
    ax = plt.gca()
    ax.grid(False)
    ax.set_xlabel('Channels')
    ax.set_ylabel('N epochs')
    plt.title('N bads and interpolated')
    plt.setp(ax, xticks=range(7, reject_log.labels.shape[1], 8), xticklabels=ch_names_)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis=u'both', which=u'both', length=0)
    plt.tight_layout(rect=[None, None, None, 1.1])
    fig_name_save = op.join(save_path, fig_name+'_fig3.png')
    fig.savefig(fig_name_save, bbox_inches='tight')
    print('# >>>>>>>>>>>>> Subject %s, bad and interpolated epochs per channel: %.1f%%' % (subject, np.mean(
        sum(reject_log.labels == 2) / reject_log.labels.shape[0]) * 100))
    print(fig_name_save)
    plt.close('all')
    print('\n')


parallel, run_func, _ = parallel_func(ar_log_summary, n_jobs=config.N_JOBS)

epoch_on_first_element = False
parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)

# epoch_on_first_element = True
# parallel(run_func(subject, epoch_on_first_element) for subject in config.subjects_list)


