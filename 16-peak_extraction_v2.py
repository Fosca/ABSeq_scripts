from ABseq_func import *
import mne
import config
import scipy
import matplotlib.pyplot as plt
import numpy as np
import os.path as op
import time
import pandas as pd
import warnings
from scipy.stats import sem
from scipy.stats import pearsonr
from mne.stats import linear_regression
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
import pickle

# /!\ DO NOT SHOW SettingWithCopyWarning /!\
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Exclude some subjects
# config.exclude_subjects.append('sub10-gp_190568')
config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
config.subjects_list.sort()

# =========================================================== #
# Options
# =========================================================== #
cleaned = True  # epochs cleaned with autoreject or not, only when using original epochs (resid_epochs=False)
resid_epochs = False  # use epochs created by regressing out surprise effects, instead of original epochs
detrend_epochs = False
baseline_epochs = False  # apply baseline to the epochs
lowpass_epochs = False  # option to filter epochs with  30Hz lowpass filter
# All above set to False since done at the evoked level
preliminary = False
n_jobs = 6

# =========================================================== #
# Output folder
# =========================================================== #
results_path = op.join(config.result_path, 'peak_data')
utils.create_folder(results_path)

# # =========================================================== #
# # Time windows and ROI
# # =========================================================== #
# # Time windows of interest
# n1m_timewin = [.070 - 0.030, .070 + 0.030]
# p2m_timewin = [.140 - 0.030, .140 + 0.030]
# n1_timewin = [.080 - 0.030, .080 + 0.030]
# p2_timewin = [.130 - 0.030, .130 + 0.030]
#
# # NEW VERSION
# n1m_timewin = [.030, .110]
# p2m_timewin = [.110, .220]
# n1_timewin = n1m_timewin
# p2_timewin = p2m_timewin
# print(n1m_timewin)
# print(p2m_timewin)

if preliminary:
    # =========================================================== #
    # Explore activation to standard items to help define time windows and ROIs
    # =========================================================== #
    plt.close('all')
    # Group evoked standard items
    item_ev, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_all', filter_not=None)
    item_ev = [item_ev['items_standard_all-'][ii][0] for ii in range(len(item_ev['items_standard_all-']))]  # correct the list of lists issue
    item_ev_mean = mne.grand_average(item_ev)
    # Group evoked violation items
    item_ev_viol, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_all', filter_not=None)
    item_ev_viol = [item_ev_viol['items_viol_all-'][ii][0] for ii in range(len(item_ev_viol['items_viol_all-']))]  # correct the list of lists issue
    item_ev_viol_mean = mne.grand_average(item_ev_viol)
    # Find timepoints of interest
    item_ev_mean.crop(tmin=0, tmax=0.300).plot_joint(topomap_args=dict(show_names=False))

    # GFP - each subject than average
    times = item_ev[0].times
    gfp_eeg_all = []
    gfp_mag_all = []
    gfp_grad_all = []
    for evoked in item_ev:
        gfp_eeg = np.sum(evoked.copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
        gfp_grad = np.sum(evoked.copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
        gfp_mag = np.sum(evoked.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
        for gfp in [gfp_eeg, gfp_grad, gfp_mag]:
            gfp = mne.baseline.rescale(gfp, times, baseline=(-0.050, 0))
        gfp_eeg_all.append(gfp_eeg)
        gfp_grad_all.append(gfp_grad)
        gfp_mag_all.append(gfp_mag)
    gfp_evoked = {'eeg': gfp_eeg_all,
                  'grad': gfp_grad_all,
                  'mag': gfp_mag_all}
    plt.close('all')
    for ch_type in ['eeg', 'mag', 'grad']:
        mean = np.mean(gfp_evoked[ch_type], axis=0)
        ub = mean + sem(gfp_evoked[ch_type], axis=0)
        lb = mean - sem(gfp_evoked[ch_type], axis=0)
        fig, ax = plt.subplots(figsize=(9, 4))
        plt.fill_between(times, ub, lb, alpha=.2)
        plt.plot(times, mean, linewidth=1.5, label=ch_type)
        ax.set(xticks=np.arange(-.100, .501, .050), xlim=(-0.100, .500))
        ax.grid(True)
        fig.savefig(op.join(results_path, ch_type + '_GFP.png'), bbox_inches='tight', dpi=300)

    # both mag & grad, scaled
    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 4))
    for ch_type in ['mag', 'grad']:
        tmp = []
        for ii in range(len(gfp_evoked[ch_type])):
            tmp.append(scale(gfp_evoked[ch_type][ii]))
        mean = np.mean(tmp, axis=0)
        ub = mean + sem(tmp, axis=0)
        lb = mean - sem(tmp, axis=0)
        plt.fill_between(times * 1000, ub, lb, alpha=.2)
        plt.plot(times * 1000, mean, linewidth=1.5, label=ch_type + ('_normGFP'))
        ax.set(xticks=np.arange(-50, 501, 10), xlim=(-50, 300))
        ax.grid(True)
        plt.xticks(rotation=90)
        plt.legend()
    fig.savefig(op.join(results_path, 'MEG_GFP.png'), bbox_inches='tight', dpi=300)
    # Peaks N1: eeg:0.084 mag:0.068 grad: 0.068
    # Peak P2: eeg:0.132 mag:0.164 grad: 0.172 /// we will use 0.164 for both meg

    # GFP - on the grand_average ?
    item_ev_mean = mne.grand_average(item_ev)
    times = item_ev_mean.times
    gfp_eeg = np.sum(item_ev_mean.copy().pick_types(eeg=True, meg=False).data ** 2, axis=0)
    gfp_grad = np.sum(item_ev_mean.copy().pick_types(eeg=False, meg='grad').data ** 2, axis=0)
    gfp_mag = np.sum(item_ev_mean.copy().pick_types(eeg=False, meg='mag').data ** 2, axis=0)
    gfp_evoked = {'eeg': mne.baseline.rescale(gfp_eeg, times, baseline=(-0.050, 0)),
                  'grad': mne.baseline.rescale(gfp_grad, times, baseline=(-0.050, 0)),
                  'mag': mne.baseline.rescale(gfp_mag, times, baseline=(-0.050, 0))}
    plt.close('all')
    for ch_type in ['eeg', 'mag', 'grad']:
        mean = gfp_evoked[ch_type]
        fig, ax = plt.subplots(figsize=(9, 4))
        plt.plot(times, mean, linewidth=1.5, label=ch_type)
        ax.set(xticks=np.arange(-.100, .501, .050), xlim=(-0.100, .500))
        ax.grid(True)

# =========================================================== #
# Create N1/P2 group-based components (topos)
# =========================================================== #
N1_peak_time = 0.068
P2_peak_time = (0.136 + 0.172) / 2
plt.close('all')

# Group evoked standard items
item_ev, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_all', filter_not=None)
item_ev = [item_ev['items_standard_all-'][ii][0] for ii in range(len(item_ev['items_standard_all-']))]  # correct the list of lists issue
item_ev_mean = mne.grand_average(item_ev)

group_n1_eeg = item_ev_mean.copy().pick_types(eeg=True, meg=False).crop(tmin=0.084 - 0.025, tmax=0.084 + 0.025)
fig = group_n1_eeg.plot_topomap(times=0.084, average=99.99)
fig.savefig(op.join(results_path, 'eeg_n1_comp.png'), bbox_inches='tight', dpi=300)
group_n1_eeg = np.mean(group_n1_eeg.data, axis=1)

group_p2_eeg = item_ev_mean.copy().pick_types(eeg=True, meg=False).crop(tmin=0.132 - 0.025, tmax=0.132 + 0.025)
fig = group_p2_eeg.plot_topomap(times=0.132, average=99.99)
fig.savefig(op.join(results_path, 'eeg_p2_comp.png'), bbox_inches='tight', dpi=300)
group_p2_eeg = np.mean(group_p2_eeg.data, axis=1)

group_n1_mag = item_ev_mean.copy().pick_types(eeg=False, meg='mag').crop(tmin=N1_peak_time - 0.025, tmax=N1_peak_time + 0.025)
fig = group_n1_mag.plot_topomap(times=N1_peak_time, average=99.99)
fig.savefig(op.join(results_path, 'mag_n1_comp.png'), bbox_inches='tight', dpi=300)
group_n1_mag = np.mean(group_n1_mag.data, axis=1)

group_p2_mag = item_ev_mean.copy().pick_types(eeg=False, meg='mag').crop(tmin=P2_peak_time - 0.025, tmax=P2_peak_time + 0.025)
fig = group_p2_mag.plot_topomap(times=P2_peak_time, average=99.99)
fig.savefig(op.join(results_path, 'mag_p2_comp.png'), bbox_inches='tight', dpi=300)
group_p2_mag = np.mean(group_p2_mag.data, axis=1)

group_n1_grad = item_ev_mean.copy().pick_types(eeg=False, meg='grad').crop(tmin=N1_peak_time - 0.025, tmax=N1_peak_time + 0.025)
fig = group_n1_grad.plot_topomap(times=N1_peak_time, average=99.99)
fig.savefig(op.join(results_path, 'grad_n1_comp.png'), bbox_inches='tight', dpi=300)
group_n1_grad = np.mean(group_n1_grad.data, axis=1)

group_p2_grad = item_ev_mean.copy().pick_types(eeg=False, meg='grad').crop(tmin=P2_peak_time - 0.025, tmax=P2_peak_time + 0.025)
fig = group_p2_grad.plot_topomap(times=P2_peak_time, average=99.99)
fig.savefig(op.join(results_path, 'grad_p2_comp.png'), bbox_inches='tight', dpi=300)
group_p2_grad = np.mean(group_p2_grad.data, axis=1)

# As a DFs of regressors
group_components_grad = pd.DataFrame(dict(group_n1_grad=group_n1_grad, group_p2_grad=group_p2_grad))
group_components_mag = pd.DataFrame(dict(group_n1_mag=group_n1_mag, group_p2_mag=group_p2_mag))
group_components_eeg = pd.DataFrame(dict(group_n1_grad=group_n1_eeg, group_p2_grad=group_p2_eeg))
# Normalize regressors
for name in list(group_components_grad):
    group_components_grad[name] = scale(group_components_grad[name])
for name in list(group_components_mag):
    group_components_mag[name] = scale(group_components_mag[name])
for name in list(group_components_eeg):
    group_components_eeg[name] = scale(group_components_eeg[name])
# Or MAG-GRAD in the same DF
group_components_mag = group_components_mag.rename(index=str, columns={'group_n1_mag': 'group_n1', 'group_p2_mag': 'group_p2'})
group_components_grad = group_components_grad.rename(index=str, columns={'group_n1_grad': 'group_n1', 'group_p2_grad': 'group_p2'})
group_components_MEG = pd.concat([group_components_mag, group_components_grad])

# =========================================================== #
# Data extraction
# =========================================================== #
for subject in config.subjects_list:
    # subject = config.subjects_list[4]
    print('\n########## peak data extraction for subject ' + subject + " ##########")
    start_time = time.time()

    # Load epochs data
    print('      Loading epochs...')
    if resid_epochs:
        epochs = epoching_funcs.load_resid_epochs_items(subject)
    else:
        if cleaned:
            epochs = epoching_funcs.load_epochs_items(subject, cleaned=True)
        else:
            epochs = epoching_funcs.load_epochs_items(subject, cleaned=False)
            epochs = epoching_funcs.update_metadata_rejected(subject, epochs)
    # crop to only what we need
    epochs = epochs.crop(tmin=-0.050, tmax=0.300)

    # Add surprise100 in metadata
    metadata = epoching_funcs.update_metadata(subject, clean=True, new_field_name=None, new_field_values=None)
    metadata["surprise100"] = metadata["surprise_100.00000"]  # "rename" the variable
    metadata = metadata[metadata.columns.drop(list(metadata.filter(regex='surprise_')))]  # remove all other surprise versions
    # add a balanced_standard yes/no column to metadata (position matched with deviants)
    metadata = epoching_funcs.metadata_balance_epochs_violation_positions(metadata)
    epochs.metadata = metadata
    print("      Preparing data took %.01f minutes ---" % ((time.time() - start_time) / 60))

    #############################"
    nn = 0
    print('      N created evoked =', sep=' ', end=' ', flush=True)
    N1_betas = []
    P2_betas = []
    for ttype in ['habituation', 'standard', 'violation']:
        for seqID in range(1, 8):
            for stimPos in range(1, 17):
                if ttype == 'habituation':
                    selec = epochs['TrialNumber <= 10 and SequenceID == ' + str(seqID) + ' and StimPosition == ' + str(stimPos)].copy()
                elif ttype == 'standard':  # For standards, taking only items from trials with no violation
                    selec = epochs['TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationInSequence == 0 and StimPosition == ' + str(stimPos)].copy()
                elif ttype == 'violation':
                    selec = epochs['TrialNumber > 10 and SequenceID == ' + str(seqID) + ' and ViolationOrNot == 1 and StimPosition == ' + str(stimPos)].copy()
                if len(selec) > 0:
                    # create metadata info for this subset of epochs
                    selecMetadata = pd.DataFrame(selec.metadata.mean()).transpose()  # "average" metadata for this subset of epochs
                    selecMetadata['nave'] = len(selec)
                    selecMetadata['trialtype'] = ttype
                    # create evoked
                    # ev = selec.average().filter(l_freq=None, h_freq=30, n_jobs=n_jobs).detrend().apply_baseline((-0.050, 0))
                    ev = selec.average().apply_baseline((-0.050, 0))  # unfiltered

                    # regression of the components (both meg together) for each timepoint
                    resN1 = []
                    resP2 = []
                    resN1_corr = []
                    resP2_corr = []
                    for timepoint in range(len(ev.times)):
                        ev_timepoint1 = ev.copy().pick_types(eeg=False, meg='mag')._data[:, timepoint]
                        ev_timepoint1 = scale(ev_timepoint1.reshape(-1, 1))
                        ev_timepoint2 = ev.copy().pick_types(eeg=False, meg='grad')._data[:, timepoint]
                        ev_timepoint2 = scale(ev_timepoint2.reshape(-1, 1))  # concatenate normalized MAG and normalized GRAD
                        ev_timepoint = np.concatenate([ev_timepoint1, ev_timepoint2])
                        # with multiple linear regression
                        reg = LinearRegression().fit(ev_timepoint, group_components_MEG)
                        resN1.append(reg.coef_[0][0])
                        resP2.append(reg.coef_[1][0])
                        # # with pearson correlation ! same results
                        # rho, pval = pearsonr(pd.DataFrame(ev_timepoint)[0], group_components_MEG['group_n1'])
                        # resN1_corr.append(rho)
                        # rho, pval = pearsonr(pd.DataFrame(ev_timepoint)[0], group_components_MEG['group_p2'])
                        # resP2_corr.append(rho)
                    # plt.figure()
                    # plt.plot(ev.times, resN1, label='N1 comp.')
                    # plt.plot(ev.times, resP2, label='P2 comp.')
                    # plt.legend()
                    # plt.figure()
                    # plt.plot(ev.times, resN1_corr, label='N1 comp.')
                    # plt.plot(ev.times, resP2_corr, label='P2 comp.')
                    # plt.legend()
                    N1_betas.append(resN1)
                    P2_betas.append(resP2)

                    # # extract peak data
                    # channel, latency, value = ev.copy().pick(picks=mag_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True, ch_type='mag')
                    # selecMetadata['n1_mag_val'] = value
                    # selecMetadata['n1_mag_lat'] = latency
                    # channel, latency, value = ev.copy().pick(picks=mag_roi).get_peak(tmin=p2m_timewin[0], tmax=p2m_timewin[1], return_amplitude=True, ch_type='mag')
                    # selecMetadata['p2_mag_val'] = value
                    # selecMetadata['p2_mag_lat'] = latency
                    # channel, latency, value = ev.copy().pick(picks=grad_roi).get_peak(tmin=n1m_timewin[0], tmax=n1m_timewin[1], return_amplitude=True, ch_type='grad', merge_grads=False)
                    # selecMetadata['n1_grad_val'] = value
                    # selecMetadata['n1_grad_lat'] = latency
                    # channel, latency, value = ev.copy().pick(picks=grad_roi).get_peak(tmin=p2m_timewin[0], tmax=p2m_timewin[1], return_amplitude=True, ch_type='grad', merge_grads=False)
                    # selecMetadata['p2_grad_val'] = value
                    # selecMetadata['p2_grad_lat'] = latency
                    # channel, latency, value = ev.copy().pick(picks=eeg_roi).get_peak(tmin=n1_timewin[0], tmax=n1_timewin[1], return_amplitude=True, ch_type='eeg')
                    # selecMetadata['n1_eeg_val'] = value
                    # selecMetadata['n1_eeg_lat'] = latency
                    # channel, latency, value = ev.copy().pick(picks=eeg_roi).get_peak(tmin=p2_timewin[0], tmax=p2_timewin[1], return_amplitude=True, ch_type='eeg')
                    # selecMetadata['p2_eeg_val'] = value
                    # selecMetadata['p2_eeg_lat'] = latency

                    # add to the results table
                    if nn == 0:
                        results_data = selecMetadata
                    else:
                        results_data = results_data.append(selecMetadata)
                    nn += 1
                    if nn % 10 == 0:
                        print(nn, sep=' ', end=' ', flush=True)
    N1_betas = np.asarray(N1_betas)
    P2_betas = np.asarray(P2_betas)
    results_data['Subject'] = subject

    # save this...
    print('\n')
    print('Saving N1 betas to ' + op.join(results_path, subject + '_N1_MEG_betas.pickle'))
    with open(op.join(results_path, subject + '_N1_MEG_betas.pickle'), 'wb') as f:
        pickle.dump(N1_betas, f, pickle.HIGHEST_PROTOCOL)
    print('Saving P2 betas to ' + op.join(results_path, subject + '_P2_MEG_betas.pickle'))
    with open(op.join(results_path, subject + '_P2_MEG_betas.pickle'), 'wb') as f:
        pickle.dump(P2_betas, f, pickle.HIGHEST_PROTOCOL)
    print('Saving metadata to ' + op.join(results_path, subject + '_metadata.pickle'))
    with open(op.join(results_path, subject + '_metadata.pickle'), 'wb') as f:
        pickle.dump(results_data, f, pickle.HIGHEST_PROTOCOL)

    print("\n      %.01f epochs per evoked on average (median = %.01f) ---" % (np.mean(results_data.nave), np.median(results_data.nave)))
    del epochs
    # output_file = op.join(results_path, subject + '_metadata_evoked.csv')
    # results_data.to_csv(output_file, index=False)
    # print('      ========> ' + output_file + " saved !")
    print("      --- Took %.01f minutes ---" % ((time.time() - start_time) / 60))

    # # ======= FIGURES (one subject)
    # plt.close('all')
    # plt.figure()
    # for seqID in range(1, 8):
    #     ids = np.where(results_data['SequenceID'] == seqID)
    #     plt.plot(ev.times, np.mean(N1_betas[ids], axis=0), label='SeqID'+str(seqID))
    # plt.legend()
    #
    # plt.figure()
    # for seqID in range(1, 8):
    #     ids = np.where(results_data['SequenceID'] == seqID)
    #     plt.plot(ev.times, np.mean(P2_betas[ids], axis=0), label='SeqID'+str(seqID))
    # plt.legend()
    #
    # plt.figure()
    # for stimPos in range(1, 17):
    #     ids = np.where(results_data['StimPosition'] == stimPos)
    #     plt.plot(ev.times, np.mean(N1_betas[ids], axis=0), label='Pos'+str(stimPos))
    # plt.legend()
    #
    # plt.figure()
    # for stimPos in range(1, 17):
    #     ids = np.where(results_data['StimPosition'] == stimPos)
    #     plt.plot(ev.times, np.mean(P2_betas[ids], axis=0), label='Pos'+str(stimPos))
    # plt.legend()
    #
    # plt.figure()
    # for trialtype in ['standard', 'violation']:
    #     ids = np.where(results_data['trialtype'] == trialtype)
    #     plt.plot(ev.times, np.mean(N1_betas[ids], axis=0), label=trialtype)
    # plt.legend()
    #
    # plt.figure()
    # for trialtype in ['standard', 'violation']:
    #     ids = np.where(results_data['trialtype'] == trialtype)
    #     plt.plot(ev.times, np.mean(P2_betas[ids], axis=0), label=trialtype)
    # plt.legend()

# =========================================================== #
# Merge group data
# =========================================================== #
# just to get times...
tmp, _ = evoked_funcs.load_evoked(subject=config.subjects_list[0], filter_name='items_standard_all', filter_not=None)
times = tmp['items_standard_all-'][0].crop(tmin=-0.050, tmax=0.300).times

# regroup all subjects data in the same lists
N1_betas_all = []
P2_betas_all = []
results_data_all = []
for subject in config.subjects_list:
    print(subject)
    with open(op.join(results_path, subject + '_N1_MEG_betas.pickle'), 'rb') as f:
        N1_betas = pickle.load(f)
    with open(op.join(results_path, subject + '_P2_MEG_betas.pickle'), 'rb') as f:
        P2_betas = pickle.load(f)
    with open(op.join(results_path, subject + '_metadata.pickle'), 'rb') as f:
        results_data = pickle.load(f)
    N1_betas_all.append(N1_betas)
    P2_betas_all.append(P2_betas)
    results_data_all.append(results_data)

# =========================================================== #
# plots
# =========================================================== #
plt.close('all')
fig, ax = plt.subplots(figsize=(9, 4))
for trialtype in ['habituation', 'standard', 'violation']:
    group_data = []
    for ii in range(len(config.subjects_list)):
        ids = np.where(results_data_all[ii]['trialtype'] == trialtype)
        group_data.append(np.mean(N1_betas_all[ii][ids], axis=0))
    mean = np.mean(group_data, axis=0)
    ub = mean + sem(group_data, axis=0)
    lb = mean - sem(group_data, axis=0)
    plt.fill_between(times * 1000, ub, lb, alpha=.2)
    plt.plot(times * 1000, mean, label=trialtype)
ax.set(xticks=np.arange(-50, 301, 25), xlim=(-50, 300))
ax.grid(True)
plt.legend()
plt.title('N1 component amplitude')
fig.savefig(op.join(results_path, 'group_N1_trialtype.png'), bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=(9, 4))
for trialtype in ['habituation', 'standard', 'violation']:
    group_data = []
    for ii in range(len(config.subjects_list)):
        ids = np.where(results_data_all[ii]['trialtype'] == trialtype)
        group_data.append(np.mean(P2_betas_all[ii][ids], axis=0))
    mean = np.mean(group_data, axis=0)
    ub = mean + sem(group_data, axis=0)
    lb = mean - sem(group_data, axis=0)
    plt.fill_between(times * 1000, ub, lb, alpha=.2)
    plt.plot(times * 1000, mean, label=trialtype)
ax.set(xticks=np.arange(-50, 301, 25), xlim=(-50, 300))
ax.grid(True)
plt.legend()
plt.title('P2 component amplitude')
fig.savefig(op.join(results_path, 'group_P2_trialtype.png'), bbox_inches='tight', dpi=300)

NUM_COLORS = 7
cm = plt.get_cmap('viridis')
colors_list = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])

fig, ax = plt.subplots(figsize=(9, 4))
for seqID in range(1, 8):
    group_data = []
    for ii in range(len(config.subjects_list)):
        ids = np.where((results_data_all[ii]['SequenceID'] == seqID) & (results_data_all[ii]['trialtype'] == 'standard'))
        group_data.append(np.mean(N1_betas_all[ii][ids], axis=0))
    mean = np.mean(group_data, axis=0)
    ub = mean + sem(group_data, axis=0)
    lb = mean - sem(group_data, axis=0)
    plt.fill_between(times * 1000, ub, lb, alpha=.2, color=colors_list[seqID - 1])
    plt.plot(times * 1000, mean, label='SeqID_' + str(seqID), color=colors_list[seqID - 1])
ax.set(xticks=np.arange(-50, 301, 25), xlim=(-50, 300))
ax.grid(True)
plt.legend()
plt.title('N1 component amplitude')
fig.savefig(op.join(results_path, 'group_N1_seqID.png'), bbox_inches='tight', dpi=300)

fig, ax = plt.subplots(figsize=(9, 4))
for seqID in range(1, 8):
    group_data = []
    for ii in range(len(config.subjects_list)):
        ids = np.where((results_data_all[ii]['SequenceID'] == seqID) & (results_data_all[ii]['trialtype'] == 'standard'))
        group_data.append(np.mean(P2_betas_all[ii][ids], axis=0))
    mean = np.mean(group_data, axis=0)
    ub = mean + sem(group_data, axis=0)
    lb = mean - sem(group_data, axis=0)
    plt.fill_between(times * 1000, ub, lb, alpha=.2, color=colors_list[seqID - 1])
    plt.plot(times * 1000, mean, label='SeqID_' + str(seqID), color=colors_list[seqID - 1])
ax.set(xticks=np.arange(-50, 301, 25), xlim=(-50, 300))
ax.grid(True)
plt.legend()
plt.title('P2 component amplitude')
fig.savefig(op.join(results_path, 'group_P2_seqID.png'), bbox_inches='tight', dpi=300)

NUM_COLORS = 16
cm = plt.get_cmap('viridis')
colors_list = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
fig, ax = plt.subplots(figsize=(9, 4))
for stimPos in range(1, 17):
    group_data = []
    for ii in range(len(config.subjects_list)):
        ids = np.where((results_data_all[ii]['StimPosition'] == stimPos) & (results_data_all[ii]['trialtype'] == 'standard'))
        group_data.append(np.mean(N1_betas_all[ii][ids], axis=0))
    mean = np.mean(group_data, axis=0)
    ub = mean + sem(group_data, axis=0)
    lb = mean - sem(group_data, axis=0)
    plt.fill_between(times * 1000, ub, lb, alpha=.2, color=colors_list[stimPos - 1])
    plt.plot(times * 1000, mean, label='Pos_' + str(stimPos), color=colors_list[stimPos - 1])
ax.set(xticks=np.arange(-50, 301, 25), xlim=(-50, 300))
ax.grid(True)
plt.legend()
plt.title('N1 component amplitude')
fig.savefig(op.join(results_path, 'group_N1_stimPos.png'), bbox_inches='tight', dpi=300)

fig, axs = plt.subplots(7, 16, figsize=(22, 16), sharex=True, sharey=True)
# fig.suptitle('N1 component amplitude', fontsize=12)
for seqID in range(1, 8):
    for stimPos in range(1, 17):
        group_data = []
        for ii in range(len(config.subjects_list)):
            ids = np.where((results_data_all[ii]['StimPosition'] == stimPos) & (results_data_all[ii]['SequenceID'] == seqID) & (results_data_all[ii]['trialtype'] == 'standard'))
            if ids:
                group_data.append(np.mean(N1_betas_all[ii][ids], axis=0))
        mean = np.mean(group_data, axis=0)
        ub = mean + sem(group_data, axis=0)
        lb = mean - sem(group_data, axis=0)
        axs[seqID - 1, stimPos - 1].fill_between(times * 1000, ub, lb, alpha=.2, color='k')
        axs[seqID - 1, stimPos - 1].plot(times * 1000, mean, label='Pos_' + str(stimPos), color='k')
        axs[seqID - 1, stimPos - 1].set(xticks=[], xlim=(-50, 300), yticks=[])
        axs[seqID - 1, stimPos - 1].set_title('Pos_' + str(stimPos))
        if stimPos == 1:
            axs[seqID - 1, stimPos - 1].set_ylabel('SeqID_' + str(seqID))
fig.savefig(op.join(results_path, 'group_N1_stimPosXseqID.png'), bbox_inches='tight', dpi=300)
plt.close()

fig, axs = plt.subplots(7, 16, figsize=(22, 16), sharex=True, sharey=True)
# fig.suptitle('N1 component amplitude', fontsize=12)
for seqID in range(1, 8):
    for stimPos in range(1, 17):
        group_data = []
        for ii in range(len(config.subjects_list)):
            ids = np.where((results_data_all[ii]['StimPosition'] == stimPos) & (results_data_all[ii]['SequenceID'] == seqID) & (results_data_all[ii]['trialtype'] == 'standard'))
            if ids:
                group_data.append(np.mean(P2_betas_all[ii][ids], axis=0))
        mean = np.mean(group_data, axis=0)
        ub = mean + sem(group_data, axis=0)
        lb = mean - sem(group_data, axis=0)
        axs[seqID - 1, stimPos - 1].fill_between(times * 1000, ub, lb, alpha=.2, color='k')
        axs[seqID - 1, stimPos - 1].plot(times * 1000, mean, label='Pos_' + str(stimPos), color='k')
        axs[seqID - 1, stimPos - 1].set(xticks=[], xlim=(-50, 300), yticks=[])
        axs[seqID - 1, stimPos - 1].set_title('Pos_' + str(stimPos))
        if stimPos == 1:
            axs[seqID - 1, stimPos - 1].set_ylabel('SeqID_' + str(seqID))
fig.savefig(op.join(results_path, 'group_P2_stimPosXseqID.png'), bbox_inches='tight', dpi=300)
plt.close()
