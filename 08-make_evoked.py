import config
import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from ABseq_func import *
from mne.parallel import parallel_func

# # make less parallel runs to limit memory usage
# N_JOBS = max(config.N_JOBS // 4, 1)
N_JOBS = 1  # config.N_JOBS

# ----------------------------------------------------------- #
# ---------- COMPUTE AND SAVE EVOKED OF INTEREST ------------ #
# ----------------------------------------------------------- #

# config.subjects_list = config.subjects_list[:11]

parallel, run_func, _ = parallel_func(evoked_funcs.create_evoked, n_jobs=N_JOBS)
parallel(run_func(subject, cleaned=True, AR_type='local') for subject in config.subjects_list)
# parallel(run_func(subject, cleaned=False) for subject in config.subjects_list)


# parallel, run_func, _ = parallel_func(evoked_funcs.create_evoked_resid, n_jobs=N_JOBS)
# parallel(run_func(subject, resid_epochs_type='reg_repeataltern_surpriseOmegainfinity') for subject in config.subjects_list)


def script_group_avg_and_plot_gfp():
    # ----------------------------------------------------------- #
    # ------------------ LOAD THE EVOKED OF INTEREST ------------ #
    # ----------------------------------------------------------- #

    # all sequences pooled together
    evoked_all_standard, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_all', filter_not=None)
    evoked_all_viol, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_all', filter_not=None)
    evoked_full_seq_all_standard, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_all', filter_not=None)

    # one key per sequence ID
    evoked_standard_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_seq', filter_not='pos')  #
    evoked_viol_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_seq', filter_not='pos')  #
    evoked_full_seq_standard_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq', filter_not=None)
    evoked_full_seq_teststandard_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_teststandard_seq', filter_not=None)
    evoked_full_seq_habituation_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_habituation_seq', filter_not=None)

    evoked_viol_seq1_pos, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq1_pos', filter_not=None)  #
    evoked_viol_seq2_pos, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq2_pos', filter_not=None)  #
    evoked_viol_seq3_pos, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq3_pos', filter_not=None)  #
    evoked_viol_seq4_pos, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq4_pos', filter_not=None)  #
    evoked_viol_seq5_pos, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq5_pos', filter_not=None)  #
    evoked_viol_seq6_pos, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq6_pos', filter_not=None)  #
    evoked_viol_seq7_pos, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq7_pos', filter_not=None)  #

    evoked_standard_seq1, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq1', filter_not='pos')  #
    evoked_standard_seq2, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq2', filter_not='pos')  #
    evoked_standard_seq3, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq3', filter_not='pos')  #
    evoked_standard_seq4, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq4', filter_not='pos')  #
    evoked_standard_seq5, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq5', filter_not='pos')  #
    evoked_standard_seq6, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq6', filter_not='pos')  #
    evoked_standard_seq7, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq7', filter_not='pos')  #

    def butterfly_grand_avg():

        evoked_funcs.plot_butterfly_items_allsubj(evoked_standard_seq, violation_or_not=0, apply_baseline=True)
        evoked_funcs.plot_butterfly_items_allsubj(evoked_viol_seq, violation_or_not=1, apply_baseline=True)

    # ----------------------------------------------------------- #
    # ------------------ EXTRACT GROUP GFP DATA ----------------- #
    # ----------------------------------------------------------- #

    def plot_gfp_super_cool(evoked_list, full_sequence=True, ch_type='eeg', save_path='', fig_name='', labels=None):
        if full_sequence:
            fig, ax = plt.subplots(1, 1, figsize=(18, 4))
            plt.axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(16):
                plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            plt.axvline(0, linestyle='-', color='black', linewidth=2)
            for xx in range(3):
                plt.axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
        ax.set_xlabel('Time (ms)')
        fig.suptitle('GFP (%s)' % ch_type + ', N subjects=' + str(len(evoked_list[next(iter(evoked_list))])), fontsize=12)
        fig_name_save = save_path + op.sep + 'GFP_' + fig_name + '_' + ch_type + '.png'
        plot_evoked_list(evoked_list, ch_type=ch_type, labels=labels)
        plt.legend(loc='upper right', fontsize=9)

        if full_sequence:
            ax.set_xlim([-500, 4250])
            # ax.set_ylim([0, 8e-25])
        else:
            ax.set_xlim([-100, 750])
        fig.savefig(fig_name_save, bbox_inches='tight', dpi=300)
        plt.close('all')

    def plot_evoked_list(evoked_list, ch_type='eeg', labels=None):
        for ll, cond in enumerate(evoked_list.keys()):
            NUM_COLORS = len(evoked_list.keys())
            if NUM_COLORS > 1:
                cm = plt.get_cmap('viridis')
                colorslist = ([cm(1. * i / (NUM_COLORS - 1)) for i in range(NUM_COLORS)])
            else:
                colorslist = 'k'
            gfp_cond, times = GFP_funcs.gfp_evoked(evoked_list[cond], baseline=-0.100)
            GFP_funcs.plot_GFP_with_sem(gfp_cond[ch_type], times * 1000, color_mean=colorslist[ll], label=labels[ll], filter=True)

        return plt.gcf()


    for ch_type in config.ch_types:
        plot_gfp_super_cool(evoked_viol_seq, full_sequence=False, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP', fig_name='viol',
                            labels=['Seq_%i' % r for r in range(1, 8)])
        plot_gfp_super_cool(evoked_standard_seq, full_sequence=False, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='standard', labels=['Seq_%i' % r for r in range(1, 8)])
        plot_gfp_super_cool(evoked_full_seq_standard_seq, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard', labels=['Seq_%i' % r for r in range(1, 8)])

        plot_gfp_super_cool(evoked_all_standard, full_sequence=False, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='all_standard', labels=['all_sequences'])
        plot_gfp_super_cool(evoked_all_viol, full_sequence=False, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='all_viol', labels=['all_sequences'])
        plot_gfp_super_cool(evoked_full_seq_all_standard, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_all_standard', labels=['all_sequences'])

    for ch_type in config.ch_types:
        plot_gfp_super_cool(evoked_viol_seq1_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq1_pos', labels=list(evoked_viol_seq1_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq2_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq2_pos', labels=list(evoked_viol_seq2_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq3_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq3_pos', labels=list(evoked_viol_seq3_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq4_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq4_pos', labels=list(evoked_viol_seq4_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq5_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq5_pos', labels=list(evoked_viol_seq5_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq6_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq6_pos', labels=list(evoked_viol_seq6_pos.keys()))
        plot_gfp_super_cool(evoked_viol_seq7_pos, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='viol_seq7_pos', labels=list(evoked_viol_seq7_pos.keys()))

    for ch_type in config.ch_types:
        plot_gfp_super_cool(evoked_standard_seq1, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq1', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq2, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq2', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq3, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq3', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq4, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq4', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq5, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq5', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq6, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq6', labels=[''])
        plot_gfp_super_cool(evoked_standard_seq7, full_sequence=True, ch_type=ch_type,
                            save_path=config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP',
                            fig_name='full_seq_standard_seq7', labels=[''])


def script_generate_heatmap_gfp_figures():
    # ----------------------------------------------------------- #
    # ---------------- GROUP GFP HEATMAP FIGURES ---------------- #
    # ----------------------------------------------------------- #

    filterdata = False
    # Load required evoked data
    evoked_full_seq_teststandard_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_teststandard_seq', filter_not=None, cleaned=True)
    evoked_full_seq_habituation_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_habituation_seq', filter_not=None, cleaned=True)
    evoked_viol_seq_pos = dict()
    for seqID in range(1, 8):
        evoked_viol_seq_pos['seq' + str(seqID)], _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq' + str(seqID) + '_pos', filter_not=None, cleaned=True)

    if config.noEEG:
        ch_types = ['mag', 'grad']
    else:
        ch_types = ['mag', 'grad', 'eeg']

    for ch_type in ch_types:
        # Compute & store average gfp vectors in a data_to_plot dict (gfp per subject, then group average)
        data_to_plot = {}
        data_to_plot['hab'] = {}
        data_to_plot['teststand'] = {}
        data_to_plot['violpos1'] = {}
        data_to_plot['violpos2'] = {}
        data_to_plot['violpos3'] = {}
        data_to_plot['violpos4'] = {}
        for seqID in range(1, 8):
            # Habituation trials
            gfp_all_subs, times = GFP_funcs.gfp_evoked(evoked_full_seq_habituation_seq['full_seq_habituation_seq' + str(seqID) + '-'])
            data_to_plot['hab']['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
            if filterdata:
                data_to_plot['hab']['seq' + str(seqID)] = savgol_filter(np.mean(gfp_all_subs[ch_type], axis=0), window_length=13, polyorder=3)
            else:
                data_to_plot['hab']['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
            # TestStandard trials
            gfp_all_subs, times = GFP_funcs.gfp_evoked(evoked_full_seq_teststandard_seq['full_seq_teststandard_seq' + str(seqID) + '-'])
            if filterdata:
                data_to_plot['teststand']['seq' + str(seqID)] = savgol_filter(np.mean(gfp_all_subs[ch_type], axis=0), window_length=13, polyorder=3)
            else:
                data_to_plot['teststand']['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
            # Violation trials
            seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqID)
            for n, devpos in enumerate(violation_positions):
                gfp_all_subs, times = GFP_funcs.gfp_evoked(evoked_viol_seq_pos['seq' + str(seqID)]['full_seq_viol_seq' + str(seqID) + '_pos' + str(devpos) + '-'])
                data_to_plot['violpos' + str(n + 1)]['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
                if filterdata:
                    data_to_plot['violpos' + str(n + 1)]['seq' + str(seqID)] = savgol_filter(np.mean(gfp_all_subs[ch_type], axis=0), window_length=13, polyorder=3)
                else:
                    data_to_plot['violpos' + str(n + 1)]['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)

        # Draw & save heatmap figure
        evoked_funcs.allsequences_heatmap_figure(data_to_plot, times, cmap_style='unilateral', fig_title='GFP ' + ch_type,
                                                 file_name=op.join(config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP', 'GFP_full_seq_allconds_heatmap_' + ch_type + '.png'))


def script_allsensors_heatmap_figures():
    # ------------------------------------------------------------------ #
    # ---------------- GROUP ALLSENSORS HEATMAP FIGURES ---------------- #
    # ------------------------------------------------------------------ #

    # ========= Load evoked data

    # # all sequences pooled together
    # evoked_all_standard = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_all', filter_not=None)
    # evoked_all_viol = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_all', filter_not=None)
    # evoked_full_seq_all_standard = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_all', filter_not=None)
    #
    # # one key per sequence ID
    # evoked_standard_seq = evoked_funcs.load_evoked(subject='all', filter_name='items_standard_seq', filter_not='pos')  #
    # evoked_viol_seq = evoked_funcs.load_evoked(subject='all', filter_name='items_viol_seq', filter_not='pos')  #
    # evoked_full_seq_standard_seq = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_standard_seq', filter_not=None)
    # evoked_full_seq_teststandard_seq = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_teststandard_seq', filter_not=None)
    # evoked_full_seq_habituation_seq = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_habituation_seq', filter_not=None)

    # ================= PLOT THE HEATMAPS OF THE GROUP-AVERAGED DATA per CHANNEL ================ #
    fig_path = op.join(config.fig_path, 'Evoked_and_GFP_plots', 'GROUP')
    utils.create_folder(fig_path)

    # evoked = evoked_all_standard[list(evoked_all_standard.keys())[0]][0]  # first key (only one key when all sequences combined)
    # evoked = evoked[0]
    evokeds_name = 'items_standard_seq'
    evoked, _ = evoked_funcs.load_evoked(subject='all', filter_name=evokeds_name, filter_not='pos')  #

    # Loop over the 3 ch_types
    plt.close('all')
    for ch_type in config.ch_types:
        fig, axes = plt.subplots(1, len(evoked.keys()), figsize=(len(evoked.keys()) * 4, 10), sharex=False, sharey=False, constrained_layout=True)
        fig.suptitle(ch_type, fontsize=12, weight='bold')
        ax = axes.ravel()[::1]
        # Loop over the different seq
        for x, condname in enumerate(evoked.keys()):
            # ---- Data
            all_evokeds = [evoked[condname][i][0] for i in range(len(evoked[condname]))]
            evokedcondmean = mne.grand_average(all_evokeds)
            if ch_type == 'eeg':
                data = evokedcondmean.copy().pick_types(eeg=True, meg=False).data
                data = data * 1e6  # scaling?
                clim = [-1.5, 1.5]
            elif ch_type == 'mag':
                data = evokedcondmean.copy().pick_types(eeg=False, meg='mag').data
                data = data * 1e15  # scaling?
                clim = [-70, 70]
            elif ch_type == 'grad':
                data = evokedcondmean.copy().pick_types(eeg=False, meg='grad').data
                data = data * 1e13  # scaling?
                clim = [-20, 20]
            minT = min(evokedcondmean.times) * 1000
            maxT = max(evokedcondmean.times) * 1000
            # ---- Plot
            im = ax[x].imshow(data, origin='upper', extent=[minT, maxT, data.shape[0], 0], aspect='auto', cmap='viridis', clim=clim)  # cmap='RdBu_r'
            ax[x].axvline(0, linestyle='-', color='black', linewidth=1)
            # for xx in range(17):
            #     ax[x].axvline(250 * xx, linestyle='--', color='black', linewidth=0.5)
            ax[x].set_xlabel('Time (ms)')
            ax[x].set_ylabel('Channels')
            ax[x].set_title(condname, loc='center', weight='normal')
        fig.colorbar(im, ax=ax, shrink=0.5, location='right')
        fig_name = fig_path + op.sep + (evokeds_name + ch_type + '.png')
        print('Saving ' + fig_name)
        plt.savefig(fig_name, dpi=300)
        plt.close(fig)

    filterdata = True

    evoked_full_seq_teststandard_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_teststandard_seq', filter_not=None)
    evoked_full_seq_habituation_seq, _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_habituation_seq', filter_not=None)
    evoked_viol_seq_pos = dict()
    for seqID in range(1, 8):
        evoked_viol_seq_pos['seq' + str(seqID)], _ = evoked_funcs.load_evoked(subject='all', filter_name='full_seq_viol_seq' + str(seqID) + '_pos', filter_not=None)

    for ch_type in config.ch_types:
        # Compute & store average gfp vectors in a data_to_plot dict (gfp per subject, then group average)
        data_to_plot = {}
        data_to_plot['hab'] = {}
        data_to_plot['teststand'] = {}
        data_to_plot['violpos1'] = {}
        data_to_plot['violpos2'] = {}
        data_to_plot['violpos3'] = {}
        data_to_plot['violpos4'] = {}
        for seqID in range(1, 8):
            # Habituation trials
            gfp_all_subs, times = GFP_funcs.gfp_evoked(evoked_full_seq_habituation_seq['full_seq_habituation_seq' + str(seqID) + '-'])
            data_to_plot['hab']['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
            if filterdata:
                data_to_plot['hab']['seq' + str(seqID)] = savgol_filter(np.mean(gfp_all_subs[ch_type], axis=0), window_length=13, polyorder=3)
            else:
                data_to_plot['hab']['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
            # TestStandard trials
            gfp_all_subs, times = GFP_funcs.gfp_evoked(evoked_full_seq_teststandard_seq['full_seq_teststandard_seq' + str(seqID) + '-'])
            if filterdata:
                data_to_plot['teststand']['seq' + str(seqID)] = savgol_filter(np.mean(gfp_all_subs[ch_type], axis=0), window_length=13, polyorder=3)
            else:
                data_to_plot['teststand']['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
            # Violation trials
            seqname, seqtxtXY, violation_positions = epoching_funcs.get_seqInfo(seqID)
            for n, devpos in enumerate(violation_positions):
                gfp_all_subs, times = GFP_funcs.gfp_evoked(evoked_viol_seq_pos['seq' + str(seqID)]['full_seq_viol_seq' + str(seqID) + '_pos' + str(devpos) + '-'])
                data_to_plot['violpos' + str(n + 1)]['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)
                if filterdata:
                    data_to_plot['violpos' + str(n + 1)]['seq' + str(seqID)] = savgol_filter(np.mean(gfp_all_subs[ch_type], axis=0), window_length=13, polyorder=3)
                else:
                    data_to_plot['violpos' + str(n + 1)]['seq' + str(seqID)] = np.mean(gfp_all_subs[ch_type], axis=0)

        # Draw & save heatmap figure
        evoked_funcs.allsequences_heatmap_figure(data_to_plot, times, cmap_style='unilateral', fig_title='GFP ' + ch_type,
                                                 file_name=op.join(config.fig_path + op.sep + 'Evoked_and_GFP_plots' + op.sep + 'GROUP', 'GFP_full_seq_allconds_heatmap_' + ch_type + '.png'))


# script_group_avg_and_plot_gfp()
script_generate_heatmap_gfp_figures()
# script_allsensors_heatmap_figures()


