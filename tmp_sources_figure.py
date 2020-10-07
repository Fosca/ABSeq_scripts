import mne
import config
import os.path as op
from ABseq_func import *
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
subjects_list = [config.subjects_list[i] for i in [0, 1, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16]]
results_path = op.join(config.result_path, 'Contrasts_tests', 'Viol_vs_Stand')
for analysis_name in ['Viol_vs_Stand_allseq', 'Viol_vs_Stand_seqID1', 'Viol_vs_Stand_seqID2', 'Viol_vs_Stand_seqID3',
                      'Viol_vs_Stand_seqID4', 'Viol_vs_Stand_seqID5', 'Viol_vs_Stand_seqID6', 'Viol_vs_Stand_seqID7']:

    # Load evoked for the 2 conditions & compute delta & compute grand averages
    all_e1 = []
    all_e2 = []
    all_delta = []
    for subject in subjects_list:
        evoked1, path_evo = evoked_funcs.load_evoked(subject=subject, filter_name=analysis_name + '_cond1', filter_not=None, cleaned=True)
        evoked2, path_evo = evoked_funcs.load_evoked(subject=subject, filter_name=analysis_name + '_cond2', filter_not=None, cleaned=True)
        e1 = next(iter(evoked1.values()))[0]
        all_e1.append(e1)
        e2 = next(iter(evoked2.values()))[0]
        # Compute evoked contrast
        delta_evoked = mne.combine_evoked([e1, e2], weights=[1, -1])
        all_e1.append(e1)
        all_e2.append(e2)
        all_delta.append(delta_evoked)
    evoked_allsub_cond1 = mne.grand_average(all_e1)
    evoked_allsub_cond2 = mne.grand_average(all_e2)
    evoked_allsub_delta = mne.grand_average(all_delta)

    # Load  morphed sources
    with open(op.join(results_path, 'allsubs_' + analysis_name + '_cond1_sources.pickle'), 'rb') as f:
        morphed_stcs_cond1 = pickle.load(f)
    with open(op.join(results_path, 'allsubs_' + analysis_name + '_cond2_sources.pickle'), 'rb') as f:
        morphed_stcs_cond2 = pickle.load(f)
    with open(op.join(results_path, 'allsubs_' + analysis_name + '_delta_sources.pickle'), 'rb') as f:
        morphed_stcs_delta = pickle.load(f)

    # Mean morphed_stcs_delta
    n_subjects = len(morphed_stcs_delta)
    mean_morphed_stcs_delta = morphed_stcs_delta[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        mean_morphed_stcs_delta._data += morphed_stcs_delta[sub].data
    mean_morphed_stcs_delta._data /= n_subjects

    # Mean morphed_stcs_cond1
    n_subjects = len(morphed_stcs_cond1)
    mean_morphed_stcs_cond1 = morphed_stcs_cond1[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        mean_morphed_stcs_cond1._data += morphed_stcs_cond1[sub].data
    mean_morphed_stcs_cond1._data /= n_subjects

    # Mean morphed_stcs_cond2
    n_subjects = len(morphed_stcs_cond2)
    mean_morphed_stcs_cond2 = morphed_stcs_cond2[0].copy()  # get copy of first instance
    for sub in range(1, n_subjects):
        mean_morphed_stcs_cond2._data += morphed_stcs_cond2[sub].data
    mean_morphed_stcs_cond2._data /= n_subjects


    # ==================================================================================================================
    # Plot from https://mne.tools/stable/auto_examples/visualization/plot_publication_figure.html
    stclist = [mean_morphed_stcs_cond1, mean_morphed_stcs_cond2, mean_morphed_stcs_delta]
    evlist = [evoked_allsub_cond1, evoked_allsub_cond2, evoked_allsub_delta]
    datainfo = ['Violation', 'Standard', 'Delta']
    for nn in range(3):
        stc = stclist[nn]
        evoked = evlist[nn]
        info = datainfo[nn]
        max_t_val = evoked.get_peak(ch_type='mag')[1]
        for max_t in [.070, .170,  max_t_val]:
            colormap = 'hot'
            clim = dict(kind='value', lims=[2, 4.5, 7])

            # Plot the STC, get the brain image, crop it:
            brain = stc.plot(views='lat', hemi='split', size=(800, 400), subject='fsaverage',
                             subjects_dir=fsMRI_dir, initial_time=max_t, background='w',
                             colorbar=False, clim=clim, colormap=colormap,
                             time_viewer=False, show_traces=False)
            screenshot = brain.screenshot()
            brain.close()
            nonwhite_pix = (screenshot != 255).any(-1)
            nonwhite_row = nonwhite_pix.any(1)
            nonwhite_col = nonwhite_pix.any(0)
            cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

            # Tweak the figure style
            plt.rcParams.update({
                'ytick.labelsize': 'small',
                'xtick.labelsize': 'small',
                'axes.labelsize': 'small',
                'axes.titlesize': 'medium',
                'grid.color': '0.75',
                'grid.linestyle': ':'})

            # figsize unit is inches
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 5.), gridspec_kw=dict(height_ratios=[3, 4]))
            fig.suptitle(analysis_name + ': ' + info, fontsize=12, fontweight='bold')
            evoked_idx = 0
            brain_idx = 1

            # plot the evoked in the desired subplot, and add a line at peak activation
            evoked.pick_types('mag').plot(axes=axes[evoked_idx])
            peak_line = axes[evoked_idx].axvline(max_t, color='#66CCEE', ls='--')
            # custom legend
            axes[evoked_idx].legend(
                [axes[evoked_idx].lines[0], peak_line], ['MEG data', 'Peak time'],
                frameon=True, columnspacing=0.1, labelspacing=0.1,
                fontsize=8, fancybox=True, handlelength=1.8)
            # remove the "N_ave" annotation
            axes[evoked_idx].texts = []
            # Remove spines and add grid
            axes[evoked_idx].grid(True)
            axes[evoked_idx].set_axisbelow(True)
            for key in ('top', 'right'):
                axes[evoked_idx].spines[key].set(visible=False)
            # Tweak the ticks and limits
            axes[evoked_idx].set(yticks=np.arange(-150, 151, 100), xticks=np.arange(-0.1, 0.751, 0.1))
            axes[evoked_idx].set(ylim=[-175, 175], xlim=[-0.1, 0.750])

            # now add the brain to the lower axes
            axes[brain_idx].imshow(cropped_screenshot)
            axes[brain_idx].axis('off')
            axes[brain_idx].set_title('Peak = ' + str('%d' % (max_t*1000)) + ' ms')

            # add a vertical colorbar with the same properties as the 3D one
            divider = make_axes_locatable(axes[brain_idx])
            cax = divider.append_axes('right', size='5%', pad=0.2)
            cbar = mne.viz.plot_brain_colorbar(cax, clim, colormap, label='Activation (F)')

            # tweak margins and spacing
            fig.subplots_adjust(left=0.15, right=0.9, bottom=0.01, top=0.83, wspace=0.1, hspace=0.8)

            # add subplot labels
            for ax, label in zip(axes, 'AB'):
                ax.text(0.03, ax.get_position().ymax, label, transform=fig.transFigure,
                        fontsize=12, fontweight='bold', va='top', ha='left')
            fig.savefig(op.join(results_path, analysis_name+'_' + info + '_'+str('%d' % (max_t*1000))+'ms.png'), dpi=600)
            plt.close(fig)
