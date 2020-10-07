import mne
import os
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *
from importlib import reload
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

# =========== FREESURFER MRI RECONSTRUCTION MUST BE DONE BEFORE =========== #
# see freesurfer_recon_all.sh

# =========== DO THIS BEFORE LAUNCHING PYTHON =========== #
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# required for mne.bem.make_watershed_bem

fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
os.environ["SUBJECTS_DIR"] = fsMRI_dir

subjects_list = config.subjects_list
subjects_list = [config.subjects_list[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]]

# ========================================================================================================================== #
# Prepare surfaces and coregistration
# ========================================================================================================================== #
for subject in subjects_list:

    # Remove temp files that may block the make_bem_model function ??
    fname = op.join(config.scripts_path, 'brain')
    if os.path.exists(fname):
        os.remove(fname)
    fname = op.join(config.scripts_path, 'inner_skull')
    if os.path.exists(fname):
        os.remove(fname)
    fname = op.join(config.scripts_path, 'outer_skin')
    if os.path.exists(fname):
        os.remove(fname)
    fname = op.join(config.scripts_path, 'outer_skull')
    if os.path.exists(fname):
        os.remove(fname)

    print(subject)
    # Create BEM surfaces & solution
    source_estimation_funcs.prepare_bem(subject, fsMRI_dir)

# /!\ GUI for the coregistration /!\
# Create coregistration "-trans.fif" file
subject = config.subjects_list[11]
mne.gui.coregistration(subject=subject, subjects_dir=fsMRI_dir)

# ========================================================================================================================== #
# Compute source space & inverse solution
# ========================================================================================================================== #
for subject in subjects_list:
    # Save bem fig
    fig = mne.viz.plot_bem(subject=subject, subjects_dir=fsMRI_dir, brain_surfaces='white', orientation='coronal', show=False)
    fig.savefig(op.join(fsMRI_dir, subject, 'bem.jpg'), dpi=300)
    plt.close(fig)

    # Source space, forward & inverse
    source_estimation_funcs.create_source_space(subject, fsMRI_dir)
    source_estimation_funcs.forward_solution(subject, fsMRI_dir)
    source_estimation_funcs.inverse_operator(subject)


def script_sanitycheck_plotactivation():
    outfold = op.join(config.fig_path, 'sources_activation')
    utils.create_folder(outfold)
    subjects_list = [config.subjects_list[i] for i in [0, 1, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16]]
    time_init = 0.170
    for subject in subjects_list:
        source_estimation_funcs.source_estimates(subject, evoked_filter_name='items_standard_all', evoked_filter_not=None)
        stc = mne.read_source_estimate(op.join(config.meg_dir, subject, 'evoked_cleaned', 'items_standard_all_dSPM_inverse'))
        morph = mne.compute_source_morph(stc, subject_from=subject, subjects_dir=fsMRI_dir, subject_to='fsaverage')
        stc_fsaverage = morph.apply(stc)
        brain = stc_fsaverage.plot(views=['lat'], surface='inflated', hemi='split', size=(1200, 600), subject='fsaverage', clim=dict(kind='value', lims=[3, 9, 15]),
                           subjects_dir=fsMRI_dir, initial_time=time_init, smoothing_steps=5, show_traces=False, time_viewer=False)
        screenshot = brain.screenshot()
        brain.close()
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        fig = plt.imshow(cropped_screenshot)
        plt.axis('off')
        plt.title(subject + ': Evoked items_standard_all, at ' + str('%d' % (time_init*1000)) + ' ms')
        plt.savefig(op.join(outfold, subject + '_items_standard_all_' + str('%d' % (time_init*1000)) + 'ms.png'), dpi=600)
        plt.close('all')


def script_plot_sources():
    method = "dSPM"
    stc = mne.read_source_estimate(op.join(config.meg_dir, subject, 'evoked_cleaned', 'items_standard_all_dSPM_inverse'))
    fig, ax = plt.subplots()
    ax.plot(1e3 * stc.times, stc.data[::100, :].T)
    ax.set(xlabel='time (ms)', ylabel='%s value' % method)
    vertno_max, time_max = stc.get_peak(hemi='lh')
    surfer_kwargs = dict(
        surface='pial', hemi='split', subjects_dir=fsMRI_dir, subject=subject,
        clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
        initial_time=time_max, time_unit='s', size=(800, 800), smoothing_steps=10)
    brain = stc.plot(**surfer_kwargs)
    brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
                   scale_factor=0.6, alpha=0.5)
    brain.add_text(0.1, 0.9, 'dSPM (plus location of maximal activation)', 'title',
                   font_size=14)

    meg_subject_dir = op.join(config.meg_dir, subject)
    # Plot evoked & sources
    plt.close('all')
    evoked1 = evoked_funcs.load_evoked(subject=subject, filter_name='items_standard_all', filter_not=None, cleaned=True)
    evoked1 = evoked1[list(evoked1.keys())[0]][0]
    evoked1.pick_types(meg='grad').apply_baseline((None, 0.))
    max_t1 = evoked1.get_peak()[1]
    print('max grad = ' + str(max_t1) + ' ms')
    stc1 = mne.read_source_estimate(op.join(meg_subject_dir, '%s_dSPM_inverse_%s' % (subject, 'items_standard_all')))
    # evoked1.plot()
    # stc1.plot(views='lat', hemi='split', size=(800, 400), subject=subject, subjects_dir=fsMRI_dir, initial_time=max_t1, time_viewer=True, colormap='hot')

    evoked2 = evoked_funcs.load_evoked(subject=subject, filter_name='items_viol_all', filter_not=None, cleaned=True)
    evoked2 = evoked2[list(evoked2.keys())[0]][0]
    evoked2.pick_types(meg='grad').apply_baseline((None, 0.))
    max_t2 = evoked2.get_peak()[1]
    print('max grad = ' + str(max_t2) + ' ms')
    stc2 = mne.read_source_estimate(op.join(meg_subject_dir, '%s_dSPM_inverse_%s' % (subject, 'items_viol_all')))
    # evoked2.plot()
    # stc2.plot(views='lat', hemi='split', size=(800, 400), subject=subject, subjects_dir=fsMRI_dir, initial_time=max_t2, time_viewer=True)

    # From https://mne.tools/dev/auto_examples/visualization/plot_publication_figure.html#sphx-glr-auto-examples-visualization-plot-publication-figure-py
    plt.close('all')
    evoked = evoked2
    stc = stc2
    max_t = max_t2  # 0.070  # max_t2

    colormap = 'hot'
    clim = dict(kind='percent', lims=[0.3, 0.65, 1.0])

    # Plot the STC, get the brain image, crop it:
    brain = stc.plot(views='lat', hemi='split', size=(800, 400), subject=subject, subjects_dir=fsMRI_dir, initial_time=max_t, background='w', colorbar=False,
                     colormap=colormap, clim='auto', time_viewer=False)
    screenshot = brain.screenshot()
    brain.close()
    nonwhite_pix = (screenshot != 255).any(-1)
    nonwhite_row = nonwhite_pix.any(1)
    nonwhite_col = nonwhite_pix.any(0)
    cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

    # # before/after results
    # fig = plt.figure(figsize=(4, 4))
    # axes = ImageGrid(fig, 111, nrows_ncols=(2, 1), axes_pad=0.5)
    # for ax, image, title in zip(axes, [screenshot, cropped_screenshot],
    #                             ['Before', 'After']):
    #     ax.imshow(image)
    #     ax.set_title('{} cropping'.format(title))

    # Tweak the figure style
    plt.rcParams.update({
        'ytick.labelsize': 'small',
        'xtick.labelsize': 'small',
        'axes.labelsize': 'small',
        'axes.titlesize': 'medium',
        'grid.color': '0.75',
        'grid.linestyle': ':',
    })

    # figsize unit is inches
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4.5, 3.), gridspec_kw=dict(height_ratios=[3, 4]))

    # we'll put the evoked plot in the upper axes, and the brain below
    evoked_idx = 0
    brain_idx = 1

    # plot the evoked in the desired subplot, and add a line at peak activation
    evoked.plot(axes=axes[evoked_idx])
    peak_line = axes[evoked_idx].axvline(max_t, color='#66CCEE', ls='--')
    # custom legend
    axes[evoked_idx].legend(
        [axes[evoked_idx].lines[0], peak_line], ['MEG data', 'Peak time'], frameon=True, columnspacing=0.1, labelspacing=0.1, fontsize=8, fancybox=True, handlelength=1.8)
    # remove the "N_ave" annotation
    axes[evoked_idx].texts = []
    # Remove spines and add grid
    axes[evoked_idx].grid(True)
    axes[evoked_idx].set_axisbelow(True)
    for key in ('top', 'right'):
        axes[evoked_idx].spines[key].set(visible=False)
    # Tweak the ticks and limits
    # axes[evoked_idx].set(yticks=np.arange(-200, 201, 100), xticks=np.arange(-0.2, 0.51, 0.1))
    # axes[evoked_idx].set(ylim=[-225, 225], xlim=[-0.2, 0.5])

    # now add the brain to the lower axes
    axes[brain_idx].imshow(cropped_screenshot)
    axes[brain_idx].axis('off')
    # add a vertical colorbar with the same properties as the 3D one
    divider = make_axes_locatable(axes[brain_idx])
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = mne.viz.plot_brain_colorbar(cax, clim=clim, colormap=colormap, label='Activation')

    # tweak margins and spacing
    fig.subplots_adjust(left=0.15, right=0.9, bottom=0.01, top=0.9, wspace=0.1, hspace=0.5)

    # add subplot labels
    for ax, label in zip(axes, 'AB'):
        ax.text(0.03, ax.get_position().ymax, label, transform=fig.transFigure, fontsize=12, fontweight='bold', va='top', ha='left')

    plt.savefig('C:/Users/sp253886/Desktop/FigMEG/test', dpi=300)


def old_script_stuff():
    # TEST PLOT SOURCES
    fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
    subject = config.subjects_list[0]
    meg_subject_dir = op.join(config.meg_dir, subject)
    stc = mne.read_source_estimate(op.join(meg_subject_dir, '%s_mne_dSPM_inverse-%s' % (subject, 'test')))

    stc.plot(subject=subject, subjects_dir=fsMRI_dir, time_viewer=True)

    evoked[0].plot()

    stc.plot(views='lat', hemi='split', size=(800, 400), subject=subject,
             subjects_dir=fsMRI_dir, initial_time=0.0,
             time_viewer=True)

    morph = mne.compute_source_morph(stc, subject_from=subject, subjects_dir=fsMRI_dir, subject_to='fsaverage')
    stc_fsaverage = morph.apply(stc)
    stc_fsaverage.plot()

    stc.plot(subjects_dir=fsMRI_dir)

    # ALTERNATIVE
    # Create BEM surfaces
    # see https://mne.tools/stable/auto_tutorials/source-modeling/plot_forward.html#sphx-glr-auto-tutorials-source-modeling-plot-forward-py
    mne.bem.make_watershed_bem(subject, overwrite=True)

    # Plot BEM surfaces
    tmp = mne.viz.plot_bem(subject=subject, subjects_dir=fsMRI_dir, brain_surfaces='white', orientation='coronal')
    plt.savefig(op.join(fsMRI_dir, subject, 'bem_fig.png'), dpi=300)

    # Coregistration? (manually)
    # see https://www.slideshare.net/mne-python/mnepython-coregistration
    mne.gui.coregistration(subject=subject, subjects_dir=fsMRI_dir)

    #
    fname_trans = op.join(config.meg_dir, subject, subject + '-trans.fif')
    datafile = op.join(config.meg_dir, subject, 'run01_raw.fif')
    info = mne.io.read_info(datafile)
    mne.viz.plot_alignment(info, fname_trans, subject=subject, dig=True, meg=['helmet', 'sensors'], subjects_dir=fsMRI_dir, surfaces='head')
