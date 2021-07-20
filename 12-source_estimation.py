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
# see Process_dicom_files (mne_flash_bem for Flash images)

# =========== DO THIS BEFORE LAUNCHING PYTHON =========== #
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# required for mne.bem.make_watershed_bem

fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
os.environ["SUBJECTS_DIR"] = fsMRI_dir

subjects_list = config.subjects_list
# subjects_list = [config.subjects_list[i] for i in [6, 8, 11, 12]]

# ========================================================================================================================== #
# Prepare BEM surfaces
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

# ========================================================================================================================== #
# /!\ GUI for the coregistration /!\ Create coregistration "-trans.fif" file
# ========================================================================================================================== #
subject = config.subjects_list[6]
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


# ========================================================================================================================== #
# sanitycheck_plot
# ========================================================================================================================== #
def script_sanitycheck_plotactivation():
    # # Exclude some subjects ?
    # config.exclude_subjects.append('sub08-cc_150418')
    # config.subjects_list = list(set(config.subjects_list) - set(config.exclude_subjects))
    # config.subjects_list.sort()

    outfold = op.join(config.fig_path, 'sources_activation')
    utils.create_folder(outfold)
    subjects_list = config.subjects_list
    time_init = 0.170
    for subject in subjects_list:
        source_estimation_funcs.source_estimates(subject, evoked_filter_name='items_standard_all', evoked_filter_not=None, apply_baseline=True)
        stc = mne.read_source_estimate(op.join(config.meg_dir, subject, 'noEEG', 'evoked_cleaned', 'items_standard_all_dSPM_inverse'))
        # stc.plot(views=['lat'], surface='inflated', hemi='split', size=(1200, 600), subject=subject, clim=dict(kind='value', lims=[2, 7.5, 13]), subjects_dir=fsMRI_dir, initial_time=time_init, smoothing_steps=5, time_viewer=False)
        morph = mne.compute_source_morph(stc, subject_from=subject, subjects_dir=fsMRI_dir, subject_to='fsaverage')
        stc_fsaverage = morph.apply(stc)
        brain = stc_fsaverage.plot(views=['lat'], surface='inflated', hemi='split', size=(1200, 600), subject='fsaverage', clim=dict(kind='value', lims=[2, 7.5, 13]),
                                   subjects_dir=fsMRI_dir, initial_time=time_init, smoothing_steps=5, time_viewer=False, backend='mayavi')
        screenshot = brain.screenshot()
        brain.close()
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        fig = plt.imshow(cropped_screenshot)
        plt.axis('off')
        plt.title(subject + ': Evoked items_standard_all, at ' + str('%d' % (time_init * 1000)) + ' ms')
        plt.savefig(op.join(outfold, subject + '_items_standard_all_' + str('%d' % (time_init * 1000)) + 'ms.png'), bbox_inches='tight', dpi=600)
        plt.close('all')

        # =============== Timecourse source figure
        output_file = op.join(outfold, subject + '_items_standard_all_timecourse.png')
        times_to_plot = [.0, .050, .100, .150, .200, .250, .300, .350, .400]
        win_size = .050
        stc = stc_fsaverage
        maxval = np.max(stc._data)
        colorlims = [maxval * .10, maxval * .30, maxval * .80]
        # plot and screenshot for each timewindow
        stc_screenshots = []
        for t in times_to_plot:
            twin_min = t
            twin_max = t + win_size
            stc_timewin = stc.copy()
            stc_timewin.crop(tmin=twin_min, tmax=twin_max)
            stc_timewin = stc_timewin.mean()
            brain = stc_timewin.plot(views=['lat'], surface='inflated', hemi='split', size=(1200, 600), subject='fsaverage', clim=dict(kind='value', lims=colorlims),
                                     subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'), background='w', smoothing_steps=5,
                                     colormap='hot', colorbar=False, time_viewer=False)
            screenshot = brain.screenshot()
            brain.close()
            nonwhite_pix = (screenshot != 255).any(-1)
            nonwhite_row = nonwhite_pix.any(1)
            nonwhite_col = nonwhite_pix.any(0)
            cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
            plt.close('all')
            stc_screenshots.append(cropped_screenshot)
        # main figure
        fig, axes = plt.subplots(len(times_to_plot), 1, figsize=(len(times_to_plot) * 1.1, 4))
        fig.suptitle('items_standard_all', fontsize=6, fontweight='bold')
        for idx in range(len(times_to_plot)):
            axes[idx].imshow(stc_screenshots[idx])
            axes[idx].axis('off')
            twin_min = times_to_plot[idx]
            twin_max = times_to_plot[idx] + win_size
            axes[idx].set_title('[%d - %d ms]' % (twin_min * 1000, twin_max * 1000), fontsize=3, y=0.8)
        # tweak margins and spacing
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.01, top=0.93, wspace=1, hspace=0.3)
        fig.savefig(output_file, bbox_inches='tight', dpi=600)
        print('========> ' + output_file + " saved !")
        plt.close(fig)
