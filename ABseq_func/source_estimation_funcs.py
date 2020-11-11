import mne
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *
import glob
import shutil
import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


# =========== DO THIS BEFORE LAUNCHING PYTHON =========== #
# required for mne.bem.make_watershed_bem
# source $FREESURFER_HOME/SetUpFreeSurfer.sh

# see https://mne.tools/stable/overview/cookbook.html#setting-up-source-space
# and https://github.com/brainthemind/CogBrainDyn_MEG_Pipeline

def prepare_bem(subject, fsMRI_dir):
    meg_subject_dir = op.join(config.meg_dir, subject)

    dcm_subdir = op.join(config.root_path, 'data', 'MRI', 'orig_dicom', subject, 'organized')
    flashfold = glob.glob(op.join(dcm_subdir, '*5°_PDW'))
    if len(flashfold) > 0:
        # In case watershed version was computed before, remove it to avoid confusion
        watersheddir = op.join(config.root_path, 'data', 'MRI', 'fs_converted', subject, 'bem', 'watershed')
        if op.exists(watersheddir):
            shutil.rmtree(watersheddir)
        # Also delete previously created symbolic links (overwrite issue...)
        bemdir = op.join(config.root_path, 'data', 'MRI', 'fs_converted', subject, 'bem')
        files_in_directory = os.listdir(bemdir)
        filtered_files = [file for file in files_in_directory if file.endswith(".surf")]
        for file in filtered_files:
            os.remove(op.join(bemdir, file))
        # Create BEM surfaces from (already converted) 5°Flash MRI using freesurfer(6.0!) mri_make_bem_surfaces
        print('Subject ' + subject + ': make_flash_bem ======================')
        mne.bem.make_flash_bem(subject, overwrite=True, show=False, subjects_dir=fsMRI_dir)
    else:
        # Create BEM surfaces from T1 MRI using freesurfer watershed
        print('Subject ' + subject + ': make_watershed_bem ======================')
        mne.bem.make_watershed_bem(subject=subject, subjects_dir=fsMRI_dir, overwrite=True)

    # BEM model meshes
    model = mne.make_bem_model(subject=subject, subjects_dir=fsMRI_dir)
    mne.write_bem_surfaces(op.join(meg_subject_dir, subject + '-5120-5120-5120-bem.fif'), model, overwrite=True)

    # BEM solution
    bem_sol = mne.make_bem_solution(model)
    mne.write_bem_solution(op.join(meg_subject_dir, subject + '-5120-5120-5120-bem-sol.fif'), bem_sol, overwrite=True)


def create_source_space(subject, fsMRI_dir):
    print('Subject ' + subject + ': create_source_space ======================')

    meg_subject_dir = op.join(config.meg_dir, subject)
    # Create source space
    src = mne.setup_source_space(subject, spacing=config.spacing, subjects_dir=fsMRI_dir)
    mne.write_source_spaces(op.join(meg_subject_dir, subject + '-oct6-src.fif'), src, overwrite=True)


def forward_solution(subject, fsMRI_dir):
    print('Subject ' + subject + ': forward_solution ======================')

    meg_subject_dir = op.join(config.meg_dir, subject)
    # Load some evoked data (just for info?)
    fname_evoked = op.join(meg_subject_dir, 'evoked_cleaned', 'items_standard_all-ave.fif')
    evoked = mne.read_evokeds(fname_evoked)
    # BEM solution
    fname_bem = op.join(meg_subject_dir, '%s-5120-5120-5120-bem-sol.fif' % subject)
    # Coregistration file
    fname_trans = fname_trans = op.join(config.meg_dir, subject, subject + '-trans.fif')
    # Source space
    src = mne.read_source_spaces(op.join(meg_subject_dir, subject + '-oct6-src.fif'))
    # Forward solution
    fwd = mne.make_forward_solution(evoked[0].info, fname_trans, src, fname_bem, mindist=config.mindist)
    extension = '_%s-fwd' % (config.spacing)
    fname_fwd = op.join(meg_subject_dir, subject + config.base_fname.format(**locals()))
    mne.write_forward_solution(fname_fwd, fwd, overwrite=True)


def compute_noise_cov(subject):
    print('Subject ' + subject + ': noise covariance ======================')

    meg_subject_dir = op.join(config.meg_dir, subject)

    # EMPTYROOM WITH MAXFILT -- TEST
    # /!\ BUT NO EEG /!\
    # /!\ AND WE SHOULD ALSO APPLY ICA + EXCLUDE BAD CHANNELS /!\ (which are different across runs...)
    # Apply maxfilter on emptyroom data
    # emptyroom = mne.io.read_raw_fif(op.join(config.meg_dir, subject, 'empty_room_raw.fif'), allow_maxshield=True)
    # emptyroom.fix_mag_coil_types()
    # emptyroom.info['bads'] = []  # /!\ bad channels are kept !! /!\
    # emptyroom_maxfilt = mne.preprocessing.maxwell_filter(emptyroom, calibration=config.mf_cal_fname,cross_talk=config.mf_ctc_fname, st_duration=config.mf_st_duration, origin=config.mf_head_origin, coord_frame="meg")
    # cov = mne.compute_raw_covariance(emptyroom_maxfilt, method=['empirical', 'shrunk'])

    # NOISE COVARIANCE FROM BASELINE OF ALL THE (LONG) EPOCHS
    epochs = epoching_funcs.load_epochs_full_sequence(subject, cleaned=True)
    cov = mne.compute_covariance(epochs, tmax=0, method=['empirical', 'shrunk'], rank='info')
    # To check:
    # fname_evoked = op.join(meg_subject_dir, 'evoked_cleaned', 'items_standard_all-ave.fif')
    # evoked = mne.read_evokeds(fname_evoked)
    # evoked[0].plot_white(cov)
    return cov


def inverse_operator(subject):
    print('Subject ' + subject + ': inverse_operator ======================')

    meg_subject_dir = op.join(config.meg_dir, subject)
    # Noise covariance
    cov = compute_noise_cov(subject)
    # Load some evoked data (just for info?)
    fname_evoked = op.join(meg_subject_dir, 'evoked_cleaned', 'items_standard_all-ave.fif')
    evoked = mne.read_evokeds(fname_evoked)
    # Load forward solution
    extension = '_%s-fwd' % (config.spacing)
    fname_fwd = op.join(meg_subject_dir, subject + config.base_fname.format(**locals()))
    forward = mne.read_forward_solution(fname_fwd)
    # Inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(evoked[0].info, forward, cov, loose=0.2, depth=0.8)
    extension = '_%s-inv' % (config.spacing)
    fname_inv = op.join(meg_subject_dir, subject + config.base_fname.format(**locals()))
    mne.minimum_norm.write_inverse_operator(fname_inv, inverse_operator)


def source_estimates(subject, evoked_filter_name=None, evoked_filter_not=None, evoked_resid=False, apply_baseline=False):
    print('Subject ' + subject + ': source_estimates: evoked >> ' + evoked_filter_name + ' ======================')

    meg_subject_dir = op.join(config.meg_dir, subject)

    # Load evoked
    if evoked_resid:
        evoked, path_evo = evoked_funcs.load_evoked(subject=subject, filter_name=evoked_filter_name, filter_not=evoked_filter_not, evoked_resid=True)
    else:
        evoked, path_evo = evoked_funcs.load_evoked(subject=subject, filter_name=evoked_filter_name, filter_not=evoked_filter_not, cleaned=True)
    evoked = evoked[list(evoked.keys())[0]][0]  # first key

    # Apply baseline
    if apply_baseline:
        evoked.apply_baseline((None, 0))

    # Load inverse operator
    extension = '_%s-inv' % (config.spacing)
    fname_inv = op.join(meg_subject_dir, subject + config.base_fname.format(**locals()))
    inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)

    # Source estimates: apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, "dSPM", pick_ori=None)
    # stc.save(op.join(meg_subject_dir, subject + '_dSPM_inverse_' + evoked_filter_name))
    stc.save(op.join(path_evo, evoked_filter_name + '_dSPM_inverse'))


def compute_sources_from_evoked(subject, evoked, morph_sources=True):

    print('Subject:' + subject + ', computing sources from evoked')
    print('     Computing sources')
    # Load inverse operator
    meg_subject_dir = op.join(config.meg_dir, subject)
    extension = '_%s-inv' % (config.spacing)
    fname_inv = op.join(meg_subject_dir, subject + config.base_fname.format(**locals()))
    inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)

    # Source estimates: apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, "dSPM", pick_ori=None)

    # Morph to fsaverage
    if morph_sources:
        print('     Morph to fsaverage')
        morph = mne.compute_source_morph(stc, subject_from=subject, subject_to='fsaverage', subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'))
        stc_fsaverage = morph.apply(stc)
        stc = stc_fsaverage

    return stc


def load_evoked_with_sources(subject, evoked_filter_name=None, evoked_filter_not=None, evoked_path='evoked_cleaned', apply_baseline=False, lowpass_evoked=True, morph_sources=True):
    """

    :param subject: subject name
    :param evoked_filter_name: element du nom de fichier à inclure
    :param evoked_filter_not: element du nom de fichier à exclure
    :param evoked_path: sous-dossier où récupérer les evoqués (sous 'subject/') : "evoked", "evoked_cleaned" ou "evoked_resid"
    :param apply_baseline: appliquer la baseline avant 0 aux évoqués
    :param morph_sources: normaliser les sources vers fsaverage
    :return:
    """

    # Load evoked
    if evoked_path == 'evoked':
        evoked, path_evo = evoked_funcs.load_evoked(subject=subject, filter_name=evoked_filter_name, filter_not=evoked_filter_not, cleaned=False, evoked_resid=False)
    elif evoked_path == 'evoked_cleaned':
        evoked, path_evo = evoked_funcs.load_evoked(subject=subject, filter_name=evoked_filter_name, filter_not=evoked_filter_not, cleaned=True, evoked_resid=False)
    elif evoked_path == 'evoked_resid':
        evoked, path_evo = evoked_funcs.load_evoked(subject=subject, filter_name=evoked_filter_name, filter_not=evoked_filter_not, cleaned=True, evoked_resid=True)
    evoked = evoked[list(evoked.keys())[0]][0]  # first key

    print('Subject ' + subject + ': evoked and sources from ' + path_evo)

    # Low-pass filter
    if lowpass_evoked:
        print('     Low pass filtering 30Hz')
        evoked = evoked.filter(l_freq=None, h_freq=30)  # default parameters (maybe should filter raw data instead of epochs...)

    # Apply baseline
    if apply_baseline:
        print('     Applying baseline to evoked')
        evoked.apply_baseline(baseline=(-0.050, 0.000))

    # Load inverse operator
    print('     Computing sources')
    meg_subject_dir = op.join(config.meg_dir, subject)
    extension = '_%s-inv' % (config.spacing)
    fname_inv = op.join(meg_subject_dir, subject + config.base_fname.format(**locals()))
    inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)

    # Source estimates: apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, "dSPM", pick_ori=None)

    # Morph to fsaverage
    if morph_sources:
        print('     Morph to fsaverage')
        morph = mne.compute_source_morph(stc, subject_from=subject, subject_to='fsaverage', subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'))
        stc_fsaverage = morph.apply(stc)
        stc = stc_fsaverage

    # Sanity check (high peak value)
    m = np.round(np.max(abs(stc.data)), 0)
    if m > 200:
        raise ValueError('/!\ Probable issue with sources ' + evoked_filter_name + ' for subject ' + subject + ': max value = ' + str(m))
    if m > 80:
        import warnings
        warnings.warn('/!\ Probable issue with sources ' + evoked_filter_name + ' for subject ' + subject + ': max value = ' + str(m))

    return evoked, stc


def source_morph(subject, source_evoked_name):
    print('Subject ' + subject + ': source_morph to fsaverage >> ' + source_evoked_name + ' ======================')

    path_evo = op.join(config.meg_dir, subject, 'evoked_cleaned')
    stc = mne.read_source_estimate(op.join(path_evo, source_evoked_name + '_dSPM_inverse'))
    morph = mne.compute_source_morph(stc, subject_from=subject,
                                     subject_to='fsaverage',
                                     subjects_dir=op.join(config.root_path, 'data', 'MRI', 'fs_converted'))
    stc_fsaverage = morph.apply(stc)
    stc_fsaverage.save(op.join(path_evo, source_evoked_name + '_dSPM_inverse_fsaverage'))

    return stc_fsaverage


def normalized_sources_from_evoked(subject, evoked):
    fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    inverse_operator = mne.minimum_norm.read_inverse_operator(op.join(config.meg_dir, subject, subject + '_oct6-inv.fif'))
    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, "dSPM", pick_ori=None)
    morph = mne.compute_source_morph(stc, subject_from=subject, subjects_dir=fsMRI_dir, subject_to='fsaverage')
    stc_fsaverage = morph.apply(stc)

    return stc_fsaverage


def sources_evoked_figure(stc, evoked, output_file, figure_title, timepoint='max', ch_type='mag', colormap='hot', colorlims='auto', signallims='fixed'):
    """
    Generates a figure showing butterfly of evoked signal (one ch_type) and corresponding sources at a given timepoint
    Uses fsaverage
    Adapted from https://mne.tools/stable/auto_examples/visualization/plot_publication_figure.html)

    :param stc: one surface source estimate (morphed to fsaverage)
    :param evoked: corresponding evoked signals
    :param timepoint: source timepoint to show, can be 'max' or a value
    :param output_file: full path and name with .png extension, to save the figure
    :param figure_title: shown in the figure
    :param ch_type: sensor type to show for evoked
    :param colormap: for the sources figures
    :param colorlims: for the sources figures, can be 'auto' or [min-mid-max]
    :param signallims: set to 'fixed' to used predefined ylims for signals plot
    """
    # Issue when the function is called when RUNNING a script: waits until figure is manually closed...
    # Turn interactive plotting off ?
    plt.ioff()

    fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
    if timepoint == 'max':
        # timeval = stc.get_peak()[1]
        timeval = evoked.get_peak(ch_type=ch_type)[1]
    else:
        timeval = timepoint
    if colorlims == 'auto':
        clim = 'auto'
    else:
        clim = dict(kind='value', lims=colorlims)

    # Plot the STC, get the brain image, crop it:
    brain = stc.plot(views='lat', hemi='split', size=(800, 400), subject='fsaverage',
                     subjects_dir=fsMRI_dir, initial_time=timeval, background='w',
                     colorbar=False, clim=clim, colormap=colormap,
                     time_viewer=False) #, show_traces=False) does not work with mne 0.19
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
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), gridspec_kw=dict(height_ratios=[3, 4]))
    fig.suptitle(figure_title, fontsize=12, fontweight='bold')
    evoked_idx = 0
    brain_idx = 1

    # plot the evoked in the desired subplot, and add a line at peak activation
    evoked.pick_types(ch_type).plot(axes=axes[evoked_idx])
    peak_line = axes[evoked_idx].axvline(timeval, color='#66CCEE', ls='--')
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
    if signallims == 'fixed':
        if ch_type == 'mag':
            axes[evoked_idx].set(yticks=np.arange(-100, 101, 50), xticks=np.arange(-0.1, 0.751, 0.1))
            axes[evoked_idx].set(ylim=[-100, 100], xlim=[-0.1, 0.750])
        elif ch_type == 'grad':
            axes[evoked_idx].set(yticks=np.arange(-50, 51, 25), xticks=np.arange(-0.1, 0.751, 0.1))
            axes[evoked_idx].set(ylim=[-50, 50], xlim=[-0.1, 0.750])

    # now add the brain to the lower axes
    axes[brain_idx].imshow(cropped_screenshot)
    axes[brain_idx].axis('off')
    axes[brain_idx].set_title('Peak = ' + str('%d' % (timeval * 1000)) + ' ms')

    # add a vertical colorbar with the same properties as the 3D one
    divider = make_axes_locatable(axes[brain_idx])
    cax = divider.append_axes('right', size='5%', pad=0.2)
    if mne.__version__ == '0.21.0':
        fmin = brain.data['fmin']
        fmid = brain.data['fmid']
        fmax = brain.data['fmax']
    else:
        fmin = brain.get_data_properties()['fmin']
        fmid = brain.get_data_properties()['fmid']
        fmax = brain.get_data_properties()['fmax']
    cbar = mne.viz.plot_brain_colorbar(cax,  dict(kind='value', lims=[fmin, fmid, fmax]), colormap, label='Activation')

    # tweak margins and spacing
    fig.subplots_adjust(left=0.15, right=0.9, bottom=0.02, top=0.87, wspace=0.1, hspace=0.2)

    # add A/B subplot labels
    # for ax, label in zip(axes, 'AB'):
    #     ax.text(0.03, ax.get_position().ymax, label, transform=fig.transFigure,
    #             fontsize=12, fontweight='bold', va='top', ha='left')
    fig.savefig(output_file, bbox_inches='tight', dpi=600)
    print('========> ' + output_file + " saved !")
    plt.close(fig)
