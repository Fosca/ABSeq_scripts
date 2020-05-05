import mne
import os
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *

# =========== FREESURFER MRI RECONSTRUCTION MUST BE DONE BEFORE =========== #
# see freesurfer_recon_all.sh

# =========== DO THIS BEFORE LAUNCHING PYTHON =========== #
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# required for mne.bem.make_watershed_bem

# TEST FREESURFER RECON-ALL IN PYTHON SCRIPT?? // no luck..
# ( $FREESURFER_HOME/SetUpFreeSurfer.sh)
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# OR TEST SOMETHIN LIKE:
# os.environ["$FREESURFER_HOME"] = '/i2bm/local/freesurfer'
# os.system('export SUBJECTS_DIR=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI')
# os.system('export SUBJECT=sub01-pa_190002')
# os.system('$FREESURFER_HOME/SetUpFreeSurfer.sh')
# os.system('source $FREESURFER_HOME/SetUpFreeSurfer.sh')
# os.system('export \PTK=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/sub01-pa_190002/*.nii')
# os.system('recon-all -s $SUBJECT -i $PTK -all')
# os.environ["$FREESURFER_HOME"] = '/i2bm/local/freesurfer'
# os.environ["SUBJECTS_DIR"] = fsMRI_dir
# mne.set_config('$FREESURFER_HOME', '/i2bm/local/freesurfer', set_env=True)

fsMRI_dir = op.join(config.root_path, 'data', 'MRI', 'fs_converted')
subject = config.subjects_list[1]

# source_estimation_funcs.prepare_bem(subject, fsMRI_dir)
# Create coregistration "-trans.fif" file
# mne.gui.coregistration(subject=subject, subjects_dir=fsMRI_dir)
# source_estimation_funcs.create_source_space(subject, fsMRI_dir)
# source_estimation_funcs.forward_solution(subject, fsMRI_dir)
# source_estimation_funcs.inverse_operator(subject)
source_estimation_funcs.source_estimates(subject, 'items_standard_all', evoked_filter_not=None)


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
