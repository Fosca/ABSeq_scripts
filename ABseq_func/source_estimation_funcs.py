import mne
import config
import matplotlib.pyplot as plt
import os.path as op
from ABseq_func import *

# =========== DO THIS BEFORE LAUNCHING PYTHON =========== #
# required for mne.bem.make_watershed_bem
# source $FREESURFER_HOME/SetUpFreeSurfer.sh

# see https://mne.tools/stable/overview/cookbook.html#setting-up-source-space
# and https://github.com/brainthemind/CogBrainDyn_MEG_Pipeline

def prepare_bem(subject, fsMRI_dir):
    meg_subject_dir = op.join(config.meg_dir, subject)

    # Create BEM surfaces from T1 MRI using freesurfer watershed
    print('Subject ' + subject + ': make_watershed_bem ======================')
    mne.bem.make_watershed_bem(subject=subject, subjects_dir=fsMRI_dir, overwrite=True)

    # BEM model meshes (alternative to freesurfer watershed?)
    model = mne.make_bem_model(subject=subject, subjects_dir=fsMRI_dir)
    mne.write_bem_surfaces(op.join(meg_subject_dir, subject + '-5120-5120-5120-bem.fif'), model)

    # BEM solution
    bem_sol = mne.make_bem_solution(model)
    mne.write_bem_solution(op.join(meg_subject_dir, subject + '-5120-5120-5120-bem-sol.fif'), bem_sol)


def create_source_space(subject, fsMRI_dir):
    print('Subject ' + subject + ': create_source_space ======================')

    meg_subject_dir = op.join(config.meg_dir, subject)
    # Create source space
    src = mne.setup_source_space(subject, spacing=config.spacing, subjects_dir=fsMRI_dir)
    mne.write_source_spaces(op.join(meg_subject_dir, subject + '-oct6-src.fif'), src)


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
    meg_subject_dir = op.join(config.meg_dir, subject)
    # Noise covariance
    emptyroom = mne.io.read_raw_fif(op.join(config.meg_dir, subject, 'empty_room_raw.fif'), allow_maxshield=True)  # is it correct to do this??
    cov = mne.compute_raw_covariance(emptyroom)
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


def source_estimates(subject, evoked_filter_name, evoked_filter_not=None):
    print('Subject ' + subject + ': source_estimates: evoked ' + evoked_filter_name + '======================')

    meg_subject_dir = op.join(config.meg_dir, subject)

    # Load evoked
    evoked = evoked_funcs.load_evoked(subject=subject, filter_name=evoked_filter_name, filter_not=evoked_filter_not, cleaned=True)
    evoked = evoked[list(evoked.keys())[0]]  # first key
    # Load inverse operator
    extension = '_%s-inv' % (config.spacing)
    fname_inv = op.join(meg_subject_dir, subject + config.base_fname.format(**locals()))
    inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)

    # Source estimates: apply inverse
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stc = mne.minimum_norm.apply_inverse(evoked[0], inverse_operator, lambda2, "dSPM", pick_ori=None)
    stc.save(op.join(meg_subject_dir, subject + '_dSPM_inverse_' + evoked_filter_name))