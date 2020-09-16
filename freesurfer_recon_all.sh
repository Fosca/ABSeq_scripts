#!/bin/bash
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/fs_converted
export SUBJECT=sub03-mr_190273
export \MRI_FILE=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/orig_nifti/$SUBJECT/*.nii

nohup recon-all -s $SUBJECT -i $MRI_FILE -all 