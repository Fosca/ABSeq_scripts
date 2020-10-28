#!/bin/bash

# ==============================
# /!\ TO DO:
# Set up parallelization, perhaps see https://github.com/dfsp-spirit/shell-tools/tree/master/gnu_parallel_reconall_minimal
# ==============================
 
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/fs_converted

declare -a sublist=('sub01-pa_190002'
                    'sub02-ch_180036'
                    'sub03-mr_190273'
                    'sub04-rf_190499'
                    'sub05-cr_170417'
                    'sub06-kc_160388'
                    'sub07-jm_100109'
                    'sub08-cc_150418'
                    'sub09-ag_170045'
                    'sub10-gp_190568'
                    'sub11-fr_190151'
                    'sub12-lg_170436'
                    'sub13-lq_180242'
                    'sub14-js_180232'
                    'sub15-ev_070110'
                    'sub16-ma_190185'
                    'sub17-mt_170249'
                    'sub18-eo_190576'
                    'sub19-mg_190180')

### Run Freesurfer recon_all
for sub in ${sublist[@]}
do
  export SUBJECT=$sub
	# If NIFTI 
	# export \MRI_FILE=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/orig_nifti/$SUBJECT/*.nii

	# If DICOM 
	all_dcm_files=(/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/orig_dicom/$SUBJECT/organized/*T1*/*/*.dcm)
	export \MRI_FILE=${all_dcm_files[1]}

	# nohup recon-all -s $SUBJECT -i $MRI_FILE -all
	recon-all -s $SUBJECT -i $MRI_FILE -all
done

### Make higher resolution surface ?
for sub in ${sublist[@]}
do
  export SUBJECT=$sub
  # mkheadsurf -subjid $SUBJECT  # not necessary?... same thing in the next command?
  mne make_scalp_surfaces -s $SUBJECT -d $SUBJECTS_DIR --force --overwrite
done

