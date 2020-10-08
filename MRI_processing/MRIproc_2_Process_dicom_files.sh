#!/bin/bash

# source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/fs_converted
export DCM_SUBJECTS_DIR=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/orig_dicom
export MNE_ROOT=/volatile/MNE-2.7.4-3514-Linux-x86_64 # contains modified version of mne_organize_dicom to deal with dcm file encoding issue
source $MNE_ROOT/bin/mne_setup_sh

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

### Run mne_organize_dicom (script to reorganize dicom files in subfolders)
for sub in ${sublist[@]}
do
  export SUBJECT=$sub
  echo "############ Processing subject" $SUBJECT "############"
  SUBDICOM=$DCM_SUBJECTS_DIR/$SUBJECT/all_imgs
  DEST=$DCM_SUBJECTS_DIR/$SUBJECT/organized
  mkdir $DEST
  cd $DEST
  ## Run script to reorganize dicom files in subfolders
  bash mne_organize_dicom $SUBDICOM
done
