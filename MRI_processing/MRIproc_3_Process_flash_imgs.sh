#!/bin/bash

export FREESURFER_HOME=/i2bm/local/freesurfer-6.0.0 # using freesurfer 6.0 since mri_make_bem_surfaces is missing on the 7
# freesurfer_init

# source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/fs_converted
export DCM_SUBJECTS_DIR=/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/data/MRI/orig_dicom
export MNE_ROOT=/volatile/MNE-2.7.4-3514-Linux-x86_64
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

for sub in ${sublist[@]}
do
  ## Run mne_flash_bem script if flash sequence exists
  FLASHFOLD=$DCM_SUBJECTS_DIR/$SUBJECT/organized/*gre_5°_PDW/
  if test -d $FLASHFOLD; then
    echo "Flash sequence found, processing..."
    ln -s $FLASHFOLD flash05
    bash mne_flash_bem --noflash30
  fi
done
