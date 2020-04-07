#!/bin/bash
        #PBS -N lreg_sub13-lq_180242.py
        #PBS -q Nspin_bigM 
        #PBS -l walltime=36:00:00
        #PBS -l nodes=1:ppn=1
        #PBS -o /neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/cluster//results_qsub/lreg/std_lreg_sub13-lq_180242.py
        #PBS -e /neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/cluster//results_qsub/lreg/err_lreg_sub13-lq_180242.py 
        cd /neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/cluster//results_qsub/
        python /neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/analysis/cluster//generated_jobs/lreg/lreg_sub13-lq_180242.py