import initialization_paths
from ABseq_func import cluster_funcs
import numpy as np

print('jaime les tomates')

# cluster_funcs.create_qsub('compute_rsa_dissim_matrix', 'rsa', 'rsa', queue='Nspin_long')
# cluster_funcs.create_qsub('compute_rsa_dissim_matrix', 'rsa', 'rsa', queue='Nspin_bigM')

# cluster_funcs.create_qsub('SVM_generate_all_sequences', 'SVM_arGlob', 'SVM_arGlob', queue='Nspin_bigM')

# cluster_funcs.create_qsub('SVM_features_sequence_structure1', 'stru1', 'stru1', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_sequence_structure2', 'stru2', 'stru2', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_sequence_structure3', 'stru3', 'stru3', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_sequence_structure4', 'stru4', 'stru4', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_sequence_structure5', 'stru5', 'stru5', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_sequence_structure6', 'stru6', 'stru6', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_sequence_structure7', 'stru7', 'stru7', queue='Nspin_bigM')

# cluster_funcs.create_qsub('SVM_quad_ordpos', 'quad_ordpos', 'quad_ordpos', queue='Nspin_long')
# cluster_funcs.create_qsub('epoch_items', 'epo_it', 'epo_it', queue='Nspin_bigM')
# cluster_funcs.create_qsub('epoch_full_trial', 'epo_full', 'epo_full', queue='Nspin_bigM')
#cluster_funcs.create_qsub('epoch_items', 'epo_it', 'epo_it', queue='Nspin_bigM', sublist_subjects=['sub10-gp_190568'])
# cluster_funcs.create_qsub('epoch_full_trial', 'epo_full', 'epo_full', queue='Nspin_bigM', sublist_subjects=['sub14-js_180232'])
# cluster_funcs.create_qsub('compute_evoked', 'evo', 'evo', queue='Nspin_bigM')


# cluster_funcs.create_qsub('SVM_features_stimID', 'stimID', 'stimID', queue='Nspin_bigM')# ok

# rerun this once autoreject has cleaned the epochs
# cluster_funcs.create_qsub('compute_evoked', 'evo', 'evo', queue='Nspin_bigM')


# cluster_funcs.create_qsub('compute_sensor_contribution_decoding_standardVSdeviant', 'loc', 'loc', queue='Nspin_bigM')
cluster_funcs.create_qsub('linear_reg', 'reg-final', 'reg-final', queue='Nspin_bigM')

# cluster_funcs.create_qsub('linear_reg1', 'lr1', 'lr1', queue='Nspin_bigM')
# cluster_funcs.create_qsub('linear_reg2', 'lr2', 'lr2', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_full_sequences_16items1', '116its', '116its', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_full_sequences_16items2', '216its', '216its', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_full_sequences_16items3', '316its', '316its', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_full_sequences_16items4', '416its', '416its', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_stimID_eeg', 'stimID_eeg', 'stimID_eeg', queue='Nspin_bigM')

#cluster_funcs.create_qsub('SVM_full_sequences_16items', '16its', '16its', queue='Nspin_bigM')# ok
#cluster_funcs.create_qsub('SVM_features_stimID_eeg', 'stimID_eeg', 'stimID_eeg', queue='Nspin_bigM')

# cluster_funcs.create_qsub('SVM_features_repeatalter', 'ra', 'ra', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_full_sequences_16items', '16its', '16its', queue='Nspin_bigM')# ok
# cluster_funcs.create_qsub('SVM_features_stimID_eeg', 'stimID_eeg', 'stimID_eeg', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_withinchunk', 'chunkpos', 'chunkpos', queue='Nspin_bigM')# ok
# cluster_funcs.create_qsub('SVM_features_withinchunk_train_quads_test_others', 'quad_ordpos', 'quad_ordpos', queue='Nspin_long')
# cluster_funcs.create_qsub('SVM_features_chunkrev', 'rev', 'rev', queue='Nspin_bigM') # ok
# cluster_funcs.create_qsub('SVM_features_number_ofOpenedChunks', 'OpenPar', 'OpenPar', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_repeatalter', 'ra', 'ra', queue='Nspin_bigM')
# cluster_funcs.create_qsub('SVM_features_chunkBeg', 'Beg', 'Beg', queue='Nspin_bigM') # ok
# cluster_funcs.create_qsub('SVM_features_chunkEnd', 'End', 'End', queue='Nspin_bigM') # ok
# cluster_funcs.create_qsub('SVM_features_withinchunk', 'chu', 'chu', queue='Nspin_bigM')
#cluster_funcs.create_qsub('SVM_features_number_ofOpenedChunks', 'op', 'op', queue='Nspin_long', sublist_subjects=['sub10-gp_190568'])

# cluster_funcs.create_qsub('SVM_features_stimID', 'stID', 'stID', queue='Nspin_bigM')
#cluster_funcs.create_qsub('SVM_features_stimID', 'stID', 'stID', queue='Nspin_long')


#cluster_funcs.create_qsub('SVM_1', 'svm1111', 'svm1111', queue='Nspin_long')
#cluster_funcs.create_qsub('SVM_features', 'feat', 'feat', queue='Nspin_long')
#cluster_funcs.create_qsub('SVM_2', 'svm2', 'svm2', queue='Nspin_long')
#cluster_funcs.create_qsub('SVM_3', 'svm3', 'svm3', queue='Nspin_long')



# cluster_funcs.create_qsub('SVM_2', 'svm2', 'svm2', queue='Global_long')
# cluster_funcs.create_qsub('SVM_3', 'svm3', 'svm3', queue='Global_long')

# cluster_funcs.create_qsub('EMS', 'ems', 'ems', queue='Nspin_bigM', sublist_subjects=['sub16-ma_190185'])
# cluster_funcs.create_qsub('linear_reg', 'lreg', 'lreg', queue='Nspin_long')
# cluster_funcs.create_qsub('surprise_omegas_analysis', 'surp_Omega', 'surp_Omega', queue='Nspin_bigM')
# cluster_funcs.create_qsub('compute_posterior_probability', 'pstprob', 'pstprob', queue='Nspin_bigM')
# cluster_funcs.create_qsub('surprise_omegas_analysis', 'coucou', 'coucou', queue='Nspin_bigM')
# cluster_funcs.create_qsub('simplified_linear_regression', 'reg', 'reg', queue='Nspin_bigM')
# cluster_funcs.create_qsub('simplified_with_complexity', 'reg_comp', 'reg_comp', queue='Nspin_bigM')

print('Tomaste')
