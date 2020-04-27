import initialization_paths
from ABseq_func import *
import numpy as np

print('jaime les tomates')

# cluster_funcs.create_qsub('epoch_items', 'epo_it', 'epo_it', queue='Nspin_bigM')
# cluster_funcs.create_qsub('epoch_full_trial', 'epo_full', 'epo_full', queue='Nspin_bigM')
# cluster_funcs.create_qsub('epoch_items', 'epo_it', 'epo_it', queue='Nspin_bigM', sublist_subjects=['sub14-js_180232'])
# cluster_funcs.create_qsub('epoch_full_trial', 'epo_full', 'epo_full', queue='Nspin_bigM', sublist_subjects=['sub14-js_180232'])
# cluster_funcs.create_qsub('compute_evoked', 'evo', 'evo', queue='Nspin_bigM')

# rerun this once autoreject has cleaned the epochs
# cluster_funcs.create_qsub('compute_evoked', 'evo', 'evo', queue='Nspin_bigM')
# cluster_funcs.create_qsub('EMS', 'ems', 'ems', queue='Nspin_bigM')
# cluster_funcs.create_qsub('EMS', 'ems', 'ems', queue='Nspin_bigM', sublist_subjects=['sub16-ma_190185'])
# cluster_funcs.create_qsub('linear_reg', 'lreg', 'lreg', queue='Nspin_bigM')
# cluster_funcs.create_qsub('surprise_omegas_analysis', 'surp_Omega', 'surp_Omega', queue='Nspin_bigM')
cluster_funcs.create_qsub('surprise_stuff', 'surstuff', 'surstuff', queue='Nspin_bigM',sublist_subjects=['sub14-js_180232'])


print('Tomaste')
