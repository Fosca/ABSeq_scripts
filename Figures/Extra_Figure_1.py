"""
==========================================================
Decoding the higher level structure features
==========================================================
# DESCRIPTION OF THE ANALYSIS
# There are two types of structure decodings :
# categorical decoding : 'ChunkBeginning', 'ChunkEnd', 'RepeatAlter', 'WithinChunkPosition'
# linear regressions : 'OpenedChunks', 'ClosedChunks', 'ChunkDepth'

"""

# ---- import the packages -------
import sys
sys.path.append("/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/")
import initialization_paths
from ABseq_func import *
import config
# ======================================= plotting the categorical results =============================================

prefix = 'full_data_clean_'
suffix = '_score_dict'
chances = {'ChunkBeginning':0.5,'ChunkEnd':0.5,'RepeatAlter':0.5,'WithinChunkPosition':0.25}

for ii, name in enumerate(chances.keys()):
    anal_name = 'feature_decoding/' + prefix + name + suffix
    SVM_funcs.plot_gat_simple(anal_name, config.subjects_list, '/feature_decoding/' + name + '/perf_leaveOneSeq',score_field='score',
                              chance=chances[name],compute_significance=[0,0.6], plot_per_subjects=True,vmin=None,vmax=None)

# ======================================= plotting the linear regression results =============================================

for name in ['OpenedChunks','ClosedChunks','ChunkDepth']:
    anal_name = 'feature_decoding/' + prefix + name + suffix
    SVM_funcs.plot_gat_simple(anal_name, config.subjects_list, '/feature_decoding/'+name+'/_leaveOneSeq', chance=0, score_field='regression',
                    compute_significance=[0,0.6],plot_per_subjects=True,vmin=-0.1,vmax=0.1)