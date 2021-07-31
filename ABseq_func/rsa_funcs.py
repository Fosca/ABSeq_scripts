import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/umne/')

from initialization_paths import initialization_paths

from ABseq_func import epoching_funcs, utils, SVM_funcs
import config
import pandas as pd
import numpy as np
from src import umne
import glob
import os.path as op

class fn_template:
    dissim = config.result_path + "rsa/dissim/{:}/{:}_{:}.dmat"

#-----------------------------------------------------------------------------------------------------------------------
def preprocess_and_compute_dissimilarity(subject, metrics, tmin=-0.4, tmax=1.,decim=1,
                                         baseline=(None, 0), clean=True,
                                         which_analysis='SequenceID_StimPosition',
                                         factors_or_interest = ('SequenceID', 'StimPosition')):

    """
    We compute the empirical dissimilarity for the data averaging the epochs across the factors of interest
    :param subject:
    :param metrics: The distance metric for the notion of similarity.
    The following parameters are parameters for the epoching:
    :param tmin:
    :param tmax:
    :param decim:
    :param baseline:
    :param rejection:
    :param which_analysis: 'primitives', 'sequences' or 'primitives_and_sequences'
    :param factors_or_interest: The factors across which the epochs are averaged to compute the dissimilarity
    :return:
    """
    if isinstance(metrics, str):
        metrics = metrics,

    if baseline is None:
        bl_str = '_no_baseline'
    else:
        bl_str = '_baseline'

    epochs_RSA = extract_good_epochs_for_RSA(subject,tmin,tmax,baseline,decim,clean)

    # ========= split half method ================
    inds_per_block, _ = SVM_funcs.train_test_different_blocks(epochs_RSA)

    epochs_1 = epochs_RSA[inds_per_block[0]]
    epochs_2 = epochs_RSA[inds_per_block[1]]

    avg_1 = umne.epochs.average_epochs_by_metadata(epochs_1, factors_or_interest)
    avg_2 = umne.epochs.average_epochs_by_metadata(epochs_2, factors_or_interest)

    del epochs_1
    del epochs_2

    for metric in metrics:
        _compute_and_save_dissimilarity(avg_1, avg_2, which_analysis + '_'.join(factors_or_interest) + bl_str, subject, metric)

    print('Saved.')


#-----------------------------------------------------------------------------------------------------------------------
def extract_good_epochs_for_RSA(subject,tmin,tmax,baseline,decim,clean):
    """
    This function computes and saves the epochs epoched for the RSA.
    :param subject:
    :param tmin:
    :param tmax:
    :param baseline:
    :param decim:
    :param reject:
    :param which_analysis:
    :return:
    """

    epochs = epoching_funcs.load_epochs_items(subject,cleaned=clean)
    epochs.crop(tmin,tmax)
    if decim is not None:
        epochs.decimate(decim)
    if baseline is not None:
        epochs.apply_baseline(True)
    epochs = epochs["TrialNumber > 10 and ViolationInSequence == 0"]

    return epochs


#-----------------------------------------------------------------------------------------------------------------------
def _compute_and_save_dissimilarity(epochs1, epochs2, subdir, subj_id, metric):

    print('\n\nComputing {:} dissimilarity (metric={:})...'.format(subdir, metric))

    dissim = umne.rsa.gen_observed_dissimilarity(epochs1, epochs2, metric=metric, sliding_window_size=100, sliding_window_step=10)

    filename = fn_template.dissim.format(subdir, metric, subj_id)
    utils.create_folder(op.split(filename)[0]+'/')
    print('Saving the dissimilarity matrix to {:}'.format(filename))
    dissim.save(filename)



# ======================================================================================================================
# ======================================================================================================================

#-----------------------------------------------------------------------------------------------------------------------
class dissimilarity:
    """
    Target dissimilarity functions
    Each function gets two dictionnaries containing several metadata fields and returns a dissimilarity score (high = dissimilar)
    """

    # ---------------------------------------------------------
    @staticmethod
    def stim_ID(stim1, stim2):
        """
        How many digits do not appear in the same locations (the digit itself doensn't matter)
        """
        #-- Array indicating to which run the trial belongs
        stim1 = stim1['StimID']
        stim2 = stim2['StimID']

        return 0 if stim1 == stim2 else 1

    # ---------------------------------------------------------
    @staticmethod
    def repeatalter(stim1, stim2):
        """
        How many digits do not appear in the same locations (the digit itself doensn't matter)
        """
        #-- Array indicating to which run the trial belongs
        repeatalter1 = stim1['RepeatAlter']
        repeatalter2 = stim2['RepeatAlter']

        return 0 if repeatalter1 == repeatalter2 else 1
    # ---------------------------------------------------------
    @staticmethod
    def Complexity(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar
        """
        comp1 = stim1['Complexity']
        comp2 = stim2['Complexity']

        return 0 if np.abs(comp2-comp1) else 1
    #---------------------------------------------------------
    @staticmethod
    def ChunkBeg(stim1, stim2):
        """
        Is the position of the first item the same ?
        """
        chunk_beg1 = stim1['ChunkBeginning']
        chunk_beg2 = stim2['ChunkBeginning']

        return 0 if chunk_beg1 == chunk_beg2 else 1
    #---------------------------------------------------------
    @staticmethod
    def ChunkEnd(stim1, stim2):
        """
        Is the position of the first item the same ?
        """
        chunk_end1 = stim1['ChunkEnd']
        chunk_end2 = stim2['ChunkEnd']

        return 0 if chunk_end1 == chunk_end2 else 1
    # ---------------------------------------------------------
    @staticmethod
    def NOpenChunks(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar
        """
        nclosedchunks1 = stim1['OpenedChunks']
        nclosedchunks2 = stim2['OpenedChunks']

        return 0 if np.abs(nclosedchunks2-nclosedchunks1) else 1
    # ---------------------------------------------------------
    @staticmethod
    def ChunkDepth(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar
        """
        nchunks1 = stim1['ChunkDepth']
        nchunks2 = stim2['ChunkDepth']

        return 0 if np.abs(nchunks2-nchunks1) else 1
    # ---------------------------------------------------------
    @staticmethod
    def WithinChunkPos(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar
        """
        chunknumb1 = stim1['ChunkNumber']
        chunknumb2 = stim2['ChunkNumber']

        return 0 if chunknumb1 == chunknumb2 else 1
    # ---------------------------------------------------------
    @staticmethod
    def Surprise(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar
        """
        Surprise1 = stim1['Surprise']
        Surprise2 = stim2['Surprise']

        return 0 if Surprise2 == Surprise1 else 1
    # ---------------------------------------------------------
    @staticmethod
    def OrdinalPos(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar
        """
        ordpos1 = stim1['WithinChunkPosition']
        ordpos2 = stim2['WithinChunkPosition']

        return 0 if ordpos1 == ordpos2 else 1
    # ---------------------------------------------------------
    @staticmethod
    def NClosedChunks(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar
        """
        nopenchunks1 = stim1['ClosedChunks']
        nopenchunks2 = stim2['ClosedChunks']

        return 0 if np.abs(nopenchunks2-nopenchunks1) else 1

# ================================================================================================================

def all_stimuli():

    # ====== we load the metadata from a given participant ==========
    metadata_path = config.data_path+'rsa/all_stimuli.pkl'
    all_stimuli = pd.read_pickle(metadata_path)
    all_stimuli = all_stimuli[np.logical_and(all_stimuli['first_or_second']==1,all_stimuli['violation']==0)]

    all_dict = []
    for primitive in ['rotp1','rotm1','rotp2','rotm2','rotp3','rotm3','sym_point','A','B','H','V']:
        presented_pairs = np.unique(all_stimuli[all_stimuli['primitive']==primitive]['position_pair'])
        for k in range(len(presented_pairs)):
            pres_pair = presented_pairs[k]
            for run_numb in range(2,6):
                all_dict.append(pd.DataFrame([dict(primitive=primitive,position_pair=pres_pair,run_number=run_numb)]))

    df = pd.concat(all_dict)

    return df

#-----------------------------------------------------------------------------------------------------------------------
def gen_predicted_dissimilarity(dissimilarity_func,md=None):
    """
    Generate a predicted dissimilarity matrix (for all stimuli)
    """
    if md is None:
        md = all_stimuli()

    result = umne.rsa.gen_predicted_dissimilarity(dissimilarity_func, md, md)

    return umne.rsa.DissimilarityMatrix([result], md, md)

#-----------------------------------------------------------------------------------------------------------------------
def reshape_matrix_2(dissimilarity_matrix,fields =('SequenceID','StimPosition')):
    """
    The goal of this function is to reshape the dissimilarity matrix. The goal is ultimately to average all the dissimilarity matrices.
    For this function, all the participants should have the same metadata of interest.
    :param diss_mat:
    :param reshape_order: the list of fields that says in which hierarchical order we want to organize the data
    :return:
    """
    meta_original = dissimilarity_matrix.md0
    mapping = {key:[] for key in fields}
    indices = {'initial_index':[],'final_index':[]}

    meta_filter = meta_original.copy()

    counter = 0
    key_values1 = np.unique(meta_original[fields[0]])
    for val1 in key_values1:
        meta_filter1 = meta_original[meta_filter[fields[0]].values == val1]
        key_values2 = np.unique(meta_filter1[fields[1]])
        for val2 in key_values2:
            meta_filter2 = meta_filter1[meta_filter1[fields[1]].values == val2]
            idx = meta_filter2.index[0]
            indices['initial_index'].append(idx)
            indices['final_index'].append(counter)
            mapping[fields[0]].append(val1)
            mapping[fields[1]].append(val2)
            counter += 1

    dissim_final = np.nan*np.ones((dissimilarity_matrix.data.shape[0],counter,counter))

    for m in range(counter):
        ind_m = indices['initial_index'][m]
        if ind_m is not None:
            for n in range(counter):
                ind_n = indices['initial_index'][n]
                if ind_n is not None:
                            dissim_final[:,m,n] = dissimilarity_matrix.data[:,ind_m,ind_n]

    meta_final = pd.DataFrame.from_dict(mapping)

    dissimilarity_matrix.data = dissim_final
    dissimilarity_matrix.md0 = meta_final
    dissimilarity_matrix.md1 = meta_final

    return dissimilarity_matrix


#-----------------------------------------------------------------------------------------------------------------------
def load_and_avg_dissimilarity_matrices(analysis_type_path,keep_initial_shape=False,fields=None):

    files = glob.glob(analysis_type_path+'/*')
    print(files)
    diss_all = []

    for file in files:
        dissimilarity_matrix = np.load(file,allow_pickle=True)
        diss_all.append(dissimilarity_matrix.data)

    diss_all = np.asarray(diss_all)
    dissimilarity_matrix.data = diss_all
    return dissimilarity_matrix