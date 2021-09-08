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
import matplotlib.pyplot as plt
cm = plt.get_cmap('viridis')

class fn_template:
    dissim = config.result_path + "rsa/dissim/{:}/{:}_{:}.dmat"

#-----------------------------------------------------------------------------------------------------------------------
def preprocess_and_compute_dissimilarity(subject, metrics, tmin=-0.4, tmax=1.,decim=1,
                                         baseline=(None, 0), clean=True,
                                         which_analysis='SequenceID_StimPosition',
                                         factors_or_interest = ('StimID','SequenceID', 'StimPosition','Complexity','RepeatAlter','ChunkBeginning','ChunkEnd','OpenedChunks','ChunkDepth','ChunkNumber','WithinChunkPosition','ClosedChunks')):

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
def extract_good_epochs_for_RSA(subject,tmin,tmax,baseline,decim,clean,recompute = False):
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

    if recompute:
        print("=== put here the code for recomputing ===")
        epochs = epoching_funcs.run_epochs(subject,epoch_on_first_element=False,tmin=tmin,tmax=tmax,baseline=baseline,whattoreturn = '')
    else:
        epochs = epoching_funcs.load_epochs_items(subject,cleaned=clean)

    epochs.pick_types(meg=True)
    epochs.crop(tmin,tmax)
    if decim is not None:
        epochs.decimate(decim)
    if baseline is not None:
        epochs.apply_baseline(True)
    epochs = epochs["TrialNumber > 10 and ViolationInSequence == 0 and StimPosition > 1"]

    return epochs


#-----------------------------------------------------------------------------------------------------------------------
def _compute_and_save_dissimilarity(epochs1, epochs2, subdir, subj_id, metric):

    print('\n\nComputing {:} dissimilarity (metric={:})...'.format(subdir, metric))

    dissim = umne.rsa.gen_observed_dissimilarity(epochs1, epochs2, metric=metric, sliding_window_size=25, sliding_window_step=4)

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
    @staticmethod
    def InfoType(stim1, stim2):
        """
        Represents the global type of information encoded in each sequences : Transition probas, chunks, nested structures, pure memory
        """
        #-- Array indicating to which run the trial belongs
        seq1 = stim1['SequenceID']
        seq2 = stim2['SequenceID']

        if (seq1 == 1 or seq1 == 2):
            if (seq2 == 1 or seq2 == 2):
                return 0
            else:
                return 1

        if (seq1 == 3 or seq1 == 4):
            if (seq2 == 3 or seq2 == 4):
                return 0
            else:
                return 1

        if (seq1 == 5 or seq1 == 6):
            if (seq2 == 5 or seq2 == 6):
                return 0
            else:
                return 1

        if seq1 == 7:
            if seq2 == 7:
                return 0
            else:
                return 1
    # ---------------------------------------------------------
    @staticmethod
    def SameSeqAndPosition(stim1, stim2):
        """
        How many digits do not appear in the same locations (the digit itself doensn't matter)
        """
        #-- Array indicating to which run the trial belongs
        seq1 = stim1['SequenceID']
        seq2 = stim2['SequenceID']
        pos1 = stim1['StimPosition']
        pos2 = stim2['StimPosition']

        return 0 if (seq1 == seq2 and pos1 == pos2) else 1

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
    def SequenceID(stim1, stim2):
        """
        How many digits do not appear in the same locations (the digit itself doensn't matter)
        """
        #-- Array indicating to which run the trial belongs
        seq1 = stim1['SequenceID']
        seq2 = stim2['SequenceID']

        return 0 if seq1 == seq2 else 1
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

        return np.abs(comp2-comp1)
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
        nopenchunks1 = stim1['OpenedChunks']
        nopenchunks2 = stim2['OpenedChunks']

        return np.abs(nopenchunks2-nopenchunks1)
    # ---------------------------------------------------------
    @staticmethod
    def ChunkDepth(stim1, stim2):
        """
        This matrix is the primitive dissimilarity if we consider all the 12 primitives as dissimilar
        """
        nchunks1 = stim1['ChunkDepth']
        nchunks2 = stim2['ChunkDepth']

        return np.abs(nchunks2-nchunks1)
    # ---------------------------------------------------------
    @staticmethod
    def ChunkNumber(stim1, stim2):
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

        return np.abs(nopenchunks2-nopenchunks1)

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
def reorder_matrix(dissimilarity_matrix, fields =('SequenceID', 'StimPosition')):
    """
    The goal of this function is to reshape the dissimilarity matrix. The goal is ultimately to average all the dissimilarity matrices.
    For this function, all the participants should have the same metadata of interest.
    :param diss_mat:
    :param reshape_order: the list of fields that says in which hierarchical order we want to organize the data
    :return:
    """
    meta_original = dissimilarity_matrix.md0
    mapping = {key:[] for key in fields}
    # indices = {'initial_index':[],'final_index':[]}
    initial_index = []
    meta_filter = meta_original.copy()

    counter = 0
    key_values1 = np.unique(meta_original[fields[0]])
    for val1 in key_values1:
        meta_filter1 = meta_original[meta_filter[fields[0]].values == val1]
        key_values2 = np.unique(meta_filter1[fields[1]])
        for val2 in key_values2:
            meta_filter2 = meta_filter1[meta_filter1[fields[1]].values == val2]
            idx = meta_filter2.index[0]
            initial_index.append(idx)
            # indices['initial_index'].append(idx)
            # indices['final_index'].append(counter)
            # mapping[fields[0]].append(val1)
            # mapping[fields[1]].append(val2)
            counter += 1

    dissim_final = np.nan*np.ones((dissimilarity_matrix.data.shape[0],counter,counter))

    for m in range(counter):
        ind_m = initial_index[m]
        if ind_m is not None:
            for n in range(counter):
                ind_n = initial_index[n]
                if ind_n is not None:
                    dissim_final[:,m,n] = dissimilarity_matrix.data[:,ind_m,ind_n]

    meta_final = meta_original.reindex(initial_index)

    dissimilarity_matrix.data = dissim_final
    dissimilarity_matrix.md0 = meta_final
    dissimilarity_matrix.md1 = meta_final

    return dissimilarity_matrix


#-----------------------------------------------------------------------------------------------------------------------
def load_and_avg_dissimilarity_matrices(analysis_type_path):

    files = glob.glob(analysis_type_path)
    print(files)
    diss_all = []

    count = 0
    for file in files:
        count+=1
        print(count)
        diss_m = np.load(file,allow_pickle=True)
        diss_m = reorder_matrix(diss_m)
        print(diss_m.data.shape)
        diss_all.append(diss_m.data)

    DISS = np.zeros((count,diss_m.data.shape[0],diss_m.data.shape[1],diss_m.data.shape[2]))
    for i in range(count):
        DISS[i,:,:,:] = diss_all[i]

    diss_all_mean = np.mean(DISS,axis=0)
    diss_m.data = diss_all_mean
    return diss_m




def plot_dissimilarity(dissim,vmin=None,vmax = None):

    matrix = dissim.data

    min_val = 0
    if vmax is None:
        max_val = np.mean(matrix) + np.std(matrix)
    else:
        max_val = vmax

    if vmin is None:
        min_val = np.mean(matrix) - np.std(matrix)
    else:
        min_val = vmin

    plt.imshow(matrix, interpolation='none', cmap=cm, origin='upper', vmin=min_val, vmax=max_val)
    plt.colorbar()
    x_ticks = extract_ticks_labels_from_md(dissim.md0)
    y_ticks = extract_ticks_labels_from_md(dissim.md1)
    y = range(matrix.shape[0])
    x = range(matrix.shape[1])
    plt.xticks(x,x_ticks,rotation='vertical')
    plt.yticks(y,y_ticks)


def extract_ticks_labels_from_md(metadata):

    xticks_labels = []
    for m in range(len(metadata)):
        string_fields = ''
        for field in metadata.keys():
            string_fields += '%s_%s_'%(field[:3],str(metadata[field][m]))

        xticks_labels.append(string_fields)

    return xticks_labels


def Predictor_dissimilarity_matrix_and_md(analysis_name):
    dis = dissimilarity
    dissim_mat = np.load(
        "/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/results/rsa/dissim/" + analysis_name + "/spearmanr_sub01-pa_190002.dmat",
        allow_pickle=True)
    dissim_mat = reorder_matrix(dissim_mat, fields=(
    'SequenceID', 'StimPosition', 'Complexity', 'RepeatAlter', 'ChunkBeginning', 'ChunkEnd', 'OpenedChunks',
    'ChunkDepth', 'ChunkNumber', 'WithinChunkPosition', 'ClosedChunks'))
    md = dissim_mat.md1
    diss_matrix = dict()

    diss_matrix['InfoType'] = gen_predicted_dissimilarity(dis.InfoType, md=md)
    diss_matrix['SameSeqAndPosition'] = gen_predicted_dissimilarity(dis.SameSeqAndPosition, md=md)
    diss_matrix['stim_ID'] = gen_predicted_dissimilarity(dis.stim_ID, md=md)
    diss_matrix['Complexity'] = gen_predicted_dissimilarity(dis.Complexity, md=md)
    diss_matrix['SequenceID'] = gen_predicted_dissimilarity(dis.SequenceID, md=md)
    diss_matrix['OrdinalPos'] = gen_predicted_dissimilarity(dis.OrdinalPos, md=md)
    diss_matrix['repeatalter'] = gen_predicted_dissimilarity(dis.repeatalter, md=md)
    diss_matrix['ChunkBeg'] = gen_predicted_dissimilarity(dis.ChunkBeg, md=md)
    diss_matrix['ChunkEnd'] = gen_predicted_dissimilarity(dis.ChunkEnd, md=md)
    diss_matrix['ChunkNumber'] = gen_predicted_dissimilarity(dis.ChunkNumber, md=md)
    diss_matrix['ChunkDepth'] = gen_predicted_dissimilarity(dis.ChunkDepth, md=md)
    diss_matrix['NOpenChunks'] = gen_predicted_dissimilarity(dis.NOpenChunks, md=md)
    diss_matrix['NClosedChunks'] = gen_predicted_dissimilarity(dis.NClosedChunks, md=md)

    return diss_matrix, md, dis, dissim_mat.times