import numpy as np
import sys
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts')
sys.path.append('/neurospin/meg/meg_tmp/ABSeq_Samuel_Fosca2019/scripts/ABSeq_scripts/umne/')
import matplotlib.cm as cm
from sklearn.svm import SVC
import mne
#MNE
#Sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#JR tools
# from jr.gat.scorers import prob_accuracy, scorer_auc, scorer_angle
# from jr.gat.classifiers import SVR_angle, AngularRegression
from sklearn.model_selection import KFold
from math import pi
from ABseq_func import epoching_funcs, utils
import config
from ABseq_func import SVM_funcs

# from sklearn.lda import LDA  # noqa
import matplotlib.pyplot as plt



def compute_sensor_weights_decoder(epochs,decoder,inds_train,labels_train,inds_test,
                                   labels_test,tmin,tmax,n_permutations,n_channels,
                                   scorer=accuracy_score,select_grad = False):

    """
    :param epochs: epochs on which the decoding is ran
    :param inds_train: indices for the epochs on which we train the decoder
    :param labels_train: labels for training
    :param inds_test: indices for testing
    :param labels_test: labels for testing
    :param tmin: for the averaging window
    :param tmax: for the averaging window
    :param n_permutations: number of permutations for the estimation of decoder the performance
    :return:
    """
    scores = []
    # --- group the channels by order of positions ----
    if select_grad:
        epochs.pick_types(meg='grad')
    else:
        epochs.pick_types(meg=True)

    # --- average through time the data ----
    if tmin is not None:
        data = epochs.crop(tmin,tmax)
        data = np.mean(data.get_data(),axis=-1)
    else:
        data = np.squeeze(epochs.get_data())
    # ---
    list_indices_channels_permutations = permutation_sensors(epochs,n_permutations,n_channels,select_grad=select_grad)

    for perm in range(n_permutations):
        print("--- running permutation %i ----"%perm)
        inds_per = list_indices_channels_permutations[perm,:,:]
        inds_per = np.hstack(inds_per)

        data_perm = data[:,inds_per]
        decoder_perm = decoder

        for k in range(len(inds_train)):
            idx_train = inds_train[k]
            lab_train = labels_train[k]
            idx_test = inds_test[k]
            lab_test = labels_test[k]
            X_train = data_perm[idx_train]
            X_test = data_perm[idx_test]
            decoder_perm.fit(X_train[:,:,np.newaxis],lab_train)
            y_pred = decoder_perm.predict(X_test[:,:,np.newaxis])
            scores_perm = scorer(y_pred = np.squeeze(y_pred), y_true = np.squeeze(lab_test))
        # ---- average across the 4 folds ----
        scores.append(np.mean(scores_perm))

    return {'scores':scores,'ch_inds':list_indices_channels_permutations}


def permutation_sensors(epochs,n_permutations, n_channels,select_grad=False):

    positions_channels = epochs._get_channel_positions(picks='meg')
    if select_grad:
        print('----------------- only the gradiometers are considered -------')
        if len(positions_channels)!=204:
            ValueError("This participant doesn't have 204 grad sensors !")

        for k in range(102):
            pos1 = positions_channels[k*2,:]
            pos2 = positions_channels[k * 2+1, :]
            if (pos1 != pos2).any():
                ValueError("There are not 2 sensors at position %i"%k)
        list_indices = []
        for nn in range(n_permutations):
            list_sensors = [i for i in range(102)]
            np.random.shuffle(list_sensors)
            list_sensors = list_sensors[:n_channels]
            positions = [[i*2,i*2+1] for i in list_sensors]
            list_indices.append(positions)
    else:
        print('----------------- all the MEG sensors are considered -------')
        if len(positions_channels)!=306:
            ValueError("This participant doesn't have 306 MEG sensors !")
    # ---- check that channels are grouped by positions, i.e. that the position of 3 consecutive channels are the same ----
        for k in range(102):
            pos1 = positions_channels[k*3,:]
            pos2 = positions_channels[k * 3+1, :]
            pos3 = positions_channels[k * 3+2, :]
            if np.logical_or(np.logical_or( (pos1 != pos2).any(), (pos1 !=pos3).any()),(pos2 !=pos3).any()):
                ValueError("There are not 3 sensors at position %i"%k)
        list_indices = []
        for nn in range(n_permutations):
            list_sensors = [i for i in range(102)]
            np.random.shuffle(list_sensors)
            list_sensors = list_sensors[:n_channels]
            positions = [[i*3,i*3+1,i*3+2] for i in list_sensors]
            list_indices.append(positions)

    list_indices = np.asarray(list_indices)

    return list_indices

def localize_standard_VS_deviant_code(subject,n_permutations = 2000,n_channels = 30,select_grad=False,cleaned=True):

    # ----------- load the epochs ---------------
    epochs = epoching_funcs.load_epochs_items(subject, cleaned=cleaned)
    epochs.pick_types(meg=True)

    # ----------- balance the position of the standard and the deviants -------
    # 'local' - Just make sure we have the same amount of standards and deviants for a given position. This may end up with
    #     1 standards/deviants for position 9 and 4 for the others.
    epochs_balanced = epoching_funcs.balance_epochs_violation_positions(epochs,balance_param="local")
    # ----------- do a sliding window to smooth the data -------
    epochs_balanced = epoching_funcs.sliding_window(epochs_balanced)

    # =============================================================================================
    toi = 0.165
    epochs_for_decoding = epochs_balanced.copy().crop(tmin=toi, tmax = toi)
    training_inds, testing_inds = SVM_funcs.train_test_different_blocks(epochs_for_decoding, return_per_seq=False)
    y_violornot = np.asarray(epochs_for_decoding.metadata['ViolationOrNot'].values)
    labels_train = [y_violornot[training_inds[i]] for i in range(2)]
    labels_test = [y_violornot[testing_inds[i]] for i in range(2)]

    performance_loc = compute_sensor_weights_decoder(epochs_for_decoding,
                                                          SVM_funcs.SVM_decoder(),
                                                          training_inds,
                                                          labels_train,
                                                          testing_inds,
                                                          labels_test, None,
                                                          None, n_permutations,
                                                          n_channels,select_grad=select_grad)

    suffix = ''
    if select_grad:
        suffix = 'only_grad'

    save_path = config.result_path + '/localization/Standard_VS_Deviant/'
    utils.create_folder(save_path)
    save_path_subject = save_path + subject + '/'+suffix
    utils.create_folder(save_path_subject)

    np.save(save_path_subject + 'results'+str(n_permutations)+'_permut'+str(n_channels)+'_chans'+'_'+str(round(toi*1000))+'.npy', performance_loc)


def compute_sensor_weights(subject,analysis_name='Standard_VS_Deviant',results_name='results.npy'):
    """
    Function to build back how much each sensor contributes to the decoding performance
    :param subject: subject id
    :param analysis_name:
    :param results_name:
    :return: list of weights for all the 306 channels
    """
    # ========== load the results =========

    res = np.load(config.result_path+'localization/'+analysis_name+'/'+subject+'/'+results_name,allow_pickle=True).item()

    if 'grad' in results_name:
        sensor_weights = {'%i' % i: [] for i in range(204)}
    else:
        sensor_weights = {'%i'%i:[] for i in range(306)}

    n_permutations = len(res['ch_inds'])
    ch_inds = res['ch_inds']
    scores = res['scores']
    for perm in range(n_permutations):
        for ch in ch_inds[perm]:
            for ii in range(len(ch)):
                sensor_weights[str(ch[ii])].append(scores[perm])

    # this list contains the number of permutations where the channel takes part
    number_permutations = []
    sensor_weights_avg = []
    for key in sensor_weights.keys():
        sensor_weights_avg.append(np.mean(sensor_weights[key]))
        number_permutations.append(len(sensor_weights[key]))

    print('----- the average number of permutation per channel in %i -----'%int(np.mean(number_permutations)))
    print('------ the standard deviation is %i ------ so in proportion %0.02f'%(int(np.std(number_permutations)),int(np.std(number_permutations))/int(np.mean(number_permutations))))


    return sensor_weights_avg


def sensor_weights_all_subj_as_epo(analysis_name='Standard_VS_Deviant',results_name='results.npy'):

    sensor_weights_all = []
    for subject in config.subjects_list:
        sensor_weights_subj = compute_sensor_weights(subject,analysis_name=analysis_name,results_name=results_name)
        sensor_weights_all.append(sensor_weights_subj)

    data = np.asarray(sensor_weights_all)

    # just check in how many permutations each sensor takes part to

    info = np.load(config.result_path+'/localization/info.npy',allow_pickle=True).item()
    if 'grad' in results_name:
        info = np.load(config.result_path+'/localization/info_grads.npy',allow_pickle=True).item()
    data = data[:,:,np.newaxis]

    return mne.EpochsArray(data,info)

def plot_weights_maps(analysis_name='Standard_VS_Deviant',results_name='results.npy',suffix='',chan_types=['mag'],chance=None,vmin=None,vmax=None,font_size = 8):

    save_path = config.fig_path+'/localization/'+analysis_name+'/'
    utils.create_folder(save_path)
    epoch = sensor_weights_all_subj_as_epo(analysis_name=analysis_name,results_name=results_name)

    for chans in chan_types:
        fig = plt.figure(figsize=(3.,2.2))
        layout = mne.find_layout(epoch.info,ch_type=chans)
        data_to_plot = np.squeeze(epoch.copy().pick_types(meg=chans).average()._data)
        if chance is None:
            if 'grad' in chan_types:
                plt.scatter(np.asarray([layout.pos[i, 0] for i in range(0,len(layout.pos),2)]), [layout.pos[i, 1] for i in range(0,len(layout.pos),2)], c=[data_to_plot[i]  for i in range(0,len(layout.pos),2)], s=30, vmin=vmin, vmax=vmax)
            else:
                plt.scatter(layout.pos[:, 0], layout.pos[:, 1], c=data_to_plot, s=30, vmin=vmin, vmax=vmax)

        else:
            if 'grad' in chan_types:
                plt.scatter(np.asarray([layout.pos[i, 0] for i in range(0,len(layout.pos),2)]), [layout.pos[i, 1] for i in range(0,len(layout.pos),2)], c=[data_to_plot[i]-chance  for i in range(0,len(layout.pos),2)], s=30, vmin=vmin, vmax=vmax)
            else:
                plt.scatter(layout.pos[:, 0], layout.pos[:, 1], c=data_to_plot - chance, s=30, vmin=vmin, vmax=vmax)

        # plt.title(analysis_name+chans)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.axis('off')
        cbar = plt.colorbar()
          # Adjust as appropriate.
        cbar.ax.tick_params(labelsize=font_size)
        plt.gcf().savefig(save_path+chans+suffix+'.png')
        plt.gcf().savefig(save_path+chans+suffix+'.svg')
        plt.gcf().show()
        fig = plt.gcf()
        plt.close('all')

    return fig


# plot_weights_maps(analysis_name='Standard_VS_Deviant',results_name='only_gradresults2000_permut20_chans_165.npy',chan_types=['grad'])
