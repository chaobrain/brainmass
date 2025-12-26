# Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak
#
# Neural Mass Model fitting module for JR with connections from pyramidal to pyramidal, excitatory, and inhibitory populations for M/EEG
# Rewritten using nmm/ modular APIs.

import math
import os
import pickle
import time
from collections import defaultdict

import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import mne
import nibabel
import numpy as np
import pandas as pd
import requests
from braintools.param import Param, Const, ReluT, ExpT, GaussianReg
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity

from brainmass.jansen_rit import JansenRitWindow


def dataloader(emp, TRperwindow):
    assert len(emp.shape) == 2
    # emp: (time_points, node_size)
    length_ts = emp.shape[0]
    node_size = emp.shape[1]
    window_size = int(length_ts / TRperwindow)
    # output: (window_size, TR_per_window, node_size)
    data_out = np.zeros((window_size, TRperwindow, node_size))
    for i_win in range(window_size):
        data_out[i_win] = emp[i_win * TRperwindow:(i_win + 1) * TRperwindow]
    return data_out


class ModelFitting:
    """
    This ModelFitting class is able to fit resting state data or evoked potential data
    for which the input training data is empty or some stimulus to one or more NMM nodes,
    and the label is an associated empirical neuroimaging recording.

    Studies which consider different kinds of input, such as if SC or some other variable
    is associated with an empirical recording, must use a different fitting class.
    """

    def __init__(self, model: JansenRitWindow):
        self.model = model
        # Use a single optimizer with model.parameters()
        self.optimizer = braintools.optim.Adam(lr=0.05)
        self.weights = model.states(brainstate.ParamState)
        self.optimizer.register_trainable_weights(self.weights)

        # define mask for getting lower triangle matrix
        self.mask = np.tril_indices(self.model.output_size, -1)

    def f_loss(self, model_state, inputs, targets):
        params = self.model.retrieve_params()
        model_state, (eeg_output, _) = self.model.update(model_state, params, inputs)

        loss_main = u.math.sqrt(u.math.mean((eeg_output - targets) ** 2))
        loss_prior = []
        for module in self.model.nodes(Param).values():
            loss_prior.append(module.reg_loss())
        loss = loss_main + sum(loss_prior)

        return loss, (model_state, eeg_output, loss_main)

    @brainstate.transform.jit(static_argnums=0)
    def f_train(self, model_state, inputs, targets):
        f_grad = brainstate.transform.grad(
            self.f_loss, self.weights, has_aux=True, return_value=True, check_states=False
        )
        grads, loss, (model_state, eeg_output, loss_main) = f_grad(model_state, inputs, targets)
        self.optimizer.step(grads)
        return loss, loss_main, model_state, eeg_output

    @brainstate.transform.jit(static_argnums=0)
    def f_predict(self, model_state, inputs):
        params = self.model.retrieve_params()
        model_state, output = self.model.update(model_state, params, inputs)
        return model_state, output

    def train(
        self,
        inputs,  # (time_dim, steps_per_TR, node_size) - external stimulation input
        target_eeg,  # (window_size, TR_per_window, output_size) - empirical data
        num_epochs: int,
        TP_per_window: int,
        warmup_window: int = 0,
    ):
        # initial state using nmm API - ModelData contains dynamics_state and delay_state
        model_state = self.model.create_initial_state()

        # Get TRs_per_window from model config
        TRs_per_window = TP_per_window
        loss_his = []  # loss placeholder to take the average for the epoch at the end of the epoch

        # LOOP 1/4: Number of Training Epochs
        for i_epoch in range(num_epochs):
            # Perform the training in windows.
            warmup_windows = 0 if i_epoch == 0 else warmup_window

            # LOOP 2/4: Number of Recordings in the Training Dataset
            external = np.zeros([TRs_per_window, self.model.steps_per_TR, self.model.node_size])
            for TR_i in range(warmup_windows):
                model_state, output = self.f_predict(model_state, external)

            loss_epoch = []
            eeg_epoch = []
            # LOOP 3/4: Number of windowed segments for the recording
            for i_win in range(target_eeg.shape[0]):
                # Slice external input from u
                # u: (time_dim, steps_per_TR, node_size)
                external = inputs[i_win * TRs_per_window:(i_win + 1) * TRs_per_window]

                # Get empirical signal window
                # windowedTS: (window_size, TR_per_window, node_size)
                ts_window = target_eeg[i_win]  # (TR_per_window, node_size)

                # LOOP 4/4: The loop within the forward model (numerical solver),
                # which is number of time points per windowed segment
                loss, loss_main, model_state, eeg_output = self.f_train(model_state, external, ts_window)

                # TRAINING_STATS: Adding Loss for every training window (corresponding to one backpropagation)
                loss_np = np.asarray(loss_main)
                loss_his.append(loss_np)
                loss_epoch.append(loss_np)
                eeg_epoch.append(eeg_output)

            # ts_sim: (total_time, output_size) or (total_time, node_size, 6)
            ts_sim = np.concatenate(eeg_epoch, axis=0)
            # For EEG, shape is (time, output_size), need transpose for correlation
            fc_sim = np.corrcoef(ts_sim[10:].T)
            # target_eeg: (window_size, TR_per_window, node_size) - reshape to (total_time, node_size)
            ts_emp = target_eeg.reshape(-1, target_eeg.shape[-1])
            fc = np.corrcoef(ts_emp.T)
            print(
                f'Epoch: {i_epoch}, '
                f'loss: {np.mean(loss_epoch)}, '
                f'Pseudo FC_cor: {np.corrcoef(fc_sim[self.mask], fc[self.mask])[0, 1]}, '
                f'cos_sim: {np.diag(cosine_similarity(ts_sim.T, ts_emp.T)).mean()}'
            )
        return np.asarray(loss_his)

    def evaluate(
        self,
        u,  # (time_dim, steps_per_TR, node_size) - external stimulation input
        target_eeg,  # (window_size, TR_per_window, output_size) - empirical data
        TR_per_window: int,
        base_window_num: int = 0,
        model_state=None,
        mask=None  # (node_size, node_size) - connection mask
    ):
        # Update connectivity mask if provided
        if mask is not None:
            self.model.mask = mask

        model_state = self.model.create_initial_state() if model_state is None else model_state

        # target_eeg: (window_size, TR_per_window, node_size) - time dimension first
        num_windows = target_eeg.shape[0]
        # u_hat: (total_time, steps_per_TR, node_size) - time dimension first
        total_time = base_window_num * TR_per_window + num_windows * TR_per_window
        u_hat = np.zeros((total_time, self.model.steps_per_TR, self.model.node_size))
        u_hat[base_window_num * TR_per_window:] = u

        # LOOP 1/2: The number of windows in a recording
        eeg_epoch = []
        state_history = defaultdict(list)
        for i_win in range(num_windows + base_window_num):
            # Slice external input from u_hat
            # u_hat: (total_time, steps_per_TR, node_size)
            external = u_hat[i_win * TR_per_window:(i_win + 1) * TR_per_window]

            # LOOP 2/2: The loop within the forward model (numerical solver),
            # which is number of time points per windowed segment
            model_state, (eeg_output, state_hist) = self.f_predict(model_state, external)
            eeg_epoch.append(eeg_output)
            for k, v in state_hist.items():
                if v is not None:
                    state_history[k].append(np.asarray(v))

        i_ignore = base_window_num * TR_per_window

        # ts_sim: (total_time, output_size) or (total_time, node_size, 6)
        ts_sim = np.concatenate(eeg_epoch, axis=0)[i_ignore:]
        state_history = {k: np.concatenate(state_history[k], axis=0) for k in state_history}
        state_history['eeg'] = ts_sim
        # For EEG, shape is (time, output_size), need transpose for correlation
        fc_sim = np.corrcoef(ts_sim.T)

        # target_eeg: (window_size, TR_per_window, node_size) - reshape to (total_time, node_size)
        ts_emp = target_eeg.reshape(-1, target_eeg.shape[-1])
        fc = np.corrcoef(ts_emp.T)
        print(
            'Pseudo FC_cor: ', np.corrcoef(fc_sim[self.mask], fc[self.mask])[0, 1],
            'cos_sim: ', np.diag(cosine_similarity(ts_sim.T, ts_emp.T)).mean()
        )
        return state_history


# download data
data_folder = 'D:/codes/projects/whole-brain-nmm-pytorch/data_momi2025'

# Load Schaefer 200-parcel atlas data
# url = ('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/'
#        'Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/'
#        'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
atlas200_file = os.path.join(data_folder, 'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
atlas200 = pd.read_csv(atlas200_file)

# Load Schaefer 1000-parcel atlas data
# url = ('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/'
#        'Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/'
#        'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
atlas1000_file = os.path.join(data_folder, 'Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
atlas1000 = pd.read_csv(atlas1000_file)

# load network colour
# url = 'https://github.com/Davi1990/DissNet/raw/main/examples/network_colour.xlsx'
net_colour_file = os.path.join(data_folder, 'network_colour.xlsx')

# load the structural connectivity file
# sc_file = 'https://raw.githubusercontent.com/GriffithsLab/PyTepFit/main/data/Schaefer2018_200Parcels_7Networks_count.csv'
sc_file = os.path.join(data_folder, 'Schaefer2018_200Parcels_7Networks_count.csv')

# distance file
dist_file = os.path.join(data_folder, 'Schaefer2018_200Parcels_7Networks_distance.csv')


def empirical_data_analysis():
    start_time = time.time()  # For estimating run time of the empirical analysis
    # Loading network colour filw from the GitHub URL
    colour = pd.read_excel(net_colour_file, header=None)[4]
    # Evoked data
    all_eeg_evoked = np.load(data_folder + '/empirical_data/all_eeg_evoked.npy')
    # Epoched example
    epo_eeg = mne.read_epochs(data_folder + '/empirical_data/example_epoched.fif', verbose=False)
    # GFMA data
    all_gfma = np.zeros((all_eeg_evoked.shape[0], all_eeg_evoked.shape[2]))

    for ses in range(all_eeg_evoked.shape[0]):
        all_gfma[ses, :] = np.std(all_eeg_evoked[ses, :, :], axis=0)  # np.mean(np.mean(epo_eeg._data, axis=0),axis=0)
        # Normalized to the baseline for comparison
        all_gfma[ses, :] = np.abs(all_gfma[ses, :] - np.mean(all_gfma[ses, :300]))

    # Load Schaefer 1000 parcels 7 networks
    with open(data_folder + '/empirical_data/dist_Schaefer_1000parcels_7net.pkl', 'rb') as handle:
        dist_Schaefer_1000parcels_7net = pickle.load(handle)

    # Extract the stimulation region data from the loaded pickle file
    stim_region = dist_Schaefer_1000parcels_7net['stim_region']

    # Plot evoked EEG GFMA at each network

    # 7 networks definition
    networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    # Create a dictionary to store the network indices
    stim_network_indices = {network: [] for network in networks}
    # Iterate over stim region
    for i, label in enumerate(stim_region):
        # Iterate over each network
        for network in networks:
            if network in label:
                stim_network_indices[network].append(i)
                break

    net_gfma = {}
    # Iterate over each network
    for network in networks:
        # GFMA for each network
        net_gfma[network] = all_gfma[stim_network_indices[network]]

    # Get the GFMA averages for each network
    averages = []
    for key, value in net_gfma.items():
        average = sum(value) / len(value)
        averages.append(average)
    averages = np.array(averages)

    # Define the desired figure size
    fig = plt.figure(figsize=(20, 6))
    for net in range(len(networks)):
        plt.plot(epo_eeg.times, averages[net, :] - np.mean(averages[net, :300]), colour[net], linewidth=5)
    plt.show()

    # Plot peaks

    # Calculate the mean array
    time_series = np.mean((averages[:, :] - np.mean(averages[:, :300])), axis=0)

    # Find peaks in the time series data
    peaks, _ = find_peaks(-time_series[:700], distance=1)  # Adjust 'distance' parameter as needed

    peak_values = time_series[peaks]

    # Get the indices of the first 3 peaks in descending order of amplitude
    first_3_peak_indices = peaks[np.argsort(peak_values)[::-1][:3]]

    # Get the actual values of the first 3 peaks
    first_3_peak_amplitudes = peak_values[np.argsort(peak_values)[::-1][:3]]

    # Plot the time series and the identified peaks
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, label='Time Series')
    plt.plot(first_3_peak_indices, first_3_peak_amplitudes, 'ro', label='First 3 Peaks')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Time Series with First 3 Peaks')
    plt.show()

    # Plot evoked response variation across sessions

    # Assuming you have a 2D array all_gfma with shape (323, 1001)
    # Calculate the mean and standard deviation along the first axis (sessions)
    mean_all_gfma = np.mean(all_gfma, axis=0)
    std_all_gfma = np.std(all_gfma, axis=0)
    # Calculate the margin of error for the confidence interval
    confidence_level = 0.95
    z_score = 1.96  # For a 95% confidence interval
    margin_of_error = z_score * (std_all_gfma / np.sqrt(len(all_gfma)))

    # Calculate the upper and lower bounds of the confidence interval
    upper_bound = mean_all_gfma + margin_of_error
    lower_bound = mean_all_gfma - margin_of_error

    upper_bound = upper_bound - np.mean(upper_bound[:300])
    lower_bound = lower_bound - np.mean(lower_bound[:300])

    if len(epo_eeg.times) == len(time_series):
        # Plot the time series and the identified peaks
        plt.figure(figsize=(20, 6))
        plt.plot(epo_eeg.times, time_series, label='Time Series')
        plt.plot(epo_eeg.times[first_3_peak_indices], first_3_peak_amplitudes, 'yo', markersize=1,
                 label='First 3 Peaks')
        plt.plot(epo_eeg.times, upper_bound, '-r', label='upper')
        plt.plot(epo_eeg.times, lower_bound, '-g', label='lower')
        plt.fill_between(epo_eeg.times, upper_bound, lower_bound, color="k", alpha=0.15)  # Use 'epo_eeg.times'
        plt.legend()
        plt.xlabel('Time (s)')  # Set the x-axis label to 'Time (s)'
        plt.ylabel('Value')
        plt.title('Time Series with First 3 Peaks')
        # plt.savefig('C:/Users/davide_momi/Desktop/peaks.png', dpi=300)
        plt.show()
    else:
        print("The lengths of 'epo_eeg.times' and 'time_series' don't match.")

    # Plot AUC

    windows = 3
    AUC = np.zeros((3, all_gfma.shape[0]))

    first_3_peak_indices_sorted = sorted(first_3_peak_indices)
    first_peak = epo_eeg.times[first_3_peak_indices_sorted[0]]
    second_peak = epo_eeg.times[first_3_peak_indices_sorted[1]]
    third_peak = epo_eeg.times[first_3_peak_indices_sorted[2]]

    for ses in range(all_gfma.shape[0]):
        AUC[0, ses] = np.trapezoid(
            all_gfma[ses, np.where(epo_eeg.times == 0)[0][0]:np.where(epo_eeg.times == first_peak)[0][0]]
            - np.mean(all_gfma[ses, :300]), dx=5
        )
        AUC[1, ses] = np.trapezoid(
            all_gfma[ses, np.where(epo_eeg.times == first_peak)[0][0]:np.where(epo_eeg.times == second_peak)[0][0]]
            - np.mean(all_gfma[ses, :300]), dx=5
        )
        AUC[2, ses] = np.trapezoid(
            all_gfma[ses, np.where(epo_eeg.times == second_peak)[0][0]:np.where(epo_eeg.times == third_peak)[0][0]]
            - np.mean(all_gfma[ses, :300]), dx=5
        )

    AUC[0, :] = AUC[0, :] / (first_3_peak_indices_sorted[0] - 300)
    AUC[1, :] = AUC[1, :] / (first_3_peak_indices_sorted[1] - first_3_peak_indices_sorted[0])
    AUC[2, :] = AUC[2, :] / (first_3_peak_indices_sorted[2] - first_3_peak_indices_sorted[1])

    net_AUC = {}

    # Iterate over each network
    for network in networks:
        # AUC for each network
        net_AUC[network] = AUC[:, stim_network_indices[network]]

    # Obtain the average AUC
    AUC_averages = np.zeros((len(networks), windows))
    for idx, key in enumerate(net_AUC.keys()):
        AUC_averages[idx, :] = np.mean(net_AUC[key], axis=1)

    AUC_averages = AUC_averages * 100000

    # Create the figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(13, 6))  # 2 rows, 1 column

    # Plot in the first subplot
    axs[0].bar(range(AUC_averages[:, 1].shape[0]), AUC_averages[:, 0], color=colour)
    axs[0].set_xticks(range(AUC_averages[:, 0].shape[0]))
    axs[0].set_xticklabels(networks, rotation=45)
    axs[0].set_xlabel('Networks')
    axs[0].set_title('Early response 0-' + str(round(first_peak * 1000)) + 'ms')
    axs[0].set_ylabel('AUC')
    axs[0].set_ylim(0, 2)  # Adjust the y-axis limits as needed

    # Plot in the second subplot (same as the first subplot)
    axs[1].bar(range(AUC_averages[:, 1].shape[0]), AUC_averages[:, 1], color=colour)
    axs[1].set_xticks(range(AUC_averages[:, 0].shape[0]))
    axs[1].set_xticklabels(networks, rotation=45)
    axs[1].set_xlabel('Networks')
    axs[1].set_title('Late response ' + str(round(first_peak * 1000)) + '-' + str(round(second_peak * 1000)) + 'ms')
    axs[1].set_ylabel('AUC')
    axs[1].set_ylim(0, 2)  # Adjust the y-axis limits as needed

    # Plot in the second subplot (same as the first subplot)
    axs[2].bar(range(AUC_averages[:, 2].shape[0]), AUC_averages[:, 2], color=colour)
    axs[2].set_xticks(range(AUC_averages[:, 0].shape[0]))
    axs[2].set_xticklabels(networks, rotation=45)
    axs[2].set_xlabel('Networks')
    axs[2].set_title('Late response ' + str(round(second_peak * 1000)) + '-' + str(round(third_peak * 1000)) + 'ms')
    axs[2].set_ylabel('AUC')
    axs[2].set_ylim(0, 2)  # Adjust the y-axis limits as needed

    plt.tight_layout()  # Adjust the spacing between subplots if needed

    plt.show()

    # Plot sEEG at each network

    # Load sEEG epochs
    with open(data_folder + '/empirical_data/all_epo_seeg.pkl', 'rb') as handle:
        all_epo_seeg = pickle.load(handle)

    all_gfma = np.zeros((len(list(all_epo_seeg.keys())), epo_eeg._data.shape[2]))

    for ses in range(len(list(all_epo_seeg.keys()))):
        epo_seeg = all_epo_seeg[list(all_epo_seeg.keys())[ses]]
        for xx in range(epo_seeg.shape[0]):
            epo_seeg[xx, :] = epo_seeg[xx, :] - np.mean(epo_seeg[xx, :300])

        all_gfma[ses, :] = np.std(epo_seeg, axis=0)

    # Load Schaefer 1000 parcels
    with open(data_folder + '/empirical_data/dist_Schaefer_1000parcels_7net.pkl', 'rb') as handle:
        dist_Schaefer_1000parcels_7net = pickle.load(handle)
    # Extract the stimulation region data from the loaded pickle file
    stim_region = dist_Schaefer_1000parcels_7net['stim_region']

    networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    # Create a dictionary to store the network indices
    stim_network_indices = {network: [] for network in networks}
    for i, label in enumerate(stim_region):
        # if dist_Schaefer_1000parcels_7net['dist'][i] < 7:
        # Iterate over each network
        for network in networks:
            if network in label:
                stim_network_indices[network].append(i)
                break

    net_gfma = {}
    # Iterate over each network
    for network in networks:
        # GFMS for each network
        net_gfma[network] = all_gfma[stim_network_indices[network]]

    # Compute the GFMA averages
    averages = []
    for key, value in net_gfma.items():
        average = sum(value) / len(value)
        averages.append(average)

    averages = np.array(averages)

    # Download the network colour file from the GitHub URL
    colour = pd.read_excel(net_colour_file, header=None)[4]

    # Define the desired figure size
    fig = plt.figure(figsize=(20, 6))
    for net in range(len(networks)):
        plt.plot(epo_eeg.times, averages[net, :] - np.mean(averages[net, :300]), colour[net], linewidth=5)
    plt.show()

    # Plot sEEG AUC

    # Calculate the mean array as you mentioned
    time_series = np.mean((averages[:, :] - np.mean(averages[:, :300])), axis=0)

    # Find peaks in the time series data
    peaks, _ = find_peaks(-time_series, width=15)  # Adjust 'distance' parameter as needed
    peak_values = time_series[peaks]

    # Get the indices of the first 3 peaks in descending order of amplitude
    first_3_peak_indices = peaks[np.argsort(peak_values)[::-1][:3]]
    first_3_peak_indices = np.array([298, 337, 378, 700])
    first_3_peak_amplitudes = time_series[first_3_peak_indices]

    windows = 3
    AUC = np.zeros((3, all_gfma.shape[0]))

    first_peak = epo_eeg.times[first_3_peak_indices[0]]
    second_peak = epo_eeg.times[first_3_peak_indices[1]]
    third_peak = epo_eeg.times[first_3_peak_indices[2]]
    fourth_peak = epo_eeg.times[first_3_peak_indices[3]]

    for ses in range(all_gfma.shape[0]):
        AUC[0, ses] = np.trapezoid(
            all_gfma[ses, np.where(epo_eeg.times == first_peak)[0][0]:np.where(epo_eeg.times == second_peak)[0][0]]
            - np.mean(all_gfma[ses, :300]), dx=5
        )
        AUC[1, ses] = np.trapezoid(
            all_gfma[ses, np.where(epo_eeg.times == second_peak)[0][0]:np.where(epo_eeg.times == third_peak)[0][0]]
            - np.mean(all_gfma[ses, :300]), dx=5)
        AUC[2, ses] = np.trapezoid(
            all_gfma[ses, np.where(epo_eeg.times == third_peak)[0][0]:np.where(epo_eeg.times == fourth_peak)[0][0]]
            - np.mean(all_gfma[ses, :300]), dx=5
        )

    AUC[0, :] = AUC[0, :] / 33
    AUC[1, :] = AUC[1, :] / 45
    AUC[2, :] = AUC[2, :] / 319

    net_AUC = {}
    # Iterate over each network
    for network in networks:
        # AUC for each network
        net_AUC[network] = AUC[:, stim_network_indices[network]]

    # Obtain AUC averages
    AUC_averages = np.zeros((len(networks), windows))
    for idx, key in enumerate(net_AUC.keys()):
        AUC_averages[idx, :] = np.mean(net_AUC[key], axis=1)

    AUC_averages = AUC_averages * 1000
    # AUC_averages = (AUC_averages / np.max(AUC_averages, axis=0)) * 100

    # Create the figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(13, 6))  # 2 rows, 1 column

    # Plot in the first subplot
    axs[0].bar(range(AUC_averages[:, 1].shape[0]), AUC_averages[:, 0], color=colour)
    axs[0].set_xticks(range(AUC_averages[:, 0].shape[0]))
    axs[0].set_xticklabels(networks, rotation=45)
    axs[0].set_xlabel('Networks')
    axs[0].set_title(
        'Early response ' + str(round(first_peak * 1000)) + '-' + str(round(second_peak * 1000)) + 'ms')
    axs[0].set_ylabel('AUC')
    axs[0].set_ylim(0, 3)  # Adjust the y-axis limits as needed

    # Plot in the second subplot (same as the first subplot)
    axs[1].bar(range(AUC_averages[:, 1].shape[0]), AUC_averages[:, 1], color=colour)
    axs[1].set_xticks(range(AUC_averages[:, 0].shape[0]))
    axs[1].set_xticklabels(networks, rotation=45)
    axs[1].set_xlabel('Networks')
    axs[1].set_title('Late response ' + str(round(second_peak * 1000)) + '-' + str(round(third_peak * 1000)) + 'ms')
    axs[1].set_ylabel('AUC')
    axs[1].set_ylim(0, 3)  # Adjust the y-axis limits as needed

    # Plot in the third subplot (same as the first subplot)
    axs[2].bar(range(AUC_averages[:, 2].shape[0]), AUC_averages[:, 2], color=colour)
    axs[2].set_xticks(range(AUC_averages[:, 0].shape[0]))
    axs[2].set_xticklabels(networks, rotation=45)
    axs[2].set_xlabel('Networks')
    axs[2].set_title('Late response ' + str(round(third_peak * 1000)) + '-' + str(round(fourth_peak * 1000)) + 'ms')
    axs[2].set_ylabel('AUC')
    axs[2].set_ylim(0, 3)  # Adjust the y-axis limits as needed

    plt.tight_layout()
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


def model_fitting():
    # Select the session number to use: Please do not change it as we are using subject-specific anatomy
    ses2use = 10

    # Load the precomputed EEG evoked response data from a file
    all_eeg_evoked = np.load(data_folder + '/empirical_data/all_eeg_evoked.npy')

    # Read the epoch data from an MNE-formatted file
    epo_eeg = mne.read_epochs(data_folder + '/empirical_data/example_epoched.fif', verbose=False)

    # Compute the average evoked response from the epochs
    evoked = epo_eeg.average()

    # Replace the data of the averaged evoked response with data from the selected session
    evoked.data = all_eeg_evoked[ses2use]

    # Load additional data from pickle files
    with open(data_folder + '/empirical_data/all_epo_seeg.pkl', 'rb') as handle:
        all_epo_seeg = pickle.load(handle)
    with open(data_folder + '/empirical_data/dist_Schaefer_1000parcels_7net.pkl', 'rb') as handle:
        dist_Schaefer_1000parcels_7net = pickle.load(handle)

    # Extract the stimulation region data from the loaded pickle file
    stim_region = dist_Schaefer_1000parcels_7net['stim_region']

    # Extract coordinates and ROI labels from the atlas data
    coords_200 = np.array([atlas200['R'], atlas200['A'], atlas200['S']]).T
    label = atlas200['ROI Name']

    # Remove network names from the ROI labels for clarity
    label_stripped_200 = []
    for xx in range(len(label)):
        label_stripped_200.append(label[xx].replace('7Networks_', ''))

    # Extract coordinates and ROI labels from the atlas data
    coords_1000 = np.array([atlas1000['R'], atlas1000['A'], atlas1000['S']]).T
    ROI_Name = atlas1000['ROI Name']

    # Remove network names from the ROI labels for clarity
    label_stripped_1000 = []
    for xx in range(len(ROI_Name)):
        label_stripped_1000.append(ROI_Name[xx].replace('7Networks_', ''))

    # Find the index of the stimulation region in the list of stripped ROI labels (1000 parcels)
    stim_idx = label_stripped_1000.index(stim_region[ses2use])

    # Use the index to get the coordinates of the stimulation region from the 1000-parcel atlas
    stim_coords = coords_1000[stim_idx]

    # Extract the network name from the stimulation region label
    # The network name is the part after the underscore in the stimulation region label
    stim_net = stim_region[ses2use].split('_')[1]

    # Define distance function
    def euclidean_distance(coord1, coord2):
        x1, y1, z1 = coord1[0], coord1[1], coord1[2]
        x2, y2, z2 = coord2[0], coord2[1], coord2[2]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    # Initialize an empty list to store distances
    distances = []
    # Iterate over each coordinate in the 200-parcel atlas
    for xx in range(coords_200.shape[0]):
        # Compute the Euclidean distance between the current coordinate and the stimulation coordinates
        # Append the computed distance to the distances list
        distances.append(euclidean_distance(coords_200[xx], stim_coords))
    # Convert the list of distances to a NumPy array for easier manipulation
    distances = np.array(distances)

    # Iterate over the indices of the distances array, sorted in ascending order
    for idx, item in enumerate(np.argsort(distances)):
        # Check if the network name of the stimulation region is present in the label of the current parcel
        if stim_net in label_stripped_200[item]:
            # If the condition is met, assign the index of the current parcel to `parcel2inject`
            parcel2inject = item
            # Exit the loop since the desired parcel has been found
            break

    # Extract the absolute values of the EEG data for the specified session
    abs_value = np.abs(all_epo_seeg[list(all_epo_seeg.keys())[ses2use]])
    # Normalize each time series by subtracting its mean
    for xx in range(abs_value.shape[0]):
        abs_value[xx, :] = abs_value[xx, :] - np.mean(abs_value[xx, :])
    # Take the absolute value of the normalized data
    abs_value = np.abs(abs_value)

    # Find the starting and ending points around the maximum value in the data
    # Get the index of the maximum value along the time axis
    starting_point = np.where(abs_value == abs_value.max())[1][0] - 10
    ending_point = np.where(abs_value == abs_value.max())[1][0] + 10

    # Compute the maximum, mean, and standard deviation of the data within the range around the maximum
    mean = np.mean(abs_value[:, starting_point:ending_point])
    std = np.std(abs_value[:, starting_point:ending_point])

    # Define a threshold as mean + 4 times the standard deviation
    thr = mean + (4 * std)

    # Count the number of unique regions affected by the threshold
    number_of_region_affected = np.unique(np.where(abs_value > thr)[0]).shape[0]

    # Load the rewritten Schaeffer 200 parcels
    img = nibabel.load(data_folder + '/calculate_distance/example_Schaefer2018_200Parcels_7Networks_rewritten.nii')

    # Get the shape and affine matrix of the image
    shape, affine = img.shape[:3], img.affine

    # Create a meshgrid of voxel coordinates
    coords = np.array(np.meshgrid(*(range(i) for i in shape), indexing='ij'))

    # Rearrange the coordinates array to have the correct shape
    coords = np.rollaxis(coords, 0, len(shape) + 1)

    # Apply the affine transformation to get the coordinates in millimeters
    mm_coords = nibabel.affines.apply_affine(affine, coords)

    # Initialize an array to store the coordinates of the 200 parcels
    sub_coords = np.zeros((3, 200))

    # Loop over each parcel (1 to 200)
    for xx in range(1, 201):
        # Find the voxel coordinates where the parcel value equals the current parcel number
        vox_x, vox_y, vox_z = np.where(img.get_fdata() == xx)
        # Calculate the mean coordinates in millimeters for the current parcel
        sub_coords[:, xx - 1] = np.mean(mm_coords[vox_x, vox_y, vox_z], axis=0)

    # Initialize an empty list to store distances
    distances = []
    # Compute the Euclidean distance between each coordinate in the
    # 200-parcel atlas and the coordinate of the parcel to inject
    for xx in range(coords_200.shape[0]):
        distances.append(euclidean_distance(sub_coords[:, xx], sub_coords[:, parcel2inject]))
    # Convert the list of distances to a NumPy array for further processing
    distances = np.array(distances)

    # Find the indices of the closest parcels to inject, based on the number of affected regions
    inject_stimulus = np.argsort(distances)[:number_of_region_affected]

    # Compute stimulus weights based on the distances
    # Adjust distances to a scale of 0 to 1 and calculate the values for the stimulus weights
    values = (np.max(distances[inject_stimulus] / 10) + 0.5) - (distances[inject_stimulus] / 10)

    # Initialize an array for stimulus weights with zeros
    stim_weights_thr = np.zeros((len(label)))

    # Assign the computed values to the stimulus weights for the selected parcels
    stim_weights_thr[inject_stimulus] = values

    old_path = data_folder + "/anatomical/example-bem"
    new_path = data_folder + "/anatomical/example-bem.fif"  # CS
    if not os.path.exists(new_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_path} to {new_path}")

    # File paths for transformation, source space, and BEM files
    trans = data_folder + '/anatomical/example-trans.fif'
    src = data_folder + '/anatomical/example-src.fif'
    bem = data_folder + '/anatomical/example-bem.fif'

    # Create a forward solution using the provided transformation, source space, and BEM files
    # Only EEG is used here; MEG is disabled
    fwd = mne.make_forward_solution(
        epo_eeg.info,
        trans=trans,
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=2,
        verbose=False,
    )

    # Convert the forward solution to a fixed orientation with surface orientation
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)
    # Update the leadfield matrix to use the fixed orientation
    leadfield = fwd_fixed['sol']['data']

    # Extract vertex indices for each hemisphere from the forward solution
    vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]

    # Read annotation files for left and right hemispheres
    lh_vertices = nibabel.freesurfer.io.read_annot(
        data_folder + '/anatomical/lh.Schaefer2018_200Parcels_7Networks_order.annot')[0]
    rh_vertices = nibabel.freesurfer.io.read_annot(
        data_folder + '/anatomical/rh.Schaefer2018_200Parcels_7Networks_order.annot')[0]

    # Extract vertices corresponding to the parcels from the annotation files
    # Add 100 to right hemisphere vertices to adjust for parcel numbering
    lh_vertices_thr = lh_vertices[vertices[0]]
    rh_vertices_thr = rh_vertices[vertices[1]] + 100
    # Combine left and right hemisphere vertices into a single array
    vertices_thr = np.concatenate([lh_vertices_thr, rh_vertices_thr])

    # Initialize a new leadfield matrix with dimensions adjusted for the number of parcels
    new_leadfield = np.zeros((leadfield.shape[0], np.unique(vertices_thr).shape[0] - 1))

    # Compute the average leadfield for parcels in the range 1-100
    for parcel in range(1, 101):
        new_leadfield[:, parcel - 1] = np.mean(leadfield[:, np.where(vertices_thr == parcel)[0]], axis=1)

    # Compute the average leadfield for parcels in the range 101-200
    for parcel in range(101, 201):
        new_leadfield[:, parcel - 1] = np.mean(leadfield[:, np.where(vertices_thr == parcel)[0]], axis=1)

    # Load structural connectivity data from a CSV file
    sc_df = pd.read_csv(sc_file, header=None, sep=' ')
    sc = sc_df.values
    # Apply log transformation and normalization to the structural connectivity matrix
    sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))

    # Save the downloaded distance data to a CSV file
    if not os.path.exists(dist_file):
        # Download distance data from Google Drive
        response = requests.get("https://drive.google.com/uc?export=download&id=1EzJNFckal6n4uXMY3h31Wtd9aqsCmgGc")
        with open(dist_file, 'wb') as f:
            f.write(response.content)

    # Load the distance data from the saved CSV file
    dist_df = pd.read_csv(dist_file, header=None, sep=' ')
    dist = dist_df.values

    # Initialize the stimulus weights for further processing
    ki0 = stim_weights_thr

    # Extract and normalize EEG data from the evoked response
    eeg_data = evoked.data
    eeg_data = eeg_data[:, 200:600].T / (np.abs(eeg_data)).max() * 2

    # Define model parameters
    node_size = sc.shape[0]
    output_size = eeg_data.shape[1]
    batch_size = 20
    step_size = 0.0001
    num_epochs = 40
    tr = 0.001
    time_dim = 400
    hidden_size = int(tr / step_size)
    TPperWindow = batch_size

    # Prepare model data structure
    # eeg_data: (time_points, output_size) - EEG data
    # dataloader output: (window_size, TR_per_window, output_size)
    data_mean = dataloader(eeg_data - eeg_data.mean(1)[:, np.newaxis], batch_size)

    # Initialize the leadfield matrix for the model
    lm = new_leadfield.copy() / 10

    # Initialize random values for the leadfield matrix
    lm_v = 0.01 * np.random.randn(output_size, 200) + 0.1

    print('lm: ', np.mean(lm), np.std(lm))

    # Create model instance with Param objects
    model = JansenRitWindow(
        node_size=node_size,
        output_size=output_size,
        tr=tr,
        step_size=step_size,
        sc=sc,
        dist=dist,
        mask=np.ones((node_size, node_size)),
        # Fixed parameters
        A=Const(3.25),
        B=Const(22),
        g=Const(200),
        g_f=Const(10),
        g_b=Const(10),
        vmax=Const(5),
        v0=Const(6),
        r=Const(0.56),
        ki=Const(ki0),
        kE=Const(0.),
        kI=Const(0.),
        # Trainable parameters with GaussianReg (fit_hyper=True)
        a=Param(100, t=ReluT(), reg=GaussianReg(100, 2, True)),
        b=Param(50, t=ReluT(), reg=GaussianReg(50, 1, True)),
        c1=Param(135, t=ReluT(), reg=GaussianReg(135, 1, True)),
        c2=Param(135 * 0.8, t=ReluT(), reg=GaussianReg(135 * 0.8, 1, True)),
        c3=Param(135 * 0.25, t=ReluT(), reg=GaussianReg(135 * 0.25, 1, True)),
        c4=Param(135 * 0.25, t=ReluT(), reg=GaussianReg(135 * 0.25, 1, True)),
        # std_in: as_log=True -> ExpT
        std_in=Param(1.1, t=ExpT(0.1), reg=GaussianReg(1.1, 0.1, True)),
        y0=Param(-2, reg=GaussianReg(-2, 0.3, True)),
        mu=1.1,
        k=Param(15, t=ReluT(5.0), reg=GaussianReg(15, 0.2, True)),
        cy0=Param(1., reg=GaussianReg(1, 0.1, True)),
        # lm with array-based regularization
        lm=Param(lm + lm_v * np.random.randn(*lm.shape)),
        w_bb=Param(np.full((node_size, node_size), 0.05)),
        w_ff=Param(np.full((node_size, node_size), 0.05)),
        w_ll=Param(np.full((node_size, node_size), 0.05)),
        state_init=lambda s, **kwargs: np.random.uniform(-0.1, 0.1, s),
        delay_init=lambda s, **kwargs: np.random.uniform(0.0, 0.1, s),
    )

    # Create objective function
    ObjFun = CostsJR(model)

    # Call model fit
    F = ModelFitting(model, ObjFun)

    # Model Training
    # u: (time_dim, hidden_size, node_size) - time dimension first
    u = np.zeros((time_dim, hidden_size, node_size))
    u[65:75] = 2000  # Apply stimulus at time steps 65-75
    F.train(inputs=u, target_eeg=data_mean, num_epochs=num_epochs, TP_per_window=TPperWindow, warmup_window=20)

    # Model Evaluation (with 20 window for warmup)
    pred = F.evaluate(u=u, target_eeg=data_mean, TR_per_window=TPperWindow, base_window_num=100)

    time_start = np.where(evoked.times == -0.1)[0][0]
    time_end = np.where(evoked.times == 0.3)[0][0]
    ts_args = dict(xlim=[-0.1, 0.3])  # Time to plot

    ch, peak_locs1 = evoked.get_peak(ch_type='eeg', tmin=-0.05, tmax=0.015)
    ch, peak_locs2 = evoked.get_peak(ch_type='eeg', tmin=0.015, tmax=0.03)
    ch, peak_locs3 = evoked.get_peak(ch_type='eeg', tmin=0.03, tmax=0.04)
    ch, peak_locs4 = evoked.get_peak(ch_type='eeg', tmin=0.04, tmax=0.06)
    ch, peak_locs5 = evoked.get_peak(ch_type='eeg', tmin=0.08, tmax=0.12)
    ch, peak_locs6 = evoked.get_peak(ch_type='eeg', tmin=0.12, tmax=0.2)
    times = [peak_locs1, peak_locs2, peak_locs3, peak_locs4, peak_locs5, peak_locs6]
    evoked_joint_st = evoked.plot_joint(ts_args=ts_args, times=times, show=False)

    # eeg_testing: (time, output_size)
    # Need to transpose to (output_size, time) to match data format
    simulated_EEG_st = evoked.copy()
    simulated_EEG_st.data[:, time_start:time_end] = pred['eeg'].T
    simulated_joint_st = simulated_EEG_st.plot_joint(ts_args=ts_args, times=times, show=True)

    return F


def apply_virtual_dissection(data):
    # Read the CSV file into a DataFrame
    atlas = pd.read_csv(atlas200_file)

    # Extract the 'ROI Name' column from the DataFrame
    label = atlas['ROI Name']

    # Create a list to store stripped labels
    label_stripped = []

    # Strip '7Networks_' from each label and append to the list
    for xx in range(len(label)):
        label_stripped.append(label[xx].replace('7Networks_', ''))

    # Define the list of network names
    networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

    # Create a dictionary to store the network indices
    network_indices = {network: [] for network in networks}

    # Iterate over each stripped label
    for i, label in enumerate(label_stripped):
        # Iterate over each network
        for network in networks:
            if network in label:
                # Append the index to the corresponding network's list in the dictionary
                network_indices[network].append(i)
                break

    # Define the stimulated network
    sti_net = 'Default'

    # Convert the list of indices for the stimulated network to a numpy array
    network_indices_arr = np.array(network_indices[sti_net])

    # Get the indices that do not belong to the stimulated network
    diff = np.array(list(set(np.arange(200)) - set(network_indices_arr)))

    # Define model parameters
    state_lb = -0.2
    state_ub = 0.2
    delays_max = 500
    when_damage = 80
    node_size = 200
    batch_size = 20
    step_size = 0.0001
    num_epochs = 150
    tr = 0.001
    state_size = 6  # 6 state variables: P, E, I, Pv, Ev, Iv
    base_batch_num = 20
    time_dim = 400
    base_batch_num = 100
    hidden_size = int(tr / step_size)
    TPperWindow = batch_size
    transient_num = 10

    final_ouput_P = []
    final_ouput_E = []
    final_ouput_I = []
    final_ouput_eeg = []

    # Initialize state using nmm API
    state = data.model.create_initial_state(init_range=(state_lb, state_ub))
    x0 = state.to_tensor()

    # Initialize delay buffer using nmm API
    delay_buffer = data.model.create_delay_buffer(init_range=(state_lb, state_ub))
    he0 = delay_buffer

    # Create input tensor u: (time_dim, hidden_size, node_size) - time dimension first
    u = np.zeros((when_damage, hidden_size, node_size))

    # Apply stimulus at time steps 65-75
    u[65:75, :, :] = 2000

    # Create a mask with ones of shape 200x200
    mask = np.ones((200, 200))

    # data_mean: (window_size, TR_per_window, output_size)
    num_windows = int(when_damage / TPperWindow)
    data_mean = np.ones((num_windows, TPperWindow, data.model.output_size))

    # Evaluate the model with the given input tensor u, empirical data data_mean, and initial states x0 and he0
    pred = data.evaluate(u=u,
                         target_eeg=data_mean,
                         TR_per_window=TPperWindow,
                         X=x0.numpy(),
                         hE=he0.numpy(),
                         base_window_num=100)

    # Append testing states - states: (total_time, node_size, 6)
    # P: index 0, E: index 1, I: index 2
    final_ouput_P.append(pred['P'])  # (time, node_size)
    final_ouput_E.append(pred['E'])
    final_ouput_I.append(pred['I'])
    final_ouput_eeg.append(pred['eeg'])

    # Update x0: take the last time step state (node_size, 6)
    x0 = data.trainingStats.states['testing'][-1]  # (node_size, 6)

    # Update he0: (delays_max, node_size)
    # Take P state history (time, node_size) last delays_max rows
    P_history = pred['P']  # (time, node_size)
    # Reverse time axis and truncate
    P_reversed = P_history[::-1]  # Time reversed
    available_len = min(delays_max, P_reversed.shape[0])
    he0[:available_len] = P_reversed[:available_len]

    # Create a mask with ones of shape 200x200
    mask = np.ones((200, 200))

    # Set the mask elements corresponding to network_indices_arr and diff to 0
    mask[np.ix_(network_indices_arr, diff)] = 0

    # Create input tensor u: (time_dim, hidden_size, node_size)
    remaining_time = int(400 - when_damage)
    u = np.zeros((remaining_time, hidden_size, node_size))

    # data_mean: (window_size, TR_per_window, output_size)
    num_windows = int(remaining_time / TPperWindow)
    data_mean = np.ones((num_windows, TPperWindow, data.model.output_size))

    # Evaluate the model with the given input tensor u, empirical data data_mean, initial states x0, he0, and mask
    pred = data.evaluate(
        u=u, target_eeg=data_mean, X=x0.numpy(), hE=he0.numpy(),
        TR_per_window=TPperWindow,
        base_window_num=0, mask=mask,
    )

    # Append testing states
    final_ouput_P.append(pred['P'])
    final_ouput_E.append(pred['E'])
    final_ouput_I.append(pred['I'])
    final_ouput_eeg.append(pred['eeg'])

    # Concatenate results along time axis (axis=0)
    # Shape: (time, node_size) or (time, output_size)
    new_P = np.concatenate((final_ouput_P[0], final_ouput_P[1]), axis=0)
    new_E = np.concatenate((final_ouput_E[0], final_ouput_E[1]), axis=0)
    new_I = np.concatenate((final_ouput_I[0], final_ouput_I[1]), axis=0)
    new_eeg = np.concatenate((final_ouput_eeg[0], final_ouput_eeg[1]), axis=0)

    # Read the epoched data from a .fif file
    epoched = mne.read_epochs(data_folder + '/empirical_data/example_epoched.fif', verbose=False)

    # Compute the average evoked response from the epoched data
    evoked = epoched.average()

    # Find the index corresponding to the time -0.1 seconds
    time_start = np.where(evoked.times == -0.1)[0][0]

    # Find the index corresponding to the time 0.3 seconds
    time_end = np.where(evoked.times == 0.3)[0][0]

    # Create a copy of the evoked data for simulation
    simulation = evoked.copy()

    # Replace the simulation data in the time range from time_start to time_end with the EEG testing data
    # eeg_testing: (time, output_size) - need to transpose to (output_size, time) to match simulation.data
    time_end = time_start + pred['eeg'].shape[0]
    simulation.data[:, time_start:time_end] = pred['eeg'].T

    # Find peak locations in specified time windows and store them
    ch, peak_locs1 = evoked.get_peak(ch_type='eeg', tmin=-0.05, tmax=0.015)
    ch, peak_locs2 = evoked.get_peak(ch_type='eeg', tmin=0.015, tmax=0.03)
    ch, peak_locs3 = evoked.get_peak(ch_type='eeg', tmin=0.03, tmax=0.04)
    ch, peak_locs4 = evoked.get_peak(ch_type='eeg', tmin=0.04, tmax=0.06)
    ch, peak_locs5 = evoked.get_peak(ch_type='eeg', tmin=0.08, tmax=0.12)
    ch, peak_locs6 = evoked.get_peak(ch_type='eeg', tmin=0.12, tmax=0.2)

    # Collect peak locations into a list
    times = [peak_locs1, peak_locs2, peak_locs3, peak_locs4, peak_locs5, peak_locs6]

    # Define plotting arguments for the time series
    ts_args = dict(xlim=[-0.1, 0.3])

    # Plot the simulated data joint view
    simulated_joint_st = simulation.plot_joint(ts_args=ts_args, times=times, show=True)

    return new_P, new_E, new_I, new_eeg


if __name__ == '__main__':
    # empirical_data_analysis()
    F = model_fitting()
    # apply_virtual_dissection(F)
