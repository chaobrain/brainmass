import os
import pickle
from collections import defaultdict

import brainstate
import brainunit as u
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.signal
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity

import braintools
from brainmass.jansen_rit_v2 import JansenRitWindow
from braintools.param import Param, Const, ReluT, ExpT, GaussianReg


def dataloader(emp, TR_per_window):
    length_ts = emp.shape[0]
    node_size = emp.shape[1]
    window_size = int(length_ts / TR_per_window)
    data_out = np.zeros((window_size, TR_per_window, node_size))
    for i_win in range(window_size):
        data_out[i_win] = emp[i_win * TR_per_window:(i_win + 1) * TR_per_window]
    return data_out


class Costs:
    def __init__(self):
        self.w_cost = 10

    def cost_eff(self, sim, emp, model):
        loss_main = u.math.sqrt(u.math.mean((sim - emp) ** 2))
        loss_prior = []
        for module in model.nodes(Param).values():
            loss_prior.append(module.reg_loss())
        loss = self.w_cost * loss_main + sum(loss_prior)
        return loss


class ModelFitting:
    def __init__(self, model: JansenRitWindow, ts, n_epoches: int):
        self.model = model
        self.n_epoches = n_epoches
        self.ts = ts
        self.cost = Costs()
        self.optimizer = braintools.optim.Adam(lr=5e-2)
        # self.optimizer = braintools.optim.Adam(lr=5e-3)
        self.weights = model.states(brainstate.ParamState)
        self.optimizer.register_trainable_weights(self.weights)

    def f_loss(self, model_state, inputs, targets):
        # Use the model.forward() function to update next state and get simulated EEG
        param = self.model.retrieve_params()
        model_state, (eeg_output, _) = self.model.update(model_state, param, inputs)
        loss = self.cost.cost_eff(eeg_output, targets, self.model)
        return loss, (model_state, eeg_output)

    @brainstate.transform.jit(static_argnums=0)
    def f_train(self, model_state, inputs, targets):
        f_grad = brainstate.transform.fwd_grad(self.f_loss, self.weights, tangent_size=128, has_aux=True, return_value=True)
        f_grad = brainstate.transform.grad(self.f_loss, self.weights, has_aux=True, return_value=True)
        grads, loss, (model_state, eeg_output) = f_grad(model_state, inputs, targets)
        self.optimizer.step(grads)
        return loss, eeg_output, model_state

    @brainstate.transform.jit(static_argnums=0)
    def f_predict(self, model_state, inputs):
        # Use the model.forward() function
        model_param = self.model.retrieve_params()
        model_state, (eeg_output, state_output) = self.model(model_state, model_param, inputs)
        return model_state, eeg_output, state_output

    def train(self, inputs):
        # initial state using nmm API - ModelData contains dynamics_state and delay_state
        model_state = self.model.create_initial_state()

        # define masks for getting lower triangle matrix indices
        mask_e = np.tril_indices(self.model.output_size, -1)

        # loss placeholder
        loss_his = []

        # define num_windows
        num_windows = self.ts.shape[0]
        TRs_per_window = int(self.ts.shape[1])  # Use actual window size from data

        for i_epoch in range(self.n_epoches):
            output_sim = defaultdict(list)
            for i_win in range(num_windows):
                # external input: u should be (time_dim, steps_per_TR, node_size) format
                # slice to get (TRs_per_window, steps_per_TR, node_size)
                external = inputs[i_win * TRs_per_window:(i_win + 1) * TRs_per_window]

                # Get the batch of empirical EEG signal.
                ts_window = self.ts[i_win]

                loss, eeg_output, model_state = self.f_train(model_state, external, ts_window)

                # Put the batch of the simulated EEG, E I M Ev Iv Mv in to placeholders
                loss_np = np.asarray(loss)
                loss_his.append(loss_np)
                output_sim['loss'].append(loss_np)
                output_sim['eeg_train'].append(np.asarray(eeg_output))

            # ts_emp: from (num_windows, TRs_per_window, node_size) to (total_TRs, node_size)
            ts_emp = self.ts.reshape(-1, self.ts.shape[-1])
            fc = np.corrcoef(ts_emp, rowvar=False)
            ts_sim = np.concatenate(output_sim['eeg_train'], axis=0)
            fc_sim = np.corrcoef(ts_sim[10:, :], rowvar=False)

            print(
                f'epoch: {i_epoch}, '
                f'loss: {np.asarray(output_sim["loss"]).mean()}, '
                f'FC cor: {np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1]}, '
                f'cos_sim: {np.diag(cosine_similarity(ts_sim.T, ts_emp.T)).mean()}'
            )

        return np.array(loss_his)

    def test(self, base_window_num, uu):
        # define some constants
        transient_num = 10

        # initial state using nmm API - ModelData contains dynamics_state and delay_state
        model_state = self.model.create_initial_state()

        # define mask for getting lower triangle matrix
        mask_e = np.tril_indices(self.model.output_size, -1)

        # define num_windows
        num_windows = self.ts.shape[0]
        TRs_per_window = int(self.ts.shape[1])

        # Create placeholders for the simulated outputs
        output_sim = defaultdict(list)

        # u_hat: (total_TRs, steps_per_TR, node_size) - external input
        total_TRs = base_window_num * TRs_per_window + self.ts.shape[0] * self.ts.shape[1]
        u_hat = np.zeros((total_TRs, hidden_size, self.model.node_size))
        # u: (TRs, steps_per_TR, node_size) - from caller
        u_hat[base_window_num * TRs_per_window:] = uu

        # Perform the testing in batches
        for TR_i in range(num_windows + base_window_num):
            external = u_hat[TR_i * TRs_per_window:(TR_i + 1) * TRs_per_window]
            model_state, eeg_output, state_output = self.f_predict(model_state, external)

            if TR_i > base_window_num - 1:
                # Map nmm output keys to original naming convention
                output_sim['eeg_test'].append(eeg_output)
                output_sim['P_test'].append(state_output['P'])
                output_sim['E_test'].append(state_output['E'])
                output_sim['I_test'].append(state_output['I'])
                output_sim['Pv_test'].append(state_output['Pv'])
                output_sim['Ev_test'].append(state_output['Ev'])
                output_sim['Iv_test'].append(state_output['Iv'])

        # ts_emp: (total_TRs, output_size)
        ts_emp = self.ts.reshape(-1, self.ts.shape[-1])
        fc = np.corrcoef(ts_emp, rowvar=False)

        # ts_sim: (total_TRs, output_size)
        ts_sim = np.concatenate(output_sim['eeg_test'], axis=0)
        fc_sim = np.corrcoef(ts_sim[transient_num:, :], rowvar=False)

        print(
            f'FC: {np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1]}, '
            f'cos_sim: {np.diag(cosine_similarity(ts_sim.T, ts_emp.T)).mean()}',
        )
        # Return all outputs, concatenated along time axis
        return {name: np.concatenate(val, axis=0) for name, val in output_sim.items()}


# We use an example dataset for one subject on a public Google Drive folder
output_dir = 'D:/codes/projects/whole-brain-nmm-pytorch/data_ismail2025/'

# We will use MEG data recorded during a covert verb generation task in verb generation trials and noise trials
# Evoked MEG data averaged across trials (-100 to 400 ms)
verb_meg_raw = np.load(os.path.join(output_dir, 'verb_evoked.npy'))  # (time, channels)
noise_meg_raw = np.load(os.path.join(output_dir, 'noise_evoked.npy'))  # (time, channels)
# Normalize both signals
verb_meg = verb_meg_raw / np.abs(verb_meg_raw).max() * 1
noise_meg = noise_meg_raw / np.abs(noise_meg_raw).max() * 1

# We will use the leadfield to simulate MEG activty from sources derived from the individual's head model
leadfield = loadmat(os.path.join(output_dir, 'leadfield_3d.mat'))  # shape (sources, sensors, 3)
lm_3d = leadfield['M']  # 3D leadfield matrix
# Convert 3D to 2D using SVD-based projection
lm = np.zeros_like(lm_3d[:, :, 0])
for sources in range(lm_3d.shape[0]):
    u_, d, v = np.linalg.svd(lm_3d[sources])
    lm[sources] = u_[:, :3].dot(np.diag(d)).dot(v[0])
# Scale the leadfield matrix
lm = lm.T / 1e-11 * 5  # Shape: (channels, sources)

# We will use the individual's weights and distance matrices
sc_df = pd.read_csv(os.path.join(output_dir, 'weights.csv'), header=None).values
sc = np.log1p(sc_df)
sc = sc / np.linalg.norm(sc)
dist = np.loadtxt(os.path.join(output_dir, 'distance.txt'))

node_size = sc.shape[0]
output_size = verb_meg.shape[0]
batch_size = 250
model_dt = 0.0001
num_epoches = 40  # used 250 in paper using 2 for example
data_dt = 0.001
state_size = 6
base_batch_num = 20
time_dim = verb_meg.shape[1]
hidden_size = int(data_dt / model_dt)

# Format input data
data_verb = dataloader(verb_meg.T, batch_size)
data_noise = dataloader(noise_meg.T, batch_size)

# To simulate the auditory inputs in this task we will stimulate the auditory cortices
# These nodes were identified using an ROI mask of left and right Heschl's gyri based on the Talairach Daemon database
ki0 = np.zeros(node_size)
ki0[[2, 183, 5]] = 1


def create_model(fit_hyper=True) -> JansenRitWindow:
    # initiate leadfield matrices
    lm_base = 0.01 * np.random.randn(output_size, node_size)
    lm_noise = 0.1 * np.random.randn(output_size, node_size)

    def create_gaussian(m_, v_):
        return GaussianReg(m_, v_, True) if fit_hyper else None

    params = dict(
        # Trainable parameters with ReLU transform and Gaussian reg
        A=Param(3.25, t=ReluT(), reg=create_gaussian(3.25, 0.1)),
        a=Param(101, t=ReluT(1.), reg=create_gaussian(101, 1.0)),
        B=Param(22, t=ReluT(), reg=create_gaussian(22, 0.5)),
        b=Param(51, t=ReluT(1.), reg=create_gaussian(51, 1.0)),
        g=Param(400, t=ReluT(0.01), reg=create_gaussian(400, 1.0)),
        g_f=Param(10, t=ReluT(0.01), reg=create_gaussian(10, 1.0)),
        g_b=Param(10, t=ReluT(0.01), reg=create_gaussian(10, 1.0)),
        c1=Param(135, t=ReluT(0.01), reg=create_gaussian(135, 1.0)),
        c2=Param(135 * 0.8, t=ReluT(0.01), reg=create_gaussian(135 * 0.8, 1.0)),
        c3=Param(135 * 0.25, t=ReluT(0.01), reg=create_gaussian(135 * 0.25, 1.0)),
        c4=Param(135 * 0.25, t=ReluT(0.01), reg=create_gaussian(135 * 0.25, 1.0)),
        # std_in uses ExpT (no reg in original code)
        std_in=Param(6.0, t=ExpT(5.0)),
        # Fixed parameters
        vmax=Const(5),
        v0=Const(6),
        r=Const(0.56),
        # Trainable with IdentityTransform
        y0=Param(-0.5, reg=create_gaussian(-0.5, 0.05)),
        mu=2.5,
        k=Param(5.5, t=ReluT(0.5), reg=create_gaussian(5.5, 0.2)),
        # Fixed parameters
        kE=Const(0),
        kI=Const(0),
        cy0=Const(5),
        # Array parameters
        ki=Const(ki0),
        lm=Param(lm + lm_base + lm_noise),
        w_bb=Param(np.full((node_size, node_size), 0.05, dtype=brainstate.environ.dftype())),
        w_ff=Param(np.full((node_size, node_size), 0.05, dtype=brainstate.environ.dftype())),
        w_ll=Param(np.full((node_size, node_size), 0.05, dtype=brainstate.environ.dftype())),
        state_init=lambda s, **kwargs: np.asarray(np.random.uniform(-0.01, 0.01, s), dtype=brainstate.environ.dftype()),
        delay_init=lambda s, **kwargs: np.asarray(np.random.uniform(-0.01, 0.01, s), dtype=brainstate.environ.dftype()),
    )

    return JansenRitWindow(
        node_size=node_size,
        output_size=output_size,
        tr=data_dt,
        step_size=model_dt,
        sc=sc,
        dist=dist,
        **params
    )


# Fit two models:
# 1) verb generation trials and noise trials
verb_model = create_model(fit_hyper=True)

# Stimulate the auditory cortices defined by roi in ki0
# stim_input: (time_dim, hidden_size, node_size) - external stimulus input
stim_input = np.zeros((time_dim, hidden_size, node_size))
# Apply stimulus at time steps 100-140
stim_input[100:140] = 5000

# Fit models
verb_F = ModelFitting(verb_model, data_verb, num_epoches)
verb_F.train(inputs=stim_input)
verb_outs = verb_F.test(base_batch_num, stim_input)
print("Finished fitting model to verb trials")

# repeat for noise
noise_model = create_model(fit_hyper=True)
noise_F = ModelFitting(noise_model, data_noise, num_epoches)
noise_F.train(inputs=stim_input)
noise_outs = noise_F.test(base_batch_num, stim_input)
print("Finished fitting model to noise trials")

# 7. Let's Compare Simulated & Empirical MEG Activity
# we will use the simulations from the fully trained model in the downloaded directory
# verb_meg_sim = np.load(os.path.join(output_dir, 'sim_verb_sensor.npy'))
# noise_meg_sim = np.load(os.path.join(output_dir, 'sim_noise_sensor.npy'))
verb_meg_sim = verb_outs['eeg_test']
noise_meg_sim = noise_outs['eeg_test']
# Use existing MEG channel structure to use MNE format
with open(os.path.join(output_dir, 'info.pkl'), 'rb') as f:
    info = pickle.load(f)

# Convert empirical data to MNE format
emp_verb_evoked = mne.EvokedArray(verb_meg[:, 0:], info, tmin=-0.1)
emp_noise_evoked = mne.EvokedArray(noise_meg[:, 0:], info, tmin=-0.1)

# Convert simulated data to MNE format
# Simulated data format is (TRs, output_size), need to transpose to MNE format (channels, times)
sim_verb_evoked = mne.EvokedArray(verb_meg_sim[0:500].T, info, tmin=-0.1)
sim_noise_evoked = mne.EvokedArray(noise_meg_sim[0:500].T, info, tmin=-0.1)
# Plot empirical verb trial
emp_verb_evoked.plot_joint(title=f"Empirical Verb", show=False, times=[0.07, 0.1, 0.1585])
# Plot simulated verb trial
sim_verb_evoked.plot_joint(title=f"Simulated Verb", show=False, times=[0.07, 0.1, 0.1585])
plt.show()
# Plot empirical noise trial
emp_noise_evoked.plot_joint(title=f"Empirical Noise", show=False, times=[0.07, 0.1, 0.1585])
# Plot simulated noise trial
sim_noise_evoked.plot_joint(title=f"Simulated Noise", show=False, times=[0.07, 0.1, 0.1585])
plt.show()

# 8. Simulate models for longer (model was fitted with 500 ms of data, we will simulate 1500 ms!)
# -------------------------------------------------------------------------
# We are interested in capturing changes in beta power between verb and noise trials observed from 700-1200 ms
# Create longer empty array with same shape and fill with the first 500 ms
sim_1500_verb = np.zeros((verb_meg.shape[0], 1500))
sim_1500_verb[:, :verb_meg.shape[1]] = verb_meg * 1.0e13
node_size = sc.shape[0]
output_size = sim_1500_verb.shape[0]
batch_size = 250
model_dt = 0.0001
data_dt = 0.001
state_size = 6
base_batch_num = 20
time_dim = sim_1500_verb.shape[1]
hidden_size = int(data_dt / model_dt)
data_mean = dataloader((sim_1500_verb - sim_1500_verb.mean(0)).T, batch_size)
verb_F.ts = data_mean
# u: (time_dim, hidden_size, node_size) - external stimulus input
uu = np.zeros((time_dim, hidden_size, node_size))
# Apply stimulus at time steps 100-140
uu[100:140] = 5000
output_test = verb_F.test(base_batch_num, uu)
# extract simulated sensor and source data for noise trials
sim_source_verb = output_test['P_test']
sim_sensor_verb = output_test['eeg_test']

# repeat for noise trials
sim_1500_noise = np.zeros((noise_meg.shape[0], 1500))
sim_1500_noise[:, :noise_meg.shape[1]] = noise_meg * 1.0e13
node_size = sc.shape[0]
output_size = sim_1500_noise.shape[0]
batch_size = 250
model_dt = 0.0001
data_dt = 0.001
state_size = 6
base_batch_num = 20
time_dim = sim_1500_noise.shape[1]
hidden_size = int(data_dt / model_dt)
data_mean = dataloader((sim_1500_noise - sim_1500_noise.mean(0)).T, batch_size)
noise_F.ts = data_mean

# u: (time_dim, hidden_size, node_size) - external stimulus input
uu = np.zeros((time_dim, hidden_size, node_size))
# Apply stimulus at time steps 100-140
uu[100:140] = 5000
output_test = noise_F.test(base_batch_num, uu)

# extract simulated sensor and source data for noise trials
sim_source_noise = output_test['P_test']
sim_sensor_noise = output_test['eeg_test']

# %%  [markdown]
# 9. Compare empirical and simulated change in beta power between verb and noise trials for one subject
# -------------------------------------------------------------------------
# We are replicating figure 1D (Adolescents) for one subject
# We will load the empirical source data (model was fitted with sensor MEG data) and simulated source from pretrained model
emp_source_noise = np.load(os.path.join(output_dir, 'emp_noise_source.npy'))
emp_source_verb = np.load(os.path.join(output_dir, 'emp_verb_source.npy'))
# sim_source_noise = np.load(os.path.join(output_dir, 'sim_noise_source.npy'))
# sim_source_verb = np.load(os.path.join(output_dir, 'sim_verb_source.npy'))

# Compute beta power
# Sampling parameters
fs = 1000  # Sampling frequency (Hz)
nperseg = 500  # Segment length (500 ms)
noverlap = 256  # 50% overlap

# Index of frequency range for beta power corresponding to (13-30 Hz)
start_freq = 7
end_freq = 16

# We focus on the frontal regions
# Define frontal ROIs of shen atlas based on mask (subtract 1 for Python indexing)
frontal_rois = np.array(
    [2, 7, 10, 17, 18, 24, 25, 26, 28, 30, 31, 33,
     37, 38, 42, 50, 56, 59, 61, 62, 65, 66, 68, 71, 77,
     78, 83, 91, 92, 94, 96, 98, 99, 100, 101, 102, 103,
     108, 110, 113, 117, 125, 126, 129, 132, 133, 135, 137,
     140, 142, 150, 158, 161, 172, 178, 180, 182, 183]
) - 1

# Separate left and right hemisphere indices
right_frontal_idx = frontal_rois[frontal_rois < 93]
left_frontal_idx = frontal_rois[frontal_rois > 93]
emp_verb_psd = scipy.signal.welch(
    emp_source_verb[:, :, 1200:1700],
    fs=fs, noverlap=noverlap, nperseg=nperseg, detrend='linear'
)
emp_noise_psd = scipy.signal.welch(
    emp_source_noise[:, :, 1200:1700],
    fs=fs, noverlap=noverlap, nperseg=nperseg, detrend='linear'
)

# Simulated data format is (TRs, node_size), transpose to (node_size, TRs) to match welch input
sim_verb_psd = scipy.signal.welch(
    sim_source_verb[800:1300].T,
    fs=fs, noverlap=noverlap, nperseg=nperseg, detrend='linear'
)
sim_noise_psd = scipy.signal.welch(
    sim_source_noise[800:1300].T,
    fs=fs, noverlap=noverlap, nperseg=nperseg, detrend='linear'
)

# We average beta power across trials
emp_verb_beta = np.mean(emp_verb_psd[1][:, :, start_freq:end_freq], axis=(2))
emp_noise_beta = np.mean(emp_noise_psd[1][:, :, start_freq:end_freq], axis=(2))
sim_verb_beta = np.mean(sim_verb_psd[1][:, start_freq:end_freq], axis=1)
sim_noise_beta = np.mean(sim_noise_psd[1][:, start_freq:end_freq], axis=1)
emp_beta_diff = (np.mean(emp_verb_beta, axis=1)) - (np.mean(emp_noise_beta, axis=1))
sim_beta_diff = (np.array(sim_verb_beta)) - (np.array(sim_noise_beta))

# We seperate right and left regions to observe ERD in the left and ERS in the right
right_emp_avg = np.mean(emp_beta_diff[right_frontal_idx])
left_emp_avg = np.mean(emp_beta_diff[left_frontal_idx])
right_sim_avg = np.mean(sim_beta_diff[right_frontal_idx])
left_sim_avg = np.mean(sim_beta_diff[left_frontal_idx])

# Plot beta power difference in left and right frontal regions
labels = ['Data', 'Simulated']
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
ax.bar(x - width / 2, [left_emp_avg, left_sim_avg], width,
       label='Left Frontal', capsize=5, color='#6a9ef9', edgecolor='#6a9ef9')
ax.bar(x + width / 2, [right_emp_avg, right_sim_avg], width,
       label='Right Frontal', capsize=5, color='#e97773', edgecolor='#e97773')
ax.set_ylabel('Verb-Noise Beta Power', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.axhline(0, color='black', linewidth=1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
