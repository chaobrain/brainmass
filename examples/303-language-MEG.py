# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import functools
import os
from typing import Union, Callable

import brainstate
import braintools
import brainunit as u
import jax.tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainstate.nn import GaussianReg, Param, Const, ReluT, ExpT
from matplotlib.gridspec import GridSpec
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity

import brainmass

Parameter = Union[brainstate.nn.Param, brainstate.typing.ArrayLike, Callable]
Initializer = Union[Callable, brainstate.typing.ArrayLike]

brainstate.environ.set(dt=0.0001 * u.second)


def get_data():
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

    # Format input data
    data_verb = verb_meg.T
    data_noise = noise_meg.T

    # To simulate the auditory inputs in this task we will stimulate the auditory cortices
    # These nodes were identified using an ROI mask of left and right Heschl's gyri
    # based on the Talairach Daemon database
    ki0 = np.zeros(node_size)
    ki0[[2, 183, 5]] = 1

    return lm, sc, dist, data_verb, data_noise, ki0


class BrainModelTR(brainstate.nn.Module):
    """
    Whole-brain neural mass model for EEG simulation.
    """

    def __init__(
        self,
        tr_dynamics: brainstate.nn.Dynamics,
        leadfield: brainstate.nn.Module,
        tr: u.Quantity = 0.001 * u.second,
    ):
        super().__init__()

        self.dynamics = tr_dynamics
        self.leadfield = leadfield

        # TR parameters
        self.tr = tr
        self.dt_per_tr = int(tr / brainstate.environ.get_dt())

    def update(self, tr_inputs, record_state: bool = False):
        def fn_tr(inp_tr):
            inp_steps = u.math.tile(u.math.expand_dims(inp_tr, 0), (self.dt_per_tr,) + (1,) * inp_tr.ndim)
            return self.dynamics(inp_steps, record_state=record_state)

        if record_state:
            activities = brainstate.transform.for_loop(fn_tr, tr_inputs)
            obv = self.leadfield(activities[0])
            return obv, activities[1]
        else:
            activities = brainstate.transform.for_loop(fn_tr, tr_inputs)
            obv = self.leadfield(activities)
            return obv


class HORN_TR(brainstate.nn.Dynamics):
    def __init__(
        self,
        n_hidden,

        # structural coupling parameters
        sc: np.ndarray,
        dist: np.ndarray,
        mu: float,

        # dynamics parameters
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * u.math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # feedback
        state_init: Callable = braintools.init.ZeroInit(),
        delay_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__(n_hidden)

        # dynamics
        dynamics = brainmass.HORNStep(
            n_hidden, alpha=alpha, omega=omega, gamma=gamma, v=v, state_init=state_init)
        delay_time = dist / mu * brainstate.environ.get_dt()
        neuron_idx = np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
        h2h = brainmass.AdditiveCoupling(
            dynamics.prefetch_delay('y', delay_time, neuron_idx, init=delay_init),
            brainmass.LaplacianConnParam(sc),
        )
        dynamics.recurrent_fn = h2h
        self.dynamics = dynamics

    def update(self, inputs, record_state: bool = False):
        brainstate.transform.for_loop(self.dynamics, inputs)
        if record_state:
            return self.dynamics.x.value, {'x': self.dynamics.x.value, 'y': self.dynamics.y.value}
        else:
            return self.dynamics.x.value


class HORNNetworkTR(BrainModelTR):
    """
    HORN neural mass network for EEG simulation.
    """

    def __init__(
        self,
        n_hidden: int,

        # structural coupling parameters
        sc: np.ndarray,
        dist: np.ndarray,
        mu: float,

        # leadfield parameters
        cy0: Parameter,
        lm: Parameter,
        y0: Parameter,

        # other parameters
        tr: u.Quantity = 0.001 * u.second,
    ):
        # dynamics parameters
        alpha: Parameter = 0.04  # excitability
        omega: Parameter = 2. * u.math.pi / 28.  # natural frequency
        gamma: Parameter = 0.01  # damping
        v: Parameter = 0.0  # feedback

        # hyperparameters
        alpha = 0.04  # excitability
        omega_base = 2. * u.math.pi / 28.  # natural frequency
        gamma_base = 0.01  # damping
        omega_min = 0.5 * omega_base
        omega_max = 2.0 * omega_base
        gamma_min = 0.5 * gamma_base
        gamma_max = 2.0 * gamma_base
        omega = braintools.init.Uniform(omega_min, omega_max)(n_hidden)
        gamma = braintools.init.Uniform(gamma_min, gamma_max)(n_hidden)

        omega = Param(omega, t=ReluT())
        gamma = Param(gamma, t=ReluT())

        # state_init: Callable = braintools.init.ZeroInit()
        # delay_init: Callable = braintools.init.ZeroInit()

        state_init = braintools.init.Uniform(-0.01, 0.01)
        delay_init = braintools.init.Uniform(-0.01, 0.01)

        dynamics = HORN_TR(
            n_hidden,
            sc=sc,
            dist=dist,
            mu=mu,
            alpha=alpha,
            omega=omega,
            gamma=gamma,
            v=v,
            state_init=state_init,
            delay_init=delay_init,
        )

        # leadfiled matrix
        leadfield = brainmass.LeadfieldReadout(lm=lm, y0=y0, cy0=cy0)

        # super initialization
        super().__init__(dynamics, leadfield, tr=tr)


class JansenRit2NetworkTR(BrainModelTR):
    """
    Jansen-Rit neural mass network for EEG simulation.
    """

    def __init__(
        self,
        node_size: int,

        # structural coupling parameters
        sc: np.ndarray,
        dist: np.ndarray,
        mu: float,

        # leadfield parameters
        cy0: Parameter,
        lm: Parameter,
        y0: Parameter,

        # other parameters
        tr: u.Quantity = 0.001 * u.second,
    ):
        # A = Const(3.25, t=ReluT(), reg=GaussianReg(3.25, 0.1, fit_hyper=True))
        # a = Const(101, t=ReluT(1.), reg=GaussianReg(101, 1.0, fit_hyper=True))
        # B = Const(22, t=ReluT(), reg=GaussianReg(22, 0.5, fit_hyper=True))
        # b = Const(51, t=ReluT(1.), reg=GaussianReg(51, 1.0, fit_hyper=True))
        # g_l = Const(400, t=ReluT(0.01), reg=GaussianReg(400, 1.0, fit_hyper=True))
        # g_f = Const(10, t=ReluT(0.01), reg=GaussianReg(10, 1.0, fit_hyper=True))
        # g_b = Const(10, t=ReluT(0.01), reg=GaussianReg(10, 1.0, fit_hyper=True))
        # c1 = Const(135, t=ReluT(0.01), reg=GaussianReg(135, 1.0, fit_hyper=True))
        # c2 = Const(135 * 0.8, t=ReluT(0.01), reg=GaussianReg(135 * 0.8, 1.0, fit_hyper=True))
        # c3 = Const(135 * 0.25, t=ReluT(0.01), reg=GaussianReg(135 * 0.25, 1.0, fit_hyper=True))
        # c4 = Const(135 * 0.25, t=ReluT(0.01), reg=GaussianReg(135 * 0.25, 1.0, fit_hyper=True))
        A = Const(3.25)
        a = Const(101)
        B = Const(22)
        b = Const(51)
        g_l = Const(400)
        g_f = Const(10)
        g_b = Const(10)
        c1 = Const(135)
        c2 = Const(135 * 0.8)
        c3 = Const(135 * 0.25)
        c4 = Const(135 * 0.25)
        # std_in uses ExpT (no reg in original code)
        std_in = Param(6.0, t=ExpT(5.0))
        # Fixed parameters
        vmax = Const(5)
        v0 = Const(6)
        r = Const(0.56)
        # Fixed parameters
        kE = Const(0)
        kI = Const(0)
        k = Param(5.5, t=ReluT(0.5), reg=GaussianReg(5.5, 0.2, fit_hyper=True))
        # Array parameters
        lm_base = 0.01 * brainstate.random.randn_like(lm)
        lm_noise = 0.1 * brainstate.random.randn_like(lm)
        lm = Param(lm + lm_base + lm_noise)
        w_bb = Param(np.full((node_size, node_size), 0.05, dtype=brainstate.environ.dftype()))
        w_ff = Param(np.full((node_size, node_size), 0.05, dtype=brainstate.environ.dftype()))
        w_ll = Param(np.full((node_size, node_size), 0.05, dtype=brainstate.environ.dftype()))
        # initialization
        state_init = braintools.init.Uniform(-0.01, 0.01)
        delay_init = braintools.init.Uniform(-0.01, 0.01)

        dynamics = brainmass.JansenRit2TR(
            in_size=node_size,

            # dynamics parameters
            A=A,
            a=a,
            B=B,
            b=b,
            vmax=vmax,
            v0=v0,
            r=r,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            std_in=std_in,
            kE=kE,
            kI=kI,
            k=k,

            # distance parameters
            delay=dist / mu,

            # structural parameters
            sc=sc,
            w_ll=w_ll,
            w_ff=w_ff,
            w_bb=w_bb,
            g_l=g_l,
            g_f=g_f,
            g_b=g_b,

            # other parameters
            state_init=state_init,
            delay_init=delay_init,
        )
        leadfield = brainmass.LeadfieldReadout(lm=lm, y0=y0, cy0=cy0)

        super().__init__(dynamics, leadfield, tr=tr)


class ModelFitting:
    def __init__(
        self,
        model: BrainModelTR,
        data: np.ndarray,
        optimizer: braintools.optim.Optimizer,
    ):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.weights = model.states(brainstate.ParamState)
        self.optimizer.register_trainable_weights(self.weights)
        # define masks for getting lower triangle matrix indices
        self.mask_e = np.tril_indices(data.shape[-1], -1)
        self.output_size = data.shape[-1]

    def f_loss(self, tr_inputs, targets, n_warmup):
        with self.model.param_precompute():
            if n_warmup > 0:
                self.model.update(np.arange(n_warmup))
            eeg_output = self.model.update(tr_inputs)
        loss_main = u.math.sqrt(u.math.mean((eeg_output - targets) ** 2))
        loss = 10. * loss_main + self.model.reg_loss()
        return loss, eeg_output

    @brainstate.transform.jit(
        static_argnums=(0, 3),
        static_argnames=['n_warmup']
    )
    def f_train(self, tr_inputs, targets, n_warmup: int):
        f_grad = brainstate.transform.grad(
            functools.partial(self.f_loss, n_warmup=n_warmup),
            self.weights, has_aux=True, return_value=True, check_states=False
        )
        grads, loss, eeg_output = f_grad(tr_inputs, targets)
        self.optimizer.step(grads)
        return loss, eeg_output

    @brainstate.transform.jit(static_argnums=0)
    def f_predict(self, inputs):
        with self.model.param_precompute():
            eeg_output, state_output = self.model(inputs, record_state=True)
        return eeg_output, state_output

    def train(self, inputs, n_epoches: int, n_warmup: int = 0):
        loss_his = []
        for i_epoch in range(n_epoches):
            self.model.init_all_states()
            loss, eeg_output = self.f_train(inputs, self.data, n_warmup=n_warmup)

            loss_np = np.asarray(loss)
            loss_his.append(loss_np)

            fc_emp = np.corrcoef(self.data, rowvar=False)
            fc_sim = np.corrcoef(eeg_output[10:, :], rowvar=False)
            cor = np.corrcoef(fc_sim[self.mask_e], fc_emp[self.mask_e])[0, 1]
            sim = np.diag(cosine_similarity(eeg_output.T, self.data.T)).mean()

            print(f'epoch = {i_epoch}, loss = {loss_np}, FC cor = {cor}, cos sim = {sim}')

        return np.array(loss_his)

    def test(self, inputs, n_warmup: int = 0):
        self.model.init_all_states()
        if n_warmup > 0:
            self.f_predict(np.zeros(n_warmup))
        eeg_output, state_output = self.f_predict(inputs)

        transient_num = 20
        fc = np.corrcoef(self.data, rowvar=False)
        fc_sim = np.corrcoef(eeg_output[transient_num:], rowvar=False)

        cor = np.corrcoef(fc_sim[self.mask_e], fc[self.mask_e])[0, 1]
        sim = np.diag(cosine_similarity(eeg_output.T, self.data.T)).mean()
        print(f'Testing FC = {cor}, cos_sim = {sim}')
        return eeg_output, state_output


def visualize_state_output(
    state_output,
    eeg_output,
    data_target,
    sc=None,
    node_indices=None,
    mode='comprehensive',
    show_statistics=False,
    show=True
):
    """
    Visualize the state output from neural mass models with multiple visualization modes.

    Parameters
    ----------
    state_output : dict
        Dictionary containing state variables (e.g., 'P', 'E', 'I') with shape (time_steps, node_size)
    eeg_output : np.ndarray
        Predicted MEG/EEG signals with shape (time_steps, channels)
    data_target : np.ndarray
        Target MEG/EEG data with shape (time_steps, channels)
    sc : np.ndarray, optional
        Structural connectivity matrix (node_size, node_size). Required for 'representative' mode.
    node_indices : list, optional
        List of node indices to highlight. Default: [2, 183, 5]
    mode : str, optional
        Visualization mode: 'comprehensive' (12-panel overview), 'representative' (4-panel focused view),
        or 'both' (show both). Default: 'comprehensive'
    show_statistics : bool, optional
        Print statistical summary for representative nodes. Default: False
    show : bool, optional
        Display the figure(s). Default: True

    Returns
    -------
    fig or tuple of figs
        The created figure(s)
    """
    if node_indices is None:
        node_indices = [2, 183, 5]

    # Extract states and convert to numpy
    state_output, eeg_output, data_target = jax.tree.map(
        np.asarray, (state_output, eeg_output, data_target)
    )
    keys = tuple(state_output.keys())

    time_steps = eeg_output.shape[0]
    node_size = state_output[keys[0]].shape[1]
    time_ms = np.arange(time_steps) * 1.0  # TR = 1ms

    # Helper function: Select representative nodes
    def select_representative_nodes(sc_mat, n_nodes=8):
        """Select representative nodes based on connectivity and spatial distribution."""
        node_degree = np.sum(sc_mat, axis=1)
        auditory_nodes = [2, 183, 5]
        available = [i for i in range(node_size) if i not in auditory_nodes]

        hub_idx = available[np.argmax(node_degree[available])]
        peripheral_idx = available[np.argmin(node_degree[available])]

        excluded = set(auditory_nodes + [hub_idx, peripheral_idx])
        spatial_candidates = [i for i in range(node_size) if i not in excluded]

        early_spatial = spatial_candidates[np.argmin(np.abs(np.array(spatial_candidates) - 30))]
        middle_spatial = spatial_candidates[np.argmin(np.abs(np.array(spatial_candidates) - 94))]
        late_spatial = spatial_candidates[np.argmin(np.abs(np.array(spatial_candidates) - 160))]

        indices = auditory_nodes + [hub_idx, peripheral_idx, early_spatial, middle_spatial, late_spatial]
        labels = [
            f'Auditory-1 (N{auditory_nodes[0]})',
            f'Auditory-2 (N{auditory_nodes[1]})',
            f'Auditory-3 (N{auditory_nodes[2]})',
            f'Hub (N{hub_idx})',
            f'Peripheral (N{peripheral_idx})',
            f'Early (N{early_spatial})',
            f'Middle (N{middle_spatial})',
            f'Late (N{late_spatial})'
        ]
        types = ['auditory', 'auditory', 'auditory', 'hub', 'peripheral', 'spatial', 'spatial', 'spatial']
        colors = ['darkred', 'red', 'lightcoral', 'blue', 'green', 'darkgray', 'gray', 'lightgray']

        return {'indices': indices, 'labels': labels, 'types': types, 'colors': colors}

    # Helper function: Print statistics
    def print_statistics(node_info):
        """Print statistical summary of state values."""
        indices = node_info['indices']
        labels = node_info['labels']

        print("\n" + "=" * 80)
        print("REPRESENTATIVE NODE STATE STATISTICS")
        print("=" * 80)

        for node_idx, label in zip(indices, labels):
            print(f"\n{label}:")
            print("-" * 60)

            for state_name in keys:
                state = state_output[state_name][:, node_idx]
                peak_val = np.max(np.abs(state))
                peak_time = np.argmax(np.abs(state))
                mean_val = np.mean(state)
                std_val = np.std(state)

                print(f"  {state_name} State:")
                print(f"    Peak Value: {peak_val:>10.4f}  (at t={peak_time} ms)")
                print(f"    Mean:       {mean_val:>10.4f}  ±{std_val:.4f}")

        print("\n" + "=" * 80 + "\n")

    # COMPREHENSIVE VISUALIZATION
    def create_comprehensive_view():
        """Create 12-panel comprehensive visualization."""
        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Time series plots for selected nodes
        colors = plt.cm.viridis(np.linspace(0, 1, len(node_indices)))
        for col, (name, state) in enumerate(state_output.items()):
            ax = fig.add_subplot(gs[0, col])
            for idx, node_idx in enumerate(node_indices):
                ax.plot(time_ms, state[:, node_idx], label=f'Node {node_idx}', color=colors[idx], linewidth=1.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('State Value')
            ax.set_title(f'{name} - Selected Nodes')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvspan(100, 140, alpha=0.2, color='red', label='Stimulus')

        # Row 2: Heatmaps for all nodes
        for col, (name, state) in enumerate(state_output.items()):
            ax = fig.add_subplot(gs[1, col])
            im = ax.imshow(state.T, aspect='auto', cmap='RdBu_r', extent=(0, time_steps, 0, node_size), origin='lower')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Node Index')
            ax.set_title(f'{name} - All Nodes')
            ax.axvline(100, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
            ax.axvline(140, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
            plt.colorbar(im, ax=ax, label='State Value')

        # Row 3: Distributions of final state values
        for col, (name, state) in enumerate(state_output.items()):
            ax = fig.add_subplot(gs[2, col])
            st = state[-1, :]
            ax.hist(st, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(st.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {st.mean():.3f}')
            ax.axvline(st.mean() + st.std(), color='orange', linestyle=':', linewidth=2, alpha=0.7)
            ax.axvline(st.mean() - st.std(), color='orange', linestyle=':',
                       linewidth=2, alpha=0.7, label=f'Std: {st.std():.3f}')
            ax.set_xlabel('Final State Value')
            ax.set_ylabel('Count')
            ax.set_title(f'{name} - Final State Distribution')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        # Row 4: MEG comparison
        n_channels = min(3, eeg_output.shape[1])
        channel_indices = np.linspace(0, eeg_output.shape[1] - 1, n_channels, dtype=int)

        for col, ch_idx in enumerate(channel_indices):
            ax = fig.add_subplot(gs[3, col])
            ax.plot(time_ms, data_target[:, ch_idx], 'k--', linewidth=2, label='Target', alpha=0.7)
            ax.plot(time_ms, eeg_output[:, ch_idx], 'b-', linewidth=1.5, label='Predicted')

            corr = np.corrcoef(data_target[:, ch_idx], eeg_output[:, ch_idx])[0, 1]
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('MEG Signal')
            ax.set_title(f'Channel {ch_idx} (corr: {corr:.3f})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvspan(100, 140, alpha=0.2, color='red')

        plt.suptitle('Neural Mass Model State Dynamics - Comprehensive View',
                     fontsize=16, fontweight='bold', y=0.995)
        return fig

    # REPRESENTATIVE VISUALIZATION
    def create_representative_view(node_info):
        """Create 4-panel representative nodes visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        indices = node_info['indices']
        labels = node_info['labels']
        colors = node_info['colors']

        # Panel 1-3: State trajectories for P, E, I
        for panel_idx in range(3):
            ax = axes.flat[panel_idx]
            state_name = keys[panel_idx] if panel_idx < len(keys) else keys[0]
            state = state_output[state_name]

            for idx, node_idx in enumerate(indices):
                ax.plot(time_ms, state[:, node_idx], label=labels[idx], color=colors[idx], linewidth=2.0)
            ax.axvspan(100, 140, alpha=0.2, color='red', zorder=0)
            ax.set_xlabel('Time (ms)', fontsize=11)
            ax.set_ylabel('State Value', fontsize=11)
            ax.set_title(f'{state_name} State - Representative Nodes', fontsize=13, fontweight='bold')
            ax.legend(fontsize=8, loc='best', ncol=2)
            ax.grid(True, alpha=0.3, color='lightgray')

        # Panel 4: Combined P-E-I for primary auditory node
        ax = axes[1, 1]
        primary_node = indices[0]
        line_styles = ['-', '--', '-.']
        state_colors = ['blue', 'red', 'green']

        for idx, (state_name, style, color) in enumerate(zip(keys, line_styles, state_colors)):
            state = state_output[state_name]
            ax.plot(time_ms, state[:, primary_node], label=f'{state_name} State',
                    linestyle=style, color=color, linewidth=2.5)

        ax.axvspan(100, 140, alpha=0.2, color='red', zorder=0)
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('State Value', fontsize=11)
        ax.set_title(f'P-E-I Dynamics in Auditory Node {primary_node}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, color='lightgray')

        plt.suptitle('Representative Brain Region State Dynamics', fontsize=16, fontweight='bold', y=0.995)
        return fig

    # Main execution logic
    if mode == 'comprehensive':
        fig = create_comprehensive_view()
        if show:
            plt.show()
        return fig

    elif mode == 'representative':
        if sc is None:
            raise ValueError("Structural connectivity matrix 'sc' is required for representative mode")
        node_info = select_representative_nodes(sc)
        if show_statistics:
            print_statistics(node_info)
        fig = create_representative_view(node_info)
        if show:
            plt.show()
        return fig

    elif mode == 'both':
        if sc is None:
            raise ValueError("Structural connectivity matrix 'sc' is required for representative mode")
        node_info = select_representative_nodes(sc)
        if show_statistics:
            print_statistics(node_info)

        fig_comp = create_comprehensive_view()
        fig_rep = create_representative_view(node_info)

        if show:
            plt.show()
        return fig_comp, fig_rep

    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'comprehensive', 'representative', or 'both'")


def train_horn():
    lm, sc, dist, data_verb, data_noise, ki0 = get_data()
    node_size = dist.shape[0]
    uu = np.zeros((data_verb.shape[0], node_size))
    uu[100:140] = 5.0 * ki0
    # lm = braintools.init.KaimingNormal()(lm.shape)

    model = HORNNetworkTR(
        node_size,
        sc=sc, dist=dist * 10, mu=1., lm=lm, cy0=Const(5),
        y0=Param(-0.5, reg=GaussianReg(-0.5, 0.05, fit_hyper=True)),
    )
    fitting = ModelFitting(model, data_verb, braintools.optim.Adam(lr=5e-2))
    fitting.train(uu, n_epoches=40)
    eeg_output, state_output = fitting.test(uu)

    # Visualize with representative mode
    visualize_state_output(state_output, eeg_output, data_verb, sc=sc, mode='both', show_statistics=True, show=True)


def train_jr():
    lm, sc, dist, data_verb, data_noise, ki0 = get_data()
    node_size = dist.shape[0]
    uu = np.zeros((data_verb.shape[0], node_size))
    uu[100:140] = 5e3 * ki0

    model = JansenRit2NetworkTR(
        node_size, sc=sc, dist=dist, mu=1., lm=lm, cy0=Const(5),
        y0=Param(-0.5, reg=GaussianReg(-0.5, 0.05, fit_hyper=True)),
    )
    fitting = ModelFitting(model, data_verb, braintools.optim.Adam(lr=5e-2))
    fitting.train(uu, n_epoches=100)
    eeg_output, state_output = fitting.test(uu)

    # Visualize with representative mode
    visualize_state_output(state_output, eeg_output, data_verb, sc=sc, mode='both', show_statistics=True, show=True)


if __name__ == '__main__':
    train_horn()
    # train_jr()
