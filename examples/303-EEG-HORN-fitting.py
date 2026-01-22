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

import braintools
import brainunit as u
import jax.tree
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics.pairwise import cosine_similarity

import brainmass
import brainstate
from brainstate.nn import Param, Const, SoftplusT

Parameter = Union[brainstate.nn.Param, brainstate.typing.ArrayLike, Callable]
Initializer = Union[Callable, brainstate.typing.ArrayLike]

brainstate.environ.set(dt=0.0001 * u.second)


def load_subject_eeg_data(
    subj: int = 2,
    scope: tuple = (200, 500),
    t_start: int = 290,
):
    data = np.load(
        os.path.join('data/ccepcoreg-eeg-data', f'sub-{subj}-eeg-data.npy'),
        allow_pickle=True
    ).item()

    n_trial = data['eeg_data'].shape[0]
    i_start = scope[0]
    i_end = scope[1]
    eeg_data = data['eeg_data'][:, i_start:i_end]
    stim_duration = np.asarray([float(s.replace('ms', '')) for s in data['stim_duration']])
    stim_duration = np.asarray(stim_duration * 20, dtype=np.int32)
    print(stim_duration)
    stim_intensity = np.asarray([float(s.replace('ma', '')) for s in data['stim_intensity']]) * 1e1
    stim_weights = data['stim_weights']
    uu = np.zeros((n_trial, eeg_data.shape[1], data['dist'].shape[0]))
    for i_trial in range(n_trial):
        ind = t_start - i_start
        dur = stim_duration[i_trial]
        uu[i_trial, ind: ind + dur] = stim_weights[i_trial] * stim_intensity[i_trial]
    return data['lm'], data['sc'], data['dist'], eeg_data, uu


class CosineSimReg(brainstate.nn.Regularization):
    def __init__(self, sc):
        super().__init__()
        self.sc = u.math.flatten(sc)

    def loss(self, value):
        return braintools.metric.cosine_similarity(
            u.math.flatten(value),
            self.sc
        )


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
        tr: u.Quantity = 1e-3 * u.second,
    ):
        super().__init__(n_hidden)

        # dynamics
        dynamics = brainmass.HORNStep(
            n_hidden, alpha=alpha, omega=omega, gamma=gamma, v=v, state_init=state_init,
        )
        dt = brainstate.environ.get_dt()
        delay_time = dist / mu * dt
        self.h2h = brainmass.AdditiveCoupling(
            dynamics.prefetch_delay(
                'y', delay_time, brainmass.delay_index(n_hidden), init=delay_init,
            ),
            # Param(sc, reg=CosineSimReg(sc)),
            Param(braintools.init.KaimingNormal()(sc.shape)),
        )
        self.dynamics = dynamics
        self.tr = tr

    def update(self, inputs, record_state: bool = False):
        def step(i):
            self.dynamics(self.h2h() + inputs)

        n_step = int(self.tr / brainstate.environ.get_dt())
        brainstate.transform.for_loop(step, np.arange(n_step))
        if record_state:
            return self.dynamics.x.value, {'x': self.dynamics.x.value, 'y': self.dynamics.y.value}
        else:
            return self.dynamics.x.value


class HORNNetwork(brainstate.nn.Module):
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
        super().__init__()

        # hyperparameters
        v: Parameter = 0.0  # feedback
        alpha = 0.04  # excitability
        omega_base = 2. * u.math.pi / 28.  # natural frequency
        gamma_base = 0.01  # damping
        omega_min = 0.5 * omega_base
        omega_max = 2.0 * omega_base
        gamma_min = 0.5 * gamma_base
        gamma_max = 2.0 * gamma_base
        omega = braintools.init.Uniform(omega_min, omega_max)(n_hidden)
        gamma = braintools.init.Uniform(gamma_min, gamma_max)(n_hidden)
        omega = Param(omega, t=SoftplusT(0.001))
        gamma = Param(gamma, t=SoftplusT(0.001))

        self.dynamics = HORN_TR(
            n_hidden,
            sc=sc,
            dist=dist,
            mu=mu,
            alpha=alpha,
            omega=omega,
            gamma=gamma,
            v=v,
            state_init=braintools.init.Uniform(-0.01, 0.01),
            delay_init=braintools.init.Uniform(-0.01, 0.01),
        )
        self.leadfield = brainmass.LeadfieldReadout(
            lm=Param(lm + 0.11 * brainstate.random.randn_like(lm)),
            y0=y0,
            cy0=cy0,
        )

        self.n_hidden = self.dynamics.varshape[0]
        self.n_output = lm.shape[0]
        self.tr = tr

    def update(self, tr_inputs, record_state: bool = False):
        fn_tr = lambda inp_tr: self.dynamics(inp_tr, record_state=record_state)

        if record_state:
            activities = brainstate.transform.for_loop(fn_tr, tr_inputs)
            obv = self.leadfield(activities[0])
            return obv, activities[1]
        else:
            activities = brainstate.transform.for_loop(fn_tr, tr_inputs)
            obv = self.leadfield(activities)
            return obv

    def save_states(self, model_path):
        model_states = self.states(brainstate.ParamState)
        braintools.file.msgpack_save(model_path, model_states)

    def load_states(self, model_path):
        model_states = self.states(brainstate.ParamState)
        braintools.file.msgpack_load(model_path, model_states)


class ModelFitting:
    def __init__(
        self,
        model: HORNNetwork,
        optimizer: braintools.optim.OptaxOptimizer,
    ):
        self.model = model
        self.optimizer = optimizer
        self.weights = model.states(brainstate.ParamState)
        self.optimizer.register_trainable_weights(self.weights)

        # define masks for getting lower triangle matrix indices
        self.mask_e = np.tril_indices(model.n_output, -1)

    def f_simulate(self, inputs, record_state=False):
        vmap_model = brainstate.nn.ModuleMapper(self.model, init_map_size=inputs.shape[0])
        vmap_model.init_all_states()
        with vmap_model.param_precompute():
            fn = functools.partial(self.model.update, record_state=record_state)
            return vmap_model.map(fn)(inputs)

    def f_loss(self, tr_inputs, targets):
        eeg_output = self.f_simulate(tr_inputs)
        loss_main = u.math.sqrt(u.math.mean((eeg_output - targets) ** 2))
        loss = 10. * loss_main + self.model.reg_loss()
        return loss, eeg_output

    @brainstate.transform.jit(static_argnums=0)
    def f_train(self, tr_inputs, targets):
        f_grad = brainstate.transform.grad(
            self.f_loss, self.weights, has_aux=True, return_value=True, check_states=False
        )
        grads, loss, eeg_output = f_grad(tr_inputs, targets)
        self.optimizer.step(grads)
        return loss, eeg_output

    @brainstate.transform.jit(static_argnums=0)
    def f_predict(self, inputs):
        return self.f_simulate(inputs, record_state=True)

    def train(self, inputs, targets, n_epoches: int, n_transient=20):
        fc_sims = [np.corrcoef(tar, rowvar=False)[self.mask_e] for tar in targets]

        loss_his = []
        for i_epoch in range(n_epoches):
            loss, eeg_output = self.f_train(inputs, targets)
            loss_his.append(np.asarray(loss))

            cors, sims = [], []
            for i_trial in range(eeg_output.shape[0]):
                eeg_out = eeg_output[i_trial]
                fc_sim = np.corrcoef(eeg_out[n_transient:], rowvar=False)
                cor = np.corrcoef(fc_sim[self.mask_e], fc_sims[i_trial])[0, 1]
                tar = targets[i_trial]
                sim = np.diag(cosine_similarity(eeg_out.T, tar.T)).mean()
                cors.append(cor)
                sims.append(sim)
            print(f'epoch = {i_epoch}, loss = {loss_his[-1]}, FC cor = {np.mean(cors)}, cos sim = {np.mean(sims)}')

        return np.array(loss_his)

    def test(self, inputs, targets, n_transient=20):
        eeg_output, state_output = self.f_predict(inputs)

        cors, sims = [], []
        for i_trial in range(eeg_output.shape[0]):
            tar = targets[i_trial]
            eeg_out = eeg_output[i_trial]
            fc = np.corrcoef(tar, rowvar=False)
            fc_sim = np.corrcoef(eeg_out[n_transient:], rowvar=False)
            cor = np.corrcoef(fc_sim[self.mask_e], fc[self.mask_e])[0, 1]
            sim = np.diag(cosine_similarity(eeg_out.T, tar.T)).mean()
            cors.append(cor)
            sims.append(sim)
        print(f'Testing FC = {np.mean(cors)}, cos_sim = {np.mean(sims)}')
        return eeg_output, state_output


def visualize_state_output(
    state_output,
    eeg_output,
    data_target,
    sc=None,
    node_indices=None,
    stimulus_window=None,
    trial_indices=None,
    show=True
):
    """
    Visualize the state output from neural mass models with support for multiple trials.

    Parameters
    ----------
    state_output : dict
        Dictionary containing state variables (e.g., 'M', 'E', 'I') with shape
        (time_steps, node_size) for single trial or (batch, time_steps, node_size) for multiple trials
    eeg_output : np.ndarray
        Predicted MEG/EEG signals with shape (time_steps, channels) for single trial
        or (batch, time_steps, channels) for multiple trials
    data_target : np.ndarray
        Target MEG/EEG data with shape (time_steps, channels) for single trial
        or (batch, time_steps, channels) for multiple trials
    sc : np.ndarray, optional
        Structural connectivity matrix (node_size, node_size).
    node_indices : list, optional
        List of node indices to highlight. Default: automatically selected from sc if available,
        otherwise [2, 183, 5]
    stimulus_window : tuple, optional
        Tuple of (start_time, end_time) for stimulus window in milliseconds. Default: (100, 140)
    trial_indices : list, optional
        List of trial indices to visualize when data has batch dimension. Default: all trials (up to 4)
    show : bool, optional
        Display the figure(s). Default: True

    Returns
    -------
    fig or list of figs
        The created figure(s)
    """
    if stimulus_window is None:
        stimulus_window = (100, 140)

    stim_start, stim_end = stimulus_window

    # Extract states and convert to numpy
    state_output, eeg_output, data_target = jax.tree.map(
        np.asarray, (state_output, eeg_output, data_target)
    )
    keys = tuple(state_output.keys())

    # Check if data has batch dimension (3D for states, 3D for eeg)
    first_state = state_output[keys[0]]
    has_batch = first_state.ndim == 3

    if has_batch:
        n_trials = first_state.shape[0]
        time_steps = first_state.shape[1]
        node_size = first_state.shape[2]

        # Select trials to visualize
        if trial_indices is None:
            trial_indices = list(range(min(n_trials, 4)))  # Default: up to 4 trials
        n_trials_to_show = len(trial_indices)
    else:
        n_trials_to_show = 1
        time_steps = eeg_output.shape[0]
        node_size = first_state.shape[1]
        trial_indices = [0]
        # Add batch dimension for uniform processing
        state_output = {k: v[np.newaxis, ...] for k, v in state_output.items()}
        eeg_output = eeg_output[np.newaxis, ...]
        data_target = data_target[np.newaxis, ...]

    time_ms = np.arange(time_steps) * 1.0  # TR = 1ms

    # Set default node_indices if not provided
    if node_indices is None:
        if sc is not None:
            node_degree = np.sum(sc, axis=1)
            node_indices = np.argsort(node_degree)[-3:].tolist()
        else:
            node_indices = [2, 183, 5] if node_size > 183 else list(range(min(3, node_size)))

    def create_trial_view(trial_idx, trial_label):
        """Create visualization for a single trial."""
        state_trial = {k: v[trial_idx] for k, v in state_output.items()}
        eeg_trial = eeg_output[trial_idx]
        target_trial = data_target[trial_idx]

        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Row 1: Time series plots for selected nodes
        colors = plt.cm.viridis(np.linspace(0, 1, len(node_indices)))
        for col, (name, state) in enumerate(state_trial.items()):
            state = u.get_magnitude(state)
            ax = fig.add_subplot(gs[0, col])
            for idx, node_idx in enumerate(node_indices):
                ax.plot(time_ms, state[:, node_idx], label=f'Node {node_idx}', color=colors[idx], linewidth=1.5)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('State Value')
            ax.set_title(f'{name} - Selected Nodes')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvspan(stim_start, stim_end, alpha=0.2, color='red', label='Stimulus')

        # Row 2: Heatmaps for all nodes
        for col, (name, state) in enumerate(state_trial.items()):
            state = u.get_magnitude(state)
            ax = fig.add_subplot(gs[1, col])
            im = ax.imshow(state.T, aspect='auto', cmap='RdBu_r', extent=(0, time_steps, 0, node_size), origin='lower')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Node Index')
            ax.set_title(f'{name} - All Nodes')
            ax.axvline(stim_start, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
            ax.axvline(stim_end, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
            plt.colorbar(im, ax=ax, label='State Value')

        # Row 3: Distributions of final state values
        for col, (name, state) in enumerate(state_trial.items()):
            state = u.get_magnitude(state)
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
        n_channels = min(3, eeg_trial.shape[1])
        channel_indices = np.linspace(0, eeg_trial.shape[1] - 1, n_channels, dtype=int)

        for col, ch_idx in enumerate(channel_indices):
            ax = fig.add_subplot(gs[3, col])
            ax.plot(time_ms, target_trial[:, ch_idx], 'k--', linewidth=2, label='Target', alpha=0.7)
            ax.plot(time_ms, eeg_trial[:, ch_idx], 'b-', linewidth=1.5, label='Predicted')

            corr = np.corrcoef(target_trial[:, ch_idx], eeg_trial[:, ch_idx])[0, 1]
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('MEG Signal')
            ax.set_title(f'Channel {ch_idx} (corr: {corr:.3f})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvspan(stim_start, stim_end, alpha=0.2, color='red')

        plt.suptitle(f'Neural Mass Model State Dynamics - {trial_label}',
                     fontsize=16, fontweight='bold', y=0.995)
        return fig

    def create_multi_trial_summary():
        """Create a summary figure comparing all trials."""
        n_states = len(keys)
        fig = plt.figure(figsize=(6 * n_trials_to_show, 4 * (n_states + 1)))
        gs = GridSpec(n_states + 1, n_trials_to_show, figure=fig, hspace=0.35, wspace=0.25)

        trial_colors = plt.cm.tab10(np.linspace(0, 1, n_trials_to_show))
        extent = (0, time_steps, 0, node_size)

        # Rows for each state variable: heatmaps across trials
        for row, (name, state_all) in enumerate(state_output.items()):
            for col, (ti, trial_idx) in enumerate(enumerate(trial_indices)):
                state = u.get_magnitude(state_all[trial_idx])
                ax = fig.add_subplot(gs[row, col])
                im = ax.imshow(state.T, aspect='auto', cmap='RdBu_r', extent=extent, origin='lower')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Node Index')
                ax.set_title(f'{name} - Trial {trial_idx}')
                ax.axvline(stim_start, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
                ax.axvline(stim_end, color='yellow', linestyle='--', linewidth=1, alpha=0.7)
                plt.colorbar(im, ax=ax, label='State Value')

        # Last row: EEG comparison for each trial
        for col, (ti, trial_idx) in enumerate(enumerate(trial_indices)):
            eeg_trial = eeg_output[trial_idx]
            target_trial = data_target[trial_idx]
            ax = fig.add_subplot(gs[n_states, col])

            # Plot mean across channels
            target_mean = target_trial.mean(axis=1)
            pred_mean = eeg_trial.mean(axis=1)
            ax.plot(time_ms, target_mean, 'k--', linewidth=2, label='Target (mean)', alpha=0.7)
            ax.plot(time_ms, pred_mean, color=trial_colors[ti], linewidth=1.5, label='Predicted (mean)')

            # Add shaded region for std
            target_std = target_trial.std(axis=1)
            pred_std = eeg_trial.std(axis=1)
            ax.fill_between(time_ms, target_mean - target_std, target_mean + target_std, color='gray', alpha=0.2)
            ax.fill_between(time_ms, pred_mean - pred_std, pred_mean + pred_std, color=trial_colors[ti], alpha=0.2)

            corr = np.corrcoef(target_mean, pred_mean)[0, 1]
            rmse = np.sqrt(np.mean((target_mean - pred_mean) ** 2))
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('EEG Signal (mean)')
            ax.set_title(f'Trial {trial_idx} (corr: {corr:.3f}, RMSE: {rmse:.3f})')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axvspan(stim_start, stim_end, alpha=0.2, color='red')

        plt.suptitle('Multi-Trial Summary - State Dynamics & EEG Comparison',
                     fontsize=16, fontweight='bold', y=0.995)
        return fig

    # Main execution logic
    figs = []

    # Create individual trial views
    for ti, trial_idx in enumerate(trial_indices):
        trial_label = f'Trial {trial_idx}' if has_batch or n_trials_to_show > 1 else 'Single Trial'
        fig = create_trial_view(trial_idx, trial_label)
        figs.append(fig)

    # Create multi-trial summary if multiple trials
    if n_trials_to_show > 1:
        fig_summary = create_multi_trial_summary()
        figs.append(fig_summary)

    if show:
        plt.show()

    return figs if len(figs) > 1 else figs[0]


def train_hdeeg_jr():
    scope = (250, 400)
    scope = (100, 800)
    lm, sc, dist, activities, inputs = load_subject_eeg_data(subj=1, scope=scope)

    model = HORNNetwork(
        dist.shape[0],
        sc=sc,
        dist=dist,
        mu=2.,
        lm=lm,
        cy0=Const(5),
        y0=Param(u.math.zeros(lm.shape[0])),
    )
    fitting = ModelFitting(model, braintools.optim.Adam(lr=5e-2))
    fitting.train(inputs, activities, n_epoches=200)
    eeg_output, state_output = fitting.test(inputs, activities)

    visualize_state_output(state_output, eeg_output, activities, sc=sc, show=True)


if __name__ == '__main__':
    pass
    train_hdeeg_jr()
