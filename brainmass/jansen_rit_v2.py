# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

from functools import partial
from typing import Callable

import brainstate.environ
import brainunit as u
import numpy as np

from braintools.param import Param, Data
from .typing import Array
from .dynamics import Dynamics
from .functions import sys2nd, sigmoid, bounded_input

__all__ = [
    "JansenRitWindow"
]


class JansenRitStep:
    def __init__(self, param):
        self.param = param
        self.u_2ndsys_ub = 500.

    def __call__(self, state: Data, input: Array, lEd: Array):
        param = self.param

        LEd_l, LEd_f, LEd_b = lEd
        P = state.P
        E = state.E
        I = state.I
        Pv = state.Pv
        Ev = state.Ev
        Iv = state.Iv

        ext = param.k * param.ki * input  # (node_size,)
        node_size = ext.shape[0]

        # 计算各群体的发放率
        rP = (
            ext +
            param.std_in * brainstate.random.randn(node_size) +  # (node_size,)
            param.g * (LEd_l + u.math.matmul(param.dg_l, P)) +  # (node_size,)
            sigmoid(state.E - I, param.vmax, param.v0, param.r)  # (node_size,) firing rate for Main population
        )
        rE = (
            param.kE +
            param.std_in * brainstate.random.randn(node_size) +  # (node_size,)
            param.g_f * (LEd_f + u.math.matmul(param.dg_f, E - I)) +  # (node_size,)
            param.c2 * sigmoid(param.c1 * P, param.vmax, param.v0, param.r)
        )
        rI = (
            param.kI +
            param.std_in * brainstate.random.randn(node_size) +  # (node_size,)
            -param.g_b * (LEd_b + u.math.matmul(param.dg_b, E - I)) +  # (node_size,)
            param.c4 * sigmoid(param.c3 * P, param.vmax, param.v0, param.r)
        )
        # Update the states by step-size.
        ddP = P + param.dt * Pv
        ddE = E + param.dt * Ev
        ddI = I + param.dt * Iv
        ddPv = Pv + param.dt * sys2nd(param.A, param.a, bounded_input(rP, self.u_2ndsys_ub), P, Pv)
        ddEv = Ev + param.dt * sys2nd(param.A, param.a, bounded_input(rE, self.u_2ndsys_ub), E, Ev)
        ddIv = Iv + param.dt * sys2nd(param.B, param.b, bounded_input(rI, self.u_2ndsys_ub), I, Iv)

        # Calculate the saturation for model states (for stability and gradient calculation).
        E = bounded_input(ddE, 1e3)
        I = bounded_input(ddI, 1e3)
        P = bounded_input(ddP, 1e3)
        Ev = bounded_input(ddEv, 1e3)
        Iv = bounded_input(ddIv, 1e3)
        Pv = bounded_input(ddPv, 1e3)
        return state.replace(E=E, I=I, P=P, Ev=Ev, Iv=Iv, Pv=Pv), None


class JansenRitTR:
    def __init__(self, param):
        self.param = param

    def __call__(self, state, inputs: Array):
        param = self.param
        step_model = JansenRitStep(param)

        Ed = state.delay[param.delay_idx, param.node_idx]
        LEd_b = u.math.sum(param.w_n_b * Ed.T, axis=1)
        LEd_f = u.math.sum(param.w_n_f * Ed.T, axis=1)
        LEd_l = u.math.sum(param.w_n_l * Ed.T, axis=1)

        state, _ = brainstate.transform.scan(partial(step_model, lEd=(LEd_b, LEd_f, LEd_l)), state, inputs)

        # hE: (delays_max, node_size)
        # E: (node_size,)
        hE = u.math.concatenate([u.math.expand_dims(state.P, 0), state.delay[:-1]], axis=0)
        state = state.replace(delay=hE)

        # Calculate EEG from leadfield and E-I difference
        eeg = param.cy0 * (param.lm @ (state.E - state.I)) - param.y0  # (output_size,)

        return state, (state.replace(delay=None), eeg)


class JansenRitWindow(Dynamics):
    """
    Jansen-Rit neural mass model for EEG simulation.

    A module for forward model (JansenRit) to simulate a batch of EEG signals.

    Attributes
    ----------
    tr : float
        Time resolution
    step_size : float
        Integration step for forward model
    steps_per_TR : int
        Number of integration steps in a tr
    TRs_per_window : int
        Number of EEG signals to simulate
    node_size : int
        Number of brain regions (ROIs)
    output_size : int
        Number of EEG channels
    sc : array
        Structural connectivity matrix (node_size x node_size)
    dist : array
        Distance matrix for conduction delay_idx
    """

    def __init__(
        self,
        node_size: int,
        step_size: float,
        output_size: int,
        tr: float,
        sc: np.ndarray,
        dist: np.ndarray,
        # Model parameters using Param API
        A: Param,
        a: Param,
        B: Param,
        b: Param,
        g: Param,
        g_f: Param,
        g_b: Param,
        c1: Param,
        c2: Param,
        c3: Param,
        c4: Param,
        std_in: Param,
        vmax: Param,
        v0: Param,
        r: Param,
        y0: Param,
        mu,
        k: Param,
        kE: Param,
        kI: Param,
        cy0: Param,
        ki: Param,
        lm: Param,
        w_bb: Param,
        w_ff: Param,
        w_ll: Param,
        state_init: Callable,
        delay_init: Callable,
        mask=None,
    ):
        super(JansenRitWindow, self).__init__()
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = step_size  # integration step 0.1 ms
        self.steps_per_TR = int(tr / step_size)
        self.node_size = node_size  # num of ROI
        self.output_size = output_size  # num of M/EEG channels
        self.sc = sc  # matrix node_size x node_size structure connectivity
        self.dist = dist

        # Register Param objects as submodules
        self.A = A
        self.a = a
        self.B = B
        self.b = b
        self.g = g
        self.g_f = g_f
        self.g_b = g_b
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.std_in = std_in
        self.vmax = vmax
        self.v0 = v0
        self.r = r
        self.y0 = y0
        self.mu = mu
        self.k = k
        self.kE = kE
        self.kI = kI
        self.cy0 = cy0
        self.ki = ki
        self.lm = lm
        self.w_bb = w_bb
        self.w_ff = w_ff
        self.w_ll = w_ll
        self.state_init = state_init
        self.delay_init = delay_init
        self.mask = mask

        # Calculate conduction delay_idx using mu.value()
        self.delay_idx = np.asarray(self.dist / self.mu, dtype=np.int64)
        self.node_idx = np.tile(np.expand_dims(np.arange(self.node_size), axis=1), (1, self.node_size))

    def create_initial_state(self, *args, **kwargs) -> Data:
        max_delay_len = self.delay_idx.max() + 1
        delay = self.delay_init((max_delay_len, self.node_size))
        state = self.state_init((self.node_size, 6))
        return Data(
            P=state[:, 0],
            E=state[:, 1],
            I=state[:, 2],
            Pv=state[:, 3],
            Ev=state[:, 4],
            Iv=state[:, 5],
            delay=delay,
        )

    def retrieve_params(self, *args, **kwargs) -> Data:
        dt = self.step_size

        # Extract all Param values at the beginning of forward pass
        A = self.A.value()
        a = self.a.value()
        B = self.B.value()
        b = self.b.value()
        g = self.g.value()
        g_f = self.g_f.value()
        g_b = self.g_b.value()
        c1 = self.c1.value()
        c2 = self.c2.value()
        c3 = self.c3.value()
        c4 = self.c4.value()
        std_in = self.std_in.value()
        vmax = self.vmax.value()
        v0 = self.v0.value()
        r = self.r.value()
        y0 = self.y0.value()
        k = self.k.value()
        kE = self.kE.value()
        kI = self.kI.value()
        cy0 = self.cy0.value()
        ki = self.ki.value()
        lm_val = self.lm.value()
        w_bb = self.w_bb.value()
        w_ff = self.w_ff.value()
        w_ll = self.w_ll.value()

        # Update the Laplacian based on the updated connection gains w_bb.
        w_b = u.math.exp(w_bb) * self.sc
        w_n_b = w_b / u.math.linalg.norm(w_b)
        if self.mask is not None:
            w_n_b = w_n_b * self.mask
        dg_b = -u.math.diag(u.math.sum(w_n_b, axis=1))

        # Update the Laplacian based on the updated connection gains w_ff.
        w_f = u.math.exp(w_ff) * self.sc
        w_n_f = w_f / u.linalg.norm(w_f)
        if self.mask is not None:
            w_n_f = w_n_f * self.mask
        dg_f = -u.math.diag(u.math.sum(w_n_f, axis=1))

        # Update the Laplacian based on the updated connection gains w_ll.
        w = u.math.exp(w_ll) * self.sc
        w = 0.5 * (w + u.math.transpose(w, (0, 1)))
        w_n_l = w / u.linalg.norm(w)
        if self.mask is not None:
            w_n_l = w_n_l * self.mask
        dg_l = -u.math.diag(u.math.sum(w_n_l, axis=1))

        # lm: (output_size, node_size), E-I: (node_size,)
        lm_t = (lm_val / u.math.sum(u.math.sqrt(lm_val ** 2), axis=1, keepdims=True))  # (output_size, node_size)
        lm_t = lm_t - u.math.mean(lm_t, axis=0, keepdims=True)

        return Data(
            dt=dt,
            A=A,
            a=a,
            B=B,
            b=b,
            g=g,
            g_f=g_f,
            g_b=g_b,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            std_in=std_in,
            vmax=vmax,
            v0=v0,
            r=r,
            y0=y0,
            k=k,
            kE=kE,
            kI=kI,
            cy0=cy0,
            ki=ki,
            lm=lm_t,
            w_n_b=w_n_b,
            w_n_f=w_n_f,
            w_n_l=w_n_l,
            dg_b=dg_b,
            dg_f=dg_f,
            dg_l=dg_l,
            delay_idx=self.delay_idx,
            node_idx=self.node_idx,
        )

    def update(self, state, param, input: Array):
        tr_model = JansenRitTR(param)
        state, (state_hist, eeg_hist) = brainstate.transform.scan(tr_model, state, input)
        return state, (eeg_hist, state_hist.to_dict())
