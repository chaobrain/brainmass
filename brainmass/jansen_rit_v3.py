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

import brainstate
import brainunit as u
import jax
import numpy as np
from brainstate.util.struct import dataclass

from braintools.param import Param, Data
from .typing import Array
from .dynamics import Dynamics
from .functions import sys2nd, sigmoid

__all__ = ["JansenRitModel"]


class JansenRitStep:
    def __init__(self, param: Data):
        self.param = param
        self.u_2ndsys_ub = 500.

    def __call__(self, state: Data, input: Array, LEd: Array):
        param = self.param
        E = state.E
        I = state.I
        P = state.P
        Ev = state.Ev
        Iv = state.Iv
        Pv = state.Pv

        ext = param.k * param.ki * input

        # Calculate firing rates (transforms handle lower bounds)
        rM = sigmoid(E - I, param.vmax, param.v0, param.r)
        rE = (
            param.std_in * brainstate.random.randn_like(E) +
            param.g * (LEd + u.math.matmul(param.dg, E)) +
            param.c2 * sigmoid(param.c1 * P, param.vmax, param.v0, param.r)
        )
        rI = param.c4 * sigmoid(param.c3 * P, param.vmax, param.v0, param.r)

        # Update states using Euler integration
        ddM = P + param.dt * Pv
        ddE = E + param.dt * Ev
        ddI = I + param.dt * Iv
        ddMv = Pv + param.dt * sys2nd(
            param.A,
            param.a,
            self.u_2ndsys_ub * u.math.tanh(rM / self.u_2ndsys_ub),
            P,
            Pv
        )
        ddEv = Ev + param.dt * sys2nd(
            param.A,
            param.a,
            ext + self.u_2ndsys_ub * u.math.tanh(rE / self.u_2ndsys_ub),
            E,
            Ev
        )
        ddIv = Iv + param.dt * sys2nd(
            param.B,
            param.b,
            self.u_2ndsys_ub * u.math.tanh(rI / self.u_2ndsys_ub),
            I,
            Iv
        )

        # Update state variables
        E = ddE
        I = ddI
        P = ddM
        Ev = ddEv
        Iv = ddIv
        Pv = ddMv
        return state.replace(E=E, I=I, P=P, Ev=Ev, Iv=Iv, Pv=Pv), None


class JansenRitTR:
    def __init__(self, param: Data):
        self.param = param

    def __call__(self, state: Data, inputs: Array):
        param = self.param
        step_model = JansenRitStep(param)

        Ed = state.delay[param.delay_idx, param.node_idx]
        LEd = u.math.sum(param.w_n * Ed.T, axis=1)
        state, _ = brainstate.transform.scan(partial(step_model, LEd=LEd), state, inputs)

        # 更新延迟缓冲区：将当前 E 添加到历史记录的开头
        # hE: (delays_max, node_size), E: (node_size,)
        hE = u.math.concatenate([u.math.expand_dims(state.P, 0), state.delay[:-1]], axis=0)
        state = state.replace(delay=hE)

        # Calculate EEG from leadfield and E-I difference
        eeg = param.cy0 * (param.lm @ (state.E - state.I)) - param.y0  # (output_size,)

        return state, (state.replace(delay=None), eeg)


class JansenRitModel(Dynamics):
    """
    Jansen-Rit neural mass model for EEG simulation.

    A module for forward model (JansenRit) to simulate a batch of EEG signals.

    Attributes
    ----------
    state_size : int
        Number of states in the JansenRit model (6)
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
        TRs_per_window: int,
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
        c1: Param,
        c2: Param,
        c3: Param,
        c4: Param,
        std_in: Param,
        vmax: Param,
        v0: Param,
        r: Param,
        y0: Param,
        mu: Param,
        k: Param,
        cy0: Param,
        ki: Param,
        lm: Param,
        w_bb: Param,
        state_init: Callable,
        delay_init: Callable,
    ):
        super(JansenRitModel, self).__init__()
        self.state_size = 6
        self.tr = tr
        self.step_size = step_size
        self.steps_per_TR = np.int64(tr / step_size)
        self.TRs_per_window = TRs_per_window
        self.node_size = node_size
        self.output_size = output_size
        self.sc = sc
        self.dist = dist

        # Register Param objects as submodules
        self.A = A
        self.a = a
        self.B = B
        self.b = b
        self.g = g
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
        self.cy0 = cy0
        self.ki = ki
        self.lm = lm
        self.w_bb = w_bb
        self.state_init = state_init
        self.delay_init = delay_init

        # Calculate conduction delay_idx using mu.value()
        self.delay_idx = np.asarray(self.dist / self.mu.value(), dtype=np.int64)
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
        cy0 = self.cy0.value()
        ki = self.ki.value()
        lm_val = self.lm.value()
        w_bb = self.w_bb.value()

        # Update connection matrix based on learnable gains
        w = u.math.exp(w_bb) * self.sc
        w_log = u.math.log1p(0.5 * (w + u.math.transpose(w, (0, 1))))
        w_n = w_log / u.linalg.norm(w_log)
        dg = -u.math.diag(u.math.sum(w_n, axis=1))

        # lm: (output_size, node_size), E-I: (node_size,)
        lm_t = (lm_val - 1 / self.output_size * u.math.matmul(u.math.ones((1, self.output_size)), lm_val))

        return Data(
            dt=dt,
            A=A,
            a=a,
            B=B,
            b=b,
            g=g,
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
            cy0=cy0,
            ki=ki,
            lm=lm_t,
            w_n=w_n,
            dg=dg,
            delay_idx=self.delay_idx,
            node_idx=self.node_idx,
        )

    def update(self, state: Data, param: Data, input: Array):
        tr_model = JansenRitTR(param)
        state, (state_hist, eeg_hist) = brainstate.transform.scan(tr_model, state, input)
        return state, (eeg_hist, state_hist.to_dict())
