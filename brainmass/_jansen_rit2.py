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

from typing import Callable, Optional

import brainstate.environ
import braintools.init
import numpy as np
from brainstate import HiddenState
from brainstate.nn import Dynamics, Module, Param, StateWithDelay

from ._common import sys2nd, sigmoid, bounded_input
from ._coupling import LaplacianConnectivity
from ._leadfield import LeadfieldReadout
from ._noise import Noise, GaussianNoise
from ._typing import Array, Parameter

__all__ = [
    "JansenRit2Step",
    "JansenRit2TR",
    "JansenRit2Window",
]

Size = brainstate.typing.Size


class JansenRit2Step(Dynamics):
    def __init__(
        self,
        in_size: Size,
        # dynamics parameters
        A: Parameter,
        a: Parameter,
        B: Parameter,
        b: Parameter,
        vmax: Parameter,
        v0: Parameter,
        r: Parameter,
        c1: Parameter,
        c2: Parameter,
        c3: Parameter,
        c4: Parameter,
        kP: Parameter = 0.,
        kE: Parameter = 0.,
        kI: Parameter = 0.,
        noise_P: Noise = None,
        noise_E: Noise = None,
        noise_I: Noise = None,
        # other parameters
        state_saturation: bool = True,
        input_saturation: bool = True,
        state_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__(in_size)

        self.node_size = in_size
        self.state_saturation = state_saturation
        self.input_saturation = input_saturation
        self.state_init = state_init
        self.u_2ndsys_ub = 500.

        self.A = Param.init(A, self.varshape)
        self.a = Param.init(a, self.varshape)
        self.B = Param.init(B, self.varshape)
        self.b = Param.init(b, self.varshape)
        self.r = Param.init(r, self.varshape)
        self.v0 = Param.init(v0, self.varshape)
        self.c1 = Param.init(c1, self.varshape)
        self.c2 = Param.init(c2, self.varshape)
        self.c3 = Param.init(c3, self.varshape)
        self.c4 = Param.init(c4, self.varshape)
        self.kP = Param.init(kP, self.varshape)
        self.kE = Param.init(kE, self.varshape)
        self.kI = Param.init(kI, self.varshape)
        self.vmax = Param.init(vmax, self.varshape)

        self.noise_E = noise_E
        self.noise_I = noise_I
        self.noise_P = noise_P

    def init_state(self, *args, **kwargs):
        self.P = HiddenState.init(self.state_init, self.varshape)
        self.E = HiddenState.init(self.state_init, self.varshape)
        self.I = HiddenState.init(self.state_init, self.varshape)
        self.Pv = HiddenState.init(self.state_init, self.varshape)
        self.Ev = HiddenState.init(self.state_init, self.varshape)
        self.Iv = HiddenState.init(self.state_init, self.varshape)

    def update(self, inp_P=None, inp_E=None, inp_I=None):
        P = self.P.value
        E = self.E.value
        I = self.I.value
        Pv = self.Pv.value
        Ev = self.Ev.value
        Iv = self.Iv.value

        A = self.A.value()
        B = self.B.value()
        a = self.a.value()
        b = self.b.value()
        vmax = self.vmax.value()
        v0 = self.v0.value()
        r = self.r.value()
        c1 = self.c1.value()
        c2 = self.c2.value()
        c3 = self.c3.value()
        c4 = self.c4.value()
        kP = self.kP.value()
        kE = self.kE.value()
        kI = self.kI.value()

        # 计算各群体的发放率
        rP = kP + sigmoid(E - I, vmax, v0, r)
        if inp_P is not None:
            rP = rP + inp_P
        if self.noise_P is not None:
            rP = rP + self.noise_P.update()

        rE = kE + c2 * sigmoid(c1 * P, vmax, v0, r)
        if inp_E is not None:
            rE = rE + inp_E
        if self.noise_E is not None:
            rE = rE + self.noise_E.update()

        rI = kI + c4 * sigmoid(c3 * P, vmax, v0, r)
        if inp_I is not None:
            rI = rI + inp_I
        if self.noise_I is not None:
            rI = rI + self.noise_I.update()

        # Update the states by step-size.
        dt = brainstate.environ.get_dt()
        ddP = P + dt * Pv
        ddE = E + dt * Ev
        ddI = I + dt * Iv
        ddPv = Pv + dt * sys2nd(A, a, bounded_input(rP, self.u_2ndsys_ub), P, Pv)
        ddEv = Ev + dt * sys2nd(A, a, bounded_input(rE, self.u_2ndsys_ub), E, Ev)
        ddIv = Iv + dt * sys2nd(B, b, bounded_input(rI, self.u_2ndsys_ub), I, Iv)

        # Calculate the saturation for model states (for stability and gradient calculation).
        self.E.value = bounded_input(ddE, 1e3)
        self.I.value = bounded_input(ddI, 1e3)
        self.P.value = bounded_input(ddP, 1e3)
        self.Ev.value = bounded_input(ddEv, 1e3)
        self.Iv.value = bounded_input(ddIv, 1e3)
        self.Pv.value = bounded_input(ddPv, 1e3)


class JansenRit2TR(Dynamics):
    def __init__(
        self,
        in_size: Size,
        tr: float,

        # neuronal dynamics parameters
        A: Parameter,
        a: Parameter,
        B: Parameter,
        b: Parameter,
        vmax: Parameter,
        v0: Parameter,
        r: Parameter,
        k: Parameter,
        c1: Parameter,
        c2: Parameter,
        c3: Parameter,
        c4: Parameter,
        kE: Parameter,
        kI: Parameter,

        # distance parameters
        delay: Array,

        # structural connectivity parameters
        sc: Array,
        w_ll: Parameter,
        w_ff: Parameter,
        w_bb: Parameter,
        g_l: Parameter,
        g_f: Parameter,
        g_b: Parameter,

        # other parameters
        std_in: Parameter = None,
        mask: Optional[Array] = None,
        state_saturation: bool = True,
        input_saturation: bool = True,
        state_init: Callable = braintools.init.ZeroInit(),
        delay_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__(in_size)

        self.k = Param.init(k)

        # single step dynamics
        self.step = JansenRit2Step(
            in_size=in_size,
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
            kE=kE,
            kI=kI,
            state_saturation=state_saturation,
            input_saturation=input_saturation,
            state_init=state_init,
            noise_P=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
            noise_E=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
            noise_I=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
        )

        # delay
        self.delay = StateWithDelay(self.step, 'E', init=delay_init)
        n_hidden = self.varshape[0]
        dt = brainstate.environ.get_dt()
        neuron_idx = np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
        self.delay_access = self.delay.access('delay', delay * dt, neuron_idx)

        # connectivity
        self.conn = LaplacianConnectivity(
            self.delay_access,
            self.step.prefetch('P'),
            self.step.prefetch('E'),
            self.step.prefetch('I'),
            sc=sc,
            w_ll=w_ll,
            w_ff=w_ff,
            w_bb=w_bb,
            g_l=g_l,
            g_f=g_f,
            g_b=g_b,
            mask=mask,
        )

    def update(self, inputs: Array, record_state: bool = False):
        k = self.k.value()

        def step(inp):
            inp_P, inp_E, inp_I = self.conn()
            self.step.update(inp_P + k * inp, inp_E, inp_I)

        assert inputs.ndim == 2, f'Expected inputs to be 2D array, but got {inputs.ndim}D array.'
        brainstate.transform.for_loop(step, inputs)
        self.delay.update(self.step.E.value)
        activity = self.step.E.value - self.step.I.value

        if record_state:
            state = dict(
                P=self.step.P.value,
                E=self.step.E.value,
                I=self.step.I.value,
            )
            return activity, state

        return activity


class JansenRit2Window(Module):
    """
    Jansen-Rit neural mass model for EEG simulation.

    A module for forward model (JansenRit) to simulate a batch of EEG signals.

    """

    def __init__(
        self,
        node_size: int,
        tr: float,
        sc: np.ndarray,
        dist: np.ndarray,
        mu: np.ndarray,
        # Model parameters using Param API
        A: Parameter,
        a: Parameter,
        B: Parameter,
        b: Parameter,
        g: Parameter,
        g_f: Parameter,
        g_b: Parameter,
        c1: Parameter,
        c2: Parameter,
        c3: Parameter,
        c4: Parameter,
        k: Parameter,
        std_in: Parameter,
        vmax: Parameter,
        v0: Parameter,
        r: Parameter,
        y0: Parameter,
        kE: Parameter,
        kI: Parameter,
        cy0: Parameter,
        lm: Parameter,
        w_bb: Parameter,
        w_ff: Parameter,
        w_ll: Parameter,
        state_init: Callable,
        delay_init: Callable,
        mask=None,
    ):
        super().__init__()

        self.dynamics = JansenRit2TR(
            in_size=node_size,
            tr=tr,

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
            g_l=g,
            g_f=g_f,
            g_b=g_b,

            # other parameters
            mask=mask,
            state_init=state_init,
            delay_init=delay_init,
        )
        self.leadfield = LeadfieldReadout(lm=lm, y0=y0, cy0=cy0)

    def update(self, inputs, record_state: bool = False):
        if record_state:
            activities, states = brainstate.transform.for_loop(
                lambda inp: self.dynamics(inp, record_state=True), inputs
            )
            return self.leadfield(activities), states
        else:
            activities = brainstate.transform.for_loop(self.dynamics, inputs)
            return self.leadfield(activities)
