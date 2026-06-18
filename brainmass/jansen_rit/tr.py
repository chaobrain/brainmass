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

from typing import Callable, Union, Optional, Tuple

import brainstate
import braintools
import brainunit as u
import jax.nn
import numpy as np
from brainstate import HiddenState
from brainstate.nn import (
    exp_euler_step, Param, Dynamics, Module, init_maybe_prefetch, Delay,
)

from ..coupling import additive_coupling, AdditiveConn, DelayedAdditiveConn
from ..noise import Noise, GaussianNoise
from ..typing import Parameter, Initializer
from ..utils import delay_index


Array = brainstate.typing.ArrayLike
Size = brainstate.typing.Size
Prefetch = Union[
    brainstate.nn.PrefetchDelayAt,
    brainstate.nn.PrefetchDelay,
    brainstate.nn.Prefetch,
    Callable,
]
from .step import Identity, JansenRitStep
from .connectivity import LaplacianConnectivity


class JansenRitTR(Dynamics):
    def __init__(
        self,
        in_size: Size,

        # distance parameters
        delay: Array,

        # structural connectivity parameters
        sc: Array,
        k: Parameter,
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
        state_init: Callable = braintools.init.Constant(0.0),
        delay_init: Callable = braintools.init.Constant(0.0),
        tr: u.Quantity = 1e-3 * u.second
    ):
        super().__init__(in_size)

        self.k = Param.init(k)
        self.tr = tr

        # single step dynamics
        self.step = JansenRitStep(
            in_size=in_size,
            Mv_init=lambda *args: state_init(*args) * u.mV / u.second,
            Ev_init=lambda *args: state_init(*args) * u.mV / u.second,
            Iv_init=lambda *args: state_init(*args) * u.mV / u.second,
            M_init=lambda *args: state_init(*args) * u.mV,
            E_init=lambda *args: state_init(*args) * u.mV,
            I_init=lambda *args: state_init(*args) * u.mV,
            noise_M=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
            noise_E=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
            noise_I=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
        )

        # delay
        n_hidden = self.varshape[0]
        dt = brainstate.environ.get_dt()
        self.delay = Delay(self.step.M_init(self.varshape), init=delay_init)
        self.delay_access = self.delay.access('delay', delay * dt, delay_index(n_hidden))

        # connectivity
        self.conn = LaplacianConnectivity(
            self.delay_access,
            self.step.prefetch('M'),
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

    def update(
        self,
        input: Array,
        record_state: bool = False,
        iter_input: bool = False
    ):
        n_step = int(self.tr / brainstate.environ.get_dt())
        if iter_input:
            assert input.shape[0] == n_step, f'Input length {input.shape[0]} does not match number of steps {n_step}'

        # Per-TR terms (constant across the sub-step loop) are computed *before*
        # the closure so they are bound when ``step`` runs, rather than relying on
        # Python late-binding of names assigned after the closure was defined.
        input = self.k.value() * input
        inp_M_tr, inp_E_tr, inp_I_tr = self.conn.update_tr()

        def step(inp):
            ext = inp if iter_input else input
            inp_M_step, inp_E_step, inp_I_step = self.conn.update()
            inp_M = (inp_M_tr + inp_M_step + ext) * u.mV
            inp_E = (inp_E_tr + inp_E_step) * u.Hz
            inp_I = (inp_I_tr + inp_I_step) * u.mV
            self.step.update(inp_M, inp_E, inp_I)

        brainstate.transform.for_loop(step, input if iter_input else np.arange(n_step))
        self.delay.update(self.step.M.value)
        activity = self.step.E.value - self.step.I.value
        activity = u.get_magnitude(activity)

        if record_state:
            state = dict(M=self.step.M.value, E=self.step.E.value, I=self.step.I.value)
            return activity, state
        return activity
