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
from .step import JansenRitStep
from .connectivity import LaplacianConnV2


class JansenRitLayer(Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        Ae: Parameter = 3.25 * u.mV,  # Excitatory gain
        Ai: Parameter = 22. * u.mV,  # Inhibitory gain
        be: Parameter = 100. * u.Hz,  # Excit. time const
        bi: Parameter = 50. * u.Hz,  # Inhib. time const.
        C: Parameter = 135.,  # Connect. const.
        a1: Parameter = 1.,  # Connect. param.
        a2: Parameter = 0.8,  # Connect. param.
        a3: Parameter = 0.25,  # Connect. param
        a4: Parameter = 0.25,  # Connect. param.
        s_max: Parameter = 5.0 * u.Hz,  # Max firing rate
        v0: Parameter = 6. * u.mV,  # Firing threshold
        r: Parameter = 0.56,  # Sigmoid steepness
        # initialization
        M_init: Callable = braintools.init.Constant(0.0 * u.mV),
        E_init: Callable = braintools.init.Constant(0.0 * u.mV),
        I_init: Callable = braintools.init.Constant(0.0 * u.mV),
        Mv_init: Callable = braintools.init.Constant(0.0 * u.mV / u.second),
        Ev_init: Callable = braintools.init.Constant(0.0 * u.mV / u.second),
        Iv_init: Callable = braintools.init.Constant(0.0 * u.mV / u.second),
        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,
        noise_M: Noise = None,
        # distance parameters
        delay: Array = None,
        delay_init: Callable = braintools.init.Constant(0.0),
        # initialization
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.Constant(0.0),
        inp_w_init: Initializer = braintools.init.KaimingNormal(),
        inp_b_init: Optional[Initializer] = braintools.init.Constant(0.0),
        # other parameters
        method: str = 'exp_euler',
    ):
        super().__init__()

        self.dynamics = JansenRitStep(
            n_hidden,
            Ae=Ae,
            Ai=Ai,
            be=be,
            bi=bi,
            C=C,
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
            s_max=s_max,
            v0=v0,
            r=r,
            Mv_init=Mv_init,
            Ev_init=Ev_init,
            Iv_init=Iv_init,
            M_init=M_init,
            E_init=E_init,
            I_init=I_init,
            noise_M=noise_M,
            noise_E=noise_E,
            noise_I=noise_I,
            method=method,
        )
        self.i2h = brainstate.nn.Linear(n_input, n_hidden, w_init=inp_w_init, b_init=inp_b_init)
        if delay is None:
            self.h2h = AdditiveConn(self.dynamics, state='M', w_init=rec_w_init, b_init=rec_b_init)
        else:
            self.h2h = DelayedAdditiveConn(self.dynamics, delay, state='M', delay_init=delay_init, w_init=rec_w_init)

    def update(self, inputs, record_state: bool = False):
        def step(inp):
            rec = self.h2h()
            out = self.dynamics(E_inp=(inp + rec) * u.Hz)
            st = dict(
                M=self.dynamics.M.value,
                E=self.dynamics.E.value,
                I=self.dynamics.I.value,
            )
            return (st, out) if record_state else out

        assert inputs.ndim == 2, 'Inputs must be 2D (time, features)'
        output = brainstate.transform.for_loop(step, self.i2h(inputs))
        return output



class JansenRit2Layer(Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        Ae: Parameter = 3.25 * u.mV,  # Excitatory gain
        Ai: Parameter = 22. * u.mV,  # Inhibitory gain
        be: Parameter = 100. * u.Hz,  # Excit. time const
        bi: Parameter = 50. * u.Hz,  # Inhib. time const.
        C: Parameter = 135.,  # Connect. const.
        a1: Parameter = 1.,  # Connect. param.
        a2: Parameter = 0.8,  # Connect. param.
        a3: Parameter = 0.25,  # Connect. param
        a4: Parameter = 0.25,  # Connect. param.
        s_max: Parameter = 5.0 * u.Hz,  # Max firing rate
        v0: Parameter = 6. * u.mV,  # Firing threshold
        r: Parameter = 0.56,  # Sigmoid steepness
        # initialization
        M_init: Callable = braintools.init.Constant(0.0 * u.mV),
        E_init: Callable = braintools.init.Constant(0.0 * u.mV),
        I_init: Callable = braintools.init.Constant(0.0 * u.mV),
        Mv_init: Callable = braintools.init.Constant(0.0 * u.mV / u.second),
        Ev_init: Callable = braintools.init.Constant(0.0 * u.mV / u.second),
        Iv_init: Callable = braintools.init.Constant(0.0 * u.mV / u.second),
        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,
        noise_M: Noise = None,
        # structural parameters
        delay: Array = None,
        delay_init: Callable = braintools.init.Constant(0.0),
        # initialization
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.Constant(0.0),
        inp_w_init: Initializer = braintools.init.KaimingNormal(),
        inp_b_init: Optional[Initializer] = braintools.init.Constant(0.0),
        # other parameters
        method: str = 'exp_euler',
    ):
        super().__init__()

        self.dynamics = JansenRitStep(
            n_hidden,
            Ae=Ae,
            Ai=Ai,
            be=be,
            bi=bi,
            C=C,
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
            s_max=s_max,
            v0=v0,
            r=r,
            Mv_init=Mv_init,
            Ev_init=Ev_init,
            Iv_init=Iv_init,
            M_init=M_init,
            E_init=E_init,
            I_init=I_init,
            noise_M=noise_M,
            noise_E=noise_E,
            noise_I=noise_I,
            method=method,
        )
        self.i2h = brainstate.nn.Linear(n_input, n_hidden, w_init=inp_w_init, b_init=inp_b_init)
        assert delay is not None, 'Delay must be provided for JansenRit2Layer'
        self.conn = LaplacianConnV2(
            self.dynamics,
            delay,
            delay_init=delay_init,
            weight=rec_w_init,
        )

    def update(self, inputs, record_state: bool = False):
        def step(inp):
            inp_M, inp_E, inp_I = self.conn()
            out = self.dynamics(
                M_inp=(inp + inp_M) * u.mV,
                E_inp=inp_E * u.Hz,
                I_inp=inp_I * u.mV,
            )
            st = dict(
                M=self.dynamics.M.value,
                E=self.dynamics.E.value,
                I=self.dynamics.I.value,
            )
            return (st, out) if record_state else out

        assert inputs.ndim == 2, 'Inputs must be 2D (time, features)'
        output = brainstate.transform.for_loop(step, self.i2h(inputs))
        return output



class JansenRitNetwork(Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: Union[int, Tuple[int]],
        n_output: int,
        Ae: Parameter = 3.25 * u.mV,  # Excitatory gain
        Ai: Parameter = 22. * u.mV,  # Inhibitory gain
        be: Parameter = 100. * u.Hz,  # Excit. time const
        bi: Parameter = 50. * u.Hz,  # Inhib. time const.
        C: Parameter = 135.,  # Connect. const.
        a1: Parameter = 1.,  # Connect. param.
        a2: Parameter = 0.8,  # Connect. param.
        a3: Parameter = 0.25,  # Connect. param
        a4: Parameter = 0.25,  # Connect. param.
        s_max: Parameter = 5.0 * u.Hz,  # Max firing rate
        v0: Parameter = 6. * u.mV,  # Firing threshold
        r: Parameter = 0.56,  # Sigmoid steepness
        # initialization
        M_init: Callable = braintools.init.Constant(0.0 * u.mV),
        E_init: Callable = braintools.init.Constant(0.0 * u.mV),
        I_init: Callable = braintools.init.Constant(0.0 * u.mV),
        Mv_init: Callable = braintools.init.Constant(0.0 * u.mV / u.second),
        Ev_init: Callable = braintools.init.Constant(0.0 * u.mV / u.second),
        Iv_init: Callable = braintools.init.Constant(0.0 * u.mV / u.second),
        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,
        noise_M: Noise = None,
        # distance parameters
        delay: Array = None,
        delay_init: Callable = braintools.init.Constant(0.0),
        # initialization
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.Constant(0.0),
        inp_w_init: Initializer = braintools.init.KaimingNormal(),
        inp_b_init: Optional[Initializer] = braintools.init.Constant(0.0),
        # other parameters
        method: str = 'exp_euler',
    ):
        super().__init__()

        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        assert isinstance(n_hidden, (list, tuple)), 'n_hidden must be int or sequence of int.'

        self.layers = []
        for hidden in n_hidden:
            layer = JansenRit2Layer(
                n_input,
                hidden,
                Ae=Ae,
                Ai=Ai,
                be=be,
                bi=bi,
                C=C,
                a1=a1,
                a2=a2,
                a3=a3,
                a4=a4,
                s_max=s_max,
                v0=v0,
                r=r,
                Mv_init=Mv_init,
                Ev_init=Ev_init,
                Iv_init=Iv_init,
                M_init=M_init,
                E_init=E_init,
                I_init=I_init,
                noise_M=noise_M,
                noise_E=noise_E,
                noise_I=noise_I,
                method=method,
                delay=delay,
                delay_init=delay_init,
                rec_w_init=rec_w_init,
                rec_b_init=rec_b_init,
                inp_w_init=inp_w_init,
                inp_b_init=inp_b_init,
            )
            self.layers.append(layer)
            n_input = hidden  # next layer input size is current layer hidden size
        self.h2o = brainstate.nn.Linear(n_input, n_output, w_init=inp_w_init, b_init=inp_b_init)

    def update(self, inputs, record_state: bool = False):
        x = inputs
        for layer in self.layers:
            # Each layer emits an mV ``eeg()`` proxy, but the next layer's input
            # path projects with a ``Linear`` and re-applies ``* u.mV`` (it expects
            # a dimensionless feature sequence). Strip units between layers so the
            # mV output of one layer becomes a valid dimensionless input to the
            # next. (For a single-layer network this is a no-op: the readout below
            # reads the last layer's dynamics state directly, not ``x``.)
            x = u.get_magnitude(layer(x))
        eeg = u.get_magnitude(self.layers[-1].dynamics.eeg())
        output = self.h2o(eeg)
        return output
