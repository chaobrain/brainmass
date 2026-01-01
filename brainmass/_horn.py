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

import math
from typing import Callable, Sequence, Optional

import brainunit as u
import jax.numpy as jnp

import brainstate
import braintools.init
from brainstate import nn
from ._typing import Initializer, Parameter

__all__ = [
    'HORNStep',
    'HORNSeqLayer',
    'HORNSeqNetwork',
]


def zeros(x):
    return 0.


class HORNStep(nn.Dynamics):
    r"""Harmonic oscillator recurrent networks (HORNs) with one-step dynmaics update.

    The update equations for a HORN network of n units in
    discrete time t result from the discretization of a second-order ODE describing a
    driven damped harmonic oscillator

    $$
    \ddot{x}(t)+2\gamma\dot{x}(t)+\omega^2x(t)=\alpha\sigma\left(I\left(t\right)+F\left(x\left(t\right),\dot{x}\left(t\right)\right)\right),
    $$

    The discrete dynamics is given by:

    $$
    \begin{aligned}
    &\mathbf{y}_{t+1}=\mathbf{y}_{t}+ h * (\boldsymbol{\alpha}\cdot\tanh\left(1/\sqrt{n}\mathbf{I}_{t+1}^{\mathrm{rec}}+\mathbf{I}_{t+1}^{\mathrm{ext}}\right)-2\boldsymbol{\gamma}\cdot\mathbf{y}_{t}-\boldsymbol{\omega}^{2}\cdot\mathbf{x}_{t}),
    \\
    &\mathbf{x}_{t+1}=\mathbf{x}_{t}+\mathbf{y}_{t+1},
    \end{aligned}
    $$

    where vectors and matrices are indicated by boldface symbols, and
    $\boldsymbol{\omega},\boldsymbol{\gamma},\boldsymbol{\alpha}$ are the natural frequencies,
    damping factors and excitability factors of the network nodes, respectively.
    Initial conditions are $\mathbf{x}_0=\mathbf{y}_0=0$ unless stated otherwise.

    $I_t+ 1^\mathrm{rec}= \mathbf{W} ^{hh}\mathbf{y} _t+ \mathbf{b} ^{hh}+ \mathbf{v} \cdot \mathbf{x}_t$
    and $I_t+1^\mathrm{ext}=\mathbf{W}^{ih}\mathbf{s}_{t+1}+\mathbf{b}^{ih}$ denote the recurrent and
    external input to each node, respectively.

    Here, the diagona entries of $\mathbf{W}^{hh}$ and v denote feedback parameters;
    $\mathbf{W}^{ih},\mathbf{b}^{ih},\mathbf{w}^{ih}$ and $\mathbf{W}^{hh},\mathbf{b}^{hh}$ denote the input and
    hidden weights and biases, respectively, and $S_{\mathrm{ext}}=(s_1,\ldots,s_T)$ the external input.

    """

    def __init__(
        self,
        in_size: int,
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # Amplitude feedback
        recurrent_fn: Callable = zeros,  # Velocity feedback
        state_init: Initializer = braintools.init.ZeroInit(),
    ):
        super().__init__(in_size)

        self.alpha = nn.Param.init(alpha, self.in_size)
        self.omega = nn.Param.init(omega, self.in_size)
        self.gamma = nn.Param.init(gamma, self.in_size)
        self.v = nn.Param.init(v, self.in_size)
        self.gain_rec = 1. / math.sqrt(self.in_size[0])
        self.state_init = state_init
        self.recurrent_fn = recurrent_fn

    def init_state(self, *args, **kwargs):
        self.x = brainstate.HiddenState(self.state_init(self.in_size))
        self.y = brainstate.HiddenState(self.state_init(self.in_size))

    def update(self, inputs):
        # one discrete dynamics step based on sympletic Euler integration

        # time step
        h = u.get_magnitude(brainstate.environ.get_dt())
        # current states
        y = self.y.value
        x = self.x.value

        # omega^2 for DHO equation
        omega_factor = self.omega.value() ** 2

        # 2 * gamma for DHO equation
        gamma_factor = 2.0 * self.gamma.value()
        v = self.v.value()
        alpha = self.alpha.value()

        # 1. integrate y_t
        # external input + recurrent input from network
        inp = (inputs + self.gain_rec * (self.recurrent_fn(y) + v * x))
        y_t = y + h * (
            alpha * jnp.tanh(inp)  # input (forcing) on y_t
            - omega_factor * x  # natural frequency term
            - gamma_factor * y  # damping term
        )

        # 2. integrate x_t with updated y_t, no input here
        x_t = x + h * y_t

        self.x.value = x_t
        self.y.value = y_t
        return x_t


class HORNSeqLayer(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # feedback
        state_init: Callable = braintools.init.ZeroInit(),
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
        inp_w_init: Initializer = braintools.init.KaimingNormal(),
        inp_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
    ):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.rec_w_init = rec_w_init
        self.rec_b_init = rec_b_init
        self.inp_w_init = inp_w_init
        self.inp_b_init = inp_b_init

        self.i2h = brainstate.nn.Linear(n_input, n_hidden, w_init=inp_w_init, b_init=inp_b_init)
        self.h2h = brainstate.nn.Linear(n_hidden, n_hidden, w_init=rec_w_init, b_init=rec_b_init)
        self.horn = HORNStep(
            n_hidden,
            alpha=alpha, omega=omega, gamma=gamma, v=v,
            state_init=state_init, recurrent_fn=self.h2h
        )

    def update(self, inputs, record_state: bool = False):
        def step(inp):
            out = self.horn(inp)
            st = dict(x=self.horn.x.value, y=self.horn.y.value)
            return (st, out) if record_state else out

        output = brainstate.transform.for_loop(step, self.i2h(inputs))
        return output


class HORNSeqNetwork(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int | Sequence[int],
        n_output: int,
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # feedback
        state_init: Callable = braintools.init.ZeroInit(),
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
        inp_w_init: Initializer = braintools.init.KaimingNormal(),
        inp_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
    ):
        super().__init__()

        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        assert isinstance(n_hidden, (list, tuple)), 'n_hidden must be int or sequence of int.'

        self.layers = []
        for hidden in n_hidden:
            layer = HORNSeqLayer(
                n_input=n_input,
                n_hidden=hidden,
                alpha=alpha,
                omega=omega,
                gamma=gamma,
                v=v,
                state_init=state_init,
                rec_w_init=rec_w_init,
                rec_b_init=rec_b_init,
                inp_w_init=inp_w_init,
                inp_b_init=inp_b_init,
            )
            self.layers.append(layer)
            n_input = hidden  # next layer input size is current layer hidden size

        self.h2o = brainstate.nn.Linear(n_input, n_output, w_init=inp_w_init, b_init=inp_b_init)

    def update(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        output = self.h2o(self.layers[-1].horn.x.value)
        return output
