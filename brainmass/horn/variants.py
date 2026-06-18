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

import brainstate
import braintools.init
import brainunit as u
import jax.numpy as jnp
import numpy as np
from brainstate.nn import Param, Module, Dynamics

from ..coupling import (
    AdditiveCoupling, additive_coupling, LaplacianConnParam, AdditiveConn, DelayedAdditiveConn
)
from ..typing import Initializer, Parameter
from .step import HORNStep
from .connections import (
    DelayedAdditiveConnTR, DelayedLaplacianConn, DelayedLaplacianConnTR,
)


class HORN_TR(Module):
    def __init__(
        self,
        n_hidden: int,
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # feedback

        # state initialization
        x_init: Initializer = braintools.init.Constant(0.0),
        y_init: Initializer = braintools.init.Constant(0.0),

        # time resolution
        tr: u.Quantity = 1. * u.ms,

        # recurrent connections
        delay: Optional[Initializer] = None,
        rec_type: str = 'additive',
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.Constant(0.0),
        delay_init: Callable = braintools.init.Constant(0.0),
    ):
        super().__init__()

        self.n_hidden = n_hidden

        # dynamics
        self.horn = HORNStep(n_hidden, alpha=alpha, omega=omega, gamma=gamma, v=v, x_init=x_init, y_init=y_init)

        # hidden-to-hidden
        if delay is None:
            self.h2h = AdditiveConn(self.horn, w_init=rec_w_init, b_init=rec_b_init)
        elif rec_type == 'additive':
            self.h2h = DelayedAdditiveConn(self.horn, delay, w_init=rec_w_init, delay_init=delay_init)
        elif rec_type == 'additive_tr':
            self.h2h = DelayedAdditiveConnTR(self.horn, delay, w_init=rec_w_init, delay_init=delay_init, tr=tr)
        elif rec_type == 'laplacian':
            self.h2h = DelayedLaplacianConn(self.horn, delay, w_init=rec_w_init, delay_init=delay_init)
        elif rec_type == 'laplacian_tr':
            self.h2h = DelayedLaplacianConnTR(self.horn, delay, w_init=rec_w_init, delay_init=delay_init, tr=tr)
        else:
            raise ValueError(f'Unknown delay_type: {rec_type}')

    def update(self, inputs, record_state: bool = False):
        inpt_tr = self.h2h.update_tr()

        def step(inp):
            inp_step = self.h2h.update()
            out = self.horn(inp + inp_step + inpt_tr)
            st = dict(x=self.horn.x.value, y=self.horn.y.value)
            return (st, out) if record_state else out

        return brainstate.transform.for_loop(step, inputs)
