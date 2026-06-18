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


class DelayedAdditiveConnTR(Module):
    def __init__(
        self,
        model: Dynamics,
        delay_time: Initializer,
        delay_init: Initializer = braintools.init.ZeroInit(),
        w_init: Callable = braintools.init.KaimingNormal(),
        k: Parameter = 1.0,
        tr: u.Quantity = 1. * u.ms,
    ):
        super().__init__()

        n_hidden = model.varshape[0]
        delay_time = braintools.init.param(delay_time, (n_hidden, n_hidden))
        neuron_idx = np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
        self.prefetch = model.prefetch_delay('y', delay_time, neuron_idx, init=delay_init, update_every=tr)
        self.weights = Param(braintools.init.param(w_init, (n_hidden, n_hidden)))
        self.k = Param.init(k)

    def update_tr(self, *args, **kwargs):
        delayed = self.prefetch()
        return additive_coupling(delayed, self.weights.value(), self.k.value())

    def update(self, *args, **kwargs):
        return 0.



class DelayedLaplacianConn(Module):
    def __init__(
        self,
        model: Dynamics,
        delay_time: Initializer,
        delay_init: Initializer = braintools.init.ZeroInit(),
        w_init: Callable = braintools.init.KaimingNormal(),
        k: Parameter = 1.0,
    ):
        super().__init__()

        n_hidden = model.varshape[0]
        delay_time = braintools.init.param(delay_time, (n_hidden, n_hidden))
        neuron_idx = np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
        self.prefetch = model.prefetch_delay('y', delay_time, neuron_idx, init=delay_init)
        self.weights = LaplacianConnParam(braintools.init.param(w_init, (n_hidden, n_hidden)))
        self.k = Param.init(k)

    def update_tr(self, *args, **kwargs):
        return 0.

    def update(self, *args, **kwargs):
        delayed = self.prefetch()
        return additive_coupling(delayed, self.weights.value(), self.k.value())



class DelayedLaplacianConnTR(Module):
    def __init__(
        self,
        model: Dynamics,
        delay_time: Initializer,
        delay_init: Initializer = braintools.init.ZeroInit(),
        w_init: Callable = braintools.init.KaimingNormal(),
        k: Parameter = 1.0,
        tr: u.Quantity = 1. * u.ms,
    ):
        super().__init__()

        n_hidden = model.varshape[0]
        delay_time = braintools.init.param(delay_time, (n_hidden, n_hidden))
        neuron_idx = np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
        self.prefetch = model.prefetch_delay('y', delay_time, neuron_idx, init=delay_init, update_every=tr)
        self.weights = LaplacianConnParam(braintools.init.param(w_init, (n_hidden, n_hidden)))
        self.k = Param.init(k)

    def update_tr(self, *args, **kwargs):
        delayed = self.prefetch()
        return additive_coupling(delayed, self.weights.value(), self.k.value())

    def update(self, *args, **kwargs):
        return 0.
