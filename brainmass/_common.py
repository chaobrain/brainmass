# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Callable

import brainstate
import braintools
import brainunit as u

from .noise import Noise

__all__ = [
    'XY_Oscillator',
]


class XY_Oscillator(brainstate.nn.Dynamics):
    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # noise parameters
        noise_x: Noise = None,
        noise_y: Noise = None,

        # other parameters
        init_x: Callable = braintools.init.Uniform(0, 0.05),
        init_y: Callable = braintools.init.Uniform(0, 0.05),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        # initializers
        assert isinstance(noise_x, Noise) or noise_x is None, "noise_x must be a Noise instance or None."
        assert isinstance(noise_y, Noise) or noise_y is None, "noise_y must be a Noise instance or None."
        assert callable(init_x), "init_x must be a callable."
        assert callable(init_y), "init_y must be a callable."
        self.init_x = init_x
        self.init_y = init_y
        self.noise_x = noise_x
        self.noise_y = noise_y
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Initialize model states ``x`` and ``y``.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.x = brainstate.HiddenState(braintools.init.param(self.init_x, self.varshape, batch_size))
        self.y = brainstate.HiddenState(braintools.init.param(self.init_y, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        """Reset model states ``x`` and ``y`` using the initializers.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional batch dimension for reinitialization. If ``None``, keeps
            current batch shape but resets values. Default is ``None``.
        """
        self.x.value = braintools.init.param(self.init_x, self.varshape, batch_size)
        self.y.value = braintools.init.param(self.init_y, self.varshape, batch_size)

    def dx(self, x, y, x_ext):
        raise NotImplementedError

    def dy(self, y, x, y_ext):
        raise NotImplementedError

    def derivative(self, state, t, x_ext, y_ext):
        x, y = state
        dxdt = self.dx(x, y, x_ext)
        dydt = self.dy(y, x, y_ext)
        return dxdt, dydt

    def update(self, x_inp=None, y_inp=None):
        x_inp = 0. if x_inp is None else x_inp
        y_inp = 0. if y_inp is None else y_inp
        if self.noise_x is not None:
            x_inp = x_inp + self.noise_x()
        if self.noise_y is not None:
            y_inp = y_inp + self.noise_y()
        if self.method == 'exp_euler':
            x = brainstate.nn.exp_euler_step(self.dx, self.x.value, self.y.value, x_inp)
            y = brainstate.nn.exp_euler_step(self.dy, self.y.value, self.x.value, y_inp)
        else:
            method = getattr(braintools.quad, f'ode_{self.method}_step')
            x, y = method(self.derivative, (self.x.value, self.y.value), 0 * u.ms, x_inp, y_inp)
        self.x.value = x
        self.y.value = y
        return x
