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

import braintools
import brainstate
import brainunit as u
import jax.numpy as jnp

from ._typing import Initializer

__all__ = [
    'Noise',
    'OUProcess',
    'GaussianNoise',
    'WhiteNoise',
    'ColoredNoise',
    'BrownianNoise',
    'PinkNoise',
    'BlueNoise',
    'VioletNoise',
]


class Noise(brainstate.nn.Dynamics):
    pass


class GaussianNoise(Noise):
    """Gaussian (white) noise process without state (i.i.d. across time)."""

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        mean: Initializer = None,
        sigma: Initializer = 1. * u.nA,
    ):
        super().__init__(in_size=in_size)

        self.sigma = sigma
        self.mean = 0. * u.get_unit(sigma) if mean is None else mean

    def update(self):
        z = brainstate.random.normal(loc=0.0, scale=1.0, size=self.varshape)
        return self.mean + self.sigma * z


class WhiteNoise(GaussianNoise):
    """Alias of GaussianNoise for semantic clarity."""


class BrownianNoise(Noise):
    """
    Brownian (red) noise: discrete-time integral of white noise.

    x[t+dt] = x[t] + sigma * sqrt(dt) * N(0, 1)
    output = mean + x
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        mean: Initializer = None,
        sigma: Initializer = 1. * u.nA,
        init: Callable = braintools.init.ZeroInit(unit=u.nA)
    ):
        super().__init__(in_size=in_size)

        self.sigma = sigma
        self.mean = 0. * u.get_unit(sigma) if mean is None else mean
        self.init = init

    def init_state(self, batch_size=None, **kwargs):
        self.x = brainstate.HiddenState(braintools.init.param(self.init, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.x.value = braintools.init.param(self.init, self.varshape, batch_size)

    def update(self):
        noise = brainstate.random.randn(*self.varshape)
        dt_sqrt = u.math.sqrt(brainstate.environ.get_dt())
        self.x.value = self.x.value + self.sigma / dt_sqrt * dt_sqrt * noise
        return self.mean + self.x.value


class ColoredNoise(Noise):
    """
    Colored noise with PSD ~ 1/f^beta generated via frequency-domain shaping.

    Note: Each update call synthesizes a fresh sample over the last axis using
    FFT shaping; there is no temporal state carried across updates.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        beta: float = 1.0,
        mean: Initializer = None,
        sigma: Initializer = 1. * u.nA,
    ):
        super().__init__(in_size=in_size)

        self.beta = beta
        self.sigma = sigma
        self.mean = 0. * u.get_unit(sigma) if mean is None else mean

    def update(self):
        size = self.varshape
        if len(size) == 0:
            # scalar: fallback to Gaussian
            z = brainstate.random.normal(loc=0.0, scale=1.0, size=())
            return self.mean + self.sigma * z

        n = size[-1]
        if n < 2:
            # not enough points to shape spectrum; fallback to Gaussian
            z = brainstate.random.normal(loc=0.0, scale=1.0, size=size)
            return self.mean + self.sigma * z

        # white noise
        x = brainstate.random.normal(loc=0.0, scale=1.0, size=size)
        xr = jnp.asarray(x)
        Xf = jnp.fft.rfft(xr, axis=-1)
        freqs = jnp.fft.rfftfreq(n)
        w = jnp.where(freqs > 0, freqs ** (-self.beta / 2.0), 0.0)
        shape_ones = (1,) * (Xf.ndim - 1) + (w.shape[0],)
        w = w.reshape(shape_ones)
        Yf = Xf * w
        y = jnp.fft.irfft(Yf, n=n, axis=-1)

        # normalize std over last axis and scale
        std = jnp.std(y, axis=-1, keepdims=True)
        y = jnp.where(std > 0, y / std, y)
        return self.mean + self.sigma * y


class PinkNoise(ColoredNoise):
    """
    Pink (1/f) noise.
    """

    def __init__(self, in_size, mean=None, sigma=1. * u.nA):
        super().__init__(in_size=in_size, beta=1.0, mean=mean, sigma=sigma)


class BlueNoise(ColoredNoise):
    """
    Blue (1/f^2) noise.
    """

    def __init__(self, in_size, mean=None, sigma=1. * u.nA):
        super().__init__(in_size=in_size, beta=-1.0, mean=mean, sigma=sigma)


class VioletNoise(ColoredNoise):
    """
    Violet (1/f^3) noise.
    """

    def __init__(self, in_size, mean=None, sigma=1. * u.nA):
        super().__init__(in_size=in_size, beta=-2.0, mean=mean, sigma=sigma)


class OUProcess(Noise):
    r"""
    The Ornstein–Uhlenbeck process.

    The Ornstein–Uhlenbeck process :math:`x_{t}` is defined by the following
    stochastic differential equation:

    .. math::

       \tau dx_{t}=-\theta \,x_{t}\,dt+\sigma \,dW_{t}

    where :math:`\theta >0` and :math:`\sigma >0` are parameters and :math:`W_{t}`
    denotes the Wiener process.

    Parameters
    ==========
    in_size: int, sequence of int
      The model size.
    mean: ArrayLike
      The noise mean value.  Default is 0 nA.
    sigma: ArrayLike
      The noise amplitude. Defualt is 1 nA.
    tau: ArrayLike
      The decay time constant. The larger the value, the slower the decay. Default is 10 ms.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        mean: Initializer = None,  # noise mean value
        sigma: Initializer = 1. * u.nA,  # noise amplitude
        tau: Initializer = 10. * u.ms,  # time constant
        init: Callable = None
    ):
        super().__init__(in_size=in_size)

        # parameters
        self.sigma = sigma
        self.mean = 0. * u.get_unit(sigma) if mean is None else mean
        self.tau = tau
        self.init = braintools.init.ZeroInit(unit=u.get_unit(sigma)) if init is None else init

    def init_state(self, batch_size=None, **kwargs):
        self.x = brainstate.HiddenState(braintools.init.param(self.init, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.x.value = braintools.init.param(self.init, self.varshape, batch_size)

    def update(self):
        df = lambda x: (self.mean - x) / self.tau
        dg = lambda x: self.sigma / u.math.sqrt(self.tau)
        self.x.value = brainstate.nn.exp_euler_step(df, dg, self.x.value)
        return self.x.value
