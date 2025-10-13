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
import jax.numpy as jnp

from ._typing import Initializer
from .noise import Noise

__all__ = [
    'WilsonCowanModel',
]


class WilsonCowanModel(brainstate.nn.Dynamics):
    r"""Wilson–Cowan neural mass model.

    The model captures the interaction between an excitatory (E) and an
    inhibitory (I) neural population. It is widely used to study neural
    oscillations, multistability, and other emergent dynamics in cortical
    circuits.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau_E : Initializer, optional
        Excitatory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_E : Initializer, optional
        Excitatory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.2``.
    theta_E : Initializer, optional
        Excitatory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.8``.
    tau_I : Initializer, optional
        Inhibitory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_I : Initializer, optional
        Inhibitory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.``.
    theta_I : Initializer, optional
        Inhibitory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.0``.
    wEE : Initializer, optional
        E→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``12.``.
    wIE : Initializer, optional
        E→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.``.
    wEI : Initializer, optional
        I→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``13.``.
    wII : Initializer, optional
        I→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``11.``.
    r : Initializer, optional
        Refractory parameter (dimensionless) that limits maximum activation.
        Broadcastable to ``in_size``. Default is ``1.``.
    noise_E : Noise or None, optional
        Additive noise process for the excitatory population. If provided, its
        output is added to ``rE_inp`` at each update. Default is ``None``.
    noise_I : Noise or None, optional
        Additive noise process for the inhibitory population. If provided, its
        output is added to ``rI_inp`` at each update. Default is ``None``.
    rE_init : Callable, optional
        Initializer for the excitatory state ``rE``. Default is
        ``braintools.init.ZeroInit()``.
    rI_init : Callable, optional
        Initializer for the inhibitory state ``rI``. Default is
        ``braintools.init.ZeroInit()``.
    method: str
        The numerical integration method to use. One of ``'exp_euler'``,
        ``'euler'``, ``'rk2'``, or ``'rk4'``, that is implemented in
        ``braintools.quad``. Default is ``'exp_euler'``.


    Attributes
    ----------
    rE : brainstate.HiddenState
        Excitatory population activity (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.
    rI : brainstate.HiddenState
        Inhibitory population activity (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.

    Notes
    -----
    The continuous-time Wilson–Cowan equations are

    .. math::

        \tau_E \frac{dr_E}{dt} = -r_E(t) + \bigl[1 - r\, r_E(t)\bigr]
        F_E\bigl(w_{EE} r_E(t) - w_{EI} r_I(t) + I_E(t)\bigr),

    .. math::

        \tau_I \frac{dr_I}{dt} = -r_I(t) + \bigl[1 - r\, r_I(t)\bigr]
        F_I\bigl(w_{IE} r_E(t) - w_{II} r_I(t) + I_I(t)\bigr),

    with the sigmoidal transfer function

    .. math::

        F_j(x) = \frac{1}{1 + e^{-a_j (x - \theta_j)}} - \frac{1}{1 + e^{a_j \theta_j}},\quad j \in \{E, I\}.

    References
    ----------
    Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions
    in localized populations of model neurons. Biophysical Journal, 12, 1–24.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # Excitatory parameters
        tau_E: Initializer = 1. * u.ms,  # excitatory time constant (ms)
        a_E: Initializer = 1.2,  # excitatory gain (dimensionless)
        theta_E: Initializer = 2.8,  # excitatory firing threshold (dimensionless)

        # Inhibitory parameters
        tau_I: Initializer = 1. * u.ms,  # inhibitory time constant (ms)
        a_I: Initializer = 1.,  # inhibitory gain (dimensionless)
        theta_I: Initializer = 4.0,  # inhibitory firing threshold (dimensionless)

        # Connection parameters
        wEE: Initializer = 12.,  # local E-E coupling (dimensionless)
        wIE: Initializer = 4.,  # local E-I coupling (dimensionless)
        wEI: Initializer = 13.,  # local I-E coupling (dimensionless)
        wII: Initializer = 11.,  # local I-I coupling (dimensionless)

        # Refractory parameter
        r: Initializer = 1.,  # refractory parameter (dimensionless)

        # noise
        noise_E: Noise = None,  # excitatory noise process
        noise_I: Noise = None,  # inhibitory noise process

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size=in_size)

        self.a_E = braintools.init.param(a_E, self.varshape)
        self.a_I = braintools.init.param(a_I, self.varshape)
        self.tau_E = braintools.init.param(tau_E, self.varshape)
        self.tau_I = braintools.init.param(tau_I, self.varshape)
        self.theta_E = braintools.init.param(theta_E, self.varshape)
        self.theta_I = braintools.init.param(theta_I, self.varshape)
        self.wEE = braintools.init.param(wEE, self.varshape)
        self.wIE = braintools.init.param(wIE, self.varshape)
        self.wEI = braintools.init.param(wEI, self.varshape)
        self.wII = braintools.init.param(wII, self.varshape)
        self.r = braintools.init.param(r, self.varshape)
        self.noise_E = noise_E
        self.noise_I = noise_I
        assert isinstance(noise_I, Noise) or noise_I is None, "noise_I must be an OUProcess or None"
        assert isinstance(noise_E, Noise) or noise_E is None, "noise_E must be an OUProcess or None"
        self.rE_init = rE_init
        self.rI_init = rI_init
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.rE = brainstate.HiddenState(braintools.init.param(self.rE_init, self.varshape, batch_size))
        self.rI = brainstate.HiddenState(braintools.init.param(self.rI_init, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.rE.value = braintools.init.param(self.rE_init, self.varshape, batch_size)
        self.rI.value = braintools.init.param(self.rI_init, self.varshape, batch_size)

    def F(self, x, a, theta):
        """Sigmoidal transfer function.

        Parameters
        ----------
        x : array-like
            Input drive.
        a : array-like
            Gain (dimensionless), broadcastable to ``x``.
        theta : array-like
            Threshold (dimensionless), broadcastable to ``x``.

        Returns
        -------
        array-like
            Output in approximately ``[0, 1]`` (subject to numerical precision),
            with the same shape as ``x``.
        """
        return 1 / (1 + jnp.exp(-a * (x - theta))) - 1 / (1 + jnp.exp(a * theta))

    def drE(self, rE, rI, ext):
        """Right-hand side for the excitatory population.

        Parameters
        ----------
        rE : array-like
            Excitatory activity (dimensionless).
        rI : array-like
            Inhibitory activity (dimensionless), broadcastable to ``rE``.
        ext : array-like or scalar
            External input to E (same shape/unit as the model state input).

        Returns
        -------
        array-like
            Time derivative ``drE/dt`` with unit of ``1/time``.
        """
        xx = self.wEE * rE - self.wIE * rI + ext
        return (-rE + (1 - self.r * rE) * self.F(xx, self.a_E, self.theta_E)) / self.tau_E

    def drI(self, rI, rE, ext):
        """Right-hand side for the inhibitory population.

        Parameters
        ----------
        rI : array-like
            Inhibitory activity (dimensionless).
        rE : array-like
            Excitatory activity (dimensionless), broadcastable to ``rI``.
        ext : array-like or scalar
            External input to I (same shape/unit as the model state input).

        Returns
        -------
        array-like
            Time derivative ``drI/dt`` with unit of ``1/time``.
        """
        xx = self.wEI * rE - self.wII * rI + ext
        return (-rI + (1 - self.r * rI) * self.F(xx, self.a_I, self.theta_I)) / self.tau_I

    def derivaitive(self, state, t, E_exp, I_exp):
        rE, rI = state
        drE_dt = self.drE(rE, rI, E_exp)
        drI_dt = self.drI(rI, rE, I_exp)
        return (drE_dt, drI_dt)

    def update(self, rE_inp=None, rI_inp=None):
        """Advance the system by one time step.

        Parameters
        ----------
        rE_inp : array-like or scalar or None, optional
            External input to the excitatory population. If ``None``, treated
            as zero. If ``noise_E`` is set, its output is added.
        rI_inp : array-like or scalar or None, optional
            External input to the inhibitory population. If ``None``, treated
            as zero. If ``noise_I`` is set, its output is added.

        Returns
        -------
        array-like
            The updated excitatory activity ``rE`` with the same shape as the
            internal state.

        Notes
        -----
        The method performs an exponential-Euler step using
        ``brainstate.nn.exp_euler_step`` for both populations and updates the
        internal states ``rE`` and ``rI`` in-place.
        """
        # excitatory input
        rE_inp = 0. if rE_inp is None else rE_inp
        rI_inp = 0. if rI_inp is None else rI_inp
        if self.noise_E is not None:
            rE_inp = rE_inp + self.noise_E()

        # inhibitory input
        if self.noise_I is not None:
            rI_inp = rI_inp + self.noise_I()

        # update the state variables
        if self.method == 'exp_euler':
            rE = brainstate.nn.exp_euler_step(self.drE, self.rE.value, self.rI.value, rE_inp)
            rI = brainstate.nn.exp_euler_step(self.drI, self.rI.value, self.rE.value, rI_inp)
        else:
            rE, rI = getattr(braintools.quad, f'ode_{self.method}_step')(
                (self.rE.value, self.rI.value),
                0. * u.ms,
                rE_inp,
                rI_inp,
            )
        self.rE.value = rE
        self.rI.value = rI
        return rE
