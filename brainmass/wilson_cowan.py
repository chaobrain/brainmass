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

from typing import Callable

import brainstate
import braintools
import brainunit as u
import jax.nn
import jax.numpy as jnp
from brainstate.nn import Param

from .noise import Noise
from .typing import Parameter

__all__ = [
    'WilsonCowanStep',
    'WilsonCowanStepNoSaturation',
    'WilsonCowanStepSymmetric',
    'WilsonCowanStepSimplified',
    'WilsonCowanStepLinear',
]


class WilsonCowanStep(brainstate.nn.Dynamics):
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
    tau_E : Parameter , optional
        Excitatory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_E : Parameter , optional
        Excitatory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.2``.
    theta_E : Parameter , optional
        Excitatory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.8``.
    tau_I : Parameter , optional
        Inhibitory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_I : Parameter , optional
        Inhibitory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.``.
    theta_I : Parameter , optional
        Inhibitory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.0``.
    wEE : Parameter , optional
        E→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``12.``.
    wIE : Parameter , optional
        E→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.``.
    wEI : Parameter , optional
        I→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``13.``.
    wII : Parameter , optional
        I→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``11.``.
    r : Parameter , optional
        Refractory parameter (dimensionless) that limits maximum activation.
        Broadcastable to ``in_size``. Default is ``1.``.
    noise_E : Noise or None, optional
        Additive noise process for the excitatory population. If provided, its
        output is added to ``rE_inp`` at each update. Default is ``None``.
    noise_I : Noise or None, optional
        Additive noise process for the inhibitory population. If provided, its
        output is added to ``rI_inp`` at each update. Default is ``None``.
    rE_init : Callable, optional
        Parameter  for the excitatory state ``rE``. Default is
        ``braintools.init.ZeroInit()``.
    rI_init : Callable, optional
        Parameter  for the inhibitory state ``rI``. Default is
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
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # Excitatory parameters
        tau_E: Parameter = 1. * u.ms,  # excitatory time constant (ms)
        a_E: Parameter = 1.2,  # excitatory gain (dimensionless)
        theta_E: Parameter = 2.8,  # excitatory firing threshold (dimensionless)

        # Inhibitory parameters
        tau_I: Parameter = 1. * u.ms,  # inhibitory time constant (ms)
        a_I: Parameter = 1.,  # inhibitory gain (dimensionless)
        theta_I: Parameter = 4.0,  # inhibitory firing threshold (dimensionless)

        # Connection parameters
        wEE: Parameter = 12.,  # local E-E coupling (dimensionless)
        wIE: Parameter = 4.,  # local E-I coupling (dimensionless)
        wEI: Parameter = 13.,  # local I-E coupling (dimensionless)
        wII: Parameter = 11.,  # local I-I coupling (dimensionless)

        # Refractory parameter
        r: Parameter = 1.,  # refractory parameter (dimensionless)

        # noise
        noise_E: Noise = None,  # excitatory noise process
        noise_I: Noise = None,  # inhibitory noise process

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size=in_size)

        self.a_E = Param.init(a_E, self.varshape)
        self.a_I = Param.init(a_I, self.varshape)
        self.tau_E = Param.init(tau_E, self.varshape)
        self.tau_I = Param.init(tau_I, self.varshape)
        self.theta_E = Param.init(theta_E, self.varshape)
        self.theta_I = Param.init(theta_I, self.varshape)
        self.wEE = Param.init(wEE, self.varshape)
        self.wIE = Param.init(wIE, self.varshape)
        self.wEI = Param.init(wEI, self.varshape)
        self.wII = Param.init(wII, self.varshape)
        self.r = Param.init(r, self.varshape)
        self.noise_E = noise_E
        self.noise_I = noise_I
        assert isinstance(noise_I, Noise) or noise_I is None, "noise_I must be an OUProcess or None"
        assert isinstance(noise_E, Noise) or noise_E is None, "noise_E must be an OUProcess or None"
        self.rE_init = rE_init
        self.rI_init = rI_init
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.rE = brainstate.HiddenState.init(self.rE_init, self.varshape, batch_size)
        self.rI = brainstate.HiddenState.init(self.rI_init, self.varshape, batch_size)

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
        # 1 / (1 + jnp.exp(-a * (x - theta))) - 1 / (1 + jnp.exp(a * theta))
        return jax.nn.sigmoid(a * (x - theta)) - jax.nn.sigmoid(-a * theta)

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
        wEE = self.wEE.value()
        wIE = self.wIE.value()
        r = self.r.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()
        xx = wEE * rE - wIE * rI + ext
        return (-rE + (1 - r * rE) * self.F(xx, a_E, theta_E)) / tau_E

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
        wEI = self.wEI.value()
        wII = self.wII.value()
        r = self.r.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()
        xx = wEI * rE - wII * rI + ext
        return (-rI + (1 - r * rI) * self.F(xx, a_I, theta_I)) / tau_I

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
            t = brainstate.environ.get('t', 0. * u.ms)
            rE, rI = getattr(braintools.quad, f'ode_{self.method}_step')(
                (self.rE.value, self.rI.value), t, rE_inp, rI_inp,
            )
        self.rE.value = rE
        self.rI.value = rI
        return rE


class WilsonCowanStepNoSaturation(brainstate.nn.Dynamics):
    r"""Wilson–Cowan neural mass model without saturation factor.

    This variant of the Wilson-Cowan model simplifies the dynamics by removing
    the saturation terms :math:`(1 - r \cdot r_E)` and :math:`(1 - r \cdot r_I)`.
    This leads to simpler analysis and potentially faster convergence while
    maintaining the core excitatory-inhibitory interaction dynamics.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau_E : Parameter , optional
        Excitatory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_E : Parameter , optional
        Excitatory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.2``.
    theta_E : Parameter , optional
        Excitatory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.8``.
    tau_I : Parameter , optional
        Inhibitory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_I : Parameter , optional
        Inhibitory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.``.
    theta_I : Parameter , optional
        Inhibitory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.0``.
    wEE : Parameter , optional
        E→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``12.``.
    wIE : Parameter , optional
        E→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.``.
    wEI : Parameter , optional
        I→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``13.``.
    wII : Parameter , optional
        I→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``11.``.
    noise_E : Noise or None, optional
        Additive noise process for the excitatory population. If provided, its
        output is added to ``rE_inp`` at each update. Default is ``None``.
    noise_I : Noise or None, optional
        Additive noise process for the inhibitory population. If provided, its
        output is added to ``rI_inp`` at each update. Default is ``None``.
    rE_init : Callable, optional
        Parameter for the excitatory state ``rE``. Default is
        ``braintools.init.ZeroInit()``.
    rI_init : Callable, optional
        Parameter for the inhibitory state ``rI``. Default is
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
    The continuous-time Wilson–Cowan equations without saturation are

    .. math::

        \tau_E \frac{dr_E}{dt} = -r_E(t) + F_E\bigl(w_{EE} r_E(t) - w_{EI} r_I(t) + I_E(t)\bigr),

    .. math::

        \tau_I \frac{dr_I}{dt} = -r_I(t) + F_I\bigl(w_{IE} r_E(t) - w_{II} r_I(t) + I_I(t)\bigr),

    with the sigmoidal transfer function

    .. math::

        F_j(x) = \frac{1}{1 + e^{-a_j (x - \theta_j)}} - \frac{1}{1 + e^{a_j \theta_j}},\quad j \in \{E, I\}.

    **Comparison to standard Wilson-Cowan:**

    - Removed saturation terms :math:`(1 - r \cdot r_E)` and :math:`(1 - r \cdot r_I)`
    - Removed parameter ``r`` (refractory parameter)
    - Simpler dynamics, potentially faster convergence
    - 10 parameters vs 11 in the standard model

    References
    ----------
    Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions
    in localized populations of model neurons. Biophysical Journal, 12, 1–24.

    Examples
    --------
    >>> model = brainmass.WilsonCowanStepNoSaturation(1)
    >>> brainstate.nn.init_all_states(model)
    >>> model.update(rE_inp=0.5)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # Excitatory parameters
        tau_E: Parameter = 1. * u.ms,
        a_E: Parameter = 1.2,
        theta_E: Parameter = 2.8,

        # Inhibitory parameters
        tau_I: Parameter = 1. * u.ms,
        a_I: Parameter = 1.,
        theta_I: Parameter = 4.0,

        # Connection parameters
        wEE: Parameter = 12.,
        wIE: Parameter = 4.,
        wEI: Parameter = 13.,
        wII: Parameter = 11.,

        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size=in_size)

        self.a_E = Param.init(a_E, self.varshape)
        self.a_I = Param.init(a_I, self.varshape)
        self.tau_E = Param.init(tau_E, self.varshape)
        self.tau_I = Param.init(tau_I, self.varshape)
        self.theta_E = Param.init(theta_E, self.varshape)
        self.theta_I = Param.init(theta_I, self.varshape)
        self.wEE = Param.init(wEE, self.varshape)
        self.wIE = Param.init(wIE, self.varshape)
        self.wEI = Param.init(wEI, self.varshape)
        self.wII = Param.init(wII, self.varshape)
        self.noise_E = noise_E
        self.noise_I = noise_I
        assert isinstance(noise_I, Noise) or noise_I is None, "noise_I must be an OUProcess or None"
        assert isinstance(noise_E, Noise) or noise_E is None, "noise_E must be an OUProcess or None"
        self.rE_init = rE_init
        self.rI_init = rI_init
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.rE = brainstate.HiddenState.init(self.rE_init, self.varshape, batch_size)
        self.rI = brainstate.HiddenState.init(self.rI_init, self.varshape, batch_size)

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
        wEE = self.wEE.value()
        wIE = self.wIE.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()
        xx = wEE * rE - wIE * rI + ext
        return (-rE + self.F(xx, a_E, theta_E)) / tau_E

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
        wEI = self.wEI.value()
        wII = self.wII.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()
        xx = wEI * rE - wII * rI + ext
        return (-rI + self.F(xx, a_I, theta_I)) / tau_I

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
            t = brainstate.environ.get('t', 0. * u.ms)
            rE, rI = getattr(braintools.quad, f'ode_{self.method}_step')(
                (self.rE.value, self.rI.value), t, rE_inp, rI_inp,
            )
        self.rE.value = rE
        self.rI.value = rI
        return rE


class WilsonCowanStepSymmetric(brainstate.nn.Dynamics):
    r"""Wilson-Cowan neural mass model with symmetric parameters.

    This variant of the Wilson-Cowan model uses symmetric parameters for the
    excitatory and inhibitory populations, i.e., both populations share the same
    time constant :math:`tau`, gain :math:`a`, and threshold :math:`theta`.
    This reduces the parameter space and can be useful for fitting or when
    assuming similar dynamics for both populations.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau : Parameter , optional
        Shared time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a : Parameter , optional
        Shared gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.1``.
    theta : Parameter , optional
        Shared threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``3.4``.
    wEE : Parameter , optional
        E→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``12.``.
    wIE : Parameter , optional
        E→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.``.
    wEI : Parameter , optional
        I→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``13.``.
    wII : Parameter , optional
        I→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``11.``.
    r : Parameter , optional
        Refractory parameter (dimensionless) that limits maximum activation.
        Broadcastable to ``in_size``. Default is ``1.``.
    noise_E : Noise or None, optional
        Additive noise process for the excitatory population. If provided, its
        output is added to ``rE_inp`` at each update. Default is ``None``.
    noise_I : Noise or None, optional
        Additive noise process for the inhibitory population. If provided, its
        output is added to ``rI_inp`` at each update. Default is ``None``.
    rE_init : Callable, optional
        Parameter for the excitatory state ``rE``. Default is
        ``braintools.init.ZeroInit()``.
    rI_init : Callable, optional
        Parameter for the inhibitory state ``rI``. Default is
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
    The continuous-time Wilson-Cowan equations with symmetric parameters are

    .. math::

        \\tau \\frac{dr_E}{dt} = -r_E(t) + \\bigl[1 - r\\, r_E(t)\\bigr]
        F\\bigl(w_{EE} r_E(t) - w_{EI} r_I(t) + I_E(t); a, \\theta\\bigr),

    .. math::

        \\tau \\frac{dr_I}{dt} = -r_I(t) + \\bigl[1 - r\\, r_I(t)\\bigr]
        F\\bigl(w_{IE} r_E(t) - w_{II} r_I(t) + I_I(t); a, \\theta\\bigr),

    with the sigmoidal transfer function

    .. math::

        F(x; a, \\theta) = \\frac{1}{1 + e^{-a (x - \\theta)}} - \\frac{1}{1 + e^{a \\theta}}.

    **Comparison to standard Wilson-Cowan:**

    - Unified parameters: :math:`\\tau` instead of :math:`\\tau_E, \\tau_I`
    - Unified sigmoid: :math:`a` instead of :math:`a_E, a_I`
    - Unified threshold: :math:`\\theta` instead of :math:`\\theta_E, \\theta_I`
    - Reduces parameter space from 11 to 6 parameters
    - Useful for fitting data when E/I symmetry is assumed

    References
    ----------
    Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions
    in localized populations of model neurons. Biophysical Journal, 12, 1–24.

    Examples
    --------
    >>> model = brainmass.WilsonCowanStepSymmetric(1)
    >>> brainstate.nn.init_all_states(model)
    >>> model.update(rE_inp=0.5)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # Shared parameters
        tau: Parameter = 1. * u.ms,
        a: Parameter = 1.1,
        theta: Parameter = 3.4,

        # Connection parameters
        wEE: Parameter = 12.,
        wIE: Parameter = 4.,
        wEI: Parameter = 13.,
        wII: Parameter = 11.,

        # Refractory parameter
        r: Parameter = 1.,

        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size=in_size)

        self.tau = Param.init(tau, self.varshape)
        self.a = Param.init(a, self.varshape)
        self.theta = Param.init(theta, self.varshape)
        self.wEE = Param.init(wEE, self.varshape)
        self.wIE = Param.init(wIE, self.varshape)
        self.wEI = Param.init(wEI, self.varshape)
        self.wII = Param.init(wII, self.varshape)
        self.r = Param.init(r, self.varshape)
        self.noise_E = noise_E
        self.noise_I = noise_I
        assert isinstance(noise_I, Noise) or noise_I is None, "noise_I must be an OUProcess or None"
        assert isinstance(noise_E, Noise) or noise_E is None, "noise_E must be an OUProcess or None"
        self.rE_init = rE_init
        self.rI_init = rI_init
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.rE = brainstate.HiddenState.init(self.rE_init, self.varshape, batch_size)
        self.rI = brainstate.HiddenState.init(self.rI_init, self.varshape, batch_size)

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
        wEE = self.wEE.value()
        wIE = self.wIE.value()
        r = self.r.value()
        a = self.a.value()
        theta = self.theta.value()
        tau = self.tau.value()
        xx = wEE * rE - wIE * rI + ext
        return (-rE + (1 - r * rE) * self.F(xx, a, theta)) / tau

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
        wEI = self.wEI.value()
        wII = self.wII.value()
        r = self.r.value()
        a = self.a.value()
        theta = self.theta.value()
        tau = self.tau.value()
        xx = wEI * rE - wII * rI + ext
        return (-rI + (1 - r * rI) * self.F(xx, a, theta)) / tau

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
            t = brainstate.environ.get('t', 0. * u.ms)
            rE, rI = getattr(braintools.quad, f'ode_{self.method}_step')(
                (self.rE.value, self.rI.value), t, rE_inp, rI_inp,
            )
        self.rE.value = rE
        self.rI.value = rI
        return rE


class WilsonCowanStepSimplified(brainstate.nn.Dynamics):
    r"""Wilson-Cowan neural mass model with simplified connectivity.

    This variant of the Wilson-Cowan model simplifies the connectivity by reducing
    the four connection weights to two parameters: one for excitatory connections
    (w_exc) and one for inhibitory connections (w_inh). This reduces the parameter
    space and can be useful for pedagogical purposes or initial exploration.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau_E : Parameter , optional
        Excitatory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_E : Parameter , optional
        Excitatory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.2``.
    theta_E : Parameter , optional
        Excitatory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.8``.
    tau_I : Parameter , optional
        Inhibitory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_I : Parameter , optional
        Inhibitory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.``.
    theta_I : Parameter , optional
        Inhibitory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.0``.
    w_exc : Parameter , optional
        Excitatory coupling strength (dimensionless). Applied to both E->E and E->I.
        Broadcastable to ``in_size``. Default is ``8.``.
    w_inh : Parameter , optional
        Inhibitory coupling strength (dimensionless). Applied to both I->E and I->I.
        Broadcastable to ``in_size``. Default is ``12.``.
    r : Parameter , optional
        Refractory parameter (dimensionless) that limits maximum activation.
        Broadcastable to ``in_size``. Default is ``1.``.
    noise_E : Noise or None, optional
        Additive noise process for the excitatory population. If provided, its
        output is added to ``rE_inp`` at each update. Default is ``None``.
    noise_I : Noise or None, optional
        Additive noise process for the inhibitory population. If provided, its
        output is added to ``rI_inp`` at each update. Default is ``None``.
    rE_init : Callable, optional
        Parameter for the excitatory state ``rE``. Default is
        ``braintools.init.ZeroInit()``.
    rI_init : Callable, optional
        Parameter for the inhibitory state ``rI``. Default is
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
    The continuous-time Wilson-Cowan equations with simplified connectivity are

    .. math::

        \\tau_E \\frac{dr_E}{dt} = -r_E(t) + \\bigl[1 - r\\, r_E(t)\\bigr]
        F_E\\bigl(w_{exc} r_E(t) - w_{inh} r_I(t) + I_E(t)\\bigr),

    .. math::

        \\tau_I \\frac{dr_I}{dt} = -r_I(t) + \\bigl[1 - r\\, r_I(t)\\bigr]
        F_I\\bigl(w_{exc} r_E(t) - w_{inh} r_I(t) + I_I(t)\\bigr),

    with the sigmoidal transfer function

    .. math::

        F_j(x) = \\frac{1}{1 + e^{-a_j (x - \\theta_j)}} - \\frac{1}{1 + e^{a_j \\theta_j}},\\quad j \\in \\{E, I\\}.

    **Comparison to standard Wilson-Cowan:**

    - Simplified connectivity: 2 weights (w_exc, w_inh) instead of 4 (wEE, wIE, wEI, wII)
    - Internal mapping: wEE = wIE = w_exc, wEI = wII = w_inh
    - Reduces parameter space from 11 to 8 parameters
    - Useful for pedagogical purposes and quick exploration

    References
    ----------
    Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions
    in localized populations of model neurons. Biophysical Journal, 12, 1–24.

    Examples
    --------
    >>> model = brainmass.WilsonCowanStepSimplified(1)
    >>> brainstate.nn.init_all_states(model)
    >>> model.update(rE_inp=0.5)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # Excitatory parameters
        tau_E: Parameter = 1. * u.ms,
        a_E: Parameter = 1.2,
        theta_E: Parameter = 2.8,

        # Inhibitory parameters
        tau_I: Parameter = 1. * u.ms,
        a_I: Parameter = 1.,
        theta_I: Parameter = 4.0,

        # Simplified connection parameters
        w_exc: Parameter = 8.,
        w_inh: Parameter = 12.,

        # Refractory parameter
        r: Parameter = 1.,

        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size=in_size)

        self.a_E = Param.init(a_E, self.varshape)
        self.a_I = Param.init(a_I, self.varshape)
        self.tau_E = Param.init(tau_E, self.varshape)
        self.tau_I = Param.init(tau_I, self.varshape)
        self.theta_E = Param.init(theta_E, self.varshape)
        self.theta_I = Param.init(theta_I, self.varshape)
        self.w_exc = Param.init(w_exc, self.varshape)
        self.w_inh = Param.init(w_inh, self.varshape)
        self.r = Param.init(r, self.varshape)
        self.noise_E = noise_E
        self.noise_I = noise_I
        assert isinstance(noise_I, Noise) or noise_I is None, "noise_I must be an OUProcess or None"
        assert isinstance(noise_E, Noise) or noise_E is None, "noise_E must be an OUProcess or None"
        self.rE_init = rE_init
        self.rI_init = rI_init
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.rE = brainstate.HiddenState.init(self.rE_init, self.varshape, batch_size)
        self.rI = brainstate.HiddenState.init(self.rI_init, self.varshape, batch_size)

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
        w_exc = self.w_exc.value()
        w_inh = self.w_inh.value()
        r = self.r.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()
        xx = w_exc * rE - w_inh * rI + ext
        return (-rE + (1 - r * rE) * self.F(xx, a_E, theta_E)) / tau_E

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
        w_exc = self.w_exc.value()
        w_inh = self.w_inh.value()
        r = self.r.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()
        xx = w_exc * rE - w_inh * rI + ext
        return (-rI + (1 - r * rI) * self.F(xx, a_I, theta_I)) / tau_I

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
            t = brainstate.environ.get('t', 0. * u.ms)
            rE, rI = getattr(braintools.quad, f'ode_{self.method}_step')(
                (self.rE.value, self.rI.value), t, rE_inp, rI_inp,
            )
        self.rE.value = rE
        self.rI.value = rI
        return rE


class WilsonCowanStepLinear(brainstate.nn.Dynamics):
    r"""Wilson-Cowan neural mass model with linear (ReLU) transfer function.

    This variant of the Wilson-Cowan model replaces the sigmoidal transfer function
    with a rectified linear unit (ReLU) function: [x]+ = max(0, x). This removes
    the need for sigmoid gain and threshold parameters, simplifies the computational
    graph, and can be more gradient-friendly for optimization tasks.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau_E : Parameter , optional
        Excitatory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    tau_I : Parameter , optional
        Inhibitory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    wEE : Parameter , optional
        E->E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``0.8``.
    wIE : Parameter , optional
        E->I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``0.3``.
    wEI : Parameter , optional
        I->E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.0``.
    wII : Parameter , optional
        I->I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``0.85``.
    r : Parameter , optional
        Refractory parameter (dimensionless) that limits maximum activation.
        Broadcastable to ``in_size``. Default is ``1.``.
    noise_E : Noise or None, optional
        Additive noise process for the excitatory population. If provided, its
        output is added to ``rE_inp`` at each update. Default is ``None``.
    noise_I : Noise or None, optional
        Additive noise process for the inhibitory population. If provided, its
        output is added to ``rI_inp`` at each update. Default is ``None``.
    rE_init : Callable, optional
        Parameter for the excitatory state ``rE``. Default is
        ``braintools.init.ZeroInit()``.
    rI_init : Callable, optional
        Parameter for the inhibitory state ``rI``. Default is
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
    The continuous-time Wilson-Cowan equations with ReLU transfer are

    .. math::

        \\tau_E \\frac{dr_E}{dt} = -r_E(t) + \\bigl[1 - r\\, r_E(t)\\bigr]
        \\bigl[w_{EE} r_E(t) - w_{EI} r_I(t) + I_E(t)\\bigr]_+,

    .. math::

        \\tau_I \\frac{dr_I}{dt} = -r_I(t) + \\bigl[1 - r\\, r_I(t)\\bigr]
        \\bigl[w_{IE} r_E(t) - w_{II} r_I(t) + I_I(t)\\bigr]_+,

    where :math:`[x]_+ = \\max(0, x)` is the rectified linear unit.

    **Comparison to standard Wilson-Cowan:**

    - ReLU transfer function instead of sigmoid
    - Removed sigmoid parameters: a_E, a_I, theta_E, theta_I
    - Reduces parameter space from 11 to 7 parameters
    - Simpler computational graph, faster evaluation
    - More gradient-friendly for optimization
    - **Important:** Default weights are scaled down by ~13-15x for stability

    References
    ----------
    Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions
    in localized populations of model neurons. Biophysical Journal, 12, 1–24.

    Examples
    --------
    >>> model = brainmass.WilsonCowanStepLinear(1)
    >>> brainstate.nn.init_all_states(model)
    >>> model.update(rE_inp=0.5)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # Time constants
        tau_E: Parameter = 1. * u.ms,
        tau_I: Parameter = 1. * u.ms,

        # Connection parameters (scaled down for stability)
        wEE: Parameter = 0.8,
        wIE: Parameter = 0.3,
        wEI: Parameter = 1.0,
        wII: Parameter = 0.85,

        # Refractory parameter
        r: Parameter = 1.,

        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size=in_size)

        self.tau_E = Param.init(tau_E, self.varshape)
        self.tau_I = Param.init(tau_I, self.varshape)
        self.wEE = Param.init(wEE, self.varshape)
        self.wIE = Param.init(wIE, self.varshape)
        self.wEI = Param.init(wEI, self.varshape)
        self.wII = Param.init(wII, self.varshape)
        self.r = Param.init(r, self.varshape)
        self.noise_E = noise_E
        self.noise_I = noise_I
        assert isinstance(noise_I, Noise) or noise_I is None, "noise_I must be an OUProcess or None"
        assert isinstance(noise_E, Noise) or noise_E is None, "noise_E must be an OUProcess or None"
        self.rE_init = rE_init
        self.rI_init = rI_init
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.rE = brainstate.HiddenState.init(self.rE_init, self.varshape, batch_size)
        self.rI = brainstate.HiddenState.init(self.rI_init, self.varshape, batch_size)

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
        wEE = self.wEE.value()
        wIE = self.wIE.value()
        r = self.r.value()
        tau_E = self.tau_E.value()
        xx = wEE * rE - wIE * rI + ext
        return (-rE + (1 - r * rE) * u.math.maximum(xx, 0.)) / tau_E

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
        wEI = self.wEI.value()
        wII = self.wII.value()
        r = self.r.value()
        tau_I = self.tau_I.value()
        xx = wEI * rE - wII * rI + ext
        return (-rI + (1 - r * rI) * u.math.maximum(xx, 0.)) / tau_I

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
            t = brainstate.environ.get('t', 0. * u.ms)
            rE, rI = getattr(braintools.quad, f'ode_{self.method}_step')(
                (self.rE.value, self.rI.value), t, rE_inp, rI_inp,
            )
        self.rE.value = rE
        self.rI.value = rI
        return rE
