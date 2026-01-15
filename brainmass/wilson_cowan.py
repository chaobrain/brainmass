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

from typing import Callable, Sequence, Optional

import braintools
import brainunit as u
import jax.nn
import numpy as np

import brainstate
from brainstate.nn import Param
from .coupling import additive_coupling
from .noise import Noise
from .typing import Parameter, Initializer

__all__ = [
    'WilsonCowanBase',
    'WilsonCowanStep',
    'WilsonCowanNoSaturationStep',
    'WilsonCowanSymmetricStep',
    'WilsonCowanSimplifiedStep',
    'WilsonCowanLinearStep',
    'WilsonCowanDivisiveStep',
    'WilsonCowanDivisiveInputStep',
    'WilsonCowanDelayedStep',
    'WilsonCowanAdaptiveStep',
    'WilsonCowanThreePopBase',
    'WilsonCowanThreePopulationStep',
]


class WilsonCowanBase(brainstate.nn.Dynamics):
    r"""Abstract base class for Wilson-Cowan neural mass model variants.

    This base class provides common functionality for all Wilson-Cowan variants,
    including state management, numerical integration, and noise handling.
    Subclasses must implement ``drE()``, ``drI()``, and optionally ``F()`` methods.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
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
    method : str, optional
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
    Subclasses must implement:

    - ``drE(rE, rI, ext)``: Right-hand side for excitatory population
    - ``drI(rI, rE, ext)``: Right-hand side for inhibitory population
    - ``F(x, *args)`` (optional): Transfer function

    This class follows the same pattern as ``XY_Oscillator`` for consistency
    with the codebase architecture.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # noise parameters
        noise_E: Noise = None,
        noise_I: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        # Validate noise parameters
        assert isinstance(noise_E, Noise) or noise_E is None, \
            "noise_E must be a Noise instance or None."
        assert isinstance(noise_I, Noise) or noise_I is None, \
            "noise_I must be a Noise instance or None."
        assert callable(rE_init), "rE_init must be a callable."
        assert callable(rI_init), "rI_init must be a callable."

        self.rE_init = rE_init
        self.rI_init = rI_init
        self.noise_E = noise_E
        self.noise_I = noise_I
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Initialize model states ``rE`` and ``rI``.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.rE = brainstate.HiddenState.init(self.rE_init, self.varshape, batch_size)
        self.rI = brainstate.HiddenState.init(self.rI_init, self.varshape, batch_size)

    def F(self, x, a, theta):
        # 1 / (1 + jnp.exp(-a * (x - theta))) - 1 / (1 + jnp.exp(a * theta))
        return jax.nn.sigmoid(a * (x - theta)) - jax.nn.sigmoid(-a * theta)

    def drE(self, rE, rI, ext):
        """Right-hand side for the excitatory population.

        Must be implemented by subclasses.

        Parameters
        ----------
        rE : array-like
            Excitatory activity (dimensionless).
        rI : array-like
            Inhibitory activity (dimensionless), broadcastable to ``rE``.
        ext : array-like or scalar
            External input to E.

        Returns
        -------
        array-like
            Time derivative ``drE/dt`` with unit of ``1/time``.
        """
        raise NotImplementedError

    def drI(self, rI, rE, ext):
        """Right-hand side for the inhibitory population.

        Must be implemented by subclasses.

        Parameters
        ----------
        rI : array-like
            Inhibitory activity (dimensionless).
        rE : array-like
            Excitatory activity (dimensionless), broadcastable to ``rI``.
        ext : array-like or scalar
            External input to I.

        Returns
        -------
        array-like
            Time derivative ``drI/dt`` with unit of ``1/time``.
        """
        raise NotImplementedError

    def derivative(self, state, t, E_ext, I_ext):
        """Compute derivatives for both populations.

        Parameters
        ----------
        state : tuple
            Tuple of (rE, rI) states.
        t : float
            Current time.
        E_ext : array-like or scalar
            External input to excitatory population.
        I_ext : array-like or scalar
            External input to inhibitory population.

        Returns
        -------
        tuple
            Tuple of (drE/dt, drI/dt) derivatives.
        """
        rE, rI = state
        drE_dt = self.drE(rE, rI, E_ext)
        drI_dt = self.drI(rI, rE, I_ext)
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
        The method performs numerical integration using the specified method
        (``exp_euler`` by default) and updates the internal states ``rE`` and
        ``rI`` in-place.
        """
        # Handle inputs
        rE_inp = 0. if rE_inp is None else rE_inp
        rI_inp = 0. if rI_inp is None else rI_inp

        # Add noise
        if self.noise_E is not None:
            rE_inp = rE_inp + self.noise_E()
        if self.noise_I is not None:
            rI_inp = rI_inp + self.noise_I()

        # Numerical integration
        if self.method == 'exp_euler':
            rE = brainstate.nn.exp_euler_step(self.drE, self.rE.value, self.rI.value, rE_inp)
            rI = brainstate.nn.exp_euler_step(self.drI, self.rI.value, self.rE.value, rI_inp)
        else:
            method = getattr(braintools.quad, f'ode_{self.method}_step')
            t = brainstate.environ.get('t', 0 * u.ms)
            rE, rI = method(self.derivative, (self.rE.value, self.rI.value), t, rE_inp, rI_inp)

        self.rE.value = rE
        self.rI.value = rI
        return rE


class WilsonCowanStep(WilsonCowanBase):
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
        super().__init__(in_size, noise_E, noise_I, rE_init, rI_init, method)

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

    def drE(self, rE, rI, ext):
        wEE = self.wEE.value()
        wIE = self.wIE.value()
        r = self.r.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()

        xx = wEE * rE - wIE * rI + ext
        return (-rE + (1 - r * rE) * self.F(xx, a_E, theta_E)) / tau_E

    def drI(self, rI, rE, ext):
        wEI = self.wEI.value()
        wII = self.wII.value()
        r = self.r.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()

        xx = wEI * rE - wII * rI + ext
        return (-rI + (1 - r * rI) * self.F(xx, a_I, theta_I)) / tau_I


class WilsonCowanNoSaturationStep(WilsonCowanBase):
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
    >>> model = brainmass.WilsonCowanNoSaturationStep(1)
    >>> model.init_all_states()
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
        super().__init__(in_size, noise_E, noise_I, rE_init, rI_init, method)

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

    def drE(self, rE, rI, ext):
        wEE = self.wEE.value()
        wIE = self.wIE.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()

        xx = wEE * rE - wIE * rI + ext
        return (-rE + self.F(xx, a_E, theta_E)) / tau_E

    def drI(self, rI, rE, ext):
        wEI = self.wEI.value()
        wII = self.wII.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()

        xx = wEI * rE - wII * rI + ext
        return (-rI + self.F(xx, a_I, theta_I)) / tau_I


class WilsonCowanSymmetricStep(WilsonCowanBase):
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

        \tau \frac{dr_E}{dt} = -r_E(t) + \bigl[1 - r\, r_E(t)\bigr]
        F\bigl(w_{EE} r_E(t) - w_{EI} r_I(t) + I_E(t); a, \theta\bigr),

    .. math::

        \tau \frac{dr_I}{dt} = -r_I(t) + \bigl[1 - r\, r_I(t)\bigr]
        F\bigl(w_{IE} r_E(t) - w_{II} r_I(t) + I_I(t); a, \theta\bigr),

    with the sigmoidal transfer function

    .. math::

        F(x; a, \theta) = \frac{1}{1 + e^{-a (x - \theta)}} - \frac{1}{1 + e^{a \theta}}.

    **Comparison to standard Wilson-Cowan:**

    - Unified parameters: :math:`\tau` instead of :math:`\tau_E, \tau_I`
    - Unified sigmoid: :math:`a` instead of :math:`a_E, a_I`
    - Unified threshold: :math:`\theta` instead of :math:`\theta_E, \theta_I`
    - Reduces parameter space from 11 to 6 parameters
    - Useful for fitting data when E/I symmetry is assumed

    References
    ----------
    Wilson, H. R., & Cowan, J. D. (1972). Excitatory and inhibitory interactions
    in localized populations of model neurons. Biophysical Journal, 12, 1–24.

    Examples
    --------
    >>> model = brainmass.WilsonCowanSymmetricStep(1)
    >>> model.init_all_states()
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
        super().__init__(in_size, noise_E, noise_I, rE_init, rI_init, method)

        self.tau = Param.init(tau, self.varshape)
        self.a = Param.init(a, self.varshape)
        self.theta = Param.init(theta, self.varshape)
        self.wEE = Param.init(wEE, self.varshape)
        self.wIE = Param.init(wIE, self.varshape)
        self.wEI = Param.init(wEI, self.varshape)
        self.wII = Param.init(wII, self.varshape)
        self.r = Param.init(r, self.varshape)

    def drE(self, rE, rI, ext):
        wEE = self.wEE.value()
        wIE = self.wIE.value()
        r = self.r.value()
        a = self.a.value()
        theta = self.theta.value()
        tau = self.tau.value()

        xx = wEE * rE - wIE * rI + ext
        return (-rE + (1 - r * rE) * self.F(xx, a, theta)) / tau

    def drI(self, rI, rE, ext):
        wEI = self.wEI.value()
        wII = self.wII.value()
        r = self.r.value()
        a = self.a.value()
        theta = self.theta.value()
        tau = self.tau.value()

        xx = wEI * rE - wII * rI + ext
        return (-rI + (1 - r * rI) * self.F(xx, a, theta)) / tau


class WilsonCowanSimplifiedStep(WilsonCowanBase):
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

        \tau_E \frac{dr_E}{dt} = -r_E(t) + \bigl[1 - r\, r_E(t)\bigr]
        F_E\bigl(w_{exc} r_E(t) - w_{inh} r_I(t) + I_E(t)\bigr),

    .. math::

        \tau_I \frac{dr_I}{dt} = -r_I(t) + \bigl[1 - r\, r_I(t)\bigr]
        F_I\bigl(w_{exc} r_E(t) - w_{inh} r_I(t) + I_I(t)\bigr),

    with the sigmoidal transfer function

    .. math::

        F_j(x) = \frac{1}{1 + e^{-a_j (x - \theta_j)}} - \frac{1}{1 + e^{a_j \theta_j}},\quad j \in \{E, I\}.

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
    >>> model = brainmass.WilsonCowanSimplifiedStep(1)
    >>> model.init_all_states()
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
        super().__init__(in_size, noise_E, noise_I, rE_init, rI_init, method)

        self.a_E = Param.init(a_E, self.varshape)
        self.a_I = Param.init(a_I, self.varshape)
        self.tau_E = Param.init(tau_E, self.varshape)
        self.tau_I = Param.init(tau_I, self.varshape)
        self.theta_E = Param.init(theta_E, self.varshape)
        self.theta_I = Param.init(theta_I, self.varshape)
        self.w_exc = Param.init(w_exc, self.varshape)
        self.w_inh = Param.init(w_inh, self.varshape)
        self.r = Param.init(r, self.varshape)

    def drE(self, rE, rI, ext):
        w_exc = self.w_exc.value()
        w_inh = self.w_inh.value()
        r = self.r.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()

        xx = w_exc * rE - w_inh * rI + ext
        return (-rE + (1 - r * rE) * self.F(xx, a_E, theta_E)) / tau_E

    def drI(self, rI, rE, ext):
        w_exc = self.w_exc.value()
        w_inh = self.w_inh.value()
        r = self.r.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()

        xx = w_exc * rE - w_inh * rI + ext
        return (-rI + (1 - r * rI) * self.F(xx, a_I, theta_I)) / tau_I


class WilsonCowanLinearStep(WilsonCowanBase):
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

        \tau_E \frac{dr_E}{dt} = -r_E(t) + \bigl[1 - r\, r_E(t)\bigr]
        \bigl[w_{EE} r_E(t) - w_{EI} r_I(t) + I_E(t)\bigr]_+,

    .. math::

        \tau_I \frac{dr_I}{dt} = -r_I(t) + \bigl[1 - r\, r_I(t)\bigr]
        \bigl[w_{IE} r_E(t) - w_{II} r_I(t) + I_I(t)\bigr]_+,

    where :math:`[x]_+ = \max(0, x)` is the rectified linear unit.

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
    >>> model = brainmass.WilsonCowanLinearStep(1)
    >>> model.init_all_states()
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
        super().__init__(in_size, noise_E, noise_I, rE_init, rI_init, method)

        self.tau_E = Param.init(tau_E, self.varshape)
        self.tau_I = Param.init(tau_I, self.varshape)
        self.wEE = Param.init(wEE, self.varshape)
        self.wIE = Param.init(wIE, self.varshape)
        self.wEI = Param.init(wEI, self.varshape)
        self.wII = Param.init(wII, self.varshape)
        self.r = Param.init(r, self.varshape)

    def drE(self, rE, rI, ext):
        wEE = self.wEE.value()
        wIE = self.wIE.value()
        r = self.r.value()
        tau_E = self.tau_E.value()
        xx = wEE * rE - wIE * rI + ext
        return (-rE + (1 - r * rE) * u.math.maximum(xx, 0.)) / tau_E

    def drI(self, rI, rE, ext):
        wEI = self.wEI.value()
        wII = self.wII.value()
        r = self.r.value()
        tau_I = self.tau_I.value()
        xx = wEI * rE - wII * rI + ext
        return (-rI + (1 - r * rI) * u.math.maximum(xx, 0.)) / tau_I


class WilsonCowanDivisiveStep(WilsonCowanBase):
    r"""Wilson-Cowan neural mass model with divisive normalization (gain modulation).

    This variant implements divisive inhibition through gain modulation, where
    inhibition divides the output gain rather than subtracting from the input.
    This form of normalization is commonly observed in visual cortex and implements
    contrast normalization.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau_E : Parameter, optional
        Excitatory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_E : Parameter, optional
        Excitatory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.2``.
    theta_E : Parameter, optional
        Excitatory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.8``.
    tau_I : Parameter, optional
        Inhibitory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_I : Parameter, optional
        Inhibitory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.``.
    theta_I : Parameter, optional
        Inhibitory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.0``.
    wEE : Parameter, optional
        E→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``6.`` (reduced from standard model for stability).
    wIE : Parameter, optional
        E→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.`` (reduced from standard model for stability).
    wEI : Parameter, optional
        I→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``6.5`` (reduced from standard model for stability).
    wII : Parameter, optional
        I→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``5.5`` (reduced from standard model for stability).
    r : Parameter, optional
        Refractory parameter (dimensionless) that limits maximum activation.
        Broadcastable to ``in_size``. Default is ``1.``.
    sigma_E : Parameter, optional
        Excitatory semisaturation constant (dimensionless) that prevents division
        by zero and controls the strength of divisive normalization. Broadcastable
        to ``in_size``. Default is ``1.``.
    sigma_I : Parameter, optional
        Inhibitory semisaturation constant (dimensionless) that prevents division
        by zero and controls the strength of divisive normalization. Broadcastable
        to ``in_size``. Default is ``1.``.
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
    method : str
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
    The continuous-time Wilson-Cowan equations with divisive gain modulation are

    .. math::

        \tau_E \frac{dr_E}{dt} = -r_E(t) + \frac{1 - r\, r_E(t)}{\sigma_E + w_{EI} r_I(t)}
        F_E\bigl(w_{EE} r_E(t) + I_E(t)\bigr),

    .. math::

        \tau_I \frac{dr_I}{dt} = -r_I(t) + \frac{1 - r\, r_I(t)}{\sigma_I + w_{II} r_I(t)}
        F_I\bigl(w_{IE} r_E(t) + I_I(t)\bigr),

    with the sigmoidal transfer function

    .. math::

        F_j(x) = \frac{1}{1 + e^{-a_j (x - \theta_j)}} - \frac{1}{1 + e^{a_j \theta_j}},\quad j \in \{E, I\}.

    Divisive normalization implements gain modulation where inhibition divides the
    saturating term rather than subtracting from the input. This is biologically
    plausible and matches normalization observed in visual cortex.

    References
    ----------
    .. [1] Carandini, M., & Heeger, D. J. (2012). Normalization as a canonical neural
           computation. Nature Reviews Neuroscience, 13(1), 51-62.
    .. [2] Reynolds, J. H., & Heeger, D. J. (2009). The normalization model of attention.
           Neuron, 61(2), 168-185.

    Examples
    --------
    >>> import brainmass
    >>> import brainunit as u
    >>> model = brainmass.WilsonCowanDivisiveStep(
    ...     in_size=100,
    ...     tau_E=1.*u.ms,
    ...     sigma_E=1.0,
    ...     sigma_I=1.0
    ... )
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

        # Connection parameters (reduced for divisive inhibition)
        wEE: Parameter = 6.,
        wIE: Parameter = 2.,
        wEI: Parameter = 6.5,
        wII: Parameter = 5.5,

        # Refractory parameter
        r: Parameter = 1.,

        # Normalization parameters
        sigma_E: Parameter = 1.,
        sigma_I: Parameter = 1.,

        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size, noise_E, noise_I, rE_init, rI_init, method)

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
        self.sigma_E = Param.init(sigma_E, self.varshape)
        self.sigma_I = Param.init(sigma_I, self.varshape)

    def drE(self, rE, rI, ext):
        wEE = self.wEE.value()
        wEI = self.wEI.value()
        r = self.r.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()
        sigma_E = self.sigma_E.value()

        xx = wEE * rE + ext
        gain = (1 - r * rE) / (sigma_E + wEI * rI)
        return (-rE + gain * self.F(xx, a_E, theta_E)) / tau_E

    def drI(self, rI, rE, ext):
        wIE = self.wIE.value()
        wII = self.wII.value()
        r = self.r.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()
        sigma_I = self.sigma_I.value()

        xx = wIE * rE + ext
        gain = (1 - r * rI) / (sigma_I + wII * rI)
        return (-rI + gain * self.F(xx, a_I, theta_I)) / tau_I


class WilsonCowanDivisiveInputStep(WilsonCowanBase):
    r"""Wilson-Cowan neural mass model with divisive normalization (input normalization).

    This variant implements divisive inhibition applied to the input before the
    transfer function. Unlike gain modulation, here the excitatory input is divided
    by inhibition before being passed through the transfer function. This provides
    a simpler mathematical formulation while still implementing contrast normalization.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau_E : Parameter, optional
        Excitatory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_E : Parameter, optional
        Excitatory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.2``.
    theta_E : Parameter, optional
        Excitatory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.8``.
    tau_I : Parameter, optional
        Inhibitory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_I : Parameter, optional
        Inhibitory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.``.
    theta_I : Parameter, optional
        Inhibitory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.0``.
    wEE : Parameter, optional
        E→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``6.`` (reduced from standard model for stability).
    wIE : Parameter, optional
        E→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.`` (reduced from standard model for stability).
    wEI : Parameter, optional
        I→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``6.5`` (reduced from standard model for stability).
    wII : Parameter, optional
        I→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``5.5`` (reduced from standard model for stability).
    r : Parameter, optional
        Refractory parameter (dimensionless) that limits maximum activation.
        Broadcastable to ``in_size``. Default is ``1.``.
    sigma_E : Parameter, optional
        Excitatory semisaturation constant (dimensionless) that prevents division
        by zero and controls the strength of divisive normalization. Broadcastable
        to ``in_size``. Default is ``1.``.
    sigma_I : Parameter, optional
        Inhibitory semisaturation constant (dimensionless) that prevents division
        by zero and controls the strength of divisive normalization. Broadcastable
        to ``in_size``. Default is ``1.``.
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
    method : str
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
    The continuous-time Wilson-Cowan equations with divisive input normalization are

    .. math::

        \tau_E \frac{dr_E}{dt} = -r_E(t) + \bigl[1 - r\, r_E(t)\bigr]
        F_E\left(\frac{w_{EE} r_E(t)}{\sigma_E + w_{EI} r_I(t)} + I_E(t)\right),

    .. math::

        \tau_I \frac{dr_I}{dt} = -r_I(t) + \bigl[1 - r\, r_I(t)\bigr]
        F_I\left(\frac{w_{IE} r_E(t)}{\sigma_I + w_{II} r_I(t)} + I_I(t)\right),

    with the sigmoidal transfer function

    .. math::

        F_j(x) = \frac{1}{1 + e^{-a_j (x - \theta_j)}} - \frac{1}{1 + e^{a_j \theta_j}},\quad j \in \{E, I\}.

    Divisive normalization is applied to the excitatory input before the transfer
    function. This is mathematically simpler than gain modulation and still implements
    contrast normalization observed in neural systems.

    References
    ----------
    .. [1] Carandini, M., & Heeger, D. J. (2012). Normalization as a canonical neural
           computation. Nature Reviews Neuroscience, 13(1), 51-62.
    .. [2] Reynolds, J. H., & Heeger, D. J. (2009). The normalization model of attention.
           Neuron, 61(2), 168-185.

    Examples
    --------
    >>> import brainmass
    >>> import brainunit as u
    >>> model = brainmass.WilsonCowanDivisiveInputStep(
    ...     in_size=100,
    ...     tau_E=1.*u.ms,
    ...     sigma_E=1.0,
    ...     sigma_I=1.0
    ... )
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

        # Connection parameters (reduced for divisive inhibition)
        wEE: Parameter = 6.,
        wIE: Parameter = 2.,
        wEI: Parameter = 6.5,
        wII: Parameter = 5.5,

        # Refractory parameter
        r: Parameter = 1.,

        # Normalization parameters
        sigma_E: Parameter = 1.,
        sigma_I: Parameter = 1.,

        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size, noise_E, noise_I, rE_init, rI_init, method)

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
        self.sigma_E = Param.init(sigma_E, self.varshape)
        self.sigma_I = Param.init(sigma_I, self.varshape)

    def drE(self, rE, rI, ext):
        wEE = self.wEE.value()
        wEI = self.wEI.value()
        r = self.r.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()
        sigma_E = self.sigma_E.value()

        normalized_input = (wEE * rE) / (sigma_E + wEI * rI)
        xx = normalized_input + ext
        return (-rE + (1 - r * rE) * self.F(xx, a_E, theta_E)) / tau_E

    def drI(self, rI, rE, ext):
        wIE = self.wIE.value()
        wII = self.wII.value()
        r = self.r.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()
        sigma_I = self.sigma_I.value()

        normalized_input = (wIE * rE) / (sigma_I + wII * rI)
        xx = normalized_input + ext
        return (-rI + (1 - r * rI) * self.F(xx, a_I, theta_I)) / tau_I


class WilsonCowanDelayedStep(WilsonCowanBase):
    r"""Wilson-Cowan neural mass model with connection delays.

    This variant incorporates explicit time delays in the connections between
    populations, modeling axonal conduction delays in neural transmission. Each
    connection (E→E, E→I, I→E, I→I) can have its own delay time.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau_E : Parameter, optional
        Excitatory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_E : Parameter, optional
        Excitatory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.2``.
    theta_E : Parameter, optional
        Excitatory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.8``.
    tau_I : Parameter, optional
        Inhibitory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_I : Parameter, optional
        Inhibitory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.``.
    theta_I : Parameter, optional
        Inhibitory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.0``.
    wEE : Parameter, optional
        E→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``12.``.
    wIE : Parameter, optional
        E→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.``.
    wEI : Parameter, optional
        I→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``13.``.
    wII : Parameter, optional
        I→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``11.``.
    r : Parameter, optional
        Refractory parameter (dimensionless) that limits maximum activation.
        Broadcastable to ``in_size``. Default is ``1.``.
    delay_EE : Parameter, optional
        E→E connection delay with unit of time (e.g., ``2. * u.ms``).
        Broadcastable to ``in_size``. Default is ``2. * u.ms``.
    delay_IE : Parameter, optional
        E→I connection delay with unit of time (e.g., ``2. * u.ms``).
        Broadcastable to ``in_size``. Default is ``2. * u.ms``.
    delay_EI : Parameter, optional
        I→E connection delay with unit of time (e.g., ``1.5 * u.ms``).
        Broadcastable to ``in_size``. Default is ``1.5 * u.ms`` (faster inhibition).
    delay_II : Parameter, optional
        I→I connection delay with unit of time (e.g., ``1.5 * u.ms``).
        Broadcastable to ``in_size``. Default is ``1.5 * u.ms`` (faster inhibition).
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
    method : str
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
    The continuous-time Wilson-Cowan equations with delays are

    .. math::

        \tau_E \frac{dr_E}{dt} = -r_E(t) + \bigl[1 - r\, r_E(t)\bigr]
        F_E\bigl(w_{EE} r_E(t - d_{EE}) - w_{EI} r_I(t - d_{EI}) + I_E(t)\bigr),

    .. math::

        \tau_I \frac{dr_I}{dt} = -r_I(t) + \bigl[1 - r\, r_I(t)\bigr]
        F_I\bigl(w_{IE} r_E(t - d_{IE}) - w_{II} r_I(t - d_{II}) + I_I(t)\bigr),

    with the sigmoidal transfer function

    .. math::

        F_j(x) = \frac{1}{1 + e^{-a_j (x - \theta_j)}} - \frac{1}{1 + e^{a_j \theta_j}},\quad j \in \{E, I\}.

    Each connection has its own delay :math:`d_{XY}` representing axonal
    conduction time. Delays are handled efficiently using circular buffers
    managed by the ``brainstate`` framework.

    References
    ----------
    .. [1] Robinson, P. A., et al. (2002). Prediction of electroencephalographic
           spectra from neurophysiology. Physical Review E, 63(2), 021903.
    .. [2] Breakspear, M., et al. (2006). A unifying explanation of primary
           generalized seizures through nonlinear brain modeling and bifurcation
           analysis. Cerebral Cortex, 16(9), 1296-1313.

    Examples
    --------
    >>> import brainmass
    >>> import brainunit as u
    >>> model = brainmass.WilsonCowanDelayedStep(
    ...     in_size=100,
    ...     delay_EE=2.*u.ms,
    ...     delay_EI=1.5*u.ms
    ... )
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

        # Refractory parameter
        r: Parameter = 1.,

        # Delay parameters
        delay_EE: Parameter = 2. * u.ms,
        delay_IE: Parameter = 2. * u.ms,
        delay_EI: Parameter = 1.5 * u.ms,
        delay_II: Parameter = 1.5 * u.ms,

        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size, noise_E, noise_I, rE_init, rI_init, method)

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

        # Set up delay buffers for each connection
        # Create neuron indices for delay mechanism
        self.delay_rE_to_E = self.prefetch_delay('rE', delay_EE, init=rE_init)
        self.delay_rE_to_I = self.prefetch_delay('rE', delay_IE, init=rE_init)
        self.delay_rI_to_E = self.prefetch_delay('rI', delay_EI, init=rI_init)
        self.delay_rI_to_I = self.prefetch_delay('rI', delay_II, init=rI_init)

    def drE(self, rE, rI, ext):
        wEE = self.wEE.value()
        wEI = self.wEI.value()
        r = self.r.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()

        # Fetch delayed values
        rE_delayed = self.delay_rE_to_E()
        rI_delayed = self.delay_rI_to_E()

        xx = wEE * rE_delayed - wEI * rI_delayed + ext
        return (-rE + (1 - r * rE) * self.F(xx, a_E, theta_E)) / tau_E

    def drI(self, rI, rE, ext):
        wIE = self.wIE.value()
        wII = self.wII.value()
        r = self.r.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()

        # Fetch delayed values
        rE_delayed = self.delay_rE_to_I()
        rI_delayed = self.delay_rI_to_I()

        xx = wIE * rE_delayed - wII * rI_delayed + ext
        return (-rI + (1 - r * rI) * self.F(xx, a_I, theta_I)) / tau_I


class WilsonCowanAdaptiveStep(WilsonCowanBase):
    r"""Wilson-Cowan neural mass model with spike-frequency adaptation.

    This variant adds adaptation currents to both excitatory and inhibitory
    populations, implementing spike-frequency adaptation. Adaptation currents
    build up with sustained activity and reduce the effective drive to each
    population, modeling neural fatigue and habituation effects.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E and I). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau_E : Parameter, optional
        Excitatory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_E : Parameter, optional
        Excitatory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.2``.
    theta_E : Parameter, optional
        Excitatory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``2.8``.
    tau_I : Parameter, optional
        Inhibitory time constant with unit of time (e.g., ``1. * u.ms``).
        Broadcastable to ``in_size``. Default is ``1. * u.ms``.
    a_I : Parameter, optional
        Inhibitory gain (dimensionless). Broadcastable to ``in_size``.
        Default is ``1.``.
    theta_I : Parameter, optional
        Inhibitory threshold (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.0``.
    wEE : Parameter, optional
        E→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``12.``.
    wIE : Parameter, optional
        E→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``4.``.
    wEI : Parameter, optional
        I→E coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``13.``.
    wII : Parameter, optional
        I→I coupling strength (dimensionless). Broadcastable to ``in_size``.
        Default is ``11.``.
    r : Parameter, optional
        Refractory parameter (dimensionless) that limits maximum activation.
        Broadcastable to ``in_size``. Default is ``1.``.
    tau_aE : Parameter, optional
        Excitatory adaptation time constant with unit of time (e.g., ``100. * u.ms``).
        Controls how fast adaptation builds up and decays. Broadcastable to
        ``in_size``. Default is ``100. * u.ms``.
    tau_aI : Parameter, optional
        Inhibitory adaptation time constant with unit of time (e.g., ``80. * u.ms``).
        Controls how fast adaptation builds up and decays. Broadcastable to
        ``in_size``. Default is ``80. * u.ms``.
    b_E : Parameter, optional
        Excitatory adaptation coupling strength (dimensionless). Controls how
        strongly activity drives adaptation. Broadcastable to ``in_size``.
        Default is ``0.1``.
    b_I : Parameter, optional
        Inhibitory adaptation coupling strength (dimensionless). Controls how
        strongly activity drives adaptation. Broadcastable to ``in_size``.
        Default is ``0.08``.
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
    aE_init : Callable, optional
        Initializer for the excitatory adaptation current ``aE``. Default is
        ``braintools.init.ZeroInit()``.
    aI_init : Callable, optional
        Initializer for the inhibitory adaptation current ``aI``. Default is
        ``braintools.init.ZeroInit()``.
    method : str
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
    aE : brainstate.HiddenState
        Excitatory adaptation current (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.
    aI : brainstate.HiddenState
        Inhibitory adaptation current (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.

    Notes
    -----
    The continuous-time Wilson-Cowan equations with adaptation are

    .. math::

        \tau_E \frac{dr_E}{dt} = -r_E(t) + \bigl[1 - r\, r_E(t)\bigr]
        F_E\bigl(w_{EE} r_E(t) - w_{EI} r_I(t) + I_E(t) - a_E(t)\bigr),

    .. math::

        \tau_I \frac{dr_I}{dt} = -r_I(t) + \bigl[1 - r\, r_I(t)\bigr]
        F_I\bigl(w_{IE} r_E(t) - w_{II} r_I(t) + I_I(t) - a_I(t)\bigr),

    .. math::

        \tau_a^E \frac{da_E}{dt} = -a_E(t) + b_E\, r_E(t),

    .. math::

        \tau_a^I \frac{da_I}{dt} = -a_I(t) + b_I\, r_I(t),

    with the sigmoidal transfer function

    .. math::

        F_j(x) = \frac{1}{1 + e^{-a_j (x - \theta_j)}} - \frac{1}{1 + e^{a_j \theta_j}},\quad j \in \{E, I\}.

    Adaptation currents increase with population activity and subtract from the
    effective input, implementing spike-frequency adaptation observed in real neurons.

    References
    ----------
    .. [1] Benda, J., & Herz, A. V. (2003). A universal model for spike-frequency
           adaptation. Neural Computation, 15(11), 2523-2564.
    .. [2] Destexhe, A. (2009). Self-sustained asynchronous irregular states and
           Up-Down states. Journal of Computational Neuroscience, 27(3), 493-506.

    Examples
    --------
    >>> import brainmass
    >>> import brainunit as u
    >>> model = brainmass.WilsonCowanAdaptiveStep(
    ...     in_size=100,
    ...     tau_aE=100.*u.ms,
    ...     b_E=0.1
    ... )
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

        # Refractory parameter
        r: Parameter = 1.,

        # Adaptation parameters
        tau_aE: Parameter = 100. * u.ms,
        tau_aI: Parameter = 80. * u.ms,
        b_E: Parameter = 0.1,
        b_I: Parameter = 0.08,

        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        aE_init: Callable = braintools.init.ZeroInit(),
        aI_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size, noise_E, noise_I, rE_init, rI_init, method)

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
        self.tau_aE = Param.init(tau_aE, self.varshape)
        self.tau_aI = Param.init(tau_aI, self.varshape)
        self.b_E = Param.init(b_E, self.varshape)
        self.b_I = Param.init(b_I, self.varshape)

        # Store adaptation initializers
        self.aE_init = aE_init
        self.aI_init = aI_init

    def init_state(self, batch_size=None, **kwargs):
        """Initialize model states ``rE``, ``rI``, ``aE``, and ``aI``.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.rE = brainstate.HiddenState.init(self.rE_init, self.varshape, batch_size)
        self.rI = brainstate.HiddenState.init(self.rI_init, self.varshape, batch_size)
        self.aE = brainstate.HiddenState.init(self.aE_init, self.varshape, batch_size)
        self.aI = brainstate.HiddenState.init(self.aI_init, self.varshape, batch_size)

    def drE(self, rE, rI, aE, ext):
        """Right-hand side for the excitatory population with adaptation.

        Parameters
        ----------
        rE : array-like
            Excitatory activity (dimensionless).
        rI : array-like
            Inhibitory activity (dimensionless).
        aE : array-like
            Excitatory adaptation current (dimensionless).
        ext : array-like or scalar
            External input to E.

        Returns
        -------
        array-like
            Time derivative ``drE/dt`` with unit of ``1/time``.
        """
        wEE = self.wEE.value()
        wEI = self.wEI.value()
        r = self.r.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()

        xx = wEE * rE - wEI * rI + ext - aE
        return (-rE + (1 - r * rE) * self.F(xx, a_E, theta_E)) / tau_E

    def drI(self, rI, rE, aI, ext):
        """Right-hand side for the inhibitory population with adaptation.

        Parameters
        ----------
        rI : array-like
            Inhibitory activity (dimensionless).
        rE : array-like
            Excitatory activity (dimensionless).
        aI : array-like
            Inhibitory adaptation current (dimensionless).
        ext : array-like or scalar
            External input to I.

        Returns
        -------
        array-like
            Time derivative ``drI/dt`` with unit of ``1/time``.
        """
        wIE = self.wIE.value()
        wII = self.wII.value()
        r = self.r.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()

        xx = wIE * rE - wII * rI + ext - aI
        return (-rI + (1 - r * rI) * self.F(xx, a_I, theta_I)) / tau_I

    def daE(self, aE, rE):
        """Right-hand side for the excitatory adaptation current.

        Parameters
        ----------
        aE : array-like
            Excitatory adaptation current (dimensionless).
        rE : array-like
            Excitatory activity (dimensionless).

        Returns
        -------
        array-like
            Time derivative ``daE/dt`` with unit of ``1/time``.
        """
        tau_aE = self.tau_aE.value()
        b_E = self.b_E.value()
        return (-aE + b_E * rE) / tau_aE

    def daI(self, aI, rI):
        """Right-hand side for the inhibitory adaptation current.

        Parameters
        ----------
        aI : array-like
            Inhibitory adaptation current (dimensionless).
        rI : array-like
            Inhibitory activity (dimensionless).

        Returns
        -------
        array-like
            Time derivative ``daI/dt`` with unit of ``1/time``.
        """
        tau_aI = self.tau_aI.value()
        b_I = self.b_I.value()
        return (-aI + b_I * rI) / tau_aI

    def derivative(self, state, t, E_ext, I_ext):
        """Compute derivatives for all four state variables.

        Parameters
        ----------
        state : tuple
            Tuple of (rE, rI, aE, aI) states.
        t : float
            Current time.
        E_ext : array-like or scalar
            External input to excitatory population.
        I_ext : array-like or scalar
            External input to inhibitory population.

        Returns
        -------
        tuple
            Tuple of (drE/dt, drI/dt, daE/dt, daI/dt) derivatives.
        """
        rE, rI, aE, aI = state
        drE_dt = self.drE(rE, rI, aE, E_ext)
        drI_dt = self.drI(rI, rE, aI, I_ext)
        daE_dt = self.daE(aE, rE)
        daI_dt = self.daI(aI, rI)
        return (drE_dt, drI_dt, daE_dt, daI_dt)

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
        The method performs numerical integration for all 4 state variables
        (rE, rI, aE, aI) using the specified method (``exp_euler`` by default).
        """
        # Handle inputs
        rE_inp = 0. if rE_inp is None else rE_inp
        rI_inp = 0. if rI_inp is None else rI_inp

        # Add noise
        if self.noise_E is not None:
            rE_inp = rE_inp + self.noise_E()
        if self.noise_I is not None:
            rI_inp = rI_inp + self.noise_I()

        # Numerical integration
        if self.method == 'exp_euler':
            rE = brainstate.nn.exp_euler_step(self.drE, self.rE.value, self.rI.value, self.aE.value, rE_inp)
            rI = brainstate.nn.exp_euler_step(self.drI, self.rI.value, self.rE.value, self.aI.value, rI_inp)
            aE = brainstate.nn.exp_euler_step(self.daE, self.aE.value, self.rE.value)
            aI = brainstate.nn.exp_euler_step(self.daI, self.aI.value, self.rI.value)
        else:
            method = getattr(braintools.quad, f'ode_{self.method}_step')
            t = brainstate.environ.get('t', 0 * u.ms)
            rE, rI, aE, aI = method(self.derivative, (self.rE.value, self.rI.value, self.aE.value, self.aI.value), t, rE_inp, rI_inp)

        # Update states
        self.rE.value = rE
        self.rI.value = rI
        self.aE.value = aE
        self.aI.value = aI

        return rE


class WilsonCowanThreePopBase(brainstate.nn.Dynamics):
    r"""Abstract base class for three-population Wilson-Cowan models.

    This base class extends the Wilson-Cowan framework to three populations:
    excitatory (E), inhibitory (I), and modulatory (M). The modulatory population
    can represent cholinergic/noradrenergic modulation, VIP interneurons, or
    long-range feedback.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E, I, M). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    noise_E : Noise or None, optional
        Additive noise process for the excitatory population. Default is ``None``.
    noise_I : Noise or None, optional
        Additive noise process for the inhibitory population. Default is ``None``.
    noise_M : Noise or None, optional
        Additive noise process for the modulatory population. Default is ``None``.
    rE_init : Callable, optional
        Initializer for the excitatory state ``rE``. Default is
        ``braintools.init.ZeroInit()``.
    rI_init : Callable, optional
        Initializer for the inhibitory state ``rI``. Default is
        ``braintools.init.ZeroInit()``.
    rM_init : Callable, optional
        Initializer for the modulatory state ``rM``. Default is
        ``braintools.init.ZeroInit()``.
    method : str, optional
        The numerical integration method to use. One of ``'exp_euler'``,
        ``'euler'``, ``'rk2'``, or ``'rk4'``, that is implemented in
        ``braintools.quad``. Default is ``'exp_euler'``.

    Attributes
    ----------
    rE : brainstate.HiddenState
        Excitatory population activity (dimensionless).
    rI : brainstate.HiddenState
        Inhibitory population activity (dimensionless).
    rM : brainstate.HiddenState
        Modulatory population activity (dimensionless).

    Notes
    -----
    Subclasses must implement:

    - ``drE(rE, rI, rM, ext)``: Right-hand side for excitatory population
    - ``drI(rI, rE, rM, ext)``: Right-hand side for inhibitory population
    - ``drM(rM, rE, rI, ext)``: Right-hand side for modulatory population
    - ``F(x, *args)`` (optional): Transfer function
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # noise parameters
        noise_E: Noise = None,
        noise_I: Noise = None,
        noise_M: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        rM_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        # Validate noise parameters
        assert isinstance(noise_E, Noise) or noise_E is None, \
            "noise_E must be a Noise instance or None."
        assert isinstance(noise_I, Noise) or noise_I is None, \
            "noise_I must be a Noise instance or None."
        assert isinstance(noise_M, Noise) or noise_M is None, \
            "noise_M must be a Noise instance or None."
        assert callable(rE_init), "rE_init must be a callable."
        assert callable(rI_init), "rI_init must be a callable."
        assert callable(rM_init), "rM_init must be a callable."

        self.rE_init = rE_init
        self.rI_init = rI_init
        self.rM_init = rM_init
        self.noise_E = noise_E
        self.noise_I = noise_I
        self.noise_M = noise_M
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Initialize model states ``rE``, ``rI``, and ``rM``.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.rE = brainstate.HiddenState.init(self.rE_init, self.varshape, batch_size)
        self.rI = brainstate.HiddenState.init(self.rI_init, self.varshape, batch_size)
        self.rM = brainstate.HiddenState.init(self.rM_init, self.varshape, batch_size)

    def F(self, x, a, theta):
        """Sigmoidal transfer function."""
        return jax.nn.sigmoid(a * (x - theta)) - jax.nn.sigmoid(-a * theta)

    def drE(self, rE, rI, rM, ext):
        """Right-hand side for the excitatory population.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def drI(self, rI, rE, rM, ext):
        """Right-hand side for the inhibitory population.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def drM(self, rM, rE, rI, ext):
        """Right-hand side for the modulatory population.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def derivative(self, state, t, E_ext, I_ext, M_ext):
        """Compute derivatives for all three populations.

        Parameters
        ----------
        state : tuple
            Tuple of (rE, rI, rM) states.
        t : float
            Current time.
        E_ext : array-like or scalar
            External input to excitatory population.
        I_ext : array-like or scalar
            External input to inhibitory population.
        M_ext : array-like or scalar
            External input to modulatory population.

        Returns
        -------
        tuple
            Tuple of (drE/dt, drI/dt, drM/dt) derivatives.
        """
        rE, rI, rM = state
        drE_dt = self.drE(rE, rI, rM, E_ext)
        drI_dt = self.drI(rI, rE, rM, I_ext)
        drM_dt = self.drM(rM, rE, rI, M_ext)
        return (drE_dt, drI_dt, drM_dt)

    def update(self, rE_inp=None, rI_inp=None, rM_inp=None):
        """Advance the system by one time step.

        Parameters
        ----------
        rE_inp : array-like or scalar or None, optional
            External input to the excitatory population. If ``None``, treated
            as zero. If ``noise_E`` is set, its output is added.
        rI_inp : array-like or scalar or None, optional
            External input to the inhibitory population. If ``None``, treated
            as zero. If ``noise_I`` is set, its output is added.
        rM_inp : array-like or scalar or None, optional
            External input to the modulatory population. If ``None``, treated
            as zero. If ``noise_M`` is set, its output is added.

        Returns
        -------
        array-like
            The updated excitatory activity ``rE`` with the same shape as the
            internal state.

        Notes
        -----
        The method performs numerical integration using the specified method
        (``exp_euler`` by default) and updates the internal states ``rE``,
        ``rI``, and ``rM`` in-place.
        """
        # Handle inputs
        rE_inp = 0. if rE_inp is None else rE_inp
        rI_inp = 0. if rI_inp is None else rI_inp
        rM_inp = 0. if rM_inp is None else rM_inp

        # Add noise
        if self.noise_E is not None:
            rE_inp = rE_inp + self.noise_E()
        if self.noise_I is not None:
            rI_inp = rI_inp + self.noise_I()
        if self.noise_M is not None:
            rM_inp = rM_inp + self.noise_M()

        # Numerical integration
        if self.method == 'exp_euler':
            rE = brainstate.nn.exp_euler_step(self.drE, self.rE.value, self.rI.value, self.rM.value, rE_inp)
            rI = brainstate.nn.exp_euler_step(self.drI, self.rI.value, self.rE.value, self.rM.value, rI_inp)
            rM = brainstate.nn.exp_euler_step(self.drM, self.rM.value, self.rE.value, self.rI.value, rM_inp)
        else:
            method = getattr(braintools.quad, f'ode_{self.method}_step')
            t = brainstate.environ.get('t', 0 * u.ms)
            rE, rI, rM = method(self.derivative, (self.rE.value, self.rI.value, self.rM.value), t, rE_inp, rI_inp, rM_inp)

        # Update states
        self.rE.value = rE
        self.rI.value = rI
        self.rM.value = rM

        return rE


class WilsonCowanThreePopulationStep(WilsonCowanThreePopBase):
    r"""Three-population Wilson-Cowan model with modulatory neurons.

    This variant extends the standard Wilson-Cowan model to include a third
    modulatory (M) population that can represent neuromodulation, attention,
    arousal, or VIP interneurons. The modulatory population receives input from
    E and I, and projects back to modulate both populations.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of each population (E, I, M). Can be an int, a tuple of
        ints, or any size compatible with ``brainstate``.
    tau_E : Parameter, optional
        Excitatory time constant with unit of time. Default is ``1. * u.ms``.
    a_E : Parameter, optional
        Excitatory gain (dimensionless). Default is ``1.2``.
    theta_E : Parameter, optional
        Excitatory threshold (dimensionless). Default is ``2.8``.
    tau_I : Parameter, optional
        Inhibitory time constant with unit of time. Default is ``1. * u.ms``.
    a_I : Parameter, optional
        Inhibitory gain (dimensionless). Default is ``1.``.
    theta_I : Parameter, optional
        Inhibitory threshold (dimensionless). Default is ``4.0``.
    tau_M : Parameter, optional
        Modulatory time constant with unit of time. Typically slower than E/I.
        Default is ``2. * u.ms``.
    a_M : Parameter, optional
        Modulatory gain (dimensionless). Default is ``1.``.
    theta_M : Parameter, optional
        Modulatory threshold (dimensionless). Default is ``3.5``.
    wEE : Parameter, optional
        E→E coupling strength. Default is ``12.``.
    wIE : Parameter, optional
        E→I coupling strength. Default is ``4.``.
    wEI : Parameter, optional
        I→E coupling strength. Default is ``13.``.
    wII : Parameter, optional
        I→I coupling strength. Default is ``11.``.
    wME : Parameter, optional
        E→M coupling strength. Default is ``8.``.
    wMI : Parameter, optional
        I→M coupling strength. Default is ``6.``.
    wEM : Parameter, optional
        M→E coupling strength (modulatory excitation). Default is ``4.``.
    wIM : Parameter, optional
        M→I coupling strength (modulatory excitation). Default is ``2.``.
    wMM : Parameter, optional
        M→M self-coupling strength. Default is ``2.``.
    r : Parameter, optional
        Refractory parameter for all populations. Default is ``1.``.
    noise_E : Noise or None, optional
        Additive noise for excitatory population. Default is ``None``.
    noise_I : Noise or None, optional
        Additive noise for inhibitory population. Default is ``None``.
    noise_M : Noise or None, optional
        Additive noise for modulatory population. Default is ``None``.
    rE_init : Callable, optional
        Initializer for ``rE``. Default is ``braintools.init.ZeroInit()``.
    rI_init : Callable, optional
        Initializer for ``rI``. Default is ``braintools.init.ZeroInit()``.
    rM_init : Callable, optional
        Initializer for ``rM``. Default is ``braintools.init.ZeroInit()``.
    method : str
        Numerical integration method. Default is ``'exp_euler'``.

    Attributes
    ----------
    rE : brainstate.HiddenState
        Excitatory population activity (dimensionless).
    rI : brainstate.HiddenState
        Inhibitory population activity (dimensionless).
    rM : brainstate.HiddenState
        Modulatory population activity (dimensionless).

    Notes
    -----
    The continuous-time three-population Wilson-Cowan equations are

    .. math::

        \tau_E \frac{dr_E}{dt} = -r_E(t) + \bigl[1 - r\, r_E(t)\bigr]
        F_E\bigl(w_{EE} r_E(t) - w_{EI} r_I(t) + w_{EM} r_M(t) + I_E(t)\bigr),

    .. math::

        \tau_I \frac{dr_I}{dt} = -r_I(t) + \bigl[1 - r\, r_I(t)\bigr]
        F_I\bigl(w_{IE} r_E(t) - w_{II} r_I(t) + w_{IM} r_M(t) + I_I(t)\bigr),

    .. math::

        \tau_M \frac{dr_M}{dt} = -r_M(t) + \bigl[1 - r\, r_M(t)\bigr]
        F_M\bigl(w_{ME} r_E(t) - w_{MI} r_I(t) + w_{MM} r_M(t) + I_M(t)\bigr),

    with the sigmoidal transfer function

    .. math::

        F_j(x) = \frac{1}{1 + e^{-a_j (x - \theta_j)}} - \frac{1}{1 + e^{a_j \theta_j}},\quad j \in \{E, I, M\}.

    The modulatory population provides excitatory modulation to both E and I
    populations, modeling attention, arousal, or neuromodulatory effects.

    References
    ----------
    .. [1] Deco, G., et al. (2014). How local excitation-inhibition ratio impacts
           the whole brain dynamics. Journal of Neuroscience, 34(23), 7886-7898.
    .. [2] Pfeffer, C. K., et al. (2013). Inhibition of inhibition in visual cortex.
           Nature Neuroscience, 16(8), 1068-1076.

    Examples
    --------
    >>> import brainmass
    >>> import brainunit as u
    >>> model = brainmass.WilsonCowanThreePopulationStep(
    ...     in_size=100,
    ...     tau_M=2.*u.ms,
    ...     wEM=4.0
    ... )
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

        # Modulatory parameters
        tau_M: Parameter = 2. * u.ms,
        a_M: Parameter = 1.,
        theta_M: Parameter = 3.5,

        # E-I connection parameters
        wEE: Parameter = 12.,
        wIE: Parameter = 4.,
        wEI: Parameter = 13.,
        wII: Parameter = 11.,

        # M connection parameters
        wME: Parameter = 8.,  # E→M
        wMI: Parameter = 6.,  # I→M
        wEM: Parameter = 4.,  # M→E (modulatory excitation)
        wIM: Parameter = 2.,  # M→I (modulatory excitation)
        wMM: Parameter = 2.,  # M→M (self-excitation)

        # Refractory parameter
        r: Parameter = 1.,

        # noise
        noise_E: Noise = None,
        noise_I: Noise = None,
        noise_M: Noise = None,

        # initialization
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        rM_init: Callable = braintools.init.ZeroInit(),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size, noise_E, noise_I, noise_M, rE_init, rI_init, rM_init, method)

        self.a_E = Param.init(a_E, self.varshape)
        self.a_I = Param.init(a_I, self.varshape)
        self.a_M = Param.init(a_M, self.varshape)
        self.tau_E = Param.init(tau_E, self.varshape)
        self.tau_I = Param.init(tau_I, self.varshape)
        self.tau_M = Param.init(tau_M, self.varshape)
        self.theta_E = Param.init(theta_E, self.varshape)
        self.theta_I = Param.init(theta_I, self.varshape)
        self.theta_M = Param.init(theta_M, self.varshape)
        self.wEE = Param.init(wEE, self.varshape)
        self.wIE = Param.init(wIE, self.varshape)
        self.wEI = Param.init(wEI, self.varshape)
        self.wII = Param.init(wII, self.varshape)
        self.wME = Param.init(wME, self.varshape)
        self.wMI = Param.init(wMI, self.varshape)
        self.wEM = Param.init(wEM, self.varshape)
        self.wIM = Param.init(wIM, self.varshape)
        self.wMM = Param.init(wMM, self.varshape)
        self.r = Param.init(r, self.varshape)

    def drE(self, rE, rI, rM, ext):
        wEE = self.wEE.value()
        wEI = self.wEI.value()
        wEM = self.wEM.value()
        r = self.r.value()
        a_E = self.a_E.value()
        theta_E = self.theta_E.value()
        tau_E = self.tau_E.value()

        xx = wEE * rE - wEI * rI + wEM * rM + ext
        return (-rE + (1 - r * rE) * self.F(xx, a_E, theta_E)) / tau_E

    def drI(self, rI, rE, rM, ext):
        wIE = self.wIE.value()
        wII = self.wII.value()
        wIM = self.wIM.value()
        r = self.r.value()
        a_I = self.a_I.value()
        theta_I = self.theta_I.value()
        tau_I = self.tau_I.value()

        xx = wIE * rE - wII * rI + wIM * rM + ext
        return (-rI + (1 - r * rI) * self.F(xx, a_I, theta_I)) / tau_I

    def drM(self, rM, rE, rI, ext):
        wME = self.wME.value()
        wMI = self.wMI.value()
        wMM = self.wMM.value()
        r = self.r.value()
        a_M = self.a_M.value()
        theta_M = self.theta_M.value()
        tau_M = self.tau_M.value()

        xx = wME * rE - wMI * rI + wMM * rM + ext
        return (-rM + (1 - r * rM) * self.F(xx, a_M, theta_M)) / tau_M


class AdditiveConn(brainstate.nn.Module):
    def __init__(
        self,
        model,
        w_init: Callable = braintools.init.KaimingNormal(),
        b_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__()

        self.model = model
        self.linear = brainstate.nn.Linear(self.model.in_size, self.model.out_size, w_init=w_init, b_init=b_init)

    def update(self, *args, **kwargs):
        return self.linear(self.model.rE.value)


class DelayedAdditiveConn(brainstate.nn.Module):
    def __init__(
        self,
        model,
        delay_time: Initializer,
        delay_init: Initializer = braintools.init.ZeroInit(),
        w_init: Callable = braintools.init.KaimingNormal(),
        k: Parameter = 1.0,
    ):
        super().__init__()

        n_hidden = model.varshape[0]
        delay_time = braintools.init.param(delay_time, (n_hidden, n_hidden))
        neuron_idx = np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
        self.prefetch = model.prefetch_delay('rE', delay_time, neuron_idx, init=delay_init)
        self.weights = Param(braintools.init.param(w_init, (n_hidden, n_hidden)))
        self.k = Param.init(k)

    def update(self, *args, **kwargs):
        delayed = self.prefetch()
        return additive_coupling(delayed, self.weights.value(), self.k.value())


class WilsonCowanSeqLayer(brainstate.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        wc_cls: type = WilsonCowanNoSaturationStep,
        delay_init: Callable = braintools.init.ZeroInit(),
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        delay: Optional[Initializer] = None,
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
        inp_w_init: Initializer = braintools.init.KaimingNormal(),
        inp_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
        **wc_kwargs
    ):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.rec_w_init = rec_w_init
        self.rec_b_init = rec_b_init
        self.inp_w_init = inp_w_init
        self.inp_b_init = inp_b_init

        self.dynamics = wc_cls(n_hidden, **wc_kwargs, rE_init=rE_init, rI_init=rI_init)
        self.i2h = brainstate.nn.Linear(n_input, n_hidden, w_init=inp_w_init, b_init=inp_b_init)
        if delay is None:
            self.h2h = AdditiveConn(self.dynamics, w_init=rec_w_init, b_init=rec_b_init)
        else:
            self.h2h = DelayedAdditiveConn(self.dynamics, delay, delay_init=delay_init, w_init=rec_w_init)

    def update(self, inputs, record_state: bool = False):
        def step(inp):
            out = self.dynamics(inp + self.h2h())
            st = dict(rE=self.dynamics.rE.value, rI=self.dynamics.rI.value)
            return (st, out) if record_state else out

        return brainstate.transform.for_loop(step, self.i2h(inputs))


class WilsonCowanSeqNetwork(brainstate.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int | Sequence[int],
        n_output: int,
        wc_cls: type = WilsonCowanNoSaturationStep,
        delay_init: Callable = braintools.init.ZeroInit(),
        rE_init: Callable = braintools.init.ZeroInit(),
        rI_init: Callable = braintools.init.ZeroInit(),
        delay: Optional[Initializer] = None,
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
        inp_w_init: Initializer = braintools.init.KaimingNormal(),
        inp_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
        **wc_kwargs
    ):
        super().__init__()

        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        assert isinstance(n_hidden, (list, tuple)), 'n_hidden must be int or sequence of int.'

        self.layers = []
        for hidden in n_hidden:
            layer = WilsonCowanSeqLayer(
                n_input=n_input,
                n_hidden=hidden,
                wc_cls=wc_cls,
                delay=delay,
                rE_init=rE_init,
                rI_init=rI_init,
                delay_init=delay_init,
                rec_w_init=rec_w_init,
                rec_b_init=rec_b_init,
                inp_w_init=inp_w_init,
                inp_b_init=inp_b_init,
                **wc_kwargs
            )
            self.layers.append(layer)
            n_input = hidden  # next layer input size is current layer hidden size

        self.h2o = brainstate.nn.Linear(n_input, n_output, w_init=inp_w_init, b_init=inp_b_init)

    def update(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        output = self.h2o(self.layers[-1].dynamics.rE.value)
        return output

    def hidden_activation(self, inputs):
        x = inputs
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs
