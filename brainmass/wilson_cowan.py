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
