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


class HORNStep(Dynamics):
    r"""Harmonic oscillator recurrent networks (HORNs) with one-step dynamics update.

    This implementation models neural dynamics as a driven damped harmonic oscillator
    where each network unit evolves according to a second-order ODE. The continuous-time
    dynamics are discretized using symplectic Euler integration.

    The continuous-time formulation for each oscillator is:

    .. math::
        \ddot{x}(t) + 2\gamma\dot{x}(t) + \omega^2 x(t) = \alpha \sigma\left(I(t) + F(x(t), \dot{x}(t))\right),

    where :math:`x` represents the position (activation state), :math:`\dot{x}` is the velocity,
    :math:`\omega` is the natural frequency, :math:`\gamma` is the damping coefficient,
    :math:`\alpha` is the excitability factor, :math:`\sigma` is a nonlinear activation (tanh),
    :math:`I(t)` is the external input, and :math:`F` denotes recurrent feedback.

    The discrete-time update equations for a HORN network of n units at time step t are:

    .. math::
        \begin{aligned}
        \mathbf{y}_{t+1} &= \mathbf{y}_{t} + h \left(\boldsymbol{\alpha} \cdot \tanh\left(\frac{1}{\sqrt{n}}\mathbf{I}_{t+1}^{\mathrm{rec}} + \mathbf{I}_{t+1}^{\mathrm{ext}}\right) - 2\boldsymbol{\gamma} \cdot \mathbf{y}_{t} - \boldsymbol{\omega}^{2} \cdot \mathbf{x}_{t}\right), \\
        \mathbf{x}_{t+1} &= \mathbf{x}_{t} + h \cdot \mathbf{y}_{t+1},
        \end{aligned}

    where boldface symbols denote vectors, :math:`h` is the integration step size,
    :math:`\boldsymbol{\omega}`, :math:`\boldsymbol{\gamma}`, and :math:`\boldsymbol{\alpha}`
    are the natural frequencies, damping factors, and excitability factors for each unit.
    Initial conditions are :math:`\mathbf{x}_0 = \mathbf{y}_0 = 0` unless specified otherwise.

    The input currents are defined as:

    .. math::
        \begin{aligned}
        \mathbf{I}_{t+1}^{\mathrm{rec}} &= \mathbf{W}^{hh} \mathbf{y}_t + \mathbf{b}^{hh} + \mathbf{v} \cdot \mathbf{x}_t, \\
        \mathbf{I}_{t+1}^{\mathrm{ext}} &= \mathbf{W}^{ih} \mathbf{s}_{t+1} + \mathbf{b}^{ih},
        \end{aligned}

    where :math:`\mathbf{I}^{\mathrm{rec}}` and :math:`\mathbf{I}^{\mathrm{ext}}` denote
    recurrent and external inputs, respectively. Here :math:`\mathbf{W}^{ih}, \mathbf{b}^{ih}`
    are the input weights and biases, :math:`\mathbf{W}^{hh}, \mathbf{b}^{hh}` are the
    hidden (recurrent) weights and biases, :math:`\mathbf{v}` is the amplitude feedback vector,
    and :math:`\mathbf{s} = (s_1, \ldots, s_T)` is the external input sequence.

    Parameters
    ----------
    in_size : int or tuple of int
        Spatial shape for parameter/state broadcasting. Specifies the dimensionality
        of the harmonic oscillator network.
    alpha : Parameter, optional
        Excitability factor (dimensionless). Controls the gain of the input forcing term.
        Broadcastable to ``in_size``. Default is ``0.04``.
    omega : Parameter, optional
        Natural frequency (radians per time step, dimensionless). Determines the
        oscillation frequency of each unit. Default is ``2π/28 ≈ 0.224``.
    gamma : Parameter, optional
        Damping coefficient (dimensionless). Controls the rate of energy dissipation.
        Broadcastable to ``in_size``. Default is ``0.01``.
    v : Parameter, optional
        Amplitude feedback coefficient (dimensionless). Provides position-based
        self-feedback in the recurrent input. Broadcastable to ``in_size``.
        Default is ``0.0`` (no amplitude feedback).
    x_init : Initializer, optional
        Initializer for the position state :math:`\mathbf{x}`.
        Default is ``braintools.init.Constant(0.0)``.
    y_init : Initializer, optional
        Initializer for the velocity state :math:`\mathbf{y}`.
        Default is ``braintools.init.Constant(0.0)``.

    Attributes
    ----------
    x : brainstate.HiddenState
        Position (activation) state vector. Shape equals ``in_size`` after ``init_state``.
        Represents the displacement from equilibrium for each oscillator.
    y : brainstate.HiddenState
        Velocity state vector. Shape equals ``in_size`` after ``init_state``.
        Represents the time derivative of the position state.

    Notes
    -----
    - **Integration method**: This implementation uses symplectic (semi-implicit) Euler
      integration, where the velocity :math:`\mathbf{y}_{t+1}` is computed first, then
      used to update the position :math:`\mathbf{x}_{t+1}`. This preserves energy
      characteristics better than standard Euler integration.

    - **Units and dimensionality**: All parameters and states are dimensionless in this
      implementation. The step size ``h`` should be chosen appropriately relative to
      the natural frequencies ``omega`` to ensure numerical stability.

    - **Recurrent structure**: The ``recurrent_fn`` callable enables flexible recurrent
      connectivity patterns. When used with ``HORNSeqLayer``, this is typically a
      linear transformation or delayed coupling operator.

    - **Feedback mechanisms**: Two feedback pathways exist:
      - Velocity feedback via ``recurrent_fn(y)``
      - Amplitude (position) feedback via ``v * x``

    References
    ----------
    .. [1] Rusch T K, Mishra S. Coupled Oscillatory Recurrent Neural Network (coRNN):
       An accurate and (gradient) stable architecture for learning long time dependencies.
       International Conference on Learning Representations (ICLR), 2021.
    .. [2] Didier Auroux, Jacques Blum. A nudging-based data assimilation method:
       the Back and Forth Nudging (BFN) algorithm. Nonlinear Processes in Geophysics,
       2008, 15(2): 305-319.

    """

    def __init__(
        self,
        in_size: int,
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # Amplitude feedback
        x_init: Initializer = braintools.init.Constant(0.0),
        y_init: Initializer = braintools.init.Constant(0.0),
    ):
        super().__init__(in_size)

        self.alpha = Param.init(alpha, self.in_size)
        self.omega = Param.init(omega, self.in_size)
        self.gamma = Param.init(gamma, self.in_size)
        self.v = Param.init(v, self.in_size)
        self.x_init = x_init
        self.y_init = y_init

    def init_state(self, *args, **kwargs):
        """Initialize position and velocity states for the HORN oscillators.

        Creates ``HiddenState`` containers for the position (``x``) and
        velocity (``y``) states using their respective initializers.
        """
        self.x = brainstate.HiddenState(self.x_init(self.in_size))
        self.y = brainstate.HiddenState(self.y_init(self.in_size))

    def update(self, inputs):
        """Perform one step of HORN dynamics using symplectic Euler integration.

        Updates the internal position (``x``) and velocity (``y``) states according
        to the driven damped harmonic oscillator equations. The velocity is updated
        first using the current position, then the new velocity is used to update
        the position (symplectic/semi-implicit Euler method).

        Parameters
        ----------
        inputs : array-like
            External input to the oscillators at the current time step. This should
            contain the combined external and recurrent inputs, shape-compatible with
            ``in_size``. Units are dimensionless.

        Returns
        -------
        array-like
            Updated position state ``x`` after one integration step. Shape matches
            ``in_size``.

        Notes
        -----
        The update follows this sequence:

        1. Compute total input: external + recurrent feedback
        2. Update velocity: ``y_{t+1} = y_t + h * (alpha*tanh(input) - omega^2*x_t - 2*gamma*y_t)``
        3. Update position: ``x_{t+1} = x_t + h * y_{t+1}``

        The symplectic integration scheme provides better energy conservation properties
        compared to standard forward Euler integration.
        """

        # current states
        y = self.y.value
        x = self.x.value

        # current parameters
        v = self.v.value()
        omega_factor = self.omega.value() ** 2
        gamma_factor = 2.0 * self.gamma.value()
        alpha = self.alpha.value()
        dt = brainstate.environ.get_dt()

        # 1. integrate y_t
        # external input + recurrent input from network
        y_t = y + dt * (
            alpha * jnp.tanh(inputs + v * x)  # input (forcing) on y_t
            - omega_factor * x  # natural frequency term
            - gamma_factor * y  # damping term
        ) / u.ms

        # 2. integrate x_t with updated y_t, no input here
        x_t = x + dt * y_t / u.ms

        self.x.value = x_t
        self.y.value = y_t
        return x_t
