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

from .coupling import AdditiveCoupling, additive_coupling, LaplacianConnParam
from .typing import Initializer, Parameter

__all__ = [
    'HORNStep',
    'HORNSeqLayer',
    'HORNSeqNetwork',
]


def zeros(x):
    return 0.


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
    state_init : Initializer, optional
        Initializer for both position and velocity states :math:`\mathbf{x}` and :math:`\mathbf{y}`.
        Default is ``braintools.init.ZeroInit()``.

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
        state_init: Initializer = braintools.init.ZeroInit(),
    ):
        super().__init__(in_size)

        self.alpha = Param.init(alpha, self.in_size)
        self.omega = Param.init(omega, self.in_size)
        self.gamma = Param.init(gamma, self.in_size)
        self.v = Param.init(v, self.in_size)
        self.state_init = state_init

    def init_state(self, *args, **kwargs):
        """Initialize position and velocity states for the HORN oscillators.

        Creates ``HiddenState`` containers for both the position (``x``) and
        velocity (``y``) states using the initializer specified during construction.
        """
        self.x = brainstate.HiddenState(self.state_init(self.in_size))
        self.y = brainstate.HiddenState(self.state_init(self.in_size))

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


class AdditiveConn(Module):
    def __init__(
        self,
        model: Module,
        w_init: Callable = braintools.init.KaimingNormal(),
        b_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__()

        self.model = model
        self.linear = brainstate.nn.Linear(self.model.in_size, self.model.out_size, w_init=w_init, b_init=b_init)

    def update_tr(self, *args, **kwargs):
        return 0.

    def update(self, *args, **kwargs):
        return self.linear(self.model.y.value)


class DelayedAdditiveConn(Module):
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
        self.weights = Param(braintools.init.param(w_init, (n_hidden, n_hidden)))
        self.k = Param.init(k)

    def update_tr(self, *args, **kwargs):
        return 0.

    def update(self, *args, **kwargs):
        delayed = self.prefetch()
        return additive_coupling(delayed, self.weights.value(), self.k.value())


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


class HORN_TR(Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # feedback

        # state initialization
        state_init: Callable = braintools.init.ZeroInit(),

        # time resolution
        tr: u.Quantity = 1. * u.ms,

        # input connections
        inp_w_init: Initializer = braintools.init.KaimingNormal(),
        inp_b_init: Optional[Initializer] = braintools.init.ZeroInit(),

        # recurrent connections
        delay: Optional[Initializer] = None,
        rec_type: str = 'additive',
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
        delay_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden

        # dynamics
        self.horn = HORNStep(n_hidden, alpha=alpha, omega=omega, gamma=gamma, v=v, state_init=state_init)

        # input-to-hidden
        self.i2h = brainstate.nn.Linear(n_input, n_hidden, w_init=inp_w_init, b_init=inp_b_init)

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

        return brainstate.transform.for_loop(step, self.i2h(inputs))


class HORNSeqLayer(Module):
    r"""Sequential layer wrapper for HORN dynamics with input and recurrent connections.

    This layer combines a ``HORNStep`` dynamics model with trainable input-to-hidden
    and hidden-to-hidden (recurrent) linear transformations to process sequential data.
    It supports optional synaptic delays in recurrent connections and processes entire
    input sequences using a for-loop scan operation.

    The layer computes:

    .. math::
        \begin{aligned}
        \mathbf{I}_{t+1}^{\mathrm{ext}} &= \mathbf{W}^{ih} \mathbf{s}_{t+1} + \mathbf{b}^{ih}, \\
        \mathbf{I}_{t+1}^{\mathrm{rec}} &= \mathbf{W}^{hh} \mathbf{y}_t + \mathbf{b}^{hh} + \mathbf{v} \cdot \mathbf{x}_t, \\
        \mathbf{out}_t &= \text{HORNStep}(\mathbf{I}_{t}^{\mathrm{ext}} + \mathbf{I}_{t}^{\mathrm{rec}}),
        \end{aligned}

    where :math:`\mathbf{s}` is the input sequence, :math:`\mathbf{W}^{ih}, \mathbf{b}^{ih}`
    are input weights/biases, :math:`\mathbf{W}^{hh}, \mathbf{b}^{hh}` are recurrent
    weights/biases, and the output is the position state :math:`\mathbf{x}` from the
    HORN dynamics.

    Parameters
    ----------
    n_input : int
        Dimensionality of input features at each time step.
    n_hidden : int
        Dimensionality of the hidden state (number of HORN oscillators).
    alpha : Parameter, optional
        Excitability factor for HORN dynamics (dimensionless).
        Broadcastable to ``n_hidden``. Default is ``0.04``.
    omega : Parameter, optional
        Natural frequency for HORN dynamics (radians per time step).
        Default is ``2π/28 ≈ 0.224``.
    gamma : Parameter, optional
        Damping coefficient for HORN dynamics (dimensionless).
        Default is ``0.01``.
    v : Parameter, optional
        Amplitude feedback coefficient (dimensionless). Default is ``0.0``.
    state_init : Callable, optional
        Initializer for HORN state variables (position and velocity).
        Default is ``braintools.init.ZeroInit()``.
    delay : Initializer, optional
        Synaptic delay configuration for recurrent connections. If provided,
        creates a ``(n_hidden, n_hidden)`` delay matrix where each entry specifies
        the number of time steps to delay that connection. When ``None``, no delays
        are used. Default is ``None``.
    rec_w_init : Initializer, optional
        Initializer for recurrent weight matrix :math:`\mathbf{W}^{hh}`.
        Default is ``braintools.init.KaimingNormal()``.
    rec_b_init : Initializer or None, optional
        Initializer for recurrent bias vector :math:`\mathbf{b}^{hh}`.
        If ``None``, no recurrent bias is used. Default is ``braintools.init.ZeroInit()``.
    inp_w_init : Initializer, optional
        Initializer for input weight matrix :math:`\mathbf{W}^{ih}`.
        Default is ``braintools.init.KaimingNormal()``.
    inp_b_init : Initializer or None, optional
        Initializer for input bias vector :math:`\mathbf{b}^{ih}`.
        If ``None``, no input bias is used. Default is ``braintools.init.ZeroInit()``.

    Attributes
    ----------
    horn : HORNStep
        The underlying HORN dynamics model instance.
    i2h : brainstate.nn.Linear
        Input-to-hidden linear transformation with shape ``(n_input, n_hidden)``.
    h2h : brainstate.nn.Linear or AdditiveCoupling
        Hidden-to-hidden recurrent transformation. When ``delay`` is ``None``, this is
        a standard ``Linear`` layer with shape ``(n_hidden, n_hidden)``. When ``delay``
        is specified, this is an ``AdditiveCoupling`` operator that applies delayed
        synaptic connections.

    Notes
    -----
    - **Sequential processing**: The ``update`` method uses ``brainstate.transform.for_loop``
      to scan over the time dimension of the input sequence, maintaining the HORN
      hidden states across time steps.

    - **Delay implementation**: When synaptic delays are specified via the ``delay``
      parameter, the layer uses an ``AdditiveCoupling`` operator that prefetches delayed
      values of the velocity state :math:`\mathbf{y}` from a ring buffer. This enables
      modeling of axonal/dendritic delays in recurrent connections.

    - **State recording**: The ``record_state`` parameter in the ``update`` method allows
      recording both position and velocity states at each time step, returning them
      alongside the output sequence.

    - **Parameter sharing**: All HORN dynamics parameters (``alpha``, ``omega``, ``gamma``,
      ``v``, ``h``) are shared across all oscillators in the layer, though they can be
      initialized as arrays to provide per-oscillator heterogeneity.

    See Also
    --------
    HORNStep : Single-step HORN dynamics
    HORNSeqNetwork : Multi-layer HORN network
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # feedback
        delay_init: Callable = braintools.init.ZeroInit(),
        state_init: Callable = braintools.init.ZeroInit(),
        delay: Optional[Initializer] = None,
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

        self.horn = HORNStep(n_hidden, alpha=alpha, omega=omega, gamma=gamma, v=v, state_init=state_init)
        self.i2h = brainstate.nn.Linear(n_input, n_hidden, w_init=inp_w_init, b_init=inp_b_init)
        if delay is None:
            self.h2h = AdditiveConn(self.horn, w_init=rec_w_init, b_init=rec_b_init)
        else:
            self.h2h = DelayedAdditiveConn(self.horn, delay, delay_init=delay_init, w_init=rec_w_init)

    def update(self, inputs, record_state: bool = False):
        def step(inp):
            out = self.horn(inp + self.h2h())
            st = dict(
                x=self.horn.x.value,
                y=self.horn.y.value,
            )
            return (st, out) if record_state else out

        output = brainstate.transform.for_loop(step, self.i2h(inputs))
        return output


class HORNSeqNetwork(Module):
    r"""Multi-layer HORN network for sequential processing tasks.

    This network stacks multiple ``HORNSeqLayer`` instances to create a deep
    recurrent architecture based on harmonic oscillator dynamics. It is designed
    for sequence-to-sequence or sequence-to-label tasks such as time series
    classification, sequence generation, or temporal pattern recognition.

    The network processes input sequences through L layers of HORN dynamics:

    .. math::
        \begin{aligned}
        \mathbf{h}^{(0)}_t &= \mathbf{s}_t, \\
        \mathbf{h}^{(\ell)}_t &= \text{HORNSeqLayer}^{(\ell)}(\mathbf{h}^{(\ell-1)}), \quad \ell = 1, \ldots, L, \\
        \mathbf{o}_T &= \mathbf{W}^{ho} \mathbf{x}^{(L)}_T + \mathbf{b}^{ho},
        \end{aligned}

    where :math:`\mathbf{s}` is the input sequence, :math:`\mathbf{h}^{(\ell)}`
    denotes the hidden state sequence at layer :math:`\ell`, :math:`\mathbf{x}^{(L)}_T`
    is the final position state from the last layer at the final time step, and
    :math:`\mathbf{o}_T` is the output.

    Parameters
    ----------
    n_input : int
        Dimensionality of input features at each time step.
    n_hidden : int or sequence of int
        Hidden state dimensionality for each layer. If an integer, creates a single
        hidden layer with that size. If a sequence (list or tuple), creates multiple
        layers where each element specifies the hidden size for that layer.
    n_output : int
        Dimensionality of the final output.
    alpha : Parameter, optional
        Excitability factor for HORN dynamics (dimensionless), shared across all layers.
        Default is ``0.04``.
    omega : Parameter, optional
        Natural frequency for HORN dynamics (radians per time step), shared across
        all layers. Default is ``2π/28 ≈ 0.224``.
    gamma : Parameter, optional
        Damping coefficient for HORN dynamics (dimensionless), shared across all layers.
        Default is ``0.01``.
    v : Parameter, optional
        Amplitude feedback coefficient (dimensionless), shared across all layers.
        Default is ``0.0``.
    state_init : Callable, optional
        Initializer for HORN state variables in all layers.
        Default is ``braintools.init.ZeroInit()``.
    delay : Initializer, optional
        Synaptic delay configuration for recurrent connections in all layers.
        If ``None``, no delays are used. Default is ``None``.
    rec_w_init : Initializer, optional
        Initializer for recurrent weight matrices in all layers.
        Default is ``braintools.init.KaimingNormal()``.
    rec_b_init : Initializer or None, optional
        Initializer for recurrent bias vectors in all layers.
        Default is ``braintools.init.ZeroInit()``.
    inp_w_init : Initializer, optional
        Initializer for input weight matrices (layer-to-layer and final output).
        Default is ``braintools.init.KaimingNormal()``.
    inp_b_init : Initializer or None, optional
        Initializer for input bias vectors (layer-to-layer and final output).
        Default is ``braintools.init.ZeroInit()``.

    Attributes
    ----------
    layers : list of HORNSeqLayer
        List of HORN sequential layers. The number of layers equals the length of
        ``n_hidden`` if it is a sequence, or 1 if ``n_hidden`` is an integer.
    h2o : brainstate.nn.Linear
        Hidden-to-output linear transformation mapping the final layer's position
        state to the output space. Shape is ``(n_hidden[-1], n_output)``.

    Notes
    -----
    - **Layer stacking**: The output of layer :math:`\ell` (the position state sequence
      :math:`\mathbf{x}^{(\ell)}`) becomes the input to layer :math:`\ell+1`. Each layer
      maintains its own position and velocity states.

    - **Final output**: The ``update`` method returns the output of the hidden-to-output
      transformation applied to the **position state** :math:`\mathbf{x}^{(L)}_T` from
      the last HORN layer at the final time step. This is suitable for sequence
      classification tasks.

    - **Prediction mode**: The ``predict`` method returns the position state sequences
      from **all layers**, which can be useful for analyzing the hierarchical
      representations learned by the network.

    - **Parameter sharing**: All HORN dynamics parameters (``alpha``, ``omega``, ``gamma``,
      ``v``, ``h``) are shared across all layers and all oscillators within each layer.
      This reduces the number of hyperparameters but can be relaxed by passing arrays
      or per-layer configurations.

    - **Flexible depth**: The network depth can be easily configured by adjusting the
      ``n_hidden`` parameter. For example, ``n_hidden=[64, 128, 64]`` creates a
      3-layer network with varying hidden sizes.

    See Also
    --------
    HORNStep : Single-step HORN dynamics
    HORNSeqLayer : Single-layer HORN sequential processor

    Examples
    --------
    Create a 2-layer HORN network for sequence classification:

    >>> import brainstate
    >>> net = HORNSeqNetwork(
    ...     n_input=10,
    ...     n_hidden=[64, 128],
    ...     n_output=5,
    ...     alpha=0.04,
    ...     omega=2*3.14159/28
    ... )
    >>> inputs = brainstate.random.randn(20, 10)  # (time_steps, input_dim)
    >>> output = net(inputs)  # (output_dim,)
    """

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
        delay_init: Callable = braintools.init.ZeroInit(),
        delay: Optional[Initializer] = None,
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
                delay=delay,
                state_init=state_init,
                delay_init=delay_init,
                rec_w_init=rec_w_init,
                rec_b_init=rec_b_init,
                inp_w_init=inp_w_init,
                inp_b_init=inp_b_init,
            )
            self.layers.append(layer)
            n_input = hidden  # next layer input size is current layer hidden size

        self.h2o = brainstate.nn.Linear(n_input, n_output, w_init=inp_w_init, b_init=inp_b_init)

    def update(self, inputs):
        """Process input sequence through all layers and return final output.

        Sequentially processes the input through each HORN layer, then applies
        a linear transformation to the final layer's position state to produce
        the output. Suitable for sequence classification or regression tasks.

        Parameters
        ----------
        inputs : array-like
            Input sequence with shape ``(T, batch?, n_input)`` where ``T`` is the
            number of time steps, ``batch?`` represents optional batch dimensions,
            and ``n_input`` is the input feature dimension.

        Returns
        -------
        array-like
            Final output with shape ``(n_output,)`` or ``(batch?, n_output)``.
            Computed by applying the hidden-to-output transformation ``h2o`` to
            the position state of the last layer.

        Notes
        -----
        This method performs the following steps:

        1. Process input through layer 1 to get hidden sequence 1
        2. Process hidden sequence 1 through layer 2 to get hidden sequence 2
        3. Continue through all remaining layers
        4. Apply linear transformation to final layer's position state

        The output uses only the final position state ``x`` from the last layer,
        not the full sequence.
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        output = self.h2o(self.layers[-1].horn.x.value)
        return output

    def hidden_activation(self, inputs):
        """Process input sequence and return outputs from all layers.

        Similar to ``update``, but returns the position state sequences from all
        intermediate layers. Useful for analyzing hierarchical representations or
        multi-scale temporal features learned by the network.

        Parameters
        ----------
        inputs : array-like
            Input sequence with shape ``(T, batch?, n_input)`` where ``T`` is the
            number of time steps, ``batch?`` represents optional batch dimensions,
            and ``n_input`` is the input feature dimension.

        Returns
        -------
        list of array-like
            List of position state sequences, one per layer. Each element has shape
            ``(T, batch?, n_hidden[i])`` where ``n_hidden[i]`` is the hidden size
            of layer ``i``. The list length equals the number of layers.

        Notes
        -----
        Unlike ``update``, this method does **not** apply the ``h2o`` transformation
        to produce a final output. Instead, it returns the raw position state sequences
        from each layer, which can be useful for:

        - Visualizing layer-wise dynamics
        - Extracting multi-scale temporal features
        - Analyzing information flow through the network
        - Debugging layer activations

        See Also
        --------
        update : Standard forward pass for task output
        """
        x = inputs
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs
