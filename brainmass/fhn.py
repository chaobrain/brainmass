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

import braintools
import brainunit as u

import brainstate
from brainstate.nn import Param, Dynamics
from .noise import Noise
from .typing import Parameter

__all__ = [
    'FitzHughNagumoStep',
]


class FitzHughNagumoStep(Dynamics):
    r"""FitzHugh–Nagumo neural mass model.

    A two-dimensional reduction of the Hodgkin–Huxley model that captures
    excitability via a fast activator (``V``) and a slow recovery variable
    (``w``). The form implemented here follows [1]_:

    .. math::

       \frac{dV}{dt} = -\alpha V^3 + \beta V^2 + \gamma V - w + I_{\text{ext}}(t),\\
       \tau \frac{dw}{dt} = V - \delta - \epsilon\, w.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node/population. Can be an ``int`` or a tuple of
        ``int``. All parameters are broadcastable to this shape.
    alpha : Parameter, optional
        Cubic nonlinearity coefficient (dimensionless). Default is ``3.0``.
    beta : Parameter, optional
        Quadratic nonlinearity coefficient (dimensionless). Default is ``4.0``.
    gamma : Parameter, optional
        Linear coefficient (dimensionless). Default is ``-1.5``.
    delta : Parameter, optional
        Offset for the recovery nullcline (dimensionless). Default is ``0.0``.
    epsilon : Parameter, optional
        Recovery coupling strength (dimensionless). Default is ``0.5``.
    tau : Parameter, optional
        Recovery time constant with unit of time (e.g., ``20.0 * u.ms``).
        Broadcastable to ``in_size``. Default is ``20.0 * u.ms``.
    noise_V : Noise or None, optional
        Additive noise process for the activator ``V``. If provided, it is
        called at each update and added to ``V_inp``. Default is ``None``.
    noise_w : Noise or None, optional
        Additive noise process for the recovery variable ``w``. If provided, it
        is called at each update and added to ``w_inp``. Default is ``None``.
    init_V : Callable, optional
        Parameter for the activator state ``V``. Default is
        ``braintools.init.Uniform(0, 0.05)``.
    init_w : Callable, optional
        Parameter for the recovery state ``w``. Default is
        ``braintools.init.Uniform(0, 0.05)``.
    method: str
        The integration method to use. Either 'exp_euler' for exponential
        Euler (default) or any method supported by ``braintools.quad``, e.g.
        'rk4', 'midpoint', 'heun', 'euler'.

    Attributes
    ----------
    V : brainstate.HiddenState
        Activator (membrane potential–like) state (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.
    w : brainstate.HiddenState
        Recovery variable state (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.

    Notes
    -----
    - Time derivatives returned by :meth:`dV` and :meth:`dw` carry unit
      ``1/ms`` to be consistent with explicit (exponential) Euler stepping with
      ``dt`` in milliseconds.
    - The parameters ``alpha``, ``beta``, ``gamma``, ``delta`` and ``epsilon``
      are dimensionless in this implementation. ``tau`` has unit of time.

    References
    ----------
    .. [1] Kostova, T., Ravindran, R., & Schonbek, M. (2004). FitzHugh–Nagumo
       revisited: Types of bifurcations, periodical forcing and stability
       regions by a Lyapunov functional. International Journal of Bifurcation
       and Chaos, 14(03), 913–925.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # fhn parameters
        alpha: Parameter = 3.0,
        beta: Parameter = 4.0,
        gamma: Parameter = -1.5,
        delta: Parameter = 0.0,
        epsilon: Parameter = 0.5,
        tau: Parameter = 20.0 * u.ms,

        # noise parameters
        noise_V: Noise = None,
        noise_w: Noise = None,

        # other parameters
        init_V: Callable = braintools.init.Uniform(0, 0.05),
        init_w: Callable = braintools.init.Uniform(0, 0.05),
        method: str = 'exp_euler',
    ):
        super().__init__(in_size=in_size)

        # model parameters
        self.alpha = Param.init(alpha, self.varshape)
        self.beta = Param.init(beta, self.varshape)
        self.gamma = Param.init(gamma, self.varshape)
        self.delta = Param.init(delta, self.varshape)
        self.epsilon = Param.init(epsilon, self.varshape)
        self.tau = Param.init(tau, self.varshape)

        # initializers
        assert isinstance(noise_V, Noise) or noise_V is None, "noise_V must be a Noise instance or None."
        assert isinstance(noise_w, Noise) or noise_w is None, "noise_w must be a Noise instance or None."
        assert callable(init_V), "init_V must be a callable."
        assert callable(init_w), "init_w must be a callable."
        self.init_V = init_V
        self.init_w = init_w
        self.noise_V = noise_V
        self.noise_w = noise_w
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.V = brainstate.HiddenState.init(self.init_V, self.varshape, batch_size)
        self.w = brainstate.HiddenState.init(self.init_w, self.varshape, batch_size)

    def dV(self, V, w, inp):
        """Right-hand side for the activator variable ``V``.

        Parameters
        ----------
        V : array-like
            Current activator value (dimensionless).
        w : array-like
            Current recovery variable (dimensionless), broadcastable to ``V``.
        inp : array-like or scalar
            External input to ``V`` (includes noise if enabled).

        Returns
        -------
        array-like
            Time derivative ``dV/dt`` with unit ``1/ms``.
        """
        alpha = self.alpha.value()
        beta = self.beta.value()
        gamma = self.gamma.value()
        return (- alpha * V ** 3 + beta * V ** 2 + gamma * V - w + inp) / u.ms

    def dw(self, w, x, inp=0.):
        """Right-hand side for the recovery variable ``w``.

        Parameters
        ----------
        w : array-like
            Current recovery variable (dimensionless).
        x : array-like
            Current activator value (dimensionless), broadcastable to ``w``.
        inp : array-like or scalar, optional
            External input to ``w`` (includes noise if enabled). Default is
            ``0.``.

        Returns
        -------
        array-like
            Time derivative ``dw/dt`` with unit ``1/ms``.
        """
        delta = self.delta.value()
        epsilon = self.epsilon.value()
        tau = self.tau.value()
        return (x - delta - epsilon * w) / tau + inp / u.ms

    def derivative(self, state, t, V_inp, w_inp):
        V, w = state
        dVdt = self.dV(V, w, V_inp)
        dwdt = self.dw(w, V, w_inp)
        return (dVdt, dwdt)

    def update(self, V_inp=None, w_inp=None):
        """Advance the system by one time step.

        Parameters
        ----------
        V_inp : array-like or scalar or None, optional
            External input to ``V``. If ``None``, treated as zero. If
            ``noise_V`` is set, its output is added. Default is ``None``.
        w_inp : array-like or scalar or None, optional
            External input to ``w``. If ``None``, treated as zero. If
            ``noise_w`` is set, its output is added. Default is ``None``.

        Returns
        -------
        array-like
            The updated activator state ``V`` with the same shape as the
            internal state.

        Notes
        -----
        Uses an exponential-Euler step via ``brainstate.nn.exp_euler_step`` for
        both state variables and updates ``V`` and ``w`` in-place.
        """
        V_inp = 0. if V_inp is None else V_inp
        w_inp = 0. if w_inp is None else w_inp
        if self.noise_V is not None:
            V_inp = V_inp + self.noise_V()
        if self.noise_w is not None:
            w_inp = w_inp + self.noise_w()
        if self.method == 'exp_euler':
            V = brainstate.nn.exp_euler_step(self.dV, self.V.value, self.w.value, V_inp)
            w = brainstate.nn.exp_euler_step(self.dw, self.w.value, self.V.value, w_inp)
        else:
            method = getattr(braintools.quad, f'ode_{self.method}_step')
            t = brainstate.environ.get('t', 0 * u.ms)
            V, w = method(self.derivative, (self.V.value, self.w.value), t, V_inp, w_inp)
        self.V.value = V
        self.w.value = w
        return V
