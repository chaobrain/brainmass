# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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
from brainstate.nn import Param
from ._base import NeuralMassDynamics
from .noise import Noise
from .typing import Parameter

__all__ = [
    'Generic2dOscillatorStep',
]


class Generic2dOscillatorStep(NeuralMassDynamics):
    r"""Generic 2D oscillator with configurable nullclines (TVB ``Generic2dOscillator``).

    A flexible planar dynamical system with a fast variable :math:`V` (membrane-like)
    and a slow recovery variable :math:`W`. By tuning the polynomial nullcline
    coefficients it reproduces a wide range of behaviours — excitability,
    bistability, relaxation oscillations — and many specific TVB 2-D models are
    special cases of it [1]_ [2]_ [3]_:

    .. math::

       \begin{aligned}
       \dot V &= d\,\tau\,\bigl(-f V^3 + e V^2 + g V + \alpha W + \gamma I + \gamma c\bigr) + s, \\
       \dot W &= \frac{d}{\tau}\,\bigl(a + b V + c_{2} V^2 - \beta W\bigr),
       \end{aligned}

    where :math:`c` is the (long-range + local) coupling input — scaled by
    :math:`\gamma` like the intrinsic drive :math:`I` — and :math:`s` is a direct
    additive stimulus/noise term. The coefficient named :math:`c_2` below is the
    quadratic-:math:`V` coefficient of the :math:`W` nullcline (the constructor
    argument is ``c``).

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node. An ``int`` or tuple of ``int``; all parameters
        broadcast to this shape.
    a : Parameter, optional
        Constant term of the :math:`W` nullcline. Default ``-2.0``.
    b : Parameter, optional
        Linear-:math:`V` coefficient of the :math:`W` nullcline. Default ``-10.0``.
    c : Parameter, optional
        Quadratic-:math:`V` coefficient of the :math:`W` nullcline (:math:`c_2` in
        the equations). Default ``0.0``.
    d : Parameter, optional
        Global time-scaling. Default ``0.02``.
    e : Parameter, optional
        Quadratic-:math:`V` coefficient of the :math:`V` nullcline. Default ``3.0``.
    f : Parameter, optional
        Cubic-:math:`V` coefficient of the :math:`V` nullcline. Default ``1.0``.
    g : Parameter, optional
        Linear-:math:`V` coefficient of the :math:`V` nullcline. Default ``0.0``.
    alpha : Parameter, optional
        :math:`W \to V` coupling strength. Default ``1.0``.
    beta : Parameter, optional
        :math:`W` decay rate. Default ``1.0``.
    gamma : Parameter, optional
        Scaling applied to the intrinsic drive :math:`I` and the coupling input.
        Default ``1.0``.
    I : Parameter, optional
        Constant external input current. Default ``0.0``.
    tau : Parameter, optional
        Time-scale separation between :math:`V` and :math:`W` (``tau > 1`` makes
        :math:`V` faster). Default ``1.0``.
    init_V, init_W : Callable, optional
        State initializers. Defaults ``braintools.init.Constant(0.0)``.
    noise_V, noise_W : Noise or None, optional
        Additive noise processes applied directly to the :math:`V` / :math:`W`
        derivatives (the TVB stimulus slot). Default ``None``.
    method : str, optional
        Integration method, ``'exp_euler'`` (default) or any ``braintools.quad``
        method (e.g. ``'rk4'``).

    Attributes
    ----------
    V : brainstate.HiddenState
        Fast variable (membrane-like). Shape ``(batch?,) + in_size``.
    W : brainstate.HiddenState
        Slow recovery variable.

    Notes
    -----
    - State variables are dimensionless; :meth:`dV` / :meth:`dW` carry unit
      ``1/ms`` so an exponential-Euler step with ``dt`` in milliseconds is
      consistent.
    - **Parameter regimes** (a few of the published ones):

      - *Excitable* (FitzHugh-Nagumo-like, the defaults):
        ``a=-2, b=-10, c=0, d=0.02``.
      - *Bistable*: ``a=1, b=0, c=-5, d=0.02`` — two stable fixed points; the basin
        is selected by the initial condition.
      - *Morris-Lecar-like*: ``a=0.5, b=0.6, c=-4, d=0.02``.

    References
    ----------
    .. [1] R. FitzHugh (1961). Impulses and physiological states in theoretical
       models of nerve membrane. Biophysical Journal, 1, 445-466.
    .. [2] A. Stefanescu, V. K. Jirsa (2008). A low dimensional description of
       globally coupled heterogeneous neural networks of excitatory and inhibitory
       neurons. PLoS Computational Biology, 4(11), e1000219.
    .. [3] P. Sanz-Leon, S. A. Knock, A. Spiegler, V. K. Jirsa (2015). Mathematical
       framework for large-scale brain network modeling in The Virtual Brain.
       NeuroImage, 111, 385-430.

    Examples
    --------
    .. code-block:: python

       >>> import brainmass
       >>> import brainstate
       >>> import brainunit as u
       >>> model = brainmass.Generic2dOscillatorStep(in_size=1)
       >>> _ = brainstate.nn.init_all_states(model)
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     V = model.update()
       >>> V.shape
       (1,)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # nullcline / dynamics parameters
        a: Parameter = -2.0,
        b: Parameter = -10.0,
        c: Parameter = 0.0,
        d: Parameter = 0.02,
        e: Parameter = 3.0,
        f: Parameter = 1.0,
        g: Parameter = 0.0,
        alpha: Parameter = 1.0,
        beta: Parameter = 1.0,
        gamma: Parameter = 1.0,
        I: Parameter = 0.0,
        tau: Parameter = 1.0,

        # initializers / noise
        init_V: Callable = braintools.init.Constant(0.0),
        init_W: Callable = braintools.init.Constant(0.0),
        noise_V: Noise = None,
        noise_W: Noise = None,
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        for name, value in dict(
            a=a, b=b, c=c, d=d, e=e, f=f, g=g,
            alpha=alpha, beta=beta, gamma=gamma, I=I, tau=tau,
        ).items():
            setattr(self, name, Param.init(value, self.varshape))

        for init in (init_V, init_W):
            assert callable(init), 'state initializers must be callable'
        assert isinstance(noise_V, Noise) or noise_V is None, 'noise_V must be a Noise instance or None'
        assert isinstance(noise_W, Noise) or noise_W is None, 'noise_W must be a Noise instance or None'
        self.init_V = init_V
        self.init_W = init_W
        self.noise_V = noise_V
        self.noise_W = noise_W
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Allocate the fast (``V``) and slow (``W``) states.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.V = brainstate.HiddenState.init(self.init_V, self.varshape, batch_size)
        self.W = brainstate.HiddenState.init(self.init_W, self.varshape, batch_size)

    def dV(self, V, W, V_inp, add=0.0):
        """Right-hand side for the fast variable ``V`` (unit ``1/ms``).

        ``V_inp`` is the coupling input (scaled by ``gamma`` like the intrinsic
        drive ``I``); ``add`` is a direct additive stimulus/noise term.
        """
        d = self.d.value()
        tau = self.tau.value()
        gamma = self.gamma.value()
        det = d * tau * (
            -self.f.value() * V ** 3
            + self.e.value() * V ** 2
            + self.g.value() * V
            + self.alpha.value() * W
            + gamma * self.I.value()
            + gamma * V_inp
        )
        return (det + add) / u.ms

    def dW(self, W, V, add=0.0):
        """Right-hand side for the slow recovery ``W`` (unit ``1/ms``)."""
        d = self.d.value()
        tau = self.tau.value()
        det = (d / tau) * (
            self.a.value() + self.b.value() * V + self.c.value() * V ** 2 - self.beta.value() * W
        )
        return (det + add) / u.ms

    def derivative(self, state, t, V_inp, add_V=0.0, add_W=0.0):
        V, W = state
        return self.dV(V, W, V_inp, add_V), self.dW(W, V, add_W)

    def update(self, V_inp=None):
        """Advance the oscillator by one time step.

        Parameters
        ----------
        V_inp : array-like or scalar or None, optional
            Coupling input to the ``V`` equation (scaled by ``gamma``). If
            ``None``, treated as zero. Default is ``None``.

        Returns
        -------
        array-like
            The updated fast variable ``V``, same shape as the internal state.
        """
        V_inp = 0.0 if V_inp is None else V_inp
        # Noise enters as a direct additive perturbation on the derivative (the TVB
        # stimulus slot), independent of the gamma coupling gain.
        add_V = self.noise_V() if self.noise_V is not None else 0.0
        add_W = self.noise_W() if self.noise_W is not None else 0.0

        V, W = self._solve_step(
            exp_euler_specs=(
                (self.dV, self.V.value, self.W.value, V_inp, add_V),
                (self.dW, self.W.value, self.V.value, add_W),
            ),
            ode_state=(self.V.value, self.W.value),
            ode_inputs=(V_inp, add_V, add_W),
        )
        self.V.value = V
        self.W.value = W
        return V
