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

from ._common import XY_Oscillator
from ._typing import Initializer
from .noise import Noise

__all__ = [
    'VanDerPolOscillator',
]


class VanDerPolOscillator(XY_Oscillator):
    r"""Van der Pol oscillator (two-dimensional form).

     In the study of dynamical systems, the van der Pol oscillator
    (named for Dutch physicist Balthasar van der Pol) is a non-conservative,
    oscillating system with non-linear damping. It evolves in time according
    to the second-order differential equation

    $$
    {d^{2}x \over dt^{2}}-\mu (1-x^{2}){dx \over dt}+x=0
    $$

    where $x$ is the position coordinate—which is a function of the time $t$—
    and $\mu$ is a scalar parameter indicating the nonlinearity and the
    strength of the damping.

    Implements the Van der Pol oscillator using the Liénard transformation
    that yields the planar system

    .. math::

        \dot x = \mu\,\left(x - \tfrac{1}{3}x^3 - y\right) + I_x(t),

    .. math::

        \dot y = \frac{1}{\mu}\,x + I_y(t),

    where :math:`x` is the state (often interpreted as position or activation),
    :math:`y` is the auxiliary variable, and :math:`\mu > 0` controls the
    nonlinearity and damping. The model exhibits a stable limit cycle for any
    :math:`\mu > 0`.

    Another commonly used form based on the transformation $y={\dot {x}}$ leads to:

    $$
    {\dot {x}}=y
    $$

    $$
    {\dot {y}}=\mu (1-x^{2})y-x
    $$

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node/population. Can be an ``int`` or a tuple of
        ``int``. All parameters are broadcastable to this shape.
    mu : Initializer, optional
        Nonlinearity/damping parameter (dimensionless). Default is ``1.0``.
    noise_x : Noise or None, optional
        Additive noise process for the :math:`x`-equation. If provided, called
        each update and added to ``x_inp``. Default is ``None``.
    noise_y : Noise or None, optional
        Additive noise process for the :math:`y`-equation. If provided, called
        each update and added to ``y_inp``. Default is ``None``.
    init_x : Callable, optional
        Initializer for the state ``x``. Default is
        ``braintools.init.Uniform(0, 0.05)``.
    init_y : Callable, optional
        Initializer for the state ``y``. Default is
        ``braintools.init.Uniform(0, 0.05)``.
    method : str, optional
        Time stepping method. One of ``'exp_euler'`` (exponential Euler; default)
        or any method supported under ``braintools.quad`` (e.g., ``'rk4'``,
        ``'midpoint'``, ``'heun'``, ``'euler'``).

    Attributes
    ----------
    x : brainstate.HiddenState
        State container for :math:`x` (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.
    y : brainstate.HiddenState
        State container for :math:`y` (dimensionless). Shape equals
        ``(batch?,) + in_size`` after ``init_state``.

    Notes
    -----
    - Time derivatives returned by :meth:`dx` and :meth:`dy` carry unit
      ``1/ms`` so that a step size ``dt`` with unit ``ms`` is consistent.
    - For ``method='exp_euler'`` the update uses ``brainstate.nn.exp_euler_step``.
      Otherwise, it dispatches to the corresponding routine in
      ``braintools.quad``.

    References
    ----------
    - van der Pol, B. (1926). On “relaxation-oscillations”. The London,
      Edinburgh, and Dublin Philosophical Magazine and Journal of Science,
      2(11), 978–992.
    - Kaplan, D., & Glass, L. (1995). Understanding Nonlinear Dynamics.
      Springer (pp. 240–244).
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # parameters
        mu: Initializer = 1.0,

        # noise parameters
        noise_x: Noise = None,
        noise_y: Noise = None,

        # other parameters
        init_x: Callable = braintools.init.Uniform(0, 0.05),
        init_y: Callable = braintools.init.Uniform(0, 0.05),
        method: str = 'exp_euler',
    ):
        super().__init__(
            in_size,
            noise_x=noise_x,
            noise_y=noise_y,
            init_x=init_x,
            init_y=init_y,
            method=method,
        )

        # model parameters
        self.mu = braintools.init.param(mu, self.varshape)

    def dx(self, x, y, inp):
        """Right-hand side for the state ``x``.

        Parameters
        ----------
        x : array-like
            Current value of ``x`` (dimensionless).
        y : array-like
            Current value of ``y`` (dimensionless), broadcastable to ``x``.
        inp : array-like or scalar
            External input to ``x`` (includes noise if enabled).

        Returns
        -------
        array-like
            Time derivative ``dx/dt`` with unit ``1/ms``.
        """
        return self.mu * (x - x ** 3 / 3 - y) / u.ms + inp / u.ms

    def dy(self, y, x, inp=0.):
        """Right-hand side for the state ``y``.

        Parameters
        ----------
        y : array-like
            Current value of ``y`` (dimensionless).
        x : array-like
            Current value of ``x`` (dimensionless), broadcastable to ``y``.
        inp : array-like or scalar, optional
            External input to ``y`` (includes noise if enabled). Default is
            ``0.``.

        Returns
        -------
        array-like
            Time derivative ``dy/dt`` with unit ``1/ms``.
        """
        return (x / self.mu + inp) / u.ms

    def derivative(self, state, t, x_inp, y_inp):
        """Vector field for ODE integrators.

        This packs :meth:`dx` and :meth:`dy` into a single callable of the form
        ``f(state, t, x_inp, y_inp)`` to be used by ``braintools.quad``
        integrators when ``method != 'exp_euler'``.

        Parameters
        ----------
        state : tuple of array-like
            Current state as ``(x, y)``.
        t : array-like or scalar
            Current time (ignored in the autonomous dynamics).
        x_inp : array-like or scalar
            External input to ``x`` passed through to :meth:`dx`.
        y_inp : array-like or scalar
            External input to ``y`` passed through to :meth:`dy`.

        Returns
        -------
        tuple of array-like
            Derivatives as ``(dx/dt, dy/dt)`` each with unit ``1/ms``.
        """
        V, w = state
        dVdt = self.dx(V, w, x_inp)
        dwdt = self.dy(w, V, y_inp)
        return (dVdt, dwdt)
