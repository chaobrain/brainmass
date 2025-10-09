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
    'StuartLandauOscillator',
]


class StuartLandauOscillator(XY_Oscillator):
    r"""Stuart–Landau oscillator (Hopf normal form).

    Implements the real two-dimensional Stuart–Landau equations that describe
    dynamics near a supercritical Hopf bifurcation:

    .. math::

        \dot x = (a - r^2)\,x - \omega\,y + I_x(t),\quad r^2 = x^2 + y^2,

    .. math::

        \dot y = (a - r^2)\,y + \omega\,x + I_y(t),

    where :math:`a` controls the bifurcation (for :math:`a>0` the system
    exhibits a stable limit cycle) and :math:`\omega` is the angular frequency.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node/population. Can be an ``int`` or a tuple of
        ``int``. All parameters are broadcastable to this shape.
    a : Initializer, optional
        Bifurcation parameter (dimensionless). Default is ``0.25``.
    w : Initializer, optional
        Angular frequency :math:`\omega` (dimensionless). Default is ``0.2``.
    noise_x : Noise or None, optional
        Additive noise process for the ``x`` component. If provided, called at
        each update and added to ``x_inp``. Default is ``None``.
    noise_y : Noise or None, optional
        Additive noise process for the ``y`` component. If provided, called at
        each update and added to ``y_inp``. Default is ``None``.
    init_x : Callable, optional
        Initializer for the state ``x``. Default is
        ``braintools.init.Uniform(0, 0.05)``.
    init_y : Callable, optional
        Initializer for the state ``y``. Default is
        ``braintools.init.Uniform(0, 0.05)``.
    method : str, optional
        Time stepping method. One of ``'exp_euler'`` (default; uses
        ``brainstate.nn.exp_euler_step``) or any supported by ``braintools.quad``
        (e.g., ``'rk4'``, ``'midpoint'``, ``'heun'``, ``'euler'``).

    Attributes
    ----------
    x : brainstate.HiddenState
        State container for the real component ``x`` (dimensionless). Shape
        equals ``(batch?,) + in_size`` after ``init_state``.
    y : brainstate.HiddenState
        State container for the imaginary component ``y`` (dimensionless).
        Shape equals ``(batch?,) + in_size`` after ``init_state``.

    Notes
    -----
    - Time derivatives returned by :meth:`dx` and :meth:`dy` carry unit
      ``1/ms`` to be consistent with explicit (exponential) Euler integration
      for a step size with unit ``ms``.
    - Integration, state initialization, and noise handling are implemented in
      the base class ``XY_Oscillator``.
    - Implementation detail: verify the cross-coupling term in :meth:`dy`
      matches the intended normal form (the standard form uses ``+ w * x``).
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # model parameters
        a: Initializer = 0.25,
        w: Initializer = 0.2,

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
        self.a = braintools.init.param(a, self.varshape, allow_none=False)
        self.w = braintools.init.param(w, self.varshape, allow_none=False)

    def dx(self, x, y, x_ext):
        """Right-hand side for the ``x`` component.

        Parameters
        ----------
        x : array-like
            Current value of ``x`` (dimensionless).
        y : array-like
            Current value of ``y`` (dimensionless), broadcastable to ``x``.
        x_ext : array-like or scalar
            External input to ``x`` (includes noise if enabled).

        Returns
        -------
        array-like
            Time derivative ``dx/dt`` with unit ``1/ms``.
        """
        return ((self.a - x * x - y * y) * x - self.w * y + x_ext) / u.ms

    def dy(self, y, x, y_ext):
        """Right-hand side for the ``y`` component.

        Parameters
        ----------
        y : array-like
            Current value of ``y`` (dimensionless).
        x : array-like
            Current value of ``x`` (dimensionless), broadcastable to ``y``.
        y_ext : array-like or scalar
            External input to ``y`` (includes noise if enabled).

        Returns
        -------
        array-like
            Time derivative ``dy/dt`` with unit ``1/ms``.
        """
        return ((self.a - x * x - y * y) * y - self.w * y + y_ext) / u.ms
