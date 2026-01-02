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
from brainstate.nn import Param
from ._common import XY_Oscillator
from ._noise import Noise
from ._typing import Parameter

__all__ = [
    'HopfStep',
]


class HopfStep(XY_Oscillator):
    r"""Normal-form Hopf oscillator (two-dimensional rate model).

    This model implements the supercritical Hopf normal form for a single node
    in terms of its real and imaginary components, often used as a simple
    mesoscopic model of oscillatory neural population activity.

    The complex form is

    .. math::
        \frac{dz}{dt} = (a + i\,\omega)\,z - \beta\,|z|^{2} z + I_{\text{ext}}(t),

    where :math:`z = x + i y` and :math:`|z|^2 = x^2 + y^2`. In real variables:

    .. math::
        \begin{aligned}
        \dot x &= (a - \beta r)\,x - \omega\,y + \text{coupled}_x + I_x(t),\\
        \dot y &= (a - \beta r)\,y + \omega\,x + \text{coupled}_y + I_y(t), \\
        r &= x^2 + y^2.
        \end{aligned}

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node. Can be an int or tuple of ints.
    a : Parameter, optional
        Bifurcation parameter (dimensionless). For ``a > 0`` the system exhibits
        a stable limit cycle; for ``a < 0`` the origin is a stable focus.
        Broadcastable to ``in_size``. Default is ``0.25``.
    w : Parameter, optional
        Angular frequency :math:`\omega` (dimensionless in this implementation).
        Broadcastable to ``in_size``. Default is ``0.2``.
    beta : Parameter, optional
        Nonlinear saturation coefficient (dimensionless) setting the limit-cycle
        amplitude (approximately :math:`\sqrt{a/\beta}` when ``a>0``).
        Broadcastable to ``in_size``. Default is ``1.0``.
    noise_x : Noise or None, optional
        Additive noise process to ``x``. If provided, called each step and added
        to ``ext_x``. Default is ``None``.
    noise_y : Noise or None, optional
        Additive noise process to ``y``. If provided, called each step and added
        to ``ext_y``. Default is ``None``.

    Attributes
    ----------
    x : brainstate.HiddenState
        State container for the real component ``x``.
    y : brainstate.HiddenState
        State container for the imaginary component ``y``.

    Notes
    -----
    Time derivatives returned by ``dx`` and ``dy`` carry unit ``1/ms`` so that
    an explicit (exponential) Euler integrator with time step ``dt`` having
    unit ``ms`` evolves the state consistently with units.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        a: Parameter = 0.25,  # Hopf bifurcation parameter
        w: Parameter = 0.2,  # Oscillator frequency
        beta: Parameter = 1.0,  # nonlinear saturation coefficient

        # noise
        noise_x: Noise = None,
        noise_y: Noise = None,

        # initialization
        init_x: Callable = braintools.init.ZeroInit(),
        init_y: Callable = braintools.init.ZeroInit(),
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

        self.a = Param.init(a, self.varshape)
        self.w = Param.init(w, self.varshape)
        self.beta = Param.init(beta, self.varshape)

    def dx(self, x, y, inp):
        """Right-hand side for ``x``.

        Parameters
        ----------
        x : array-like
            Current real component.
        y : array-like
            Current imaginary component (broadcastable to ``x``).
        inp : array-like or scalar
            External input to ``x`` (includes coupling and noise).

        Returns
        -------
        array-like
            Time derivative ``dx/dt`` with unit ``1/ms``.
        """
        a = self.a.value()
        w = self.w.value()
        beta = self.beta.value()
        r = x ** 2 + y ** 2
        dx_dt = (a - beta * r) * x - w * y + inp
        return dx_dt / u.ms

    def dy(self, y, x, inp):
        """Right-hand side for ``y``.

        Parameters
        ----------
        y : array-like
            Current imaginary component.
        x : array-like
            Current real component (broadcastable to ``y``).
        inp : array-like or scalar
            External input to ``y`` (includes coupling and noise).

        Returns
        -------
        array-like
            Time derivative ``dy/dt`` with unit ``1/ms``.
        """
        a = self.a.value()
        beta = self.beta.value()
        w = self.w.value()
        r = x ** 2 + y ** 2
        dy_dt = (a - beta * r) * y + w * x + inp
        return dy_dt / u.ms
