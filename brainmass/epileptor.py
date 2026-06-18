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
    'EpileptorStep',
]


class EpileptorStep(NeuralMassDynamics):
    r"""Epileptor model of seizure dynamics (Jirsa et al., 2014).

    The Hindmarsh-Rose-Jirsa Epileptor is a six-dimensional neural-mass model that
    reproduces the full taxonomy of epileptic seizures: the autonomous transition
    from interictal (between-seizure) to ictal (seizure) activity and back,
    driven by a slow *permittivity* variable [1]_ [2]_. It couples three
    subsystems on separate time scales:

    - a fast subsystem :math:`(x_1, y_1)` generating ictal fast discharges,
    - an ultra-slow subsystem :math:`(x_2, y_2)` for spike-and-wave events,
    - a slow permittivity variable :math:`z` that ramps seizures on and off,

    plus a low-pass filter :math:`g` linking the two subsystems.

    .. math::

       \begin{aligned}
       \dot x_1 &= t_t\,(y_1 - z + I_{\mathrm{ext}} + K_{vf} c_1 + f_1(x_1, x_2)\,x_1), \\
       \dot y_1 &= t_t\,(c - d\,x_1^2 - y_1), \\
       \dot z   &= t_t\,r\,(h(x_1, z) - z + K_s c_1), \\
       \dot x_2 &= t_t\,(-y_2 + x_2 - x_2^3 + I_{\mathrm{ext2}} + b_b\,g - 0.3(z - 3.5) + K_f c_2), \\
       \dot y_2 &= t_t\,(-y_2 + f_2(x_2))/\tau, \\
       \dot g   &= t_t\,(-0.01\,(g - 0.1\,x_1)),
       \end{aligned}

    with the piecewise nonlinearities

    .. math::

       f_1 = \begin{cases} -a x_1^2 + b x_1 & x_1 < 0 \\ \mathrm{slope} - x_2 + 0.6 (z - 4)^2 & x_1 \ge 0 \end{cases},
       \quad
       f_2 = \begin{cases} 0 & x_2 < -0.25 \\ a_a (x_2 + 0.25) & x_2 \ge -0.25 \end{cases},

    and the permittivity coupling :math:`h`, a blend (via ``modification``
    :math:`\in [0, 1]`) of a linear and a sigmoidal form,

    .. math::

       h = \mathrm{mod}\,\bigl(x_0 + \tfrac{3}{1 + e^{-(x_1 + 0.5)/0.1}}\bigr)
         + (1 - \mathrm{mod})\,\bigl(4(x_1 - x_0) + z_{\mathrm{nl}}\bigr),
       \quad z_{\mathrm{nl}} = \begin{cases} -0.1 z^7 & z < 0 \\ 0 & z \ge 0 \end{cases}.

    Here :math:`c_1` and :math:`c_2` are external/coupling inputs to populations 1
    and 2 (``x1_inp`` and ``x2_inp``); :math:`c_1` additionally drives the
    permittivity through :math:`K_s`.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the population. All parameters broadcast to this shape.
    a, b : Parameter, optional
        Quadratic / linear coefficients of :math:`f_1`. Defaults ``1.0, 3.0``.
    c, d : Parameter, optional
        Additive / quadratic coefficients of :math:`\dot y_1`. Defaults ``1.0, 5.0``.
    r : Parameter, optional
        Permittivity rate (inverse slow time constant). Default ``0.00035`` — the
        seizure-cycle time scale; larger values speed seizures up.
    x0 : Parameter, optional
        Epileptogenicity parameter (seizure threshold). Default ``-1.6`` (an
        epileptogenic node). More negative values render the node healthy.
    Iext : Parameter, optional
        External input to population 1. Default ``3.1``.
    slope : Parameter, optional
        Linear coefficient in the :math:`x_1 \ge 0` branch of :math:`f_1`. Default ``0.0``.
    Iext2 : Parameter, optional
        External input to population 2. Default ``0.45``.
    tau : Parameter, optional
        Time scale of :math:`y_2`. Default ``10.0``.
    aa : Parameter, optional
        Slope of :math:`f_2`. Default ``6.0``.
    bb : Parameter, optional
        Coupling from the filter :math:`g` into :math:`x_2`. Default ``2.0``.
    Kvf, Kf, Ks : Parameter, optional
        Very-fast (:math:`x_1`), fast (:math:`x_2`) and slow (:math:`z`) coupling
        scales. Defaults ``0.0``.
    tt : Parameter, optional
        Global time-scale factor. Default ``1.0``.
    modification : Parameter, optional
        Blend in ``[0, 1]`` selecting the nonlinear (``1``) vs linear (``0``)
        permittivity influence on :math:`z`. Default ``0.0``.
    init_x1, init_y1, init_z, init_x2, init_y2, init_g : Callable, optional
        State initializers. Defaults reproduce the canonical initial condition
        ``(-1.5, -10.0, 3.5, -1.0, 0.0, 0.0)``.
    noise_x1, noise_x2 : Noise or None, optional
        Additive noise processes for populations 1 and 2. Default ``None``.
    method : str, optional
        Integration method, ``'exp_euler'`` (default; well-suited to the stiff slow
        ``z``) or any ``braintools.quad`` method (e.g. ``'rk4'``).

    Attributes
    ----------
    x1, y1 : brainstate.HiddenState
        Fast subsystem (membrane-like activity and recovery).
    z : brainstate.HiddenState
        Slow permittivity variable driving seizure onset/offset.
    x2, y2 : brainstate.HiddenState
        Ultra-slow subsystem (spike-and-wave).
    g : brainstate.HiddenState
        Low-pass filter of ``x1``.

    Notes
    -----
    - State variables are dimensionless; every right-hand side carries unit
      ``1/ms`` so an exponential-Euler step with ``dt`` in milliseconds is
      consistent.
    - The ``z`` time scale is *stiff and slow* (``r = 3.5e-4``): a full
      interictal-ictal cycle spans :math:`\mathcal{O}(1/r)` time units. Raising
      ``r`` rescales time and makes seizures observable over fewer steps.
    - :meth:`lfp` returns the standard local-field-potential proxy
      :math:`x_2 - x_1`, which :meth:`update` returns.

    References
    ----------
    .. [1] V. K. Jirsa, W. C. Stacey, P. P. Quilichini, A. I. Ivanov, C. Bernard
       (2014). On the nature of seizure dynamics. Brain, 137(8), 2210-2230.
       https://doi.org/10.1093/brain/awu133
    .. [2] T. Proix, F. Bartolomei, P. Chauvel, C. Bernard, V. K. Jirsa (2014).
       Permittivity coupling across brain regions determines seizure recruitment in
       partial epilepsy. Journal of Neuroscience, 34(45), 15009-15021.

    Examples
    --------
    .. code-block:: python

       >>> import brainmass
       >>> import brainstate
       >>> import brainunit as u
       >>> model = brainmass.EpileptorStep(in_size=1)
       >>> _ = brainstate.nn.init_all_states(model)
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     lfp = model.update()
       >>> lfp.shape
       (1,)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # population 1
        a: Parameter = 1.0,
        b: Parameter = 3.0,
        c: Parameter = 1.0,
        d: Parameter = 5.0,
        r: Parameter = 0.00035,
        x0: Parameter = -1.6,
        Iext: Parameter = 3.1,
        slope: Parameter = 0.0,
        # population 2
        Iext2: Parameter = 0.45,
        tau: Parameter = 10.0,
        aa: Parameter = 6.0,
        bb: Parameter = 2.0,
        # coupling
        Kvf: Parameter = 0.0,
        Kf: Parameter = 0.0,
        Ks: Parameter = 0.0,
        # global
        tt: Parameter = 1.0,
        modification: Parameter = 0.0,

        # initializers reproducing the canonical IC
        init_x1: Callable = braintools.init.Constant(-1.5),
        init_y1: Callable = braintools.init.Constant(-10.0),
        init_z: Callable = braintools.init.Constant(3.5),
        init_x2: Callable = braintools.init.Constant(-1.0),
        init_y2: Callable = braintools.init.Constant(0.0),
        init_g: Callable = braintools.init.Constant(0.0),
        noise_x1: Noise = None,
        noise_x2: Noise = None,
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        for name, value in dict(
            a=a, b=b, c=c, d=d, r=r, x0=x0, Iext=Iext, slope=slope,
            Iext2=Iext2, tau=tau, aa=aa, bb=bb,
            Kvf=Kvf, Kf=Kf, Ks=Ks, tt=tt, modification=modification,
        ).items():
            setattr(self, name, Param.init(value, self.varshape))

        for init in (init_x1, init_y1, init_z, init_x2, init_y2, init_g):
            assert callable(init), 'state initializers must be callable'
        assert isinstance(noise_x1, Noise) or noise_x1 is None, 'noise_x1 must be a Noise or None'
        assert isinstance(noise_x2, Noise) or noise_x2 is None, 'noise_x2 must be a Noise or None'
        self.init_x1 = init_x1
        self.init_y1 = init_y1
        self.init_z = init_z
        self.init_x2 = init_x2
        self.init_y2 = init_y2
        self.init_g = init_g
        self.noise_x1 = noise_x1
        self.noise_x2 = noise_x2
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Allocate the six Epileptor states at their canonical initial values."""
        self.x1 = brainstate.HiddenState.init(self.init_x1, self.varshape, batch_size)
        self.y1 = brainstate.HiddenState.init(self.init_y1, self.varshape, batch_size)
        self.z = brainstate.HiddenState.init(self.init_z, self.varshape, batch_size)
        self.x2 = brainstate.HiddenState.init(self.init_x2, self.varshape, batch_size)
        self.y2 = brainstate.HiddenState.init(self.init_y2, self.varshape, batch_size)
        self.g = brainstate.HiddenState.init(self.init_g, self.varshape, batch_size)

    def _f1(self, x1, x2, z):
        """Piecewise nonlinearity coupling the fast subsystem to populations."""
        a = self.a.value()
        b = self.b.value()
        slope = self.slope.value()
        if_neg = -a * x1 ** 2 + b * x1
        if_pos = slope - x2 + 0.6 * (z - 4.0) ** 2
        return u.math.where(x1 < 0.0, if_neg, if_pos)

    def _f2(self, x2):
        """Piecewise nonlinearity in the :math:`y_2` recovery equation."""
        aa = self.aa.value()
        return u.math.where(x2 < -0.25, u.math.zeros_like(x2), aa * (x2 + 0.25))

    def _h(self, x1, z):
        """Permittivity influence on ``z`` (linear/nonlinear blend)."""
        x0 = self.x0.value()
        mod = self.modification.value()
        z_nl = u.math.where(z < 0.0, -0.1 * z ** 7, u.math.zeros_like(z))
        h_nonlinear = x0 + 3.0 / (1.0 + u.math.exp(-(x1 + 0.5) / 0.1))
        h_linear = 4.0 * (x1 - x0) + z_nl
        return mod * h_nonlinear + (1.0 - mod) * h_linear

    def dx1(self, x1, y1, z, x2, x1_inp, add=0.0):
        """Right-hand side for the fast activity ``x1`` (unit ``1/ms``).

        ``x1_inp`` is the population-1 coupling current (scaled by ``Kvf``); ``add``
        is a direct additive perturbation (e.g. noise) applied at the derivative
        level, independent of the coupling gain.
        """
        tt = self.tt.value()
        f1 = self._f1(x1, x2, z)
        det = tt * (y1 - z + self.Iext.value() + self.Kvf.value() * x1_inp + f1 * x1)
        return (det + add) / u.ms

    def dy1(self, y1, x1):
        """Right-hand side for the fast recovery ``y1`` (unit ``1/ms``)."""
        tt = self.tt.value()
        return tt * (self.c.value() - self.d.value() * x1 ** 2 - y1) / u.ms

    def dz(self, z, x1, x1_inp):
        """Right-hand side for the slow permittivity ``z`` (unit ``1/ms``)."""
        tt = self.tt.value()
        h = self._h(x1, z)
        return tt * self.r.value() * (h - z + self.Ks.value() * x1_inp) / u.ms

    def dx2(self, x2, y2, z, g, x2_inp, add=0.0):
        """Right-hand side for the ultra-slow activity ``x2`` (unit ``1/ms``).

        ``x2_inp`` is the population-2 coupling current (scaled by ``Kf``); ``add``
        is a direct additive perturbation (e.g. noise) applied at the derivative
        level, independent of the coupling gain.
        """
        tt = self.tt.value()
        det = tt * (
            -y2 + x2 - x2 ** 3 + self.Iext2.value() + self.bb.value() * g
            - 0.3 * (z - 3.5) + self.Kf.value() * x2_inp
        )
        return (det + add) / u.ms

    def dy2(self, y2, x2):
        """Right-hand side for the ultra-slow recovery ``y2`` (unit ``1/ms``)."""
        tt = self.tt.value()
        return tt * (-y2 + self._f2(x2)) / self.tau.value() / u.ms

    def dg(self, g, x1):
        """Right-hand side for the low-pass filter ``g`` (unit ``1/ms``)."""
        tt = self.tt.value()
        return tt * (-0.01 * (g - 0.1 * x1)) / u.ms

    def derivative(self, state, t, x1_inp, x2_inp, add1=0.0, add2=0.0):
        x1, y1, z, x2, y2, g = state
        return (
            self.dx1(x1, y1, z, x2, x1_inp, add1),
            self.dy1(y1, x1),
            self.dz(z, x1, x1_inp),
            self.dx2(x2, y2, z, g, x2_inp, add2),
            self.dy2(y2, x2),
            self.dg(g, x1),
        )

    def lfp(self):
        """Local-field-potential proxy ``x2 - x1`` (the standard Epileptor output)."""
        return self.x2.value - self.x1.value

    def update(self, x1_inp=None, x2_inp=None):
        """Advance the six Epileptor states by one time step.

        Parameters
        ----------
        x1_inp : array-like or scalar or None, optional
            Coupling input to population 1 (also drives ``z`` through ``Ks``). If
            ``None``, treated as zero; ``noise_x1`` is added when set.
        x2_inp : array-like or scalar or None, optional
            Coupling input to population 2. If ``None``, treated as zero;
            ``noise_x2`` is added when set.

        Returns
        -------
        array-like
            The local-field-potential proxy :meth:`lfp` (``x2 - x1``).
        """
        x1_inp = 0.0 if x1_inp is None else x1_inp
        x2_inp = 0.0 if x2_inp is None else x2_inp
        # Noise enters as a direct additive perturbation on the derivative (TVB-style
        # additive noise), independent of the Kvf/Kf coupling gains.
        add1 = self.noise_x1() if self.noise_x1 is not None else 0.0
        add2 = self.noise_x2() if self.noise_x2 is not None else 0.0

        x1, y1, z, x2, y2, g = self._solve_step(
            exp_euler_specs=(
                (self.dx1, self.x1.value, self.y1.value, self.z.value, self.x2.value, x1_inp, add1),
                (self.dy1, self.y1.value, self.x1.value),
                (self.dz, self.z.value, self.x1.value, x1_inp),
                (self.dx2, self.x2.value, self.y2.value, self.z.value, self.g.value, x2_inp, add2),
                (self.dy2, self.y2.value, self.x2.value),
                (self.dg, self.g.value, self.x1.value),
            ),
            ode_state=(
                self.x1.value, self.y1.value, self.z.value,
                self.x2.value, self.y2.value, self.g.value,
            ),
            ode_inputs=(x1_inp, x2_inp, add1, add2),
        )
        self.x1.value = x1
        self.y1.value = y1
        self.z.value = z
        self.x2.value = x2
        self.y2.value = y2
        self.g.value = g
        return self.lfp()
