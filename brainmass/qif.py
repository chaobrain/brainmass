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

import brainstate
import braintools
import brainunit as u

from ._typing import Initializer
from .noise import Noise

__all__ = [
    'QIF',
]


class QIF(brainstate.nn.Dynamics):
    r"""Quadratic integrate-and-fire mean-field model (Montbrió–Pazó–Roxin).

    Implements the exact mean-field reduction of a population of all-to-all
    coupled QIF neurons with a Lorentzian distribution of background
    excitabilities [1]_. The macroscopic dynamics of the population firing rate
    :math:`r(t)` and mean membrane potential :math:`v(t)` follow

    .. math::
       \begin{aligned}
       \tau \, \dot r(t) &= \frac{\Delta}{\pi} + 2 \, \tau \, r(t) \, v(t), \\
       \tau \, \dot v(t) &= v(t)^2 + \bar\eta + I(t) + J \, \tau \, r(t)
                             - \bigl(\pi \, \tau \, r(t)\bigr)^2,\quad
       \end{aligned}

    where :math:`\bar\eta` is the mean excitability, :math:`\Delta` the
    Lorentzian half-width at half-maximum (HWHM), :math:`J` the recurrent
    coupling strength, and :math:`I(t)` an external input to the mean potential.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the population. Can be an ``int`` or a tuple of
        ``int``. All parameters are broadcastable to this shape.
    tau : Initializer, optional
        Population time constant with unit of time (e.g., ``1. * u.ms``).
        Default is ``1. * u.ms``.
    eta : Initializer, optional
        Mean of the Lorentzian excitability distribution (dimensionless).
        Default is ``-5.0``.
    delta : Initializer, optional
        HWHM of the Lorentzian excitability distribution.
        Default is ``1.0 * u.Hz``.
    J : Initializer, optional
        Recurrent coupling strength (dimensionless). Default is ``15.``.
    init_r : Callable, optional
        Initializer for the firing-rate state ``r``. Default is
        ``braintools.init.Uniform(0, 0.05)``.
    init_v : Callable, optional
        Initializer for the mean-membrane-potential state ``v``. Default is
        ``braintools.init.Uniform(0, 0.05)``.
    noise_r : Noise or None, optional
        Additive noise process for the rate dynamics. If provided, its output
        is added to ``r_inp`` at each update. Default is ``None``.
    noise_v : Noise or None, optional
        Additive noise process for the potential dynamics. If provided, its
        output is added to ``v_inp`` at each update. Default is ``None``.
    method: str
        Integration method to use. Either 'exp_euler' (default) or one of the
        methods implemented in `braintools.quad` (e.g., 'rk4', 'rk2', 'dopri5').
        The exponential-Euler method is recommended for efficiency and
        stability.

        .. warning::
            The exponential-Euler method is only valid for this model because
            the rate equation is linear in ``r``. Other methods are provided
            for comparison but may be less efficient and/or unstable.

    Attributes
    ----------
    r : brainstate.HiddenState
        Population firing rate (dimensionless). Shape equals ``(batch?,) + in_size``
        after ``init_state``.
    v : brainstate.HiddenState
        Population mean membrane potential (dimensionless in this implementation).
        Shape equals ``(batch?,) + in_size`` after ``init_state``.

    Notes
    -----
    - Time derivatives returned by :meth:`dr` and :meth:`dv` carry unit
      ``1/ms`` so that an exponential-Euler step with ``dt`` in milliseconds is
      consistent.
    - The mean-field reduction is exact under all-to-all coupling, infinite
      population size, and Lorentzian excitability assumptions; it closely
      approximates large sparse networks [2]_.

    References
    ----------
    .. [1] E. Montbrió, D. Pazó, A. Roxin (2015). Macroscopic description for
       networks of spiking neurons. Physical Review X, 5:021028.
       https://doi.org/10.1103/PhysRevX.5.021028
    .. [2] R. Gast, H. Schmidt, T. R. Knösche (2020). A Mean-Field Description of
       Bursting Dynamics in Spiking Neural Networks with Short-Term Adaptation.
        Neural Computation, 32(9), 1615–1634.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # model parameters
        tau: Initializer = 1. * u.ms,
        eta: Initializer = -5.0,
        delta: Initializer = 1.0 * u.Hz,
        J: Initializer = 15.,

        # initializers
        init_r: Callable = braintools.init.Uniform(0, 0.05, unit=u.Hz),
        init_v: Callable = braintools.init.Uniform(0, 0.05),
        noise_r: Noise = None,
        noise_v: Noise = None,
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        # time integration method
        self.tau = braintools.init.param(tau, self.varshape, allow_none=False)

        # the mean of a Lorenzian distribution over the neural excitability in the population
        self.eta = braintools.init.param(eta, self.varshape, allow_none=False)

        # the half-width at half maximum of the Lorenzian distribution over the neural excitability
        self.delta = braintools.init.param(delta, self.varshape, allow_none=False)

        # the strength of the recurrent coupling inside the population
        self.J = braintools.init.param(J, self.varshape, allow_none=False)

        # noise and initializers
        assert callable(init_r), 'init_r must be callable'
        assert callable(init_v), 'init_v must be callable'
        assert isinstance(noise_v, Noise) or noise_v is None, 'noise_v must be a Noise instance or None'
        assert isinstance(noise_r, Noise) or noise_r is None, 'noise_r must be a Noise instance or None'
        self.init_r = init_r
        self.init_v = init_v
        self.noise_r = noise_r
        self.noise_v = noise_v
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Initialize firing-rate and mean-potential states.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.r = brainstate.HiddenState(braintools.init.param(self.init_r, self.varshape, batch_size))
        self.v = brainstate.HiddenState(braintools.init.param(self.init_v, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        """Reset states ``r`` and ``v`` using the configured initializers.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional batch dimension for reinitialization. If ``None``, keeps
            current batch shape but resets values. Default is ``None``.
        """
        self.r.value = braintools.init.param(self.init_r, self.varshape, batch_size)
        self.v.value = braintools.init.param(self.init_v, self.varshape, batch_size)

    def dr(self, r, v, r_ext):
        """Right-hand side for the firing rate ``r``.

        Parameters
        ----------
        r : array-like
            Current firing rate (dimensionless).
        v : array-like
            Current mean membrane potential (dimensionless), broadcastable to ``r``.
        r_ext : array-like or scalar
            External input to the rate equation (includes noise if enabled).

        Returns
        -------
        array-like
            Time derivative ``dr/dt`` with unit ``1/ms``.
        """
        return self.delta / u.math.pi / self.tau + (2. * v * r + r_ext) / u.ms

    def dv(self, v, r, v_ext):
        """Right-hand side for the mean membrane potential ``v``.

        Parameters
        ----------
        v : array-like
            Current mean membrane potential (dimensionless).
        r : array-like
            Current firing rate (dimensionless), broadcastable to ``v``.
        v_ext : array-like or scalar
            External input to the potential equation (includes noise if enabled).

        Returns
        -------
        array-like
            Time derivative ``dv/dt`` with unit ``1/ms``.
        """
        return (v ** 2 + self.eta + v_ext + self.J * r * self.tau -
                (u.math.pi * r * self.tau) ** 2) / self.tau

    def derivative(self, state, t, r_ext, v_ext):
        r, v = state
        drdt = self.dr(r, v, r_ext)
        dvdt = self.dv(v, r, v_ext)
        return drdt, dvdt

    def update(self, r_inp=None, v_inp=None):
        """Advance the population by one time step.

        Parameters
        ----------
        r_inp : array-like or scalar or None, optional
            External input to the rate equation. If ``None``, treated as zero.
            If ``noise_r`` is set, its output is added. Default is ``None``.
        v_inp : array-like or scalar or None, optional
            External input to the potential equation. If ``None``, treated as
            zero. If ``noise_v`` is set, its output is added. Default is ``None``.

        Returns
        -------
        array-like
            The updated firing rate ``r`` with the same shape as the internal
            state.

        Notes
        -----
        Performs an exponential-Euler step using ``brainstate.nn.exp_euler_step``
        for both equations and updates ``r`` and ``v`` in-place.
        """
        r_inp = 0. * u.Hz if r_inp is None else r_inp
        if self.noise_r is not None:
            r_inp = r_inp + self.noise_r()
        v_inp = 0. if v_inp is None else v_inp
        if self.noise_v is not None:
            v_inp = v_inp + self.noise_v()

        if self.method == 'exp_euler':
            r = brainstate.nn.exp_euler_step(self.dr, self.r.value, self.v.value, r_inp)
            v = brainstate.nn.exp_euler_step(self.dv, self.v.value, self.r.value, v_inp)
        else:
            r, v = getattr(braintools.quad, f'ode_{self.method}_step')(
                (self.r.value, self.v.value), 0. * u.ms, r_inp, v_inp
            )
        self.r.value = r
        self.v.value = v
        return r
