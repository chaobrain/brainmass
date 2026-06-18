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
    'CoombesByrneStep',
]


class CoombesByrneStep(NeuralMassDynamics):
    r"""Coombes-Byrne next-generation neural mass model (2D).

    Exact mean-field reduction of an infinite population of all-to-all coupled
    quadratic-integrate-and-fire (QIF) / :math:`\theta`-neurons *with
    conductance-based synapses*, obtained through the Ott-Antonsen ansatz [1]_.
    Like the Montbrio-Pazo-Roxin model (:class:`~brainmass.MontbrioPazoRoxinStep`)
    it tracks the population firing rate :math:`r(t)` and mean membrane potential
    :math:`v(t)`, but it adds a synaptic conductance proportional to the firing
    rate, :math:`g = \kappa\,\pi\,r`, which couples reciprocally into both
    equations:

    .. math::

       \begin{aligned}
       \dot r(t) &= \frac{\Delta}{\pi} + 2\,v\,r - g\,r, \\
       \dot v(t) &= v^2 - (\pi r)^2 + \eta + (v_{\mathrm{syn}} - v)\,g + I(t),
       \end{aligned}

    with :math:`g = \kappa\,\pi\,r`. Here :math:`\Delta` is the half-width at
    half-maximum of the Lorentzian background-excitability distribution,
    :math:`\eta` the mean excitability, :math:`\kappa` the synaptic conductance
    scale, :math:`v_{\mathrm{syn}}` the synaptic reversal potential, and
    :math:`I(t)` an external/coupling input to the mean potential.

    The conductance term makes the rate equation quadratically damped in
    :math:`r` (the :math:`-g\,r = -\kappa\pi r^2` term), giving richer dynamics
    than the standard QIF mean field.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the population. An ``int`` or tuple of ``int``; all
        parameters are broadcastable to this shape.
    Delta : Parameter, optional
        HWHM of the Lorentzian excitability distribution (dimensionless).
        Default is ``1.0``.
    eta : Parameter, optional
        Mean background excitability (dimensionless). Default is ``2.0``.
    k : Parameter, optional
        Synaptic conductance scaling :math:`\kappa` (dimensionless). Setting
        ``k = 0`` removes the conductance and recovers the Montbrio-Pazo-Roxin
        field with ``J = 0``. Default is ``1.0``.
    v_syn : Parameter, optional
        Synaptic reversal potential :math:`v_{\mathrm{syn}}` (dimensionless).
        Default is ``-4.0``.
    init_r : Callable, optional
        Initializer for the firing-rate state ``r``. Default is
        ``braintools.init.Constant(0.1)``.
    init_v : Callable, optional
        Initializer for the mean-potential state ``v``. Default is
        ``braintools.init.Constant(0.0)``.
    noise_r : Noise or None, optional
        Additive noise process for the rate dynamics. If provided, its output is
        added to ``r_inp`` at each update. Default is ``None``.
    noise_v : Noise or None, optional
        Additive noise process for the potential dynamics. If provided, its
        output is added to ``v_inp`` at each update. Default is ``None``.
    method : str, optional
        Integration method. Either ``'exp_euler'`` (default) or any method in
        ``braintools.quad`` (e.g. ``'rk4'``, ``'rk2'``, ``'heun'``).

    Attributes
    ----------
    r : brainstate.HiddenState
        Population firing rate (dimensionless). Shape ``(batch?,) + in_size``.
    v : brainstate.HiddenState
        Population mean membrane potential (dimensionless).

    Notes
    -----
    - State variables are dimensionless; the per-variable right-hand sides
      returned by :meth:`dr` / :meth:`dv` carry unit ``1/ms`` so an
      exponential-Euler step with ``dt`` in milliseconds is consistent (the same
      convention used by :class:`~brainmass.FitzHughNagumoStep`).
    - **Relationship to Montbrio-Pazo-Roxin.** With :math:`\kappa = 0` the
      conductance :math:`g` vanishes and the equations collapse to
      :math:`\dot r = \Delta/\pi + 2vr`, :math:`\dot v = v^2 - (\pi r)^2 + \eta + I`,
      i.e. :class:`~brainmass.MontbrioPazoRoxinStep` with recurrent coupling
      ``J = 0`` (at unit time constant). The conductance regime (``k > 0``) is
      what distinguishes the next-generation mass.
    - The model can equivalently be written in the complex Kuramoto-Daido form
      via :math:`Z = (1 - \bar W)/(1 + \bar W)` with :math:`\bar W = \pi r - i v`;
      this implementation uses the real ``(r, v)`` coordinates.

    References
    ----------
    .. [1] S. Coombes and A. Byrne (2019). Next generation neural mass models.
       In *Nonlinear Dynamics in Computational Neuroscience*, pp. 1-16. Springer.
       https://doi.org/10.1007/978-3-319-71048-8_1
    .. [2] E. Montbrió, D. Pazó, A. Roxin (2015). Macroscopic description for
       networks of spiking neurons. Physical Review X, 5:021028.

    Examples
    --------
    .. code-block:: python

       >>> import brainmass
       >>> import brainstate
       >>> import brainunit as u
       >>> model = brainmass.CoombesByrneStep(in_size=1)
       >>> _ = brainstate.nn.init_all_states(model)
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     r = model.update()
       >>> r.shape
       (1,)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # model parameters
        Delta: Parameter = 1.0,
        eta: Parameter = 2.0,
        k: Parameter = 1.0,
        v_syn: Parameter = -4.0,

        # initializers / noise
        init_r: Callable = braintools.init.Constant(0.1),
        init_v: Callable = braintools.init.Constant(0.0),
        noise_r: Noise = None,
        noise_v: Noise = None,
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        # the HWHM of the Lorentzian excitability distribution
        self.Delta = Param.init(Delta, self.varshape)
        # the mean background excitability
        self.eta = Param.init(eta, self.varshape)
        # the synaptic conductance scaling (kappa)
        self.k = Param.init(k, self.varshape)
        # the synaptic reversal potential
        self.v_syn = Param.init(v_syn, self.varshape)

        # initializers and noise
        assert callable(init_r), 'init_r must be callable'
        assert callable(init_v), 'init_v must be callable'
        assert isinstance(noise_r, Noise) or noise_r is None, 'noise_r must be a Noise instance or None'
        assert isinstance(noise_v, Noise) or noise_v is None, 'noise_v must be a Noise instance or None'
        self.init_r = init_r
        self.init_v = init_v
        self.noise_r = noise_r
        self.noise_v = noise_v
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Allocate firing-rate and mean-potential states.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.r = brainstate.HiddenState.init(self.init_r, self.varshape, batch_size)
        self.v = brainstate.HiddenState.init(self.init_v, self.varshape, batch_size)

    def _g(self, r):
        """Synaptic conductance ``g = k * pi * r`` (dimensionless)."""
        return self.k.value() * u.math.pi * r

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
        Delta = self.Delta.value()
        g = self._g(r)
        return (Delta / u.math.pi + 2.0 * v * r - g * r + r_ext) / u.ms

    def dv(self, v, r, v_ext):
        """Right-hand side for the mean membrane potential ``v``.

        Parameters
        ----------
        v : array-like
            Current mean membrane potential (dimensionless).
        r : array-like
            Current firing rate (dimensionless), broadcastable to ``v``.
        v_ext : array-like or scalar
            External input to the potential equation (includes noise/coupling if
            enabled).

        Returns
        -------
        array-like
            Time derivative ``dv/dt`` with unit ``1/ms``.
        """
        eta = self.eta.value()
        v_syn = self.v_syn.value()
        g = self._g(r)
        return (v ** 2 - (u.math.pi * r) ** 2 + eta + (v_syn - v) * g + v_ext) / u.ms

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
            External input to the rate equation. If ``None``, treated as zero. If
            ``noise_r`` is set, its output is added. Default is ``None``.
        v_inp : array-like or scalar or None, optional
            External input to the potential equation (the coupling port). If
            ``None``, treated as zero. If ``noise_v`` is set, its output is added.
            Default is ``None``.

        Returns
        -------
        array-like
            The updated firing rate ``r`` (the coupling observable), same shape as
            the internal state.
        """
        r_inp = 0.0 if r_inp is None else r_inp
        if self.noise_r is not None:
            r_inp = r_inp + self.noise_r()
        v_inp = 0.0 if v_inp is None else v_inp
        if self.noise_v is not None:
            v_inp = v_inp + self.noise_v()

        r, v = self._solve_step(
            exp_euler_specs=(
                (self.dr, self.r.value, self.v.value, r_inp),
                (self.dv, self.v.value, self.r.value, v_inp),
            ),
            ode_state=(self.r.value, self.v.value),
            ode_inputs=(r_inp, v_inp),
        )
        self.r.value = r
        self.v.value = v
        return r
