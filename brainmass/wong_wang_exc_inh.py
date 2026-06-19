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
    'WongWangExcInhStep',
]


class WongWangExcInhStep(NeuralMassDynamics):
    r"""Reduced Wong-Wang excitatory-inhibitory mean-field model (Deco et al., 2014).

    A two-population reduction of the Wong-Wang (2006) attractor network with
    *separate* excitatory and inhibitory pools, coupled locally and (in a network)
    through long-range excitatory connections. It is the BOLD/FC workhorse of
    whole-brain modelling: the local excitation-inhibition balance set by the
    feedback-inhibition weight :math:`J_i` shapes the resting-state dynamics
    [1]_ [2]_. The model integrates the NMDA synaptic gating variables
    :math:`S_E, S_I`:

    .. math::

       \begin{aligned}
       x_E &= w_p J_N S_E - J_i S_I + W_E I_o + G J_N c + I_{\mathrm{ext}}, \\
       H_E &= \frac{a_E x_E - b_E}{1 - e^{-d_E (a_E x_E - b_E)}}, \qquad
       \dot S_E = -\frac{S_E}{\tau_E} + (1 - S_E)\,\gamma_E\,H_E, \\[4pt]
       x_I &= J_N S_E - S_I + W_I I_o + \lambda G J_N c, \\
       H_I &= \frac{a_I x_I - b_I}{1 - e^{-d_I (a_I x_I - b_I)}}, \qquad
       \dot S_I = -\frac{S_I}{\tau_I} + \gamma_I\,H_I,
       \end{aligned}

    where :math:`c` is the (long-range + local) coupling input, :math:`H_{E/I}` are
    the population firing rates (the f-I transfer functions), and
    :math:`G J_N c` is the scaled coupling current shared by both populations
    (the inhibitory pool receives a fraction :math:`\lambda` of it).

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node. An ``int`` or tuple of ``int``; all parameters
        broadcast to this shape.
    a_e, b_e, d_e : Parameter, optional
        Excitatory f-I curve gain / shift / scaling. Defaults ``310.0``, ``125.0``,
        ``0.160``.
    gamma_e : Parameter, optional
        Excitatory kinetic parameter. Default ``0.641 / 1000``.
    tau_e : Parameter, optional
        Excitatory NMDA decay time constant (ms). Default ``100.0``.
    w_p : Parameter, optional
        Local excitatory recurrence weight. Default ``1.4``.
    W_e : Parameter, optional
        Excitatory external-input scaling. Default ``1.0``.
    a_i, b_i, d_i : Parameter, optional
        Inhibitory f-I curve gain / shift / scaling. Defaults ``615.0``, ``177.0``,
        ``0.087``.
    gamma_i : Parameter, optional
        Inhibitory kinetic parameter. Default ``1.0 / 1000``.
    tau_i : Parameter, optional
        Inhibitory time constant (ms). Default ``10.0``.
    W_i : Parameter, optional
        Inhibitory external-input scaling. Default ``0.7``.
    J_N : Parameter, optional
        NMDA synaptic coupling current (nA). Default ``0.15``.
    J_i : Parameter, optional
        Local feedback-inhibition weight (the E/I-balance knob). Default ``1.0``.
    I_o : Parameter, optional
        Background (overall effective external) input current. Default ``0.382``.
    I_ext : Parameter, optional
        Additional external stimulation current to the excitatory pool. Default ``0.0``.
    G : Parameter, optional
        Global coupling strength applied to the network input. Default ``2.0``.
    lamda : Parameter, optional
        Fraction of the coupling current received by the inhibitory pool. Default ``0.0``.
    init_S_e, init_S_i : Callable, optional
        State initializers. Defaults ``braintools.init.Constant(0.001)``.
    noise_e, noise_i : Noise or None, optional
        Additive noise processes applied directly to the :math:`S_E` / :math:`S_I`
        derivatives. Default ``None``.
    method : str, optional
        Integration method, ``'exp_euler'`` (default) or any ``braintools.quad``
        method (e.g. ``'rk4'``).

    Attributes
    ----------
    S_e : brainstate.HiddenState
        Excitatory synaptic gating (dimensionless, in ``[0, 1]`` at the fixed
        point). Shape ``(batch?,) + in_size``.
    S_i : brainstate.HiddenState
        Inhibitory synaptic gating.

    Notes
    -----
    - State variables are dimensionless; :meth:`dS_e` / :meth:`dS_i` carry unit
      ``1/ms`` so an exponential-Euler step with ``dt`` in milliseconds is
      consistent (``tau`` are in ms, ``gamma`` per-ms).
    - :meth:`H_e` / :meth:`H_i` return the population firing rates (the auxiliary
      transfer-function values) at the current state; they are recomputed each
      :meth:`update` rather than stored.
    - The f-I transfer function has a removable singularity where its argument
      :math:`a x - b = 0`; in the physiological (low-activity) operating regime the
      argument stays well away from zero, so the literal TVB form is used.

    References
    ----------
    .. [1] K.-F. Wong, X.-J. Wang (2006). A recurrent network mechanism of time
       integration in perceptual decisions. Journal of Neuroscience, 26(4),
       1314-1328.
    .. [2] G. Deco, A. Ponce-Alvarez, P. Hagmann, G. L. Romani, D. Mantini,
       M. Corbetta (2014). How local excitation-inhibition ratio impacts the whole
       brain dynamics. Journal of Neuroscience, 34(23), 7886-7898.
       https://doi.org/10.1523/JNEUROSCI.5068-13.2014

    Examples
    --------
    .. code-block:: python

       >>> import brainmass
       >>> import brainstate
       >>> import brainunit as u
       >>> model = brainmass.WongWangExcInhStep(in_size=1)
       >>> _ = brainstate.nn.init_all_states(model)
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     S_e = model.update()
       >>> S_e.shape
       (1,)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # excitatory population
        a_e: Parameter = 310.0,
        b_e: Parameter = 125.0,
        d_e: Parameter = 0.160,
        gamma_e: Parameter = 0.641 / 1000.0,
        tau_e: Parameter = 100.0,
        w_p: Parameter = 1.4,
        W_e: Parameter = 1.0,
        # inhibitory population
        a_i: Parameter = 615.0,
        b_i: Parameter = 177.0,
        d_i: Parameter = 0.087,
        gamma_i: Parameter = 1.0 / 1000.0,
        tau_i: Parameter = 10.0,
        W_i: Parameter = 0.7,
        # synaptic weights
        J_N: Parameter = 0.15,
        J_i: Parameter = 1.0,
        # external input
        I_o: Parameter = 0.382,
        I_ext: Parameter = 0.0,
        # coupling
        G: Parameter = 2.0,
        lamda: Parameter = 0.0,

        # initializers / noise
        init_S_e: Callable = braintools.init.Constant(0.001),
        init_S_i: Callable = braintools.init.Constant(0.001),
        noise_e: Noise = None,
        noise_i: Noise = None,
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        for name, value in dict(
            a_e=a_e, b_e=b_e, d_e=d_e, gamma_e=gamma_e, tau_e=tau_e, w_p=w_p, W_e=W_e,
            a_i=a_i, b_i=b_i, d_i=d_i, gamma_i=gamma_i, tau_i=tau_i, W_i=W_i,
            J_N=J_N, J_i=J_i, I_o=I_o, I_ext=I_ext, G=G, lamda=lamda,
        ).items():
            setattr(self, name, Param.init(value, self.varshape))

        for init in (init_S_e, init_S_i):
            assert callable(init), 'state initializers must be callable'
        assert isinstance(noise_e, Noise) or noise_e is None, 'noise_e must be a Noise instance or None'
        assert isinstance(noise_i, Noise) or noise_i is None, 'noise_i must be a Noise instance or None'
        self.init_S_e = init_S_e
        self.init_S_i = init_S_i
        self.noise_e = noise_e
        self.noise_i = noise_i
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Allocate the excitatory (``S_e``) and inhibitory (``S_i``) gating states.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.S_e = brainstate.HiddenState.init(self.init_S_e, self.varshape, batch_size)
        self.S_i = brainstate.HiddenState.init(self.init_S_i, self.varshape, batch_size)

    @staticmethod
    def _transfer(x, a, b, d):
        """f-I transfer function ``(a x - b) / (1 - exp(-d (a x - b)))`` (firing rate)."""
        xs = a * x - b
        return xs / (1.0 - u.math.exp(-d * xs))

    def _x_e(self, S_e, S_i, coupling):
        """Excitatory input current ``x_E``."""
        coup_total = self.G.value() * self.J_N.value() * coupling
        return (
            self.w_p.value() * self.J_N.value() * S_e
            - self.J_i.value() * S_i
            + self.W_e.value() * self.I_o.value()
            + coup_total
            + self.I_ext.value()
        )

    def _x_i(self, S_e, S_i, coupling):
        """Inhibitory input current ``x_I``."""
        coup_total = self.G.value() * self.J_N.value() * coupling
        return (
            self.J_N.value() * S_e
            - S_i
            + self.W_i.value() * self.I_o.value()
            + self.lamda.value() * coup_total
        )

    def H_e(self, coupling=0.0):
        """Excitatory population firing rate ``H_E`` at the current state."""
        x_e = self._x_e(self.S_e.value, self.S_i.value, coupling)
        return self._transfer(x_e, self.a_e.value(), self.b_e.value(), self.d_e.value())

    def H_i(self, coupling=0.0):
        """Inhibitory population firing rate ``H_I`` at the current state."""
        x_i = self._x_i(self.S_e.value, self.S_i.value, coupling)
        return self._transfer(x_i, self.a_i.value(), self.b_i.value(), self.d_i.value())

    def dS_e(self, S_e, S_i, coupling, add=0.0):
        """Right-hand side for the excitatory gating ``S_e`` (unit ``1/ms``)."""
        x_e = self._x_e(S_e, S_i, coupling)
        h_e = self._transfer(x_e, self.a_e.value(), self.b_e.value(), self.d_e.value())
        det = -S_e / self.tau_e.value() + (1.0 - S_e) * h_e * self.gamma_e.value()
        return (det + add) / u.ms

    def dS_i(self, S_i, S_e, coupling, add=0.0):
        """Right-hand side for the inhibitory gating ``S_i`` (unit ``1/ms``)."""
        x_i = self._x_i(S_e, S_i, coupling)
        h_i = self._transfer(x_i, self.a_i.value(), self.b_i.value(), self.d_i.value())
        det = -S_i / self.tau_i.value() + h_i * self.gamma_i.value()
        return (det + add) / u.ms

    def derivative(self, state, t, coupling, add_e=0.0, add_i=0.0):
        S_e, S_i = state
        return self.dS_e(S_e, S_i, coupling, add_e), self.dS_i(S_i, S_e, coupling, add_i)

    def update(self, coupling=None):
        """Advance the excitatory/inhibitory gating by one time step.

        Parameters
        ----------
        coupling : array-like or scalar or None, optional
            Network coupling input (scaled internally by ``G * J_N``). If ``None``,
            treated as zero. Default is ``None``.

        Returns
        -------
        array-like
            The updated excitatory gating ``S_e`` (the coupling observable), same
            shape as the internal state.
        """
        coupling = 0.0 if coupling is None else coupling
        add_e = self.noise_e() if self.noise_e is not None else 0.0
        add_i = self.noise_i() if self.noise_i is not None else 0.0

        S_e, S_i = self._solve_step(
            exp_euler_specs=(
                (self.dS_e, self.S_e.value, self.S_i.value, coupling, add_e),
                (self.dS_i, self.S_i.value, self.S_e.value, coupling, add_i),
            ),
            ode_state=(self.S_e.value, self.S_i.value),
            ode_inputs=(coupling, add_e, add_i),
        )
        self.S_e.value = S_e
        self.S_i.value = S_i
        return S_e
