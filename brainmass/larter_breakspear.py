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
    'LarterBreakspearStep',
]


class LarterBreakspearStep(NeuralMassDynamics):
    r"""Larter-Breakspear conductance-based neural mass model.

    A modified Morris-Lecar mean-field model with three state variables: the mean
    membrane potential of pyramidal cells :math:`V`, the potassium gating variable
    :math:`W`, and the inhibitory interneuron activity :math:`Z` [1]_ [2]_. Voltage
    sigmoidally gates Ca\ :sup:`2+`, Na\ :sup:`+` and K\ :sup:`+` conductances, and a
    voltage-dependent firing rate feeds excitatory recurrence; the model exhibits
    fixed points, limit cycles and chaos depending on parameters.

    Auxiliary (instantaneous) variables — sigmoidal channel gating and firing rates:

    .. math::

       \begin{aligned}
       m_{\mathrm{Ca}} &= \tfrac12\bigl(1 + \tanh\tfrac{V - T_{\mathrm{Ca}}}{\delta_{\mathrm{Ca}}}\bigr),\quad
       m_{\mathrm{Na}}  = \tfrac12\bigl(1 + \tanh\tfrac{V - T_{\mathrm{Na}}}{\delta_{\mathrm{Na}}}\bigr),\quad
       m_{K} = \tfrac12\bigl(1 + \tanh\tfrac{V - T_K}{\delta_K}\bigr), \\
       Q_V &= \tfrac12 Q_V^{\max}\bigl(1 + \tanh\tfrac{V - V_T}{\delta_V}\bigr),\quad
       Q_Z = \tfrac12 Q_Z^{\max}\bigl(1 + \tanh\tfrac{Z - Z_T}{\delta_Z}\bigr).
       \end{aligned}

    State equations (with long-range coupling :math:`c` entering on :math:`V`):

    .. math::

       \begin{aligned}
       \dot V &= t_s\bigl(-I_{\mathrm{Ca}} - I_K - I_L - I_{\mathrm{Na}}
                 - a_{ie} Z Q_Z + a_{ne} I_{\mathrm{ext}}\bigr), \\
       \dot W &= t_s\,\phi\,(m_K - W)/\tau_K, \\
       \dot Z &= t_s\,b\,(a_{ni} I_{\mathrm{ext}} + a_{ei} V Q_V),
       \end{aligned}

    with currents
    :math:`I_{\mathrm{Ca}} = (g_{\mathrm{Ca}} + (1-C) r_{\mathrm{NMDA}} a_{ee} Q_V + C r_{\mathrm{NMDA}} a_{ee} c)\,m_{\mathrm{Ca}}(V - V_{\mathrm{Ca}})`,
    :math:`I_{\mathrm{Na}} = (g_{\mathrm{Na}} m_{\mathrm{Na}} + (1-C) a_{ee} Q_V + C a_{ee} c)(V - V_{\mathrm{Na}})`,
    :math:`I_K = g_K W (V - V_K)` and :math:`I_L = g_L (V - V_L)`.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the population. All parameters broadcast to this shape.
    gCa, gK, gL, gNa : Parameter, optional
        Ca, K, leak and Na conductances. Defaults ``1.1, 2.0, 0.5, 6.7``.
    TCa, TK, TNa : Parameter, optional
        Channel activation thresholds. Defaults ``-0.01, 0.0, 0.3``.
    d_Ca, d_K, d_Na : Parameter, optional
        Channel activation slopes. Defaults ``0.15, 0.3, 0.15``.
    VCa, VK, VL, VNa : Parameter, optional
        Nernst / reversal potentials. Defaults ``1.0, -0.7, -0.5, 0.53``.
    phi : Parameter, optional
        Temperature scaling of K kinetics. Default ``0.7``.
    tau_K : Parameter, optional
        Potassium relaxation time constant (dimensionless). Default ``1.0``.
    aee, aei, aie, ane, ani : Parameter, optional
        Synaptic coupling strengths (E->E, E->I, I->E, ext->E, ext->I). Defaults
        ``0.4, 2.0, 2.0, 1.0, 0.4``.
    b : Parameter, optional
        Inhibitory feedback strength. Default ``0.1``.
    C : Parameter, optional
        Long-range vs local coupling weight. Default ``0.1``.
    Iext : Parameter, optional
        External input current. Default ``0.3``.
    rNMDA : Parameter, optional
        NMDA receptor strength. Default ``0.25``.
    VT, ZT : Parameter, optional
        Firing thresholds for pyramidal / inhibitory cells. Default ``0.0``.
    d_V : Parameter, optional
        Pyramidal firing-rate slope. Governs the dynamical regime: ``d_V < 0.55``
        gives fixed points, ``0.55 < d_V < 0.59`` limit cycles, ``d_V > 0.59``
        chaos. Default ``0.65``.
    d_Z : Parameter, optional
        Inhibitory firing-rate slope. Default ``0.7``.
    QV_max, QZ_max : Parameter, optional
        Maximum firing rates. Default ``1.0``.
    t_scale : Parameter, optional
        Global time-scale factor. Default ``1.0``.
    init_V, init_W, init_Z : Callable, optional
        State initializers. Default ``braintools.init.Constant(0.0)``.
    noise_V : Noise or None, optional
        Additive noise process for the ``V`` (membrane) dynamics. Default ``None``.
    method : str, optional
        Integration method, ``'exp_euler'`` (default) or any ``braintools.quad``
        method (e.g. ``'rk4'``).

    Attributes
    ----------
    V : brainstate.HiddenState
        Mean membrane potential of pyramidal cells (dimensionless).
    W : brainstate.HiddenState
        Potassium channel gating variable (dimensionless).
    Z : brainstate.HiddenState
        Inhibitory interneuron activity (dimensionless).

    Notes
    -----
    - State variables are dimensionless; each right-hand side carries unit ``1/ms``
      so an exponential-Euler step with ``dt`` in milliseconds is consistent.
    - The external input ``V_inp`` is the long-range coupling :math:`c` entering the
      Ca and Na currents exactly as in the upstream TVB model (the local-coupling
      term is taken as zero for an isolated node).

    References
    ----------
    .. [1] R. Larter, B. Speelman, R. M. Worth (1999). A coupled ordinary
       differential equation lattice model for the simulation of epileptic seizures.
       Chaos, 9(3), 795-804.
    .. [2] M. Breakspear, J. R. Terry, K. J. Friston (2003). Modulation of excitatory
       synaptic coupling facilitates synchronization and complex dynamics in a
       biophysical model of neuronal dynamics. Network: Computation in Neural
       Systems, 14(4), 703-732. https://doi.org/10.1088/0954-898X_14_4_305

    Examples
    --------
    .. code-block:: python

       >>> import brainmass
       >>> import brainstate
       >>> import brainunit as u
       >>> model = brainmass.LarterBreakspearStep(in_size=1)
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

        # ion-channel conductances
        gCa: Parameter = 1.1,
        gK: Parameter = 2.0,
        gL: Parameter = 0.5,
        gNa: Parameter = 6.7,
        # channel activation thresholds
        TCa: Parameter = -0.01,
        TK: Parameter = 0.0,
        TNa: Parameter = 0.3,
        # channel activation slopes
        d_Ca: Parameter = 0.15,
        d_K: Parameter = 0.3,
        d_Na: Parameter = 0.15,
        # reversal potentials
        VCa: Parameter = 1.0,
        VK: Parameter = -0.7,
        VL: Parameter = -0.5,
        VNa: Parameter = 0.53,
        # kinetics
        phi: Parameter = 0.7,
        tau_K: Parameter = 1.0,
        # synaptic couplings
        aee: Parameter = 0.4,
        aei: Parameter = 2.0,
        aie: Parameter = 2.0,
        ane: Parameter = 1.0,
        ani: Parameter = 0.4,
        # other
        b: Parameter = 0.1,
        C: Parameter = 0.1,
        Iext: Parameter = 0.3,
        rNMDA: Parameter = 0.25,
        # firing-rate function parameters
        VT: Parameter = 0.0,
        d_V: Parameter = 0.65,
        ZT: Parameter = 0.0,
        d_Z: Parameter = 0.7,
        QV_max: Parameter = 1.0,
        QZ_max: Parameter = 1.0,
        # time scaling
        t_scale: Parameter = 1.0,

        # initializers / noise
        init_V: Callable = braintools.init.Constant(0.0),
        init_W: Callable = braintools.init.Constant(0.0),
        init_Z: Callable = braintools.init.Constant(0.0),
        noise_V: Noise = None,
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        for name, value in dict(
            gCa=gCa, gK=gK, gL=gL, gNa=gNa,
            TCa=TCa, TK=TK, TNa=TNa,
            d_Ca=d_Ca, d_K=d_K, d_Na=d_Na,
            VCa=VCa, VK=VK, VL=VL, VNa=VNa,
            phi=phi, tau_K=tau_K,
            aee=aee, aei=aei, aie=aie, ane=ane, ani=ani,
            b=b, C=C, Iext=Iext, rNMDA=rNMDA,
            VT=VT, d_V=d_V, ZT=ZT, d_Z=d_Z,
            QV_max=QV_max, QZ_max=QZ_max, t_scale=t_scale,
        ).items():
            setattr(self, name, Param.init(value, self.varshape))

        assert callable(init_V), 'init_V must be callable'
        assert callable(init_W), 'init_W must be callable'
        assert callable(init_Z), 'init_Z must be callable'
        assert isinstance(noise_V, Noise) or noise_V is None, 'noise_V must be a Noise or None'
        self.init_V = init_V
        self.init_W = init_W
        self.init_Z = init_Z
        self.noise_V = noise_V
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Allocate the ``V``, ``W`` and ``Z`` states."""
        self.V = brainstate.HiddenState.init(self.init_V, self.varshape, batch_size)
        self.W = brainstate.HiddenState.init(self.init_W, self.varshape, batch_size)
        self.Z = brainstate.HiddenState.init(self.init_Z, self.varshape, batch_size)

    @staticmethod
    def _sigmoid_gate(x, threshold, slope):
        """Half-activation sigmoid ``0.5 * (1 + tanh((x - threshold) / slope))``."""
        return 0.5 * (1.0 + u.math.tanh((x - threshold) / slope))

    def QV(self, V):
        """Pyramidal-cell firing rate as a function of ``V`` (dimensionless)."""
        return self.QV_max.value() * self._sigmoid_gate(V, self.VT.value(), self.d_V.value())

    def QZ(self, Z):
        """Inhibitory-cell firing rate as a function of ``Z`` (dimensionless)."""
        return self.QZ_max.value() * self._sigmoid_gate(Z, self.ZT.value(), self.d_Z.value())

    def dV(self, V, W, Z, V_inp):
        """Right-hand side for the membrane potential ``V`` (unit ``1/ms``).

        ``V_inp`` is the long-range coupling :math:`c` entering the Ca/Na currents.
        """
        C = self.C.value()
        aee = self.aee.value()
        rNMDA = self.rNMDA.value()
        QV = self.QV(V)
        m_Ca = self._sigmoid_gate(V, self.TCa.value(), self.d_Ca.value())
        m_Na = self._sigmoid_gate(V, self.TNa.value(), self.d_Na.value())

        I_Ca = (
            self.gCa.value()
            + (1.0 - C) * rNMDA * aee * QV
            + C * rNMDA * aee * V_inp
        ) * m_Ca * (V - self.VCa.value())
        I_K = self.gK.value() * W * (V - self.VK.value())
        I_L = self.gL.value() * (V - self.VL.value())
        I_Na = (
            self.gNa.value() * m_Na
            + (1.0 - C) * aee * QV
            + C * aee * V_inp
        ) * (V - self.VNa.value())
        I_inh = self.aie.value() * Z * self.QZ(Z)
        I_ext = self.ane.value() * self.Iext.value()

        return self.t_scale.value() * (-I_Ca - I_K - I_L - I_Na - I_inh + I_ext) / u.ms

    def dW(self, W, V):
        """Right-hand side for the potassium gating variable ``W`` (unit ``1/ms``)."""
        m_K = self._sigmoid_gate(V, self.TK.value(), self.d_K.value())
        return self.t_scale.value() * self.phi.value() * (m_K - W) / self.tau_K.value() / u.ms

    def dZ(self, Z, V):
        """Right-hand side for the inhibitory activity ``Z`` (unit ``1/ms``)."""
        drive = self.ani.value() * self.Iext.value() + self.aei.value() * V * self.QV(V)
        return self.t_scale.value() * self.b.value() * drive / u.ms

    def derivative(self, state, t, V_inp):
        V, W, Z = state
        return self.dV(V, W, Z, V_inp), self.dW(W, V), self.dZ(Z, V)

    def update(self, V_inp=None):
        """Advance ``(V, W, Z)`` by one time step.

        Parameters
        ----------
        V_inp : array-like or scalar or None, optional
            Long-range coupling input on the membrane equation. If ``None``,
            treated as zero. If ``noise_V`` is set, its output is added.

        Returns
        -------
        array-like
            The updated membrane potential ``V`` (the coupling observable).
        """
        V_inp = 0.0 if V_inp is None else V_inp
        if self.noise_V is not None:
            V_inp = V_inp + self.noise_V()

        V, W, Z = self._solve_step(
            exp_euler_specs=(
                (self.dV, self.V.value, self.W.value, self.Z.value, V_inp),
                (self.dW, self.W.value, self.V.value),
                (self.dZ, self.Z.value, self.V.value),
            ),
            ode_state=(self.V.value, self.W.value, self.Z.value),
            ode_inputs=(V_inp,),
        )
        self.V.value = V
        self.W.value = W
        self.Z.value = Z
        return V
