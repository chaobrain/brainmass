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


import brainunit as u
import jax.numpy as jnp

import brainstate
from brainstate.nn import Param
from .noise import Noise
from .typing import Parameter

__all__ = [
    'WongWangStep',
]


class WongWangStep(brainstate.nn.Dynamics):
    r"""Wong-Wang reduced neural-mass model for perceptual decision-making.

    Implements the reduced two-variable model of Wong & Wang (2006). It
    describes the competitive dynamics between two neural populations (e.g.
    left- vs right-motion detectors) through slow NMDA-mediated recurrent
    excitation, capturing the temporal integration of sensory evidence during
    perceptual decision-making.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Shape of the population (number of independent decision units).
    tau_S : Parameter, default ``0.1 * u.second``
        NMDA receptor time constant.
    gamma : Parameter, default 0.641
        Saturation factor for the synaptic gating variables.
    a : Parameter, default ``270. * (u.Hz / u.nA)``
        Gain of the input-output (f-I) function.
    theta : Parameter, default ``0.31 * u.nA``
        Firing threshold of the input-output function.
    J_N11 : Parameter, default ``0.2609 * u.nA``
        Self-excitation strength of population 1.
    J_N22 : Parameter, default ``0.2609 * u.nA``
        Self-excitation strength of population 2.
    J_N12 : Parameter, default ``0.0497 * u.nA``
        Cross-inhibition strength from population 2 to population 1.
    J_N21 : Parameter, default ``0.0497 * u.nA``
        Cross-inhibition strength from population 1 to population 2.
    J_A_ext : Parameter, default ``0.0002243 * (u.nA / u.Hz)``
        External input strength (AMPA).
    mu_0 : Parameter, default ``30. * u.Hz``
        Baseline external input rate.
    I_0 : Parameter, default ``0.3255 * u.nA``
        Background input current.
    noise_s1 : Noise, optional
        Noise process added to the input current of population 1. ``None``
        disables noise on that population.
    noise_s2 : Noise, optional
        Noise process added to the input current of population 2. ``None``
        disables noise on that population.

    Notes
    -----
    The model evolves the synaptic gating variables :math:`S_1` and :math:`S_2`
    of the two competing populations:

    .. math::

        \frac{dS_1}{dt} = -\frac{S_1}{\tau_S} + (1-S_1)\gamma r_1 , \qquad
        \frac{dS_2}{dt} = -\frac{S_2}{\tau_S} + (1-S_2)\gamma r_2 ,

    where the firing rates follow the threshold-linear f-I curve

    .. math::

        r_i = \phi(I_i) = \begin{cases}
            a(I_i - \theta) & \text{if } I_i > \theta \\
            0 & \text{otherwise} ,
        \end{cases}

    and the total input currents are

    .. math::

        I_1 = J_{N,11}S_1 - J_{N,12}S_2 + J_{A,ext}\mu_0(1+c) + I_0 + I_{noise,1} ,

    .. math::

        I_2 = J_{N,22}S_2 - J_{N,21}S_1 + J_{A,ext}\mu_0(1-c) + I_0 + I_{noise,2} ,

    with motion coherence :math:`c \in [-1, 1]`.

    The network exhibits a spontaneous (symmetric) state at :math:`c = 0`, a
    decision state in which one population wins for :math:`|c| > 0`, bistable
    attractor dynamics, and slow (:math:`\tau_S = 100` ms) temporal integration
    of evidence over hundreds of milliseconds.

    References
    ----------
    .. [1] Wong, K.-F. & Wang, X.-J. "A Recurrent Network Mechanism of Time
           Integration in Perceptual Decisions." J. Neurosci. 26, 1314-1328
           (2006).
    .. [2] Deco, G. et al. "The role of rhythm in cognition." Front. Hum.
           Neurosci. 5, 29 (2011).

    Examples
    --------
    .. code-block:: python

        >>> import brainstate
        >>> import brainmass
        >>> model = brainmass.WongWangStep(in_size=100)
        >>> model.init_all_states(batch_size=1)
        >>> # Simulate decision-making with rightward motion (c=0.32).
        >>> for t in range(1000):
        ...     r1, r2 = model.update(coherence=0.32)
        >>> # S1 and S2 activities are accessible via model.S1.value / model.S2.value.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # NMDA synaptic parameters
        tau_S: Parameter = 0.1 * u.second,  # NMDA time constant (ms)
        gamma: Parameter = 0.641,  # saturation factor

        # Input-output function parameters
        a: Parameter = 270. * (u.Hz / u.nA),  # gain (Hz/nA)
        theta: Parameter = 0.31 * u.nA,  # firing threshold (nA)

        # Network connectivity (nA)
        J_N11: Parameter = 0.2609 * u.nA,  # self-excitation pop 1
        J_N22: Parameter = 0.2609 * u.nA,  # self-excitation pop 2
        J_N12: Parameter = 0.0497 * u.nA,  # cross-inhibition 2->1
        J_N21: Parameter = 0.0497 * u.nA,  # cross-inhibition 2->1
        J_A_ext: Parameter = 0.0002243 * (u.nA / u.Hz),  # external input strength (nA·Hz⁻¹)

        # External input
        mu_0: Parameter = 30. * u.Hz,  # baseline input rate (Hz)
        I_0: Parameter = 0.3255 * u.nA,  # background input current (nA)

        # Noise processes
        noise_s1: Noise = None,
        noise_s2: Noise = None,
    ):
        super().__init__(in_size=in_size)

        # NMDA parameters
        self.tau_S = Param.init(tau_S, self.varshape)
        self.gamma = Param.init(gamma, self.varshape)

        # I-O function parameters
        self.a = Param.init(a, self.varshape)
        self.theta = Param.init(theta, self.varshape)

        # Network connectivity
        self.J_N11 = Param.init(J_N11, self.varshape)
        self.J_N22 = Param.init(J_N22, self.varshape)
        self.J_N12 = Param.init(J_N12, self.varshape)
        self.J_N21 = Param.init(J_N21, self.varshape)
        self.J_A_ext = Param.init(J_A_ext, self.varshape)

        # External input
        self.mu_0 = Param.init(mu_0, self.varshape)
        self.I_0 = Param.init(I_0, self.varshape)

        # Noise processes
        self.noise_s1 = noise_s1
        self.noise_s2 = noise_s2

    def init_state(self, batch_size=None, **kwargs):
        """Initialize the synaptic gating variables S1 and S2."""
        self.S1 = brainstate.HiddenState.init(jnp.zeros, self.varshape, batch_size)
        self.S2 = brainstate.HiddenState.init(jnp.zeros, self.varshape, batch_size)

    def phi(self, I):
        """Threshold-linear input-output transfer function (f-I curve).

        Parameters
        ----------
        I : ArrayLike
            Input current (in units of ``nA``).

        Returns
        -------
        ArrayLike
            Firing rate (in units of ``Hz``); ``a * (I - theta)`` above
            threshold and ``0`` otherwise.
        """
        theta = self.theta.value()
        a = self.a.value()
        return u.math.where(I > theta, a * (I - theta), 0. * u.Hz)

    def compute_inputs(self, coherence=0., noise_1_val=0. * u.nA, noise_2_val=0. * u.nA):
        """Compute the total input currents to both populations.

        Parameters
        ----------
        coherence : float, default 0.0
            Motion coherence level, ``c`` in ``[-1, 1]``.
        noise_1_val : ArrayLike, default ``0. * u.nA``
            Noise input added to population 1.
        noise_2_val : ArrayLike, default ``0. * u.nA``
            Noise input added to population 2.

        Returns
        -------
        I1 : ArrayLike
            Total input current to population 1.
        I2 : ArrayLike
            Total input current to population 2.
        """
        J_A_ext = self.J_A_ext.value()
        mu_0 = self.mu_0.value()
        J_N11 = self.J_N11.value()
        J_N22 = self.J_N22.value()
        J_N12 = self.J_N12.value()
        J_N21 = self.J_N21.value()
        I_0 = self.I_0.value()

        # External stimulus inputs
        I_stim_1 = J_A_ext * mu_0 * (1 + coherence)
        I_stim_2 = J_A_ext * mu_0 * (1 - coherence)

        # Recurrent inputs
        I_rec_1 = J_N11 * self.S1.value - J_N12 * self.S2.value
        I_rec_2 = J_N22 * self.S2.value - J_N21 * self.S1.value

        # Total inputs (including background current I_0)
        I1 = I_rec_1 + I_stim_1 + I_0 + noise_1_val
        I2 = I_rec_2 + I_stim_2 + I_0 + noise_2_val

        return I1, I2

    def dS1_dt(self, S1, r1):
        """Differential equation for synaptic gating variable S1."""
        tau_S = self.tau_S.value()
        gamma = self.gamma.value()
        return (-S1 / tau_S + (1 - S1) * gamma * r1).to(u.Hz)

    def dS2_dt(self, S2, r2):
        """Differential equation for synaptic gating variable S2."""
        tau_S = self.tau_S.value()
        gamma = self.gamma.value()
        return (-S2 / tau_S + (1 - S2) * gamma * r2).to(u.Hz)

    def update(self, coherence=0.):
        """Advance the Wong-Wang model by one time step.

        Parameters
        ----------
        coherence : float, default 0.0
            Motion coherence level, ``c`` in ``[-1, 1]``. Positive values
            favour population 1, negative values favour population 2.

        Returns
        -------
        r1 : ArrayLike
            Firing rate of population 1 (in units of ``Hz``).
        r2 : ArrayLike
            Firing rate of population 2 (in units of ``Hz``).
        """
        # Add noise if specified
        noise_1_val = 0. * u.nA if self.noise_s1 is None else self.noise_s1()
        noise_2_val = 0. * u.nA if self.noise_s2 is None else self.noise_s2()

        # Compute input currents
        I1, I2 = self.compute_inputs(coherence, noise_1_val, noise_2_val)

        # Compute firing rates
        r1 = self.phi(I1)
        r2 = self.phi(I2)

        # Update synaptic gating variables using Euler integration
        self.S1.value = brainstate.nn.exp_euler_step(self.dS1_dt, self.S1.value, r1)
        self.S2.value = brainstate.nn.exp_euler_step(self.dS2_dt, self.S2.value, r2)

        # Clamp S values to [0, 1] range
        self.S1.value = jnp.clip(self.S1.value, 0., 1.)
        self.S2.value = jnp.clip(self.S2.value, 0., 1.)

        return r1, r2

    def get_decision(self, threshold=15. * u.Hz):
        """Return the current decision based on a firing-rate threshold.

        Parameters
        ----------
        threshold : ArrayLike, default ``15. * u.Hz``
            Firing-rate threshold a population must exceed to count as the
            winner.

        Returns
        -------
        ArrayLike
            Decision code: ``1`` if population 1 wins, ``-1`` if population 2
            wins, and ``0`` if undecided.
        """
        I1, I2 = self.compute_inputs()
        r1 = self.phi(I1)
        r2 = self.phi(I2)
        return jnp.where((r1 > threshold) & (r1 > r2), 1, jnp.where((r2 > threshold) & (r2 > r1), -1, 0))
