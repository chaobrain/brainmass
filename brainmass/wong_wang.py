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


import brainunit as u
import jax.numpy as jnp

import braintools
import brainstate
from .noise import Noise
from ._typing import Initializer

__all__ = [
    'WongWangModel',
]


class WongWangModel(brainstate.nn.Dynamics):
    r"""
    The Wong-Wang neural mass model for perceptual decision-making.
    
    This model implements the reduced two-variable neural mass model described in:
    Wong, K.-F. & Wang, X.-J. "A Recurrent Network Mechanism of Time Integration 
    in Perceptual Decisions." J. Neurosci. 26, 1314–1328 (2006).
    
    The model describes the competitive dynamics between two neural populations 
    (e.g., left vs right motion detection) through slow NMDA-mediated recurrent 
    excitation, capturing the temporal integration of sensory evidence during 
    perceptual decision-making.

    Mathematical Description
    ========================
    
    The model is governed by two coupled differential equations for the synaptic 
    gating variables S1 and S2 of the competing neural populations:
    
    .. math::
        \frac{dS_1}{dt} = -\frac{S_1}{\tau_S} + (1-S_1)\gamma r_1
        
    .. math::
        \frac{dS_2}{dt} = -\frac{S_2}{\tau_S} + (1-S_2)\gamma r_2
    
    where the firing rates r1 and r2 are given by:
    
    .. math::
        r_i = \phi(I_i) = \begin{cases}
            a(I_i - \theta) & \text{if } I_i > \theta \\
            0 & \text{otherwise}
        \end{cases}
    
    The total input current to each population is:
    
    .. math::
        I_1 = J_{N,11}S_1 - J_{N,12}S_2 + J_{A,ext}\mu_0(1+c) + I_{noise,1}
        
    .. math::
        I_2 = J_{N,22}S_2 - J_{N,21}S_1 + J_{A,ext}\mu_0(1-c) + I_{noise,2}

    Parameters
    ==========
    
    Synaptic Parameters:
        - $\tau_S$ = 100 ms : NMDA receptor time constant
        - $\gamma$ = 0.641 : Saturation factor for synaptic gating
        
    Input-Output Function:
        - $\alpha$ = 270 Hz/nA : Gain parameter
        - $\theta$ = 0.31 nA : Firing threshold
        
    Network Connectivity (typical values):
        - J_N,11 = J_N,22 = 0.2609 nA : Self-excitation strength
        - J_N,12 = J_N,21 = 0.0497 nA : Cross-inhibition strength
        - J_A,ext = 0.00052 nA : External input strength
        
    External Input:
        - $\mu_0$ = 30 Hz : Baseline external input rate
        - $c \in [-1, 1]$ : Motion coherence (stimulus strength)


    Network Behavior
    ================
    
    The model exhibits rich dynamics depending on the stimulus strength:
    
    1. **Spontaneous State**: At c=0 (no coherence), both populations have equal 
       activity, representing uncertainty.
       
    2. **Decision State**: For $|c| > 0$, one population gradually wins the competition,
       representing a perceptual choice.
       
    3. **Bistability**: The network can exhibit bistable attractor dynamics where
       the system can remain in either of two decision states.
       
    4. **Integration Time**: The slow NMDA dynamics ($\tau_S$ = 100ms) enable temporal
       integration of sensory evidence over hundreds of milliseconds.


    Usage Example
    =============
    
    >>> model = WongWangModel(in_size=100)
    >>> model.init_state(batch_size=1)
    >>> 
    >>> # Simulate decision making with rightward motion (c=0.32)
    >>> for t in range(1000):
    ...     output = model.update(coherence=0.32)
    ...     # S1 and S2 activities accessible via model.S1.value, model.S2.value
    
    References
    ==========
    
    - Wong, K.-F. & Wang, X.-J. A Recurrent Network Mechanism of Time Integration 
      in Perceptual Decisions. J. Neurosci. 26, 1314–1328 (2006).
    - Deco, G. et al. The role of rhythm in cognition. Front. Hum. Neurosci. 5, 29 (2011).
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # NMDA synaptic parameters
        tau_S: Initializer = 0.1 * u.second,  # NMDA time constant (ms)
        gamma: Initializer = 0.641,  # saturation factor

        # Input-output function parameters  
        a: Initializer = 270. * (u.Hz / u.nA),  # gain (Hz/nA)
        theta: Initializer = 0.31 * u.nA,  # firing threshold (nA)

        # Network connectivity (nA)
        J_N11: Initializer = 0.2609 * u.nA,  # self-excitation pop 1
        J_N22: Initializer = 0.2609 * u.nA,  # self-excitation pop 2
        J_N12: Initializer = 0.0497 * u.nA,  # cross-inhibition 2->1
        J_N21: Initializer = 0.0497 * u.nA,  # cross-inhibition 2->1
        J_A_ext: Initializer = 0.0002243 * (u.nA / u.Hz),  # external input strength (nA·Hz⁻¹)

        # External input
        mu_0: Initializer = 30. * u.Hz,  # baseline input rate (Hz)
        I_0: Initializer = 0.3255 * u.nA,  # background input current (nA)

        # Noise processes
        noise_s1: Noise = None,
        noise_s2: Noise = None,
    ):
        super().__init__(in_size=in_size)

        # NMDA parameters
        self.tau_S = braintools.init.param(tau_S, self.varshape)
        self.gamma = braintools.init.param(gamma, self.varshape)

        # I-O function parameters
        self.a = braintools.init.param(a, self.varshape)
        self.theta = braintools.init.param(theta, self.varshape)

        # Network connectivity
        self.J_N11 = braintools.init.param(J_N11, self.varshape)
        self.J_N22 = braintools.init.param(J_N22, self.varshape)
        self.J_N12 = braintools.init.param(J_N12, self.varshape)
        self.J_N21 = braintools.init.param(J_N21, self.varshape)
        self.J_A_ext = braintools.init.param(J_A_ext, self.varshape)

        # External input
        self.mu_0 = braintools.init.param(mu_0, self.varshape)
        self.I_0 = braintools.init.param(I_0, self.varshape)

        # Noise processes
        self.noise_s1 = noise_s1
        self.noise_s2 = noise_s2

    def init_state(self, batch_size=None, **kwargs):
        """Initialize the synaptic gating variables S1 and S2."""
        size = self.varshape if batch_size is None else (batch_size,) + self.varshape
        self.S1 = brainstate.HiddenState(braintools.init.param(jnp.zeros, size))
        self.S2 = brainstate.HiddenState(braintools.init.param(jnp.zeros, size))

    def reset_state(self, batch_size=None, **kwargs):
        """Reset the synaptic gating variables to initial conditions."""
        size = self.varshape if batch_size is None else (batch_size,) + self.varshape
        self.S1.value = braintools.init.param(jnp.zeros, size)
        self.S2.value = braintools.init.param(jnp.zeros, size)

    def phi(self, I):
        """
        Input-output transfer function (f-I curve).
        
        Args:
            I: Input current (nA)
            
        Returns:
            Firing rate (Hz)
        """
        return u.math.where(I > self.theta, self.a * (I - self.theta), 0. * u.Hz)

    def compute_inputs(self, coherence=0., noise_1_val=0. * u.nA, noise_2_val=0. * u.nA):
        """
        Compute total input currents to both populations.
        
        Args:
            coherence: Motion coherence level, c ∈ [-1, 1]
            noise_1_val: Noise input to population 1
            noise_2_val: Noise input to population 2
            
        Returns:
            Tuple of (I1, I2) input currents
        """
        # External stimulus inputs
        I_stim_1 = self.J_A_ext * self.mu_0 * (1 + coherence)
        I_stim_2 = self.J_A_ext * self.mu_0 * (1 - coherence)

        # Recurrent inputs
        I_rec_1 = self.J_N11 * self.S1.value - self.J_N12 * self.S2.value
        I_rec_2 = self.J_N22 * self.S2.value - self.J_N21 * self.S1.value

        # Total inputs (including background current I_0)
        I1 = I_rec_1 + I_stim_1 + self.I_0 + noise_1_val
        I2 = I_rec_2 + I_stim_2 + self.I_0 + noise_2_val

        return I1, I2

    def dS1_dt(self, S1, r1):
        """Differential equation for synaptic gating variable S1."""
        return (-S1 / self.tau_S + (1 - S1) * self.gamma * r1).to(u.Hz)

    def dS2_dt(self, S2, r2):
        """Differential equation for synaptic gating variable S2."""
        return (-S2 / self.tau_S + (1 - S2) * self.gamma * r2).to(u.Hz)

    def update(self, coherence=0.):
        """
        Update the Wong-Wang model for one time step.
        
        Args:
            coherence: Motion coherence level, c ∈ [-1, 1]. Positive values 
                      favor population 1, negative values favor population 2.

        Returns:
            Tuple of (r1, r2) firing rates of the two populations
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
        """
        Get the current decision based on firing rate threshold.
        
        Args:
            threshold: Firing rate threshold for decision (Hz)
            
        Returns:
            Decision: 1 if population 1 wins, -1 if population 2 wins, 0 if undecided
        """
        I1, I2 = self.compute_inputs()
        r1 = self.phi(I1)
        r2 = self.phi(I2)
        return jnp.where((r1 > threshold) & (r1 > r2), 1, jnp.where((r2 > threshold) & (r2 > r1), -1, 0))
