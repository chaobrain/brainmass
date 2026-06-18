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

from typing import Callable, Union, Optional, Tuple

import brainstate
import braintools
import brainunit as u
import jax.nn
import numpy as np
from brainstate import HiddenState
from brainstate.nn import (
    exp_euler_step, Param, Dynamics, Module, init_maybe_prefetch, Delay,
)

from ..coupling import additive_coupling, AdditiveConn, DelayedAdditiveConn
from ..noise import Noise, GaussianNoise
from ..typing import Parameter, Initializer
from ..utils import delay_index


Array = brainstate.typing.ArrayLike
Size = brainstate.typing.Size
Prefetch = Union[
    brainstate.nn.PrefetchDelayAt,
    brainstate.nn.PrefetchDelay,
    brainstate.nn.Prefetch,
    Callable,
]


class LaplacianConnectivity(Module):
    r"""
    Laplacian connectivity for multi-region Jansen-Rit neural mass models.

    This class implements a three-pathway graph Laplacian coupling mechanism
    designed for spatially-extended Jansen-Rit neural mass networks. It computes
    coupling inputs for the pyramidal (M), excitatory (E), and inhibitory (I)
    populations based on delayed activity from connected regions.

    Mathematical Model
    ------------------

    The connectivity implements three distinct coupling pathways:

    1. **Lateral pathway (l)**: Symmetric coupling to pyramidal population
    2. **Feedforward pathway (f)**: Directed coupling to excitatory interneurons
    3. **Feedback pathway (b)**: Directed coupling to inhibitory interneurons

    For each pathway :math:`p \in \{l, f, b\}`, normalized weights are computed as:

    .. math::

        W_p = \text{normalize}(\exp(w_p) \odot SC)

    where :math:`w_p` are trainable log-weights, :math:`SC` is the structural
    connectivity matrix (fixed), and :math:`\odot` denotes element-wise multiplication.

    The Laplacian-based coupling consists of two terms:

    .. math::

        \text{LE}_p(i) = g_p \sum_j W_p^{(ij)} \cdot x^D_j

    .. math::

        \text{dg}_p(i) = -g_p \left(\sum_j W_p^{(ij)}\right) \cdot y_i

    where :math:`x^D` is the delayed inter-regional signal, :math:`y` is the local
    state, and :math:`g_p` are global pathway gains.

    The outputs are combined as:

    .. math::

        \begin{aligned}
        \text{inp}_M &= \text{LE}_l + \text{dg}_l \\
        \text{inp}_E &= \text{LE}_f + \text{dg}_f \\
        \text{inp}_I &= -(\text{LE}_b + \text{dg}_b)
        \end{aligned}

    where:

    - For lateral pathway: :math:`y_i = \text{EEG}_i = E_i - I_i` (difference of
      excitatory and inhibitory PSPs)
    - For feedforward pathway: :math:`y_i = M_i` (pyramidal membrane potential)
    - For feedback pathway: :math:`y_i = \text{EEG}_i = E_i - I_i`

    Parameters
    ----------
    delayed_x : Prefetch
        Delayed inter-regional signal accessor, typically returning the delayed
        EEG-like proxy (:math:`E - I`) with shape ``(n_regions,)`` or batched.
    M : Prefetch
        Accessor for pyramidal population membrane potential with shape ``(n_regions,)``.
    E : Prefetch
        Accessor for excitatory interneuron postsynaptic potential with shape ``(n_regions,)``.
    I : Prefetch
        Accessor for inhibitory interneuron postsynaptic potential with shape ``(n_regions,)``.
    sc : ArrayLike
        Structural connectivity matrix with shape ``(n_regions, n_regions)``. This
        is a fixed, non-trainable template that scales the learned weights.
    w_ll : Initializer
        Initializer for lateral pathway log-weights. Will be exponentiated and
        normalized with symmetric normalization during precompute.
    w_ff : Initializer
        Initializer for feedforward pathway log-weights. Will be exponentiated
        and normalized during precompute.
    w_bb : Initializer
        Initializer for feedback pathway log-weights. Will be exponentiated
        and normalized during precompute.
    g_l : Parameter, default 1.0
        Global gain for lateral pathway.
    g_f : Parameter, default 1.0
        Global gain for feedforward pathway.
    g_b : Parameter, default 1.0
        Global gain for feedback pathway.
    mask : ArrayLike, optional
        Optional binary mask with shape ``(n_regions, n_regions)`` applied to
        normalized weights. Default is ``None`` (no masking).

    Notes
    -----
    - The lateral pathway uses symmetric normalization: :math:`W_l = 0.5(W + W^T)`
    - The feedforward and feedback pathways use standard L2 normalization
    - All trainable weights (:math:`w_*`) are stored in log-space for numerical stability
    - The precompute mechanism ensures normalized weights are cached during forward pass
    - LE terms represent long-range excitation from delayed inter-regional signals
    - dg terms represent local inhibition from the diagonal Laplacian

    References
    ----------
    .. [1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual
           evoked potential generation in a mathematical model of coupled cortical
           columns. *Biological Cybernetics*, 73(4), 357-366.
    .. [2] David, O., & Friston, K. J. (2003). A neural mass model for MEG/EEG:
           coupling and neuronal dynamics. *NeuroImage*, 20(3), 1743-1755.
    .. [3] Momi, D., Wang, Z., & Griffiths, J. D. (2023). TMS-evoked responses are
           driven by recurrent large-scale network dynamics. *eLife*, 12, e83232.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        delayed_x: Prefetch,
        M: Prefetch,
        E: Prefetch,
        I: Prefetch,
        sc: Array,
        w_ll: Initializer,
        w_ff: Initializer,
        w_bb: Initializer,
        g_l: Parameter = 1.0,
        g_f: Parameter = 1.0,
        g_b: Parameter = 1.0,
        mask: Optional[Array] = None,
    ):
        super().__init__()

        self.delayed_x = delayed_x
        self.M = M
        self.E = E
        self.I = I

        # Structural connectivity matrix (non-trainable)
        self.sc = sc
        shape = sc.shape

        # Optional binary mask (non-trainable, applied after weight normalization)
        self.mask = braintools.init.param(mask, shape)
        self.w_ff = Param.init(w_ff, shape)
        self.w_ff.precompute = self._normalize
        self.w_bb = Param.init(w_bb, shape)
        self.w_bb.precompute = self._normalize
        self.w_ll = Param.init(w_ll, shape)
        self.w_ll.precompute = self._symmetric_normalize
        self.g_l = Param.init(g_l)
        self.g_f = Param.init(g_f)
        self.g_b = Param.init(g_b)

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        init_maybe_prefetch(self.delayed_x)
        init_maybe_prefetch(self.M)
        init_maybe_prefetch(self.E)
        init_maybe_prefetch(self.I)

    def _normalize(self, w: Array) -> Tuple[Array, Array]:
        """Normalize weights with standard L2 normalization."""
        w_b = u.math.exp(w) * self.sc
        w_n = w_b / u.math.linalg.norm(w_b)
        if self.mask is not None:
            w_n = w_n * self.mask
        diag = -u.math.sum(w_n, axis=1)
        return w_n, diag

    def _symmetric_normalize(self, w: Array) -> Tuple[Array, Array]:
        """Normalize weights with symmetric normalization (for lateral pathway)."""
        w = u.math.exp(w) * self.sc
        w = 0.5 * (w + u.math.transpose(w, (0, 1)))
        w_n = w / u.linalg.norm(w)
        if self.mask is not None:
            w_n = w_n * self.mask
        diag = -u.math.sum(w_n, axis=1)
        return w_n, diag

    def update_tr(self, *args, **kwargs):
        # Get pathway gains
        g_l = self.g_l.value()
        g_f = self.g_f.value()
        g_b = self.g_b.value()

        # Get delayed inter-regional signal and normalized weights
        delay_x = u.get_magnitude(self.delayed_x())
        w_n_b, dg_b = self.w_bb.value()  # feedback pathway weights (normalized via precompute)
        w_n_f, dg_f = self.w_ff.value()  # feedforward pathway weights (normalized via precompute)
        w_n_l, dg_l = self.w_ll.value()  # lateral pathway weights (symmetric normalized via precompute)

        # Long-range excitation (LE) terms from delayed inter-regional signal
        LEd_b = u.math.sum(w_n_b * delay_x, axis=1)
        LEd_f = u.math.sum(w_n_f * delay_x, axis=1)
        LEd_l = u.math.sum(w_n_l * delay_x, axis=1)

        return g_l * LEd_l, g_f * LEd_f, -g_b * LEd_b

    def update(self, *args, **kwargs) -> Tuple[Array, Array, Array]:
        """
        Compute Laplacian coupling inputs for pyramidal, excitatory, and inhibitory populations.
        """
        # Get pathway gains
        g_l = self.g_l.value()
        g_f = self.g_f.value()
        g_b = self.g_b.value()

        # Get delayed inter-regional signal and normalized weights
        w_n_b, dg_b = self.w_bb.value()  # feedback pathway weights (normalized via precompute)
        w_n_f, dg_f = self.w_ff.value()  # feedforward pathway weights (normalized via precompute)
        w_n_l, dg_l = self.w_ll.value()  # lateral pathway weights (symmetric normalized via precompute)

        # Get local population states
        M = u.get_magnitude(self.M())
        E = u.get_magnitude(self.E())
        I = u.get_magnitude(self.I())
        eeg = E - I  # EEG-like proxy (difference of excitatory and inhibitory PSPs)

        # Combine LE and dg terms for each population
        inp_M = g_l * dg_l * M  # pyramidal population (lateral pathway)
        inp_E = g_f * dg_f * eeg  # excitatory interneurons (feedforward pathway)
        inp_I = -g_b * dg_b * eeg  # inhibitory interneurons (feedback pathway)

        # Return inputs for pyramidal, excitatory, and inhibitory populations
        return inp_M, inp_E, inp_I



class LaplacianConnV2(Module):
    def __init__(
        self,
        dynamics: Dynamics,
        delays: Array,
        delay_init: Callable = braintools.init.ZeroInit(),
        weight: Initializer = braintools.init.KaimingNormal(),
    ):
        super().__init__()

        n_hidden = dynamics.varshape[0]
        self.dynamics = dynamics
        delays = braintools.init.param(delays, (n_hidden, n_hidden))
        self.delay_prefetch = self.dynamics.prefetch_delay('M', delays, delay_index(n_hidden), init=delay_init)
        self.w_b = brainstate.ParamState.init(weight, (n_hidden, n_hidden))
        self.w_f = brainstate.ParamState.init(weight, (n_hidden, n_hidden))
        self.w_l = brainstate.ParamState.init(weight, (n_hidden, n_hidden))

    def update(self, *args, **kwargs) -> Tuple[Array, Array, Array]:
        """
        Compute Laplacian coupling inputs for pyramidal, excitatory, and inhibitory populations.
        """
        E = self.dynamics.E.value / u.mV
        I = self.dynamics.I.value / u.mV
        eeg = u.get_magnitude(E - I)

        # Combine LE and dg terms for each population
        inp_M = additive_coupling(u.get_magnitude(self.delay_prefetch()), self.w_b.value)
        inp_E = self.w_f.value @ eeg
        inp_I = - self.w_l.value @ eeg

        # Return inputs for pyramidal, excitatory, and inhibitory populations
        return inp_M, inp_E, inp_I
