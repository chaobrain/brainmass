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

from typing import Tuple, Callable, Optional, Union

import brainstate
import brainstate.environ
import braintools
import braintools.init
import brainunit as u
import jax
import numpy as np
from brainstate import HiddenState
from brainstate.nn import Dynamics, Delay, Param, Module, init_maybe_prefetch

from brainmass import (
    LeadfieldReadout,
    Noise, GaussianNoise,
    Parameter, Initializer,
    sys2nd, sigmoid, bounded_input
)

__all__ = [
    "JansenRit2Step",
    "JansenRit2TR",
    "JansenRit2Window",
]

Size = brainstate.typing.Size
Array = brainstate.typing.ArrayLike
Prefetch = Union[
    brainstate.nn.PrefetchDelayAt,
    brainstate.nn.PrefetchDelay,
    brainstate.nn.Prefetch,
    Callable,
]


class JansenRit2Step(Dynamics):
    def __init__(
        self,
        in_size: Size,
        # dynamics parameters
        A: Parameter = 3.25,
        B: Parameter = 22.,
        a: Parameter = 100.,
        b: Parameter = 50.,
        vmax: Parameter = 5.0,
        v0: Parameter = 6.0,
        r: Parameter = 0.56,
        c1: Parameter = 135.,
        c2: Parameter = 135 * 0.8,
        c3: Parameter = 135 * 0.25,
        c4: Parameter = 135 * 0.25,
        kP: Parameter = 0.,
        kE: Parameter = 0.,
        kI: Parameter = 0.,
        noise_P: Noise = None,
        noise_E: Noise = None,
        noise_I: Noise = None,
        # other parameters
        state_saturation: bool = True,
        input_saturation: bool = True,
        state_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__(in_size)

        self.node_size = in_size
        self.state_saturation = state_saturation
        self.input_saturation = input_saturation
        self.state_init = state_init
        self.u_2ndsys_ub = 500.

        self.A = Param.init(A, self.varshape)
        self.a = Param.init(a, self.varshape)
        self.B = Param.init(B, self.varshape)
        self.b = Param.init(b, self.varshape)
        self.r = Param.init(r, self.varshape)
        self.v0 = Param.init(v0, self.varshape)
        self.c1 = Param.init(c1, self.varshape)
        self.c2 = Param.init(c2, self.varshape)
        self.c3 = Param.init(c3, self.varshape)
        self.c4 = Param.init(c4, self.varshape)
        self.kP = Param.init(kP, self.varshape)
        self.kE = Param.init(kE, self.varshape)
        self.kI = Param.init(kI, self.varshape)
        self.vmax = Param.init(vmax, self.varshape)

        self.noise_E = noise_E
        self.noise_I = noise_I
        self.noise_P = noise_P

    def init_state(self, *args, **kwargs):
        self.P = HiddenState.init(self.state_init, self.varshape)
        self.E = HiddenState.init(self.state_init, self.varshape)
        self.I = HiddenState.init(self.state_init, self.varshape)
        self.Pv = HiddenState.init(self.state_init, self.varshape)
        self.Ev = HiddenState.init(self.state_init, self.varshape)
        self.Iv = HiddenState.init(self.state_init, self.varshape)

    def update(self, inp_P=None, inp_E=None, inp_I=None):
        P = self.P.value
        E = self.E.value
        I = self.I.value
        Pv = self.Pv.value
        Ev = self.Ev.value
        Iv = self.Iv.value

        A = self.A.value()
        B = self.B.value()
        a = self.a.value()
        b = self.b.value()
        vmax = self.vmax.value()
        v0 = self.v0.value()
        r = self.r.value()
        c1 = self.c1.value()
        c2 = self.c2.value()
        c3 = self.c3.value()
        c4 = self.c4.value()
        kP = self.kP.value()
        kE = self.kE.value()
        kI = self.kI.value()

        # 计算各群体的发放率
        rP = kP + sigmoid(E - I, vmax, v0, r)
        if inp_P is not None:
            rP = rP + inp_P
        if self.noise_P is not None:
            rP = rP + self.noise_P.update()

        rE = kE + c2 * sigmoid(c1 * P, vmax, v0, r)
        if inp_E is not None:
            rE = rE + inp_E
        if self.noise_E is not None:
            rE = rE + self.noise_E.update()

        rI = kI + c4 * sigmoid(c3 * P, vmax, v0, r)
        if inp_I is not None:
            rI = rI + inp_I
        if self.noise_I is not None:
            rI = rI + self.noise_I.update()

        # Update the states by step-size.
        dt = brainstate.environ.get_dt()
        ddP = P + dt * Pv / u.second
        ddE = E + dt * Ev / u.second
        ddI = I + dt * Iv / u.second
        ddPv = Pv + dt * sys2nd(A, a, bounded_input(rP, self.u_2ndsys_ub), P, Pv) / u.second
        ddEv = Ev + dt * sys2nd(A, a, bounded_input(rE, self.u_2ndsys_ub), E, Ev) / u.second
        ddIv = Iv + dt * sys2nd(B, b, bounded_input(rI, self.u_2ndsys_ub), I, Iv) / u.second

        # Calculate the saturation for model states (for stability and gradient calculation).
        self.E.value = bounded_input(ddE, 1e3)
        self.I.value = bounded_input(ddI, 1e3)
        self.P.value = bounded_input(ddP, 1e3)
        self.Ev.value = bounded_input(ddEv, 1e3)
        self.Iv.value = bounded_input(ddIv, 1e3)
        self.Pv.value = bounded_input(ddPv, 1e3)


class LaplacianConnectivity(Module):
    r"""
    Laplacian connectivity for multi-region Jansen-Rit neural mass models.

    This class implements a three-pathway graph Laplacian coupling mechanism
    designed for spatially-extended Jansen-Rit neural mass networks. It computes
    coupling inputs for the pyramidal (P), excitatory (E), and inhibitory (I)
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
        \text{inp}_P &= \text{LE}_l + \text{dg}_l \\
        \text{inp}_E &= \text{LE}_f + \text{dg}_f \\
        \text{inp}_I &= -(\text{LE}_b + \text{dg}_b)
        \end{aligned}

    where:

    - For lateral pathway: :math:`y_i = \text{EEG}_i = E_i - I_i` (difference of
      excitatory and inhibitory PSPs)
    - For feedforward pathway: :math:`y_i = P_i` (pyramidal membrane potential)
    - For feedback pathway: :math:`y_i = \text{EEG}_i = E_i - I_i`

    Parameters
    ----------
    delayed_x : Prefetch
        Delayed inter-regional signal accessor, typically returning the delayed
        EEG-like proxy (:math:`E - I`) with shape ``(n_regions,)`` or batched.
    P : Prefetch
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
        P: Prefetch,
        E: Prefetch,
        I: Prefetch,
        sc: Array,
        w_ll: Initializer,
        w_ff: Initializer,
        w_bb: Initializer,
        g_l: Parameter,
        g_f: Parameter,
        g_b: Parameter,
        mask: Optional[Array] = None,
    ):
        super().__init__()

        self.delayed_x = delayed_x
        self.P = P
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
        init_maybe_prefetch(self.P)
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
        delay_x = self.delayed_x()
        w_n_b, dg_b = self.w_bb.value()  # feedback pathway weights (normalized via precompute)
        w_n_f, dg_f = self.w_ff.value()  # feedforward pathway weights (normalized via precompute)
        w_n_l, dg_l = self.w_ll.value()  # lateral pathway weights (symmetric normalized via precompute)

        # Long-range excitation (LE) terms from delayed inter-regional signal
        LEd_b = u.math.sum(w_n_b * delay_x, axis=1)
        LEd_f = u.math.sum(w_n_f * delay_x, axis=1)
        LEd_l = u.math.sum(w_n_l * delay_x, axis=1)

        return g_l * LEd_l, g_f * LEd_f, -g_b * LEd_b

    def update_step(self, *args, **kwargs) -> Tuple[Array, Array, Array]:
        """Compute Laplacian coupling inputs for pyramidal, excitatory, and inhibitory populations.

        Returns
        -------
        inp_P : ArrayLike
            Input to pyramidal population from lateral pathway.
        inp_E : ArrayLike
            Input to excitatory interneurons from feedforward pathway.
        inp_I : ArrayLike
            Input to inhibitory interneurons from feedback pathway (negated).
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
        P = self.P()
        E = self.E()
        I = self.I()
        eeg = E - I  # EEG-like proxy (difference of excitatory and inhibitory PSPs)

        # Combine LE and dg terms for each population
        inp_P = g_l * dg_l * P  # pyramidal population (lateral pathway)
        inp_E = g_f * dg_f * eeg  # excitatory interneurons (feedforward pathway)
        inp_I = -g_b * dg_b * eeg  # inhibitory interneurons (feedback pathway)

        # Return inputs for pyramidal, excitatory, and inhibitory populations
        return inp_P, inp_E, inp_I


class JansenRit2TR(Dynamics):
    def __init__(
        self,
        in_size: Size,

        # neuronal dynamics parameters
        A: Parameter,
        a: Parameter,
        B: Parameter,
        b: Parameter,
        vmax: Parameter,
        v0: Parameter,
        r: Parameter,
        k: Parameter,
        c1: Parameter,
        c2: Parameter,
        c3: Parameter,
        c4: Parameter,
        kE: Parameter,
        kI: Parameter,

        # distance parameters
        delay: Array,

        # structural connectivity parameters
        sc: Array,
        w_ll: Parameter,
        w_ff: Parameter,
        w_bb: Parameter,
        g_l: Parameter,
        g_f: Parameter,
        g_b: Parameter,

        # other parameters
        std_in: Parameter = None,
        mask: Optional[Array] = None,
        state_saturation: bool = True,
        input_saturation: bool = True,
        state_init: Callable = braintools.init.ZeroInit(),
        delay_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__(in_size)

        self.k = Param.init(k)

        # single step dynamics
        self.step = JansenRit2Step(
            in_size=in_size,
            A=A,
            a=a,
            B=B,
            b=b,
            vmax=vmax,
            v0=v0,
            r=r,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            kE=kE,
            kI=kI,
            state_saturation=state_saturation,
            input_saturation=input_saturation,
            state_init=state_init,
            noise_P=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
            noise_E=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
            noise_I=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
        )

        # delay
        n_hidden = self.varshape[0]
        dt = brainstate.environ.get_dt()
        self.delay = Delay(
            jax.ShapeDtypeStruct(self.step.varshape, dtype=brainstate.environ.dftype()),
            init=delay_init
        )
        neuron_idx = np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
        self.delay_access = self.delay.access('delay', delay * dt, neuron_idx)

        # connectivity
        self.conn = LaplacianConnectivity(
            self.delay_access,
            self.step.prefetch('P'),
            self.step.prefetch('E'),
            self.step.prefetch('I'),
            sc=sc,
            w_ll=w_ll,
            w_ff=w_ff,
            w_bb=w_bb,
            g_l=g_l,
            g_f=g_f,
            g_b=g_b,
            mask=mask,
        )

    def update(self, inputs: Array, record_state: bool = False):
        k = self.k.value()
        inp_P_tr, inp_E_tr, inp_I_tr = self.conn.update_tr()

        def step(inp):
            ext = k * inp
            inp_P_step, inp_E_step, inp_I_step = self.conn.update_step()
            self.step.update(
                inp_P_tr + inp_P_step + ext,
                inp_E_tr + inp_E_step,
                inp_I_tr + inp_I_step
            )

        assert inputs.ndim == 2, f'Expected inputs to be 2D array, but got {inputs.ndim}D array.'
        brainstate.transform.for_loop(step, inputs)
        self.delay.update(self.step.P.value)
        activity = self.step.E.value - self.step.I.value

        if record_state:
            state = dict(P=self.step.P.value, E=self.step.E.value, I=self.step.I.value)
            return activity, state

        return activity


class JansenRit2Window(Module):
    """
    Jansen-Rit neural mass model for EEG simulation.

    A module for forward model (JansenRit) to simulate a batch of EEG signals.

    """

    def __init__(
        self,
        node_size: int,
        sc: np.ndarray,
        dist: np.ndarray,
        mu: np.ndarray,
        # Model parameters using Param API
        A: Parameter,
        a: Parameter,
        B: Parameter,
        b: Parameter,
        g_l: Parameter,
        g_f: Parameter,
        g_b: Parameter,
        c1: Parameter,
        c2: Parameter,
        c3: Parameter,
        c4: Parameter,
        k: Parameter,
        std_in: Parameter,
        vmax: Parameter,
        v0: Parameter,
        r: Parameter,
        y0: Parameter,
        kE: Parameter,
        kI: Parameter,
        cy0: Parameter,
        lm: Parameter,
        w_bb: Parameter,
        w_ff: Parameter,
        w_ll: Parameter,
        state_init: Callable,
        delay_init: Callable,
        mask=None,
    ):
        super().__init__()

        self.dynamics = JansenRit2TR(
            in_size=node_size,

            # dynamics parameters
            A=A,
            a=a,
            B=B,
            b=b,
            vmax=vmax,
            v0=v0,
            r=r,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            std_in=std_in,
            kE=kE,
            kI=kI,
            k=k,

            # distance parameters
            delay=dist / mu,

            # structural parameters
            sc=sc,
            w_ll=w_ll,
            w_ff=w_ff,
            w_bb=w_bb,
            g_l=g_l,
            g_f=g_f,
            g_b=g_b,

            # other parameters
            mask=mask,
            state_init=state_init,
            delay_init=delay_init,
        )
        self.leadfield = LeadfieldReadout(lm=lm, y0=y0, cy0=cy0)

    def set_mask(self, mask):
        self.dynamics.conn.mask = mask

    def update(self, inputs, record_state: bool = False):
        if record_state:
            fn = lambda inp: self.dynamics(inp, record_state=True)
            activities, states = brainstate.transform.for_loop(fn, inputs)
            return self.leadfield(activities), states
        else:
            activities = brainstate.transform.for_loop(self.dynamics, inputs)
            return self.leadfield(activities)
