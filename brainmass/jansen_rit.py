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

import brainmass
from .noise import Noise, GaussianNoise
from .typing import Parameter, Initializer

__all__ = [
    'JansenRitStep',
    "JansenRitTR",
]

Array = brainstate.typing.ArrayLike
Size = brainstate.typing.Size
Prefetch = Union[
    brainstate.nn.PrefetchDelayAt,
    brainstate.nn.PrefetchDelay,
    brainstate.nn.Prefetch,
    Callable,
]


class Identity:
    def __call__(self, x):
        return x


class JansenRitStep(brainstate.nn.Dynamics):
    r"""
    Jansen-Rit neural mass model.

    This implementation follows the standard three-population Jansen–Rit formulation
    with state variables for the pyramidal (M), excitatory interneuron (E), and
    inhibitory interneuron (I) membrane potentials and their first derivatives
    (Mv, Ev, Iv):

    $$
    \begin{aligned}
    &\dot{M}= M_v, \\
    &\dot{E}= E_v, \\
    &\dot{I}= I_v, \\
    &\dot{M}_v= A_e b_e\,\text{scale}\big(S(E - I + M_{\text{inp}})\big) - 2 b_e M_v - b_e^2 M, \\
    &\dot{E}_v= A_e b_e\,\text{scale}\big(E_{\text{inp}} + C a_2 S(C a_1 M)\big) - 2 b_e E_v - b_e^2 E, \\
    &\dot{I}_v= A_i b_i\,\text{scale}\big(C a_4 S(C a_3 M + I_{\text{inp}})\big) - 2 b_i I_v - b_i^2 I.
    \end{aligned}
    $$

    The static nonlinearity maps membrane potential to firing rate:

    $$
    S(v) = \frac{s_{\max}}{1 + e^{\, r (v_0 - v)/\mathrm{mV}}},
    $$

    yielding values in $[0, s_{\max}]$. Here, $v$ is in mV, $s_{\max}$ in s$^{-1}$,
    $v_0$ in mV, and $r$ is dimensionless.

    Inputs and units:

    - `M_inp` (mV) shifts the pyramidal population input inside the sigmoid in $\dot{M}_v$.
    - `E_inp` (s$^{-1}$) is added to the excitatory firing-rate drive in $\dot{E}_v$.
    - `I_inp` (mV) shifts the inhibitory population input inside the sigmoid in $\dot{I}_v$.

    The EEG-like output proxy returned by `eeg()` is the difference between excitatory
    and inhibitory postsynaptic potentials at the pyramidal population, i.e. `E - I`.

    Standard parameter settings for the Jansen–Rit model. Only parameters with a
    specified "Range" are estimated in this study.

    .. list-table::
       :widths: 12 30 14 18
       :header-rows: 1

       * - Parameter
         - Description
         - Default
         - Range
       * - Ae
         - Excitatory gain
         - 3.25 mV
         - 2.6-9.75 mV
       * - Ai
         - Inhibitory gain
         - 22 mV
         - 17.6-110.0 mV
       * - be
         - Excitatory time const.
         - 100 s^-1
         - 5-150 s^-1
       * - bi
         - Inhibitory time const.
         - 50 s^-1
         - 25-75 s^-1
       * - C
         - Connectivity constant
         - 135
         - 65-1350
       * - a1
         - Connectivity parameter
         - 1.0
         - 0.5-1.5
       * - a2
         - Connectivity parameter
         - 0.8
         - 0.4-1.2
       * - a3
         - Connectivity parameter
         - 0.25
         - 0.125-0.375
       * - a4
         - Connectivity parameter
         - 0.25
         - 0.125-0.375
       * - smax
         - Max firing rate
         - 2.5 s^-1
         - -
       * - v0
         - Firing threshold
         - 6 mV
         - -
       * - r
         - Sigmoid steepness
         - 0.56
         - -

    Parameters
    ----------
    in_size : `brainstate.typing.Size`
        Variable shape for parameter/state broadcasting.
    Ae : `ArrayLike` or `Callable`, default `3.25 * u.mV`
        Excitatory gain (mV).
    Ai : `ArrayLike` or `Callable`, default `22. * u.mV`
        Inhibitory gain (mV).
    be : `ArrayLike` or `Callable`, default `100. * u.Hz`
        Excitatory inverse time constant (s^-1).
    bi : `ArrayLike` or `Callable`, default `50. * u.Hz`
        Inhibitory inverse time constant (s^-1).
    C : `ArrayLike` or `Callable`, default `135.`
        Global connectivity scaling (dimensionless).
    a1, a2, a3, a4 : `ArrayLike` or `Callable`, defaults `1., 0.8, 0.25, 0.25`
        Connectivity parameters (dimensionless) used as in the equations above.
    s_max : `ArrayLike` or `Callable`, default `2.5 * u.Hz`
        Maximum firing rate for the sigmoid, units s^-1.
    v0 : `ArrayLike` or `Callable`, default `6. * u.mV`
        Sigmoid midpoint (mV).
    r : `ArrayLike` or `Callable`, default `0.56`
        Sigmoid steepness (dimensionless).
    M_init, E_init, I_init : `Callable`, defaults `ZeroInit(unit=u.mV)`
        Initializers for membrane potentials (mV).
    Mv_init, Ev_init, Iv_init : `Callable`, defaults `ZeroInit(unit=u.mV/u.second)`
        Initializers for potential derivatives (mV/s).
    fr_scale : `Callable`, default `Identity()`
        Optional scaling applied to firing-rate drives; receives rates in s^-1
        and returns scaled rates.
    noise_E, noise_I, noise_M : `Noise` or `None`, default `None`
        Optional additive noise sources applied to `E_inp`, `I_inp`, and `M_inp`
        respectively.
    method : `str`, default `'exp_euler'`
        Integrator name. `'exp_euler'` uses `brainstate.nn.exp_euler_step`; any
        other value dispatches to `braintools.quad.ode_{method}_step`.

    Notes
    -----
    - In this implementation `fr_scale` is applied to the firing-rate drive terms
      and defaults to the identity.
    - Variable naming: $(M, E, I)$ correspond to pyramidal, excitatory, and inhibitory
      population membrane potentials (mV); $(M_v, E_v, I_v)$ are their time derivatives (mV/s).

    References
    ----------
    - [1] Nunez P L, Srinivasan R. Electric fields of the brain: the neurophysics of EEG. Oxford University Press, 2006.
    - [2] Jansen B H, Rit V G. Electroencephalogram and visual evoked potential generation in a mathematical model of coupled cortical columns. Biological Cybernetics, 1995, 73(4): 357–366.
    - [3] David O, Friston K J. A neural mass model for MEG/EEG: coupling and neuronal dynamics. NeuroImage, 2003, 20(3): 1743–1755.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        Ae: Parameter = 3.25 * u.mV,  # Excitatory gain
        Ai: Parameter = 22. * u.mV,  # Inhibitory gain
        be: Parameter = 100. * u.Hz,  # Excit. time const
        bi: Parameter = 50. * u.Hz,  # Inhib. time const.
        C: Parameter = 135.,  # Connect. const.
        a1: Parameter = 1.,  # Connect. param.
        a2: Parameter = 0.8,  # Connect. param.
        a3: Parameter = 0.25,  # Connect. param
        a4: Parameter = 0.25,  # Connect. param.
        s_max: Parameter = 5.0 * u.Hz,  # Max firing rate
        v0: Parameter = 6. * u.mV,  # Firing threshold
        r: Parameter = 0.56,  # Sigmoid steepness
        M_init: Callable = braintools.init.ZeroInit(unit=u.mV),
        E_init: Callable = braintools.init.ZeroInit(unit=u.mV),
        I_init: Callable = braintools.init.ZeroInit(unit=u.mV),
        Mv_init: Callable = braintools.init.ZeroInit(unit=u.mV / u.second),
        Ev_init: Callable = braintools.init.ZeroInit(unit=u.mV / u.second),
        Iv_init: Callable = braintools.init.ZeroInit(unit=u.mV / u.second),
        fr_scale: Callable = Identity(),
        noise_E: Noise = None,
        noise_I: Noise = None,
        noise_M: Noise = None,
        method: str = 'exp_euler'
    ):
        super().__init__(in_size)

        # parameters
        self.Ae = Param.init(Ae, self.varshape)
        self.Ai = Param.init(Ai, self.varshape)
        self.be = Param.init(be, self.varshape)
        self.bi = Param.init(bi, self.varshape)
        self.a1 = Param.init(a1, self.varshape)
        self.a2 = Param.init(a2, self.varshape)
        self.a3 = Param.init(a3, self.varshape)
        self.a4 = Param.init(a4, self.varshape)
        self.v0 = Param.init(v0, self.varshape)
        self.C = Param.init(C, self.varshape)
        self.r = Param.init(r, self.varshape)
        self.s_max = Param.init(s_max, self.varshape)

        # initialization
        assert callable(M_init), 'M_init must be a callable function'
        assert callable(E_init), 'E_init must be a callable function'
        assert callable(I_init), 'I_init must be a callable function'
        assert callable(Mv_init), 'Mv_init must be a callable function'
        assert callable(Ev_init), 'Ev_init must be a callable function'
        assert callable(Iv_init), 'Iv_init must be a callable function'
        self.M_init = M_init
        self.E_init = E_init
        self.I_init = I_init
        self.Mv_init = Mv_init
        self.Ev_init = Ev_init
        self.Iv_init = Iv_init

        # noise
        self.noise_E = noise_E
        self.noise_I = noise_I
        self.noise_M = noise_M

        assert callable(fr_scale), 'fr_scale must be a callable function'
        self.fr_scale = fr_scale
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.M = HiddenState.init(self.M_init, self.varshape, batch_size)
        self.E = HiddenState.init(self.E_init, self.varshape, batch_size)
        self.I = HiddenState.init(self.I_init, self.varshape, batch_size)
        self.Mv = HiddenState.init(self.Mv_init, self.varshape, batch_size)
        self.Ev = HiddenState.init(self.Ev_init, self.varshape, batch_size)
        self.Iv = HiddenState.init(self.Iv_init, self.varshape, batch_size)

    def S(self, v):
        # Sigmoid ranges from 0 to s_max, centered at v0
        s_max = self.s_max.value()
        v0 = self.v0.value()
        r = self.r.value()
        return s_max * jax.nn.sigmoid(-r * (v0 - v) / u.mV)

    def dMv(self, Mv, M, E, I, inp):
        # Pyramidal population driven by the difference of PSPs (no extra C here)
        be = self.be.value()
        Ae = self.Ae.value()
        fr = self.S(E - I + inp)
        return Ae * be * self.fr_scale(fr) - 2 * be * Mv - be ** 2 * M

    def dEv(self, Ev, M, E, inp=0. * u.Hz):
        # Excitatory interneuron population: A*a*(p + C2*S(C1*M)) - 2*a*y' - a^2*y
        C = self.C.value()
        a2 = self.a2.value()
        a1 = self.a1.value()
        Ae = self.Ae.value()
        be = self.be.value()
        s_M = C * a2 * self.S(C * a1 * M)
        fr_total = self.fr_scale(inp + s_M)
        return Ae * be * fr_total - 2 * be * Ev - be ** 2 * E

    def dIv(self, Iv, M, I, inp):
        # Inhibitory interneuron population: B*b*(C4*S(C3*M)) - 2*b*y' - b^2*y
        C = self.C.value()
        a3 = self.a3.value()
        a4 = self.a4.value()
        Ai = self.Ai.value()
        bi = self.bi.value()
        s_M = C * a4 * self.S(C * a3 * M + inp)
        fr_total = self.fr_scale(s_M)
        return Ai * bi * fr_total - 2 * bi * Iv - bi ** 2 * I

    def derivative(self, state, t, M_inp, E_inp, I_inp):
        M, E, I, Mv, Ev, Iv = state
        dM = Mv
        dE = Ev
        dI = Iv
        dMv = self.dMv(Mv, M, E, I, M_inp)
        dEv = self.dEv(Ev, M, E, E_inp)
        dIv = self.dIv(Iv, M, I, I_inp)
        return (dM, dE, dI, dMv, dEv, dIv)

    def update(
        self,
        M_inp=0. * u.mV,
        E_inp=0. * u.Hz,
        I_inp=0. * u.mV,
    ):
        M_inp = M_inp if self.noise_M is None else M_inp + self.noise_M()
        E_inp = E_inp if self.noise_E is None else E_inp + self.noise_E()
        I_inp = I_inp if self.noise_I is None else I_inp + self.noise_I()
        if self.method == 'exp_euler':
            dt = brainstate.environ.get_dt()
            M = self.M.value + self.Mv.value * dt
            E = self.E.value + self.Ev.value * dt
            I = self.I.value + self.Iv.value * dt
            Mv = exp_euler_step(self.dMv, self.Mv.value, self.M.value, self.E.value, self.I.value, M_inp)
            Ev = exp_euler_step(self.dEv, self.Ev.value, self.M.value, self.E.value, E_inp)
            Iv = exp_euler_step(self.dIv, self.Iv.value, self.M.value, self.I.value, I_inp)
        else:
            method = getattr(braintools.quad, f'ode_{self.method}_step')
            state = (self.M.value, self.E.value, self.I.value, self.Mv.value, self.Ev.value, self.Iv.value)
            M, E, I, Mv, Ev, Iv = method(self.derivative, state, 0. * u.ms, M_inp, E_inp, I_inp)
        self.M.value = M
        self.E.value = E
        self.I.value = I
        self.Mv.value = Mv
        self.Ev.value = Ev
        self.Iv.value = Iv
        return self.eeg()

    def eeg(self):
        # EEG-like proxy: difference between excitatory and inhibitory PSPs at pyramidal
        return self.E.value - self.I.value


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
        g_l: Parameter,
        g_f: Parameter,
        g_b: Parameter,
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

    def update_step(self, *args, **kwargs) -> Tuple[Array, Array, Array]:
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


class JansenRitTR(Dynamics):
    def __init__(
        self,
        in_size: Size,

        # distance parameters
        delay: Array,

        # structural connectivity parameters
        sc: Array,
        k: Parameter,
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
        tr: u.Quantity = 1e-3 * u.second
    ):
        super().__init__(in_size)

        self.k = Param.init(k)
        self.tr = tr

        # single step dynamics
        self.step = JansenRitStep(
            in_size=in_size,
            Mv_init=lambda *args: state_init(*args) * u.mV / u.second,
            Ev_init=lambda *args: state_init(*args) * u.mV / u.second,
            Iv_init=lambda *args: state_init(*args) * u.mV / u.second,
            M_init=lambda *args: state_init(*args) * u.mV,
            E_init=lambda *args: state_init(*args) * u.mV,
            I_init=lambda *args: state_init(*args) * u.mV,
            noise_M=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
            noise_E=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
            noise_I=GaussianNoise(in_size, sigma=std_in) if std_in is not None else None,
        )

        # delay
        n_hidden = self.varshape[0]
        dt = brainstate.environ.get_dt()
        self.delay = Delay(self.step.M_init(self.varshape), init=delay_init)
        self.delay_access = self.delay.access('delay', delay * dt, brainmass.delay_index(n_hidden))

        # connectivity
        self.conn = LaplacianConnectivity(
            self.delay_access,
            self.step.prefetch('M'),
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

    def update(
        self,
        input: Array,
        record_state: bool = False,
        iter_input: bool = False
    ):
        def step(inp):
            ext = inp if iter_input else input
            inp_M_step, inp_E_step, inp_I_step = self.conn.update_step()
            inp_M = (inp_M_tr + inp_M_step + ext) * u.mV
            inp_E = (inp_E_tr + inp_E_step) * u.Hz
            inp_I = (inp_I_tr + inp_I_step) * u.mV
            self.step.update(inp_M, inp_E, inp_I)

        n_step = int(self.tr / brainstate.environ.get_dt())
        if iter_input:
            assert input.shape[0] == n_step, f'Input length {input.shape[0]} does not match number of steps {n_step}'

        input = self.k.value() * input
        inp_M_tr, inp_E_tr, inp_I_tr = self.conn.update_tr()
        brainstate.transform.for_loop(step, input if iter_input else np.arange(n_step))
        self.delay.update(self.step.M.value)
        activity = self.step.E.value - self.step.I.value
        activity = u.get_magnitude(activity)

        if record_state:
            state = dict(M=self.step.M.value, E=self.step.E.value, I=self.step.I.value)
            return activity, state
        return activity
