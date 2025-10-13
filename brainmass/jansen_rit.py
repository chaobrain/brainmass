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

from typing import Union, Callable

import brainstate
import braintools
import brainunit as u
from brainstate.nn import exp_euler_step

from .noise import Noise
from ._typing import Initializer

__all__ = [
    'JansenRitModel',
]


class Identity:
    def __call__(self, x):
        return x


class JansenRitModel(brainstate.nn.Dynamics):
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

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        Ae: Initializer = 3.25 * u.mV,  # Excitatory gain
        Ai: Initializer = 22. * u.mV,  # Inhibitory gain
        be: Initializer = 100. * u.Hz,  # Excit. time const
        bi: Initializer = 50. * u.Hz,  # Inhib. time const.
        C: Initializer = 135.,  # Connect. const.
        a1: Initializer = 1.,  # Connect. param.
        a2: Initializer = 0.8,  # Connect. param.
        a3: Initializer = 0.25,  # Connect. param
        a4: Initializer = 0.25,  # Connect. param.
        s_max: Initializer = 5.0 * u.Hz,  # Max firing rate
        v0: Initializer = 6. * u.mV,  # Firing threshold
        r: Initializer = 0.56,  # Sigmoid steepness
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

        self.Ae = braintools.init.param(Ae, self.varshape)
        self.Ai = braintools.init.param(Ai, self.varshape)
        self.be = braintools.init.param(be, self.varshape)
        self.bi = braintools.init.param(bi, self.varshape)
        self.a1 = braintools.init.param(a1, self.varshape)
        self.a2 = braintools.init.param(a2, self.varshape)
        self.a3 = braintools.init.param(a3, self.varshape)
        self.a4 = braintools.init.param(a4, self.varshape)
        self.v0 = braintools.init.param(v0, self.varshape)
        self.C = braintools.init.param(C, self.varshape)
        self.r = braintools.init.param(r, self.varshape)
        self.s_max = braintools.init.param(s_max, self.varshape)

        assert callable(fr_scale), 'fr_scale must be a callable function'
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
        self.fr_scale = fr_scale
        self.noise_E = noise_E
        self.noise_I = noise_I
        self.noise_M = noise_M
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        self.M = brainstate.HiddenState(braintools.init.param(self.M_init, self.varshape, batch_size))
        self.E = brainstate.HiddenState(braintools.init.param(self.E_init, self.varshape, batch_size))
        self.I = brainstate.HiddenState(braintools.init.param(self.I_init, self.varshape, batch_size))
        self.Mv = brainstate.HiddenState(braintools.init.param(self.Mv_init, self.varshape, batch_size))
        self.Ev = brainstate.HiddenState(braintools.init.param(self.Ev_init, self.varshape, batch_size))
        self.Iv = brainstate.HiddenState(braintools.init.param(self.Iv_init, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.M.value = braintools.init.param(self.M_init, self.varshape, batch_size)
        self.E.value = braintools.init.param(self.E_init, self.varshape, batch_size)
        self.I.value = braintools.init.param(self.I_init, self.varshape, batch_size)
        self.Mv.value = braintools.init.param(self.Mv_init, self.varshape, batch_size)
        self.Ev.value = braintools.init.param(self.Ev_init, self.varshape, batch_size)
        self.Iv.value = braintools.init.param(self.Iv_init, self.varshape, batch_size)

    def S(self, v):
        # Sigmoid ranges from 0 to s_max, centered at v0
        return self.s_max / (1 + u.math.exp(self.r * (self.v0 - v) / u.mV))

    def dMv(self, Mv, M, E, I, inp):
        # Pyramidal population driven by the difference of PSPs (no extra C here)
        fr = self.S(E - I + inp)
        return self.Ae * self.be * self.fr_scale(fr) - 2 * self.be * Mv - self.be ** 2 * M

    def dEv(self, Ev, M, E, inp=0. * u.Hz):
        # Excitatory interneuron population: A*a*(p + C2*S(C1*M)) - 2*a*y' - a^2*y
        s_M = self.C * self.a2 * self.S(self.C * self.a1 * M)
        fr_total = self.fr_scale(inp + s_M)
        return self.Ae * self.be * fr_total - 2 * self.be * Ev - self.be ** 2 * E

    def dIv(self, Iv, M, I, inp):
        # Inhibitory interneuron population: B*b*(C4*S(C3*M)) - 2*b*y' - b^2*y
        s_M = self.C * self.a4 * self.S(self.C * self.a3 * M + inp)
        fr_total = self.fr_scale(s_M)
        return self.Ai * self.bi * fr_total - 2 * self.bi * Iv - self.bi ** 2 * I

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
