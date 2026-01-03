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


from typing import Union, Tuple, Callable, Optional

import braintools
import brainunit as u

import brainstate
from brainstate.nn import Param, Module, init_maybe_prefetch
from ._common import set_module_as
from ._typing import Parameter, Initializer

# Typing alias for static type hints
Prefetch = Union[
    brainstate.nn.PrefetchDelayAt,
    brainstate.nn.PrefetchDelay,
    brainstate.nn.Prefetch,
    Callable,
]
# Runtime check tuple for isinstance
_PREFETCH_TYPES: Tuple[type, ...] = (
    brainstate.nn.PrefetchDelayAt,
    brainstate.nn.PrefetchDelay,
    brainstate.nn.Prefetch,
)
Array = brainstate.typing.ArrayLike

__all__ = [
    'DiffusiveCoupling',
    'AdditiveCoupling',
    'LaplacianConnectivity',
    'diffusive_coupling',
    'additive_coupling',
]


def _check_type(x):
    if not (isinstance(x, _PREFETCH_TYPES) or callable(x)):
        raise TypeError(f'The argument must be a Prefetch or Callable, got {x}')
    return x


@set_module_as('brainmass')
def diffusive_coupling(
    delayed_x: Callable | Array,
    y: Callable | Array,
    conn: Array,
    k: Array,
):
    r"""
    Diffusive coupling kernel (function form).

    Computes, for each target unit i over the last axis, the diffusive term

        current_i = k * sum_j conn[i, j] * (x_{i, j} - y_i)

    with full support for leading batch/time dimensions and unit-safe algebra.

    Parameters
    ----------
    delayed_x : Callable, ArrayLike
        Zero-arg callable returning the source signal with shape ``(..., N_out, N_in)``
        or flattened ``(..., N_out*N_in)``. Typically a ``Prefetch`` that reads
        a state from another module.
    y : Callable, ArrayLike
        Zero-arg callable returning the target signal with shape ``(..., N_out)``.
    conn : ArrayLike
        Connection weights. Either ``(N_out, N_in)`` or flattened ``(N_out*N_in,)``.
    k : ArrayLike
        Global coupling strength. Can be scalar or broadcastable to the output shape ``(..., N_out)``.

    Returns
    -------
    ArrayLike
        Coupling output with shape ``(..., N_out)``. If inputs carry units, the
        result preserves unit consistency via `brainunit`.

    Raises
    ------
    ValueError
        If shapes are incompatible with the expected conventions.
    """
    # y: (..., N_out)
    y_val = y() if callable(y) else y
    if y_val.ndim < 1:
        raise ValueError(f'y must have at least 1 dimension; got shape {y_val.shape}')
    n_out = y_val.shape[-1]
    y_exp = u.math.expand_dims(y_val, axis=-1)  # (..., N_out, 1)

    # x expected shape on trailing dims: (N_out, N_in) or flattened N_out*N_in
    x_val = delayed_x() if callable(delayed_x) else delayed_x
    if x_val.ndim < 1:
        raise ValueError(f'x must have at least 1 dimension; got shape {x_val.shape}')

    # Build (N_out, N_in) connection matrix
    if conn.ndim == 1:
        if conn.size % n_out != 0:
            raise ValueError(
                f'Flattened connection length {conn.size} is not divisible by N_out={n_out}.'
            )
        n_in = conn.size // n_out
        conn2d = u.math.reshape(conn, (n_out, n_in))
    else:
        conn2d = conn
        if conn2d.shape[0] != n_out:
            raise ValueError(
                f'Connection rows ({conn2d.shape[0]}) must match y size ({n_out}).'
            )
        n_in = conn2d.shape[1]

    # Reshape x to (..., N_out, N_in)
    if x_val.ndim >= 2 and x_val.shape[-2:] == (n_out, n_in):
        x_mat = x_val
    elif x_val.shape[-1] == n_out * n_in:
        x_mat = u.math.reshape(x_val, (*x_val.shape[:-1], n_out, n_in))
    else:
        raise ValueError(
            f'x has incompatible shape {x_val.shape}; expected (..., {n_out}, {n_in}) '
            f'or flattened (..., {n_out * n_in}).'
        )

    # Broadcast conn across leading dims if needed
    diff = x_mat - y_exp  # (..., N_out, N_in)
    diffusive = diff * conn2d  # broadcasting on leading dims
    return k * diffusive.sum(axis=-1)  # (..., N_out)


@set_module_as('brainmass')
def additive_coupling(
    delayed_x: Callable | Array,
    conn: Array,
    k: Array
):
    r"""
    Additive coupling kernel (function form).

    Computes, for each target unit i over the last axis, the additive term

        current_i = k * sum_j conn[i, j] * x_{i, j}

    with full support for leading batch/time dimensions and unit-safe algebra.

    Parameters
    ----------
    delayed_x : Callable
        Zero-arg callable returning the source signal with shape ``(..., N_out, N_in)``
        or flattened ``(..., N_out*N_in)``. Typically a ``Prefetch``.
    conn : ArrayLike
        Connection weights with shape ``(N_out, N_in)``.
    k : ArrayLike
        Global coupling strength. Scalar or broadcastable to ``(..., N_out)``.

    Returns
    -------
    ArrayLike
        Coupling output with shape ``(..., N_out)``. Units are preserved when
        inputs are `Quantity`.

    Raises
    ------
    ValueError
        If shapes are incompatible with the expected conventions.
    """
    # x expected trailing dims to match connection (N_out, N_in) or flattened N_out*N_in
    x_val = delayed_x() if callable(delayed_x) else delayed_x
    n_out, n_in = conn.shape

    if x_val.ndim >= 2 and x_val.shape[-2:] == (n_out, n_in):
        x_mat = x_val
    elif x_val.shape[-1] == n_out * n_in:
        x_mat = u.math.reshape(x_val, (*x_val.shape[:-1], n_out, n_in))
    else:
        raise ValueError(
            f'x has incompatible shape {x_val.shape}; expected (..., {n_out}, {n_in}) '
            f'or flattened (..., {n_out * n_in}).'
        )

    additive = conn * x_mat  # broadcasting on leading dims
    return k * additive.sum(axis=-1)  # (..., N_out)


class DiffusiveCoupling(Module):
    r"""
    Diffusive coupling.

    This class implements a diffusive coupling mechanism for neural network modules.
    It simulates the following model:

    $$
    \mathrm{current}_i = k * \sum_j g_{ij} * (x_{D_{ij}} - y_i)
    $$

    where:
        - $\mathrm{current}_i$: the output current for neuron $i$
        - $g_{ij}$: the connection strength between neuron $i$ and neuron $j$
        - $x_{D_{ij}}$: the delayed state variable for neuron $j$, as seen by neuron $i$
        - $y_i$: the state variable for neuron i

    Parameters
    ----------
    x : Prefetch
        The delayed state variable for the source units.
    y : Prefetch
        The delayed state variable for the target units.
    conn : brainstate.typing.Array
        The connection matrix (1D or 2D array) specifying the coupling strengths between units.
    k: float
        The global coupling strength. Default is 1.0.

    """
    __module__ = 'brainmass'

    def __init__(
        self,
        x: Prefetch,
        y: Prefetch,
        conn: Parameter,
        k: Parameter = 1.0
    ):
        super().__init__()
        self.x = _check_type(x)
        self.y = _check_type(y)

        # global coupling strength
        self.k = Param.init(k)

        # Connection matrix (support 1D flattened (N_out*N_in,) or 2D (N_out, N_in))
        self.conn = Param.init(conn)
        ndim = self.conn.value().ndim
        if ndim not in (1, 2):
            raise ValueError(
                f'Connection must be 1D (flattened) or 2D matrix; got {ndim}D.'
            )

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        init_maybe_prefetch(self.x)
        init_maybe_prefetch(self.y)

    def update(self, *args, **kwargs):
        return diffusive_coupling(self.x, self.y, self.conn.value(), self.k.value())


class AdditiveCoupling(Module):
    r"""
    Additive coupling.

    This class implements an additive coupling mechanism for neural network modules.
    It simulates the following model:

    $$
    \mathrm{current}_i = k * \sum_j g_{ij} * x_{D_{ij}}
    $$

    where:
        - $\mathrm{current}_i$: the output current for neuron $i$
        - $g_{ij}$: the connection strength between neuron $i$ and neuron $j$
        - $x_{D_{ij}}$: the delayed state variable for neuron $j$, as seen by neuron $i$

    Parameters
    ----------
    x : Prefetch, Callable
        The delayed state variable for the source units.
    conn : brainstate.typing.Array
        The connection matrix (1D or 2D array) specifying the coupling strengths between units.
    k: float
        The global coupling strength. Default is 1.0.

    """
    __module__ = 'brainmass'

    def __init__(
        self,
        x: Prefetch,
        conn: Parameter,
        k: Parameter = 1.0
    ):
        super().__init__()
        self.x = _check_type(x)

        # global coupling strength
        self.k = Param.init(k)

        # Connection matrix
        self.conn = Param.init(conn)
        ndim = self.conn.value().ndim
        if ndim != 2:
            raise ValueError(f'Only support 2D connection matrix; got {ndim}D.')

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        init_maybe_prefetch(self.x)

    def update(self, *args, **kwargs):
        return additive_coupling(self.x, self.conn.value(), self.k.value())


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

    def _normalize(self, w_bb: Array) -> Array:
        """Normalize weights with standard L2 normalization."""
        w_b = u.math.exp(w_bb) * self.sc
        w_n_b = w_b / u.math.linalg.norm(w_b)
        if self.mask is not None:
            w_n_b = w_n_b * self.mask
        return w_n_b

    def _symmetric_normalize(self, w_ll: Array) -> Array:
        """Normalize weights with symmetric normalization (for lateral pathway)."""
        w = u.math.exp(w_ll) * self.sc
        w = 0.5 * (w + u.math.transpose(w, (0, 1)))
        w_n_l = w / u.linalg.norm(w)
        if self.mask is not None:
            w_n_l = w_n_l * self.mask
        return w_n_l

    def update(
        self,
        *args,
        **kwargs
    ) -> Tuple[Array, Array, Array]:
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
        delay_x = self.delayed_x()
        w_n_b = self.w_bb.value()  # feedback pathway weights (normalized via precompute)
        w_n_f = self.w_ff.value()  # feedforward pathway weights (normalized via precompute)
        w_n_l = self.w_ll.value()  # lateral pathway weights (symmetric normalized via precompute)

        # Long-range excitation (LE) terms from delayed inter-regional signal
        LEd_b = g_b * u.math.sum(w_n_b * delay_x, axis=1)
        LEd_f = g_f * u.math.sum(w_n_f * delay_x, axis=1)
        LEd_l = g_l * u.math.sum(w_n_l * delay_x, axis=1)

        # Get local population states
        P = self.P()
        E = self.E()
        I = self.I()
        eeg = E - I  # EEG-like proxy (difference of excitatory and inhibitory PSPs)

        # Local inhibition (diagonal Laplacian) terms
        dg_f = -u.math.sum(w_n_f, axis=1) * g_f * P  # feedforward pathway uses pyramidal state
        dg_b = -u.math.sum(w_n_b, axis=1) * g_b * eeg  # feedback pathway uses EEG proxy
        dg_l = -u.math.sum(w_n_l, axis=1) * g_l * eeg  # lateral pathway uses EEG proxy

        # Combine LE and dg terms for each population
        inp_P = LEd_l + dg_l  # pyramidal population (lateral pathway)
        inp_E = LEd_f + dg_f  # excitatory interneurons (feedforward pathway)
        inp_I = LEd_b + dg_b  # inhibitory interneurons (feedback pathway)

        # Return inputs for pyramidal, excitatory, and inhibitory populations
        return inp_P, inp_E, -inp_I
