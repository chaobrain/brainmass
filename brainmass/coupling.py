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


from typing import Union, Tuple, Callable, Literal, Optional

import braintools
import brainstate
import brainunit as u
import numpy as np
from brainstate.nn import Param, Module, init_maybe_prefetch, Transform, IdentityT, Regularization

from .typing import Parameter, Initializer
from .utils import set_module_as

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
    'SigmoidalCoupling',
    'HyperbolicTangentCoupling',
    'SigmoidalJansenRitCoupling',
    'diffusive_coupling',
    'additive_coupling',
    'sigmoidal_coupling',
    'hyperbolic_tangent_coupling',
    'sigmoidal_jansen_rit_coupling',
    'laplacian_connectivity',
    'LaplacianConnParam',
]


def _check_type(x):
    if not (isinstance(x, _PREFETCH_TYPES) or callable(x)):
        raise TypeError(f'The argument must be a Prefetch or Callable, got {x}')
    return x


def _coupling_conn_xmat(delayed_x: Callable | Array, conn: Array):
    r"""Resolve ``(conn2d, x_mat)`` for additive-style couplings (no target ``y``).

    Shared shape machinery for the post-/pre-nonlinearity couplings, which weight
    a source signal by a connectivity matrix and sum over the source axis. Unlike
    :func:`diffusive_coupling`, there is no target signal to infer ``N_out`` from,
    so a flattened ``conn`` is assumed **square** (``N_out == N_in``).

    Parameters
    ----------
    delayed_x : Callable or ArrayLike
        Zero-arg callable (e.g. a ``Prefetch`` / ``prefetch_delay``) or array
        returning the source signal, shaped ``(..., N_out, N_in)`` or flattened
        ``(..., N_out * N_in)``.
    conn : ArrayLike
        Connection weights, either ``(N_out, N_in)`` or square-flattened
        ``(N_out * N_in,)`` with ``N_out == N_in``.

    Returns
    -------
    conn2d : ArrayLike
        The ``(N_out, N_in)`` connection matrix.
    x_mat : ArrayLike
        The source signal reshaped to ``(..., N_out, N_in)``.

    Raises
    ------
    ValueError
        If ``conn`` is neither 1-D (square) nor 2-D, a flattened ``conn`` is not a
        perfect square, or ``x`` is incompatible with ``(..., N_out, N_in)``.
    """
    x_val = delayed_x() if callable(delayed_x) else delayed_x
    if x_val.ndim < 1:
        raise ValueError(f'x must have at least 1 dimension; got shape {x_val.shape}')

    if conn.ndim == 2:
        n_out, n_in = conn.shape
        conn2d = conn
    elif conn.ndim == 1:
        n = int(round(float(np.sqrt(conn.size))))
        if n * n != conn.size:
            raise ValueError(
                f'Flattened conn length {conn.size} is not a perfect square; these '
                f'couplings have no target to infer N_out, so a 1-D conn must be '
                f'square (N_out == N_in). Pass a 2-D (N_out, N_in) conn for '
                f'non-square connectivity.'
            )
        n_out = n_in = n
        conn2d = u.math.reshape(conn, (n, n))
    else:
        raise ValueError(
            f'conn must be 1-D (square-flattened) or 2-D; got {conn.ndim}-D.'
        )

    if x_val.ndim >= 2 and x_val.shape[-2:] == (n_out, n_in):
        x_mat = x_val
    elif x_val.shape[-1] == n_out * n_in:
        x_mat = u.math.reshape(x_val, (*x_val.shape[:-1], n_out, n_in))
    else:
        raise ValueError(
            f'x has incompatible shape {x_val.shape}; expected (..., {n_out}, {n_in}) '
            f'or flattened (..., {n_out * n_in}).'
        )
    return conn2d, x_mat


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
    conn : Param, array_like
        The connection matrix (1D or 2D array) specifying the coupling strengths between units.
    k: Param, array_like
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


@set_module_as('brainmass')
def additive_coupling(
    delayed_x: Callable | Array,
    conn: Array,
    k: Array = 1.0,
    b: Array = 0.0,
):
    r"""
    Additive (linear) coupling kernel (function form).

    Computes, for each target unit i over the last axis, the additive term

        current_i = k * sum_j conn[i, j] * x_{i, j} + b

    with full support for leading batch/time dimensions and unit-safe algebra.
    This is TVB's ``Linear`` coupling; the global coupling strength ``k`` is TVB's
    ``G`` (``G ≡ k``) and ``b`` is its offset/bias.

    Parameters
    ----------
    delayed_x : Callable
        Zero-arg callable returning the source signal with shape ``(..., N_out, N_in)``
        or flattened ``(..., N_out*N_in)``. Typically a ``Prefetch``.
    conn : ArrayLike
        Connection weights with shape ``(N_out, N_in)``.
    k : ArrayLike
        Global coupling strength (TVB ``G``). Scalar or broadcastable to ``(..., N_out)``.
    b : ArrayLike, default 0.0
        Additive offset/bias, added after the weighted sum. The default ``0.0`` is
        the additive identity, so it reproduces the bias-free coupling bit-for-bit
        (and, by brainunit's convention, does not disturb a unit-carrying output).
        A non-zero ``b`` should carry the same units as ``k * sum_j conn x``.

    Returns
    -------
    ArrayLike
        Coupling output with shape ``(..., N_out)``. Units are preserved when
        inputs are `Quantity`.

    Raises
    ------
    ValueError
        If shapes are incompatible with the expected conventions.

    Examples
    --------
    >>> import brainmass
    >>> import jax.numpy as jnp
    >>> conn = jnp.ones((2, 2))
    >>> x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    >>> out = brainmass.additive_coupling(x, conn, k=1.0, b=0.5)
    >>> [float(v) for v in out]
    [3.5, 7.5]
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
    return k * additive.sum(axis=-1) + b  # (..., N_out); b=0.0 is the identity


class AdditiveCoupling(Module):
    r"""
    Additive (linear) coupling.

    This class implements an additive coupling mechanism for neural network modules.
    It simulates the following model:

    $$
    \mathrm{current}_i = k * \sum_j g_{ij} * x_{D_{ij}} + b
    $$

    where:
        - $\mathrm{current}_i$: the output current for neuron $i$
        - $g_{ij}$: the connection strength between neuron $i$ and neuron $j$
        - $x_{D_{ij}}$: the delayed state variable for neuron $j$, as seen by neuron $i$
        - $b$: an additive offset/bias

    This is TVB's ``Linear`` coupling; the global strength ``k`` is TVB's ``G``
    (``G ≡ k``).

    Parameters
    ----------
    x : Prefetch, Callable
        The delayed state variable for the source units.
    conn : Param, array_like
        The connection matrix (1D or 2D array) specifying the coupling strengths between units.
    k: Param, array_like
        The global coupling strength. Default is 1.0.
    b: Param, array_like
        Additive offset/bias added after the weighted sum. Default is ``0.0``, which
        reproduces the bias-free coupling bit-for-bit. Pass a trainable ``Param`` to
        fit it (``Param.init(0.0)`` yields a non-trainable ``Const``, so the default
        adds no trainable state).

    """
    __module__ = 'brainmass'

    def __init__(
        self,
        x: Prefetch,
        conn: Parameter,
        k: Parameter = 1.0,
        b: Parameter = 0.0,
    ):
        super().__init__()
        self.x = _check_type(x)

        # global coupling strength
        self.k = Param.init(k)
        # additive offset/bias (b=0.0 -> Const, no trainable state added)
        self.b = Param.init(b)

        # Connection matrix
        self.conn = Param.init(conn)
        ndim = self.conn.value().ndim
        if ndim != 2:
            raise ValueError(f'Only support 2D connection matrix; got {ndim}D.')

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        init_maybe_prefetch(self.x)

    def update(self, *args, **kwargs):
        return additive_coupling(
            self.x, self.conn.value(), self.k.value(), self.b.value()
        )


@set_module_as('brainmass')
def sigmoidal_coupling(
    delayed_x: Callable | Array,
    conn: Array,
    k: Array = 1.0,
    a: Array = 1.0,
    b: Array = 0.0,
    slope: Array = 1.0,
    midpoint: Array = 0.0,
):
    r"""
    Sigmoidal coupling kernel (function form, post-nonlinearity).

    The TVB ``Sigmoidal`` coupling: a logistic nonlinearity is applied **after** the
    network sum, so each target's coupled input saturates smoothly,

    .. math::

        c_i = k \, \sigma\!\left(\mathrm{slope} \cdot
              \left(a \sum_j w_{ij} x_j + b - \mathrm{midpoint}\right)\right),
        \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.

    Parameters
    ----------
    delayed_x : Callable or ArrayLike
        Zero-arg callable (e.g. a ``Prefetch`` / ``prefetch_delay``) or array
        returning the source signal, shaped ``(..., N_out, N_in)`` or flattened
        ``(..., N_out * N_in)``.
    conn : ArrayLike
        Connection weights, ``(N_out, N_in)`` or square-flattened ``(N_out * N_in,)``.
    k : ArrayLike, default 1.0
        Global coupling strength (TVB ``G``; ``G ≡ k``). Scales the saturated output.
    a : ArrayLike, default 1.0
        Linear scaling of the network sum before the sigmoid.
    b : ArrayLike, default 0.0
        Linear offset added to the network sum before the sigmoid.
    slope : ArrayLike, default 1.0
        Steepness of the logistic nonlinearity.
    midpoint : ArrayLike, default 0.0
        Centre of the logistic nonlinearity.

    Returns
    -------
    ArrayLike
        Coupling output with shape ``(..., N_out)``. The logistic is dimensionless,
        so the output carries the units of ``k``.

    See Also
    --------
    hyperbolic_tangent_coupling : symmetric (``tanh``) post-nonlinearity.
    sigmoidal_jansen_rit_coupling : pre-nonlinearity Jansen-Rit sigmoid.
    additive_coupling : the underlying linear (``k * sum + b``) coupling.

    Notes
    -----
    The argument of the logistic must be dimensionless; the network sum is reduced
    to its magnitude (``brainunit.get_magnitude``) before the nonlinearity, mirroring
    the Jansen-Rit house style. At zero net input the output is
    ``k * sigma(slope * (b - midpoint))`` (``= k * sigma(-slope * midpoint)`` for the
    default ``a = 1, b = 0``).

    References
    ----------
    .. [1] Sanz-Leon, P., Knock, S. A., Spiegler, A., & Jirsa, V. K. (2015).
           Mathematical framework for large-scale brain network modeling in The
           Virtual Brain. NeuroImage, 111, 385-430.

    Examples
    --------
    >>> import brainmass
    >>> import jax.numpy as jnp
    >>> conn = jnp.ones((2, 2))
    >>> x = jnp.zeros((2, 2))  # zero net input
    >>> out = brainmass.sigmoidal_coupling(x, conn, k=1.0, slope=1.0, midpoint=0.0)
    >>> [round(float(v), 3) for v in out]
    [0.5, 0.5]
    """
    conn2d, x_mat = _coupling_conn_xmat(delayed_x, conn)
    net = u.get_magnitude((conn2d * x_mat).sum(axis=-1))  # (..., N_out)
    return k * u.math.sigmoid(slope * (a * net + b - midpoint))


class SigmoidalCoupling(Module):
    r"""
    Sigmoidal coupling (TVB ``Sigmoidal``, post-nonlinearity).

    Applies a logistic nonlinearity after the network sum:

    $$
    c_i = k \, \sigma\!\left(\mathrm{slope} \cdot
          \left(a \sum_j g_{ij} x_{D_{ij}} + b - \mathrm{midpoint}\right)\right)
    $$

    where $\sigma$ is the logistic function and $g_{ij}$ the connection strength.
    ``k`` is the global coupling strength (TVB ``G``; ``G ≡ k``).

    Parameters
    ----------
    x : Prefetch, Callable
        The (optionally delayed) source state, shaped ``(..., N_out, N_in)``.
    conn : Param, array_like
        Connection matrix ``(N_out, N_in)`` or square-flattened ``(N_out * N_in,)``.
    k : Param, array_like, default 1.0
        Global coupling strength (TVB ``G``).
    a : Param, array_like, default 1.0
        Linear scaling of the network sum before the sigmoid.
    b : Param, array_like, default 0.0
        Linear offset added before the sigmoid.
    slope : Param, array_like, default 1.0
        Steepness of the logistic.
    midpoint : Param, array_like, default 0.0
        Centre of the logistic.

    See Also
    --------
    HyperbolicTangentCoupling : symmetric (``tanh``) post-nonlinearity.
    SigmoidalJansenRitCoupling : pre-nonlinearity Jansen-Rit sigmoid.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        x: Prefetch,
        conn: Parameter,
        k: Parameter = 1.0,
        a: Parameter = 1.0,
        b: Parameter = 0.0,
        slope: Parameter = 1.0,
        midpoint: Parameter = 0.0,
    ):
        super().__init__()
        self.x = _check_type(x)

        self.k = Param.init(k)
        self.a = Param.init(a)
        self.b = Param.init(b)
        self.slope = Param.init(slope)
        self.midpoint = Param.init(midpoint)

        self.conn = Param.init(conn)
        ndim = self.conn.value().ndim
        if ndim not in (1, 2):
            raise ValueError(
                f'Connection must be 1D (square-flattened) or 2D matrix; got {ndim}D.'
            )

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        init_maybe_prefetch(self.x)

    def update(self, *args, **kwargs):
        return sigmoidal_coupling(
            self.x, self.conn.value(), self.k.value(), self.a.value(),
            self.b.value(), self.slope.value(), self.midpoint.value(),
        )


@set_module_as('brainmass')
def hyperbolic_tangent_coupling(
    delayed_x: Callable | Array,
    conn: Array,
    k: Array = 0.5,
    scale: Array = 2.0,
):
    r"""
    Hyperbolic-tangent coupling kernel (function form, post-nonlinearity).

    The TVB ``HyperbolicTangent`` coupling: a symmetric saturating nonlinearity is
    applied **after** the network sum,

    .. math::

        c_i = k \, \tanh\!\left(\mathrm{scale} \sum_j w_{ij} x_j\right).

    Parameters
    ----------
    delayed_x : Callable or ArrayLike
        Zero-arg callable (e.g. a ``Prefetch`` / ``prefetch_delay``) or array
        returning the source signal, shaped ``(..., N_out, N_in)`` or flattened
        ``(..., N_out * N_in)``.
    conn : ArrayLike
        Connection weights, ``(N_out, N_in)`` or square-flattened ``(N_out * N_in,)``.
    k : ArrayLike, default 0.5
        Global coupling strength (TVB ``G``; ``G ≡ k``). The output saturates to
        ``±k``.
    scale : ArrayLike, default 2.0
        Scaling of the network sum before the ``tanh``.

    Returns
    -------
    ArrayLike
        Coupling output with shape ``(..., N_out)``. ``tanh`` is dimensionless, so the
        output carries the units of ``k``.

    See Also
    --------
    sigmoidal_coupling : asymmetric (logistic) post-nonlinearity.
    additive_coupling : the underlying linear coupling.

    Notes
    -----
    The argument of ``tanh`` must be dimensionless; the network sum is reduced to its
    magnitude (``brainunit.get_magnitude``) before the nonlinearity. As ``|sum| → ∞``
    the output saturates to ``±k``.

    References
    ----------
    .. [1] Sanz-Leon, P., Knock, S. A., Spiegler, A., & Jirsa, V. K. (2015).
           Mathematical framework for large-scale brain network modeling in The
           Virtual Brain. NeuroImage, 111, 385-430.

    Examples
    --------
    >>> import brainmass
    >>> import jax.numpy as jnp
    >>> conn = jnp.ones((2, 2))
    >>> x = jnp.full((2, 2), 1e3)  # large positive net input saturates to +k
    >>> out = brainmass.hyperbolic_tangent_coupling(x, conn, k=0.5, scale=2.0)
    >>> [round(float(v), 3) for v in out]
    [0.5, 0.5]
    """
    conn2d, x_mat = _coupling_conn_xmat(delayed_x, conn)
    net = u.get_magnitude((conn2d * x_mat).sum(axis=-1))  # (..., N_out)
    return k * u.math.tanh(scale * net)


class HyperbolicTangentCoupling(Module):
    r"""
    Hyperbolic-tangent coupling (TVB ``HyperbolicTangent``, post-nonlinearity).

    Applies a symmetric saturating nonlinearity after the network sum:

    $$
    c_i = k \, \tanh\!\left(\mathrm{scale} \sum_j g_{ij} x_{D_{ij}}\right)
    $$

    where $g_{ij}$ is the connection strength. ``k`` is the global coupling strength
    (TVB ``G``; ``G ≡ k``) and the output saturates to $\pm k$.

    Parameters
    ----------
    x : Prefetch, Callable
        The (optionally delayed) source state, shaped ``(..., N_out, N_in)``.
    conn : Param, array_like
        Connection matrix ``(N_out, N_in)`` or square-flattened ``(N_out * N_in,)``.
    k : Param, array_like, default 0.5
        Global coupling strength (TVB ``G``).
    scale : Param, array_like, default 2.0
        Scaling of the network sum before the ``tanh``.

    See Also
    --------
    SigmoidalCoupling : asymmetric (logistic) post-nonlinearity.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        x: Prefetch,
        conn: Parameter,
        k: Parameter = 0.5,
        scale: Parameter = 2.0,
    ):
        super().__init__()
        self.x = _check_type(x)

        self.k = Param.init(k)
        self.scale = Param.init(scale)

        self.conn = Param.init(conn)
        ndim = self.conn.value().ndim
        if ndim not in (1, 2):
            raise ValueError(
                f'Connection must be 1D (square-flattened) or 2D matrix; got {ndim}D.'
            )

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        init_maybe_prefetch(self.x)

    def update(self, *args, **kwargs):
        return hyperbolic_tangent_coupling(
            self.x, self.conn.value(), self.k.value(), self.scale.value()
        )


@set_module_as('brainmass')
def sigmoidal_jansen_rit_coupling(
    delayed_x: Callable | Array,
    conn: Array,
    k: Array = 1.0,
    cmin: Array = 0.0,
    cmax: Array = 0.005,
    midpoint: Array = 6.0,
    r: Array = 0.56,
):
    r"""
    Sigmoidal Jansen-Rit coupling kernel (function form, pre-nonlinearity).

    The TVB ``SigmoidalJansenRit`` coupling: the sigmoid is applied to each source
    **before** the network sum (a firing-rate transfer of the presynaptic potential),

    .. math::

        c_i = k \sum_j w_{ij} \, \sigma_{\mathrm{JR}}(x_j),
        \qquad
        \sigma_{\mathrm{JR}}(x) = c_{\min}
        + \frac{c_{\max} - c_{\min}}{1 + e^{\,r\,(\mathrm{midpoint} - x)}}.

    The source ``x_j`` is whatever the caller prefetches -- e.g. the Jansen-Rit
    ``y1 - y2`` pyramidal input.

    Parameters
    ----------
    delayed_x : Callable or ArrayLike
        Zero-arg callable (e.g. a ``Prefetch`` / ``prefetch_delay``) or array
        returning the presynaptic source, shaped ``(..., N_out, N_in)`` or flattened
        ``(..., N_out * N_in)``.
    conn : ArrayLike
        Connection weights, ``(N_out, N_in)`` or square-flattened ``(N_out * N_in,)``.
    k : ArrayLike, default 1.0
        Global coupling strength (TVB ``G``; ``G ≡ k``).
    cmin : ArrayLike, default 0.0
        Lower asymptote of the sigmoid (firing rate as ``x → -∞``).
    cmax : ArrayLike, default 0.005
        Upper asymptote of the sigmoid (firing rate as ``x → +∞``).
    midpoint : ArrayLike, default 6.0
        Half-activation potential (centre of the sigmoid).
    r : ArrayLike, default 0.56
        Steepness of the sigmoid.

    Returns
    -------
    ArrayLike
        Coupling output with shape ``(..., N_out)``. ``σ_JR`` is dimensionless, so the
        output carries the units of ``k * conn``.

    See Also
    --------
    sigmoidal_coupling : post-nonlinearity (sigmoid after the sum).
    brainmass.JansenRitStep : the Jansen-Rit neural mass whose output this couples.

    Notes
    -----
    The sigmoid argument must be dimensionless; the source is reduced to its magnitude
    (``brainunit.get_magnitude``) before the nonlinearity. At ``x = midpoint`` the
    transfer equals ``(cmin + cmax) / 2``, so ``c_i = k * (cmin + cmax) / 2 * Σ_j
    w_ij``; far below/above ``midpoint`` it tends to ``cmin`` / ``cmax`` respectively.

    References
    ----------
    .. [1] Jansen, B. H., & Rit, V. G. (1995). Electroencephalogram and visual evoked
           potential generation in a mathematical model of coupled cortical columns.
           Biological Cybernetics, 73(4), 357-366.

    Examples
    --------
    >>> import brainmass
    >>> import jax.numpy as jnp
    >>> conn = jnp.array([[0.2, 0.3], [0.5, 0.1]])
    >>> x = jnp.full((2, 2), 6.0)  # at the midpoint -> sigma = (cmin + cmax) / 2
    >>> out = brainmass.sigmoidal_jansen_rit_coupling(x, conn, k=1.0)
    >>> [round(float(v), 6) for v in out]
    [0.00125, 0.0015]
    """
    conn2d, x_mat = _coupling_conn_xmat(delayed_x, conn)
    xmag = u.get_magnitude(x_mat)  # (..., N_out, N_in), dimensionless
    sigma = cmin + (cmax - cmin) / (1.0 + u.math.exp(r * (midpoint - xmag)))
    return k * (conn2d * sigma).sum(axis=-1)  # (..., N_out)


class SigmoidalJansenRitCoupling(Module):
    r"""
    Sigmoidal Jansen-Rit coupling (TVB ``SigmoidalJansenRit``, pre-nonlinearity).

    Applies the Jansen-Rit firing-rate sigmoid to each source **before** the network
    sum:

    $$
    c_i = k \sum_j g_{ij} \, \sigma_{\mathrm{JR}}(x_{D_{ij}}),
    \qquad
    \sigma_{\mathrm{JR}}(x) = c_{\min}
    + \frac{c_{\max} - c_{\min}}{1 + e^{\,r\,(\mathrm{midpoint} - x)}}.
    $$

    ``k`` is the global coupling strength (TVB ``G``; ``G ≡ k``). The source is
    whatever the caller prefetches (e.g. the Jansen-Rit ``y1 - y2``).

    Parameters
    ----------
    x : Prefetch, Callable
        The (optionally delayed) presynaptic source, shaped ``(..., N_out, N_in)``.
    conn : Param, array_like
        Connection matrix ``(N_out, N_in)`` or square-flattened ``(N_out * N_in,)``.
    k : Param, array_like, default 1.0
        Global coupling strength (TVB ``G``).
    cmin : Param, array_like, default 0.0
        Lower asymptote of the sigmoid.
    cmax : Param, array_like, default 0.005
        Upper asymptote of the sigmoid.
    midpoint : Param, array_like, default 6.0
        Half-activation potential.
    r : Param, array_like, default 0.56
        Steepness of the sigmoid.

    See Also
    --------
    SigmoidalCoupling : post-nonlinearity sigmoid.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        x: Prefetch,
        conn: Parameter,
        k: Parameter = 1.0,
        cmin: Parameter = 0.0,
        cmax: Parameter = 0.005,
        midpoint: Parameter = 6.0,
        r: Parameter = 0.56,
    ):
        super().__init__()
        self.x = _check_type(x)

        self.k = Param.init(k)
        self.cmin = Param.init(cmin)
        self.cmax = Param.init(cmax)
        self.midpoint = Param.init(midpoint)
        self.r = Param.init(r)

        self.conn = Param.init(conn)
        ndim = self.conn.value().ndim
        if ndim not in (1, 2):
            raise ValueError(
                f'Connection must be 1D (square-flattened) or 2D matrix; got {ndim}D.'
            )

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        init_maybe_prefetch(self.x)

    def update(self, *args, **kwargs):
        return sigmoidal_jansen_rit_coupling(
            self.x, self.conn.value(), self.k.value(), self.cmin.value(),
            self.cmax.value(), self.midpoint.value(), self.r.value(),
        )


@set_module_as('brainmass')
def laplacian_connectivity(
    W: Array,
    *,
    normalize: Optional[Literal["rw", "sym"]] = None,
    eps: float = 1e-12,
    return_diag: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    r"""
    Build graph Laplacian matrix from adjacency/connectivity matrix.

    The graph Laplacian is a fundamental matrix representation used in spectral graph
    theory, graph signal processing, and network analysis. Given an adjacency matrix W
    and degree matrix D = diag(sum_j W_ij), this function computes one of three standard
    Laplacian forms.

    **Unnormalized Laplacian** (``normalize=None``):

    $$
    L = W - D
    $$

    **Random Walk Normalized Laplacian** (``normalize="rw"``):

    $$
    L_{\mathrm{rw}} = D^{-1} W = D^{-1} L - I
    $$

    This form is asymmetric and commonly used in diffusion processes and random walks on graphs.

    **Symmetric Normalized Laplacian** (``normalize="sym"``):

    $$
    L_{\mathrm{sym}} = D^{-1/2} W D^{-1/2} = D^{-1/2} L D^{-1/2} - I
    $$

    This form is symmetric, preserves spectral properties, and is widely used in spectral clustering
    and graph neural networks.

    Parameters
    ----------
    W : ArrayLike
        Adjacency or connectivity matrix with shape ``(N, N)`` representing weighted edges
        between N nodes. Should contain non-negative weights. For directed graphs, W[i, j]
        represents edge weight from node j to node i.
    normalize : {None, "rw", "sym"}, optional
        Normalization mode for the Laplacian:

        - ``None`` (default): Returns unnormalized Laplacian L = W - D
        - ``"rw"``: Returns random walk normalized Laplacian L_rw = D^{-1}W - I
        - ``"sym"``: Returns symmetric normalized Laplacian L_sym = D^{-1/2}W D^{-1/2} - I
    eps : float, default=1e-12
        Small constant added for numerical stability when computing D^{-1} or D^{-1/2},
        preventing division by zero for isolated nodes (zero degree).
    return_diag : bool, default=False
        If True, return a tuple ``(L, d)`` where ``L`` is the Laplacian matrix and ``d`` is
        the degree vector (row sums of W). If False (default), return only the Laplacian matrix.

    Returns
    -------
    ArrayLike or tuple of ArrayLike
        If ``return_diag=False`` (default): Returns the graph Laplacian matrix with shape ``(N, N)``
        and dtype as input W.
        If ``return_diag=True``: Returns a tuple ``(L, d)`` where ``L`` is the Laplacian matrix
        with shape ``(N, N)`` and ``d`` is the degree vector with shape ``(N,)``.
        If W carries units via `brainunit`, the output preserves unit consistency.

    Raises
    ------
    ValueError
        If ``normalize`` is not one of {None, "rw", "sym"}.

    Notes
    -----
    - **Assumptions**: This function assumes non-negative edge weights. For directed graphs,
      interpretation requires care as the degree matrix D uses row sums.
    - **Numerical stability**: The ``eps`` parameter prevents division-by-zero errors for
      isolated nodes with degree zero. Nodes with degree < eps will be treated as having
      degree = eps.
    - **Unit safety**: Fully compatible with `brainunit` for unit-safe array operations.
    - **Use cases**:
        - Unnormalized: Best for preserving absolute connectivity structure and scale
        - Random walk: Suitable for diffusion analysis and probabilistic processes
        - Symmetric: Preferred for spectral analysis, clustering, and eigendecomposition

    Examples
    --------
    Compute the (combinatorial) Laplacian for a simple 3-node graph:

    >>> import brainmass
    >>> import brainunit as u
    >>> W = u.math.asarray([[0., 1., 1.],
    ...                      [1., 0., 1.],
    ...                      [1., 1., 0.]])
    >>> L = brainmass.laplacian_connectivity(W)
    >>> L.shape
    (3, 3)
    >>> # every row of the combinatorial Laplacian sums to zero
    >>> bool(abs(L.sum(axis=1)).max() < 1e-6)
    True

    Compute the symmetric normalized Laplacian:

    >>> L_sym = brainmass.laplacian_connectivity(W, normalize="sym")
    >>> L_sym.shape
    (3, 3)
    """
    W = u.math.asarray(W)
    d = u.math.sum(W, axis=-1)  # (N,)
    if normalize is None:
        L = W - u.math.diag(d)
        return (L, d) if return_diag else L

    n = W.shape[-1]
    I = u.math.eye(n, dtype=W.dtype, unit=u.get_unit(W))

    if normalize == "rw":
        inv_d = 1.0 / u.math.maximum(d, eps)
        DinvW = W * inv_d[:, None]
        L = DinvW - I
        return (L, d) if return_diag else L

    if normalize == "sym":
        inv_sqrt_d = 1.0 / u.math.sqrt(u.math.maximum(d, eps))
        Wn = (W * inv_sqrt_d[:, None]) * inv_sqrt_d[None, :]
        L = Wn - I
        return (L, d) if return_diag else L

    raise ValueError(
        f"Unknown normalize={normalize}, "
        f"only None, 'rw', 'sym' are supported."
    )


class LaplacianConnParam(Param):
    r"""
    Graph Laplacian connectivity module.

    This module computes the graph Laplacian matrix from a given adjacency/connectivity
    matrix using one of three standard forms: unnormalized, random walk normalized,
    or symmetric normalized.

    Parameters
    ----------
    W : Param, array_like
        Adjacency or connectivity matrix with shape ``(N, N)`` representing weighted edges
        between N nodes.
    normalize : {None, "rw", "sym"}, optional
        Normalization mode for the Laplacian:

        - ``None`` (default): Returns unnormalized Laplacian L = W - D
        - ``"rw"``: Returns random walk normalized Laplacian L_rw = D^{-1}W - I
        - ``"sym"``: Returns symmetric normalized Laplacian L_sym = D^{-1/2}W D^{-1/2} - I
    eps : float, default=1e-12
        Small constant added for numerical stability when computing D^{-1} or D^{-1/2}.
    t : Transform, optional
        Optional transform applied to W before computing the Laplacian. Default is IdentityT (no transform).
    return_diag : bool, default=False
        If True, the module's value will be a tuple (L, d) where L is the Laplacian matrix
        and d is the degree vector. If False (default), the module's value will be just the Laplacian matrix L.

    """
    __module__ = 'brainmass'

    def __init__(
        self,
        W: Array,
        t: Transform = IdentityT(),
        reg: Regularization = None,
        mask: Optional[Array] = None,
        fit: bool = True,
        normalize: Optional[Literal["rw", "sym"]] = None,
        eps: float = 1e-12,
        return_diag: bool = False,
    ):
        # ``normalize`` is the normalization *mode* string ("rw"/"sym"/None); the
        # precompute callback is the (renamed) ``_laplacian`` method, so the two
        # no longer collide on the ``normalize`` name. precompute is invoked
        # lazily by Param.value()/cache(), after these attributes are set.
        super().__init__(W, fit=fit, precompute=self._laplacian, t=t, reg=reg)
        self.mask = mask
        self.original_W = W
        self.normalize = normalize
        self.return_diag = return_diag
        self.eps = eps
        if mask is not None:
            if mask.shape != W.shape:
                raise ValueError(
                    f'Mask shape {mask.shape} must match W shape {W.shape}.'
                )

    def _laplacian(self, weight):
        weight = u.math.exp(u.get_magnitude(weight)) * self.original_W
        if self.mask is not None:
            weight = weight * self.mask
        return laplacian_connectivity(
            weight,
            normalize=self.normalize,
            eps=self.eps,
            return_diag=self.return_diag,
        )


class AdditiveConn(Module):
    r"""Additive recurrent connection: a dense linear map of a model state.

    Reads a named hidden state of ``model`` and applies a trainable linear
    projection. Shared by the HORN layers (``state='y'``) and the Jansen-Rit
    layers (``state='M'``); ``brainunit.get_magnitude`` is a no-op on the unitless
    HORN state and strips units from the unit-carrying Jansen-Rit state.

    Parameters
    ----------
    model : brainstate.nn.Module
        The dynamics module whose state is read.
    state : str, default 'y'
        Name of the model attribute (a ``State``) to read; e.g. ``'y'`` (HORN
        velocity) or ``'M'`` (Jansen-Rit pyramidal potential).
    w_init : Callable, default KaimingNormal
        Initializer for the linear weight matrix.
    b_init : Callable, default ZeroInit
        Initializer for the linear bias.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        model: Module,
        state: str = 'y',
        w_init: Callable = braintools.init.KaimingNormal(),
        b_init: Callable = braintools.init.Constant(0.0),
    ):
        super().__init__()

        self.model = model
        self.state = state
        self.linear = brainstate.nn.Linear(
            self.model.in_size, self.model.out_size, w_init=w_init, b_init=b_init
        )

    def update_tr(self, *args, **kwargs):
        return 0.

    def update(self, *args, **kwargs):
        x = u.get_magnitude(getattr(self.model, self.state).value)
        return self.linear(x)


class DelayedAdditiveConn(Module):
    r"""Delayed additive recurrent connection.

    Prefetches a delayed model state and applies :func:`additive_coupling`.
    Shared by the HORN layers (``state='y'``) and the Jansen-Rit layers
    (``state='M'``).

    Parameters
    ----------
    model : brainstate.nn.Dynamics
        The dynamics module whose delayed state is read.
    delay_time : Initializer or ArrayLike
        Per-connection delay times, shaped ``(n_hidden, n_hidden)``.
    state : str, default 'y'
        Name of the model state to prefetch with delay.
    delay_init : Initializer, default ZeroInit
        Initializer for the delay buffer.
    w_init : Callable, default KaimingNormal
        Initializer for the coupling weight matrix.
    k : Parameter, default 1.0
        Global coupling strength.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        model: Module,
        delay_time: Initializer,
        state: str = 'y',
        delay_init: Initializer = braintools.init.Constant(0.0),
        w_init: Callable = braintools.init.KaimingNormal(),
        k: Parameter = 1.0,
    ):
        super().__init__()

        n_hidden = model.varshape[0]
        delay_time = braintools.init.param(delay_time, (n_hidden, n_hidden))
        neuron_idx = np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
        self.prefetch = model.prefetch_delay(state, delay_time, neuron_idx, init=delay_init)
        self.weights = Param(braintools.init.param(w_init, (n_hidden, n_hidden)))
        self.k = Param.init(k)

    def update_tr(self, *args, **kwargs):
        return 0.

    def update(self, *args, **kwargs):
        delayed = u.get_magnitude(self.prefetch())
        return additive_coupling(delayed, self.weights.value(), self.k.value())
