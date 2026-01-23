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

import brainstate
import brainunit as u
from brainstate.nn import Param, Module, init_maybe_prefetch

from .typing import Parameter
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
    'diffusive_coupling',
    'additive_coupling',
    'laplacian_connectivity',
    'LaplacianConnParam',
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
    k: Array = 1.0
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
    conn : Param, array_like
        The connection matrix (1D or 2D array) specifying the coupling strengths between units.
    k: Param, array_like
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
    Compute unnormalized Laplacian for a simple 3-node graph:

    >>> import brainunit as u
    >>> W = u.math.asarray([[0., 1., 1.],
    ...                      [1., 0., 1.],
    ...                      [1., 1., 0.]])
    >>> L = laplacian_connectivity(W)
    >>> # L = [[ 2, -1, -1],
    >>> #      [-1,  2, -1],
    >>> #      [-1, -1,  2]]

    Compute symmetric normalized Laplacian:

    >>> L_sym = laplacian_connectivity(W, normalize="sym")
    >>> # L_sym = [[ 1.0, -0.5, -0.5],
    >>> #          [-0.5,  1.0, -0.5],
    >>> #          [-0.5, -0.5,  1.0]]
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

    """
    __module__ = 'brainmass'

    def __init__(
        self,
        W: Array,
        mask: Optional[Array] = None,
        fit: bool = True,
        normalize: Optional[Literal["rw", "sym"]] = None,
        eps: float = 1e-12,
        return_diag: bool = False,
    ):
        super().__init__(W, fit=fit, precompute=self.normalize)
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

    def normalize(self, weight):
        weight = u.math.exp(u.get_magnitude(weight)) * self.original_W
        if self.mask is not None:
            weight = weight * self.mask
        return laplacian_connectivity(
            weight,
            normalize=self.normalize,
            eps=self.eps,
            return_diag=self.return_diag,
        )
