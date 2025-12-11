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


from typing import Union, Tuple, Callable

import brainstate
import brainunit as u
from brainstate.nn._dynamics import maybe_init_prefetch

from ._common import set_module_as

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

__all__ = [
    'DiffusiveCoupling',
    'AdditiveCoupling',
    'diffusive_coupling',
    'additive_coupling',
]


def _check_type(x):
    if not (isinstance(x, _PREFETCH_TYPES) or callable(x)):
        raise TypeError(f'The argument must be a Prefetch or Callable, got {x}')
    return x


@set_module_as('brainmass')
def diffusive_coupling(
    delayed_x: Callable | brainstate.typing.ArrayLike,
    y: Callable | brainstate.typing.ArrayLike,
    conn: brainstate.typing.ArrayLike,
    k: brainstate.typing.ArrayLike,
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
    delayed_x: Callable | brainstate.typing.ArrayLike,
    conn: brainstate.typing.ArrayLike,
    k: brainstate.typing.ArrayLike
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


class DiffusiveCoupling(brainstate.nn.Module):
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

    Attributes
    ----------
    x : Prefetch
        The delayed state variable for the source units.
    y : Prefetch
        The delayed state variable for the target units.
    conn : Array
        The connection matrix.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        x: Prefetch,
        y: Prefetch,
        conn: brainstate.typing.Array,
        k: float = 1.0
    ):
        super().__init__()
        self.x = _check_type(x)
        self.y = _check_type(y)
        self.k = k

        # Connection matrix (support 1D flattened (N_out*N_in,) or 2D (N_out, N_in))
        self.conn = u.math.asarray(conn)
        if self.conn.ndim not in (1, 2):
            raise ValueError(
                f'Connection must be 1D (flattened) or 2D matrix; got {self.conn.ndim}D.'
            )

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        maybe_init_prefetch(self.x)
        maybe_init_prefetch(self.y)

    def update(self):
        return diffusive_coupling(self.x, self.y, self.conn, self.k)


class AdditiveCoupling(brainstate.nn.Module):
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

    Attributes
    ----------
    x : Prefetch
        The delayed state variable for the source units.
    conn : Array
        The connection matrix.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        x: Prefetch,
        conn: brainstate.typing.Array,
        k: float = 1.0
    ):
        super().__init__()
        self.x = _check_type(x)
        self.k = k

        # Connection matrix
        self.conn = u.math.asarray(conn)
        if self.conn.ndim != 2:
            raise ValueError(f'Only support 2D connection matrix; got {self.conn.ndim}D.')

    @brainstate.nn.call_order(2)
    def init_state(self, *args, **kwargs):
        maybe_init_prefetch(self.x)

    def update(self):
        return additive_coupling(self.x, self.conn, self.k)
