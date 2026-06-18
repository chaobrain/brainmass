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

from typing import Union, Sequence, Dict, Tuple, List, Callable

import brainstate
import brainunit as u
import jax.tree
import numpy as np

from .typing import Array

__all__ = [
    'sys2nd',
    'sigmoid',
    'bounded_input',
    'process_sequence',
    'set_module_as',
    'delay_index',
]


def sys2nd(
    A: Array,
    a: Array,
    u: Array,
    x: Array,
    v: Array
) -> Array:
    r"""Compute the acceleration of a second-order linear system.

    Implements the canonical second-order kinetic block used by neural-mass
    models (e.g. Jansen-Rit). The system

    .. math::

        \frac{d^2 x}{dt^2} + 2 a \frac{dx}{dt} + a^2 x = A\,a\,u

    is written in state-space form with ``v = dx/dt`` as

    .. math::

        \frac{dv}{dt} = A\,a\,u - 2 a v - a^2 x .

    Parameters
    ----------
    A : Array
        Amplitude gain parameter.
    a : Array
        Time-constant parameter (units of inverse time).
    u : Array
        Input signal.
    x : Array
        Position state (the integrated output).
    v : Array
        Velocity state (the derivative of ``x``).

    Returns
    -------
    Array
        ``dv/dt`` — the acceleration, i.e. the derivative of the velocity state.
    """
    return A * a * u - 2 * a * v - a ** 2 * x


def sigmoid(
    x: Array,
    vmax: Array,
    v0: Array,
    r: Array
) -> Array:
    r"""Convert membrane potential to firing rate via a sigmoid.

    .. math::

        S(x) = \frac{v_{max}}{1 + \exp\!\big(r (v_0 - x)\big)}

    Parameters
    ----------
    x : Array
        Input membrane potential.
    vmax : Array
        Maximum firing rate.
    v0 : Array
        Firing threshold (potential at half-maximum rate).
    r : Array
        Steepness of the sigmoid.

    Returns
    -------
    Array
        Firing rate in the open interval ``(0, vmax)``.
    """
    # vmax / (1 + u.math.exp(r * (v0 - x)))
    return vmax * u.math.sigmoid(-r * (v0 - x))


def bounded_input(
    x: Array,
    bound: float = 500.0
) -> Array:
    r"""Apply a ``tanh`` bound to an input signal.

    Prevents numerical instability by smoothly limiting the magnitude of the
    inputs fed to a second-order system.

    Parameters
    ----------
    x : Array
        Input signal.
    bound : float, default 500.0
        Maximum absolute value of the output.

    Returns
    -------
    Array
        The bounded input ``bound * tanh(x / bound)``.
    """
    return bound * u.math.tanh(x / bound)


def process_sequence(
    data: brainstate.typing.PyTree,
    mode: Union[str, Callable] = 'stack',
) -> Union[Array, Tuple, List, Dict, Sequence]:
    """Aggregate a sequence of data items along the leading dimension.

    The sequence is first stacked along a new leading axis and then reduced
    according to ``mode``. The reduction is applied recursively over arbitrary
    PyTrees (dicts, tuples, lists, and nested combinations thereof).

    Parameters
    ----------
    data : PyTree
        A stacked PyTree of items, with the items enumerated along the leading
        axis of every leaf. All leaves must share a compatible structure.
    mode : str or Callable, default 'stack'
        How to reduce along the leading axis:

        - ``'stack'`` : return the data unchanged (identity).
        - ``'last'`` / ``'first'`` : take the last/first item.
        - ``'avg'`` / ``'mean'`` : mean over the leading axis.
        - ``'max'`` / ``'min'`` : max/min over the leading axis.
        - callable : apply the callable to each leaf.

    Returns
    -------
    Array or tuple or list or dict or Sequence
        Aggregated data whose structure matches the input leaves:

        - ``mode='stack'`` : structure with the new leading dimension retained.
        - ``mode='last'`` / ``'first'`` : a single item.
        - ``mode='avg'``/``'mean'``/``'max'``/``'min'`` : reduced PyTree.
        - ``mode`` callable : the result of applying the callable.

    Raises
    ------
    ValueError
        If ``mode`` is an unknown string.

    Notes
    -----
    - All leaves must share the same type and structure.
    - Dictionary keys (and tuple/list lengths) must match across elements.
    - The reduction is recursive and handles nested structures, e.g. a dict of
      tuples of arrays.
    """
    msg = (
        f"Unknown mode: {mode}. "
        f"Supported modes: 'stack', 'last', 'first', 'avg', 'mean', 'max', 'min', or callable."
    )

    if callable(mode):
        fn = mode
    elif isinstance(mode, str):
        if mode == 'stack':
            fn = lambda x: x
        elif mode == 'last':
            fn = lambda x: x[-1]
        elif mode == 'first':
            fn = lambda x: x[0]
        elif mode in ('avg', 'mean'):
            fn = lambda x: x.mean(axis=0)
        elif mode == 'max':
            fn = lambda x: x.max(axis=0)
        elif mode == 'min':
            fn = lambda x: x.min(axis=0)
        else:
            raise ValueError(msg)
    else:
        raise ValueError(msg)
    return jax.tree.map(fn, data)


def set_module_as(module: str):
    """Return a decorator that rebinds a function's ``__module__`` attribute.

    Parameters
    ----------
    module : str
        The module path to assign to the decorated function's ``__module__``.

    Returns
    -------
    Callable
        A decorator that sets ``fun.__module__ = module`` and returns ``fun``.
    """

    def wrapper(fun: Callable):
        fun.__module__ = module
        return fun

    return wrapper


def delay_index(n_hidden: int):
    """Build the neuron-index matrix used to address per-connection delays.

    Parameters
    ----------
    n_hidden : int
        Number of hidden units (nodes).

    Returns
    -------
    numpy.ndarray
        An integer array of shape ``(n_hidden, n_hidden)`` whose every row is
        ``arange(n_hidden)``; row ``i`` selects the source-neuron index for
        each delayed connection into target ``i``.
    """
    return np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
