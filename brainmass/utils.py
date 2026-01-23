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
    """
    Second-order system dynamics.

    Implements the derivative of a second-order linear system:
        d²x/dt² + 2a·dx/dt + a²·x = A·a·u

    Which can be written as:
        dv/dt = A·a·u - 2·a·v - a²·x

    where v = dx/dt

    Args:
        A: Amplitude gain parameter.
        a: Time constant parameter (1/time).
        u: Input signal.
        x: Position state (integrated output).
        v: Velocity state (derivative of position).

    Returns:
        dv/dt - the acceleration (derivative of velocity).
    """
    return A * a * u - 2 * a * v - a ** 2 * x


def sigmoid(
    x: Array,
    vmax: Array,
    v0: Array,
    r: Array
) -> Array:
    """
    Sigmoidal firing rate function.

    Converts membrane potential to firing rate using a sigmoid function.

    S(x) = vmax / (1 + exp(r·(v0 - x)))

    Args:
        x: Input membrane potential.
        vmax: Maximum firing rate.
        v0: Firing threshold (potential at half-max rate).
        r: Steepness of the sigmoid.

    Returns:
        Firing rate in range (0, vmax).
    """
    # vmax / (1 + u.math.exp(r * (v0 - x)))
    return vmax * u.math.sigmoid(-r * (v0 - x))


def bounded_input(
    x: Array,
    bound: float = 500.0
) -> Array:
    """
    Apply tanh bounding to input signal.

    Prevents numerical instability by limiting the magnitude of inputs
    to the second-order system.

    Args:
        x: Input signal.
        bound: Maximum absolute value.

    Returns:
        Bounded input: bound * tanh(u / bound)
    """
    return bound * u.math.tanh(x / bound)


def process_sequence(
    data: brainstate.typing.PyTree,
    mode: Union[str, Callable] = 'stack',
) -> Union[Array, Tuple, List, Dict, Sequence]:
    """
    Stack a sequence of data items along a new dimension.

    This is the inverse operation of slice_data - while slice_data reduces
    a dimension via aggregation, stack_data creates a new dimension by
    stacking multiple items together.


    Returns:
        Aggregated data with structure matching input elements:
        - mode='stack': Array/dict/tuple/list with new dimension at `dim`
        - mode='last'/'first': Single item (same as data[-1] or data[0])
        - mode='avg'/'mean'/'max'/'min': Aggregated tensor/dict/tuple/list
        - mode=callable: Result of applying callable to stacked data
        For dicts/tuples/lists, aggregation is applied recursively to each element.

    Raises:
        ValueError: If data is empty (cannot infer structure/type) or if
            mode is an unknown string.
        TypeError: If sequence contains mixed or incompatible types, or if
            mode is not a string or callable.


    Notes:
        - All elements in data must have the same type and structure
        - NumPy arrays are automatically converted to float32 tensors
        - Dictionary keys must match across all elements
        - Tuple/list lengths must match across all elements
        - Recursive: handles nested structures (e.g., dict of tuples)
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
    def wrapper(fun: Callable):
        fun.__module__ = module
        return fun

    return wrapper


def delay_index(n_hidden: int):
    return np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
