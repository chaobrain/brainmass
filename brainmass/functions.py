"""
Core dynamics functions for neural mass models.

Provides fundamental mathematical functions used in Jansen-Rit
and related neural mass models.
"""

from typing import Union, Sequence, Dict, Tuple, List, Callable

import brainunit as u
import numpy as np

from .typing import Array

__all__ = [
    'sys2nd',
    'sigmoid',
    'bounded_input',
    'euler_step',
    'slice_data',
    'process_sequence',
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
    return vmax / (1 + u.math.exp(r * (v0 - x)))


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


def euler_step(
    x: Array,
    dx: Array,
    dt: float
) -> Array:
    """
    Euler integration step.

    Args:
        x: Current state value.
        dx: Derivative of state.
        dt: Time step.

    Returns:
        Updated state: x + dt * dx
    """
    return x + dt * dx


def slice_data(
    data: Union[Array, np.ndarray, Sequence, Dict],
    mode: Union[str, Callable] = 'last'
) -> Union[Array, Dict, Sequence]:
    """
    Slice data along first dimension using specified aggregation mode.

    Reduces the first dimension (typically time/steps) of the input data
    using one of several aggregation modes. Supports tensors, numpy arrays,
    sequences, and dictionaries with recursive processing.

    Args:
        data: Input data to slice. Supported types:
            - Array: Sliced along dim 0
            - np.ndarray: Converted to tensor, then sliced
            - dict: Recursively slice each value
            - tuple/list: Slice each element, preserve type
            - Primitives (None, int, float, str, bool): Passed through unchanged
        mode: Slicing mode. Supported modes:
            - 'last': Return last element (data[-1])
            - 'first': Return first element (data[0])
            - 'mean': Average along dim 0 (data.mean(dim=0))
            - 'max': Maximum along dim 0 (data.max(dim=0).values)
            - Callable: The callable function.

    Returns:
        Sliced data with same structure but first dimension reduced.
        For tensors: returns tensor with dim 0 removed.
        For dicts: returns dict with same keys, sliced values.
        For sequences: returns same type (tuple/list) with sliced elements.
        For primitives: returns unchanged.

    Raises:
        ValueError: If mode is not one of the supported modes.
        TypeError: If data type is not supported.

    Examples:
        >>> # Array slicing
        >>> tensor = randn(10, 5, 3)  # (time, batch, features)
        >>> slice_data(tensor, mode='last').shape
        Size([5, 3])
        >>> slice_data(tensor, mode='mean').shape
        Size([5, 3])

        >>> # Dictionary slicing
        >>> data = {'x': randn(10, 5), 'y': randn(10, 5)}
        >>> result = slice_data(data, mode='first')
        >>> result['x'].shape
        Size([5])

        >>> # Sequence slicing
        >>> seq = [randn(10, 3), randn(10, 3)]
        >>> result = slice_data(seq, mode='max')
        >>> len(result)
        2

    Notes:
        - For tensors, operates on the first dimension (dim=0)
        - Scalar tensors (dim=0) are returned unchanged
        - Dictionary and sequence processing is recursive
        - NumPy arrays are converted to float32 tensors
    """
    # Step 1
    if isinstance(data, Array):
        data = u.math.asarray(data)

    # Step 2: Handle torch tensors (main logic)
    if isinstance(data, Array):
        # Handle scalar tensors (no dimension to slice)
        if u.math.ndim(data) == 0:
            return data

        # Apply mode-based slicing on dim 0
        if callable(mode):
            return mode(data)
        elif isinstance(mode, str):
            if mode == 'last':
                return data[-1]
            elif mode == 'first':
                return data[0]
            elif mode == 'mean':
                return data.mean(axis=0)
            elif mode == 'max':
                return data.max(axis=0)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            raise TypeError(
                f"Unsupported data type for slice_data: {type(data).__name__}. "
            )

    # Step 3: Handle dictionaries (recursive)
    if isinstance(data, dict):
        return {
            key: slice_data(value, mode=mode)
            for key, value in data.items()
        }

    # Step 4: Handle sequences (tuple/list) - element-wise
    if isinstance(data, (tuple, list)):
        sliced_items = [slice_data(item, mode=mode) for item in data]
        # Preserve type: return tuple if input was tuple, list if input was list
        return type(data)(sliced_items)

    # Step 5: Handle primitives (pass through)
    if data is None or isinstance(data, (int, float, str, bool)):
        return data

    # Step 6: Unsupported type
    raise TypeError(
        f"Unsupported data type for slice_data: {type(data).__name__}. "
        f"Expected Array, np.ndarray, dict, tuple, list, or primitive type."
    )


def process_sequence(
    data: Sequence[Union[Array, np.ndarray, Sequence, Dict]],
    mode: Union[str, Callable] = 'stack',
) -> Union[Array, Tuple, List, Dict, Sequence]:
    """
    Stack a sequence of data items along a new dimension.

    This is the inverse operation of slice_data - while slice_data reduces
    a dimension via aggregation, stack_data creates a new dimension by
    stacking multiple items together.

    Args:
        data: Sequence (list/tuple) of data items to stack. All items must
            have compatible structure. Supported element types:
            - Array: Stacked using u.math.concatenate
            - np.ndarray: Converted to tensor, then stacked
            - tuple: Stack each element at same index (recursive)
            - list: Stack each element at same index (recursive)
            - dict: Stack values for each key (recursive)
            - Primitives (None, int, float, str, bool): Returned as-is
        mode: Aggregation mode. Supported modes:
            - 'stack': Stack all items along dim (default, creates new dimension)
            - 'last': Return last item in sequence (data[-1])
            - 'first': Return first item in sequence (data[0])
            - 'avg' or 'mean': Average all items (element-wise mean)
            - 'max': Element-wise maximum across all items
            - 'min': Element-wise minimum across all items
            - Callable: Apply custom function to stacked data (always stacked on dim 0)

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

    Examples:
        >>> # Stack tensors
        >>> tensors = [randn(5, 3), randn(5, 3), randn(5, 3)]
        >>> process_sequence(tensors).shape
        Size([3, 5, 3])  # New dimension of size 3 at position 0

        >>> # Stack dictionaries (common in neural mass models)
        >>> dicts = [
        ...     {'P': randn(10), 'E': randn(10)},
        ...     {'P': randn(10), 'E': randn(10)}
        ... ]
        >>> result = process_sequence(dicts)
        >>> result['P'].shape
        Size([2, 10])  # 2 time steps, 10 nodes

        >>> # Stack tuples (multiple outputs per step)
        >>> tuples = [
        ...     (randn(5), randn(3)),
        ...     (randn(5), randn(3))
        ... ]
        >>> result = process_sequence(tuples)
        >>> len(result), result[0].shape
        (2, Size([2, 5]))

        >>> # Custom dimension
        >>> tensors = [randn(3, 4), randn(3, 4)]
        >>> process_sequence(tensors, dim=1).shape
        Size([3, 2, 4])  # Stacked along dim 1

        >>> # Aggregation modes
        >>> tensors = [randn(5, 3), randn(5, 3), randn(5, 3)]
        >>> process_sequence(tensors, mode='last').shape
        Size([5, 3])  # Returns last tensor
        >>> process_sequence(tensors, mode='mean').shape
        Size([5, 3])  # Averaged across 3 tensors

        >>> # Custom aggregation with callable
        >>> process_sequence(tensors, mode=lambda x: x.median(dim=0).values).shape
        Size([5, 3])  # Median across 3 tensors

    Notes:
        - All elements in data must have the same type and structure
        - NumPy arrays are automatically converted to float32 tensors
        - Dictionary keys must match across all elements
        - Tuple/list lengths must match across all elements
        - Recursive: handles nested structures (e.g., dict of tuples)
    """
    # Step 1: Validate non-empty sequence
    if not data or len(data) == 0:
        raise ValueError(
            "Cannot stack empty sequence. "
            "Provide at least one element to infer structure and type."
        )

    # Step 3: Type dispatch based on first element
    first = data[0]

    # Step 3a: Handle torch tensors
    if isinstance(first, Array):
        # Apply mode-based aggregation
        if callable(mode):
            stacked = u.math.concatenate(data, axis=0)
            return mode(stacked)
        elif isinstance(mode, str):
            if mode == 'stack':
                return u.math.concatenate(data, axis=0)
            elif mode == 'last':
                return data[-1]
            elif mode == 'first':
                return data[0]
            elif mode in ('avg', 'mean'):
                stacked = u.math.concatenate(data, axis=0)
                return stacked.mean(axis=0)
            elif mode == 'max':
                stacked = u.math.concatenate(data, axis=0)
                return stacked.max(axis=0)
            elif mode == 'min':
                stacked = u.math.concatenate(data, axis=0)
                return stacked.min(axis=0)
            else:
                raise ValueError(
                    f"Unknown mode: {mode}. "
                    f"Supported modes: 'stack', 'last', 'first', 'avg', 'mean', 'max', 'min', or callable."
                )
        else:
            raise TypeError(f"mode must be str or callable, got {type(mode).__name__}")

    # Step 3b: Handle tuples (element-wise stacking)
    elif isinstance(first, tuple):
        # Validate all are tuples with same length
        if not all(isinstance(item, tuple) and len(item) == len(first) for item in data):
            raise TypeError(
                "Mixed types in sequence or tuples with inconsistent lengths. "
                "All elements must be tuples with same structure."
            )
        # Stack each position recursively
        return tuple(
            process_sequence([item[i] for item in data], mode=mode)
            for i in range(len(first))
        )

    # Step 3c: Handle dictionaries (key-wise stacking)
    elif isinstance(first, dict):
        # Validate all are dicts with same keys
        first_keys = set(first.keys())
        if not all(isinstance(item, dict) and set(item.keys()) == first_keys for item in data):
            raise TypeError(
                "Mixed types in sequence or dicts with inconsistent keys. "
                "All elements must be dicts with same keys."
            )
        # Stack each value recursively
        return {
            key: process_sequence([item[key] for item in data], mode=mode)
            for key in first.keys()
        }

    # Step 3d: Handle lists (element-wise stacking)
    elif isinstance(first, list):
        # Validate all are lists with same length
        if not all(isinstance(item, list) and len(item) == len(first) for item in data):
            raise TypeError(
                "Mixed types in sequence or lists with inconsistent lengths. "
                "All elements must be lists with same structure."
            )
        # Stack element-wise and return as list
        return [
            process_sequence([item[i] for item in data], mode=mode)
            for i in range(len(first))
        ]

    # Step 3e: Handle primitives (pass through)
    elif first is None or isinstance(first, (int, float, str, bool)):
        # Primitives don't stack - return original sequence
        return data

    # Step 3f: Unsupported type
    else:
        raise TypeError(
            f"Cannot stack sequence with elements of type {type(first).__name__}. "
            f"Supported types: Array, np.ndarray, tuple, list, dict, or primitive types."
        )
