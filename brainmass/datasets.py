# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""An extensible registry of small, bundled example datasets.

The registry lets the tutorials and the gallery run with **no external
download**: every entry is either a tiny, license-clean, deterministic file
bundled under ``brainmass/_data/`` or a synthetic generator. Registering a new
dataset is a single :func:`register_dataset` call, so the showcase tracks can
add their own data without changing this module.

Built-in entries
----------------
``example_connectome``
    A small (N=8) structural connectivity bundle -- a :class:`Connectome` of
    symmetric, zero-diagonal ``weights`` and unit-aware Euclidean ``distances``
    (in :data:`brainunit.mm`). Synthetic; see ``brainmass/_data/_generate.py``.
``example_signal``
    A short multi-region target time series for the fitting tutorials, as a
    :class:`Signal` carrying the sampling ``dt`` (in :data:`brainunit.ms`) and a
    precomputed functional-connectivity matrix.
``delayed_match_task``
    A purely synthetic delayed-match-to-sample task generator (no bundled binary)
    yielding ``(inputs, targets)`` for the HORN training tutorial.

See Also
--------
brainmass.viz : thin plotting helpers for the data these loaders return.
brainmass.list_models : a typed catalogue of the bundled neural-mass models.
"""

import os
from typing import Callable, List, NamedTuple, Tuple

import brainunit as u
import numpy as np

__all__ = [
    'register_dataset',
    'list_datasets',
    'load_dataset',
    'Connectome',
    'Signal',
    'delayed_match_task',
]

#: Directory holding the bundled ``.npz`` data files.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_data')

#: The dataset registry: ``name -> (loader, description)``.
_REGISTRY: dict = {}


class Connectome(NamedTuple):
    """A small structural-connectivity bundle.

    Attributes
    ----------
    weights : numpy.ndarray
        ``(N, N)`` structural-connectivity weights -- symmetric, zero diagonal,
        values in ``[0, 1]``.
    distances : brainunit.Quantity
        ``(N, N)`` inter-region Euclidean distances, unit-aware (millimetres),
        symmetric with a zero diagonal.
    labels : numpy.ndarray
        ``(N,)`` region labels.
    """

    weights: np.ndarray
    distances: u.Quantity
    labels: np.ndarray


class Signal(NamedTuple):
    """A short multi-region target time series and its functional connectivity.

    Attributes
    ----------
    signal : numpy.ndarray
        ``(time, region)`` time series.
    dt : brainunit.Quantity
        The sampling step, unit-aware (milliseconds).
    labels : numpy.ndarray
        ``(region,)`` region labels.
    fc : numpy.ndarray
        ``(region, region)`` functional-connectivity (Pearson correlation) matrix.
    """

    signal: np.ndarray
    dt: u.Quantity
    labels: np.ndarray
    fc: np.ndarray


def register_dataset(
    name: str,
    loader: Callable[[], object],
    *,
    description: str = None,
) -> None:
    """Register a named dataset loader.

    Parameters
    ----------
    name : str
        The unique name under which the dataset is looked up by
        :func:`load_dataset`.
    loader : callable
        A zero-argument callable returning the dataset when called. It is invoked
        lazily, on each :func:`load_dataset` call.
    description : str, optional
        A one-line human-readable description shown by :func:`list_datasets`.
        Defaults to an empty string.

    Raises
    ------
    TypeError
        If ``loader`` is not callable.
    ValueError
        If a dataset with the same ``name`` is already registered.

    See Also
    --------
    load_dataset : Look up and call a registered loader.
    list_datasets : List the registered datasets.

    Examples
    --------
    .. code-block:: python

        >>> from brainmass import datasets
        >>> datasets.register_dataset('ones', lambda: [1, 1, 1], description='a list')
        >>> datasets.load_dataset('ones')
        [1, 1, 1]
        >>> datasets._REGISTRY.pop('ones')  # doctest: +ELLIPSIS
        (...)
    """
    if not callable(loader):
        raise TypeError(
            f"loader for dataset '{name}' must be callable, got {type(loader).__name__}."
        )
    if name in _REGISTRY:
        raise ValueError(
            f"dataset '{name}' is already registered; choose a different name."
        )
    _REGISTRY[name] = (loader, '' if description is None else str(description))


def list_datasets() -> List[Tuple[str, str]]:
    """List the registered datasets.

    Returns
    -------
    list of (str, str)
        ``(name, description)`` pairs, sorted by name.

    See Also
    --------
    load_dataset : Load one of the listed datasets.

    Examples
    --------
    .. code-block:: python

        >>> from brainmass import datasets
        >>> names = [name for name, _ in datasets.list_datasets()]
        >>> 'example_connectome' in names
        True
    """
    return [(name, desc) for name, (_, desc) in sorted(_REGISTRY.items())]


def load_dataset(name: str):
    """Load a registered dataset by name.

    Parameters
    ----------
    name : str
        The name the dataset was registered under.

    Returns
    -------
    object
        Whatever the registered loader returns (e.g. a :class:`Connectome`,
        a :class:`Signal`, or an ``(inputs, targets)`` tuple).

    Raises
    ------
    KeyError
        If ``name`` is not registered. The message lists the available names.

    See Also
    --------
    list_datasets : Discover the available dataset names.

    Examples
    --------
    .. code-block:: python

        >>> from brainmass import datasets
        >>> conn = datasets.load_dataset('example_connectome')
        >>> conn.weights.shape
        (8, 8)
    """
    if name not in _REGISTRY:
        available = ', '.join(sorted(_REGISTRY)) or '<none>'
        raise KeyError(
            f"unknown dataset '{name}'. Available datasets: {available}."
        )
    loader, _ = _REGISTRY[name]
    return loader()


# ---------------------------------------------------------------------------
# Built-in loaders (bundled data + synthetic generators)
# ---------------------------------------------------------------------------

def _load_example_connectome() -> Connectome:
    """Load the bundled ``example_connectome.npz`` as a :class:`Connectome`."""
    with np.load(os.path.join(_DATA_DIR, 'example_connectome.npz'), allow_pickle=False) as data:
        weights = np.array(data['weights'])
        distances = np.array(data['distances'])
        labels = np.array(data['labels'])
    return Connectome(weights=weights, distances=distances * u.mm, labels=labels)


def _load_example_signal() -> Signal:
    """Load the bundled ``example_signal.npz`` as a :class:`Signal`."""
    with np.load(os.path.join(_DATA_DIR, 'example_signal.npz'), allow_pickle=False) as data:
        signal = np.array(data['signal'])
        dt = float(data['dt'])
        labels = np.array(data['labels'])
        fc = np.array(data['fc'])
    return Signal(signal=signal, dt=dt * u.ms, labels=labels, fc=fc)


def delayed_match_task(
    n_samples: int = 128,
    *,
    seq_len: int = 10,
    n_symbols: int = 4,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic delayed-match-to-sample task.

    Each trial presents a *cue* symbol, a delay, and a *probe* symbol, all
    one-hot encoded over ``n_symbols`` along the last axis. The binary target is
    ``1`` if the probe matches the cue and ``0`` otherwise. The data is fully
    synthetic (no bundled binary, no MNIST) and deterministic given ``seed``.

    Parameters
    ----------
    n_samples : int, optional
        Number of trials. Default ``128``.
    seq_len : int, optional
        Length of each input sequence (must be at least 2). The cue is at
        ``t=0`` and the probe at ``t=seq_len-1``; the steps in between are the
        (empty) delay. Default ``10``.
    n_symbols : int, optional
        The one-hot symbol alphabet size (last-axis depth). Default ``4``.
    seed : int, optional
        Seed for the deterministic generator. Default ``0``.

    Returns
    -------
    inputs : numpy.ndarray
        ``(n_samples, seq_len, n_symbols)`` float one-hot input sequences.
    targets : numpy.ndarray
        ``(n_samples,)`` int match labels in ``{0, 1}``.

    See Also
    --------
    load_dataset : ``load_dataset('delayed_match_task')`` returns the default task.

    Examples
    --------
    .. code-block:: python

        >>> from brainmass import datasets
        >>> inputs, targets = datasets.delayed_match_task(n_samples=8, seq_len=6)
        >>> inputs.shape
        (8, 6, 4)
        >>> sorted(set(targets.tolist())) == [0, 1] or set(targets.tolist()) <= {0, 1}
        True
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}.")
    if n_symbols < 2:
        raise ValueError(f"n_symbols must be >= 2, got {n_symbols}.")

    rng = np.random.default_rng(seed)
    cues = rng.integers(0, n_symbols, size=n_samples)
    # Force a balanced ~50/50 match/non-match split deterministically.
    match = np.zeros(n_samples, dtype=np.int64)
    match[: n_samples // 2] = 1
    match = rng.permutation(match)

    probes = cues.copy()
    nonmatch = match == 0
    # Offset non-matching probes by a non-zero amount (mod n_symbols).
    offsets = rng.integers(1, n_symbols, size=n_samples)
    probes[nonmatch] = (cues[nonmatch] + offsets[nonmatch]) % n_symbols

    inputs = np.zeros((n_samples, seq_len, n_symbols), dtype=np.float64)
    inputs[np.arange(n_samples), 0, cues] = 1.0
    inputs[np.arange(n_samples), seq_len - 1, probes] = 1.0
    return inputs, match.astype(np.int64)


def _load_delayed_match_task() -> Tuple[np.ndarray, np.ndarray]:
    """Loader wrapper returning the default :func:`delayed_match_task`."""
    return delayed_match_task()


# Register the built-in datasets.
register_dataset(
    'example_connectome',
    _load_example_connectome,
    description='Small (N=8) synthetic structural connectome (weights + mm distances).',
)
register_dataset(
    'example_signal',
    _load_example_signal,
    description='Short multi-region target time series (+ FC) for fitting tutorials.',
)
register_dataset(
    'delayed_match_task',
    _load_delayed_match_task,
    description='Synthetic delayed-match-to-sample task (inputs, targets) for HORN training.',
)
