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

"""Provenance script for the bundled example datasets.

This script regenerates the small, license-clean, fully **synthetic** ``.npz``
files that ``brainmass.datasets`` ships under ``brainmass/_data/``. They are
deterministic (fixed seed), so re-running this script reproduces the committed
bytes exactly. There is **no third-party data** in any bundled file.

Run from the repository root::

    python -m brainmass._data._generate

Bundled files
-------------
``example_connectome.npz``
    A small (N=8) structural-connectivity bundle: a symmetric, zero-diagonal
    weight matrix ``weights`` (in [0, 1)), a symmetric zero-diagonal
    Euclidean ``distances`` matrix in millimetres, and ``labels`` (N region
    names). Generated from 8 points on a ring in a 2-D plane so the distance
    matrix is a genuine metric.

``example_signal.npz``
    A short multi-region target time series ``signal`` shaped ``(time, region)``
    for the fitting tutorials, its ``dt`` (milliseconds), the region ``labels``,
    and the precomputed functional-connectivity matrix ``fc``.
"""

import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Region count for the bundled example connectome / signal.
_N_REGIONS = 8

#: Region labels shared by the bundled connectome and signal.
_LABELS = np.array(
    ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'],
    dtype='<U8',
)


def _make_connectome():
    """Build the synthetic ``(weights, distances, labels)`` arrays.

    Returns
    -------
    weights : numpy.ndarray
        ``(N, N)`` symmetric, zero-diagonal weights in ``[0, 1)``.
    distances : numpy.ndarray
        ``(N, N)`` symmetric, zero-diagonal Euclidean distances in millimetres.
    labels : numpy.ndarray
        ``(N,)`` region labels.
    """
    rng = np.random.default_rng(0)
    n = _N_REGIONS

    # Place the N regions on a ring in a 2-D plane (radius 40 mm); the pairwise
    # Euclidean distances are then a genuine metric (symmetric, zero diagonal).
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    coords = 40.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(distances, 0.0)

    # Random symmetric weights in [0, 1), zero diagonal (no self-connections),
    # biased so that nearer regions are more strongly connected.
    raw = rng.random((n, n))
    raw = 0.5 * (raw + raw.T)  # symmetrise
    proximity = 1.0 / (1.0 + distances / 40.0)
    weights = raw * proximity
    np.fill_diagonal(weights, 0.0)
    # Normalise into [0, 1).
    weights = weights / (weights.max() * 1.0001)
    weights = 0.5 * (weights + weights.T)  # re-symmetrise after the scalar divide
    np.fill_diagonal(weights, 0.0)

    return weights.astype(np.float64), distances.astype(np.float64), _LABELS


def _make_signal(weights):
    """Build the synthetic example signal ``(signal, dt, labels, fc)``.

    A short multi-region time series produced by a simple linear network of
    damped oscillators coupled through ``weights`` and driven by small noise.
    Deterministic (fixed seed).

    Parameters
    ----------
    weights : numpy.ndarray
        The connectome weights used to couple the regions.

    Returns
    -------
    signal : numpy.ndarray
        ``(time, region)`` time series.
    dt_ms : float
        Sampling step in milliseconds.
    labels : numpy.ndarray
        Region labels.
    fc : numpy.ndarray
        ``(region, region)`` functional-connectivity (Pearson correlation) matrix.
    """
    rng = np.random.default_rng(1)
    n = _N_REGIONS
    n_time = 500
    dt_ms = 1.0

    # Region-specific angular frequencies (cycles over the window).
    freqs = np.linspace(0.02, 0.06, n)
    t = np.arange(n_time)

    # Base oscillations with distinct phases.
    phases = np.linspace(0.0, np.pi, n)
    base = np.sin(2.0 * np.pi * freqs[None, :] * t[:, None] + phases[None, :])

    # Mix regions through the (row-normalised) connectome so correlated structure
    # mirrors the connectome, then add small Gaussian observation noise.
    row_sum = weights.sum(1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    mixing = np.eye(n) + 0.5 * (weights / row_sum)
    signal = base @ mixing.T
    signal = signal + 0.05 * rng.standard_normal((n_time, n))

    # Functional connectivity = Pearson correlation across regions.
    fc = np.corrcoef(signal.T)

    return signal.astype(np.float64), float(dt_ms), _LABELS, fc.astype(np.float64)


def main():
    """Regenerate every bundled ``.npz`` file in place."""
    weights, distances, labels = _make_connectome()
    np.savez(
        os.path.join(_HERE, 'example_connectome.npz'),
        weights=weights,
        distances=distances,
        labels=labels,
    )

    signal, dt_ms, sig_labels, fc = _make_signal(weights)
    np.savez(
        os.path.join(_HERE, 'example_signal.npz'),
        signal=signal,
        dt=np.asarray(dt_ms, dtype=np.float64),
        labels=sig_labels,
        fc=fc,
    )
    print('Wrote example_connectome.npz and example_signal.npz to', _HERE)


if __name__ == '__main__':
    main()
