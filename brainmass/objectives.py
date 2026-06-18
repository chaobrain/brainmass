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

"""Composable objective (loss / score) builders over simulated trajectories.

Each function here is a *builder*: it takes configuration and returns a small
``callable(prediction, target)`` that is jit / grad / vmap-safe and unit-aware.
The callables are thin wrappers over :mod:`braintools.metric` -- they reimplement
no metric maths -- and are meant to be composed (see :func:`combine`) into the
loss a fitter (goal-08) minimises.

A *prediction* / *target* is a region time-series array shaped ``(time, regions)``
(the natural output of :meth:`brainmass.Simulator.run`). Functional-connectivity
objectives are scale-invariant and operate on magnitudes; the time-series RMSE is
unit-checked (subtracting incompatible units raises).
"""

import braintools
import brainunit as u
import jax.numpy as jnp

__all__ = [
    'timeseries_rmse',
    'fc_corr',
    'fc_rmse',
    'cosine_sim',
    'fcd',
    'combine',
]


def _mag(x):
    """Return the bare magnitude of ``x`` (a no-op for unitless arrays)."""
    return u.get_magnitude(x)


def timeseries_rmse():
    r"""Build a root-mean-square-error loss between two time series.

    Returns
    -------
    callable
        ``loss(prediction, target) -> scalar`` computing
        ``sqrt(mean((prediction - target) ** 2))``. The subtraction is
        unit-checked: incompatible units raise before the magnitude is taken.

    See Also
    --------
    fc_rmse : RMSE between functional-connectivity matrices.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainmass import objectives
    >>> loss = objectives.timeseries_rmse()
    >>> x = jnp.zeros((10, 3))
    >>> float(loss(x + 2.0, x))
    2.0
    """

    def loss(prediction, target):
        diff = _mag(prediction - target)
        return jnp.sqrt(jnp.mean(diff ** 2))

    return loss


def fc_corr(as_loss=False):
    r"""Build a functional-connectivity correlation score.

    Computes the correlation between the static functional-connectivity (FC)
    matrices of the prediction and the target via
    :func:`braintools.metric.functional_connectivity` and
    :func:`braintools.metric.matrix_correlation`.

    Parameters
    ----------
    as_loss : bool, default False
        If ``True``, return ``1 - corr`` (a quantity to *minimise*); otherwise
        return the raw correlation (in ``[-1, 1]``, to *maximise*).

    Returns
    -------
    callable
        ``score(prediction, target) -> scalar``.

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> from brainmass import objectives
    >>> rng = np.random.default_rng(0)
    >>> x = jnp.asarray(rng.standard_normal((200, 5)))
    >>> score = objectives.fc_corr()
    >>> float(score(x, x))
    1.0
    """

    def score(prediction, target):
        fc_p = braintools.metric.functional_connectivity(_mag(prediction))
        fc_t = braintools.metric.functional_connectivity(_mag(target))
        corr = braintools.metric.matrix_correlation(fc_p, fc_t)
        return 1.0 - corr if as_loss else corr

    return score


def fc_rmse():
    r"""Build a functional-connectivity RMSE loss.

    Returns
    -------
    callable
        ``loss(prediction, target) -> scalar`` computing the RMSE between the
        prediction's and target's static FC matrices.

    See Also
    --------
    fc_corr : correlation between FC matrices.
    """

    def loss(prediction, target):
        fc_p = braintools.metric.functional_connectivity(_mag(prediction))
        fc_t = braintools.metric.functional_connectivity(_mag(target))
        diff = fc_p - fc_t
        return jnp.sqrt(jnp.mean(diff ** 2))

    return loss


def cosine_sim(as_loss=False, epsilon=0.0):
    r"""Build a cosine-similarity score between two (flattened) time series.

    Thin wrapper over :func:`braintools.metric.cosine_similarity`; the inputs are
    flattened to a single vector so the result is a scalar.

    Parameters
    ----------
    as_loss : bool, default False
        If ``True``, return ``1 - cos`` (to *minimise*); otherwise the raw cosine
        similarity (to *maximise*).
    epsilon : float, default 0.0
        Numerical floor forwarded to :func:`braintools.metric.cosine_similarity`.

    Returns
    -------
    callable
        ``score(prediction, target) -> scalar``.
    """

    def score(prediction, target):
        cos = braintools.metric.cosine_similarity(
            _mag(prediction).reshape(-1),
            _mag(target).reshape(-1),
            epsilon=epsilon,
        )
        return 1.0 - cos if as_loss else cos

    return score


def fcd(window_size=30, step_size=5, as_loss=False):
    r"""Build a functional-connectivity-dynamics (FCD) objective.

    Surfaces :func:`braintools.metric.functional_connectivity_dynamics`, which had
    no call sites in the package. The FCD matrix captures how the sliding-window
    functional connectivity itself evolves over time.

    Parameters
    ----------
    window_size, step_size : int
        Sliding-window length and stride (in samples) forwarded to
        :func:`braintools.metric.functional_connectivity_dynamics`.
    as_loss : bool, default False
        If ``True``, return ``1 - corr`` (to *minimise*); otherwise the raw FCD
        matrix correlation (to *maximise*).

    Returns
    -------
    callable
        ``fn(prediction, target=None)``. With ``target=None`` it returns the
        prediction's FCD **matrix** (surfacing the metric); with a target it
        returns the correlation between the two FCD matrices.
    """

    def fn(prediction, target=None):
        fcd_p = braintools.metric.functional_connectivity_dynamics(
            _mag(prediction), window_size=window_size, step_size=step_size
        )
        if target is None:
            return fcd_p
        fcd_t = braintools.metric.functional_connectivity_dynamics(
            _mag(target), window_size=window_size, step_size=step_size
        )
        corr = braintools.metric.matrix_correlation(fcd_p, fcd_t)
        return 1.0 - corr if as_loss else corr

    return fn


def combine(*weighted_objectives):
    r"""Combine weighted objective callables into a single objective.

    Parameters
    ----------
    *weighted_objectives : tuple of (float, callable)
        Pairs of ``(weight, objective)`` where each ``objective`` is a callable
        returned by the builders in this module (signature
        ``objective(prediction, target)``).

    Returns
    -------
    callable
        ``loss(prediction, target=None) -> scalar`` equal to
        ``sum(weight * objective(prediction, target))``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from brainmass import objectives
    >>> loss = objectives.combine(
    ...     (2.0, objectives.timeseries_rmse()),
    ...     (0.5, objectives.timeseries_rmse()),
    ... )
    >>> x = jnp.zeros((10, 3))
    >>> float(loss(x + 1.0, x))   # (2.0 + 0.5) * 1.0
    2.5
    """

    def loss(prediction, target=None):
        total = 0.0
        for weight, objective in weighted_objectives:
            total = total + weight * objective(prediction, target)
        return total

    return loss
