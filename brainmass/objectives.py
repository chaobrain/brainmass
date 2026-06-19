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
import jax
import jax.numpy as jnp

__all__ = [
    'timeseries_rmse',
    'fc_corr',
    'fc_rmse',
    'cosine_sim',
    'fcd',
    'fcd_distribution',
    'ks_distance',
    'wasserstein_1d',
    'fcd_ks',
    'fcd_wasserstein',
    'combine',
]

#: Default evaluation grid for FCD off-diagonal distributions (correlations live
#: in [-1, 1]); 100 points on [-0.99, 0.99] matching the tvboptim convention.
_DEFAULT_FCD_MIDPOINTS = jnp.linspace(-0.99, 0.99, 100)


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


def fcd_distribution(fcd_matrix, midpoints=None, n_diag=1, bw_method=None, normalize=True):
    r"""Kernel-density estimate of the FCD off-diagonal value distribution.

    The standard FCD fitting target is the *distribution* of the upper-triangle
    (off-diagonal) values of the FCD matrix -- not the matrix itself. This
    surfaces that distribution as a smooth density on a fixed grid via
    :func:`jax.scipy.stats.gaussian_kde` (delegated; ``braintools`` provides no
    KDE).

    Parameters
    ----------
    fcd_matrix : array
        Square FCD matrix (e.g. from :func:`fcd` or
        :func:`braintools.metric.functional_connectivity_dynamics`).
    midpoints : array, optional
        Evaluation grid. Default: 100 points on ``[-0.99, 0.99]`` (FCD values are
        correlations).
    n_diag : int, default 1
        Diagonal offset for the upper-triangle extraction (``1`` excludes the
        main diagonal).
    bw_method : optional
        Bandwidth selector forwarded to :func:`jax.scipy.stats.gaussian_kde`
        (default ``None`` = Scott's rule). Smaller values give sharper densities.
    normalize : bool, default True
        Renormalise the evaluated density to integrate to 1 over ``midpoints``
        (KDE on a finite grid never integrates to exactly 1).

    Returns
    -------
    jax.Array
        Density evaluated on ``midpoints``.

    See Also
    --------
    fcd_ks, fcd_wasserstein : objective builders comparing two FCD distributions.
    """
    if midpoints is None:
        midpoints = _DEFAULT_FCD_MIDPOINTS
    vals = fcd_matrix[jnp.triu_indices(fcd_matrix.shape[0], k=n_diag)]
    kde = jax.scipy.stats.gaussian_kde(vals, bw_method=bw_method)
    density = kde.evaluate(midpoints)
    if normalize:
        dx = midpoints[1] - midpoints[0]
        density = density / (jnp.sum(density) * dx)
    return density


def ks_distance(p, q):
    r"""Kolmogorov-Smirnov statistic between two 1-D densities / histograms.

    Each cumulative sum is normalised to a proper CDF before the supremum is
    taken, so the result lies in ``[0, 1]`` independent of bin width and is
    directly comparable to :func:`scipy.stats.ks_2samp`.

    .. math::

        D = \sup_x \left| F_p(x) - F_q(x) \right|.

    Parameters
    ----------
    p, q : array
        Densities or (unnormalised) histograms on a shared, ordered grid.

    Returns
    -------
    jax.Array
        Scalar KS statistic in ``[0, 1]``.

    Notes
    -----
    The supremum (``max``) makes this **non-smooth**: its gradient is the
    indicator at the argmax, so prefer :func:`wasserstein_1d` (and
    :func:`fcd_wasserstein`) when the distance is a *fitting* loss. Use KS for
    evaluation / reporting, where literature comparability matters.

    See Also
    --------
    wasserstein_1d : smooth, gradient-friendly distributional distance.
    """
    p_cdf = jnp.cumsum(p)
    q_cdf = jnp.cumsum(q)
    p_cdf = p_cdf / p_cdf[-1]
    q_cdf = q_cdf / q_cdf[-1]
    return jnp.max(jnp.abs(p_cdf - q_cdf))


def wasserstein_1d(p, q, x):
    r"""Wasserstein-1 distance between two 1-D densities on a shared grid.

    .. math::

        W_1(p, q) = \int \left| F_p(x) - F_q(x) \right| \, dx,

    discretised on the uniform grid ``x``. Both inputs are normalised internally
    (CDFs run 0..1), so ``p`` and ``q`` may be densities or unnormalised
    histograms; the returned value lives in the same units as ``x``. Unlike
    :func:`ks_distance` this is **smooth and differentiable** in both inputs,
    making it the preferred distributional loss for gradient-based fitting.
    Matches :func:`scipy.stats.wasserstein_distance` as the grid is refined.

    Parameters
    ----------
    p, q : array
        Densities or histograms on the shared grid ``x``.
    x : array
        Uniform evaluation grid (same length as ``p`` / ``q``).

    Returns
    -------
    jax.Array
        Scalar Wasserstein-1 distance.

    See Also
    --------
    ks_distance : the non-smooth KS counterpart.
    """
    dx = x[1] - x[0]
    p_cdf = jnp.cumsum(p)
    q_cdf = jnp.cumsum(q)
    p_cdf = p_cdf / p_cdf[-1]
    q_cdf = q_cdf / q_cdf[-1]
    return jnp.sum(jnp.abs(p_cdf - q_cdf)) * dx


def _fcd_distribution_pair(prediction, target, window_size, step_size,
                           midpoints, bw_method, n_diag):
    """Compute the FCD off-diagonal density of prediction and target."""
    fcd_p = braintools.metric.functional_connectivity_dynamics(
        _mag(prediction), window_size=window_size, step_size=step_size
    )
    fcd_t = braintools.metric.functional_connectivity_dynamics(
        _mag(target), window_size=window_size, step_size=step_size
    )
    dp = fcd_distribution(fcd_p, midpoints=midpoints, n_diag=n_diag, bw_method=bw_method)
    dq = fcd_distribution(fcd_t, midpoints=midpoints, n_diag=n_diag, bw_method=bw_method)
    return dp, dq


def fcd_wasserstein(window_size=30, step_size=5, midpoints=None, bw_method=None, n_diag=1):
    r"""Build a Wasserstein FCD-distribution loss (smooth, grad-friendly).

    Compares the *distribution* of FCD off-diagonal values of the prediction and
    target -- the standard FCD fitting target -- rather than the FCD matrix
    correlation (:func:`fcd`). The :func:`wasserstein_1d` distance is smooth, so
    this is the recommended FCD objective for gradient-based fitting.

    Parameters
    ----------
    window_size, step_size : int
        Sliding-window length and stride (in samples) for
        :func:`braintools.metric.functional_connectivity_dynamics`.
    midpoints : array, optional
        FCD-value evaluation grid (default 100 points on ``[-0.99, 0.99]``).
    bw_method : optional
        KDE bandwidth forwarded to :func:`fcd_distribution`.
    n_diag : int, default 1
        Upper-triangle diagonal offset.

    Returns
    -------
    callable
        ``loss(prediction, target) -> scalar`` Wasserstein-1 distance between the
        two FCD distributions (a quantity to *minimise*; ``0`` on identity).

    See Also
    --------
    fcd_ks : the non-smooth KS counterpart.
    fcd : FCD *matrix* correlation objective.

    Examples
    --------
    >>> import numpy as np, jax.numpy as jnp
    >>> from brainmass import objectives
    >>> rng = np.random.default_rng(0)
    >>> x = jnp.asarray(rng.standard_normal((200, 6)))
    >>> loss = objectives.fcd_wasserstein(window_size=30, step_size=5)
    >>> float(loss(x, x))
    0.0
    """
    if midpoints is None:
        midpoints = _DEFAULT_FCD_MIDPOINTS

    def loss(prediction, target):
        dp, dq = _fcd_distribution_pair(
            prediction, target, window_size, step_size, midpoints, bw_method, n_diag
        )
        return wasserstein_1d(dp, dq, midpoints)

    return loss


def fcd_ks(window_size=30, step_size=5, midpoints=None, bw_method=None, n_diag=1):
    r"""Build a Kolmogorov-Smirnov FCD-distribution loss.

    Like :func:`fcd_wasserstein` but using the :func:`ks_distance` statistic
    between the two FCD off-diagonal distributions. The KS statistic is the
    common literature metric for FCD comparison, but it is **non-smooth** (a
    supremum); prefer :func:`fcd_wasserstein` when the loss drives a gradient
    optimiser, and use this for evaluation / reporting.

    Parameters
    ----------
    window_size, step_size : int
        Sliding-window length and stride (in samples) for
        :func:`braintools.metric.functional_connectivity_dynamics`.
    midpoints : array, optional
        FCD-value evaluation grid (default 100 points on ``[-0.99, 0.99]``).
    bw_method : optional
        KDE bandwidth forwarded to :func:`fcd_distribution`.
    n_diag : int, default 1
        Upper-triangle diagonal offset.

    Returns
    -------
    callable
        ``loss(prediction, target) -> scalar`` KS distance between the two FCD
        distributions (a quantity to *minimise*; ``0`` on identity).

    See Also
    --------
    fcd_wasserstein : the smooth, gradient-friendly counterpart.
    """
    if midpoints is None:
        midpoints = _DEFAULT_FCD_MIDPOINTS

    def loss(prediction, target):
        dp, dq = _fcd_distribution_pair(
            prediction, target, window_size, step_size, midpoints, bw_method, n_diag
        )
        return ks_distance(dp, dq)

    return loss


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
