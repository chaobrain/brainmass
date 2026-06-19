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

"""Tests for :mod:`brainmass.objectives`.

Each builder is checked against a direct :mod:`braintools.metric` computation (so
the wrapper adds no drift), plus the known fixed points (self-correlation 1,
self-RMSE 0), the ``as_loss`` conversions, unit-safety, composition, and
gradient-safety.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools
import brainunit as u

from brainmass import objectives


@pytest.fixture
def signals():
    """A (prediction, target) pair of (time, regions) arrays."""
    rng = np.random.default_rng(0)
    pred = jnp.asarray(rng.standard_normal((200, 6)).astype('float32'))
    target = jnp.asarray(rng.standard_normal((200, 6)).astype('float32'))
    return pred, target


# --------------------------------------------------------------------------- #
# timeseries_rmse                                                             #
# --------------------------------------------------------------------------- #

def test_timeseries_rmse_matches_manual(signals):
    pred, target = signals
    got = objectives.timeseries_rmse()(pred, target)
    expected = jnp.sqrt(jnp.mean((pred - target) ** 2))
    assert np.allclose(float(got), float(expected))


def test_timeseries_rmse_zero_on_identity(signals):
    pred, _ = signals
    assert float(objectives.timeseries_rmse()(pred, pred)) == 0.0


def test_timeseries_rmse_known_value():
    x = jnp.zeros((10, 3))
    assert np.isclose(float(objectives.timeseries_rmse()(x + 2.0, x)), 2.0)


def test_timeseries_rmse_unit_safe():
    x = jnp.ones((10, 2))
    # Same units: result is a dimensionless scalar magnitude.
    same = objectives.timeseries_rmse()(x * u.mV, (x + 1.0) * u.mV)
    assert np.isclose(float(same), 1.0)
    # Incompatible units must raise (no silent unit dropping).
    with pytest.raises(Exception):
        objectives.timeseries_rmse()(x * u.mV, x * u.Hz)


# --------------------------------------------------------------------------- #
# fc_corr / fc_rmse                                                           #
# --------------------------------------------------------------------------- #

def test_fc_corr_matches_manual(signals):
    pred, target = signals
    got = objectives.fc_corr()(pred, target)
    fc_p = braintools.metric.functional_connectivity(np.asarray(pred))
    fc_t = braintools.metric.functional_connectivity(np.asarray(target))
    expected = braintools.metric.matrix_correlation(fc_p, fc_t)
    assert np.allclose(float(got), float(expected))


def test_fc_corr_self_is_one(signals):
    pred, _ = signals
    assert np.isclose(float(objectives.fc_corr()(pred, pred)), 1.0, atol=1e-5)


def test_fc_corr_as_loss(signals):
    pred, target = signals
    score = objectives.fc_corr()(pred, target)
    loss = objectives.fc_corr(as_loss=True)(pred, target)
    assert np.allclose(float(loss), 1.0 - float(score))


def test_fc_rmse_matches_manual_and_zero_on_identity(signals):
    pred, target = signals
    got = objectives.fc_rmse()(pred, target)
    fc_p = braintools.metric.functional_connectivity(np.asarray(pred))
    fc_t = braintools.metric.functional_connectivity(np.asarray(target))
    expected = jnp.sqrt(jnp.mean((fc_p - fc_t) ** 2))
    assert np.allclose(float(got), float(expected))
    assert float(objectives.fc_rmse()(pred, pred)) == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# cosine_sim                                                                  #
# --------------------------------------------------------------------------- #

def test_cosine_sim_self_is_one(signals):
    pred, _ = signals
    assert np.isclose(float(objectives.cosine_sim()(pred, pred)), 1.0, atol=1e-5)


def test_cosine_sim_matches_manual(signals):
    pred, target = signals
    got = objectives.cosine_sim()(pred, target)
    expected = braintools.metric.cosine_similarity(
        np.asarray(pred).reshape(-1), np.asarray(target).reshape(-1)
    )
    assert np.allclose(float(got), float(expected))


def test_cosine_sim_as_loss(signals):
    pred, target = signals
    score = objectives.cosine_sim()(pred, target)
    loss = objectives.cosine_sim(as_loss=True)(pred, target)
    assert np.allclose(float(loss), 1.0 - float(score))


# --------------------------------------------------------------------------- #
# fcd                                                                         #
# --------------------------------------------------------------------------- #

def test_fcd_surfaces_matrix_when_no_target(signals):
    pred, _ = signals
    mat = objectives.fcd(window_size=30, step_size=5)(pred)
    expected = braintools.metric.functional_connectivity_dynamics(
        np.asarray(pred), window_size=30, step_size=5
    )
    assert np.asarray(mat).shape == np.asarray(expected).shape
    assert np.asarray(mat).ndim == 2


def test_fcd_self_correlation_is_one(signals):
    pred, _ = signals
    assert np.isclose(float(objectives.fcd()(pred, pred)), 1.0, atol=1e-5)


def test_fcd_as_loss(signals):
    pred, target = signals
    score = objectives.fcd()(pred, target)
    loss = objectives.fcd(as_loss=True)(pred, target)
    assert np.allclose(float(loss), 1.0 - float(score))


# --------------------------------------------------------------------------- #
# combine                                                                     #
# --------------------------------------------------------------------------- #

def test_combine_is_weighted_sum(signals):
    pred, target = signals
    rmse = objectives.timeseries_rmse()
    fcl = objectives.fc_corr(as_loss=True)
    combined = objectives.combine((1.0, rmse), (0.5, fcl))
    got = combined(pred, target)
    expected = 1.0 * rmse(pred, target) + 0.5 * fcl(pred, target)
    assert np.allclose(float(got), float(expected))


def test_combine_zero_on_identity(signals):
    pred, _ = signals
    combined = objectives.combine(
        (1.0, objectives.timeseries_rmse()),
        (1.0, objectives.fc_corr(as_loss=True)),
    )
    assert np.isclose(float(combined(pred, pred)), 0.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# gradient-safety                                                             #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('builder', [
    objectives.timeseries_rmse(),
    objectives.fc_corr(as_loss=True),
    objectives.fc_rmse(),
    objectives.cosine_sim(as_loss=True),
])
def test_objectives_are_grad_safe(builder, signals):
    pred, target = signals

    def loss(scale):
        return builder(scale * pred, target)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))


# --------------------------------------------------------------------------- #
# FCD-distribution primitives: ks_distance / wasserstein_1d / fcd_distribution #
# --------------------------------------------------------------------------- #

from scipy.stats import ks_2samp, wasserstein_distance  # noqa: E402


def test_ks_distance_matches_scipy_exactly():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(300)
    b = rng.standard_normal(250) + 0.4
    # Build per-value "histograms" on the pooled sorted unique grid so that the
    # normalised cumulative sums ARE the empirical CDFs evaluated at the data
    # points -- the exact quantity scipy's ks_2samp maximises.
    grid = np.unique(np.concatenate([a, b]))
    pa = np.array([np.sum(a == g) for g in grid], dtype=float)
    pb = np.array([np.sum(b == g) for g in grid], dtype=float)
    got = float(objectives.ks_distance(jnp.asarray(pa), jnp.asarray(pb)))
    expected = float(ks_2samp(a, b).statistic)
    assert np.isclose(got, expected, atol=1e-6)


def test_ks_distance_in_unit_interval():
    rng = np.random.default_rng(2)
    p = jnp.asarray(np.abs(rng.standard_normal(50)))
    q = jnp.asarray(np.abs(rng.standard_normal(50)))
    d = float(objectives.ks_distance(p, q))
    assert 0.0 <= d <= 1.0


def test_wasserstein_1d_matches_scipy_on_fine_grid():
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, 4000)
    b = rng.normal(0.6, 1.3, 4000)
    grid = np.linspace(-8.0, 8.0, 4000)
    # densities (normalised histograms) of each sample on the shared grid
    pa, _ = np.histogram(a, bins=np.r_[grid, grid[-1] + (grid[1] - grid[0])], density=True)
    pb, _ = np.histogram(b, bins=np.r_[grid, grid[-1] + (grid[1] - grid[0])], density=True)
    got = float(objectives.wasserstein_1d(jnp.asarray(pa), jnp.asarray(pb), jnp.asarray(grid)))
    expected = float(wasserstein_distance(a, b))
    assert np.isclose(got, expected, rtol=2e-2, atol=2e-2)


def test_wasserstein_1d_zero_for_identical_inputs():
    rng = np.random.default_rng(4)
    p = jnp.asarray(np.abs(rng.standard_normal(100)))
    x = jnp.linspace(0.0, 5.0, 100)
    assert float(objectives.wasserstein_1d(p, p, x)) == pytest.approx(0.0, abs=1e-6)


def test_wasserstein_1d_is_grad_safe():
    rng = np.random.default_rng(5)
    p = jnp.asarray(np.abs(rng.standard_normal(100)))
    q = jnp.asarray(np.abs(rng.standard_normal(100)))
    x = jnp.linspace(0.0, 5.0, 100)

    def loss(scale):
        return objectives.wasserstein_1d(scale * p, q, x)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))


def test_fcd_distribution_shape_and_normalisation(signals):
    pred, _ = signals
    fcd_mat = objectives.fcd(window_size=30, step_size=5)(pred)
    midpoints = jnp.linspace(-0.99, 0.99, 100)
    density = objectives.fcd_distribution(fcd_mat, midpoints=midpoints)
    assert np.asarray(density).shape == (100,)
    dx = float(midpoints[1] - midpoints[0])
    integral = float(jnp.sum(density) * dx)
    assert np.isclose(integral, 1.0, atol=1e-3)


def test_fcd_distribution_default_midpoints(signals):
    pred, _ = signals
    fcd_mat = objectives.fcd(window_size=30, step_size=5)(pred)
    density = objectives.fcd_distribution(fcd_mat)
    assert np.asarray(density).shape == (100,)   # default 100-point grid


def test_fcd_distribution_unnormalised_branch(signals):
    pred, _ = signals
    fcd_mat = objectives.fcd(window_size=30, step_size=5)(pred)
    midpoints = jnp.linspace(-0.99, 0.99, 100)
    raw = objectives.fcd_distribution(fcd_mat, midpoints=midpoints, normalize=False)
    norm = objectives.fcd_distribution(fcd_mat, midpoints=midpoints, normalize=True)
    dx = float(midpoints[1] - midpoints[0])
    # The un-normalised density does not integrate to 1; normalising fixes that.
    assert not np.isclose(float(jnp.sum(raw) * dx), 1.0, atol=1e-3)
    assert np.isclose(float(jnp.sum(norm) * dx), 1.0, atol=1e-3)


# --------------------------------------------------------------------------- #
# FCD-distribution objective builders: fcd_ks / fcd_wasserstein               #
# --------------------------------------------------------------------------- #

def test_fcd_wasserstein_zero_on_identity(signals):
    pred, _ = signals
    loss = objectives.fcd_wasserstein(window_size=30, step_size=5)
    assert float(loss(pred, pred)) == pytest.approx(0.0, abs=1e-6)


def test_fcd_ks_zero_on_identity(signals):
    pred, _ = signals
    loss = objectives.fcd_ks(window_size=30, step_size=5)
    assert float(loss(pred, pred)) == pytest.approx(0.0, abs=1e-6)


def test_fcd_wasserstein_positive_for_different_signals(signals):
    pred, target = signals
    loss = objectives.fcd_wasserstein(window_size=30, step_size=5)
    assert float(loss(pred, target)) > 0.0


def test_fcd_ks_positive_for_different_signals(signals):
    pred, target = signals
    loss = objectives.fcd_ks(window_size=30, step_size=5)
    assert float(loss(pred, target)) > 0.0


def test_fcd_wasserstein_is_grad_safe(signals):
    pred, target = signals

    def loss(scale):
        return objectives.fcd_wasserstein(window_size=30, step_size=5)(scale * pred, target)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))


def test_fcd_wasserstein_combines(signals):
    pred, target = signals
    combined = objectives.combine(
        (1.0, objectives.timeseries_rmse()),
        (0.5, objectives.fcd_wasserstein(window_size=30, step_size=5)),
    )
    got = float(combined(pred, target))
    expected = (1.0 * float(objectives.timeseries_rmse()(pred, target))
                + 0.5 * float(objectives.fcd_wasserstein(window_size=30, step_size=5)(pred, target)))
    assert np.isclose(got, expected)


@pytest.mark.parametrize('builder', [objectives.fcd_wasserstein, objectives.fcd_ks])
def test_fcd_builders_accept_explicit_midpoints(builder, signals):
    pred, target = signals
    midpoints = jnp.linspace(-0.95, 0.95, 64)
    loss = builder(window_size=30, step_size=5, midpoints=midpoints)
    val = float(loss(pred, target))
    assert np.isfinite(val) and val > 0.0


def test_fcd_distribution_degenerate_constant_series_is_nan():
    # A constant (zero-variance) series gives degenerate FCD off-diagonal values;
    # the KDE covariance is singular, so the density -- and any distance built on
    # it -- is nan. Documented: FCD-distribution objectives need non-degenerate,
    # non-constant input.
    const = jnp.ones((200, 6))
    loss = objectives.fcd_wasserstein(window_size=30, step_size=5)
    assert not np.isfinite(float(loss(const, const)))
