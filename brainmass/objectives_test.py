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
