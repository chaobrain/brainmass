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

"""Tests for :class:`brainmass.KuramotoNetwork`.

Covers construction, the connectivity variants and their validation, the noise
path, and -- as a meaningful dynamical oracle -- the Kuramoto order parameter
:math:`R = |\\langle e^{i\\theta} \\rangle|`, which must approach 1 (full phase
synchrony) under strong coupling and stay flat with no coupling.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import brainstate
import brainunit as u
from brainstate.nn import Param

import brainmass


def _order_parameter(theta):
    """Kuramoto order parameter ``R = |mean(exp(i*theta))|`` in ``[0, 1]``."""
    theta = np.asarray(u.get_magnitude(theta))
    return float(np.abs(np.mean(np.exp(1j * theta))))


def _run(model, n_steps, theta_inp=None):
    """Run ``model`` for ``n_steps`` and return the stacked phase trajectory."""
    def step(i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            return model.update(theta_inp)

    return brainstate.transform.for_loop(step, np.arange(n_steps))


def test_construct_and_init_state(dt):
    """Default construction allocates a phase state of the requested shape."""
    model = brainmass.KuramotoNetwork(6)
    brainstate.nn.init_all_states(model)
    assert model.theta.value.shape == (6,)


def test_update_returns_phase(dt, seeded):
    """A single update returns the new phase with the state shape."""
    model = brainmass.KuramotoNetwork(5, omega=0.5, K=1.0)
    brainstate.nn.init_all_states(model)
    with brainstate.environ.context(i=0, t=0.0 * u.ms):
        out = model.update()
    assert out.shape == (5,)
    assert jnp.all(jnp.isfinite(out))


def test_strong_coupling_synchronizes(dt, seeded):
    """Identical oscillators with strong coupling reach full phase synchrony."""
    model = brainmass.KuramotoNetwork(8, omega=0.0, K=8.0, alpha=0.0)
    brainstate.nn.init_all_states(model)
    r_initial = _order_parameter(model.theta.value)
    theta = _run(model, 2000)
    r_final = _order_parameter(theta[-1])
    assert r_final > r_initial
    assert r_final > 0.95, f"expected near-synchrony, got R={r_final:.3f}"


def test_zero_coupling_does_not_synchronize(dt, seeded):
    """With ``K=0`` (and ``omega=0``) phases are frozen, so coherence is unchanged."""
    model = brainmass.KuramotoNetwork(8, omega=0.0, K=0.0)
    brainstate.nn.init_all_states(model)
    r_initial = _order_parameter(model.theta.value)
    theta = _run(model, 500)
    r_final = _order_parameter(theta[-1])
    assert r_final == pytest.approx(r_initial, abs=1e-5)


@pytest.mark.parametrize("exclude_self", [True, False])
@pytest.mark.parametrize("normalize_by_n", [True, False])
def test_coupling_flags(dt, seeded, exclude_self, normalize_by_n):
    """All combinations of ``exclude_self`` / ``normalize_by_n`` run finitely."""
    model = brainmass.KuramotoNetwork(
        5, omega=0.1, K=2.0, alpha=0.3,
        exclude_self=exclude_self, normalize_by_n=normalize_by_n)
    brainstate.nn.init_all_states(model)
    theta = _run(model, 50)
    assert jnp.all(jnp.isfinite(theta))


def test_explicit_connectivity_matrix(dt, seeded):
    """A 2-D weight matrix is accepted and produces finite dynamics."""
    w = np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)
    model = brainmass.KuramotoNetwork(4, K=2.0, conn=w)
    brainstate.nn.init_all_states(model)
    theta = _run(model, 50)
    assert theta.shape == (50, 4)
    assert jnp.all(jnp.isfinite(theta))


def test_flattened_connectivity_matrix(dt, seeded):
    """A flattened ``(N*N,)`` weight vector behaves like its 2-D form."""
    w = (np.ones((4, 4), dtype=np.float32) - np.eye(4, dtype=np.float32)).reshape(-1)
    model = brainmass.KuramotoNetwork(4, K=2.0, conn=w)
    brainstate.nn.init_all_states(model)
    with brainstate.environ.context(i=0, t=0.0 * u.ms):
        out = model.update()
    assert out.shape == (4,)


def test_connectivity_validation(dt):
    """Malformed connectivity raises a clear ``ValueError`` at update time."""
    # 1-D length not divisible by N.
    m1 = brainmass.KuramotoNetwork(4, K=1.0, conn=np.ones(5, dtype=np.float32))
    brainstate.nn.init_all_states(m1)
    with brainstate.environ.context(i=0, t=0.0 * u.ms):
        with pytest.raises(ValueError, match="not divisible"):
            m1.update()

    # 1-D divisible but not square (N_in != N).
    m2 = brainmass.KuramotoNetwork(4, K=1.0, conn=np.ones(8, dtype=np.float32))
    brainstate.nn.init_all_states(m2)
    with brainstate.environ.context(i=0, t=0.0 * u.ms):
        with pytest.raises(ValueError, match="square"):
            m2.update()

    # 2-D wrong shape.
    m3 = brainmass.KuramotoNetwork(4, K=1.0, conn=np.ones((3, 4), dtype=np.float32))
    brainstate.nn.init_all_states(m3)
    with brainstate.environ.context(i=0, t=0.0 * u.ms):
        with pytest.raises(ValueError, match="square"):
            m3.update()

    # 3-D connectivity.
    m4 = brainmass.KuramotoNetwork(2, K=1.0, conn=np.ones((2, 2, 2), dtype=np.float32))
    brainstate.nn.init_all_states(m4)
    with brainstate.environ.context(i=0, t=0.0 * u.ms):
        with pytest.raises(ValueError, match="1D flattened or 2D"):
            m4.update()


def test_noise_is_seeded_and_reproducible(dt):
    """The noise path runs and is reproducible under a fixed seed."""
    def run_with_noise():
        brainstate.random.seed(11)
        noise = brainmass.GaussianNoise(5, sigma=0.2)
        model = brainmass.KuramotoNetwork(5, omega=0.1, K=1.0, noise_theta=noise)
        brainstate.nn.init_all_states(model)
        return np.asarray(u.get_magnitude(_run(model, 30)))

    a = run_with_noise()
    b = run_with_noise()
    assert np.array_equal(a, b)
    assert np.all(np.isfinite(a))


def test_invalid_noise_type_raises(dt):
    """A non-``Noise`` ``noise_theta`` is rejected at construction."""
    with pytest.raises(AssertionError, match="noise_theta"):
        brainmass.KuramotoNetwork(4, noise_theta="not-a-noise")


def test_gradient_flows_through_coupling(dt):
    """A gradient flows through the network w.r.t. a trainable coupling ``K``."""
    def loss(k):
        brainstate.random.seed(0)
        model = brainmass.KuramotoNetwork(6, omega=0.2, K=Param(k, fit=True))
        brainstate.nn.init_all_states(model)
        theta = _run(model, 40)
        return jnp.sum(jnp.sin(u.get_magnitude(theta)) ** 2)

    value, grad = jax.value_and_grad(loss)(jnp.asarray(1.5))
    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)
