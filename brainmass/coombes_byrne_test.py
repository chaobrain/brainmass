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
"""Tests for :class:`brainmass.CoombesByrneStep`.

Always-on validation strategy (TVB / tvboptim are not importable in CI):

- ``test_rhs_matches_reference`` — the right-hand side equals an embedded
  transcription of tvboptim's ``CoombesByrne2D`` dynamics (the JAX-faithful TVB
  form) to ``rtol = 1e-6``. This is the equation-level reference regression.
- ``test_trajectory_matches_reference_rk4`` — a short trajectory integrated with
  ``method='rk4'`` matches an independent RK4 integration of the reference field
  (correlation >= 0.99, small relative RMSE), mirroring tvboptim's
  ``test_tvb_comparison``.
- ``test_coombes_byrne_reduces_to_mpr`` — at ``k = 0`` the conductance vanishes
  and the field reduces to Montbrio-Pazo-Roxin with ``J = 0``.
"""

import importlib.util

import braintools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
import brainunit as u
from brainstate.nn import Param

import brainmass

# ---------------------------------------------------------------------------
# Live-reference gating (tvb / tvboptim are not importable in CI).
# ---------------------------------------------------------------------------
_HAS_TVB = importlib.util.find_spec("tvb") is not None
_HAS_TVBOPTIM = importlib.util.find_spec("tvboptim") is not None
requires_tvb = pytest.mark.skipif(not _HAS_TVB, reason="TVB not installed")
requires_tvboptim = pytest.mark.skipif(
    not _HAS_TVBOPTIM, reason="tvboptim not installed"
)


# ---------------------------------------------------------------------------
# Embedded reference: tvboptim CoombesByrne2D.dynamics (dimensionless).
# ---------------------------------------------------------------------------
def _cb_reference(r, v, *, Delta, eta, k, v_syn, coup):
    """Reference (dr/dt, dv/dt) per unit time, transcribed from tvboptim."""
    g = k * jnp.pi * r
    dr = Delta / jnp.pi + 2.0 * v * r - g * r
    dv = v**2 - (jnp.pi * r) ** 2 + eta + (v_syn - v) * g + coup
    return dr, dv


def _mpr_reference(r, v, *, Delta, eta, J, coup):
    """Reference Montbrio-Pazo-Roxin field (tau = 1), the k -> 0 limit of CB."""
    dr = Delta / jnp.pi + 2.0 * v * r
    dv = v**2 - (jnp.pi * r) ** 2 + eta + J * r + coup
    return dr, dv


def _rk4_trajectory(rhs, r0, v0, dt, n_steps):
    """Classic RK4 integration of an autonomous 2-state field, recording post-step."""
    def step(state, _):
        r, v = state
        k1r, k1v = rhs(r, v)
        k2r, k2v = rhs(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v)
        k3r, k3v = rhs(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v)
        k4r, k4v = rhs(r + dt * k3r, v + dt * k3v)
        r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        new = (r, v)
        return new, new

    _, traj = jax.lax.scan(step, (r0, v0), jnp.arange(n_steps))
    return traj  # (r_traj, v_traj), each (n_steps, ...)


def _mag_per_ms(quantity):
    """Strip the 1/ms unit a brainmass RHS carries -> dimensionless per-time value."""
    return u.get_magnitude(quantity * u.ms)


# ---------------------------------------------------------------------------
# RHS fidelity (the reference regression).
# ---------------------------------------------------------------------------
def test_rhs_matches_reference(seeded):
    n = 6
    rng = np.random.default_rng(0)
    params = dict(Delta=0.7, eta=1.3, k=0.9, v_syn=-3.5)
    model = brainmass.CoombesByrneStep(n, **params)
    model.init_all_states()

    for _ in range(20):
        r = jnp.asarray(rng.uniform(0.0, 2.0, size=n))
        v = jnp.asarray(rng.uniform(-3.0, 3.0, size=n))
        r_inp = jnp.asarray(rng.uniform(-0.5, 0.5, size=n))
        v_inp = jnp.asarray(rng.uniform(-1.0, 1.0, size=n))
        model.r.value = r
        model.v.value = v

        dr, dv = model.derivative((r, v), 0.0 * u.ms, r_inp, v_inp)
        # The external input on the v-port plays the role of the coupling term;
        # the r-port adds directly to dr.
        ref_dr, ref_dv = _cb_reference(r, v, coup=v_inp, **params)
        ref_dr = ref_dr + r_inp

        np.testing.assert_allclose(_mag_per_ms(dr), np.asarray(ref_dr), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(_mag_per_ms(dv), np.asarray(ref_dv), rtol=1e-6, atol=1e-6)


def test_trajectory_matches_reference_rk4(dt):
    params = dict(Delta=1.0, eta=2.0, k=1.0, v_syn=-4.0)
    r0, v0 = 0.1, 0.0
    n_steps = 400
    model = brainmass.CoombesByrneStep(
        1,
        init_r=braintools.init.Constant(r0),
        init_v=braintools.init.Constant(v0),
        method="rk4",
        **params,
    )
    out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["r", "v"], jit=True)
    r_bm = np.asarray(out["r"]).reshape(-1)
    v_bm = np.asarray(out["v"]).reshape(-1)

    dt_ms = u.get_magnitude(dt / u.ms)
    rhs = lambda r, v: _cb_reference(r, v, coup=0.0, **params)
    r_ref, v_ref = _rk4_trajectory(rhs, jnp.asarray(r0), jnp.asarray(v0), dt_ms, n_steps)
    r_ref = np.asarray(r_ref).reshape(-1)
    v_ref = np.asarray(v_ref).reshape(-1)

    for a, b in ((r_bm, r_ref), (v_bm, v_ref)):
        corr = np.corrcoef(a, b)[0, 1]
        rel_rmse = np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min())
        assert corr >= 0.99, f"corr={corr}"
        assert rel_rmse <= 0.01, f"rel_rmse={rel_rmse}"


def test_coombes_byrne_reduces_to_mpr(seeded):
    """At k=0 the conductance vanishes -> MPR field with J=0 (relationship for goal-10)."""
    n = 5
    rng = np.random.default_rng(1)
    params = dict(Delta=1.0, eta=-5.0, k=0.0, v_syn=-4.0)
    model = brainmass.CoombesByrneStep(n, **params)
    model.init_all_states()
    r = jnp.asarray(rng.uniform(0.0, 1.5, size=n))
    v = jnp.asarray(rng.uniform(-2.0, 2.0, size=n))
    model.r.value = r
    model.v.value = v
    dr, dv = model.derivative((r, v), 0.0 * u.ms, 0.0, 0.0)
    ref_dr, ref_dv = _mpr_reference(r, v, Delta=1.0, eta=-5.0, J=0.0, coup=0.0)
    np.testing.assert_allclose(_mag_per_ms(dr), np.asarray(ref_dr), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(_mag_per_ms(dv), np.asarray(ref_dv), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Integrator paths, shapes, units, batching, noise, gradient.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["exp_euler", "rk4"])
def test_integrator_paths_run_bounded(method, dt):
    model = brainmass.CoombesByrneStep(3, method=method)
    out = brainmass.Simulator(model, dt=dt).run(100 * dt, monitors=["r", "v"])
    assert jnp.all(jnp.isfinite(out["r"]))
    assert jnp.all(jnp.isfinite(out["v"]))


def test_shapes_and_units():
    n = 4
    model = brainmass.CoombesByrneStep(n)
    model.init_all_states()
    assert model.r.value.shape == (n,)
    assert model.v.value.shape == (n,)
    dr, dv = model.derivative((model.r.value, model.v.value), 0.0 * u.ms, 0.0, 0.0)
    # RHS carries unit 1/ms.
    assert u.get_unit(dr * u.ms).is_unitless
    assert u.get_unit(dv * u.ms).is_unitless


def test_batched(dt):
    n, b = 3, 5
    model = brainmass.CoombesByrneStep(n)
    out = brainmass.Simulator(model, dt=dt).run(20 * dt, monitors=["r"], batch_size=b)
    assert out["r"].shape[-2:] == (b, n)


def test_update_accepts_explicit_inputs():
    n = 2
    model = brainmass.CoombesByrneStep(n)
    model.init_all_states()
    with brainstate.environ.context(dt=0.1 * u.ms):
        r = model.update(r_inp=0.05, v_inp=0.1)
    assert r.shape == (n,)
    assert jnp.all(jnp.isfinite(r))


def test_noise_changes_trajectory(seeded, dt):
    n = 3
    model = brainmass.CoombesByrneStep(
        n,
        noise_r=brainmass.GaussianNoise(n, sigma=0.1),
        noise_v=brainmass.GaussianNoise(n, sigma=0.2),
    )
    out_noisy = brainmass.Simulator(model, dt=dt).run(50 * dt, monitors=["r", "v"])

    model_clean = brainmass.CoombesByrneStep(n)
    out_clean = brainmass.Simulator(model_clean, dt=dt).run(50 * dt, monitors=["r", "v"])
    assert not jnp.allclose(out_noisy["v"], out_clean["v"])
    assert not jnp.allclose(out_noisy["r"], out_clean["r"])


def test_gradient_ad_vs_fd(dt):
    """AD through Simulator.run wrt eta matches a finite-difference estimate."""
    n_steps = 60

    def loss(eta_val):
        brainstate.random.seed(0)
        model = brainmass.CoombesByrneStep(1, eta=Param(eta_val, fit=True), v_syn=-4.0)
        out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["r"], jit=False)
        return jnp.sum(u.get_magnitude(out["r"]) ** 2)

    eta0 = -3.0
    g_ad = jax.grad(loss)(eta0)
    eps = 1e-3
    g_fd = (loss(eta0 + eps) - loss(eta0 - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(g_ad), np.asarray(g_fd), rtol=2e-2, atol=1e-4)


@requires_tvb
@requires_tvboptim
def test_live_tvb_comparison():  # pragma: no cover - skipped unless TVB present
    """Live regression against TVB; skipped in CI (TVB not importable)."""
    raise AssertionError("placeholder: enable when tvb + tvboptim are installed")
