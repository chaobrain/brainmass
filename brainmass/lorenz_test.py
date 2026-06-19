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
"""Tests for :class:`brainmass.LorenzStep`.

Validation strategy (the Lorenz flow is chaotic, so trajectory parity must use a
SHORT horizon — sensitive dependence forbids long-run agreement):

- ``test_rhs_matches_reference`` — the right-hand side equals an embedded
  transcription of tvboptim's ``Lorenz`` dynamics to ``rtol = 1e-6``.
- ``test_trajectory_matches_reference_rk4`` — a short ``method='rk4'`` trajectory
  matches an independent RK4 integration of the reference field (corr >= 0.99).
- ``test_lorenz_is_chaotic`` — two trajectories 1e-8 apart diverge by orders of
  magnitude while staying bounded (positive largest Lyapunov exponent).
"""

import braintools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
import brainunit as u
from brainstate.nn import Param

import brainmass

_P = dict(sigma=10.0, rho=28.0, beta=8.0 / 3.0)


# ---------------------------------------------------------------------------
# Embedded reference: tvboptim Lorenz.dynamics (dimensionless).
# ---------------------------------------------------------------------------
def _lorenz_reference(x, y, z, *, sigma, rho, beta, coup):
    """Reference (dx, dy, dz) per unit time, transcribed from tvboptim."""
    dx = sigma * (y - x) + coup
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


def _rk4_trajectory(rhs, y0, dt, n_steps):
    """Classic RK4 integration of the autonomous 3-state field, recording post-step."""
    def step(state, _):
        s = jnp.asarray(state)
        k1 = jnp.asarray(rhs(*s))
        k2 = jnp.asarray(rhs(*(s + 0.5 * dt * k1)))
        k3 = jnp.asarray(rhs(*(s + 0.5 * dt * k2)))
        k4 = jnp.asarray(rhs(*(s + dt * k3)))
        new = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return new, new

    _, traj = jax.lax.scan(step, jnp.asarray(y0), jnp.arange(n_steps))
    return traj  # (n_steps, 3)


def _mag_per_ms(quantity):
    return u.get_magnitude(quantity * u.ms)


# ---------------------------------------------------------------------------
# RHS fidelity (the reference regression).
# ---------------------------------------------------------------------------
def test_rhs_matches_reference(seeded):
    n = 6
    rng = np.random.default_rng(0)
    params = dict(sigma=12.0, rho=25.0, beta=2.5)
    model = brainmass.LorenzStep(n, **params)
    model.init_all_states()
    for _ in range(20):
        x = jnp.asarray(rng.uniform(-20.0, 20.0, size=n))
        y = jnp.asarray(rng.uniform(-25.0, 25.0, size=n))
        z = jnp.asarray(rng.uniform(0.0, 50.0, size=n))
        coup = jnp.asarray(rng.uniform(-5.0, 5.0, size=n))
        model.x.value, model.y.value, model.z.value = x, y, z

        dx, dy, dz = model.derivative((x, y, z), 0.0 * u.ms, coup)
        ref_dx, ref_dy, ref_dz = _lorenz_reference(x, y, z, coup=coup, **params)
        np.testing.assert_allclose(_mag_per_ms(dx), np.asarray(ref_dx), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(_mag_per_ms(dy), np.asarray(ref_dy), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(_mag_per_ms(dz), np.asarray(ref_dz), rtol=1e-6, atol=1e-6)


def test_trajectory_matches_reference_rk4():
    dt = 0.01 * u.ms
    n_steps = 250  # short horizon: chaos forbids long-run parity
    y0 = (1.0, 1.0, 1.0)
    model = brainmass.LorenzStep(
        1,
        init_x=braintools.init.Constant(y0[0]),
        init_y=braintools.init.Constant(y0[1]),
        init_z=braintools.init.Constant(y0[2]),
        method="rk4",
        **_P,
    )
    out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["x", "y", "z"], jit=True)
    traj_bm = np.stack([np.asarray(out[k]).reshape(-1) for k in ("x", "y", "z")], axis=1)

    dt_ms = u.get_magnitude(dt / u.ms)
    rhs = lambda x, y, z: _lorenz_reference(x, y, z, coup=0.0, **_P)
    traj_ref = np.asarray(_rk4_trajectory(rhs, y0, dt_ms, n_steps))

    for j in range(3):
        a, b = traj_bm[:, j], traj_ref[:, j]
        corr = np.corrcoef(a, b)[0, 1]
        rel_rmse = np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min())
        assert corr >= 0.99, f"state {j}: corr={corr}"
        assert rel_rmse <= 0.02, f"state {j}: rel_rmse={rel_rmse}"


def test_lorenz_is_chaotic():
    """Sensitive dependence: a tiny IC perturbation diverges by >1e3, staying bounded.

    The perturbation is 1e-3 (comfortably above the float32 epsilon ~1e-7, so it is
    a genuinely resolved difference rather than round-off) and chaos amplifies it.
    """
    dt = 0.01 * u.ms
    n_steps = 1500
    eps = 1e-3

    def run(x0):
        m = brainmass.LorenzStep(
            1,
            init_x=braintools.init.Constant(x0),
            init_y=braintools.init.Constant(1.0),
            init_z=braintools.init.Constant(1.0),
            **_P,
        )
        out = brainmass.Simulator(m, dt=dt).run(n_steps * dt, monitors=["x", "y", "z"], jit=True)
        return np.stack([np.asarray(out[k]).reshape(-1) for k in ("x", "y", "z")], axis=1)

    a = run(1.0)
    b = run(1.0 + eps)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    # Bounded on the attractor (well within |state| < 100).
    assert np.abs(a).max() < 100.0 and np.abs(b).max() < 100.0
    final_sep = np.linalg.norm(a[-1] - b[-1])
    assert final_sep / eps > 1e3, f"separation grew only {final_sep / eps:.3g}x"
    # Positive largest Lyapunov exponent (per natural time unit = per ms here).
    lyap = np.log(final_sep / eps) / (n_steps * u.get_magnitude(dt / u.ms))
    assert lyap > 0.0, f"lyap={lyap}"


# ---------------------------------------------------------------------------
# Integrator paths, shapes, units, batching, noise, gradient.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["exp_euler", "rk4"])
def test_integrator_paths_run_bounded(method):
    dt = 0.01 * u.ms
    model = brainmass.LorenzStep(3, method=method)
    out = brainmass.Simulator(model, dt=dt).run(500 * dt, monitors=["x", "y", "z"])
    for k in ("x", "y", "z"):
        assert jnp.all(jnp.isfinite(out[k]))
        assert jnp.all(jnp.abs(out[k]) < 100.0)


def test_shapes_and_units():
    n = 4
    model = brainmass.LorenzStep(n)
    model.init_all_states()
    assert model.x.value.shape == (n,)
    assert model.y.value.shape == (n,)
    assert model.z.value.shape == (n,)
    dx, dy, dz = model.derivative((model.x.value, model.y.value, model.z.value), 0.0 * u.ms, 0.0)
    for d in (dx, dy, dz):
        assert u.get_unit(d * u.ms).is_unitless


def test_batched():
    dt = 0.01 * u.ms
    n, b = 3, 5
    model = brainmass.LorenzStep(n)
    out = brainmass.Simulator(model, dt=dt).run(20 * dt, monitors=["x"], batch_size=b)
    assert out["x"].shape[-2:] == (b, n)


def test_update_accepts_explicit_coupling():
    n = 2
    model = brainmass.LorenzStep(n)
    model.init_all_states()
    with brainstate.environ.context(dt=0.01 * u.ms):
        x = model.update(x_inp=0.5)
    assert x.shape == (n,)
    assert jnp.all(jnp.isfinite(x))


def test_noise_changes_trajectory(seeded):
    dt = 0.01 * u.ms
    n = 3
    model = brainmass.LorenzStep(n, noise_x=brainmass.GaussianNoise(n, sigma=0.5))
    out_noisy = brainmass.Simulator(model, dt=dt).run(100 * dt, monitors=["x"])
    model_clean = brainmass.LorenzStep(n)
    out_clean = brainmass.Simulator(model_clean, dt=dt).run(100 * dt, monitors=["x"])
    assert not jnp.allclose(out_noisy["x"], out_clean["x"])


def test_invalid_args():
    with pytest.raises(AssertionError):
        brainmass.LorenzStep(1, init_x=None)
    with pytest.raises(AssertionError):
        brainmass.LorenzStep(1, noise_x=object())


def test_gradient_ad_vs_fd():
    """AD through a SHORT Simulator.run wrt rho matches finite differences.

    The horizon is deliberately short: gradients through a chaotic flow grow
    exponentially, so AD-vs-FD agreement is only meaningful before trajectories
    decorrelate.
    """
    dt = 0.01 * u.ms
    n_steps = 30

    def loss(rho_val):
        brainstate.random.seed(0)
        model = brainmass.LorenzStep(1, rho=Param(rho_val, fit=True))
        out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["x"], jit=False)
        return jnp.sum(u.get_magnitude(out["x"]) ** 2)

    rho0 = 28.0
    g_ad = jax.grad(loss)(rho0)
    eps = 1e-3
    g_fd = (loss(rho0 + eps) - loss(rho0 - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(g_ad), np.asarray(g_fd), rtol=2e-2, atol=1e-4)
