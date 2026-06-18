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
"""Tests for :class:`brainmass.EpileptorStep`.

Always-on validation (TVB / tvboptim not importable in CI):

- ``test_rhs_matches_reference`` — RHS equals an embedded transcription of
  tvboptim's ``Epileptor`` dynamics, across both branches of every piecewise term
  and both ``modification`` settings (rtol 1e-6).
- ``test_trajectory_matches_reference_rk4`` — ``method='rk4'`` trajectory matches an
  independent RK4 integration of the reference (corr >= 0.99).
- ``test_seizure_onset_and_offset`` — the published feature: an epileptogenic node
  (``x0 = -1.6``) autonomously enters and leaves the ictal regime, whereas a healthy
  node (``x0 = -2.4``) never seizes.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainstate
import brainunit as u
from brainstate.nn import Param

import brainmass

_HAS_TVB = importlib.util.find_spec("tvb") is not None
_HAS_TVBOPTIM = importlib.util.find_spec("tvboptim") is not None
requires_tvb = pytest.mark.skipif(not _HAS_TVB, reason="TVB not installed")
requires_tvboptim = pytest.mark.skipif(not _HAS_TVBOPTIM, reason="tvboptim not installed")


# Default parameters, verbatim from tvboptim Epileptor.DEFAULT_PARAMS
# (``s`` is in that bunch but unused by the dynamics, so it is omitted here).
_P = dict(
    a=1.0, b=3.0, c=1.0, d=5.0, r=0.00035, x0=-1.6, Iext=3.1, slope=0.0,
    Iext2=0.45, tau=10.0, aa=6.0, bb=2.0,
    Kvf=0.0, Kf=0.0, Ks=0.0, tt=1.0,
)


def _epi_reference(x1, y1, z, x2, y2, g, *, c_pop1, c_pop2, modification, **p):
    """Reference Epileptor derivative array, transcribed from tvboptim."""
    f1 = jnp.where(x1 < 0.0, -p["a"] * x1 ** 2 + p["b"] * x1,
                   p["slope"] - x2 + 0.6 * (z - 4.0) ** 2)
    dx1 = p["tt"] * (y1 - z + p["Iext"] + p["Kvf"] * c_pop1 + f1 * x1)
    dy1 = p["tt"] * (p["c"] - p["d"] * x1 ** 2 - y1)

    z_nl = jnp.where(z < 0.0, -0.1 * z ** 7, 0.0)
    h_nl = p["x0"] + 3.0 / (1.0 + jnp.exp(-(x1 + 0.5) / 0.1))
    h_lin = 4.0 * (x1 - p["x0"]) + z_nl
    h = modification * h_nl + (1.0 - modification) * h_lin
    dz = p["tt"] * p["r"] * (h - z + p["Ks"] * c_pop1)

    dx2 = p["tt"] * (-y2 + x2 - x2 ** 3 + p["Iext2"] + p["bb"] * g
                     - 0.3 * (z - 3.5) + p["Kf"] * c_pop2)
    f2 = jnp.where(x2 < -0.25, 0.0, p["aa"] * (x2 + 0.25))
    dy2 = p["tt"] * (-y2 + f2) / p["tau"]
    dg = p["tt"] * (-0.01 * (g - 0.1 * x1))
    return jnp.stack([dx1, dy1, dz, dx2, dy2, dg])


def _rk4(rhs, y0, dt, n_steps):
    def step(y, _):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y, y

    _, traj = jax.lax.scan(step, y0, jnp.arange(n_steps))
    return traj


def _mag_per_ms(q):
    return u.get_magnitude(q * u.ms)


_STATE = ("x1", "y1", "z", "x2", "y2", "g")


@pytest.mark.parametrize("modification", [0.0, 1.0])
def test_rhs_matches_reference(modification, seeded):
    """RHS matches the reference across both piecewise branches of f1/f2/z_nl.

    Uses non-zero coupling gains (Kvf, Kf, Ks) so the coupling terms are exercised.
    """
    n = 8
    rng = np.random.default_rng(0)
    coupling = dict(Kvf=1.5, Kf=0.7, Ks=2.0)
    p = {**_P, **coupling}
    model = brainmass.EpileptorStep(n, modification=modification, **coupling)
    model.init_all_states()
    for _ in range(25):
        # Ranges chosen to straddle x1=0, x2=-0.25 and z=0 (both branches).
        x1 = jnp.asarray(rng.uniform(-2.0, 2.0, size=n))
        y1 = jnp.asarray(rng.uniform(-12.0, 0.0, size=n))
        z = jnp.asarray(rng.uniform(-1.0, 5.0, size=n))
        x2 = jnp.asarray(rng.uniform(-1.5, 1.0, size=n))
        y2 = jnp.asarray(rng.uniform(-1.0, 1.0, size=n))
        g = jnp.asarray(rng.uniform(-1.0, 1.0, size=n))
        c1 = jnp.asarray(rng.uniform(-0.3, 0.3, size=n))
        c2 = jnp.asarray(rng.uniform(-0.3, 0.3, size=n))
        for name, val in zip(_STATE, (x1, y1, z, x2, y2, g)):
            getattr(model, name).value = val

        ders = model.derivative((x1, y1, z, x2, y2, g), 0.0 * u.ms, c1, c2)
        got = np.stack([_mag_per_ms(d) for d in ders])
        ref = _epi_reference(x1, y1, z, x2, y2, g, c_pop1=c1, c_pop2=c2,
                             modification=modification, **p)
        np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-6, atol=1e-6)


def test_trajectory_matches_reference_rk4(dt):
    model = brainmass.EpileptorStep(1, method="rk4")
    n_steps = 800
    out = brainmass.Simulator(model, dt=dt).run(
        n_steps * dt, monitors=list(_STATE), jit=True
    )
    bm = np.stack([np.asarray(out[k]).reshape(-1) for k in _STATE])

    dt_ms = u.get_magnitude(dt / u.ms)
    ic = jnp.asarray([-1.5, -10.0, 3.5, -1.0, 0.0, 0.0])
    rhs = lambda y: _epi_reference(*y, c_pop1=0.0, c_pop2=0.0, modification=0.0, **_P)
    ref = np.asarray(_rk4(rhs, ic, dt_ms, n_steps)).T  # (6, n_steps)

    # Flatten across all states (tvboptim's comparison metric).
    a, b = bm.flatten(), ref.flatten()
    corr = np.corrcoef(a, b)[0, 1]
    rel_rmse = np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min())
    assert corr >= 0.99, f"corr={corr}"
    assert rel_rmse <= 0.01, f"rel_rmse={rel_rmse}"


def test_seizure_onset_and_offset(dt):
    """Published feature: epileptogenic node seizes (onset+offset); healthy does not.

    ``r`` is raised from its default ``3.5e-4`` so a few seizure cycles fit in a
    tractable number of steps; this only rescales the slow time axis.
    """
    n_steps = 30000
    w = 1000

    def run(x0):
        model = brainmass.EpileptorStep(1, x0=x0, r=0.0006)
        out = brainmass.Simulator(model, dt=dt).run(
            n_steps * dt, monitors={"lfp": lambda m: m.lfp(), "x1": "x1"}
        )
        return (np.asarray(out["lfp"]).reshape(-1), np.asarray(out["x1"]).reshape(-1))

    lfp, x1 = run(-1.6)            # epileptogenic
    nwin = len(lfp) // w
    wstd = lfp[: nwin * w].reshape(nwin, w).std(axis=1)
    # Ictal bursts (high std) alternate with interictal quiet (low std): onset+offset.
    assert wstd.max() > 0.3, f"no ictal burst, max std={wstd.max()}"
    assert wstd.min() < 0.1, f"never quiescent, min std={wstd.min()}"
    assert wstd.max() > 5 * wstd.min()
    ictal = wstd > 0.5 * (wstd.max() + wstd.min())
    transitions = int(np.sum(np.abs(np.diff(ictal.astype(int)))))
    assert transitions >= 2, "expected at least one onset and one offset"
    assert x1.max() > 0.0, "epileptogenic node never entered the ictal regime"

    lfp_h, x1_h = run(-2.4)        # healthy
    assert x1_h.max() < 0.0, "healthy node entered the ictal regime"
    assert (lfp.max() - lfp.min()) > 1.5 * (lfp_h.max() - lfp_h.min())


def test_stiff_slow_z_stable():
    """The default stiff slow ``z`` stays bounded under exp_euler over a long run."""
    model = brainmass.EpileptorStep(1)  # default r=3.5e-4
    out = brainmass.Simulator(model, dt=0.1 * u.ms).run(
        20000 * (0.1 * u.ms), monitors=["z", "x1"]
    )
    assert jnp.all(jnp.isfinite(out["z"]))
    z = np.asarray(out["z"]).reshape(-1)
    # z evolves slowly and stays in a physiological band around its IC (3.5).
    assert z.min() > 2.0 and z.max() < 5.0


@pytest.mark.parametrize("method", ["exp_euler", "rk4"])
def test_integrator_paths_run_bounded(method, dt):
    model = brainmass.EpileptorStep(3, method=method)
    out = brainmass.Simulator(model, dt=dt).run(300 * dt, monitors=list(_STATE))
    for k in _STATE:
        assert jnp.all(jnp.isfinite(out[k]))


def test_shapes_units_and_lfp():
    n = 4
    model = brainmass.EpileptorStep(n)
    model.init_all_states()
    for name in _STATE:
        assert getattr(model, name).value.shape == (n,)
    ders = model.derivative(
        tuple(getattr(model, k).value for k in _STATE), 0.0 * u.ms, 0.0, 0.0
    )
    for d in ders:
        assert u.get_unit(d * u.ms).is_unitless
    # lfp == x2 - x1
    np.testing.assert_allclose(
        np.asarray(model.lfp()), np.asarray(model.x2.value - model.x1.value)
    )


def test_update_returns_lfp_and_accepts_inputs():
    n = 2
    model = brainmass.EpileptorStep(n)
    model.init_all_states()
    with brainstate.environ.context(dt=0.1 * u.ms):
        lfp = model.update(x1_inp=0.05, x2_inp=0.02)
    assert lfp.shape == (n,)
    np.testing.assert_allclose(
        np.asarray(lfp), np.asarray(model.x2.value - model.x1.value)
    )


def test_batched(dt):
    n, b = 3, 4
    model = brainmass.EpileptorStep(n)
    out = brainmass.Simulator(model, dt=dt).run(50 * dt, monitors=["x1"], batch_size=b)
    assert out["x1"].shape[-2:] == (b, n)


def test_noise_changes_trajectory(seeded, dt):
    n = 3
    model = brainmass.EpileptorStep(
        n,
        noise_x1=brainmass.GaussianNoise(n, sigma=0.05),
        noise_x2=brainmass.GaussianNoise(n, sigma=0.05),
    )
    out_noisy = brainmass.Simulator(model, dt=dt).run(100 * dt, monitors=["x1", "x2"])
    clean = brainmass.EpileptorStep(n)
    out_clean = brainmass.Simulator(clean, dt=dt).run(100 * dt, monitors=["x1", "x2"])
    assert not jnp.allclose(out_noisy["x1"], out_clean["x1"])
    assert not jnp.allclose(out_noisy["x2"], out_clean["x2"])


def test_gradient_ad_vs_fd(dt):
    n_steps = 100

    def loss(iext_val):
        brainstate.random.seed(0)
        model = brainmass.EpileptorStep(1, Iext=Param(iext_val, fit=True))
        out = brainmass.Simulator(model, dt=dt).run(
            n_steps * dt, monitors={"lfp": lambda m: m.lfp()}, jit=False
        )
        return jnp.sum(u.get_magnitude(out["lfp"]) ** 2)

    x0 = 3.1
    g_ad = jax.grad(loss)(x0)
    eps = 1e-3
    g_fd = (loss(x0 + eps) - loss(x0 - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(g_ad), np.asarray(g_fd), rtol=2e-2, atol=1e-3)


@requires_tvb
@requires_tvboptim
def test_live_tvb_comparison():  # pragma: no cover
    raise AssertionError("placeholder: enable when tvb + tvboptim are installed")
