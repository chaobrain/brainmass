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
"""Tests for :class:`brainmass.LarterBreakspearStep`.

Always-on validation (TVB / tvboptim not importable in CI):

- ``test_rhs_matches_reference`` — RHS equals an embedded transcription of
  tvboptim's ``LarterBreakspear`` dynamics (rtol 1e-6).
- ``test_trajectory_matches_reference_rk4`` — ``method='rk4'`` trajectory matches an
  independent RK4 integration of the reference field (corr >= 0.99).
- ``test_limit_cycle_vs_fixed_point`` — the published regime feature: ``d_V`` in the
  limit-cycle band sustains oscillations whereas a small ``d_V`` relaxes to a fixed
  point.
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


# Default parameters, verbatim from tvboptim LarterBreakspear.DEFAULT_PARAMS.
_P = dict(
    gCa=1.1, gK=2.0, gL=0.5, gNa=6.7,
    TCa=-0.01, TK=0.0, TNa=0.3,
    d_Ca=0.15, d_K=0.3, d_Na=0.15,
    VCa=1.0, VK=-0.7, VL=-0.5, VNa=0.53,
    phi=0.7, tau_K=1.0,
    aee=0.4, aei=2.0, aie=2.0, ane=1.0, ani=0.4,
    b=0.1, C=0.1, Iext=0.3, rNMDA=0.25,
    VT=0.0, d_V=0.65, ZT=0.0, d_Z=0.7,
    QV_max=1.0, QZ_max=1.0, t_scale=1.0,
)


def _lb_reference(V, W, Z, *, c_inst, c_del, **p):
    """Reference (dV, dW, dZ) per unit time, transcribed from tvboptim."""
    m_Ca = 0.5 * (1.0 + jnp.tanh((V - p["TCa"]) / p["d_Ca"]))
    m_Na = 0.5 * (1.0 + jnp.tanh((V - p["TNa"]) / p["d_Na"]))
    m_K = 0.5 * (1.0 + jnp.tanh((V - p["TK"]) / p["d_K"]))
    QV = 0.5 * p["QV_max"] * (1.0 + jnp.tanh((V - p["VT"]) / p["d_V"]))
    QZ = 0.5 * p["QZ_max"] * (1.0 + jnp.tanh((Z - p["ZT"]) / p["d_Z"]))
    I_Ca = (
        p["gCa"]
        + (1.0 - p["C"]) * p["rNMDA"] * p["aee"] * (QV + c_inst)
        + p["C"] * p["rNMDA"] * p["aee"] * c_del
    ) * m_Ca * (V - p["VCa"])
    I_K = p["gK"] * W * (V - p["VK"])
    I_L = p["gL"] * (V - p["VL"])
    I_Na = (
        p["gNa"] * m_Na
        + (1.0 - p["C"]) * p["aee"] * (QV + c_inst)
        + p["C"] * p["aee"] * c_del
    ) * (V - p["VNa"])
    I_inh = p["aie"] * Z * QZ
    I_ext = p["ane"] * p["Iext"]
    dV = p["t_scale"] * (-I_Ca - I_K - I_L - I_Na - I_inh + I_ext)
    dW = p["t_scale"] * p["phi"] * (m_K - W) / p["tau_K"]
    dZ = p["t_scale"] * p["b"] * (p["ani"] * p["Iext"] + p["aei"] * V * QV)
    return jnp.stack([dV, dW, dZ])


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


def test_rhs_matches_reference(seeded):
    n = 5
    rng = np.random.default_rng(0)
    model = brainmass.LarterBreakspearStep(n)
    model.init_all_states()
    for _ in range(20):
        V = jnp.asarray(rng.uniform(-1.0, 1.0, size=n))
        W = jnp.asarray(rng.uniform(0.0, 1.0, size=n))
        Z = jnp.asarray(rng.uniform(-0.5, 0.5, size=n))
        V_inp = jnp.asarray(rng.uniform(-0.2, 0.2, size=n))
        model.V.value = V
        model.W.value = W
        model.Z.value = Z
        dV, dW, dZ = model.derivative((V, W, Z), 0.0 * u.ms, V_inp)
        ref = _lb_reference(V, W, Z, c_inst=0.0, c_del=V_inp, **_P)
        got = np.stack([_mag_per_ms(dV), _mag_per_ms(dW), _mag_per_ms(dZ)])
        np.testing.assert_allclose(got, np.asarray(ref), rtol=1e-6, atol=1e-6)


def test_trajectory_matches_reference_rk4(dt):
    model = brainmass.LarterBreakspearStep(1, method="rk4")
    n_steps = 600
    out = brainmass.Simulator(model, dt=dt).run(
        n_steps * dt, monitors=["V", "W", "Z"], jit=True
    )
    bm = np.stack([np.asarray(out[k]).reshape(-1) for k in ("V", "W", "Z")])

    dt_ms = u.get_magnitude(dt / u.ms)
    rhs = lambda y: _lb_reference(y[0], y[1], y[2], c_inst=0.0, c_del=0.0, **_P)
    traj = _rk4(rhs, jnp.zeros(3), dt_ms, n_steps)  # IC (0,0,0)
    ref = np.asarray(traj).T  # (3, n_steps)

    for a, b in zip(bm, ref):
        corr = np.corrcoef(a, b)[0, 1]
        rel_rmse = np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min() + 1e-12)
        assert corr >= 0.99, f"corr={corr}"
        assert rel_rmse <= 0.02, f"rel_rmse={rel_rmse}"


def test_limit_cycle_vs_fixed_point(dt):
    """Published regime feature: the limit-cycle band sustains oscillations whereas
    a sub-threshold ``d_V`` relaxes onto a fixed point.

    With the upstream default parameters the clean separation is ``d_V = 0.50``
    (fixed point) vs ``d_V = 0.57`` (limit-cycle band ``0.55 < d_V < 0.59``); pushing
    ``d_V`` much lower (< ~0.45) drives the membrane variable divergent rather than to
    a fixed point, so the test stays inside the physiological band.
    """
    n_steps = 8000

    def late_std(d_V):
        model = brainmass.LarterBreakspearStep(1, d_V=d_V, method="rk4")
        out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["V"])
        V = np.asarray(out["V"]).reshape(-1)
        return V[n_steps // 2:].std()

    std_cycle = late_std(0.57)   # limit-cycle band (0.55 < d_V < 0.59)
    std_fixed = late_std(0.50)   # below the cycle threshold -> fixed point
    assert std_cycle > 1e-2, f"expected oscillation, got std={std_cycle}"
    assert std_fixed < 1e-3, f"expected fixed point, got std={std_fixed}"
    assert std_cycle > 50 * std_fixed


@pytest.mark.parametrize("method", ["exp_euler", "rk4"])
def test_integrator_paths_run_bounded(method, dt):
    model = brainmass.LarterBreakspearStep(3, method=method)
    out = brainmass.Simulator(model, dt=dt).run(200 * dt, monitors=["V", "W", "Z"])
    for k in ("V", "W", "Z"):
        assert jnp.all(jnp.isfinite(out[k]))


def test_shapes_and_units():
    n = 4
    model = brainmass.LarterBreakspearStep(n)
    model.init_all_states()
    for s in (model.V, model.W, model.Z):
        assert s.value.shape == (n,)
    dV, dW, dZ = model.derivative(
        (model.V.value, model.W.value, model.Z.value), 0.0 * u.ms, 0.0
    )
    for d in (dV, dW, dZ):
        assert u.get_unit(d * u.ms).is_unitless


def test_update_accepts_explicit_input():
    n = 2
    model = brainmass.LarterBreakspearStep(n)
    model.init_all_states()
    with brainstate.environ.context(dt=0.1 * u.ms):
        V = model.update(V_inp=0.05)
    assert V.shape == (n,)
    assert jnp.all(jnp.isfinite(V))


def test_batched(dt):
    n, b = 3, 4
    model = brainmass.LarterBreakspearStep(n)
    out = brainmass.Simulator(model, dt=dt).run(20 * dt, monitors=["V"], batch_size=b)
    assert out["V"].shape[-2:] == (b, n)


def test_noise_changes_trajectory(seeded, dt):
    n = 3
    model = brainmass.LarterBreakspearStep(
        n, noise_V=brainmass.GaussianNoise(n, sigma=0.05)
    )
    out_noisy = brainmass.Simulator(model, dt=dt).run(50 * dt, monitors=["V"])
    out_clean = brainmass.LarterBreakspearStep(n)
    out_clean = brainmass.Simulator(out_clean, dt=dt).run(50 * dt, monitors=["V"])
    assert not jnp.allclose(out_noisy["V"], out_clean["V"])


def test_gradient_ad_vs_fd(dt):
    n_steps = 80

    def loss(iext_val):
        brainstate.random.seed(0)
        model = brainmass.LarterBreakspearStep(1, Iext=Param(iext_val, fit=True))
        out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["V"], jit=False)
        return jnp.sum(u.get_magnitude(out["V"]) ** 2)

    x0 = 0.3
    g_ad = jax.grad(loss)(x0)
    eps = 1e-3
    g_fd = (loss(x0 + eps) - loss(x0 - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(g_ad), np.asarray(g_fd), rtol=2e-2, atol=1e-3)
