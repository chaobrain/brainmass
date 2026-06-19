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
"""Tests for :class:`brainmass.WongWangExcInhStep`.

Validation strategy:

- ``test_rhs_matches_reference`` — the right-hand side equals an embedded
  transcription of tvboptim's ``WongWangExcInh`` dynamics to ``rtol = 1e-6``.
  States are sampled in the physiological (low-activity) regime so the f-I
  transfer-function argument stays away from its removable singularity.
- ``test_trajectory_matches_reference_rk4`` — a short ``method='rk4'`` trajectory
  matches an independent RK4 integration of the reference field (corr >= 0.99).
- ``test_converges_to_resting_fixed_point`` / ``test_monostable`` /
  ``test_excitatory_input_raises_rate`` — published-feature fallbacks: relaxation
  to the Deco-2014 resting state (~3 Hz), monostability, and input responsiveness.
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

# Full default parameter set (matches the constructor defaults / TVB).
_P = dict(
    a_e=310.0, b_e=125.0, d_e=0.160, gamma_e=0.641 / 1000.0, tau_e=100.0, w_p=1.4, W_e=1.0,
    a_i=615.0, b_i=177.0, d_i=0.087, gamma_i=1.0 / 1000.0, tau_i=10.0, W_i=0.7,
    J_N=0.15, J_i=1.0, I_o=0.382, I_ext=0.0, G=2.0, lamda=0.0,
)


# ---------------------------------------------------------------------------
# Embedded reference: tvboptim WongWangExcInh.dynamics (dimensionless).
# ---------------------------------------------------------------------------
def _wwei_reference(S_e, S_i, *, a_e, b_e, d_e, gamma_e, tau_e, w_p, W_e,
                    a_i, b_i, d_i, gamma_i, tau_i, W_i,
                    J_N, J_i, I_o, I_ext, G, lamda, coupling, add_e, add_i):
    """Reference (dS_e, dS_i) per unit time, transcribed from tvboptim."""
    coup_total = G * J_N * coupling
    x_e = w_p * J_N * S_e - J_i * S_i + W_e * I_o + coup_total + I_ext
    xse = a_e * x_e - b_e
    H_e = xse / (1.0 - jnp.exp(-d_e * xse))
    dS_e = -S_e / tau_e + (1.0 - S_e) * H_e * gamma_e + add_e
    x_i = J_N * S_e - S_i + W_i * I_o + lamda * coup_total
    xsi = a_i * x_i - b_i
    H_i = xsi / (1.0 - jnp.exp(-d_i * xsi))
    dS_i = -S_i / tau_i + H_i * gamma_i + add_i
    return dS_e, dS_i


def _rk4_trajectory(rhs, Se0, Si0, dt, n_steps):
    """Classic RK4 integration of the autonomous 2-state field, recording post-step."""
    def step(state, _):
        e, i = state
        k1e, k1i = rhs(e, i)
        k2e, k2i = rhs(e + 0.5 * dt * k1e, i + 0.5 * dt * k1i)
        k3e, k3i = rhs(e + 0.5 * dt * k2e, i + 0.5 * dt * k2i)
        k4e, k4i = rhs(e + dt * k3e, i + dt * k3i)
        e = e + dt / 6.0 * (k1e + 2 * k2e + 2 * k3e + k4e)
        i = i + dt / 6.0 * (k1i + 2 * k2i + 2 * k3i + k4i)
        new = (e, i)
        return new, new

    _, traj = jax.lax.scan(step, (Se0, Si0), jnp.arange(n_steps))
    return traj


def _mag_per_ms(quantity):
    return u.get_magnitude(quantity * u.ms)


# ---------------------------------------------------------------------------
# RHS fidelity (the reference regression).
# ---------------------------------------------------------------------------
def test_rhs_matches_reference(seeded):
    n = 6
    rng = np.random.default_rng(0)
    model = brainmass.WongWangExcInhStep(n, **_P)
    model.init_all_states()
    for _ in range(20):
        # Physiological regime: keep the f-I argument away from the a*x - b = 0
        # removable singularity (here both arguments stay safely negative).
        S_e = jnp.asarray(rng.uniform(0.0, 0.2, size=n))
        S_i = jnp.asarray(rng.uniform(0.1, 0.5, size=n))
        coup = jnp.asarray(rng.uniform(-0.05, 0.05, size=n))
        add_e = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=n))
        add_i = jnp.asarray(rng.uniform(-1e-3, 1e-3, size=n))
        model.S_e.value, model.S_i.value = S_e, S_i

        dS_e, dS_i = model.derivative((S_e, S_i), 0.0 * u.ms, coup, add_e, add_i)
        ref_e, ref_i = _wwei_reference(S_e, S_i, coupling=coup,
                                       add_e=add_e, add_i=add_i, **_P)
        np.testing.assert_allclose(_mag_per_ms(dS_e), np.asarray(ref_e), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(_mag_per_ms(dS_i), np.asarray(ref_i), rtol=1e-6, atol=1e-8)


def test_trajectory_matches_reference_rk4(dt):
    Se0, Si0, n_steps = 0.001, 0.001, 600
    model = brainmass.WongWangExcInhStep(
        1,
        init_S_e=braintools.init.Constant(Se0),
        init_S_i=braintools.init.Constant(Si0),
        method="rk4",
        **_P,
    )
    out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["S_e", "S_i"], jit=True)
    Se_bm = np.asarray(out["S_e"]).reshape(-1)
    Si_bm = np.asarray(out["S_i"]).reshape(-1)

    dt_ms = u.get_magnitude(dt / u.ms)
    rhs = lambda e, i: _wwei_reference(e, i, coupling=0.0, add_e=0.0, add_i=0.0, **_P)
    Se_ref, Si_ref = _rk4_trajectory(rhs, jnp.asarray(Se0), jnp.asarray(Si0), dt_ms, n_steps)
    Se_ref = np.asarray(Se_ref).reshape(-1)
    Si_ref = np.asarray(Si_ref).reshape(-1)

    for a, b in ((Se_bm, Se_ref), (Si_bm, Si_ref)):
        corr = np.corrcoef(a, b)[0, 1]
        rel_rmse = np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min())
        assert corr >= 0.99, f"corr={corr}"
        assert rel_rmse <= 0.01, f"rel_rmse={rel_rmse}"


def test_converges_to_resting_fixed_point(dt):
    """From rest the model settles to the Deco-2014 resting state (~3 Hz rates)."""
    model = brainmass.WongWangExcInhStep(1, **_P)
    out = brainmass.Simulator(model, dt=dt).run(
        30000 * dt,
        monitors={"S_e": "S_e", "S_i": "S_i",
                  "H_e": lambda m: m.H_e(), "H_i": lambda m: m.H_i()},
    )
    Se = np.asarray(out["S_e"]).reshape(-1)
    Si = np.asarray(out["S_i"]).reshape(-1)
    He = np.asarray(out["H_e"]).reshape(-1)
    Hi = np.asarray(out["H_i"]).reshape(-1)
    assert np.all(np.isfinite(Se)) and np.all(np.isfinite(Si))
    # Converged to a fixed point.
    assert Se[-200:].std() < 1e-4 and Si[-200:].std() < 1e-4
    # Physiological low-activity rates (a few Hz), gating in [0, 1].
    assert 2.0 < He[-1] < 6.0, f"H_e={He[-1]}"
    assert 2.0 < Hi[-1] < 8.0, f"H_i={Hi[-1]}"
    assert 0.0 < Se[-1] < 1.0 and 0.0 < Si[-1] < 1.0


def test_monostable(dt):
    """Unlike the decision-model WongWangStep, the default E-I regime is monostable."""
    def settle(s0):
        model = brainmass.WongWangExcInhStep(
            1,
            init_S_e=braintools.init.Constant(s0),
            init_S_i=braintools.init.Constant(s0),
            **_P,
        )
        out = brainmass.Simulator(model, dt=dt).run(30000 * dt, monitors=["S_e"])
        return float(np.asarray(out["S_e"]).reshape(-1)[-1])

    low = settle(0.001)
    high = settle(0.9)
    np.testing.assert_allclose(low, high, atol=1e-3)


def test_excitatory_input_raises_rate(dt):
    """A stronger external drive to the E pool raises its steady-state firing rate."""
    def rate(I_ext):
        model = brainmass.WongWangExcInhStep(1, **dict(_P, I_ext=I_ext))
        out = brainmass.Simulator(model, dt=dt).run(
            30000 * dt, monitors={"H_e": lambda m: m.H_e()}
        )
        return float(np.asarray(out["H_e"]).reshape(-1)[-1])

    assert rate(0.1) > rate(0.0) + 0.1


# ---------------------------------------------------------------------------
# Integrator paths, shapes, units, batching, noise, gradient.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["exp_euler", "rk4"])
def test_integrator_paths_run_bounded(method, dt):
    model = brainmass.WongWangExcInhStep(3, method=method)
    out = brainmass.Simulator(model, dt=dt).run(2000 * dt, monitors=["S_e", "S_i"])
    assert jnp.all(jnp.isfinite(out["S_e"]))
    assert jnp.all(jnp.isfinite(out["S_i"]))


def test_shapes_and_units():
    n = 4
    model = brainmass.WongWangExcInhStep(n)
    model.init_all_states()
    assert model.S_e.value.shape == (n,)
    assert model.S_i.value.shape == (n,)
    dS_e, dS_i = model.derivative((model.S_e.value, model.S_i.value), 0.0 * u.ms, 0.0)
    assert u.get_unit(dS_e * u.ms).is_unitless
    assert u.get_unit(dS_i * u.ms).is_unitless
    # Auxiliary firing rates are finite at the initial state.
    assert jnp.isfinite(model.H_e()).all() and jnp.isfinite(model.H_i()).all()


def test_batched(dt):
    n, b = 3, 5
    model = brainmass.WongWangExcInhStep(n)
    out = brainmass.Simulator(model, dt=dt).run(20 * dt, monitors=["S_e"], batch_size=b)
    assert out["S_e"].shape[-2:] == (b, n)


def test_update_accepts_explicit_coupling():
    n = 2
    model = brainmass.WongWangExcInhStep(n)
    model.init_all_states()
    with brainstate.environ.context(dt=0.1 * u.ms):
        S_e = model.update(coupling=0.05)
    assert S_e.shape == (n,)
    assert jnp.all(jnp.isfinite(S_e))


def test_noise_changes_trajectory(seeded, dt):
    n = 3
    model = brainmass.WongWangExcInhStep(
        n,
        noise_e=brainmass.GaussianNoise(n, sigma=1e-3),
        noise_i=brainmass.GaussianNoise(n, sigma=1e-3),
    )
    out_noisy = brainmass.Simulator(model, dt=dt).run(2000 * dt, monitors=["S_e", "S_i"])
    model_clean = brainmass.WongWangExcInhStep(n)
    out_clean = brainmass.Simulator(model_clean, dt=dt).run(2000 * dt, monitors=["S_e", "S_i"])
    assert not jnp.allclose(out_noisy["S_e"], out_clean["S_e"])
    assert not jnp.allclose(out_noisy["S_i"], out_clean["S_i"])


def test_invalid_args():
    with pytest.raises(AssertionError):
        brainmass.WongWangExcInhStep(1, init_S_e=None)
    with pytest.raises(AssertionError):
        brainmass.WongWangExcInhStep(1, noise_e=object())


def test_gradient_ad_vs_fd(dt):
    """AD through Simulator.run wrt the recurrence weight w_p matches finite differences."""
    n_steps = 400

    def loss(wp_val):
        brainstate.random.seed(0)
        model = brainmass.WongWangExcInhStep(1, w_p=Param(wp_val, fit=True))
        out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["S_e"], jit=False)
        return jnp.sum(u.get_magnitude(out["S_e"]) ** 2)

    wp0 = 1.4
    g_ad = jax.grad(loss)(wp0)
    eps = 1e-3
    g_fd = (loss(wp0 + eps) - loss(wp0 - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(g_ad), np.asarray(g_fd), rtol=2e-2, atol=1e-5)
