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
"""Tests for :class:`brainmass.Generic2dOscillatorStep`.

Validation strategy:

- ``test_rhs_matches_reference`` — the right-hand side equals an embedded
  transcription of tvboptim's ``Generic2dOscillator`` dynamics to ``rtol = 1e-6``,
  exercised across the three published parameter regimes (excitable, bistable,
  Morris-Lecar) and with non-zero coupling + stimulus inputs.
- ``test_trajectory_matches_reference_rk4`` — a short ``method='rk4'`` trajectory
  matches an independent RK4 integration of the reference field (corr >= 0.99).
- ``test_converges_to_analytic_fixed_point`` / ``test_excitable_input_integration``
  — published-feature fallbacks: the bistable-nullcline configuration relaxes to
  the analytic nullcline-intersection root, and the excitable regime integrates a
  constant drive to a monotonically input-dependent fixed point.
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

# Full parameter sets for the three published regimes (defaults for the rest).
_DEFAULTS = dict(e=3.0, f=1.0, g=0.0, alpha=1.0, beta=1.0, gamma=1.0, I=0.0, tau=1.0)
_EXCITABLE = dict(_DEFAULTS, a=-2.0, b=-10.0, c=0.0, d=0.02)
_BISTABLE = dict(_DEFAULTS, a=1.0, b=0.0, c=-5.0, d=0.02)
_MORRIS_LECAR = dict(_DEFAULTS, a=0.5, b=0.6, c=-4.0, d=0.02)


# ---------------------------------------------------------------------------
# Embedded reference: tvboptim Generic2dOscillator.dynamics (dimensionless).
# ---------------------------------------------------------------------------
def _g2d_reference(V, W, *, a, b, c, d, e, f, g, alpha, beta, gamma, I, tau, coup, stim_V, stim_W):
    """Reference (dV, dW) per unit time, transcribed from tvboptim."""
    dV = d * tau * (-f * V ** 3 + e * V ** 2 + g * V + alpha * W + gamma * I + gamma * coup) + stim_V
    dW = (d / tau) * (a + b * V + c * V ** 2 - beta * W) + stim_W
    return dV, dW


def _rk4_trajectory(rhs, V0, W0, dt, n_steps):
    """Classic RK4 integration of the autonomous 2-state field, recording post-step."""
    def step(state, _):
        V, W = state
        k1v, k1w = rhs(V, W)
        k2v, k2w = rhs(V + 0.5 * dt * k1v, W + 0.5 * dt * k1w)
        k3v, k3w = rhs(V + 0.5 * dt * k2v, W + 0.5 * dt * k2w)
        k4v, k4w = rhs(V + dt * k3v, W + dt * k3w)
        V = V + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        W = W + dt / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        new = (V, W)
        return new, new

    _, traj = jax.lax.scan(step, (V0, W0), jnp.arange(n_steps))
    return traj


def _mag_per_ms(quantity):
    return u.get_magnitude(quantity * u.ms)


# ---------------------------------------------------------------------------
# RHS fidelity across the published regimes (the reference regression).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("regime", [_EXCITABLE, _BISTABLE, _MORRIS_LECAR],
                         ids=["excitable", "bistable", "morris_lecar"])
def test_rhs_matches_reference(regime, seeded):
    n = 6
    rng = np.random.default_rng(0)
    model = brainmass.Generic2dOscillatorStep(n, **regime)
    model.init_all_states()
    for _ in range(20):
        V = jnp.asarray(rng.uniform(-3.0, 3.0, size=n))
        W = jnp.asarray(rng.uniform(-3.0, 3.0, size=n))
        coup = jnp.asarray(rng.uniform(-1.0, 1.0, size=n))
        stim_V = jnp.asarray(rng.uniform(-0.5, 0.5, size=n))
        stim_W = jnp.asarray(rng.uniform(-0.5, 0.5, size=n))
        model.V.value, model.W.value = V, W

        dV, dW = model.derivative((V, W), 0.0 * u.ms, coup, stim_V, stim_W)
        ref_dV, ref_dW = _g2d_reference(V, W, coup=coup, stim_V=stim_V, stim_W=stim_W, **regime)
        np.testing.assert_allclose(_mag_per_ms(dV), np.asarray(ref_dV), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(_mag_per_ms(dW), np.asarray(ref_dW), rtol=1e-6, atol=1e-6)


def test_trajectory_matches_reference_rk4(dt):
    regime = _BISTABLE
    V0, W0, n_steps = 0.5, -2.0, 400
    model = brainmass.Generic2dOscillatorStep(
        1,
        init_V=braintools.init.Constant(V0),
        init_W=braintools.init.Constant(W0),
        method="rk4",
        **regime,
    )
    out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["V", "W"], jit=True)
    V_bm = np.asarray(out["V"]).reshape(-1)
    W_bm = np.asarray(out["W"]).reshape(-1)

    dt_ms = u.get_magnitude(dt / u.ms)
    rhs = lambda V, W: _g2d_reference(V, W, coup=0.0, stim_V=0.0, stim_W=0.0, **regime)
    V_ref, W_ref = _rk4_trajectory(rhs, jnp.asarray(V0), jnp.asarray(W0), dt_ms, n_steps)
    V_ref = np.asarray(V_ref).reshape(-1)
    W_ref = np.asarray(W_ref).reshape(-1)

    for a, b in ((V_bm, V_ref), (W_bm, W_ref)):
        corr = np.corrcoef(a, b)[0, 1]
        rel_rmse = np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min())
        assert corr >= 0.99, f"corr={corr}"
        assert rel_rmse <= 0.01, f"rel_rmse={rel_rmse}"


def test_converges_to_analytic_fixed_point(dt):
    """The bistable-nullcline regime relaxes to the analytic nullcline root.

    Fixed points solve V^3 + 2 V^2 - 1 = 0 (with the default e=3, f=1, alpha=1 and
    a=1, b=0, c=-5, beta=1); the global attractor is V* = (-1-sqrt(5))/2 = -1.618...,
    with W* = V*^3 - 3 V*^2. ``d`` is raised from the published 0.02 only to speed
    convergence (``d`` rescales time; the fixed points are unchanged).
    """
    regime = dict(_BISTABLE, d=1.0)
    Vstar = (-1.0 - np.sqrt(5.0)) / 2.0          # -1.6180339...
    Wstar = Vstar ** 3 - 3.0 * Vstar ** 2        # -12.0901699...
    for V0, W0 in ((2.0, -4.0), (-2.0, -4.0), (0.5, 0.0)):
        model = brainmass.Generic2dOscillatorStep(
            1,
            init_V=braintools.init.Constant(V0),
            init_W=braintools.init.Constant(W0),
            **regime,
        )
        out = brainmass.Simulator(model, dt=dt).run(8000 * dt, monitors=["V", "W"])
        Vf = float(np.asarray(out["V"]).reshape(-1)[-1])
        Wf = float(np.asarray(out["W"]).reshape(-1)[-1])
        np.testing.assert_allclose(Vf, Vstar, atol=1e-3)
        np.testing.assert_allclose(Wf, Wstar, atol=1e-2)


def test_excitable_input_integration(dt):
    """Excitable regime: a constant drive settles to a monotonically input-shifted FP.

    ``d`` is raised from the published 0.02 to 0.2 only to speed convergence (it
    rescales time, not the fixed points). At ``d=0.2`` the steady state stays a
    clean fixed point across the drive range tested.
    """
    regime = dict(_EXCITABLE, d=0.2)

    def settle(I):
        model = brainmass.Generic2dOscillatorStep(1, **dict(regime, I=I))
        out = brainmass.Simulator(model, dt=dt).run(8000 * dt, monitors=["V"])
        V = np.asarray(out["V"]).reshape(-1)
        return V[-1], V[-200:].std()

    v0, s0 = settle(0.0)
    v3, s3 = settle(3.0)
    # Converged to a fixed point (no oscillation).
    assert s0 < 1e-3 and s3 < 1e-3, f"std0={s0}, std3={s3}"
    # Stronger drive raises the steady state (input integration).
    assert v3 > v0 + 0.1, f"v0={v0}, v3={v3}"


# ---------------------------------------------------------------------------
# Integrator paths, shapes, units, batching, noise, gradient.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["exp_euler", "rk4"])
def test_integrator_paths_run_bounded(method, dt):
    model = brainmass.Generic2dOscillatorStep(3, method=method)
    out = brainmass.Simulator(model, dt=dt).run(200 * dt, monitors=["V", "W"])
    assert jnp.all(jnp.isfinite(out["V"]))
    assert jnp.all(jnp.isfinite(out["W"]))


def test_shapes_and_units():
    n = 4
    model = brainmass.Generic2dOscillatorStep(n)
    model.init_all_states()
    assert model.V.value.shape == (n,)
    assert model.W.value.shape == (n,)
    dV, dW = model.derivative((model.V.value, model.W.value), 0.0 * u.ms, 0.0)
    assert u.get_unit(dV * u.ms).is_unitless
    assert u.get_unit(dW * u.ms).is_unitless


def test_batched(dt):
    n, b = 3, 5
    model = brainmass.Generic2dOscillatorStep(n)
    out = brainmass.Simulator(model, dt=dt).run(20 * dt, monitors=["V"], batch_size=b)
    assert out["V"].shape[-2:] == (b, n)


def test_update_accepts_explicit_coupling():
    n = 2
    model = brainmass.Generic2dOscillatorStep(n)
    model.init_all_states()
    with brainstate.environ.context(dt=0.1 * u.ms):
        V = model.update(V_inp=0.3)
    assert V.shape == (n,)
    assert jnp.all(jnp.isfinite(V))


def test_noise_changes_trajectory(seeded, dt):
    n = 3
    model = brainmass.Generic2dOscillatorStep(
        n,
        noise_V=brainmass.GaussianNoise(n, sigma=0.1),
        noise_W=brainmass.GaussianNoise(n, sigma=0.1),
    )
    out_noisy = brainmass.Simulator(model, dt=dt).run(50 * dt, monitors=["V", "W"])
    model_clean = brainmass.Generic2dOscillatorStep(n)
    out_clean = brainmass.Simulator(model_clean, dt=dt).run(50 * dt, monitors=["V", "W"])
    assert not jnp.allclose(out_noisy["V"], out_clean["V"])
    assert not jnp.allclose(out_noisy["W"], out_clean["W"])


def test_invalid_args():
    with pytest.raises(AssertionError):
        brainmass.Generic2dOscillatorStep(1, init_V=None)
    with pytest.raises(AssertionError):
        brainmass.Generic2dOscillatorStep(1, noise_V=object())


def test_gradient_ad_vs_fd(dt):
    """AD through Simulator.run wrt the drive I matches a finite-difference estimate."""
    n_steps = 60

    def loss(I_val):
        brainstate.random.seed(0)
        model = brainmass.Generic2dOscillatorStep(1, I=Param(I_val, fit=True), d=0.5)
        out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["V"], jit=False)
        return jnp.sum(u.get_magnitude(out["V"]) ** 2)

    I0 = 1.0
    g_ad = jax.grad(loss)(I0)
    eps = 1e-3
    g_fd = (loss(I0 + eps) - loss(I0 - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(g_ad), np.asarray(g_fd), rtol=2e-2, atol=1e-4)
