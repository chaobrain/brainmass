# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brainstate.nn import Param

import brainmass


class TestThresholdLinearModel:
    def test_initialization_defaults(self):
        m = brainmass.ThresholdLinearStep(in_size=1)
        assert m.in_size == (1,)
        assert m.tau_E.val == 2e-2 * u.second
        assert m.tau_I.val == 1e-2 * u.second
        assert m.beta_E.val == 0.066
        assert m.beta_I.val == 0.351
        assert m.noise_E is None
        assert m.noise_I is None

    def test_state_initialization_and_reset(self):
        m = brainmass.ThresholdLinearStep(
            in_size=4,
            init_E=braintools.init.Constant(0.0),
            init_I=braintools.init.Constant(0.0),
        )
        m.init_state()
        assert m.E.value.shape == (4,)
        assert m.I.value.shape == (4,)
        assert u.math.allclose(m.E.value, jnp.zeros((4,)))
        assert u.math.allclose(m.I.value, jnp.zeros((4,)))

        # With batch
        m.init_state(batch_size=3)
        assert m.E.value.shape == (3, 4)
        assert m.I.value.shape == (3, 4)
        assert u.math.allclose(m.E.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.I.value, jnp.zeros((3, 4)))

        # Modify and reset
        m.E.value = jnp.ones((3, 4)) * 0.2
        m.I.value = -jnp.ones((3, 4)) * 0.3
        m.init_state(batch_size=3)
        assert u.math.allclose(m.E.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.I.value, jnp.zeros((3, 4)))

    def test_update_basic_and_nonnegativity(self):
        m = brainmass.ThresholdLinearStep(
            in_size=2,
            init_E=braintools.init.Constant(0.0),
            init_I=braintools.init.Constant(0.0),
        )
        m.init_state()

        with brainstate.environ.context(dt=0.5 * u.ms):
            out = m.update(E_inp=jnp.array([1.0, -1.0]), I_inp=jnp.array([0.0, 0.0]))

        # Shapes
        assert out.shape == (2,)
        assert m.E.value.shape == (2,)
        assert m.I.value.shape == (2,)
        # Nonnegativity enforced
        assert u.math.all(m.E.value >= 0.0)
        assert u.math.all(m.I.value >= 0.0)
        # Positive drive should increase E from zero
        assert m.E.value[0] > 0.0
        # Negative drive is rectified to 0 input; from zero remains nonnegative
        assert m.E.value[1] >= 0.0

    def test_beta_effect_on_gain(self):
        # Larger beta_E should yield larger E update from zero under same positive input
        m1 = brainmass.ThresholdLinearStep(
            in_size=1,
            beta_E=0.05,
            init_E=braintools.init.Constant(0.0),
            init_I=braintools.init.Constant(0.0),
        )
        m2 = brainmass.ThresholdLinearStep(
            in_size=1,
            beta_E=0.20,
            init_E=braintools.init.Constant(0.0),
            init_I=braintools.init.Constant(0.0),
        )
        m1.init_state()
        m2.init_state()
        with brainstate.environ.context(dt=0.1 * u.ms):
            e1 = m1.update(E_inp=1.0, I_inp=0.0)
            e2 = m2.update(E_inp=1.0, I_inp=0.0)
        assert e2[()] > e1[()]

    def test_shapes_batch_and_multidimensional(self):
        sz = (2, 3)
        m = brainmass.ThresholdLinearStep(
            in_size=sz,
            init_E=braintools.init.Constant(0.0),
            init_I=braintools.init.Constant(0.0),
        )
        m.init_state(batch_size=4)
        with brainstate.environ.context(dt=0.1 * u.ms):
            out = m.update(jnp.zeros((4,) + sz), jnp.zeros((4,) + sz))
        assert out.shape == (4,) + sz
        assert m.E.value.shape == (4,) + sz
        assert m.I.value.shape == (4,) + sz

    def test_invalid_noise_and_initializers(self):
        # Invalid noise types should raise
        try:
            _ = brainmass.ThresholdLinearStep(1, noise_E=object())
            assert False, "Expected assertion for invalid noise_E"
        except AssertionError:
            pass
        try:
            _ = brainmass.ThresholdLinearStep(1, noise_I=object())
            assert False, "Expected assertion for invalid noise_I"
        except AssertionError:
            pass
        # Invalid initializers should raise
        try:
            _ = brainmass.ThresholdLinearStep(1, init_E=None)
            assert False, "Expected assertion for invalid init_E"
        except AssertionError:
            pass
        try:
            _ = brainmass.ThresholdLinearStep(1, init_I=None)
            assert False, "Expected assertion for invalid init_I"
        except AssertionError:
            pass

    def test_noise_paths_change_trajectory(self):
        # Exercise the noise_E / noise_I injection branches of update().
        brainstate.random.seed(0)
        n = 3
        m = brainmass.ThresholdLinearStep(
            n,
            noise_E=brainmass.GaussianNoise(n, sigma=0.05),
            noise_I=brainmass.GaussianNoise(n, sigma=0.05),
        )
        m.init_state()
        with brainstate.environ.context(dt=0.1 * u.ms):
            e1 = m.update(E_inp=1.0, I_inp=1.0)
        m_clean = brainmass.ThresholdLinearStep(n)
        m_clean.init_state()
        with brainstate.environ.context(dt=0.1 * u.ms):
            e2 = m_clean.update(E_inp=1.0, I_inp=1.0)
        assert e1.shape == (n,)
        assert not jnp.allclose(e1, e2)


# ===========================================================================
# LinearStep (TVB ``Linear`` model): dx/dt = gamma * x + coupling.
#
# Validation:
# - ``test_linear_rhs_matches_reference`` — the RHS equals the embedded TVB
#   transcription to rtol 1e-6 (equation-level reference regression).
# - ``test_linear_matches_analytic_decay`` — with zero coupling the activity
#   relaxes as the closed-form x(t) = x0 * exp(gamma * t); the (exact for linear
#   systems) exp-Euler step reproduces it (corr >= 0.99, tight RMSE).
# ===========================================================================
def _linear_reference(x, *, gamma, coupling):
    """Reference dx/dt per unit time, transcribed from tvboptim ``Linear``."""
    return gamma * x + coupling


def _mag_per_ms(quantity):
    """Strip the 1/ms unit a brainmass RHS carries -> dimensionless per-time value."""
    return u.get_magnitude(quantity * u.ms)


def test_linear_rhs_matches_reference(seeded):
    n = 6
    rng = np.random.default_rng(0)
    gamma = -7.0
    model = brainmass.LinearStep(n, gamma=gamma)
    model.init_all_states()
    for _ in range(20):
        x = jnp.asarray(rng.uniform(-3.0, 3.0, size=n))
        coup = jnp.asarray(rng.uniform(-2.0, 2.0, size=n))
        model.x.value = x
        (dx,) = model.derivative((x,), 0.0 * u.ms, coup)
        ref = _linear_reference(x, gamma=gamma, coupling=coup)
        np.testing.assert_allclose(_mag_per_ms(dx), np.asarray(ref), rtol=1e-6, atol=1e-6)


def test_linear_matches_analytic_decay(dt):
    """Zero-coupling relaxation matches x(t) = x0 * exp(gamma t) (exp-Euler is exact)."""
    gamma, x0, n_steps = -5.0, 0.8, 200
    model = brainmass.LinearStep(1, gamma=gamma, init_x=braintools.init.Constant(x0))
    out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["x"], jit=True)
    x_bm = np.asarray(out["x"]).reshape(-1)

    dt_ms = u.get_magnitude(dt / u.ms)
    t = (np.arange(n_steps) + 1) * dt_ms  # Simulator records post-step values
    x_ref = x0 * np.exp(gamma * t)

    corr = np.corrcoef(x_bm, x_ref)[0, 1]
    assert corr >= 0.99, f"corr={corr}"
    np.testing.assert_allclose(x_bm, x_ref, rtol=1e-4, atol=1e-6)


def test_linear_converges_to_forced_fixed_point(dt):
    """With constant coupling c, the activity settles to the fixed point -c/gamma."""
    gamma, c = -4.0, 2.0
    model = brainmass.LinearStep(1, gamma=gamma, init_x=braintools.init.Constant(0.0))
    out = brainmass.Simulator(model, dt=dt).run(
        4000 * dt, monitors=["x"], inputs=lambda i, t: c
    )
    x_final = float(np.asarray(out["x"]).reshape(-1)[-1])
    np.testing.assert_allclose(x_final, -c / gamma, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("method", ["exp_euler", "rk4"])
def test_linear_integrator_paths_bounded(method, dt):
    model = brainmass.LinearStep(3, method=method)
    out = brainmass.Simulator(model, dt=dt).run(100 * dt, monitors=["x"])
    assert jnp.all(jnp.isfinite(out["x"]))


def test_linear_shapes_and_units():
    n = 4
    model = brainmass.LinearStep(n)
    model.init_all_states()
    assert model.x.value.shape == (n,)
    (dx,) = model.derivative((model.x.value,), 0.0 * u.ms, 0.0)
    assert u.get_unit(dx * u.ms).is_unitless


def test_linear_batched(dt):
    n, b = 3, 5
    model = brainmass.LinearStep(n)
    out = brainmass.Simulator(model, dt=dt).run(20 * dt, monitors=["x"], batch_size=b)
    assert out["x"].shape[-2:] == (b, n)


def test_linear_noise_changes_trajectory(seeded, dt):
    n = 3
    model = brainmass.LinearStep(n, noise_x=brainmass.GaussianNoise(n, sigma=0.1))
    out_noisy = brainmass.Simulator(model, dt=dt).run(50 * dt, monitors=["x"])
    model_clean = brainmass.LinearStep(n)
    out_clean = brainmass.Simulator(model_clean, dt=dt).run(50 * dt, monitors=["x"])
    assert not jnp.allclose(out_noisy["x"], out_clean["x"])


def test_linear_invalid_args():
    with pytest.raises(AssertionError):
        brainmass.LinearStep(1, init_x=None)
    with pytest.raises(AssertionError):
        brainmass.LinearStep(1, noise_x=object())


def test_linear_gradient_ad_vs_fd(dt):
    """AD through Simulator.run wrt gamma matches a finite-difference estimate."""
    n_steps = 80

    def loss(gamma_val):
        brainstate.random.seed(0)
        model = brainmass.LinearStep(1, gamma=Param(gamma_val, fit=True),
                                     init_x=braintools.init.Constant(1.0))
        out = brainmass.Simulator(model, dt=dt).run(n_steps * dt, monitors=["x"], jit=False)
        return jnp.sum(u.get_magnitude(out["x"]) ** 2)

    g0 = -3.0
    g_ad = jax.grad(loss)(g0)
    eps = 1e-3
    g_fd = (loss(g0 + eps) - loss(g0 - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(g_ad), np.asarray(g_fd), rtol=2e-2, atol=1e-4)
