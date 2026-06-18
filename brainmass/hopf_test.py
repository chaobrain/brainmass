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
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brainstate.nn import Param
from jax.test_util import check_grads

import brainmass


class TestHopfModel:
    def test_initialization_basic(self):
        m = brainmass.HopfStep(in_size=1)
        assert m.in_size == (1,)
        assert m.a.val == 0.25
        assert m.w.val == 0.2
        assert m.beta.val == 1.0
        assert m.noise_x is None
        assert m.noise_y is None

    def test_initialization_custom(self):
        m = brainmass.HopfStep(
            in_size=(2, 3),
            a=0.1,
            w=1.5,
            beta=2.0,
        )
        assert m.in_size == (2, 3)
        assert m.a.val == 0.1
        assert m.w.val == 1.5
        assert m.beta.val == 2.0

    def test_state_initialization_and_reset(self):
        m = brainmass.HopfStep(in_size=4)
        m.init_state()
        assert isinstance(m.x, brainstate.HiddenState)
        assert isinstance(m.y, brainstate.HiddenState)
        assert m.x.value.shape == (4,)
        assert m.y.value.shape == (4,)
        assert u.math.allclose(m.x.value, jnp.zeros((4,)))
        assert u.math.allclose(m.y.value, jnp.zeros((4,)))

        # With batch dimension
        m.init_state(batch_size=3)
        assert m.x.value.shape == (3, 4)
        assert m.y.value.shape == (3, 4)
        assert u.math.allclose(m.x.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.y.value, jnp.zeros((3, 4)))

        # Modify and reset
        m.x.value = jnp.ones((3, 4)) * 0.5
        m.y.value = jnp.ones((3, 4)) * -0.2
        m.init_state(batch_size=3)
        assert u.math.allclose(m.x.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.y.value, jnp.zeros((3, 4)))

    def test_dx_dy_units_and_finiteness(self):
        m = brainmass.HopfStep(in_size=1)
        x = jnp.array([0.1])
        y = jnp.array([0.2])
        inp = jnp.array([0.3])

        dx_dt = m.dx(x, y, inp)
        dy_dt = m.dy(y, x, inp)

        # Units are derivatives w.r.t. time, i.e., 1/ms
        assert u.get_unit(dx_dt).dim == (1 / u.ms).dim
        assert u.get_unit(dy_dt).dim == (1 / u.ms).dim
        assert u.math.isfinite(dx_dt).item()
        assert u.math.isfinite(dy_dt).item()

    def test_update_single_step_changes_state(self):
        m = brainmass.HopfStep(in_size=2)
        m.init_state()

        # Finite external drive to x only
        ext_x = jnp.array([0.5, -0.5])

        with brainstate.environ.context(dt=0.1 * u.ms):
            _ = m.update(x_inp=ext_x)

        # Check states updated and have correct shapes
        assert m.x.value.shape == (2,)
        assert m.y.value.shape == (2,)
        # x should move; y should also move due to cross term w * x
        assert not u.math.allclose(m.x.value, jnp.zeros((2,)))

    def test_growth_and_decay_regimes(self):
        # Growth for a > 0
        m1 = brainmass.HopfStep(in_size=1, a=0.2, beta=1.0, w=0.8)
        m1.init_state()
        m1.x.value = jnp.array([1e-2])
        m1.y.value = jnp.array([0.0])

        def step1(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                _ = m1.update(0.0, 0.0)
                return jnp.sqrt(m1.x.value ** 2 + m1.y.value ** 2)

        brainstate.environ.set(dt=0.1 * u.ms)
        r_series1 = brainstate.transform.for_loop(step1, np.arange(2000))
        assert jnp.all(r_series1[-1] > r_series1[0])

        # Decay for a < 0
        m2 = brainmass.HopfStep(in_size=1, a=-0.2, beta=1.0, w=0.8)
        m2.init_state()
        m2.x.value = jnp.array([0.2])
        m2.y.value = jnp.array([0.0])

        def step2(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                _ = m2.update(0.0, 0.0)
                return jnp.sqrt(m2.x.value ** 2 + m2.y.value ** 2)

        r_series2 = brainstate.transform.for_loop(step2, np.arange(2000))
        assert jnp.all(r_series2[-1] < r_series2[0])

    def test_batch_and_multidimensional_update_shapes(self):
        m = brainmass.HopfStep(in_size=(2, 3))
        m.init_state(batch_size=4)

        cx = jnp.zeros((4, 2, 3))
        cy = jnp.zeros((4, 2, 3))

        with brainstate.environ.context(dt=0.05 * u.ms):
            _ = m.update(cx, cy)

        assert m.x.value.shape == (4, 2, 3)
        assert m.y.value.shape == (4, 2, 3)

    def test_noise_assertions_single_side(self):
        # Only one noise provided should raise when updating
        n = brainmass.OUProcess(1, sigma=1.0)

        m_x = brainmass.HopfStep(in_size=1, noise_x=n, noise_y=None)
        brainstate.nn.init_all_states(m_x)
        with brainstate.environ.context(dt=0.1 * u.ms):
            try:
                _ = m_x.update(0.0, 0.0)
                assert False, "Expected assertion when only noise_x is provided"
            except AssertionError:
                pass


# --------------------------------------------------------------------------- #
# Differentiability / gradient tests (migrated from gradient_test.py)
# --------------------------------------------------------------------------- #
# Short horizon keeps the tests fast while still exercising recurrence.
_GRAD_N_STEP = 40
_GRAD_IN_SIZE = 2
_GRAD_SEED = 0

# Differentiate w.r.t. the bifurcation parameter ``a``. ``strong=True``: the
# Hopf gradient is robustly far from zero, so the extra "non-trivial gradient"
# assertion applies.
_HOPF_GRAD_SPEC = dict(
    build=lambda p: brainmass.HopfStep(_GRAD_IN_SIZE, a=Param(p, fit=True)),
    drive=lambda m: m.update(0.1, 0.0),
    read=lambda m: m.x.value,
    p0=0.3,
    strong=True,
)


def _grad_scalar_loss(model, drive, read):
    """Run a short deterministic simulation and return ``sum(read ** 2)``."""
    def step(i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            drive(model)
            return read(model)

    xs = brainstate.transform.for_loop(step, np.arange(_GRAD_N_STEP))
    return jnp.sum(u.get_magnitude(xs) ** 2)


def _grad_pure_loss(spec):
    """Return a pure, seeded ``f(pval) -> scalar`` for a model spec."""
    def f(pval):
        brainstate.random.seed(_GRAD_SEED)
        model = spec["build"](pval)
        brainstate.nn.init_all_states(model)
        return _grad_scalar_loss(model, spec["drive"], spec["read"])

    return f


class TestHopfGradient:
    """A gradient flows through a short for_loop simulation of HopfStep."""

    def test_grad_flows_through_for_loop(self, dt):
        spec = _HOPF_GRAD_SPEC
        brainstate.random.seed(_GRAD_SEED)
        model = spec["build"](jnp.asarray(spec["p0"], dtype=float))
        brainstate.nn.init_all_states(model)

        weights = model.states(brainstate.ParamState)
        assert len(weights) == 1, "exactly one trainable Param expected"

        def loss():
            return _grad_scalar_loss(model, spec["drive"], spec["read"])

        grads, value = brainstate.transform.grad(loss, weights, return_value=True)()
        leaves = jax.tree.leaves(grads)
        assert leaves, "grad tree is empty -- no ParamState was differentiated"
        for g in leaves:
            assert jnp.all(jnp.isfinite(g)), "non-finite gradient"
        assert jnp.isfinite(value)

    def test_value_and_grad(self, dt):
        spec = _HOPF_GRAD_SPEC
        f = _grad_pure_loss(spec)
        value, grad = jax.value_and_grad(f)(jnp.asarray(spec["p0"], dtype=float))
        assert jnp.isfinite(value)
        assert jnp.isfinite(grad)
        if spec["strong"]:
            assert abs(float(grad)) > 1e-3, (
                f"gradient ~0 ({float(grad):.2e}); flow likely blocked"
            )

    def test_ad_matches_finite_difference(self, dt):
        spec = _HOPF_GRAD_SPEC
        f = _grad_pure_loss(spec)
        p0 = float(spec["p0"])
        ad = float(jax.grad(f)(jnp.asarray(p0)))

        eps = 1e-3 * max(1.0, abs(p0))
        fd = (float(f(jnp.asarray(p0 + eps))) - float(f(jnp.asarray(p0 - eps)))) / (2 * eps)

        assert ad == pytest.approx(fd, rel=2e-2, abs=1e-6), (
            f"AD={ad:.6g} disagrees with finite-difference={fd:.6g}"
        )

    def test_check_grads(self, dt):
        spec = _HOPF_GRAD_SPEC
        f = _grad_pure_loss(spec)
        check_grads(f, (jnp.asarray(float(spec["p0"])),), order=1, modes=["rev"],
                    rtol=2e-2, atol=2e-2)


def test_grad_through_stochastic_model_is_deterministic(dt):
    """Gradient through a *stochastic* model is finite and reproducible when seeded.

    Seeding the RNG before building the model makes the noise realisation -- and
    therefore the gradient -- bit-for-bit reproducible, which is what allows
    stochastic models to be fitted with gradient methods at all.
    """
    def grad_once():
        brainstate.random.seed(123)
        noise = brainmass.GaussianNoise(_GRAD_IN_SIZE, sigma=0.3)  # unitless to match Hopf x
        model = brainmass.HopfStep(_GRAD_IN_SIZE, a=Param(jnp.asarray(0.3), fit=True),
                                   noise_x=noise)
        brainstate.nn.init_all_states(model)
        weights = model.states(brainstate.ParamState)

        def loss():
            return _grad_scalar_loss(model, lambda m: m.update(0.1, 0.0), lambda m: m.x.value)

        grads, value = brainstate.transform.grad(loss, weights, return_value=True)()
        return float(jax.tree.leaves(grads)[0]), float(value)

    g1, v1 = grad_once()
    g2, v2 = grad_once()
    assert np.isfinite(g1) and np.isfinite(v1)
    assert g1 == g2, "seeded stochastic gradient is not reproducible"
    assert v1 == v2


def test_vmap_batched_gradients(dt):
    """Batched (``vmap``) gradients over a sweep of parameter values are finite."""
    f = _grad_pure_loss(_HOPF_GRAD_SPEC)
    p_values = jnp.array([0.1, 0.2, 0.3, 0.4])
    batched = jax.vmap(jax.grad(f))(p_values)
    assert batched.shape == (4,)
    assert jnp.all(jnp.isfinite(batched))
    # Different operating points should give different gradients.
    assert float(jnp.std(batched)) > 1e-3


def test_zero_noise_limit_recovers_deterministic_gradient(dt):
    """With zero-amplitude noise the stochastic path equals the noise-free path."""
    def grad_with(noise):
        brainstate.random.seed(7)
        model = brainmass.HopfStep(_GRAD_IN_SIZE, a=Param(jnp.asarray(0.3), fit=True),
                                   noise_x=noise)
        brainstate.nn.init_all_states(model)
        weights = model.states(brainstate.ParamState)
        grads = brainstate.transform.grad(
            lambda: _grad_scalar_loss(model, lambda m: m.update(0.1, 0.0),
                                      lambda m: m.x.value),
            weights,
        )()
        return float(jax.tree.leaves(grads)[0])

    g_zero_noise = grad_with(brainmass.GaussianNoise(_GRAD_IN_SIZE, sigma=0.0))
    g_no_noise = grad_with(None)
    assert g_zero_noise == pytest.approx(g_no_noise, rel=1e-5, abs=1e-6)


# --------------------------------------------------------------------------- #
# Pre-refactor golden trajectory (goal-05 characterization)
# --------------------------------------------------------------------------- #
# Goal-05 re-parented ``HopfStep`` (the ``XY_Oscillator`` family) onto the
# unified ``NeuralMassDynamics`` base; that refactor had to be behaviour-
# preserving. This pins a seeded, deterministic 20-step ``exp_euler`` trajectory
# against values captured from ``origin/main`` *before* the refactor. The
# trajectory was bit-identical pre/post; ``rtol`` only guards future XLA/platform
# drift.
_HOPF_PRE_REFACTOR_GOLDEN = [
    [0.009900662116706371, 0.009900662116706371, 0.0, 0.0],
    [0.019605040550231934, 0.019605040550231934, 0.00019604526460170746, 0.00019604526460170746],
    [0.029112322255969048, 0.029112322255969048, 0.0005803543026559055, 0.0005803543026559055],
    [0.03842140734195709, 0.03842140734195709, 0.0011452524922788143, 0.0011452524922788143],
    [0.04753096401691437, 0.04753096401691437, 0.0018831479828804731, 0.0018831479828804731],
    [0.05643950402736664, 0.05643950402736664, 0.0027865113224834204, 0.0027865113224834204],
    [0.06514541804790497, 0.06514541804790497, 0.0038478609640151262, 0.0038478609640151262],
    [0.07364705204963684, 0.07364705204963684, 0.0050597526133060455, 0.0050597526133060455],
    [0.08194273710250854, 0.08194273710250854, 0.0064147706143558025, 0.0064147706143558025],
    [0.09003085643053055, 0.09003085643053055, 0.007905526086688042, 0.007905526086688042],
    [0.09790986776351929, 0.09790986776351929, 0.009524653665721416, 0.009524653665721416],
    [0.10557834804058075, 0.10557834804058075, 0.011264817789196968, 0.011264817789196968],
    [0.11303503811359406, 0.11303503811359406, 0.013118712231516838, 0.013118712231516838],
    [0.12027886509895325, 0.12027886509895325, 0.015079069882631302, 0.015079069882631302],
    [0.12730897963047028, 0.12730897963047028, 0.01713867299258709, 0.01713867299258709],
    [0.1341247707605362, 0.1341247707605362, 0.01929035782814026, 0.01929035782814026],
    [0.14072589576244354, 0.14072589576244354, 0.021527033299207687, 0.021527033299207687],
    [0.14711228013038635, 0.14711228013038635, 0.02384168654680252, 0.02384168654680252],
    [0.15328416228294373, 0.15328416228294373, 0.026227403432130814, 0.026227403432130814],
    [0.15924207866191864, 0.15924207866191864, 0.028677375987172127, 0.028677375987172127],
]


def test_trajectory_matches_pre_refactor_golden():
    """HopfStep reproduces its pre-refactor seeded ``exp_euler`` trajectory."""
    brainstate.environ.set(dt=0.1 * u.ms)
    brainstate.random.seed(0)
    np.random.seed(0)
    model = brainmass.HopfStep(2, a=-0.2)
    brainstate.nn.init_all_states(model)
    traj = []
    for i in range(20):
        with brainstate.environ.context(i=i, t=i * 0.1 * u.ms):
            out = model.update(0.1, 0.0)
        traj.append(np.asarray(u.get_magnitude(out)).reshape(-1))
    got = np.stack(traj)
    golden = np.asarray(_HOPF_PRE_REFACTOR_GOLDEN)
    assert got.shape == golden.shape
    np.testing.assert_allclose(
        got, golden, rtol=1e-5, atol=1e-6,
        err_msg="hopf: trajectory drifted from pre-refactor golden",
    )
