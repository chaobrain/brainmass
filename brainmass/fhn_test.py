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
from jax.test_util import check_grads

import brainmass


class TestFitzHughNagumoModel:
    def test_initialization_basic(self):
        # The current implementation asserts callability of noise arguments
        # even when None, so we provide Noise objects to pass validation.
        nV = brainmass.OUProcess(1, sigma=0.01)
        nW = brainmass.OUProcess(1, sigma=0.01)
        m = brainmass.FitzHughNagumoStep(in_size=1, noise_V=nV, noise_w=nW)
        assert m.in_size == (1,)
        assert m.alpha.val == 3.0
        assert m.beta.val == 4.0
        assert m.gamma.val == -1.5
        assert m.delta.val == 0.0
        assert m.epsilon.val == 0.5
        # tau carries time unit
        assert u.get_unit(m.tau.val).dim == u.ms.dim

    def test_state_initialization_and_reset(self):
        nV = brainmass.OUProcess(4, sigma=0.0)
        nW = brainmass.OUProcess(4, sigma=0.0)
        m = brainmass.FitzHughNagumoStep(
            in_size=4,
            init_V=braintools.init.ZeroInit(),
            init_w=braintools.init.ZeroInit(),
            noise_V=nV,
            noise_w=nW,
        )

        m.init_state()
        assert isinstance(m.V, brainstate.HiddenState)
        assert isinstance(m.w, brainstate.HiddenState)
        assert m.V.value.shape == (4,)
        assert m.w.value.shape == (4,)
        assert u.math.allclose(m.V.value, jnp.zeros((4,)))
        assert u.math.allclose(m.w.value, jnp.zeros((4,)))

        # With batch
        m.init_state(batch_size=2)
        assert m.V.value.shape == (2, 4)
        assert m.w.value.shape == (2, 4)
        assert u.math.allclose(m.V.value, jnp.zeros((2, 4)))
        assert u.math.allclose(m.w.value, jnp.zeros((2, 4)))

        # Modify and reset
        m.V.value = jnp.ones((2, 4)) * 0.3
        m.w.value = jnp.ones((2, 4)) * -0.1
        m.init_state(batch_size=2)
        assert u.math.allclose(m.V.value, jnp.zeros((2, 4)))
        assert u.math.allclose(m.w.value, jnp.zeros((2, 4)))

    def test_dv_dw_units_and_finiteness(self):
        nV = brainmass.OUProcess(1, sigma=0.0)
        nW = brainmass.OUProcess(1, sigma=0.0)
        m = brainmass.FitzHughNagumoStep(in_size=1, noise_V=nV, noise_w=nW)

        V = jnp.array([0.1])
        w = jnp.array([0.2])
        inp = jnp.array([0.3])

        dV_dt = m.dV(V, w, inp)
        dw_dt = m.dw(w, V, inp)

        assert u.get_unit(dV_dt).dim == (1 / u.ms).dim
        assert u.get_unit(dw_dt).dim == (1 / u.ms).dim
        assert u.math.isfinite(dV_dt).item()
        assert u.math.isfinite(dw_dt).item()

    def test_update_single_step_changes_state(self):
        nV = brainmass.OUProcess(2, sigma=0.0)
        nW = brainmass.OUProcess(2, sigma=0.0)
        m = brainmass.FitzHughNagumoStep(
            in_size=2,
            init_V=braintools.init.ZeroInit(),
            init_w=braintools.init.ZeroInit(),
            noise_V=nV,
            noise_w=nW,
        )
        brainstate.nn.init_all_states(m)

        V_inp = jnp.array([0.5, -0.5])
        w_inp = jnp.array([0.0, 0.0])

        with brainstate.environ.context(dt=0.1 * u.ms):
            V_next = m.update(V_inp=V_inp, w_inp=w_inp)

        assert V_next.shape == (2,)
        assert m.V.value.shape == (2,)
        assert m.w.value.shape == (2,)
        # Expect some change from zero initial conditions under nonzero input
        assert not u.math.allclose(m.V.value, jnp.zeros((2,)))

    def test_batch_and_multidimensional_update_shapes(self):
        n = (2, 3)
        nV = brainmass.OUProcess(n, sigma=0.0)
        nW = brainmass.OUProcess(n, sigma=0.0)
        m = brainmass.FitzHughNagumoStep(
            in_size=n,
            init_V=braintools.init.ZeroInit(),
            init_w=braintools.init.ZeroInit(),
            noise_V=nV,
            noise_w=nW,
        )
        brainstate.nn.init_all_states(m, batch_size=4)

        V_inp = jnp.zeros((4,) + n)
        w_inp = jnp.zeros((4,) + n)

        with brainstate.environ.context(dt=0.05 * u.ms):
            _ = m.update(V_inp, w_inp)

        assert m.V.value.shape == (4,) + n
        assert m.w.value.shape == (4,) + n

    def test_invalid_noise_type_raises(self):
        # Not a Noise instance
        try:
            _ = brainmass.FitzHughNagumoStep(in_size=1, noise_V=object(), noise_w=object())
            assert False, "Expected assertion for invalid noise types"
        except AssertionError:
            pass


# --------------------------------------------------------------------------- #
# Differentiability / gradient tests (migrated from gradient_test.py)
# --------------------------------------------------------------------------- #
_GRAD_N_STEP = 40
_GRAD_IN_SIZE = 2
_GRAD_SEED = 0

# Differentiate w.r.t. the cubic coefficient ``alpha``. ``strong=True``: the
# FitzHugh-Nagumo gradient is robustly far from zero.
_FHN_GRAD_SPEC = dict(
    build=lambda p: brainmass.FitzHughNagumoStep(_GRAD_IN_SIZE, alpha=Param(p, fit=True)),
    drive=lambda m: m.update(V_inp=0.5),
    read=lambda m: m.V.value,
    p0=3.0,
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


class TestFitzHughNagumoGradient:
    """A gradient flows through a short for_loop simulation of FitzHughNagumoStep."""

    def test_grad_flows_through_for_loop(self, dt):
        spec = _FHN_GRAD_SPEC
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
        spec = _FHN_GRAD_SPEC
        f = _grad_pure_loss(spec)
        value, grad = jax.value_and_grad(f)(jnp.asarray(spec["p0"], dtype=float))
        assert jnp.isfinite(value)
        assert jnp.isfinite(grad)
        if spec["strong"]:
            assert abs(float(grad)) > 1e-3, (
                f"gradient ~0 ({float(grad):.2e}); flow likely blocked"
            )

    def test_ad_matches_finite_difference(self, dt):
        spec = _FHN_GRAD_SPEC
        f = _grad_pure_loss(spec)
        p0 = float(spec["p0"])
        ad = float(jax.grad(f)(jnp.asarray(p0)))

        eps = 1e-3 * max(1.0, abs(p0))
        fd = (float(f(jnp.asarray(p0 + eps))) - float(f(jnp.asarray(p0 - eps)))) / (2 * eps)

        assert ad == pytest.approx(fd, rel=2e-2, abs=1e-6), (
            f"AD={ad:.6g} disagrees with finite-difference={fd:.6g}"
        )

    def test_check_grads(self, dt):
        spec = _FHN_GRAD_SPEC
        f = _grad_pure_loss(spec)
        check_grads(f, (jnp.asarray(float(spec["p0"])),), order=1, modes=["rev"],
                    rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------- #
# Noise-injection branch coverage (migrated from branch_coverage_test.py)
# --------------------------------------------------------------------------- #
def _reproducible_noise_run(make_model, seed=0, n_steps=20):
    """Run a noisy model twice under a fixed seed and return both trajectories."""
    def once():
        brainstate.random.seed(seed)
        model = make_model()
        brainstate.nn.init_all_states(model)

        def step(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update()

        return np.asarray(u.get_magnitude(
            brainstate.transform.for_loop(step, np.arange(n_steps))))

    return once(), once()


def test_fhn_noise_paths_are_seeded(dt):
    """Both noise channels of FitzHugh-Nagumo run and are reproducible."""
    a, b = _reproducible_noise_run(
        lambda: brainmass.FitzHughNagumoStep(
            2,
            noise_V=brainmass.GaussianNoise(2, sigma=0.1),
            noise_w=brainmass.GaussianNoise(2, sigma=0.1)))
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Pre-refactor golden trajectory (goal-05 characterization)
# --------------------------------------------------------------------------- #
# Goal-05 re-parented ``FitzHughNagumoStep`` onto the unified
# ``NeuralMassDynamics`` base; that refactor had to be behaviour-preserving. This
# pins a seeded, deterministic 20-step ``exp_euler`` trajectory against values
# captured from ``origin/main`` *before* the refactor. The trajectory was
# bit-identical pre/post; ``rtol`` only guards future XLA/platform drift.
_FHN_PRE_REFACTOR_GOLDEN = [
    [0.027692005038261414, 0.027171337977051735],
    [0.0517396405339241, 0.05015530437231064],
    [0.07325849682092667, 0.07070139050483704],
    [0.09278653562068939, 0.08931697905063629],
    [0.11072073876857758, 0.10637781769037247],
    [0.1273615062236786, 0.12216896563768387],
    [0.1429411619901657, 0.13691124320030212],
    [0.1576429009437561, 0.1507789045572281],
    [0.17161361873149872, 0.16391165554523468],
    [0.18497298657894135, 0.17642317712306976],
    [0.19781982898712158, 0.18840722739696503],
    [0.21023687720298767, 0.19994203746318817],
    [0.2222941815853119, 0.21109361946582794],
    [0.23405176401138306, 0.2219182550907135],
    [0.24556156992912292, 0.23246443271636963],
    [0.25686901807785034, 0.24277424812316895],
    [0.26801419258117676, 0.2528845965862274],
    [0.27903270721435547, 0.2628280520439148],
    [0.2899564802646637, 0.2726336121559143],
    [0.30081433057785034, 0.28232723474502563],
]


def test_trajectory_matches_pre_refactor_golden():
    """FitzHughNagumoStep reproduces its pre-refactor seeded ``exp_euler`` trajectory."""
    brainstate.environ.set(dt=0.1 * u.ms)
    brainstate.random.seed(0)
    np.random.seed(0)
    model = brainmass.FitzHughNagumoStep(2)
    brainstate.nn.init_all_states(model)
    traj = []
    for i in range(20):
        with brainstate.environ.context(i=i, t=i * 0.1 * u.ms):
            out = model.update(V_inp=0.3)
        traj.append(np.asarray(u.get_magnitude(out)).reshape(-1))
    got = np.stack(traj)
    golden = np.asarray(_FHN_PRE_REFACTOR_GOLDEN)
    assert got.shape == golden.shape
    np.testing.assert_allclose(
        got, golden, rtol=1e-5, atol=1e-6,
        err_msg="fhn: trajectory drifted from pre-refactor golden",
    )
