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


class TestQIFModel:
    def test_initialization_basic(self):
        m = brainmass.MontbrioPazoRoxinStep(in_size=1)
        assert m.in_size == (1,)
        assert m.tau.value() == 1.0 * u.ms
        assert m.eta.val == -5.0
        assert m.delta.val == 1.0 * u.Hz
        assert m.J.val == 15.0
        assert m.noise_r is None
        assert m.noise_v is None

    def test_custom_parameters(self):
        m = brainmass.MontbrioPazoRoxinStep(
            in_size=(2, 3),
            tau=2.0 * u.ms,
            eta=-3.0,
            delta=0.5 * u.Hz,
            J=12.0,
        )
        assert m.in_size == (2, 3)
        assert m.tau.val == 2.0 * u.ms
        assert m.eta.val == -3.0
        assert m.delta.val == 0.5 * u.Hz
        assert m.J.val == 12.0

    def test_state_initialization_and_reset(self):
        m = brainmass.MontbrioPazoRoxinStep(
            in_size=4,
            init_r=braintools.init.Constant(0.0),
            init_v=braintools.init.Constant(0.0),
        )

        # init without batch
        m.init_state()
        assert isinstance(m.r, brainstate.HiddenState)
        assert isinstance(m.v, brainstate.HiddenState)
        assert m.r.value.shape == (4,)
        assert m.v.value.shape == (4,)
        assert u.math.allclose(m.r.value, jnp.zeros((4,)))
        assert u.math.allclose(m.v.value, jnp.zeros((4,)))

        # with batch
        m.init_state(batch_size=3)
        assert m.r.value.shape == (3, 4)
        assert m.v.value.shape == (3, 4)
        assert u.math.allclose(m.r.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.v.value, jnp.zeros((3, 4)))

        # modify and reset
        m.r.value = jnp.ones((3, 4)) * 0.1
        m.v.value = jnp.ones((3, 4)) * -0.2
        m.init_state(batch_size=3)
        assert u.math.allclose(m.r.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.v.value, jnp.zeros((3, 4)))

    def test_derivative_units_and_finiteness(self):
        m = brainmass.MontbrioPazoRoxinStep(in_size=1)
        r = jnp.array([0.05]) * u.Hz
        v = jnp.array([0.1])
        rex = jnp.array([0.0]) * u.Hz
        vex = jnp.array([0.2])

        dr_dt = m.dr(r, v, rex)
        dv_dt = m.dv(v, r, vex)

        assert u.get_unit(dr_dt).dim == (u.Hz / u.ms).dim
        assert u.get_unit(dv_dt).dim == (1 / u.ms).dim
        assert u.math.isfinite(dr_dt).item()
        assert u.math.isfinite(dv_dt).item()

    def test_update_single_step_changes_state(self):
        m = brainmass.MontbrioPazoRoxinStep(
            in_size=2,
            init_r=braintools.init.Constant(0.0 * u.Hz),
            init_v=braintools.init.Constant(0.0),
        )
        m.init_state()

        r_inp = jnp.array([0.0, 0.0]) * u.Hz
        v_inp = jnp.array([1.0, -1.0])

        with brainstate.environ.context(dt=0.1 * u.ms):
            r_next = m.update(r_inp=r_inp, v_inp=v_inp)

        assert r_next.shape == (2,)
        assert m.r.value.shape == (2,)
        assert m.v.value.shape == (2,)
        # Expect change from zeros under nonzero v input
        assert not u.math.allclose(m.r.value, jnp.zeros((2,)) * u.Hz)

    def test_batch_and_multidimensional_update_shapes(self):
        sz = (2, 3)
        m = brainmass.MontbrioPazoRoxinStep(
            in_size=sz,
            init_r=braintools.init.Constant(0.0 * u.Hz),
            init_v=braintools.init.Constant(0.0),
        )
        m.init_state(batch_size=4)

        r_inp = jnp.zeros((4,) + sz) * u.Hz
        v_inp = jnp.zeros((4,) + sz)

        with brainstate.environ.context(dt=0.05 * u.ms):
            _ = m.update(r_inp, v_inp)

        assert m.r.value.shape == (4,) + sz
        assert m.v.value.shape == (4,) + sz

    def test_parameter_arrays(self):
        # Provide an array tau to ensure broadcasting works per element
        tau_arr = jnp.ones((3,)) * (2.0 * u.ms)
        m = brainmass.MontbrioPazoRoxinStep(in_size=3, tau=tau_arr)
        m.init_state()
        with brainstate.environ.context(dt=0.1 * u.ms):
            out = m.update()
        assert out.shape == (3,)


# --------------------------------------------------------------------------- #
# Differentiability / gradient tests (migrated from gradient_test.py)
# --------------------------------------------------------------------------- #
_GRAD_N_STEP = 40
_GRAD_IN_SIZE = 2
_GRAD_SEED = 0

# Differentiate w.r.t. the excitability ``eta`` while driving the membrane
# potential, so the population is in an active regime where the parameter
# genuinely influences the firing rate. ``strong=True``: the gradient is then
# robustly far from zero.
_MONTBRIO_GRAD_SPEC = dict(
    build=lambda p: brainmass.MontbrioPazoRoxinStep(_GRAD_IN_SIZE, eta=Param(p, fit=True)),
    drive=lambda m: m.update(v_inp=1.0),
    read=lambda m: m.r.value,
    p0=-5.0,
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


class TestMontbrioGradient:
    """A gradient flows through a short for_loop simulation of MontbrioPazoRoxinStep."""

    def test_grad_flows_through_for_loop(self, dt):
        spec = _MONTBRIO_GRAD_SPEC
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
        spec = _MONTBRIO_GRAD_SPEC
        f = _grad_pure_loss(spec)
        value, grad = jax.value_and_grad(f)(jnp.asarray(spec["p0"], dtype=float))
        assert jnp.isfinite(value)
        assert jnp.isfinite(grad)
        if spec["strong"]:
            assert abs(float(grad)) > 1e-3, (
                f"gradient ~0 ({float(grad):.2e}); flow likely blocked"
            )

    def test_ad_matches_finite_difference(self, dt):
        spec = _MONTBRIO_GRAD_SPEC
        f = _grad_pure_loss(spec)
        p0 = float(spec["p0"])
        ad = float(jax.grad(f)(jnp.asarray(p0)))

        eps = 1e-3 * max(1.0, abs(p0))
        fd = (float(f(jnp.asarray(p0 + eps))) - float(f(jnp.asarray(p0 - eps)))) / (2 * eps)

        assert ad == pytest.approx(fd, rel=2e-2, abs=1e-6), (
            f"AD={ad:.6g} disagrees with finite-difference={fd:.6g}"
        )

    def test_check_grads(self, dt):
        spec = _MONTBRIO_GRAD_SPEC
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


def test_montbrio_noise_paths_are_seeded(dt):
    """Both noise channels of MontbrioPazoRoxin run and are reproducible."""
    a, b = _reproducible_noise_run(
        lambda: brainmass.MontbrioPazoRoxinStep(
            2,
            noise_r=brainmass.GaussianNoise(2, sigma=1.0 * u.Hz),
            noise_v=brainmass.GaussianNoise(2, sigma=0.1)))
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# Pre-refactor golden trajectory (goal-05 characterization)
# --------------------------------------------------------------------------- #
# Goal-05 re-parented ``MontbrioPazoRoxinStep`` onto the unified
# ``NeuralMassDynamics`` base; that refactor had to be behaviour-preserving. This
# pins a seeded, deterministic 20-step ``exp_euler`` trajectory against values
# captured from ``origin/main`` *before* the refactor. The trajectory was
# bit-identical pre/post; ``rtol`` only guards future XLA/platform drift.
_MONTBRIO_PRE_REFACTOR_GOLDEN = [
    [0.03183098882436752, 0.03183098882436752],
    [0.05953194200992584, 0.05953194200992584],
    [0.0793565958738327, 0.0793565958738327],
    [0.09063003212213516, 0.09063003212213516],
    [0.09501630812883377, 0.09501630812883377],
    [0.09497888386249542, 0.09497888386249542],
    [0.09267318993806839, 0.09267318993806839],
    [0.08955812454223633, 0.08955812454223633],
    [0.08645164966583252, 0.08645164966583252],
    [0.08372881263494492, 0.08372881263494492],
    [0.0815066397190094, 0.0815066397190094],
    [0.0797722265124321, 0.0797722265124321],
    [0.07845857739448547, 0.07845857739448547],
    [0.07748459279537201, 0.07748459279537201],
    [0.0767737329006195, 0.0767737329006195],
    [0.07626111805438995, 0.07626111805438995],
    [0.07589495182037354, 0.07589495182037354],
    [0.0756353884935379, 0.0756353884935379],
    [0.07545255124568939, 0.07545255124568939],
    [0.07532444596290588, 0.07532444596290588],
]


def test_trajectory_matches_pre_refactor_golden():
    """MontbrioPazoRoxinStep reproduces its pre-refactor seeded ``exp_euler`` trajectory."""
    brainstate.environ.set(dt=0.1 * u.ms)
    brainstate.random.seed(0)
    np.random.seed(0)
    model = brainmass.MontbrioPazoRoxinStep(
        2,
        init_r=braintools.init.Constant(0.0 * u.Hz),
        init_v=braintools.init.Constant(0.0),
    )
    brainstate.nn.init_all_states(model)
    traj = []
    for i in range(20):
        with brainstate.environ.context(i=i, t=i * 0.1 * u.ms):
            out = model.update(v_inp=0.5)
        traj.append(np.asarray(u.get_magnitude(out)).reshape(-1))
    got = np.stack(traj)
    golden = np.asarray(_MONTBRIO_PRE_REFACTOR_GOLDEN)
    assert got.shape == golden.shape
    np.testing.assert_allclose(
        got, golden, rtol=1e-5, atol=1e-6,
        err_msg="montbrio: trajectory drifted from pre-refactor golden",
    )
