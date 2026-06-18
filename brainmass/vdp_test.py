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
import jax.numpy as jnp
import numpy as np

import brainmass


class TestVanDerPolOscillator:
    def test_initialization_basic(self):
        # XY_Oscillator asserts noise callability even if None; pass Noise
        nx = brainmass.OUProcess(1, sigma=0.0)
        ny = brainmass.OUProcess(1, sigma=0.0)
        m = brainmass.VanDerPolStep(in_size=1, mu=2.0, noise_x=nx, noise_y=ny)
        assert m.in_size == (1,)
        assert m.mu.val == 2.0
        assert m.noise_x is nx
        assert m.noise_y is ny

    def test_state_initialization_and_reset(self):
        nx = brainmass.OUProcess(4, sigma=0.0)
        ny = brainmass.OUProcess(4, sigma=0.0)
        m = brainmass.VanDerPolStep(
            in_size=4,
            init_x=braintools.init.ZeroInit(),
            init_y=braintools.init.ZeroInit(),
            noise_x=nx,
            noise_y=ny,
        )
        m.init_state()
        assert m.x.value.shape == (4,)
        assert m.y.value.shape == (4,)
        assert u.math.allclose(m.x.value, jnp.zeros((4,)))
        assert u.math.allclose(m.y.value, jnp.zeros((4,)))

        # batch
        m.init_state(batch_size=3)
        assert m.x.value.shape == (3, 4)
        assert m.y.value.shape == (3, 4)
        assert u.math.allclose(m.x.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.y.value, jnp.zeros((3, 4)))

        # reset back to zero
        m.x.value = jnp.ones((3, 4))
        m.y.value = -jnp.ones((3, 4))
        m.init_state(batch_size=3)
        assert u.math.allclose(m.x.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.y.value, jnp.zeros((3, 4)))

    def test_dx_dy_units_and_finiteness(self):
        nx = brainmass.OUProcess(1, sigma=0.0)
        ny = brainmass.OUProcess(1, sigma=0.0)
        m = brainmass.VanDerPolStep(in_size=1, mu=1.5, noise_x=nx, noise_y=ny)
        x = jnp.array([0.1])
        y = jnp.array([0.2])
        inp = jnp.array([0.3])
        dx = m.dx(x, y, inp)
        dy = m.dy(y, x, inp)
        assert u.get_unit(dx).dim == (1 / u.ms).dim
        assert u.get_unit(dy).dim == (1 / u.ms).dim
        assert u.math.isfinite(dx).item()
        assert u.math.isfinite(dy).item()

    def test_update_exp_euler_changes_state(self):
        nx = brainmass.OUProcess(2, sigma=0.0)
        ny = brainmass.OUProcess(2, sigma=0.0)
        m = brainmass.VanDerPolStep(
            in_size=2,
            init_x=braintools.init.ZeroInit(),
            init_y=braintools.init.ZeroInit(),
            noise_x=nx,
            noise_y=ny,
            mu=1.0,
            method='exp_euler',
        )
        brainstate.nn.init_all_states(m)
        ex = jnp.array([0.5, -0.2])
        ey = jnp.array([0.0, 0.0])
        with brainstate.environ.context(dt=0.1 * u.ms):
            _ = m.update(ex, ey)
        assert m.x.value.shape == (2,)
        assert m.y.value.shape == (2,)
        assert not u.math.allclose(m.x.value, jnp.zeros((2,)))

    def test_update_rk4_path(self):
        # Exercise non-exp_euler integrator path
        nx = brainmass.OUProcess(2, sigma=0.0)
        ny = brainmass.OUProcess(2, sigma=0.0)
        m = brainmass.VanDerPolStep(
            in_size=2,
            init_x=braintools.init.ZeroInit(),
            init_y=braintools.init.ZeroInit(),
            noise_x=nx,
            noise_y=ny,
            mu=1.0,
            method='rk4',
        )
        brainstate.nn.init_all_states(m)
        with brainstate.environ.context(dt=0.1 * u.ms):
            _ = m.update(jnp.zeros((2,)), jnp.zeros((2,)))
        assert m.x.value.shape == (2,)
        assert m.y.value.shape == (2,)

    def test_derivative_wrapper(self):
        nx = brainmass.OUProcess(1, sigma=0.0)
        ny = brainmass.OUProcess(1, sigma=0.0)
        m = brainmass.VanDerPolStep(in_size=1, noise_x=nx, noise_y=ny)
        x = jnp.array([0.1])
        y = jnp.array([0.0])
        dx, dy = m.derivative((x, y), 0.0, 0.0, 0.0)
        assert dx.shape == (1,)
        assert dy.shape == (1,)
        assert u.get_unit(dx).dim == (1 / u.ms).dim
        assert u.get_unit(dy).dim == (1 / u.ms).dim


# --------------------------------------------------------------------------- #
# Pre-refactor golden trajectory (goal-05 characterization)
# --------------------------------------------------------------------------- #
# Goal-05 re-parented ``VanDerPolStep`` onto the unified ``NeuralMassDynamics``
# base; that refactor had to be behaviour-preserving. This pins a seeded,
# deterministic 20-step ``exp_euler`` trajectory against values captured from
# ``origin/main`` *before* the refactor. The trajectory was bit-identical
# pre/post; ``rtol`` only guards future XLA/platform drift.
_VDP_PRE_REFACTOR_GOLDEN = [
    [0.010361060500144958, 0.008537974208593369, 0.00523275276646018, 0.01725160889327526],
    [0.022566400468349457, 0.017678774893283844, 0.005750805605202913, 0.017678506672382355],
    [0.03735793009400368, 0.028748206794261932, 0.006879125721752644, 0.018562445417046547],
    [0.05516974255442619, 0.04207060858607292, 0.008747022598981857, 0.019999856129288673],
    [0.07649889588356018, 0.05801885947585106, 0.011505509726703167, 0.022103385999798775],
    [0.10191033780574799, 0.07701975852251053, 0.015330454334616661, 0.025004329159855843],
    [0.13203904032707214, 0.09955871105194092, 0.02042597159743309, 0.02885531634092331],
    [0.16758665442466736, 0.1261826753616333, 0.027027923613786697, 0.033833250403404236],
    [0.20930832624435425, 0.15749940276145935, 0.035407256335020065, 0.04014238342642784],
    [0.2579832077026367, 0.1941702961921692, 0.04587267339229584, 0.04801735281944275],
    [0.3143593966960907, 0.23689216375350952, 0.05877183377742767, 0.05772586911916733],
    [0.37906163930892944, 0.2863616645336151, 0.07448980212211609, 0.06957047432661057],
    [0.4524499475955963, 0.3432137966156006, 0.0934428870677948, 0.08388856053352356],
    [0.5344231128692627, 0.4079245924949646, 0.1160653829574585, 0.10104925185441971],
    [0.6241783499717712, 0.4806699752807617, 0.14278653264045715, 0.12144548445940018],
    [0.7199731469154358, 0.5611404180526733, 0.17399545013904572, 0.1454789787530899],
    [0.8189802765846252, 0.6483305096626282, 0.2099941074848175, 0.1735360026359558],
    [0.9173604249954224, 0.740352988243103, 0.250943124294281, 0.20595252513885498],
    [1.010647177696228, 0.834363579750061, 0.29681113362312317, 0.24297016859054565],
    [1.0944160223007202, 0.926695704460144, 0.3473435044288635, 0.2846883535385132],
]


def test_trajectory_matches_pre_refactor_golden():
    """VanDerPolStep reproduces its pre-refactor seeded ``exp_euler`` trajectory."""
    brainstate.environ.set(dt=0.1 * u.ms)
    brainstate.random.seed(0)
    np.random.seed(0)
    model = brainmass.VanDerPolStep(in_size=2, mu=2.0)
    brainstate.nn.init_all_states(model)
    traj = []
    for i in range(20):
        with brainstate.environ.context(i=i, t=i * 0.1 * u.ms):
            out = model.update(0.1, 0.0)
        traj.append(np.asarray(u.get_magnitude(out)).reshape(-1))
    got = np.stack(traj)
    golden = np.asarray(_VDP_PRE_REFACTOR_GOLDEN)
    assert got.shape == golden.shape
    np.testing.assert_allclose(
        got, golden, rtol=1e-5, atol=1e-6,
        err_msg="vdp: trajectory drifted from pre-refactor golden",
    )
