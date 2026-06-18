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


class TestStuartLandauOscillator:
    def test_initialization_basic(self):
        # XY_Oscillator asserts callability of noise args, so pass Noise objects
        nx = brainmass.OUProcess(1, sigma=0.0)
        ny = brainmass.OUProcess(1, sigma=0.0)
        m = brainmass.StuartLandauStep(in_size=1, noise_x=nx, noise_y=ny)
        assert m.in_size == (1,)
        assert m.a.val == 0.25
        assert m.w.val == 0.2
        assert m.noise_x is nx
        assert m.noise_y is ny

    def test_state_initialization_and_reset(self):
        nx = brainmass.OUProcess(4, sigma=0.0)
        ny = brainmass.OUProcess(4, sigma=0.0)
        m = brainmass.StuartLandauStep(
            in_size=4,
            init_x=braintools.init.ZeroInit(),
            init_y=braintools.init.ZeroInit(),
            noise_x=nx,
            noise_y=ny,
        )
        m.init_state()
        assert isinstance(m.x, brainstate.HiddenState)
        assert isinstance(m.y, brainstate.HiddenState)
        assert m.x.value.shape == (4,)
        assert m.y.value.shape == (4,)
        assert u.math.allclose(m.x.value, jnp.zeros((4,)))
        assert u.math.allclose(m.y.value, jnp.zeros((4,)))

        # With batch
        m.init_state(batch_size=3)
        assert m.x.value.shape == (3, 4)
        assert m.y.value.shape == (3, 4)
        assert u.math.allclose(m.x.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.y.value, jnp.zeros((3, 4)))

        # Modify and reset
        m.x.value = jnp.ones((3, 4)) * 0.1
        m.y.value = jnp.ones((3, 4)) * -0.2
        m.init_state(batch_size=3)
        assert u.math.allclose(m.x.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.y.value, jnp.zeros((3, 4)))

    def test_dx_dy_units_and_finiteness(self):
        nx = brainmass.OUProcess(1, sigma=0.0)
        ny = brainmass.OUProcess(1, sigma=0.0)
        m = brainmass.StuartLandauStep(in_size=1, noise_x=nx, noise_y=ny)
        x = jnp.array([0.1])
        y = jnp.array([0.2])
        inp = jnp.array([0.3])
        dx = m.dx(x, y, inp)
        dy = m.dy(y, x, inp)
        assert u.get_unit(dx).dim == (1 / u.ms).dim
        assert u.get_unit(dy).dim == (1 / u.ms).dim
        assert u.math.isfinite(dx).item()
        assert u.math.isfinite(dy).item()

    def test_update_single_step_changes_x(self):
        # Only test that x changes under x input; y change depends on specific form
        nx = brainmass.OUProcess(2, sigma=0.0)
        ny = brainmass.OUProcess(2, sigma=0.0)
        m = brainmass.StuartLandauStep(
            in_size=2,
            init_x=braintools.init.ZeroInit(),
            init_y=braintools.init.ZeroInit(),
            noise_x=nx,
            noise_y=ny,
        )
        brainstate.nn.init_all_states(m)
        ext_x = jnp.array([0.5, -0.5])
        ext_y = jnp.array([0.0, 0.0])

        with brainstate.environ.context(dt=0.1 * u.ms):
            _ = m.update(ext_x, ext_y)

        assert m.x.value.shape == (2,)
        assert m.y.value.shape == (2,)
        assert not u.math.allclose(m.x.value, jnp.zeros((2,)))

    def test_batch_and_multidimensional_update_shapes(self):
        sz = (2, 3)
        nx = brainmass.OUProcess(sz, sigma=0.0)
        ny = brainmass.OUProcess(sz, sigma=0.0)
        m = brainmass.StuartLandauStep(
            in_size=sz,
            init_x=braintools.init.ZeroInit(),
            init_y=braintools.init.ZeroInit(),
            noise_x=nx,
            noise_y=ny,
        )
        brainstate.nn.init_all_states(m, batch_size=4)
        ex = jnp.zeros((4,) + sz)
        ey = jnp.zeros((4,) + sz)
        with brainstate.environ.context(dt=0.05 * u.ms):
            _ = m.update(ex, ey)
        assert m.x.value.shape == (4,) + sz
        assert m.y.value.shape == (4,) + sz


# --------------------------------------------------------------------------- #
# Pre-refactor golden trajectory (goal-05 characterization)
# --------------------------------------------------------------------------- #
# Goal-05 re-parented ``StuartLandauStep`` onto the unified ``NeuralMassDynamics``
# base; that refactor had to be behaviour-preserving. This pins a seeded,
# deterministic 20-step ``exp_euler`` trajectory against values captured from
# ``origin/main`` *before* the refactor. The trajectory was bit-identical
# pre/post; ``rtol`` only guards future XLA/platform drift.
_SL_PRE_REFACTOR_GOLDEN = [
    [0.010394351556897163, 0.010848547331988811, 0.005353895481675863, 0.017655402421951294],
    [0.020674780011177063, 0.020890861749649048, 0.00569985993206501, 0.01832125335931778],
    [0.031207121908664703, 0.031172238290309906, 0.006262579932808876, 0.019206644967198372],
    [0.041991546750068665, 0.041692472994327545, 0.007052441593259573, 0.020321443676948547],
    [0.05302712321281433, 0.05245029926300049, 0.008079998195171356, 0.02167561464011669],
    [0.06431165337562561, 0.06344323605298996, 0.009355908259749413, 0.023279158398509026],
    [0.07584153860807419, 0.07466745376586914, 0.010890859179198742, 0.025142038241028786],
    [0.0876116082072258, 0.08611762523651123, 0.012695485725998878, 0.027274081483483315],
    [0.0996149331331253, 0.09778675436973572, 0.014780262485146523, 0.0296848863363266],
    [0.11184269189834595, 0.10966604948043823, 0.017155397683382034, 0.03238370269536972],
    [0.12428397685289383, 0.12174473702907562, 0.019830696284770966, 0.035379309207201004],
    [0.13692563772201538, 0.1340099275112152, 0.02281542494893074, 0.03867986425757408],
    [0.1497521549463272, 0.14644649624824524, 0.026118144392967224, 0.0422927588224411],
    [0.16274546086788177, 0.15903691947460175, 0.02974654920399189, 0.0462244488298893],
    [0.17588485777378082, 0.17176122963428497, 0.033707279711961746, 0.05048029124736786],
    [0.18914695084095, 0.18459689617156982, 0.03800573572516441, 0.05506434664130211],
    [0.20250558853149414, 0.19751882553100586, 0.042645882815122604, 0.059979211539030075],
    [0.21593187749385834, 0.2104993313550949, 0.04763004928827286, 0.06522583216428757],
    [0.22939422726631165, 0.22350826859474182, 0.05295874923467636, 0.07080332934856415],
    [0.24285849928855896, 0.23651304841041565, 0.05863048881292343, 0.0767088383436203],
]


def test_trajectory_matches_pre_refactor_golden():
    """StuartLandauStep reproduces its pre-refactor seeded ``exp_euler`` trajectory."""
    brainstate.environ.set(dt=0.1 * u.ms)
    brainstate.random.seed(0)
    np.random.seed(0)
    model = brainmass.StuartLandauStep(in_size=2)
    brainstate.nn.init_all_states(model)
    traj = []
    for i in range(20):
        with brainstate.environ.context(i=i, t=i * 0.1 * u.ms):
            out = model.update(0.1, 0.0)
        traj.append(np.asarray(u.get_magnitude(out)).reshape(-1))
    got = np.stack(traj)
    golden = np.asarray(_SL_PRE_REFACTOR_GOLDEN)
    assert got.shape == golden.shape
    np.testing.assert_allclose(
        got, golden, rtol=1e-5, atol=1e-6,
        err_msg="sl: trajectory drifted from pre-refactor golden",
    )
