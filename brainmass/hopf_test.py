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
import jax.numpy as jnp
import numpy as np

import brainmass


class TestHopfModel:
    def test_initialization_basic(self):
        m = brainmass.HopfStep(in_size=1)
        assert m.in_size == (1,)
        assert m.a == 0.25
        assert m.w == 0.2
        assert m.K_gl == 1.0
        assert m.beta == 1.0
        assert m.noise_x is None
        assert m.noise_y is None

    def test_initialization_custom(self):
        m = brainmass.HopfStep(
            in_size=(2, 3),
            a=0.1,
            w=1.5,
            K_gl=0.8,
            beta=2.0,
        )
        assert m.in_size == (2, 3)
        assert m.a == 0.1
        assert m.w == 1.5
        assert m.K_gl == 0.8
        assert m.beta == 2.0

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
        m.reset_state(batch_size=3)
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
