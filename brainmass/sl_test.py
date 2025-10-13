# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

import brainmass
import braintools


class TestStuartLandauOscillator:
    def test_initialization_basic(self):
        # XY_Oscillator asserts callability of noise args, so pass Noise objects
        nx = brainmass.OUProcess(1, sigma=0.0)
        ny = brainmass.OUProcess(1, sigma=0.0)
        m = brainmass.StuartLandauOscillator(in_size=1, noise_x=nx, noise_y=ny)
        assert m.in_size == (1,)
        assert m.a == 0.25
        assert m.w == 0.2
        assert m.noise_x is nx
        assert m.noise_y is ny

    def test_state_initialization_and_reset(self):
        nx = brainmass.OUProcess(4, sigma=0.0)
        ny = brainmass.OUProcess(4, sigma=0.0)
        m = brainmass.StuartLandauOscillator(
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
        m.reset_state(batch_size=3)
        assert u.math.allclose(m.x.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.y.value, jnp.zeros((3, 4)))

    def test_dx_dy_units_and_finiteness(self):
        nx = brainmass.OUProcess(1, sigma=0.0)
        ny = brainmass.OUProcess(1, sigma=0.0)
        m = brainmass.StuartLandauOscillator(in_size=1, noise_x=nx, noise_y=ny)
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
        m = brainmass.StuartLandauOscillator(
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
        m = brainmass.StuartLandauOscillator(
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
