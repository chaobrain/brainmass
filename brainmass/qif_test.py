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


class TestQIFModel:
    def test_initialization_basic(self):
        m = brainmass.QIF(in_size=1)
        assert m.in_size == (1,)
        assert m.tau == 1.0 * u.ms
        assert m.eta == -5.0
        assert m.delta == 1.0 * u.Hz
        assert m.J == 15.0
        assert m.noise_r is None
        assert m.noise_v is None

    def test_custom_parameters(self):
        m = brainmass.QIF(
            in_size=(2, 3),
            tau=2.0 * u.ms,
            eta=-3.0,
            delta=0.5 * u.Hz,
            J=12.0,
        )
        assert m.in_size == (2, 3)
        assert m.tau == 2.0 * u.ms
        assert m.eta == -3.0
        assert m.delta == 0.5 * u.Hz
        assert m.J == 12.0

    def test_state_initialization_and_reset(self):
        m = brainmass.QIF(
            in_size=4,
            init_r=braintools.init.ZeroInit(),
            init_v=braintools.init.ZeroInit(),
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
        m.reset_state(batch_size=3)
        assert u.math.allclose(m.r.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.v.value, jnp.zeros((3, 4)))

    def test_derivative_units_and_finiteness(self):
        m = brainmass.QIF(in_size=1)
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
        m = brainmass.QIF(
            in_size=2,
            init_r=braintools.init.ZeroInit(unit=u.Hz),
            init_v=braintools.init.ZeroInit(),
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
        m = brainmass.QIF(
            in_size=sz,
            init_r=braintools.init.ZeroInit(unit=u.Hz),
            init_v=braintools.init.ZeroInit(),
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
        m = brainmass.QIF(in_size=3, tau=tau_arr)
        m.init_state()
        with brainstate.environ.context(dt=0.1 * u.ms):
            out = m.update()
        assert out.shape == (3,)
