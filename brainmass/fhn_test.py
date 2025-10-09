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

import braintools
import brainmass


class TestFitzHughNagumoModel:
    def test_initialization_basic(self):
        # The current implementation asserts callability of noise arguments
        # even when None, so we provide Noise objects to pass validation.
        nV = brainmass.OUProcess(1, sigma=0.01)
        nW = brainmass.OUProcess(1, sigma=0.01)
        m = brainmass.FitzHughNagumoModel(in_size=1, noise_V=nV, noise_w=nW)
        assert m.in_size == (1,)
        assert m.alpha == 3.0
        assert m.beta == 4.0
        assert m.gamma == -1.5
        assert m.delta == 0.0
        assert m.epsilon == 0.5
        # tau carries time unit
        assert u.get_unit(m.tau).dim == u.ms.dim

    def test_state_initialization_and_reset(self):
        nV = brainmass.OUProcess(4, sigma=0.0)
        nW = brainmass.OUProcess(4, sigma=0.0)
        m = brainmass.FitzHughNagumoModel(
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
        m.reset_state(batch_size=2)
        assert u.math.allclose(m.V.value, jnp.zeros((2, 4)))
        assert u.math.allclose(m.w.value, jnp.zeros((2, 4)))

    def test_dv_dw_units_and_finiteness(self):
        nV = brainmass.OUProcess(1, sigma=0.0)
        nW = brainmass.OUProcess(1, sigma=0.0)
        m = brainmass.FitzHughNagumoModel(in_size=1, noise_V=nV, noise_w=nW)

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
        m = brainmass.FitzHughNagumoModel(
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
        m = brainmass.FitzHughNagumoModel(
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
            _ = brainmass.FitzHughNagumoModel(in_size=1, noise_V=object(), noise_w=object())
            assert False, "Expected assertion for invalid noise types"
        except AssertionError:
            pass
