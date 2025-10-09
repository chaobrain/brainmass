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


class TestThresholdLinearModel:
    def test_initialization_defaults(self):
        m = brainmass.ThresholdLinearModel(in_size=1)
        assert m.in_size == (1,)
        assert m.tau_E == 2e-2 * u.second
        assert m.tau_I == 1e-2 * u.second
        assert m.beta_E == 0.066
        assert m.beta_I == 0.351
        assert m.noise_E is None
        assert m.noise_I is None

    def test_state_initialization_and_reset(self):
        m = brainmass.ThresholdLinearModel(
            in_size=4,
            init_E=braintools.init.ZeroInit(),
            init_I=braintools.init.ZeroInit(),
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
        m.reset_state(batch_size=3)
        assert u.math.allclose(m.E.value, jnp.zeros((3, 4)))
        assert u.math.allclose(m.I.value, jnp.zeros((3, 4)))

    def test_update_basic_and_nonnegativity(self):
        m = brainmass.ThresholdLinearModel(
            in_size=2,
            init_E=braintools.init.ZeroInit(),
            init_I=braintools.init.ZeroInit(),
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
        m1 = brainmass.ThresholdLinearModel(
            in_size=1,
            beta_E=0.05,
            init_E=braintools.init.ZeroInit(),
            init_I=braintools.init.ZeroInit(),
        )
        m2 = brainmass.ThresholdLinearModel(
            in_size=1,
            beta_E=0.20,
            init_E=braintools.init.ZeroInit(),
            init_I=braintools.init.ZeroInit(),
        )
        m1.init_state()
        m2.init_state()
        with brainstate.environ.context(dt=0.1 * u.ms):
            e1 = m1.update(E_inp=1.0, I_inp=0.0)
            e2 = m2.update(E_inp=1.0, I_inp=0.0)
        assert e2[()] > e1[()]

    def test_shapes_batch_and_multidimensional(self):
        sz = (2, 3)
        m = brainmass.ThresholdLinearModel(
            in_size=sz,
            init_E=braintools.init.ZeroInit(),
            init_I=braintools.init.ZeroInit(),
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
            _ = brainmass.ThresholdLinearModel(1, noise_E=object())
            assert False, "Expected assertion for invalid noise_E"
        except AssertionError:
            pass
        try:
            _ = brainmass.ThresholdLinearModel(1, noise_I=object())
            assert False, "Expected assertion for invalid noise_I"
        except AssertionError:
            pass
        # Invalid initializers should raise
        try:
            _ = brainmass.ThresholdLinearModel(1, init_E=None)
            assert False, "Expected assertion for invalid init_E"
        except AssertionError:
            pass
        try:
            _ = brainmass.ThresholdLinearModel(1, init_I=None)
            assert False, "Expected assertion for invalid init_I"
        except AssertionError:
            pass
