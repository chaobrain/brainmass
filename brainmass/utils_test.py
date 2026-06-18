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

import jax.numpy as jnp
import numpy as np
import pytest

import brainunit as u

from brainmass.utils import (
    sys2nd, sigmoid, bounded_input, process_sequence, set_module_as, delay_index,
)


class TestSys2nd:
    def test_acceleration_formula(self):
        # dv/dt = A*a*u - 2*a*v - a**2 * x
        A, a, inp, x, v = 2.0, 3.0, 1.0, 0.5, 0.25
        out = sys2nd(A, a, inp, x, v)
        assert np.isclose(out, A * a * inp - 2 * a * v - a ** 2 * x)


class TestSigmoid:
    def test_half_max_at_threshold(self):
        # At x == v0 the argument is 0 -> sigmoid(0) = 0.5 -> vmax/2.
        out = sigmoid(jnp.asarray(1.0), vmax=10.0, v0=1.0, r=0.5)
        assert np.isclose(float(out), 5.0)

    def test_monotonic_and_bounded(self):
        xs = jnp.linspace(-10, 10, 50)
        ys = np.asarray(sigmoid(xs, vmax=5.0, v0=0.0, r=1.0))
        assert np.all(ys > 0) and np.all(ys < 5.0)
        assert np.all(np.diff(ys) >= -1e-6)  # non-decreasing


class TestBoundedInput:
    def test_saturates_for_large_input(self):
        out = bounded_input(jnp.asarray(1e6), bound=500.0)
        assert np.isclose(float(out), 500.0, atol=1e-3)

    def test_approx_linear_for_small_input(self):
        out = bounded_input(jnp.asarray(1.0), bound=500.0)
        assert np.isclose(float(out), 1.0, atol=1e-2)


class TestProcessSequence:
    def setup_method(self):
        self.data = jnp.asarray(np.arange(12, dtype='float32').reshape(4, 3))

    def test_stack_identity(self):
        out = process_sequence(self.data, mode='stack')
        assert out.shape == (4, 3)
        assert np.allclose(np.asarray(out), np.asarray(self.data))

    def test_first_and_last(self):
        assert np.allclose(np.asarray(process_sequence(self.data, 'first')), np.asarray(self.data[0]))
        assert np.allclose(np.asarray(process_sequence(self.data, 'last')), np.asarray(self.data[-1]))

    @pytest.mark.parametrize('mode', ['avg', 'mean'])
    def test_mean(self, mode):
        out = process_sequence(self.data, mode)
        assert np.allclose(np.asarray(out), np.asarray(self.data).mean(axis=0))

    def test_max_min(self):
        assert np.allclose(np.asarray(process_sequence(self.data, 'max')), np.asarray(self.data).max(axis=0))
        assert np.allclose(np.asarray(process_sequence(self.data, 'min')), np.asarray(self.data).min(axis=0))

    def test_callable_mode(self):
        out = process_sequence(self.data, mode=lambda x: x.sum(axis=0))
        assert np.allclose(np.asarray(out), np.asarray(self.data).sum(axis=0))

    def test_pytree_dict(self):
        tree = {'a': self.data, 'b': self.data + 1}
        out = process_sequence(tree, 'mean')
        assert set(out.keys()) == {'a', 'b'}
        assert np.allclose(np.asarray(out['a']), np.asarray(self.data).mean(axis=0))

    def test_unknown_string_mode_raises(self):
        with pytest.raises(ValueError):
            process_sequence(self.data, mode='nonsense')

    def test_non_str_non_callable_mode_raises(self):
        with pytest.raises(ValueError):
            process_sequence(self.data, mode=123)


class TestSetModuleAs:
    def test_sets_dunder_module(self):
        @set_module_as('brainmass.somewhere')
        def fn():
            return 1

        assert fn.__module__ == 'brainmass.somewhere'
        assert fn() == 1


class TestDelayIndex:
    def test_shape_and_rows(self):
        idx = delay_index(4)
        assert idx.shape == (4, 4)
        # every row is arange(n)
        for row in idx:
            assert list(row) == [0, 1, 2, 3]
