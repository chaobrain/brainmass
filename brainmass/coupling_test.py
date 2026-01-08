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
import pytest

import brainmass


class _Holder(brainstate.nn.Dynamics):
    """Minimal dynamics container exposing attributes for Prefetch.

    We do not implement update logic; we only host attributes 'x' and 'y'
    that Prefetch will read via getattr.
    """

    def __init__(self):
        # Provide a dummy in_size to satisfy Dynamics constructor
        super().__init__(in_size=(1,))
        self.x = None
        self.y = None


class TestDiffusiveCoupling:
    def test_basic_2d_conn(self):
        n_out, n_in = 2, 3
        holder = _Holder()
        holder.x = jnp.array([[2.0, 4.0, 6.0],
                              [1.0, 3.0, 5.0]])
        holder.y = jnp.array([1.0, 2.0])

        x_ref = brainstate.nn.Prefetch(holder, 'x')
        y_ref = brainstate.nn.Prefetch(holder, 'y')
        conn = jnp.ones((n_out, n_in))
        k = 0.5

        coup = brainmass.DiffusiveCoupling(x_ref, y_ref, conn=conn, k=k)
        coup.init_state()
        out = coup.update()

        # Expected: k * sum_j (x_ij - y_i)
        exp = k * jnp.array([
            jnp.sum(holder.x[0] - holder.y[0]),
            jnp.sum(holder.x[1] - holder.y[1]),
        ])
        assert out.shape == (n_out,)
        assert u.math.allclose(out, exp)

    def test_basic_1d_conn_flat(self):
        n_out, n_in = 2, 3
        holder = _Holder()
        holder.x = jnp.array([[2.0, 4.0, 6.0],
                              [1.0, 3.0, 5.0]])
        holder.y = jnp.array([1.0, 2.0])

        x_ref = brainstate.nn.Prefetch(holder, 'x')
        y_ref = brainstate.nn.Prefetch(holder, 'y')
        conn = jnp.ones((n_out, n_in)).reshape(-1)  # flattened

        coup = brainmass.DiffusiveCoupling(x_ref, y_ref, conn=conn, k=1.0)
        coup.init_state()
        out = coup.update()

        exp = jnp.array([
            jnp.sum(holder.x[0] - holder.y[0]),
            jnp.sum(holder.x[1] - holder.y[1]),
        ])
        assert u.math.allclose(out, exp)

    def test_batch_support(self):
        batch, n_out, n_in = 4, 2, 3
        holder = _Holder()
        x_single = jnp.array([[1.0, 0.0, -1.0],
                              [2.0, 2.0, 2.0]])
        y_single = jnp.array([0.5, 1.5])
        holder.x = jnp.broadcast_to(x_single, (batch, n_out, n_in))
        holder.y = jnp.broadcast_to(y_single, (batch, n_out))

        x_ref = brainstate.nn.Prefetch(holder, 'x')
        y_ref = brainstate.nn.Prefetch(holder, 'y')
        conn = jnp.ones((n_out, n_in))

        coup = brainmass.DiffusiveCoupling(x_ref, y_ref, conn=conn, k=1.0)
        coup.init_state()
        out = coup.update()

        # Expected computed per batch equals the single computation broadcast
        row0 = jnp.sum(x_single[0] - y_single[0])
        row1 = jnp.sum(x_single[1] - y_single[1])
        exp = jnp.stack([jnp.array([row0, row1])] * batch)
        assert out.shape == (batch, n_out)
        assert u.math.allclose(out, exp)

    def test_type_checking(self):
        holder = _Holder()
        holder.x = jnp.array([1.0, 2.0])
        holder.y = jnp.array([0.5, 1.5])
        y_ref = brainstate.nn.Prefetch(holder, 'y')

        with pytest.raises(TypeError):
            brainmass.DiffusiveCoupling(holder.x, y_ref, conn=jnp.ones((2, 1)))

    def test_shape_validation_errors(self):
        n_out, n_in = 2, 3
        holder = _Holder()
        # x flattened with wrong size (not divisible by n_out)
        holder.x = jnp.arange(7.0)  # wrong
        holder.y = jnp.array([0.0, 0.0])
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        y_ref = brainstate.nn.Prefetch(holder, 'y')
        conn = jnp.ones((n_out, n_in))

        coup = brainmass.DiffusiveCoupling(x_ref, y_ref, conn=conn)
        coup.init_state()
        with pytest.raises(ValueError):
            _ = coup.update()


class TestAdditiveCoupling:
    def test_basic_2d_conn(self):
        n_out, n_in = 2, 3
        holder = _Holder()
        holder.x = jnp.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]])

        x_ref = brainstate.nn.Prefetch(holder, 'x')
        conn = jnp.array([[1.0, 0.0, 1.0],
                          [0.0, 1.0, 1.0]])
        k = 2.0

        coup = brainmass.AdditiveCoupling(x_ref, conn=conn, k=k)
        coup.init_state()
        out = coup.update()

        exp = k * jnp.array([
            jnp.sum(conn[0] * holder.x[0]),
            jnp.sum(conn[1] * holder.x[1]),
        ])
        assert out.shape == (n_out,)
        assert u.math.allclose(out, exp)

    def test_flattened_x(self):
        n_out, n_in = 2, 3
        holder = _Holder()
        holder.x = jnp.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]]).reshape(-1)  # flattened
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        conn = jnp.array([[1.0, 0.0, 1.0],
                          [0.0, 1.0, 1.0]])

        coup = brainmass.AdditiveCoupling(x_ref, conn=conn, k=1.0)
        coup.init_state()
        out = coup.update()

        exp = jnp.array([
            jnp.sum(conn[0] * jnp.array([1.0, 2.0, 3.0])),
            jnp.sum(conn[1] * jnp.array([4.0, 5.0, 6.0])),
        ])
        assert u.math.allclose(out, exp)

    def test_batch_support(self):
        batch, n_out, n_in = 5, 2, 3
        holder = _Holder()
        x_single = jnp.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]])
        holder.x = jnp.broadcast_to(x_single, (batch, n_out, n_in))
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        conn = jnp.array([[1.0, 1.0, 0.0],
                          [0.5, 0.5, 0.5]])

        coup = brainmass.AdditiveCoupling(x_ref, conn=conn, k=1.0)
        coup.init_state()
        out = coup.update()

        row0 = jnp.sum(conn[0] * x_single[0])
        row1 = jnp.sum(conn[1] * x_single[1])
        exp = jnp.stack([jnp.array([row0, row1])] * batch)
        assert out.shape == (batch, n_out)
        assert u.math.allclose(out, exp)

    def test_init_validation(self):
        holder = _Holder()
        holder.x = jnp.array([1.0, 2.0, 3.0])
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        with pytest.raises(ValueError):
            # AdditiveCoupling only supports 2D connection matrix
            brainmass.AdditiveCoupling(x_ref, conn=jnp.ones(3))

    def test_shape_error(self):
        n_out, n_in = 2, 3
        holder = _Holder()
        # Wrong x trailing shape
        holder.x = jnp.ones((n_out, n_in + 1))
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        conn = jnp.ones((n_out, n_in))

        coup = brainmass.AdditiveCoupling(x_ref, conn=conn)
        coup.init_state()
        with pytest.raises(ValueError):
            _ = coup.update()
