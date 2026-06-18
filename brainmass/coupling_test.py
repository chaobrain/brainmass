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


class TestLaplacianConnParam:
    """LaplacianConnParam: precompute method must not collide with the mode attr."""

    @staticmethod
    def _W():
        return jnp.array([[0., 1., 1.],
                          [1., 0., 1.],
                          [1., 1., 0.]])

    def test_normalization_method_renamed_no_collision(self):
        # The precompute callback was renamed to ``_laplacian``; ``normalize`` is
        # the mode string. On the buggy version ``_laplacian`` does not exist.
        p = brainmass.LaplacianConnParam(self._W(), normalize='sym')
        assert callable(getattr(p, '_laplacian', None))
        assert p.normalize == 'sym'

    @pytest.mark.parametrize('mode', [None, 'rw', 'sym'])
    def test_value_matches_reference(self, mode):
        W = self._W()
        p = brainmass.LaplacianConnParam(W, normalize=mode)
        got = p.value()
        exp = brainmass.laplacian_connectivity(u.math.exp(W) * W, normalize=mode)
        assert u.math.allclose(got, exp)

    def test_value_with_mask(self):
        W = self._W()
        mask = jnp.array([[0., 1., 0.],
                          [1., 0., 1.],
                          [0., 1., 0.]])
        p = brainmass.LaplacianConnParam(W, normalize='sym', mask=mask)
        got = p.value()
        exp = brainmass.laplacian_connectivity(u.math.exp(W) * W * mask, normalize='sym')
        assert u.math.allclose(got, exp)

    def test_return_diag_returns_tuple(self):
        W = self._W()
        p = brainmass.LaplacianConnParam(W, normalize='rw', return_diag=True)
        L, d = p.value()
        expL, expd = brainmass.laplacian_connectivity(
            u.math.exp(W) * W, normalize='rw', return_diag=True)
        assert u.math.allclose(L, expL)
        assert u.math.allclose(d, expd)

    def test_mask_shape_mismatch_raises(self):
        W = self._W()
        with pytest.raises(ValueError):
            brainmass.LaplacianConnParam(W, mask=jnp.ones((2, 2)))


class TestSharedRecurrentConn:
    """Direct tests for the deduplicated AdditiveConn / DelayedAdditiveConn.

    These classes were unified into coupling.py from horn.py + jansen_rit.py.
    They read a named state via ``u.get_magnitude`` so they work both on a
    unitless state (HORN 'y') and a unit-carrying state (Jansen 'M').
    """

    def test_additive_conn_reads_named_state_and_update_tr_zero(self):
        import brainstate
        from brainmass.coupling import AdditiveConn
        brainstate.environ.set(dt=0.1 * u.ms)
        model = brainmass.HORNStep(4)
        brainstate.nn.init_all_states(model)
        conn = AdditiveConn(model, state='y')
        brainstate.nn.init_all_states(conn)
        out = conn.update()
        assert out.shape == (4,)
        assert np.all(np.isfinite(np.asarray(out)))
        # update_tr is a no-op contributing zero for the non-delayed conn.
        assert float(np.asarray(conn.update_tr())) == 0.0

    def test_delayed_additive_conn_update_tr_zero(self):
        import brainstate
        from brainmass.coupling import DelayedAdditiveConn
        brainstate.environ.set(dt=0.1 * u.ms)
        model = brainmass.HORNStep(4)
        brainstate.nn.init_all_states(model)
        np.random.seed(0)  # deterministic delay matrix
        delay = jnp.asarray(np.random.randint(1, 4, size=(4, 4)).astype('float32')) * u.ms
        conn = DelayedAdditiveConn(model, delay_time=delay, state='y')
        brainstate.nn.init_all_states(conn)
        assert float(np.asarray(conn.update_tr())) == 0.0
        out = conn.update()
        assert out.shape == (4,)
        assert np.all(np.isfinite(np.asarray(out)))
