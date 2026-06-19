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
import jax
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


# ---------------------------------------------------------------------------
# goal-11: nonlinear couplings (parity with tvboptim) + additive Linear bias
# ---------------------------------------------------------------------------


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class TestAdditiveBias:
    """Linear bias ``b`` added to additive coupling (back-compat: b=0 == old)."""

    def test_kernel_b_zero_matches_old(self):
        # Default b=0 must reproduce the bias-free additive kernel bit-for-bit.
        conn = jnp.array([[1.0, 0.0, 1.0],
                          [0.0, 1.0, 1.0]])
        x = jnp.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
        out0 = brainmass.additive_coupling(x, conn, k=2.0)
        out_b0 = brainmass.additive_coupling(x, conn, k=2.0, b=0.0)
        assert np.array_equal(np.asarray(out0), np.asarray(out_b0))

    def test_kernel_b_shifts(self):
        conn = jnp.array([[1.0, 1.0],
                          [1.0, 1.0]])
        x = jnp.array([[1.0, 2.0],
                       [3.0, 4.0]])
        base = brainmass.additive_coupling(x, conn, k=1.0)
        shifted = brainmass.additive_coupling(x, conn, k=1.0, b=0.5)
        assert u.math.allclose(shifted, base + 0.5)

    def test_module_bias(self):
        holder = _Holder()
        holder.x = jnp.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]])
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        conn = jnp.array([[1.0, 0.0, 1.0],
                          [0.0, 1.0, 1.0]])
        coup = brainmass.AdditiveCoupling(x_ref, conn=conn, k=2.0, b=0.3)
        coup.init_state()
        out = coup.update()
        exp = 2.0 * jnp.array([jnp.sum(conn[0] * holder.x[0]),
                               jnp.sum(conn[1] * holder.x[1])]) + 0.3
        assert u.math.allclose(out, exp)

    def test_module_default_bias_backcompat(self):
        # AdditiveCoupling with no b must equal the old behaviour exactly.
        holder = _Holder()
        holder.x = jnp.array([[1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]])
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        conn = jnp.array([[1.0, 0.0, 1.0],
                          [0.0, 1.0, 1.0]])
        coup = brainmass.AdditiveCoupling(x_ref, conn=conn, k=2.0)
        coup.init_state()
        out = coup.update()
        exp = 2.0 * jnp.array([jnp.sum(conn[0] * holder.x[0]),
                               jnp.sum(conn[1] * holder.x[1])])
        assert np.array_equal(np.asarray(out), np.asarray(exp))

    def test_unitful_default_bias_preserves_units(self):
        # Unit-carrying additive coupling must still work with the default bias.
        conn = jnp.array([[1.0, 1.0],
                          [1.0, 1.0]])
        x = jnp.array([[1.0, 2.0],
                       [3.0, 4.0]]) * u.nA
        out = brainmass.additive_coupling(x, conn, k=1.0)
        assert u.get_unit(out) == u.nA
        assert u.math.allclose(out, jnp.array([3.0, 7.0]) * u.nA)


class TestSigmoidalCoupling:
    """TVB Sigmoidal: post-nonlinearity c = k*sigma(slope*(a*sum + b - midpoint))."""

    def test_closed_form_zero_input(self):
        # Zero net input -> k*sigma(-slope*midpoint) (with a=1, b=0).
        conn = jnp.ones((2, 2))
        x = jnp.zeros((2, 2))
        k, slope, midpoint = 0.7, 2.0, 3.0
        out = brainmass.sigmoidal_coupling(
            x, conn, k=k, slope=slope, midpoint=midpoint)
        exp = k * _sigmoid(-slope * midpoint)
        assert u.math.allclose(out, jnp.full((2,), exp), atol=1e-6)

    def test_kernel_value(self):
        conn = jnp.ones((2, 2))
        x = jnp.array([[1.0, 1.0],
                       [2.0, 2.0]])  # row sums -> [2, 4]
        out = brainmass.sigmoidal_coupling(x, conn, k=1.0, a=1.0, b=0.0,
                                           slope=1.0, midpoint=0.0)
        exp = _sigmoid(np.array([2.0, 4.0]))
        assert u.math.allclose(out, jnp.asarray(exp), atol=1e-6)

    def test_module_matches_kernel(self):
        holder = _Holder()
        holder.x = jnp.array([[0.5, 1.5],
                              [-1.0, 2.0]])
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        conn = jnp.array([[1.0, 0.5],
                          [0.5, 1.0]])
        coup = brainmass.SigmoidalCoupling(x_ref, conn=conn, k=1.3, a=0.8,
                                           b=0.1, slope=1.5, midpoint=0.2)
        coup.init_state()
        out = coup.update()
        exp = brainmass.sigmoidal_coupling(holder.x, conn, k=1.3, a=0.8,
                                           b=0.1, slope=1.5, midpoint=0.2)
        assert out.shape == (2,)
        assert u.math.allclose(out, exp)

    def test_saturation_bounded(self):
        # Sigmoid saturates in (0, k); huge |input| -> no NaN/Inf.
        conn = jnp.ones((2, 2))
        x_pos = jnp.full((2, 2), 1e6)
        x_neg = jnp.full((2, 2), -1e6)
        k = 2.0
        out_pos = brainmass.sigmoidal_coupling(x_pos, conn, k=k, slope=1.0)
        out_neg = brainmass.sigmoidal_coupling(x_neg, conn, k=k, slope=1.0)
        assert np.all(np.isfinite(np.asarray(out_pos)))
        assert np.all(np.isfinite(np.asarray(out_neg)))
        assert u.math.allclose(out_pos, jnp.full((2,), k), atol=1e-5)
        assert u.math.allclose(out_neg, jnp.zeros((2,)), atol=1e-5)

    def test_batched(self):
        batch = 4
        conn = jnp.ones((2, 3))
        x_single = jnp.array([[1.0, 0.0, -1.0],
                              [2.0, 2.0, 2.0]])
        x = jnp.broadcast_to(x_single, (batch, 2, 3))
        out = brainmass.sigmoidal_coupling(x, conn, k=1.0)
        exp_single = brainmass.sigmoidal_coupling(x_single, conn, k=1.0)
        assert out.shape == (batch, 2)
        assert u.math.allclose(out, jnp.broadcast_to(exp_single, (batch, 2)))

    def test_flattened_conn_square(self):
        conn2d = jnp.array([[1.0, 0.5],
                            [0.5, 1.0]])
        conn1d = conn2d.reshape(-1)
        x = jnp.array([[1.0, 2.0],
                       [3.0, 4.0]])
        out2d = brainmass.sigmoidal_coupling(x, conn2d, k=1.0)
        out1d = brainmass.sigmoidal_coupling(x, conn1d, k=1.0)
        assert u.math.allclose(out2d, out1d)

    def test_units_carry_k(self):
        conn = jnp.ones((2, 2))
        x = jnp.array([[1.0, 1.0],
                       [2.0, 2.0]])
        out = brainmass.sigmoidal_coupling(x, conn, k=1.0 * u.nA)
        assert u.get_unit(out) == u.nA


class TestHyperbolicTangentCoupling:
    """TVB HyperbolicTangent: post-nonlinearity c = k*tanh(scale*sum)."""

    def test_saturation_pm_k(self):
        conn = jnp.ones((2, 2))
        k = 0.5
        out_pos = brainmass.hyperbolic_tangent_coupling(
            jnp.full((2, 2), 1e6), conn, k=k, scale=2.0)
        out_neg = brainmass.hyperbolic_tangent_coupling(
            jnp.full((2, 2), -1e6), conn, k=k, scale=2.0)
        assert u.math.allclose(out_pos, jnp.full((2,), k), atol=1e-6)
        assert u.math.allclose(out_neg, jnp.full((2,), -k), atol=1e-6)
        assert np.all(np.isfinite(np.asarray(out_pos)))

    def test_kernel_value(self):
        conn = jnp.ones((2, 2))
        x = jnp.array([[0.1, 0.2],
                       [0.3, 0.4]])  # row sums -> [0.3, 0.7]
        out = brainmass.hyperbolic_tangent_coupling(x, conn, k=0.5, scale=2.0)
        exp = 0.5 * np.tanh(2.0 * np.array([0.3, 0.7]))
        assert u.math.allclose(out, jnp.asarray(exp), atol=1e-6)

    def test_defaults(self):
        # Documented defaults k=0.5, scale=2.0.
        conn = jnp.ones((2, 2))
        x = jnp.array([[0.1, 0.2],
                       [0.3, 0.4]])
        out_default = brainmass.hyperbolic_tangent_coupling(x, conn)
        out_explicit = brainmass.hyperbolic_tangent_coupling(x, conn, k=0.5, scale=2.0)
        assert u.math.allclose(out_default, out_explicit)

    def test_module_matches_kernel(self):
        holder = _Holder()
        holder.x = jnp.array([[0.5, -0.5],
                              [1.0, 0.2]])
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        conn = jnp.array([[1.0, 0.5],
                          [0.5, 1.0]])
        coup = brainmass.HyperbolicTangentCoupling(x_ref, conn=conn, k=0.5, scale=2.0)
        coup.init_state()
        out = coup.update()
        exp = brainmass.hyperbolic_tangent_coupling(holder.x, conn, k=0.5, scale=2.0)
        assert out.shape == (2,)
        assert u.math.allclose(out, exp)

    def test_batched_and_flattened_conn(self):
        batch = 3
        conn2d = jnp.array([[1.0, 0.5],
                            [0.5, 1.0]])
        x_single = jnp.array([[0.2, 0.4],
                              [0.6, 0.8]])
        x = jnp.broadcast_to(x_single, (batch, 2, 2))
        out = brainmass.hyperbolic_tangent_coupling(x, conn2d.reshape(-1), k=0.5)
        exp_single = brainmass.hyperbolic_tangent_coupling(x_single, conn2d, k=0.5)
        assert out.shape == (batch, 2)
        assert u.math.allclose(out, jnp.broadcast_to(exp_single, (batch, 2)))


class TestSigmoidalJansenRitCoupling:
    """TVB SigmoidalJansenRit: pre-nonlinearity c = k*sum_j w_ij*sigma_JR(x_j)."""

    @staticmethod
    def _sigma_jr(x, cmin, cmax, midpoint, r):
        return cmin + (cmax - cmin) / (1.0 + np.exp(r * (midpoint - x)))

    def test_value_at_midpoint(self):
        # At x = midpoint, sigma_JR = (cmin + cmax) / 2.
        cmin, cmax, midpoint, r = 0.0, 0.005, 6.0, 0.56
        conn = jnp.array([[0.2, 0.3],
                          [0.5, 0.1]])
        x = jnp.full((2, 2), midpoint)
        out = brainmass.sigmoidal_jansen_rit_coupling(
            x, conn, k=1.0, cmin=cmin, cmax=cmax, midpoint=midpoint, r=r)
        exp = 1.0 * conn.sum(axis=1) * (cmin + cmax) / 2.0
        assert u.math.allclose(out, exp, atol=1e-9)

    def test_far_limits(self):
        cmin, cmax, midpoint, r = 0.0, 0.005, 6.0, 0.56
        conn = jnp.array([[1.0, 1.0],
                          [1.0, 1.0]])
        # Far above midpoint -> cmax; far below -> cmin.
        x_hi = jnp.full((2, 2), 1e3)
        x_lo = jnp.full((2, 2), -1e3)
        out_hi = brainmass.sigmoidal_jansen_rit_coupling(
            x_hi, conn, cmin=cmin, cmax=cmax, midpoint=midpoint, r=r)
        out_lo = brainmass.sigmoidal_jansen_rit_coupling(
            x_lo, conn, cmin=cmin, cmax=cmax, midpoint=midpoint, r=r)
        assert u.math.allclose(out_hi, conn.sum(axis=1) * cmax, atol=1e-9)
        assert u.math.allclose(out_lo, conn.sum(axis=1) * cmin, atol=1e-9)

    def test_kernel_value(self):
        cmin, cmax, midpoint, r = 0.0, 0.005, 6.0, 0.56
        conn = jnp.array([[0.2, 0.3],
                          [0.5, 0.1]])
        x = jnp.array([[5.0, 6.0],
                       [7.0, 4.0]])
        out = brainmass.sigmoidal_jansen_rit_coupling(
            x, conn, k=2.0, cmin=cmin, cmax=cmax, midpoint=midpoint, r=r)
        s = self._sigma_jr(np.asarray(x), cmin, cmax, midpoint, r)
        exp = 2.0 * np.sum(np.asarray(conn) * s, axis=1)
        assert u.math.allclose(out, jnp.asarray(exp), atol=1e-9)

    def test_module_matches_kernel(self):
        holder = _Holder()
        holder.x = jnp.array([[5.0, 6.0],
                              [7.0, 4.0]])
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        conn = jnp.array([[0.2, 0.3],
                          [0.5, 0.1]])
        coup = brainmass.SigmoidalJansenRitCoupling(
            x_ref, conn=conn, k=2.0, cmax=0.005, midpoint=6.0, r=0.56)
        coup.init_state()
        out = coup.update()
        exp = brainmass.sigmoidal_jansen_rit_coupling(
            holder.x, conn, k=2.0, cmax=0.005, midpoint=6.0, r=0.56)
        assert out.shape == (2,)
        assert u.math.allclose(out, exp)

    def test_batched_and_flattened_conn(self):
        batch = 3
        conn2d = jnp.array([[0.2, 0.3],
                            [0.5, 0.1]])
        x_single = jnp.array([[5.0, 6.0],
                              [7.0, 4.0]])
        x = jnp.broadcast_to(x_single, (batch, 2, 2))
        out = brainmass.sigmoidal_jansen_rit_coupling(x, conn2d.reshape(-1))
        exp_single = brainmass.sigmoidal_jansen_rit_coupling(x_single, conn2d)
        assert out.shape == (batch, 2)
        assert u.math.allclose(out, jnp.broadcast_to(exp_single, (batch, 2)))


class TestCouplingShapeValidation:
    def test_sigmoidal_bad_x_shape_raises(self):
        conn = jnp.ones((2, 3))
        x = jnp.ones((2, 4))  # trailing neither (2,3) nor 6
        with pytest.raises(ValueError):
            brainmass.sigmoidal_coupling(x, conn)

    def test_flattened_conn_non_square_raises(self):
        conn = jnp.ones(6)  # not a perfect square
        x = jnp.ones((2, 3))
        with pytest.raises(ValueError):
            brainmass.hyperbolic_tangent_coupling(x, conn)

    def test_flattened_x_path(self):
        # Flattened source (..., N_out*N_in) must reshape and match the 2-D form.
        conn = jnp.array([[1.0, 0.5],
                          [0.5, 1.0]])
        x2d = jnp.array([[1.0, 2.0],
                         [3.0, 4.0]])
        x_flat = x2d.reshape(-1)
        out_2d = brainmass.sigmoidal_coupling(x2d, conn, k=1.0)
        out_flat = brainmass.sigmoidal_coupling(x_flat, conn, k=1.0)
        assert u.math.allclose(out_2d, out_flat)

    def test_kernel_3d_conn_raises(self):
        x = jnp.ones((2, 2))
        with pytest.raises(ValueError):
            brainmass.sigmoidal_coupling(x, jnp.ones((2, 2, 2)))

    def test_kernel_0d_x_raises(self):
        with pytest.raises(ValueError):
            brainmass.sigmoidal_coupling(jnp.asarray(1.0), jnp.ones((2, 2)))

    def test_module_conn_ndim_raises(self):
        holder = _Holder()
        holder.x = jnp.ones((2, 2, 2))
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        with pytest.raises(ValueError):
            brainmass.SigmoidalCoupling(x_ref, conn=jnp.ones((2, 2, 2)))
        with pytest.raises(ValueError):
            brainmass.HyperbolicTangentCoupling(x_ref, conn=jnp.ones((2, 2, 2)))
        with pytest.raises(ValueError):
            brainmass.SigmoidalJansenRitCoupling(x_ref, conn=jnp.ones((2, 2, 2)))


class TestCouplingGradients:
    """Gradients flow w.r.t. trainable scalars (FD-checked)."""

    @staticmethod
    def _fd(f, x, eps=1e-4):
        return (f(x + eps) - f(x - eps)) / (2 * eps)

    def test_grad_sigmoidal_k_slope_midpoint(self):
        conn = jnp.ones((2, 2))
        x = jnp.array([[1.0, 0.5],
                       [0.2, 0.8]])

        def loss(p):
            k, slope, midpoint = p
            out = brainmass.sigmoidal_coupling(x, conn, k=k, slope=slope,
                                               midpoint=midpoint)
            return jnp.sum(out ** 2)

        p0 = jnp.array([1.0, 1.5, 0.3])
        g = jax.grad(loss)(p0)
        assert np.all(np.isfinite(np.asarray(g)))
        assert np.any(np.abs(np.asarray(g)) > 1e-6)
        # FD check each component.
        for i in range(3):
            def fi(v, i=i):
                return loss(p0.at[i].set(v))
            fd = self._fd(fi, float(p0[i]))
            assert np.allclose(np.asarray(g[i]), fd, rtol=2e-2, atol=1e-5)

    def test_grad_tanh_k(self):
        conn = jnp.ones((2, 2))
        x = jnp.array([[0.3, 0.1],
                       [0.2, 0.4]])

        def loss(k):
            return jnp.sum(brainmass.hyperbolic_tangent_coupling(x, conn, k=k) ** 2)

        g = jax.grad(loss)(0.5)
        fd = self._fd(loss, 0.5)
        assert np.isfinite(float(g)) and abs(float(g)) > 1e-6
        assert np.allclose(float(g), fd, rtol=2e-2, atol=1e-5)

    def test_grad_additive_bias(self):
        conn = jnp.ones((2, 2))
        x = jnp.array([[1.0, 2.0],
                       [3.0, 4.0]])

        def loss(b):
            return jnp.sum(brainmass.additive_coupling(x, conn, k=1.0, b=b) ** 2)

        g = jax.grad(loss)(0.5)
        fd = self._fd(loss, 0.5)
        assert np.isfinite(float(g)) and abs(float(g)) > 1e-6
        assert np.allclose(float(g), fd, rtol=2e-2, atol=1e-5)


class TestCouplingTrainableParam:
    """Constrained-Param round-trip: explicit Param exposes a trainable ParamState."""

    def test_trainable_k_roundtrip(self):
        holder = _Holder()
        holder.x = jnp.ones((2, 2))
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        k = brainstate.nn.Param(0.5, t=brainstate.nn.SigmoidT(0.0, 1.0))
        coup = brainmass.SigmoidalCoupling(x_ref, conn=jnp.ones((2, 2)), k=k)
        # Constrained value round-trips through the transform.
        assert u.math.allclose(coup.k.value(), jnp.asarray(0.5), atol=1e-5)
        # And it is exposed as a trainable ParamState.
        ps = coup.states(brainstate.ParamState)
        assert len(ps) >= 1

    def test_default_scalars_not_trainable(self):
        holder = _Holder()
        holder.x = jnp.ones((2, 2))
        x_ref = brainstate.nn.Prefetch(holder, 'x')
        coup = brainmass.SigmoidalCoupling(x_ref, conn=jnp.ones((2, 2)))
        # Defaults are Const -> no trainable ParamState.
        ps = coup.states(brainstate.ParamState)
        assert len(ps) == 0


class TestCouplingNetworkIntegration:
    """Wire a new coupling into a 2-node Network and run it under Simulator."""

    def test_sigmoidal_network_with_delays(self):
        brainstate.random.seed(0)
        brainstate.environ.set(dt=0.1 * u.ms)  # prefetch_delay sizes from delay/dt
        N = 2
        conn = np.array([[0.0, 0.2],
                         [0.3, 0.0]])
        distance = np.array([[0.0, 1.0],
                             [1.0, 0.0]])
        node = brainmass.HopfStep(N, a=0.1)
        net = brainmass.Network(
            node, conn=conn, distance=distance, speed=1.0,
            coupling='sigmoidal', coupled_var='x', k=0.5)
        sim = brainmass.Simulator(net, dt=0.1 * u.ms)
        res = sim.run(5.0 * u.ms, monitors=lambda m: m.node.x.value)
        out = np.asarray(u.get_magnitude(res['output']))
        assert out.shape == (50, N)
        assert np.all(np.isfinite(out))

    def test_tanh_network_instantaneous(self):
        brainstate.random.seed(0)
        brainstate.environ.set(dt=0.1 * u.ms)  # prefetch_delay sizes from delay/dt
        N = 2
        conn = np.array([[0.0, 0.2],
                         [0.3, 0.0]])
        node = brainmass.HopfStep(N, a=0.1)
        net = brainmass.Network(
            node, conn=conn, coupling='tanh', coupled_var='x', k=0.5)
        sim = brainmass.Simulator(net, dt=0.1 * u.ms)
        res = sim.run(3.0 * u.ms, monitors=lambda m: m.node.x.value)
        out = np.asarray(u.get_magnitude(res['output']))
        assert out.shape == (30, N)
        assert np.all(np.isfinite(out))

    def test_sigmoidal_jansen_rit_network(self):
        brainstate.random.seed(0)
        brainstate.environ.set(dt=0.1 * u.ms)  # prefetch_delay sizes from delay/dt
        N = 2
        conn = np.array([[0.0, 0.2],
                         [0.3, 0.0]])
        node = brainmass.HopfStep(N, a=0.1)
        net = brainmass.Network(
            node, conn=conn, coupling='sigmoidal_jansen_rit', coupled_var='x', k=1.0)
        sim = brainmass.Simulator(net, dt=0.1 * u.ms)
        res = sim.run(3.0 * u.ms, monitors=lambda m: m.node.x.value)
        out = np.asarray(u.get_magnitude(res['output']))
        assert out.shape == (30, N)
        assert np.all(np.isfinite(out))

    def test_bad_coupling_name_raises(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        node = brainmass.HopfStep(2, a=0.1)
        with pytest.raises(ValueError):
            brainmass.Network(node, conn=np.zeros((2, 2)),
                              coupling='nope', coupled_var='x')
