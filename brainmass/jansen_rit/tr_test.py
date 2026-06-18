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


class TestJansenRitTR:
    """JansenRitTR.update: per-TR terms must be ready before the sub-step loop."""

    @staticmethod
    def _make(n=3, tr=1e-3 * u.second, dt=1e-4 * u.second, seed=0):
        brainstate.environ.set(dt=dt)
        brainstate.random.seed(seed)
        np.random.seed(seed)  # sc/delay/w below use numpy's RNG
        sc = jnp.asarray(np.random.rand(n, n))
        sc = sc - jnp.diag(jnp.diag(sc))
        delay = jnp.asarray(np.random.randint(1, 5, size=(n, n)).astype(float))
        w = jnp.asarray(np.random.randn(n, n) * 0.1)
        model = brainmass.JansenRitTR(
            in_size=n, delay=delay, sc=sc, k=1.0,
            w_ll=w, w_ff=w, w_bb=w, g_l=1.0, g_f=1.0, g_b=1.0, tr=tr,
        )
        brainstate.nn.init_all_states(model)
        return model, n

    def test_update_runs_and_returns_finite(self):
        model, n = self._make()
        out = model.update(jnp.zeros(n))
        out = np.asarray(u.get_magnitude(out))
        assert out.shape == (n,)
        assert np.all(np.isfinite(out))

    def test_update_record_state(self):
        model, n = self._make()
        out, state = model.update(jnp.zeros(n), record_state=True)
        assert set(state.keys()) == {'M', 'E', 'I'}
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))

    def test_non_integer_n_step(self):
        # tr/dt = 1e-3 / 0.3e-3 = 3.33 -> int() floors to 3; must still run.
        model, n = self._make(tr=1e-3 * u.second, dt=0.3e-3 * u.second)
        out = model.update(jnp.zeros(n))
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))

    def test_iter_input_path(self):
        model, n = self._make(tr=1e-3 * u.second, dt=1e-4 * u.second)
        n_step = int(1e-3 / 1e-4)  # 10
        inp = jnp.zeros((n_step, n))
        out = model.update(inp, iter_input=True)
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))


class TestJansenRitTRMask:
    """Cover the mask branch in LaplacianConnectivity._normalize / _symmetric_normalize."""

    def test_masked_update_runs(self):
        n = 3
        brainstate.environ.set(dt=1e-4 * u.second)
        brainstate.random.seed(0)
        np.random.seed(0)  # sc/delay/w/mask below use numpy's RNG
        sc = jnp.asarray(np.random.rand(n, n))
        sc = sc - jnp.diag(jnp.diag(sc))
        delay = jnp.asarray(np.random.randint(1, 5, size=(n, n)).astype(float))
        w = jnp.asarray(np.random.randn(n, n) * 0.1)
        mask = jnp.asarray((np.random.rand(n, n) > 0.3).astype('float32'))
        model = brainmass.JansenRitTR(
            in_size=n, delay=delay, sc=sc, k=1.0,
            w_ll=w, w_ff=w, w_bb=w, g_l=1.0, g_f=1.0, g_b=1.0,
            tr=1e-3 * u.second, mask=mask,
        )
        brainstate.nn.init_all_states(model)
        out = model.update(jnp.zeros(n))
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))
