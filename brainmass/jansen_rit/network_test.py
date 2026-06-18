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


def _jr_seq(T, n, seed=0):
    brainstate.random.seed(seed)
    np.random.seed(seed)  # the sequence below uses numpy's RNG, not brainstate's
    return jnp.asarray(np.random.randn(T, n).astype('float32'))


class TestJansenRitLayer:
    """Cover JansenRitLayer (additive + delayed) — uses the deduped AdditiveConn/DelayedAdditiveConn."""

    def test_additive_path(self):
        from brainmass.jansen_rit import JansenRitLayer
        brainstate.environ.set(dt=1e-4 * u.second)
        m = JansenRitLayer(n_input=3, n_hidden=5)
        brainstate.nn.init_all_states(m)
        out = m.update(_jr_seq(6, 3))
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))

    def test_delayed_path_and_record_state(self):
        from brainmass.jansen_rit import JansenRitLayer
        brainstate.environ.set(dt=1e-4 * u.second)
        np.random.seed(0)  # deterministic delay matrix
        delay = jnp.asarray(np.random.randint(1, 4, size=(5, 5)).astype(float)) * u.ms
        m = JansenRitLayer(n_input=3, n_hidden=5, delay=delay)
        brainstate.nn.init_all_states(m)
        st, out = m.update(_jr_seq(6, 3), record_state=True)
        assert set(st.keys()) == {'M', 'E', 'I'}
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))


class TestJansenRit2LayerAndNetwork:
    """Cover JansenRit2Layer / LaplacianConnV2 / JansenRitNetwork."""

    def test_jansen_rit2_layer_record_state(self):
        from brainmass.jansen_rit import JansenRit2Layer
        brainstate.environ.set(dt=1e-4 * u.second)
        np.random.seed(0)  # deterministic delay matrix
        delay = jnp.asarray(np.random.randint(1, 4, size=(5, 5)).astype(float)) * u.ms
        m = JansenRit2Layer(n_input=3, n_hidden=5, delay=delay)
        brainstate.nn.init_all_states(m)
        st, out = m.update(_jr_seq(6, 3), record_state=True)
        assert set(st.keys()) == {'M', 'E', 'I'}
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))

    def test_jansen_rit2_layer_requires_delay(self):
        from brainmass.jansen_rit import JansenRit2Layer
        brainstate.environ.set(dt=1e-4 * u.second)
        with pytest.raises(AssertionError):
            JansenRit2Layer(n_input=3, n_hidden=5)  # delay is required

    def test_network_single_and_multi_layer(self):
        from brainmass.jansen_rit import JansenRitNetwork
        brainstate.environ.set(dt=1e-4 * u.second)
        np.random.seed(0)  # deterministic delay matrix
        delay = jnp.asarray(np.random.randint(1, 4, size=(5, 5)).astype(float)) * u.ms
        net = JansenRitNetwork(n_input=3, n_hidden=5, n_output=2, delay=delay)
        brainstate.nn.init_all_states(net)
        out = net.update(_jr_seq(6, 3))
        assert out.shape == (2,)  # network emits a single readout from the final state
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))

    def test_network_multi_layer_runs(self):
        """Stacking ≥2 JansenRit2Layers now runs (goal-05 fix).

        Previously this was ``xfail(strict)``: a layer emits an mV ``eeg()``
        proxy, and the next layer re-multiplied its input by ``u.mV`` ->
        ``UnitMismatchError``. ``JansenRitNetwork.update`` now strips units
        between layers, so a two-layer stack produces a finite readout.
        """
        from brainmass.jansen_rit import JansenRitNetwork
        brainstate.environ.set(dt=1e-4 * u.second)
        np.random.seed(0)  # deterministic delay matrix
        delay = jnp.asarray(np.random.randint(1, 4, size=(5, 5)).astype(float)) * u.ms
        net = JansenRitNetwork(n_input=3, n_hidden=[5, 5], n_output=2, delay=delay)
        brainstate.nn.init_all_states(net)
        out = net.update(_jr_seq(6, 3))
        assert out.shape == (2,)
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))
