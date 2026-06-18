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

import brainstate
from brainstate.nn import Param

import brainmass


def _make(normalize=True, demean=True, M=3, N=5, seed=0):
    brainstate.random.seed(seed)
    lm = jnp.asarray(np.random.randn(M, N).astype('float32'))
    readout = brainmass.LeadfieldReadout(
        lm=lm,
        y0=Param.init(0.0),
        cy0=Param.init(1.0),
        normalize=normalize,
        demean=demean,
    )
    brainstate.nn.init_all_states(readout)
    return readout, M, N


class TestLeadfieldReadout:
    @pytest.mark.parametrize('normalize,demean', [
        (True, True), (True, False), (False, True), (False, False),
    ])
    def test_update_1d(self, normalize, demean):
        readout, M, N = _make(normalize, demean)
        out = readout.update(jnp.ones(N))
        assert out.shape == (M,)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_update_batched_vmap(self):
        readout, M, N = _make()
        T = 7
        out = readout.update(jnp.ones((T, N)))
        assert out.shape == (T, M)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_row_normalization_makes_unit_l1_rows(self):
        # normalize_leadfield computes sum(sqrt(w**2)) == sum(|w|), i.e. the L1
        # norm (despite the historical "L2" label). With demean=False to isolate
        # normalization, each row's sum of absolute values should be ~1.
        readout, M, N = _make(normalize=True, demean=False)
        lm = np.asarray(readout.lm.value())
        l1_norms = np.abs(lm).sum(axis=1)
        assert np.allclose(l1_norms, 1.0, atol=1e-5)

    def test_demean_removes_column_mean(self):
        readout, M, N = _make(normalize=False, demean=True)
        lm = np.asarray(readout.lm.value())
        # demean subtracts the per-column (across-channel) mean -> column means ~0
        assert np.allclose(lm.mean(axis=0), 0.0, atol=1e-5)
