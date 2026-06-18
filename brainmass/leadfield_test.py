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
    np.random.seed(seed)  # the lead-field matrix below uses numpy's RNG
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
        # The default ``normalize=True`` is L1 row-normalisation: with demean=False
        # to isolate it, each row's sum of absolute values should be ~1. (goal-05
        # made L1 the documented default and the code now uses sum(|w|) directly.)
        readout, M, N = _make(normalize=True, demean=False)
        lm = np.asarray(readout.lm.value())
        l1_norms = np.abs(lm).sum(axis=1)
        assert np.allclose(l1_norms, 1.0, atol=1e-5)

    def test_demean_removes_column_mean(self):
        readout, M, N = _make(normalize=False, demean=True)
        lm = np.asarray(readout.lm.value())
        # demean subtracts the per-column (across-channel) mean -> column means ~0
        assert np.allclose(lm.mean(axis=0), 0.0, atol=1e-5)

    def test_normalize_true_equals_l1(self):
        # ``normalize=True`` is the backward-compatible alias for L1.
        lm_true = np.asarray(_make(normalize=True, demean=False)[0].lm.value())
        lm_l1 = np.asarray(_make(normalize='l1', demean=False)[0].lm.value())
        assert np.allclose(lm_true, lm_l1, atol=0.0)

    def test_l2_normalization_makes_unit_l2_rows(self):
        # 'l2' scales each row by its Euclidean norm -> unit-L2 rows.
        readout, M, N = _make(normalize='l2', demean=False)
        lm = np.asarray(readout.lm.value())
        l2_norms = np.sqrt((lm ** 2).sum(axis=1))
        assert np.allclose(l2_norms, 1.0, atol=1e-5)

    def test_l1_and_l2_differ(self):
        # The two norms produce genuinely different matrices (sanity check that
        # 'l2' is not silently doing L1).
        lm_l1 = np.asarray(_make(normalize='l1', demean=False)[0].lm.value())
        lm_l2 = np.asarray(_make(normalize='l2', demean=False)[0].lm.value())
        assert not np.allclose(lm_l1, lm_l2, atol=1e-3)

    def test_invalid_normalize_raises(self):
        with pytest.raises(ValueError):
            _make(normalize='l3')
