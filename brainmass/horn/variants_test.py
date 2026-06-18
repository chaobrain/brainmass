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
import brainunit as u

import brainmass
from brainmass.horn import HORN_TR


def _seq(T, n, seed=0):
    brainstate.random.seed(seed)
    np.random.seed(seed)  # the sequence below uses numpy's RNG, not brainstate's
    return jnp.asarray(np.random.randn(T, n).astype('float32'))


class TestHORN_TR:
    @pytest.mark.parametrize('rec_type,with_delay', [
        (None, False),
        ('additive', True),
        ('additive_tr', True),
        ('laplacian', True),
        ('laplacian_tr', True),
    ])
    def test_rec_types(self, rec_type, with_delay):
        brainstate.environ.set(dt=0.1 * u.ms)
        n = 5
        np.random.seed(0)  # deterministic delay matrix
        delay = (jnp.asarray(np.random.randint(1, 4, size=(n, n)).astype('float32')) * u.ms
                 if with_delay else None)
        kwargs = dict(tr=0.5 * u.ms)
        if rec_type is not None:
            kwargs.update(delay=delay, rec_type=rec_type)
        m = HORN_TR(n, **kwargs)
        brainstate.nn.init_all_states(m)
        out = m.update(_seq(6, n))
        assert np.all(np.isfinite(np.asarray(u.get_magnitude(out))))

    def test_unknown_rec_type_raises(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        np.random.seed(0)  # deterministic delay matrix
        delay = jnp.asarray(np.random.randint(1, 4, size=(5, 5)).astype('float32')) * u.ms
        with pytest.raises(ValueError):
            HORN_TR(5, delay=delay, rec_type='nope')
