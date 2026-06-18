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

import brainstate
import brainunit as u

import brainmass


class TestHORNStep:
    def test_init_and_update_shape(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        m = brainmass.HORNStep(4)
        brainstate.nn.init_all_states(m)
        assert m.x.value.shape == (4,)
        out = m.update(jnp.ones(4))
        assert out.shape == (4,)
        assert np.all(np.isfinite(np.asarray(out)))


# --------------------------------------------------------------------------- #
# Pre-refactor golden trajectory (goal-05 characterization)
# --------------------------------------------------------------------------- #
# Goal-05 split ``horn.py`` into a package; that refactor had to be
# behaviour-preserving. This pins a seeded, deterministic 20-step ``exp_euler``
# trajectory against values captured from ``origin/main`` *before* the refactor.
# The trajectory was bit-identical pre/post; ``rtol`` only guards future
# XLA/platform drift.
_HORN_PRE_REFACTOR_GOLDEN = [
    [0.00030463768052868545, 0.00030463768052868545, 0.00030463768052868545, 0.00030463768052868545],
    [0.0009131503757089376, 0.0009131503757089376, 0.0009131503757089376, 0.0009131503757089376],
    [0.0018246239051222801, 0.0018246239051222801, 0.0018246239051222801, 0.0018246239051222801],
    [0.0030379933305084705, 0.0030379933305084705, 0.0030379933305084705, 0.0030379933305084705],
    [0.004552043974399567, 0.004552043974399567, 0.004552043974399567, 0.004552043974399567],
    [0.006365411914885044, 0.006365411914885044, 0.006365411914885044, 0.006365411914885044],
    [0.008476585149765015, 0.008476585149765015, 0.008476585149765015, 0.008476585149765015],
    [0.010883905924856663, 0.010883905924856663, 0.010883905924856663, 0.010883905924856663],
    [0.013585569337010384, 0.013585569337010384, 0.013585569337010384, 0.013585569337010384],
    [0.016579626128077507, 0.016579626128077507, 0.016579626128077507, 0.016579626128077507],
    [0.019863983616232872, 0.019863983616232872, 0.019863983616232872, 0.019863983616232872],
    [0.0234364066272974, 0.0234364066272974, 0.0234364066272974, 0.0234364066272974],
    [0.027294522151350975, 0.027294522151350975, 0.027294522151350975, 0.027294522151350975],
    [0.03143581375479698, 0.03143581375479698, 0.03143581375479698, 0.03143581375479698],
    [0.035857632756233215, 0.035857632756233215, 0.035857632756233215, 0.035857632756233215],
    [0.04055718705058098, 0.04055718705058098, 0.04055718705058098, 0.04055718705058098],
    [0.045531559735536575, 0.045531559735536575, 0.045531559735536575, 0.045531559735536575],
    [0.05077769234776497, 0.05077769234776497, 0.05077769234776497, 0.05077769234776497],
    [0.056292399764060974, 0.056292399764060974, 0.056292399764060974, 0.056292399764060974],
    [0.06207237020134926, 0.06207237020134926, 0.06207237020134926, 0.06207237020134926],
]


def test_trajectory_matches_pre_refactor_golden():
    """HORNStep reproduces its pre-refactor seeded ``exp_euler`` trajectory."""
    brainstate.environ.set(dt=0.1 * u.ms)
    brainstate.random.seed(0)
    np.random.seed(0)
    model = brainmass.HORNStep(4)
    brainstate.nn.init_all_states(model)
    traj = []
    for i in range(20):
        with brainstate.environ.context(i=i, t=i * 0.1 * u.ms):
            out = model.update(jnp.ones(4))
        traj.append(np.asarray(u.get_magnitude(out)).reshape(-1))
    got = np.stack(traj)
    golden = np.asarray(_HORN_PRE_REFACTOR_GOLDEN)
    assert got.shape == golden.shape
    np.testing.assert_allclose(
        got, golden, rtol=1e-5, atol=1e-6,
        err_msg="horn: trajectory drifted from pre-refactor golden",
    )
