# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
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

"""Golden-trajectory characterization tests for the goal-05 structure refactor.

The refactor that unified the two-variable models onto
``NeuralMassDynamics`` (``XY_Oscillator``, ``FitzHughNagumoStep``,
``MontbrioPazoRoxinStep``) and split ``jansen_rit.py`` / ``horn.py`` into
packages was required to be *behaviour preserving*. These tests pin a seeded,
deterministic ``exp_euler`` trajectory for every touched model against golden
values captured from ``origin/main`` **before** the refactor (the values in
``_characterization_goldens.json``). At refactor time every trajectory was in
fact bit-identical pre/post; the tolerances below (``rtol=1e-5``) only guard
against future XLA/platform drift.

If one of these fails after an intentional change, regenerate the fixture with
the same recipe (see ``_CASES`` / ``_run`` below) and review the diff.
"""

import json
import pathlib

import numpy as np
import pytest

import braintools
import brainstate
import brainunit as u
import jax.numpy as jnp

import brainmass

_DT = 0.1
_NSTEPS = 20
_SEED = 0

_GOLDENS = json.loads(
    (pathlib.Path(__file__).parent / '_characterization_goldens.json').read_text()
)


def _run(build, drive):
    """Replay a model's seeded ``exp_euler`` trajectory.

    Mirrors the generator used to capture the golden fixture: fixed ``dt``,
    fixed RNG seed, ``_NSTEPS`` calls to ``drive`` (which calls ``update``),
    collecting the flattened magnitude of each step's output.
    """
    brainstate.environ.set(dt=_DT * u.ms)
    brainstate.random.seed(_SEED)
    np.random.seed(_SEED)
    model = build()
    brainstate.nn.init_all_states(model)
    traj = []
    for i in range(_NSTEPS):
        with brainstate.environ.context(i=i, t=i * _DT * u.ms):
            out = drive(model)
        traj.append(np.asarray(u.get_magnitude(out)).reshape(-1))
    return np.stack(traj)


# (build, drive) recipes -- one per model touched by the goal-05 refactor.
_CASES = {
    # XY_Oscillator family migrated onto the unified NeuralMassDynamics base.
    'hopf': (lambda: brainmass.HopfStep(2, a=-0.2),
             lambda m: m.update(0.1, 0.0)),
    'sl': (lambda: brainmass.StuartLandauStep(in_size=2),
           lambda m: m.update(0.1, 0.0)),
    'vdp': (lambda: brainmass.VanDerPolStep(in_size=2, mu=2.0),
            lambda m: m.update(0.1, 0.0)),
    # Two-variable models migrated onto the shared base.
    'fhn': (lambda: brainmass.FitzHughNagumoStep(2),
            lambda m: m.update(V_inp=0.3)),
    'montbrio': (lambda: brainmass.MontbrioPazoRoxinStep(
        2, init_r=braintools.init.ZeroInit(unit=u.Hz),
        init_v=braintools.init.ZeroInit()),
        lambda m: m.update(v_inp=0.5)),
    # Models whose source module was split into a package.
    'jansen': (lambda: brainmass.JansenRitStep(in_size=3),
               lambda m: m.update()),
    'horn': (lambda: brainmass.HORNStep(4),
             lambda m: m.update(jnp.ones(4))),
}


@pytest.mark.parametrize('name', sorted(_CASES))
def test_trajectory_matches_pre_refactor_golden(name):
    """Each touched model reproduces its pre-refactor ``exp_euler`` trajectory."""
    build, drive = _CASES[name]
    got = _run(build, drive)
    golden = np.asarray(_GOLDENS[name])
    assert got.shape == golden.shape, (
        f"{name}: shape {got.shape} != golden {golden.shape}"
    )
    np.testing.assert_allclose(
        got, golden, rtol=1e-5, atol=1e-6,
        err_msg=f"{name}: trajectory drifted from pre-refactor golden",
    )


def test_every_touched_model_has_a_golden():
    """Guard: the fixture and the recipe set stay in sync."""
    assert set(_CASES) == set(_GOLDENS)
