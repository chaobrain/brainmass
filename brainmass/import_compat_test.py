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

"""Backward-compatible import surface after the goal-05 structure refactor.

``jansen_rit.py`` and ``horn.py`` became packages, and the two-variable models
were re-parented onto ``NeuralMassDynamics``. None of that may change the public
import surface: every ``brainmass.__all__`` symbol must still resolve, every
old fully-qualified module path (``brainmass.jansen_rit.X`` / ``brainmass.horn.X``)
must still import the *same* object, the ``Model`` -> ``Step`` deprecation shim
must still work, and moved classes must remain valid (pytree-flattenable) models.
"""

import importlib

import jax
import numpy as np
import pytest

import brainstate

import brainmass


def test_all_public_symbols_resolve():
    """Every name advertised in ``brainmass.__all__`` is importable and non-None."""
    missing = [name for name in brainmass.__all__ if getattr(brainmass, name, None) is None]
    assert not missing, f"brainmass.__all__ names failed to resolve: {missing}"


# --- Old fully-qualified module paths must keep resolving to the same object ---

# Names that lived at ``brainmass.jansen_rit.<name>`` before the package split.
_JANSEN_PATH_SYMBOLS = [
    'JansenRitStep', 'JansenRitTR', 'Identity',
    'LaplacianConnectivity', 'LaplacianConnV2',
    'JansenRitLayer', 'JansenRit2Layer', 'JansenRitNetwork',
]

# Names that lived at ``brainmass.horn.<name>`` before the package split
# (public classes + connection helpers + re-exported coupling operators).
_HORN_PATH_SYMBOLS = [
    'HORNStep', 'HORNSeqLayer', 'HORNSeqNetwork', 'HORN_TR',
    'DelayedAdditiveConnTR', 'DelayedLaplacianConn', 'DelayedLaplacianConnTR',
    'AdditiveConn', 'AdditiveCoupling', 'DelayedAdditiveConn',
    'LaplacianConnParam', 'additive_coupling',
]


@pytest.mark.parametrize('modpath, symbols', [
    ('brainmass.jansen_rit', _JANSEN_PATH_SYMBOLS),
    ('brainmass.horn', _HORN_PATH_SYMBOLS),
])
def test_old_module_paths_still_resolve(modpath, symbols):
    """``brainmass.<pkg>.<name>`` keeps importing (back-compat for the flat modules).

    This is also the path pickle uses for instances created before the split, so
    it must keep resolving even for the formerly-internal helper classes.
    """
    mod = importlib.import_module(modpath)
    missing = [s for s in symbols if getattr(mod, s, None) is None]
    assert not missing, f"{modpath} no longer exposes: {missing}"


@pytest.mark.parametrize('modpath, symbols', [
    ('brainmass.jansen_rit', ['JansenRitStep', 'JansenRitTR']),
    ('brainmass.horn', ['HORNStep', 'HORNSeqLayer', 'HORNSeqNetwork']),
])
def test_top_level_and_package_are_the_same_object(modpath, symbols):
    """The top-level export and the package attribute are one identical class.

    A duplicated class object would mean a second pytree/graph registration and
    subtle ``isinstance`` / state-handling bugs, so identity matters here.
    """
    mod = importlib.import_module(modpath)
    for s in symbols:
        assert getattr(brainmass, s) is getattr(mod, s), f"{s} diverged from brainmass.{s}"


def test_from_import_forms_work():
    """The ``from brainmass.<pkg> import <name>`` spellings used by tests/tutorials work."""
    from brainmass.jansen_rit import JansenRitNetwork, JansenRitStep, JansenRitTR  # noqa: F401
    from brainmass.horn import HORN_TR, HORNStep, AdditiveConn  # noqa: F401


def test_model_to_step_deprecation_shim():
    """The legacy ``<Model>`` aliases still resolve (to the ``<Step>`` classes) with a warning."""
    with pytest.warns(DeprecationWarning):
        alias = brainmass.JansenRitModel
    assert alias is brainmass.JansenRitStep


@pytest.mark.parametrize('factory', [
    lambda: brainmass.JansenRitStep(in_size=3),
    lambda: brainmass.HORNStep(4),
    lambda: brainmass.HopfStep(2, a=-0.2),
    lambda: brainmass.FitzHughNagumoStep(2),
])
def test_moved_classes_pytree_roundtrip(factory):
    """A moved/re-parented class still produces a valid, pytree-flattenable model.

    Guards against the class object being re-registered or its ``HiddenState`` /
    ``Param`` states breaking when the defining module moved.
    """
    model = factory()
    brainstate.nn.init_all_states(model)
    values = {k: v.value for k, v in model.states().items()}
    assert values, "model exposes no states"
    leaves, treedef = jax.tree.flatten(values)
    assert leaves, "state pytree has no leaves"
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert set(rebuilt) == set(values)
    # Compare via the (unit-stripped) pytree leaves: some states are unit-bearing
    # ``Quantity`` objects that intentionally refuse silent ``np.asarray`` coercion.
    rebuilt_leaves = jax.tree.leaves(rebuilt)
    assert len(rebuilt_leaves) == len(leaves)
    for got, exp in zip(rebuilt_leaves, leaves):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(exp))
