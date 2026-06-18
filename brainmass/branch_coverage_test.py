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

"""Targeted tests for otherwise-uncovered branches.

These cover the small, easy-to-miss code paths that the model behaviour tests do
not reach: the abstract base ``XY_Oscillator`` right-hand sides, the noise
injection branches of the FitzHugh-Nagumo and Montbrio-Pazo-Roxin models, and
the deprecated-alias ``__getattr__`` shim of the package.
"""

import warnings

import numpy as np
import pytest

import brainstate
import brainunit as u

import brainmass
from brainmass._xy_model import XY_Oscillator


def test_xy_oscillator_base_rhs_not_implemented(dt):
    """The abstract ``XY_Oscillator`` leaves ``dx``/``dy`` to subclasses."""
    model = XY_Oscillator(2)
    brainstate.nn.init_all_states(model)
    with pytest.raises(NotImplementedError):
        model.dx(model.x.value, model.y.value, 0.0)
    with pytest.raises(NotImplementedError):
        model.dy(model.y.value, model.x.value, 0.0)


def _reproducible_noise_run(make_model, seed=0, n_steps=20):
    """Run a noisy model twice under a fixed seed and return both trajectories."""
    def once():
        brainstate.random.seed(seed)
        model = make_model()
        brainstate.nn.init_all_states(model)

        def step(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update()

        return np.asarray(u.get_magnitude(
            brainstate.transform.for_loop(step, np.arange(n_steps))))

    return once(), once()


def test_montbrio_noise_paths_are_seeded(dt):
    """Both noise channels of MontbrioPazoRoxin run and are reproducible."""
    a, b = _reproducible_noise_run(
        lambda: brainmass.MontbrioPazoRoxinStep(
            2,
            noise_r=brainmass.GaussianNoise(2, sigma=1.0 * u.Hz),
            noise_v=brainmass.GaussianNoise(2, sigma=0.1)))
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, b)


def test_fhn_noise_paths_are_seeded(dt):
    """Both noise channels of FitzHugh-Nagumo run and are reproducible."""
    a, b = _reproducible_noise_run(
        lambda: brainmass.FitzHughNagumoStep(
            2,
            noise_V=brainmass.GaussianNoise(2, sigma=0.1),
            noise_w=brainmass.GaussianNoise(2, sigma=0.1)))
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, b)


def test_deprecated_model_alias_warns_and_resolves():
    """Legacy ``*Model`` names resolve to their ``*Step`` class with a warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls = brainmass.WilsonCowanModel  # legacy alias for WilsonCowanStep
    assert cls is brainmass.WilsonCowanStep
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_unknown_attribute_raises():
    """An attribute that is neither current nor a legacy alias raises."""
    with pytest.raises(AttributeError):
        _ = brainmass.DefinitelyNotAModelName
