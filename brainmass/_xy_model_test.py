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

"""Tests for the abstract ``XY_Oscillator`` base shared by the two-variable models."""

import pytest

import brainstate

from brainmass._xy_model import XY_Oscillator


def test_xy_oscillator_base_rhs_not_implemented(dt):
    """The abstract ``XY_Oscillator`` leaves ``dx``/``dy`` to subclasses."""
    model = XY_Oscillator(2)
    brainstate.nn.init_all_states(model)
    with pytest.raises(NotImplementedError):
        model.dx(model.x.value, model.y.value, 0.0)
    with pytest.raises(NotImplementedError):
        model.dy(model.y.value, model.x.value, 0.0)
