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

"""Harmonic Oscillatory Recurrent Network (HORN) models.

This package was split out of the former single ``horn.py`` module (goal-05
structure refactor). Every symbol that used to live at ``brainmass.horn.<name>``
is re-exported here, so existing imports such as
``from brainmass.horn import HORN_TR`` keep working.

Submodules
----------
step
    The single-node ``HORNStep`` dynamics.
connections
    Delayed recurrent coupling operators (``DelayedAdditiveConnTR``,
    ``DelayedLaplacianConn``, ``DelayedLaplacianConnTR``).
variants
    The time-resolved ``HORN_TR`` network.
sequential
    Sequence models (``HORNSeqLayer``, ``HORNSeqNetwork``).
"""

# Coupling operators re-exported so the old flat-module attribute paths
# (e.g. ``brainmass.horn.AdditiveConn``) keep resolving.
from ..coupling import (
    AdditiveConn,
    AdditiveCoupling,
    DelayedAdditiveConn,
    LaplacianConnParam,
    additive_coupling,
)
from .connections import (
    DelayedAdditiveConnTR,
    DelayedLaplacianConn,
    DelayedLaplacianConnTR,
)
from .sequential import HORNSeqLayer, HORNSeqNetwork
from .step import HORNStep
from .variants import HORN_TR

__all__ = [
    'HORNStep',
    'HORNSeqLayer',
    'HORNSeqNetwork',
    # Kept importable for backward compatibility with the old flat module path
    # ``brainmass.horn.<name>``.
    'HORN_TR',
    'DelayedAdditiveConnTR',
    'DelayedLaplacianConn',
    'DelayedLaplacianConnTR',
    'AdditiveConn',
    'AdditiveCoupling',
    'DelayedAdditiveConn',
    'LaplacianConnParam',
    'additive_coupling',
]
