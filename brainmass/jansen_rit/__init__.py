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

"""Jansen-Rit neural-mass model and its network variants.

This package was split out of the former single ``jansen_rit.py`` module
(goal-05 structure refactor). Every symbol that used to live at
``brainmass.jansen_rit.<name>`` is re-exported here, so existing imports such
as ``from brainmass.jansen_rit import JansenRitNetwork`` keep working.

Submodules
----------
step
    The single-node ``JansenRitStep`` dynamics (and the ``Identity`` helper).
connectivity
    Laplacian coupling operators (``LaplacianConnectivity``, ``LaplacianConnV2``).
tr
    The time-resolved ``JansenRitTR`` network.
network
    Layered network builders (``JansenRitLayer``, ``JansenRit2Layer``,
    ``JansenRitNetwork``).
"""

from .connectivity import LaplacianConnectivity, LaplacianConnV2
from .network import JansenRit2Layer, JansenRitLayer, JansenRitNetwork
from .step import Identity, JansenRitStep
from .tr import JansenRitTR

__all__ = [
    'JansenRitStep',
    'JansenRitTR',
    # Network / connectivity helpers kept importable for backward compatibility
    # with the old flat module path ``brainmass.jansen_rit.<name>``.
    'Identity',
    'LaplacianConnectivity',
    'LaplacianConnV2',
    'JansenRitLayer',
    'JansenRit2Layer',
    'JansenRitNetwork',
]
