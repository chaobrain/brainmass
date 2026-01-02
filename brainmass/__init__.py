# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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


__version__ = "0.0.5"
__version_info__ = tuple(map(int, __version__.split(".")))

# Common utilities
from ._common import (
    XY_Oscillator,
    sys2nd,
    sigmoid,
    bounded_input,
    euler_step,
    process_sequence,
)
# Coupling mechanisms
from ._coupling import (
    DiffusiveCoupling,
    AdditiveCoupling,
    diffusive_coupling,
    additive_coupling,
)
# Neural mass models
from ._fhn import FitzHughNagumoModel
# Forward models and lead field
from ._forward_model import (
    BOLDSignal,
    LeadFieldModel,
    EEGLeadFieldModel,
    MEGLeadFieldModel,
)
from ._hopf import HopfOscillator
# HORN models
from ._horn import (
    HORNStep,
    HORNSeqLayer,
    HORNSeqNetwork,
)
from ._jansen_rit import JansenRitStep
from ._kuramoto import KuramotoNetwork
from ._linear import ThresholdLinearModel
# Noise processes
from ._noise import (
    Noise,
    OUProcess,
    GaussianNoise,
    WhiteNoise,
    ColoredNoise,
    BrownianNoise,
    PinkNoise,
    BlueNoise,
    VioletNoise,
)
from ._qif import QIF
from ._sl import StuartLandauOscillator
# Type aliases
from ._typing import (
    Initializer,
    Array,
    Parameter,
)
from ._vdp import VanDerPolOscillator
from ._wilson_cowan import WilsonCowanModel
from ._wong_wang import WongWangModel

__all__ = [
    # Version
    '__version__',
    '__version_info__',

    # Common utilities
    'XY_Oscillator',
    'sys2nd',
    'sigmoid',
    'bounded_input',
    'euler_step',
    'process_sequence',

    # Type aliases
    'Initializer',
    'Array',
    'Parameter',

    # Noise processes
    'Noise',
    'OUProcess',
    'GaussianNoise',
    'WhiteNoise',
    'ColoredNoise',
    'BrownianNoise',
    'PinkNoise',
    'BlueNoise',
    'VioletNoise',

    # Neural mass models
    'FitzHughNagumoModel',
    'HopfOscillator',
    'JansenRitStep',
    'WilsonCowanModel',
    'WongWangModel',
    'VanDerPolOscillator',
    'QIF',
    'ThresholdLinearModel',
    'KuramotoNetwork',
    'StuartLandauOscillator',

    # Forward models and lead field
    'BOLDSignal',
    'LeadFieldModel',
    'EEGLeadFieldModel',
    'MEGLeadFieldModel',

    # Coupling mechanisms
    'DiffusiveCoupling',
    'AdditiveCoupling',
    'diffusive_coupling',
    'additive_coupling',

    # HORN models
    'HORNStep',
    'HORNSeqLayer',
    'HORNSeqNetwork',
]
