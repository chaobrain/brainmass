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

# Coupling mechanisms
from .coupling import (
    DiffusiveCoupling,
    AdditiveCoupling,
    diffusive_coupling,
    additive_coupling,
    laplacian_connectivity,
    LaplacianConnParam,
)
# Neural mass models
from .fhn import FitzHughNagumoStep
# Forward models and lead field
from .forward_model import (
    BOLDSignal,
    LeadFieldModel,
    EEGLeadFieldModel,
    MEGLeadFieldModel,
)
from .hopf import HopfStep
# HORN models
from .horn import (
    HORNStep,
    HORNSeqLayer,
    HORNSeqNetwork,
)
from .jansen_rit import (
    JansenRitStep,
    JansenRitTR,
)
from .kuramoto import KuramotoNetwork
from .leadfield import LeadfieldReadout
from .linear import ThresholdLinearStep
# Noise processes
from .noise import (
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
from .qif import MontbrioPazoRoxinStep
from .sl import StuartLandauStep
# Type aliases
from .typing import (
    Initializer,
    Array,
    Parameter,
)
# Common utilities
from .utils import (
    sys2nd,
    sigmoid,
    bounded_input,
    process_sequence,
    delay_index,
)
from .vdp import VanDerPolStep
from .wilson_cowan import (
    WilsonCowanStep,
    WilsonCowanStepNoSaturation,
    WilsonCowanStepSymmetric,
    WilsonCowanStepSimplified,
    WilsonCowanStepLinear,
)
from .wong_wang import WongWangStep
from ._xy_model import (
    XY_Oscillator,
)

__all__ = [
    # Version
    '__version__',
    '__version_info__',

    # Common utilities
    'XY_Oscillator',
    'sys2nd',
    'sigmoid',
    'bounded_input',
    'process_sequence',
    'delay_index',

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
    'FitzHughNagumoStep',
    'HopfStep',
    'WilsonCowanStep',
    'WilsonCowanStepNoSaturation',
    'WilsonCowanStepSymmetric',
    'WilsonCowanStepSimplified',
    'WilsonCowanStepLinear',
    'WongWangStep',
    'VanDerPolStep',
    'MontbrioPazoRoxinStep',
    'ThresholdLinearStep',
    'LeadfieldReadout',
    'KuramotoNetwork',
    'StuartLandauStep',

    # Jansen-Rit model
    'JansenRitStep',
    'JansenRitTR',

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
    'laplacian_connectivity',
    'LaplacianConnParam',

    # HORN models
    'HORNStep',
    'HORNSeqLayer',
    'HORNSeqNetwork',
]
