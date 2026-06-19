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


from ._version import (
    __version__,
    __version_info__,
)
from ._xy_model import (
    XY_Oscillator,
)
# Coupling mechanisms
from .coupling import (
    DiffusiveCoupling,
    AdditiveCoupling,
    SigmoidalCoupling,
    HyperbolicTangentCoupling,
    SigmoidalJansenRitCoupling,
    diffusive_coupling,
    additive_coupling,
    sigmoidal_coupling,
    hyperbolic_tangent_coupling,
    sigmoidal_jansen_rit_coupling,
    laplacian_connectivity,
    LaplacianConnParam,
)
# Neural mass models
from .coombes_byrne import CoombesByrneStep
from .epileptor import EpileptorStep
from .fhn import FitzHughNagumoStep
from .generic_2d_oscillator import Generic2dOscillatorStep
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
from .larter_breakspear import LarterBreakspearStep
from .lorenz import LorenzStep
from .leadfield import LeadfieldReadout
from .linear import ThresholdLinearStep, LinearStep
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
    WilsonCowanNoSaturationStep,
    WilsonCowanSymmetricStep,
    WilsonCowanSimplifiedStep,
    WilsonCowanLinearStep,
    WilsonCowanDivisiveStep,
    WilsonCowanDivisiveInputStep,
    WilsonCowanDelayedStep,
    WilsonCowanAdaptiveStep,
    WilsonCowanThreePopBase,
    WilsonCowanThreePopulationStep,
)
from .wong_wang import WongWangStep
from .wong_wang_exc_inh import WongWangExcInhStep
# Orchestration layer
from . import objectives
from .fitter import Fitter, FitResult
from .network import Network
from .simulator import Simulator

__all__ = [
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
    'WilsonCowanNoSaturationStep',
    'WilsonCowanSymmetricStep',
    'WilsonCowanSimplifiedStep',
    'WilsonCowanLinearStep',
    'WilsonCowanDivisiveStep',
    'WilsonCowanDivisiveInputStep',
    'WilsonCowanDelayedStep',
    'WilsonCowanAdaptiveStep',
    'WilsonCowanThreePopBase',
    'WilsonCowanThreePopulationStep',
    'WongWangStep',
    'VanDerPolStep',
    'MontbrioPazoRoxinStep',
    'CoombesByrneStep',
    'LarterBreakspearStep',
    'EpileptorStep',
    'Generic2dOscillatorStep',
    'WongWangExcInhStep',
    'LorenzStep',
    'ThresholdLinearStep',
    'LinearStep',
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
    'SigmoidalCoupling',
    'HyperbolicTangentCoupling',
    'SigmoidalJansenRitCoupling',
    'diffusive_coupling',
    'additive_coupling',
    'sigmoidal_coupling',
    'hyperbolic_tangent_coupling',
    'sigmoidal_jansen_rit_coupling',
    'laplacian_connectivity',
    'LaplacianConnParam',

    # HORN models
    'HORNStep',
    'HORNSeqLayer',
    'HORNSeqNetwork',

    # Orchestration layer
    'Network',
    'Simulator',
    'Fitter',
    'FitResult',
    'objectives',
]


def __getattr__(old_name):
    name = old_name.replace('Model', 'Step')
    if name in __all__:
        import warnings
        from . import __name__ as module_name
        module = __import__(module_name, fromlist=[name])
        warnings.warn(
            f"{module_name}.{name} is deprecated, use {module_name}.{old_name} instead.",
            DeprecationWarning
        )
        return getattr(module, name)
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
