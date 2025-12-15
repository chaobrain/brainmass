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

from ._coupling import *
from ._coupling import __all__ as coupling_all
from ._fhn import *
from ._fhn import __all__ as fhn_all
from ._forward_model import *
from ._forward_model import __all__ as forward_model_all
from ._hopf import *
from ._hopf import __all__ as hopf_all
from ._jansen_rit import *
from ._jansen_rit import __all__ as jansen_rit_all
from ._linear import *
from ._linear import __all__ as linear_all
from ._noise import *
from ._noise import __all__ as noise_all
from ._qif import *
from ._qif import __all__ as qif_all
from ._sl import *
from ._sl import __all__ as sl_all
from ._vdp import *
from ._vdp import __all__ as vdp_all
from ._wilson_cowan import *
from ._wilson_cowan import __all__ as wilson_cowan_all
from ._wong_wang import *
from ._wong_wang import __all__ as wong_wang_all

__all__ = forward_model_all + coupling_all + jansen_rit_all + noise_all + wilson_cowan_all + wong_wang_all + hopf_all
__all__ = __all__ + fhn_all + linear_all + vdp_all + qif_all + sl_all + ['ArrayParam']
del forward_model_all, coupling_all, jansen_rit_all, noise_all, wilson_cowan_all, wong_wang_all, hopf_all
del fhn_all, linear_all, vdp_all, qif_all, sl_all


def __getattr__(name):
    if name == 'ArrayParam':
        import warnings
        import braintools
        warnings.warn(
            "brainmass.ArrayParam is deprecated and will be removed in a future version. "
            "Please use braintools.param.Param instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return braintools.param.Param
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
