# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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


__version__ = "0.0.4"
__version_info__ = (0, 0, 4)

from .coupling import *
from .coupling import __all__ as coupling_all
from .fhn import *
from .fhn import __all__ as fhn_all
from .forward_model import *
from .forward_model import __all__ as forward_model_all
from .hopf import *
from .hopf import __all__ as hopf_all
from .jansen_rit import *
from .jansen_rit import __all__ as jansen_rit_all
from .linear import *
from .linear import __all__ as linear_all
from .noise import *
from .noise import __all__ as noise_all
from .param import *
from .param import __all__ as param_all
from .qif import *
from .qif import __all__ as qif_all
from .sl import *
from .sl import __all__ as sl_all
from .vdp import *
from .vdp import __all__ as vdp_all
from .wilson_cowan import *
from .wilson_cowan import __all__ as wilson_cowan_all
from .wong_wang import *
from .wong_wang import __all__ as wong_wang_all

__all__ = forward_model_all + coupling_all + jansen_rit_all + noise_all + wilson_cowan_all + wong_wang_all + hopf_all
__all__ = __all__ + param_all + fhn_all + linear_all + vdp_all + qif_all + sl_all
del forward_model_all, coupling_all, jansen_rit_all, noise_all, wilson_cowan_all, wong_wang_all, hopf_all
del param_all, fhn_all, linear_all, vdp_all, qif_all, sl_all
