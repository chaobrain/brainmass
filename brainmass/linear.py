# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Callable

import braintools
import brainstate
import brainunit as u

from ._typing import Initializer
from .noise import Noise

__all__ = [
    'ThresholdLinearModel',
]


class ThresholdLinearModel(brainstate.nn.Dynamics):
    r"""Threshold-linear two-population rate model.

    This model describes excitatory (E) and inhibitory (I) population rates
    with threshold-linear input-output functions [1]_. The continuous-time
    dynamics are

    .. math::

       \begin{aligned}
        &\tau_{E} \frac{d \nu_{E}}{d t} = -\nu_{E} + \beta_{E}\,[I_{E}]_{+}, \\
        &\tau_{I} \frac{d \nu_{I}}{d t} = -\nu_{I} + \beta_{I}\,[I_{I}]_{+},
       \end{aligned}

    where :math:`[x]_+ = \max(x, 0)` is the rectifier. :math:`\nu_E` and
    :math:`\nu_I` denote E and I firing rates; :math:`\tau_E` and
    :math:`\tau_I` are their intrinsic time constants; :math:`\beta_E` and
    :math:`\beta_I` are gains.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape for the E and I populations. Can be an ``int`` or a
        tuple of ``int``. All parameters are broadcastable to this shape.
    tau_E : Initializer, optional
        Excitatory time constant with unit of time (e.g., ``2e-2 * u.second``).
        Default is ``2e-2 * u.second``.
    tau_I : Initializer, optional
        Inhibitory time constant with unit of time (e.g., ``1e-2 * u.second``).
        Default is ``1e-2 * u.second``.
    beta_E : Initializer, optional
        Excitatory gain (dimensionless). Default is ``0.066``.
    beta_I : Initializer, optional
        Inhibitory gain (dimensionless). Default is ``0.351``.
    init_E : Callable, optional
        Initializer for the excitatory rate state ``E``. Default is
        ``braintools.init.ZeroInit()``.
    init_I : Callable, optional
        Initializer for the inhibitory rate state ``I``. Default is
        ``braintools.init.ZeroInit()``.
    noise_E : Noise or None, optional
        Additive noise process for the E population. If provided, called each
        update and added to ``E_inp``. Default is ``None``.
    noise_I : Noise or None, optional
        Additive noise process for the I population. If provided, called each
        update and added to ``I_inp``. Default is ``None``.

    Attributes
    ----------
    E : brainstate.HiddenState
        Excitatory rate state (dimensionless). Shape equals ``(batch?,) + in_size``
        after ``init_state``.
    I : brainstate.HiddenState
        Inhibitory rate state (dimensionless). Shape equals ``(batch?,) + in_size``
        after ``init_state``.

    Notes
    -----
    - Time derivatives implicitly have unit ``1/time`` determined by the
      ``brainunit`` time units of ``tau_E`` and ``tau_I``.
    - The rectification is applied to the external drives, and the states are
      clipped to be non-negative after each update.

    References
    ----------
    .. [1] Chaudhuri, R., et al. (2015). A large-scale circuit mechanism for
       hierarchical dynamical processing in the primate cortex. Neuron, 88(2),
       419–431.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        tau_E: Initializer = 2e-2 * u.second,
        tau_I: Initializer = 1e-2 * u.second,
        beta_E: Initializer = .066,
        beta_I: Initializer = .351,
        init_E: Callable = braintools.init.ZeroInit(),
        init_I: Callable = braintools.init.ZeroInit(),
        noise_E: Noise = None,
        noise_I: Noise = None,
    ):
        super().__init__(in_size)

        # parameters
        self.tau_E = braintools.init.param(tau_E, self.varshape)
        self.tau_I = braintools.init.param(tau_I, self.varshape)
        self.beta_E = braintools.init.param(beta_E, self.varshape)
        self.beta_I = braintools.init.param(beta_I, self.varshape)

        # initializers and noise processes
        assert callable(init_E), "init_E must be a callable function"
        assert callable(init_I), "init_I must be a callable function"
        assert isinstance(noise_E, Noise) or noise_E is None, "noise_E must be an instance of Noise or None"
        assert isinstance(noise_I, Noise) or noise_I is None, "noise_I must be an instance of Noise or None"
        self.init_E = init_E
        self.init_I = init_I
        self.noise_E = noise_E
        self.noise_I = noise_I

    def init_state(self, batch_size=None, **kwargs):
        """Initialize excitatory and inhibitory states.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.E = brainstate.HiddenState(braintools.init.param(self.init_E, self.varshape, batch_size))
        self.I = brainstate.HiddenState(braintools.init.param(self.init_I, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        """Reset E and I states using the configured initializers.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional batch dimension for reinitialization. If ``None``, keeps
            current batch shape but resets values. Default is ``None``.
        """
        self.E.value = braintools.init.param(self.init_E, self.varshape, batch_size)
        self.I.value = braintools.init.param(self.init_I, self.varshape, batch_size)

    def update(self, E_inp=None, I_inp=None):
        """Advance the system by one time step.

        Parameters
        ----------
        E_inp : array-like or scalar or None, optional
            External input drive to the excitatory population. If ``None``,
            treated as zero. If ``noise_E`` is set, its output is added.
        I_inp : array-like or scalar or None, optional
            External input drive to the inhibitory population. If ``None``,
            treated as zero. If ``noise_I`` is set, its output is added.

        Returns
        -------
        array-like
            The updated excitatory rate ``E`` with the same shape as the
            internal state.

        Notes
        -----
        Uses exponential-Euler updates via ``brainstate.nn.exp_euler_step`` with
        rectification applied to inputs and non-negativity enforced on states.
        """
        E_inp = 0. if E_inp is None else E_inp
        if self.noise_E is not None:
            E_inp = E_inp + self.noise_E()
        I_inp = 0. if I_inp is None else I_inp
        if self.noise_I is not None:
            I_inp = I_inp + self.noise_I()

        dE = lambda E: (-E + self.beta_E * u.math.maximum(E_inp, 0.)) / self.tau_E
        E = brainstate.nn.exp_euler_step(dE, self.E.value)
        self.E.value = u.math.maximum(E, 0.)

        dI = lambda I: (I - self.beta_I * u.math.maximum(I_inp, 0.)) / self.tau_I
        I = brainstate.nn.exp_euler_step(dI, self.I.value)
        self.I.value = u.math.maximum(I, 0.)
        return self.E.value
