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

from typing import Callable

import brainstate
import braintools
import brainunit as u
from brainstate.nn import Dynamics, Param

from ._base import NeuralMassDynamics
from .noise import Noise
from .typing import Parameter

__all__ = [
    'ThresholdLinearStep',
    'LinearStep',
]


class ThresholdLinearStep(Dynamics):
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
    tau_E : Parameter , optional
        Excitatory time constant with unit of time (e.g., ``2e-2 * u.second``).
        Default is ``2e-2 * u.second``.
    tau_I : Parameter , optional
        Inhibitory time constant with unit of time (e.g., ``1e-2 * u.second``).
        Default is ``1e-2 * u.second``.
    beta_E : Parameter , optional
        Excitatory gain (dimensionless). Default is ``0.066``.
    beta_I : Parameter , optional
        Inhibitory gain (dimensionless). Default is ``0.351``.
    init_E : Callable, optional
        Parameter  for the excitatory rate state ``E``. Default is
        ``braintools.init.Constant(0.0)``.
    init_I : Callable, optional
        Parameter  for the inhibitory rate state ``I``. Default is
        ``braintools.init.Constant(0.0)``.
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
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        tau_E: Parameter  = 2e-2 * u.second,
        tau_I: Parameter  = 1e-2 * u.second,
        beta_E: Parameter  = .066,
        beta_I: Parameter  = .351,
        init_E: Callable = braintools.init.Constant(0.0),
        init_I: Callable = braintools.init.Constant(0.0),
        noise_E: Noise = None,
        noise_I: Noise = None,
    ):
        super().__init__(in_size)

        # parameters
        self.tau_E = Param.init(tau_E, self.varshape)
        self.tau_I = Param.init(tau_I, self.varshape)
        self.beta_E = Param.init(beta_E, self.varshape)
        self.beta_I = Param.init(beta_I, self.varshape)

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
        self.E = brainstate.HiddenState.init(self.init_E, self.varshape, batch_size)
        self.I = brainstate.HiddenState.init(self.init_I, self.varshape, batch_size)

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
        tau_E = self.tau_E.value()
        tau_I = self.tau_I.value()
        beta_E = self.beta_E.value()
        beta_I = self.beta_I.value()

        E_inp = 0. if E_inp is None else E_inp
        if self.noise_E is not None:
            E_inp = E_inp + self.noise_E()
        I_inp = 0. if I_inp is None else I_inp
        if self.noise_I is not None:
            I_inp = I_inp + self.noise_I()

        dE = lambda E: (-E + beta_E * u.math.maximum(E_inp, 0.)) / tau_E
        E = brainstate.nn.exp_euler_step(dE, self.E.value)
        self.E.value = u.math.maximum(E, 0.)

        dI = lambda I: (I - beta_I * u.math.maximum(I_inp, 0.)) / tau_I
        I = brainstate.nn.exp_euler_step(dI, self.I.value)
        self.I.value = u.math.maximum(I, 0.)
        return self.E.value


class LinearStep(NeuralMassDynamics):
    r"""Linear neural-mass node with damping (TVB ``Linear`` model).

    A single-state linear model with a damping coefficient, used in The Virtual
    Brain as a canonical baseline for validating simulation pipelines and network
    coupling without nonlinear complications [1]_. Despite its simplicity it is a
    genuine network node: the long-range and local coupling enter additively.

    .. math::

       \frac{dx}{dt} = \gamma\,x + c,

    where :math:`x` is the (dimensionless) node activity, :math:`\gamma` the
    damping coefficient, and :math:`c` the summed coupling/external input. For a
    stable node :math:`\gamma` must be negative and its magnitude should exceed the
    node's in-degree; with :math:`c = 0` the activity relaxes exponentially,
    :math:`x(t) = x_0 e^{\gamma t}`.

    This is distinct from :class:`~brainmass.ThresholdLinearStep`, which is a
    two-population (E/I) *threshold*-linear rate model.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node. An ``int`` or tuple of ``int``; ``gamma``
        broadcasts to this shape.
    gamma : Parameter, optional
        Damping coefficient (dimensionless). Must be negative for stability.
        Default is ``-10.0``.
    init_x : Callable, optional
        Initializer for the activity state ``x``. Default is
        ``braintools.init.Constant(0.01)``.
    noise_x : Noise or None, optional
        Additive noise process. If provided, its output is added to the input
        ``x_inp`` at each update. Default is ``None``.
    method : str, optional
        Integration method, ``'exp_euler'`` (default) or any ``braintools.quad``
        method (e.g. ``'rk4'``). The exponential-Euler step is *exact* for this
        linear system.

    Attributes
    ----------
    x : brainstate.HiddenState
        Node activity (dimensionless). Shape ``(batch?,) + in_size``.

    Notes
    -----
    The state is dimensionless and :meth:`dx` carries unit ``1/ms``, so an
    exponential-Euler step with ``dt`` in milliseconds is consistent (the
    convention shared by the other ``*Step`` models in this package).

    References
    ----------
    .. [1] P. Sanz-Leon, S. A. Knock, A. Spiegler, V. K. Jirsa (2015). Mathematical
       framework for large-scale brain network modeling in The Virtual Brain.
       NeuroImage, 111, 385-430. https://doi.org/10.1016/j.neuroimage.2015.01.002

    Examples
    --------
    .. code-block:: python

       >>> import brainmass
       >>> import brainstate
       >>> import brainunit as u
       >>> model = brainmass.LinearStep(in_size=1, gamma=-5.0)
       >>> _ = brainstate.nn.init_all_states(model)
       >>> with brainstate.environ.context(dt=0.1 * u.ms):
       ...     x = model.update()
       >>> x.shape
       (1,)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,
        gamma: Parameter = -10.0,
        init_x: Callable = braintools.init.Constant(0.01),
        noise_x: Noise = None,
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)
        self.gamma = Param.init(gamma, self.varshape)

        assert callable(init_x), 'init_x must be callable'
        assert isinstance(noise_x, Noise) or noise_x is None, 'noise_x must be a Noise instance or None'
        self.init_x = init_x
        self.noise_x = noise_x
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Allocate the activity state ``x``.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.x = brainstate.HiddenState.init(self.init_x, self.varshape, batch_size)

    def dx(self, x, x_inp):
        """Right-hand side for the activity ``x``.

        Parameters
        ----------
        x : array-like
            Current activity (dimensionless).
        x_inp : array-like or scalar
            Summed coupling/external input (includes noise if enabled).

        Returns
        -------
        array-like
            Time derivative ``dx/dt`` with unit ``1/ms``.
        """
        return (self.gamma.value() * x + x_inp) / u.ms

    def derivative(self, state, t, x_inp):
        (x,) = state
        return (self.dx(x, x_inp),)

    def update(self, x_inp=None):
        """Advance the node by one time step.

        Parameters
        ----------
        x_inp : array-like or scalar or None, optional
            Summed coupling/external input. If ``None``, treated as zero. If
            ``noise_x`` is set, its output is added. Default is ``None``.

        Returns
        -------
        array-like
            The updated activity ``x``, same shape as the internal state.
        """
        x_inp = 0.0 if x_inp is None else x_inp
        if self.noise_x is not None:
            x_inp = x_inp + self.noise_x()

        (x,) = self._solve_step(
            exp_euler_specs=((self.dx, self.x.value, x_inp),),
            ode_state=(self.x.value,),
            ode_inputs=(x_inp,),
        )
        self.x.value = x
        return x
