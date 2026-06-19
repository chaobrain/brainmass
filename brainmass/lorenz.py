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

from typing import Callable

import braintools
import brainunit as u

import brainstate
from brainstate.nn import Param
from ._base import NeuralMassDynamics
from .noise import Noise
from .typing import Parameter

__all__ = [
    'LorenzStep',
]


class LorenzStep(NeuralMassDynamics):
    r"""Lorenz chaotic system as a network node.

    The classic three-variable Lorenz (1963) system [1]_, a low-dimensional
    deterministic flow that exhibits sensitive dependence on initial conditions
    (deterministic chaos) for the standard parameters. The Virtual Brain ships it
    as a canonical non-neural test fixture for validating integration and network
    coupling. Structural coupling enters the :math:`x` equation:

    .. math::

       \begin{aligned}
       \dot x &= \sigma\,(y - x) + c, \\
       \dot y &= x\,(\rho - z) - y, \\
       \dot z &= x\,y - \beta\,z,
       \end{aligned}

    where :math:`c` is the coupling/external input to :math:`x`. With the default
    :math:`(\sigma, \rho, \beta) = (10, 28, 8/3)` the system has a positive largest
    Lyapunov exponent (:math:`\approx 0.9` per natural time unit): nearby
    trajectories separate exponentially while remaining bounded on the Lorenz
    attractor.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape of the node. An ``int`` or tuple of ``int``; all parameters
        broadcast to this shape.
    sigma : Parameter, optional
        Prandtl number :math:`\sigma`. Default is ``10.0``.
    rho : Parameter, optional
        Rayleigh number :math:`\rho`. Default is ``28.0`` (chaotic regime).
    beta : Parameter, optional
        Geometric parameter :math:`\beta`. Default is ``8/3``.
    init_x, init_y, init_z : Callable, optional
        State initializers. Defaults reproduce the canonical initial condition
        ``(1.0, 1.0, 1.0)``.
    noise_x : Noise or None, optional
        Additive noise process for the :math:`x` equation. If provided, its output
        is added to the coupling input ``x_inp`` at each update. Default is ``None``.
    method : str, optional
        Integration method, ``'exp_euler'`` (default) or any ``braintools.quad``
        method (e.g. ``'rk4'``).

    Attributes
    ----------
    x, y, z : brainstate.HiddenState
        The three Lorenz coordinates (dimensionless). Shape ``(batch?,) + in_size``.

    Notes
    -----
    - State variables are dimensionless; each right-hand side carries unit ``1/ms``
      so an exponential-Euler step with ``dt`` in milliseconds is consistent. One
      "natural" Lorenz time unit therefore corresponds to ``1 ms`` here; a step of
      ``dt = 0.01 * u.ms`` reproduces the classic dimensionless ``dt = 0.01``.
    - Because the flow is chaotic, two integrations that differ only in round-off
      (or integrator) diverge after a short horizon. Trajectory comparisons must
      use a short horizon; long-run agreement is *not* expected and is not a bug.

    References
    ----------
    .. [1] E. N. Lorenz (1963). Deterministic nonperiodic flow. Journal of the
       Atmospheric Sciences, 20(2), 130-141.
       https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2

    Examples
    --------
    .. code-block:: python

       >>> import brainmass
       >>> import brainstate
       >>> import brainunit as u
       >>> model = brainmass.LorenzStep(in_size=1)
       >>> _ = brainstate.nn.init_all_states(model)
       >>> with brainstate.environ.context(dt=0.01 * u.ms):
       ...     x = model.update()
       >>> x.shape
       (1,)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        # model parameters
        sigma: Parameter = 10.0,
        rho: Parameter = 28.0,
        beta: Parameter = 8.0 / 3.0,

        # initializers / noise
        init_x: Callable = braintools.init.Constant(1.0),
        init_y: Callable = braintools.init.Constant(1.0),
        init_z: Callable = braintools.init.Constant(1.0),
        noise_x: Noise = None,
        method: str = 'exp_euler',
    ):
        super().__init__(in_size)

        self.sigma = Param.init(sigma, self.varshape)
        self.rho = Param.init(rho, self.varshape)
        self.beta = Param.init(beta, self.varshape)

        for init in (init_x, init_y, init_z):
            assert callable(init), 'state initializers must be callable'
        assert isinstance(noise_x, Noise) or noise_x is None, 'noise_x must be a Noise instance or None'
        self.init_x = init_x
        self.init_y = init_y
        self.init_z = init_z
        self.noise_x = noise_x
        self.method = method

    def init_state(self, batch_size=None, **kwargs):
        """Allocate the three Lorenz coordinates at their canonical initial values.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.x = brainstate.HiddenState.init(self.init_x, self.varshape, batch_size)
        self.y = brainstate.HiddenState.init(self.init_y, self.varshape, batch_size)
        self.z = brainstate.HiddenState.init(self.init_z, self.varshape, batch_size)

    def dx(self, x, y, x_inp):
        """Right-hand side for ``x`` (unit ``1/ms``); ``x_inp`` is the coupling input."""
        return (self.sigma.value() * (y - x) + x_inp) / u.ms

    def dy(self, y, x, z):
        """Right-hand side for ``y`` (unit ``1/ms``)."""
        return (x * (self.rho.value() - z) - y) / u.ms

    def dz(self, z, x, y):
        """Right-hand side for ``z`` (unit ``1/ms``)."""
        return (x * y - self.beta.value() * z) / u.ms

    def derivative(self, state, t, x_inp):
        x, y, z = state
        return self.dx(x, y, x_inp), self.dy(y, x, z), self.dz(z, x, y)

    def update(self, x_inp=None):
        """Advance the Lorenz node by one time step.

        Parameters
        ----------
        x_inp : array-like or scalar or None, optional
            Structural coupling/external input to the ``x`` equation. If ``None``,
            treated as zero. If ``noise_x`` is set, its output is added. Default is
            ``None``.

        Returns
        -------
        array-like
            The updated ``x`` coordinate, same shape as the internal state.
        """
        x_inp = 0.0 if x_inp is None else x_inp
        if self.noise_x is not None:
            x_inp = x_inp + self.noise_x()

        x, y, z = self._solve_step(
            exp_euler_specs=(
                (self.dx, self.x.value, self.y.value, x_inp),
                (self.dy, self.y.value, self.x.value, self.z.value),
                (self.dz, self.z.value, self.x.value, self.y.value),
            ),
            ode_state=(self.x.value, self.y.value, self.z.value),
            ode_inputs=(x_inp,),
        )
        self.x.value = x
        self.y.value = y
        self.z.value = z
        return x
