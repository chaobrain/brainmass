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

from typing import Callable, Optional

import braintools
import brainstate
import brainunit as u

from ._typing import Initializer
from .noise import Noise

__all__ = [
    'KuramotoNetwork',
]


class KuramotoNetwork(brainstate.nn.Dynamics):
    r"""Kuramoto phase oscillator network (with optional phase lag).

    Implements the (Sakaguchi–)Kuramoto model for a population of coupled
    phase oscillators with states :math:`\theta_i \in \mathbb{R}` evolving as

    .. math::

        \dot{\theta}_i = \omega_i + \frac{K}{\mathcal{N}}
        \sum_{j=1}^{N} w_{ij} \sin\bigl(\theta_j - \theta_i - \alpha\bigr)
        + I_i(t),

    where :math:`\omega_i` is the natural frequency, :math:`K` is a global
    coupling strength, :math:`w_{ij}` are (optional) connection weights, and
    :math:`\alpha` is a phase lag. If no weight matrix is provided, the model
    defaults to all-to-all coupling with :math:`w_{ij}=1` and (by default)
    normalization :math:`\mathcal{N} = N`.

    Parameters
    ----------
    in_size : brainstate.typing.Size
        Spatial shape for the oscillator array. For network coupling, the last
        dimension is treated as the node index :math:`N`.
    omega : Initializer, optional
        Natural frequency for each oscillator (dimensionless here; overall
        derivative has unit ``1/ms`` due to division by ``u.ms``). Broadcastable
        to ``in_size``. Default is ``0.0``.
    K : Initializer, optional
        Global coupling strength (dimensionless). Broadcastable to ``in_size``
        when used without explicit ``conn``. Default is ``0.0``.
    alpha : Initializer, optional
        Phase lag :math:`\alpha` (dimensionless, radians). Broadcastable to
        ``in_size``. Default is ``0.0``.
    conn : array-like or None, optional
        Connection weights :math:`w_{ij}` with shape ``(N, N)`` or flattened
        ``(N*N,)``. If ``None`` (default), uses all-to-all unit weights.
    normalize_by_n : bool, optional
        If ``True`` divides the coupling sum by :math:`N`. Default is ``True``.
    exclude_self : bool, optional
        If ``True`` excludes self-coupling terms (diagonal), which matters when
        ``alpha != 0``. Default is ``True``.
    noise_theta : Noise or None, optional
        Additive noise process for the phase dynamics. If provided, its output
        is added to ``theta_inp`` each update. Default is ``None``.
    init_theta : Callable, optional
        Initializer for the phase state ``theta`` (dimensionless, radians).
        Default is ``braintools.init.Uniform(0.0, 2 * u.math.pi)``.

    Attributes
    ----------
    theta : brainstate.HiddenState
        Phase state with shape ``(batch?,) + in_size`` after ``init_state``.

    Notes
    -----
    - Time derivatives returned by :meth:`dtheta` carry unit ``1/ms`` to be
      consistent with explicit (exponential) Euler integration.
    - If ``conn`` is provided, it broadcasts across any leading batch/time
      dimensions in ``theta``. If absent, coupling is computed from the
      pairwise phase differences without an explicit matrix.
    - This implementation does not wrap the phase to ``[0, 2\pi)``. If phase
      wrapping is required for downstream use or visualization, apply it
      externally.

    References
    ----------
    - Kuramoto, Y. (1975). Self-entrainment of a population of coupled
      non-linear oscillators. In International Symposium on Mathematical
      Problems in Theoretical Physics.
    - Sakaguchi, H., & Kuramoto, Y. (1986). A soluble active rotator model
      showing phase transitions via mutual entertainment. Progress of
      Theoretical Physics, 76(3), 576–581.
    """

    def __init__(
        self,
        in_size: brainstate.typing.Size,

        omega: Initializer = 0.0,
        K: Initializer = 0.0,
        alpha: Initializer = 0.0,
        conn: Optional[brainstate.typing.ArrayLike] = None,
        normalize_by_n: bool = True,
        exclude_self: bool = True,

        noise_theta: Noise = None,
        init_theta: Callable = braintools.init.Uniform(0.0, 2.0 * u.math.pi),
    ):
        super().__init__(in_size=in_size)

        # parameters
        self.omega = braintools.init.param(omega, self.varshape)
        self.K = braintools.init.param(K, self.varshape)
        self.alpha = braintools.init.param(alpha, self.varshape)

        # coupling configuration
        self.conn = None if conn is None else u.math.asarray(conn)
        self.normalize_by_n = normalize_by_n
        self.exclude_self = exclude_self

        # noise and initializer
        assert isinstance(noise_theta, Noise) or noise_theta is None, (
            "noise_theta must be a Noise instance or None."
        )
        self.noise_theta = noise_theta
        self.init_theta = init_theta

    def init_state(self, batch_size=None, **kwargs):
        """Initialize phase state ``theta``.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional leading batch dimension. If ``None``, no batch dimension is
            used. Default is ``None``.
        """
        self.theta = brainstate.HiddenState(
            braintools.init.param(self.init_theta, self.varshape, batch_size)
        )

    def reset_state(self, batch_size=None, **kwargs):
        """Reset phase state ``theta`` using the initializer.

        Parameters
        ----------
        batch_size : int or None, optional
            Optional batch dimension for reinitialization. If ``None``, keeps
            current batch shape but resets values. Default is ``None``.
        """
        self.theta.value = braintools.init.param(self.init_theta, self.varshape, batch_size)

    def _pairwise_coupling(self, theta):
        """Compute coupling term for each oscillator.

        Parameters
        ----------
        theta : array-like
            Current phases with shape ``(..., N)`` over the last dimension.

        Returns
        -------
        array-like
            Coupling drive with shape ``(..., N)`` (dimensionless). This is the
            quantity added to :math:`\omega` before scaling by ``1/ms``.

        Notes
        -----
        - If ``conn`` is ``None``, all-to-all unit weights are assumed.
        - If ``exclude_self`` is ``True``, diagonal self-interactions are
          removed (relevant only if ``alpha != 0``).
        - If ``normalize_by_n`` is ``True``, the sum is divided by :math:`N`.
        """
        # theta: (..., N)
        if theta.ndim < 1:
            return 0.0

        n = theta.shape[-1]

        # Broadcast pairwise differences: (..., N_out, N_in)
        theta_i = u.math.expand_dims(theta, axis=-1)      # (..., N, 1)
        theta_j = u.math.expand_dims(theta, axis=-2)      # (..., 1, N)
        delta = theta_j - theta_i - self.alpha  # broadcast alpha
        s_mat = u.math.sin(delta)               # (..., N, N)

        if self.exclude_self:
            # Zero out diagonal terms to remove self-coupling.
            eye = u.math.eye(n)
            s_mat = s_mat * (1.0 - eye)  # broadcasts across leading dims

        if self.conn is not None:
            conn = self.conn
            if conn.ndim == 1:
                if conn.size % n != 0:
                    raise ValueError(
                        f'Flattened connection length {conn.size} is not divisible by N={n}.'
                    )
                n_in = conn.size // n
                if n_in != n:
                    raise ValueError(
                        f'For Kuramoto, expected square connectivity (N={n}); got N_in={n_in}.'
                    )
                conn2d = u.math.reshape(conn, (n, n))
            elif conn.ndim == 2:
                if conn.shape != (n, n):
                    raise ValueError(
                        f'Connectivity must be square (N,N)={(n,n)}; got {conn.shape}.'
                    )
                conn2d = conn
            else:
                raise ValueError('Connectivity must be 1D flattened or 2D matrix.')

            weighted = s_mat * conn2d  # (..., N, N)
            coupling = weighted.sum(axis=-1)  # (..., N)
        else:
            coupling = s_mat.sum(axis=-1)  # (..., N)

        if self.normalize_by_n and n > 0:
            coupling = coupling / n

        return self.K * coupling

    def dtheta(self, theta, drive):
        """Phase dynamics right-hand side.

        Parameters
        ----------
        theta : array-like
            Current phase (unused here since ``drive`` already encodes the
            dependence via :meth:`_pairwise_coupling`).
        drive : array-like
            External drive combining coupling and user-provided input.

        Returns
        -------
        array-like
            Time derivative ``dtheta/dt`` with unit ``1/ms``.
        """
        return (self.omega + drive) / u.ms

    def update(self, theta_inp=None):
        """Advance the system by one time step.

        Parameters
        ----------
        theta_inp : array-like or scalar or None, optional
            External input to the phase dynamics. If ``None``, treated as zero.
            If ``noise_theta`` is set, its output is added. Default is ``None``.

        Returns
        -------
        array-like
            The updated phase ``theta`` with the same shape as the internal
            state.

        Notes
        -----
        - Computes coupling from the current phases and performs an
          exponential-Euler step using ``brainstate.nn.exp_euler_step``.
        - No phase wrapping is applied.
        """
        theta_inp = 0.0 if theta_inp is None else theta_inp
        if self.noise_theta is not None:
            theta_inp = theta_inp + self.noise_theta()

        coup = self._pairwise_coupling(self.theta.value)
        total_drive = coup + theta_inp

        theta_next = brainstate.nn.exp_euler_step(self.dtheta, self.theta.value, total_drive)
        self.theta.value = theta_next
        return theta_next
