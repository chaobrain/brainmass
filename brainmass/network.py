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

"""A reusable whole-brain connectome builder.

:class:`Network` encapsulates the wiring that is otherwise copy-pasted across the
examples -- zero the structural-connectivity diagonal, turn a distance matrix into
conduction delays (``distance / speed``), tile the neighbour indices, prefetch the
delayed source state, and feed a :class:`~brainmass.DiffusiveCoupling` (or additive /
Laplacian) current back into the node -- into one ``brainstate`` ``Module`` that the
:class:`~brainmass.Simulator` can drive directly.
"""

import braintools
import brainstate
import brainunit as u
import numpy as np
from brainstate.nn import Param

from .coupling import DiffusiveCoupling, AdditiveCoupling, LaplacianConnParam
from .utils import delay_index

__all__ = [
    'Network',
]


class Network(brainstate.nn.Module):
    r"""Wire a node model into a delay-coupled whole-brain network.

    The network reads a single ``node`` (a ``Dynamics`` sized for *N* regions) and
    couples its regions through a structural-connectivity matrix, optionally with
    distance-dependent conduction delays. ``update`` computes the coupling current
    and feeds it back as the node's first positional input -- the same
    ``current = coupling(); node(current)`` idiom hand-written in the examples.

    Parameters
    ----------
    node : brainstate.nn.Dynamics
        The per-region dynamics, already sized for *N* regions (``node.varshape[0]
        == N``) and carrying any per-region noise. Exposed afterwards as
        :attr:`node`; its states are reachable as ``network.node.<coupled_var>``.
    conn : array_like or brainstate.nn.Param
        Structural connectivity. A plain ``(N, N)`` (or flattened ``(N * N,)``)
        array has its diagonal zeroed unless ``self_connection`` is ``True``; a
        ``Param`` / :class:`~brainmass.LaplacianConnParam` is passed through
        untouched (the caller owns its structure) and may be trained.
    distance : array_like, optional
        Inter-region distance matrix ``(N, N)``. Combined with ``speed`` into
        conduction delays ``distance / speed``. If either ``distance`` or ``speed``
        is ``None`` the coupling is instantaneous (zero delays).
    speed : float or brainunit.Quantity, optional
        Conduction speed. If ``distance`` carries length units and ``speed`` carries
        ``length / time`` units the delay is unit-correct; if both are plain numbers
        the quotient is interpreted as milliseconds (matching the examples).
    coupling : {'diffusive', 'additive', 'laplacian'}, default 'diffusive'
        Coupling kernel. ``'diffusive'`` uses
        :class:`~brainmass.DiffusiveCoupling` (``k * sum_j conn_ij (x_j - x_i)``);
        ``'additive'`` uses :class:`~brainmass.AdditiveCoupling`
        (``k * sum_j conn_ij x_j``); ``'laplacian'`` wraps ``conn`` in a
        :class:`~brainmass.LaplacianConnParam` and applies it additively.
    coupled_var : str
        Name of the node state variable to couple (e.g. ``'rE'``, ``'x'``, ``'V'``).
        Validated at initialisation; an unknown name raises ``ValueError``.
    k : float, brainunit.Quantity or brainstate.nn.Param, default 1.0
        Global coupling strength. Pass a trainable ``Param`` to fit it.
    delay_init : Callable, default ``braintools.init.Uniform(0., 0.05)``
        Initializer for the delay buffer's history.
    self_connection : bool, default False
        If ``False`` (default), the connectivity diagonal is zeroed (no
        self-coupling). Only applies to plain-array ``conn``.
    noise : brainstate.nn.Module, optional
        Optional network-level noise process (e.g. an :class:`~brainmass.OUProcess`
        sized for *N*). Its output is added to the coupling current each step.

    See Also
    --------
    brainmass.Simulator : drives the network and collects trajectories.
    brainmass.DiffusiveCoupling : the underlying diffusive coupling kernel.

    Notes
    -----
    The delay-buffer shape ``(max_delay_steps, N)`` does not depend on whether the
    ``(delay, index)`` pair is supplied flattened (as in ``examples/100``) or as
    ``(N, N)`` matrices (as here), so a seeded run reproduces the hand-wired
    examples bit-for-bit. The self-delay (delay-matrix diagonal) is always zeroed.

    Examples
    --------
    A four-region Hopf network driven by the simulator:

    >>> import brainmass
    >>> import brainunit as u
    >>> import numpy as np
    >>> N = 4
    >>> conn = np.ones((N, N)) * 0.1
    >>> node = brainmass.HopfStep(N, a=0.1)
    >>> net = brainmass.Network(node, conn=conn, coupled_var='x', k=0.5)
    >>> sim = brainmass.Simulator(net, dt=0.1 * u.ms)
    >>> res = sim.run(5.0 * u.ms, monitors=lambda m: m.node.x.value)
    >>> res['output'].shape
    (50, 4)
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        node,
        *,
        conn,
        distance=None,
        speed=None,
        coupling='diffusive',
        coupled_var,
        k=1.0,
        delay_init=braintools.init.Uniform(0., 0.05),
        self_connection=False,
        noise=None,
    ):
        super().__init__()
        self.node = node
        self.noise = noise
        self.coupled_var = coupled_var
        self.self_connection = self_connection
        n_node = node.varshape[0]
        self.n_node = n_node

        conn_for_coupling = self._build_conn(conn, n_node, self_connection)
        delay_time = self._build_delays(distance, speed, n_node)
        idx = delay_index(n_node)

        # Delayed read of the coupled source state, shaped (..., N, N).
        src = node.prefetch_delay(coupled_var, (delay_time, idx), init=delay_init)

        if coupling == 'diffusive':
            tgt = node.prefetch(coupled_var)
            self.coupling = DiffusiveCoupling(src, tgt, conn_for_coupling, k=k)
        elif coupling == 'additive':
            self.coupling = AdditiveCoupling(src, conn_for_coupling, k=k)
        elif coupling == 'laplacian':
            lap = (
                conn_for_coupling
                if isinstance(conn_for_coupling, LaplacianConnParam)
                else LaplacianConnParam(conn_for_coupling)
            )
            self.coupling = AdditiveCoupling(src, lap, k=k)
        else:
            raise ValueError(
                f"coupling must be 'diffusive', 'additive' or 'laplacian'; "
                f"got {coupling!r}."
            )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _build_conn(conn, n_node, self_connection):
        """Validate connectivity; zero the diagonal of a plain array if asked."""
        if isinstance(conn, Param):
            # A Param / LaplacianConnParam owns its own structure (and may train).
            return conn
        conn_arr = u.math.asarray(conn)
        if conn_arr.ndim == 1:
            if conn_arr.size != n_node * n_node:
                raise ValueError(
                    f"flattened conn has length {conn_arr.size}, expected "
                    f"{n_node * n_node} (= {n_node} x {n_node})."
                )
            conn_arr = conn_arr.reshape(n_node, n_node)
        elif conn_arr.ndim == 2:
            if conn_arr.shape != (n_node, n_node):
                raise ValueError(
                    f"conn has shape {conn_arr.shape}, expected "
                    f"({n_node}, {n_node})."
                )
        else:
            raise ValueError(
                f"conn must be 1-D (flattened) or 2-D; got {conn_arr.ndim}-D."
            )
        if not self_connection:
            conn_arr = conn_arr * (1.0 - np.eye(n_node))
        return conn_arr

    @staticmethod
    def _build_delays(distance, speed, n_node):
        """Conduction delays ``distance / speed`` (ms); zeros if either is None."""
        if distance is None or speed is None:
            return u.math.zeros((n_node, n_node)) * u.ms
        raw = u.math.asarray(distance) / speed
        # Plain quotient is interpreted as milliseconds (matching the examples);
        # a unit-carrying quotient is already a time and is used as-is.
        delay_time = raw if isinstance(raw, u.Quantity) else raw * u.ms
        # No self-delay (zero the diagonal).
        return delay_time * (1.0 - np.eye(n_node))

    # -------------------------------------------------------------- simulation

    @brainstate.nn.call_order(1)
    def init_state(self, *args, **kwargs):
        """Validate ``coupled_var`` once the node's states exist."""
        names = {
            name for name, val in vars(self.node).items()
            if isinstance(val, brainstate.State)
        }
        if self.coupled_var not in names:
            raise ValueError(
                f"coupled_var {self.coupled_var!r} is not a state variable of "
                f"{type(self.node).__name__}; available states: {sorted(names)}."
            )

    def update(self, *node_inputs):
        """Advance one step: coupling current (+ noise) -> node.

        Parameters
        ----------
        *node_inputs
            Extra inputs forwarded to the node after the coupling current (e.g. a
            second drive supplied by :meth:`brainmass.Simulator.run`). The coupling
            current is always the node's first positional input.

        Returns
        -------
        Any
            The node's ``update`` return value.
        """
        current = self.coupling()
        if self.noise is not None:
            current = current + self.noise()
        return self.node(current, *node_inputs)
