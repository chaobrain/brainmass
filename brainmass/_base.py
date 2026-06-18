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

"""Shared base class for brainmass neural-mass models.

This module defines :class:`NeuralMassDynamics`, the single source of truth for
the *integrator-selection* contract that the two-variable neural-mass models
share. Before this base existed, :class:`~brainmass.XY_Oscillator` (the parent
of :class:`~brainmass.HopfStep`, :class:`~brainmass.StuartLandauStep` and
:class:`~brainmass.VanDerPolStep`), :class:`~brainmass.FitzHughNagumoStep` and
:class:`~brainmass.MontbrioPazoRoxinStep` each re-implemented the same
``update`` branch::

    if self.method == 'exp_euler':
        x = brainstate.nn.exp_euler_step(self.dx, ...)
        ...
    else:
        ode = getattr(braintools.quad, f'ode_{self.method}_step')
        x, ... = ode(self.derivative, state, t, *inputs)

That duplication is now owned by :meth:`NeuralMassDynamics._solve_step`. Each
model keeps its own ``update`` (so its public keyword signature is unchanged)
but delegates the actual stepping to the base.
"""

import braintools
import brainunit as u

import brainstate

__all__ = [
    'NeuralMassDynamics',
]


class NeuralMassDynamics(brainstate.nn.Dynamics):
    """Base class for neural-mass models with selectable time integration.

    This is an *implementation* base: it does not allocate any state or
    parameters itself. Concrete subclasses are responsible for

    - storing the integration ``method`` string (set in ``__init__``),
    - allocating their :class:`~brainstate.HiddenState` variables in
      ``init_state``,
    - defining the per-variable right-hand-side methods (e.g. ``dx``/``dy``),
    - defining a ``derivative(state, t, *inputs)`` aggregator returning the
      tuple of time derivatives, and
    - implementing a public ``update(...)`` that assembles the step arguments
      and calls :meth:`_solve_step`.

    The only behaviour provided here is the integrator dispatch, so that the
    ``exp_euler`` versus ``braintools.quad.ode_<method>_step`` choice lives in
    exactly one place.

    See Also
    --------
    brainmass.XY_Oscillator : two-variable oscillator base built on this class.
    """

    def _solve_step(self, exp_euler_specs, ode_state, ode_inputs):
        """Advance the state by one step, dispatching on ``self.method``.

        Parameters
        ----------
        exp_euler_specs : sequence of tuple
            One ``(d_func, primary_value, *extra_args)`` entry per state
            variable, forwarded verbatim to
            :func:`brainstate.nn.exp_euler_step` when ``self.method`` is
            ``'exp_euler'`` (the default). The entries are evaluated against the
            *current* state values supplied by the caller, so the per-variable
            updates all read the pre-step state regardless of evaluation order.
        ode_state : tuple
            Current state values, passed as the second positional argument to
            the ``braintools.quad`` solver for any non-exp-euler ``method``.
        ode_inputs : tuple
            Extra inputs forwarded to ``self.derivative`` by the solver (the
            same external inputs that appear in ``exp_euler_specs``).

        Returns
        -------
        tuple
            Updated state values, in the same order as ``ode_state`` and
            ``exp_euler_specs``.

        Notes
        -----
        For ``method != 'exp_euler'`` the current time ``t`` is read from
        :func:`brainstate.environ.get` (defaulting to ``0 * u.ms``); the models
        in this package are autonomous, so the value only matters for
        non-autonomous extensions.
        """
        if self.method == 'exp_euler':
            return tuple(
                brainstate.nn.exp_euler_step(*spec) for spec in exp_euler_specs
            )
        ode_step = getattr(braintools.quad, f'ode_{self.method}_step')
        t = brainstate.environ.get('t', 0. * u.ms)
        return ode_step(self.derivative, ode_state, t, *ode_inputs)
