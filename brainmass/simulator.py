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

"""A thin, brainstate-native simulation driver.

:class:`Simulator` collapses the copy-pasted ``environ.set(dt)`` ->
``init_all_states()`` -> ``for_loop(step_run, indices)`` -> collect-``.value``
idiom into one reusable object that drives any ``brainstate`` ``Dynamics`` /
``Module`` -- a single node or a whole-brain network -- and returns a unit-aware,
structured result.
"""

import math
import warnings

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    'Simulator',
]


class Simulator:
    r"""Drive a ``Dynamics`` / ``Module`` and collect monitored trajectories.

    The simulator wraps the standard brainmass run loop -- set ``dt``,
    initialise states, step the model inside ``environ.context(i=, t=)`` with
    :func:`brainstate.transform.for_loop`, and stack the recorded values -- in a
    single :meth:`run` call. It reimplements none of that machinery; it only
    composes it.

    Parameters
    ----------
    model : brainstate.nn.Dynamics or brainstate.nn.Module
        The model to drive. Exposed afterwards as :attr:`model`. May be a single
        node or a network whose ``update`` reads its own internal coupling.
    dt : brainunit.Quantity, optional
        Integration time step. If ``None`` (default), the step is read from the
        global environment (``brainstate.environ.get('dt')``) at :meth:`run`
        time; if that is also unset, :meth:`run` raises ``ValueError``.

    See Also
    --------
    brainmass.objectives : composable loss builders over simulated trajectories.

    Notes
    -----
    The result of :meth:`run` is a plain ``dict`` (a valid JAX pytree, so it is
    safe to return through ``jit`` / ``grad`` / ``vmap``) mapping each monitor
    name to its stacked trajectory, plus a ``'ts'`` time axis. ``ts[k]`` is the
    simulation time at the **end** of the ``k``-th recorded step (the recorded
    value is the post-``update`` state).

    Examples
    --------
    >>> import brainmass
    >>> import brainunit as u
    >>> node = brainmass.HopfStep(2, a=-0.2)
    >>> sim = brainmass.Simulator(node, dt=0.1 * u.ms)
    >>> res = sim.run(10.0 * u.ms, monitors=['x'])
    >>> res['x'].shape
    (100, 2)
    >>> res['ts'].shape
    (100,)

    A derived observable (here ``E - I``) is monitored with a callable:

    >>> jr = brainmass.JansenRitStep(in_size=1)
    >>> sim = brainmass.Simulator(jr, dt=0.1 * u.ms)
    >>> res = sim.run(5.0 * u.ms, monitors=lambda m: m.eeg(), transient=1.0 * u.ms)
    >>> res['output'].shape
    (40, 1)
    """
    __module__ = 'brainmass'

    def __init__(self, model, dt=None):
        self.model = model
        self.dt = dt

    # ------------------------------------------------------------------ helpers

    def _resolve_dt(self):
        """Return the run ``dt`` (arg overrides environment); raise if unset."""
        dt = self.dt
        if dt is None:
            dt = brainstate.environ.get('dt', None)
        if dt is None:
            raise ValueError(
                "dt is not set: pass Simulator(model, dt=...) or call "
                "brainstate.environ.set(dt=...) before run()."
            )
        return dt

    @staticmethod
    def _resolve_steps(duration, dt):
        """Number of dt-steps in ``duration``; warn (and floor) if not a multiple."""
        ratio = float(u.get_magnitude(duration / dt))
        nearest = round(ratio)
        if abs(ratio - nearest) < 1e-6:
            n_steps = int(nearest)
        else:
            n_steps = int(math.floor(ratio))
            warnings.warn(
                f"duration {duration} is not an integer multiple of dt {dt}; "
                f"truncating to {n_steps} steps.",
                stacklevel=3,
            )
        if n_steps <= 0:
            raise ValueError(
                f"duration {duration} yields {n_steps} steps; must be >= 1 dt."
            )
        return n_steps

    @staticmethod
    def _resolve_transient(transient, dt, n_steps):
        """Leading steps to discard from the output (Quantity duration or count)."""
        if transient is None:
            return 0
        if isinstance(transient, u.Quantity):
            n_tr = int(round(float(u.get_magnitude(transient / dt))))
        else:
            n_tr = int(transient)
        if n_tr < 0:
            raise ValueError(f"transient must be >= 0, got {transient!r}.")
        if n_tr >= n_steps:
            raise ValueError(
                f"transient ({n_tr} steps) must be shorter than the run "
                f"({n_steps} steps); nothing would be recorded."
            )
        return n_tr

    @staticmethod
    def _resolve_sample_every(sample_every):
        """Validate and normalise the output downsampling stride."""
        if sample_every is None:
            return 1
        k = int(sample_every)
        if k < 1:
            raise ValueError(f"sample_every must be >= 1, got {sample_every!r}.")
        return k

    def _build_measure(self, monitors):
        """Build ``measure(update_return) -> dict`` for the requested monitors."""
        model = self.model
        if monitors is None:
            return lambda r: {'output': r}
        if callable(monitors):
            return lambda r: {'output': monitors(model)}
        if isinstance(monitors, dict):
            for name, spec in monitors.items():
                if isinstance(spec, str) and not hasattr(model, spec):
                    raise ValueError(
                        f"monitor {name!r} -> {spec!r} is not an attribute of {model}."
                    )

            def measure(r):
                out = {}
                for name, spec in monitors.items():
                    out[name] = getattr(model, spec).value if isinstance(spec, str) else spec(model)
                return out

            return measure
        # list / tuple of state-attribute names
        names = list(monitors)
        for name in names:
            if not hasattr(model, name):
                raise ValueError(f"monitor {name!r} is not an attribute of {model}.")
        return lambda r: {name: getattr(model, name).value for name in names}

    @staticmethod
    def _build_inputs(inputs, n_steps):
        """Build ``get_inputs(i, t) -> tuple`` forwarded to ``model.update``."""
        if inputs is None:
            return lambda i, t: ()
        if callable(inputs):
            def get_inputs(i, t):
                r = inputs(i, t)
                return r if isinstance(r, tuple) else (r,)
            return get_inputs
        # array-like: one row per step. Convert to a JAX array so it can be
        # indexed by the traced loop counter (a NumPy array cannot).
        arr = inputs if isinstance(inputs, u.Quantity) else jnp.asarray(inputs)
        length = arr.shape[0] if hasattr(arr, 'shape') else len(arr)
        if length != n_steps:
            raise ValueError(
                f"inputs has length {length} but the run has {n_steps} steps."
            )
        return lambda i, t: (arr[i],)

    # --------------------------------------------------------------------- run

    def run(
        self,
        duration,
        *,
        inputs=None,
        monitors=None,
        transient=None,
        sample_every=None,
        batch_size=None,
        init_states=True,
        jit=True,
    ):
        """Simulate ``model`` for ``duration`` and return the recorded trajectories.

        Parameters
        ----------
        duration : brainunit.Quantity
            Total simulated time. The number of integration steps is
            ``int(duration / dt)``; a non-integer multiple is floored with a
            warning.
        inputs : None, array-like or callable, optional
            External drive forwarded to ``model.update``:

            - ``None`` (default): call ``model.update()`` with no arguments.
            - array-like of shape ``(n_steps, ...)``: row ``i`` is passed as the
              single argument ``model.update(inputs[i])``.
            - ``callable(i, t)``: its return is splatted -- ``model.update(*r)``
              if it returns a tuple, else ``model.update(r)``.
        monitors : None, list of str, callable or dict, optional
            What to record each step:

            - ``None`` (default): the return value of ``update()``, under key
              ``'output'``.
            - list of state-attribute names (e.g. ``['rE']``): each
              ``getattr(model, name).value``.
            - ``callable(model) -> value``: a derived observable, under key
              ``'output'`` (e.g. ``lambda m: m.eeg()``).
            - dict mapping output name to a state-attribute name or a
              ``callable(model)``.
        transient : None, brainunit.Quantity or int, optional
            Leading transient to discard from the outputs, as a duration or a
            step count. Must be shorter than the run.
        sample_every : None or int, optional
            Record every ``k``-th step (output downsampling; generalises the
            Jansen-Rit TR loop). ``None`` records every step.
        batch_size : None or int, optional
            If given, initialise states with this batch size (batched initial
            conditions); outputs gain a leading batch axis.
        init_states : bool, default True
            Call ``init_all_states`` before running. Set ``False`` to continue
            from the model's current state.
        jit : bool, default True
            Compile the run with :func:`brainstate.transform.jit`.

        Returns
        -------
        dict
            Maps each monitor name (or ``'output'``) to its stacked trajectory
            and ``'ts'`` to the time axis. Each trajectory has a leading axis of
            length ``(n_steps - transient) // sample_every``.

        Raises
        ------
        ValueError
            If ``dt`` is unset, ``transient`` is not shorter than the run, a
            monitored name is not a model attribute, ``sample_every`` < 1, or the
            ``inputs`` length does not match the step count.
        """
        dt = self._resolve_dt()
        n_steps = self._resolve_steps(duration, dt)
        k = self._resolve_sample_every(sample_every)
        n_tr = self._resolve_transient(transient, dt, n_steps)
        # Validated early (independent of state initialisation).
        get_inputs = self._build_inputs(inputs, n_steps)

        # ``dt`` must be live in the environment for both state initialisation
        # (delay buffers are sized from ``delay / dt``) and the run loop.
        with brainstate.environ.context(dt=dt):
            if init_states:
                if batch_size is None:
                    brainstate.nn.init_all_states(self.model)
                else:
                    brainstate.nn.init_all_states(self.model, batch_size)

            # Built after init so monitored state attributes already exist.
            measure = self._build_measure(monitors)

            def step_run(i):
                t = i * dt
                with brainstate.environ.context(i=i, t=t):
                    r = self.model.update(*get_inputs(i, t))
                return measure(r)

            def core():
                return brainstate.transform.for_loop(step_run, np.arange(n_steps))

            rec = brainstate.transform.jit(core)() if jit else core()

        sl = slice(n_tr, None, k)
        out = jax.tree.map(lambda a: a[sl], rec)
        times = (np.arange(n_steps) + 1) * dt
        out['ts'] = times[sl]
        return out
