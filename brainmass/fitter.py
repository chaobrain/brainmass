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

"""One reusable model fitter spanning three optimizer backends.

:class:`Fitter` collapses the copy-pasted ``ModelFitting`` classes -- the
``weights = model.states(ParamState)`` -> ``register_trainable_weights`` ->
jitted ``grad`` -> ``optimizer.step`` loop, repeated verbatim in every fitting
tutorial -- into a single object. You write the objective once and swap the
optimizer backend:

- ``backend='grad'`` -- gradient descent through the differentiable model with a
  :mod:`braintools.optim` optax optimizer (the canonical loop above).
- ``backend='nevergrad'`` -- gradient-free evolutionary search over a bounded
  ``search_space`` with :class:`braintools.optim.NevergradOptimizer`.
- ``backend='scipy'`` -- :class:`braintools.optim.ScipyOptimizer` (gradient or
  gradient-free SciPy methods) over the same ``search_space``.

It reimplements none of that machinery; it only composes the existing pieces --
brainstate :class:`~brainstate.nn.Param` (constrained parameters + transforms +
``reg_loss`` + ``param_precompute``), :class:`brainmass.Simulator`,
:mod:`brainmass.objectives`, and the :mod:`braintools.optim` optimizers.
"""

import braintools
import brainstate
import jax.numpy as jnp
import numpy as np
from brainstate.nn import Param

from . import objectives as _objectives

__all__ = [
    'Fitter',
    'FitResult',
]

_BACKENDS = ('grad', 'nevergrad', 'scipy')


# --------------------------------------------------------------------------- #
# Parameter / search-space helpers                                            #
# --------------------------------------------------------------------------- #

def _trainable_params(model):
    """Return ``{dotted_name: Param}`` for every trainable ``Param`` in ``model``.

    Walks ``model.nodes()`` and keeps the :class:`~brainstate.nn.Param`
    submodules whose ``fit`` flag is ``True`` (so ``Const`` / ``fit=False``
    parameters are excluded). The name is the dotted node path, e.g. ``'k'`` for
    a top-level attribute or ``'node.k'`` for a nested one.
    """
    out = {}
    for path, node in model.nodes().items():
        if isinstance(node, Param) and node.fit:
            out['.'.join(str(p) for p in path)] = node
    return out


def _finite_box(transform):
    """Return the ``(lower, upper)`` constrained interval of ``transform``, or None.

    Only transforms that define a *finite* interval expose usable derivative-free
    bounds: :class:`~brainstate.nn.SigmoidT` / ``TanhT`` / ``SoftsignT`` /
    ``ScaledSigmoidT`` (via ``lower`` + ``width``) and :class:`~brainstate.nn.ClipT`
    (via ``lower`` + ``upper``). Half-line / unbounded transforms (``ReluT``,
    ``SoftplusT``, ``PositiveT``, ``IdentityT`` ...) return ``None`` and require an
    explicit ``search_space`` entry.
    """
    lower = getattr(transform, 'lower', None)
    if lower is None:
        return None
    width = getattr(transform, 'width', None)
    if width is not None:
        return lower, lower + width
    upper = getattr(transform, 'upper', None)
    if upper is not None:
        return lower, upper
    return None


def _shaped_bounds(lo, hi, shape):
    """Broadcast a scalar ``(lo, hi)`` pair to the parameter's value shape."""
    if shape == ():
        return (lo, hi)
    return (jnp.full(shape, lo), jnp.full(shape, hi))


def _build_search_space(params, user_search_space):
    """Resolve ``{name: (low, high)}`` bounds for the derivative-free backends.

    User-supplied bounds override the bounds derived from each parameter's
    transform. Raises ``ValueError`` for a parameter that has neither a finite
    transform interval nor an explicit entry, and for an explicit entry naming a
    parameter that is not trainable.
    """
    user = dict(user_search_space or {})
    for name in user:
        if name not in params:
            raise ValueError(
                f"search_space names unknown parameter {name!r}; trainable "
                f"parameters are {sorted(params)}."
            )
    bounds = {}
    for name, p in params.items():
        shape = jnp.shape(p.value())
        if name in user:
            lo, hi = user[name]
        else:
            box = _finite_box(p.t)
            if box is None:
                raise ValueError(
                    f"cannot derive search bounds for trainable parameter "
                    f"{name!r}: its transform {type(p.t).__name__} defines no "
                    f"finite interval. Pass search_space={{{name!r}: (low, high)}}."
                )
            lo, hi = box
        bounds[name] = _shaped_bounds(lo, hi, shape)
    return bounds


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #

class FitResult:
    """Outcome of a :meth:`Fitter.fit` call.

    Attributes
    ----------
    backend : str
        The optimizer backend that produced this result (``'grad'`` /
        ``'nevergrad'`` / ``'scipy'``).
    best_loss : float
        Lowest loss observed over the run.
    best_params : dict
        ``{name: value}`` of the trainable parameters at the best-seen point, in
        the *constrained* (physical) space (i.e. ``Param.value()``).
    history : list of float
        Per-iteration loss. For ``grad`` this is one entry per optimization step;
        for ``scipy`` it is the best loss per restart; for ``nevergrad`` it is the
        loss of every evaluated candidate.
    n_steps : int
        Number of optimization iterations actually run (may be less than the
        requested ``n_steps`` if a callback requested early stopping).
    prediction : Any or None
        The model prediction at the best-seen point (objective path), else ``None``.
    optimizer : Any
        The underlying :mod:`braintools.optim` optimizer object.
    raw : Any
        Backend-specific raw result (a SciPy ``OptimizeResult`` for ``scipy``,
        the best-parameter mapping for ``nevergrad``, ``None`` for ``grad``).
    model : brainstate.nn.Module
        The fitted model, holding the best-seen parameters.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        *,
        backend,
        best_loss,
        best_params,
        history,
        n_steps,
        prediction=None,
        optimizer=None,
        raw=None,
        model=None,
    ):
        self.backend = backend
        self.best_loss = best_loss
        self.best_params = best_params
        self.history = history
        self.n_steps = n_steps
        self.prediction = prediction
        self.optimizer = optimizer
        self.raw = raw
        self.model = model

    def __repr__(self):
        names = ', '.join(self.best_params)
        return (
            f"FitResult(backend={self.backend!r}, best_loss={self.best_loss:.6g}, "
            f"n_steps={self.n_steps}, params=[{names}])"
        )


# --------------------------------------------------------------------------- #
# Fitter                                                                      #
# --------------------------------------------------------------------------- #

class Fitter:
    r"""Fit a model's trainable parameters to data behind one ``.fit`` call.

    Parameters
    ----------
    model : brainstate.nn.Module
        The model to fit. Its trainable :class:`~brainstate.nn.Param` parameters
        (``fit=True``) are the optimization variables.
    optimizer : optional
        Interpreted by ``backend``:

        - ``'grad'``: a :class:`braintools.optim.OptaxOptimizer` instance
          (e.g. ``braintools.optim.Adam(lr=0.05)``). Defaults to ``Adam(lr=1e-2)``.
        - ``'nevergrad'`` / ``'scipy'``: an options ``dict`` forwarded to the
          optimizer constructor (e.g. ``{'method': 'DE', 'n_sample': 8}`` or
          ``{'method': 'L-BFGS-B'}``), a method-name ``str``, or ``None``. The
          actual optimizer is constructed inside :meth:`fit` (it needs the loss
          and bounds) and exposed afterwards as :attr:`optimizer`.
    loss_fn : callable, optional
        ``loss_fn(model) -> (scalar_loss, aux)``. When given it is the *entire*
        loss (you own any regularization) and ``objective`` / ``predict`` /
        ``target`` are unused. Mutually exclusive with ``predict``.
    objective : callable, optional
        ``objective(prediction, target) -> scalar`` (e.g. from
        :mod:`brainmass.objectives`). Used with ``predict``. Defaults to
        :func:`brainmass.objectives.timeseries_rmse`.
    predict : callable, optional
        ``predict(model) -> prediction``; typically a :class:`brainmass.Simulator`
        closure. Required unless ``loss_fn`` is given. The objective-path loss is
        ``objective(prediction[transient:], target) + model.reg_loss()``.
    backend : {'grad', 'nevergrad', 'scipy'}, default 'grad'
        Which optimizer backend to use.
    callbacks : list of callable, optional
        Each ``callback(info) -> bool | None`` is called once per step with
        ``info = {'step', 'loss', 'best_loss', 'model'}``. Returning ``True``
        stops the run early (``grad`` backend only).
    transient : int, optional
        Number of leading samples discarded from the prediction (axis 0) before
        the objective is applied. ``None`` keeps the whole prediction.
    search_space : dict, optional
        ``{name: (low, high)}`` constrained bounds for the derivative-free
        backends, overriding/augmenting the bounds derived from each parameter's
        transform.

    See Also
    --------
    brainmass.Simulator : builds the ``predict`` closure.
    brainmass.objectives : composable objective callables.

    Notes
    -----
    The ``grad`` backend reproduces the canonical hand-rolled loop exactly:
    ``model.states(ParamState)`` are registered as trainable weights, the loss is
    evaluated inside ``model.param_precompute()``, and
    ``brainstate.transform.grad(..., has_aux=True, return_value=True)`` feeds
    ``optimizer.step``. ``model.reg_loss()`` is added automatically. The
    derivative-free backends evaluate one candidate at a time (setting parameters
    via ``Param.set_value`` then running ``predict``) -- ``vmap`` over parameter
    writes is unsupported, so candidates are looped, which is why these backends
    suit a small number of scalar parameters.

    Examples
    --------
    >>> import braintools, brainstate, brainunit as u
    >>> import jax.numpy as jnp
    >>> import brainmass
    >>> from brainstate.nn import Param, SigmoidT
    >>> class Toy(brainstate.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.k = Param(1.0, t=SigmoidT(0.5, 3.0))
    ...     def update(self):
    ...         return self.k.value()
    >>> def predict(m):
    ...     sim = brainmass.Simulator(m, dt=0.1 * u.ms)
    ...     return jnp.mean(sim.run(1.0 * u.ms, monitors=None)['output'])
    >>> fitter = brainmass.Fitter(Toy(), braintools.optim.Adam(lr=0.1), predict=predict)
    >>> result = fitter.fit(target=jnp.asarray(2.0), n_steps=30)
    >>> result.backend
    'grad'
    >>> list(result.best_params)
    ['k']
    >>> bool(result.best_loss <= result.history[0])
    True
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        model,
        optimizer=None,
        *,
        loss_fn=None,
        objective=None,
        predict=None,
        backend='grad',
        callbacks=None,
        transient=None,
        search_space=None,
    ):
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}.")
        if loss_fn is None and predict is None:
            raise ValueError(
                "provide either loss_fn=callable(model)->(loss, aux) or "
                "predict=callable(model)->prediction."
            )
        if loss_fn is not None and predict is not None:
            raise ValueError("pass loss_fn or predict, not both.")

        self.model = model
        self.backend = backend
        self._optimizer_arg = optimizer
        self._optimizer = None
        self.loss_fn = loss_fn
        self.predict = predict
        self.objective = (
            objective if objective is not None
            else (_objectives.timeseries_rmse() if loss_fn is None else None)
        )
        self.callbacks = list(callbacks or [])
        self.transient = transient
        self.search_space = search_space

    # ----------------------------------------------------------------- helpers

    @property
    def optimizer(self):
        """The underlying optimizer (constructed lazily for derivative-free backends)."""
        return self._optimizer if self._optimizer is not None else self._optimizer_arg

    def _trim(self, prediction):
        """Drop the leading ``transient`` samples from a prediction (axis 0)."""
        if self.transient is None:
            return prediction
        return prediction[self.transient:]

    def _eval(self, model, target):
        """Return ``(loss, aux)`` for ``model`` (objective path adds ``reg_loss``)."""
        if self.loss_fn is not None:
            return self.loss_fn(model)
        prediction = self.predict(model)
        loss = self.objective(self._trim(prediction), target) + model.reg_loss()
        return loss, prediction

    def _eval_candidate(self, params, values, target):
        """Set ``values`` onto ``params`` then return the scalar loss."""
        for name, val in values.items():
            params[name].set_value(val)
        loss, _ = self._eval(self.model, target)
        return loss

    @staticmethod
    def _apply_params(params, values):
        """Write ``values`` (constrained space) back onto ``params``."""
        for name, val in values.items():
            params[name].set_value(val)

    def _run_callbacks(self, step, loss, best_loss):
        """Invoke every callback; return ``True`` if any requests early stop."""
        info = {'step': step, 'loss': loss, 'best_loss': best_loss, 'model': self.model}
        stop = False
        for cb in self.callbacks:
            if cb(info) is True:
                stop = True
        return stop

    def _derivfree_opts(self):
        """Normalise the ``optimizer`` argument into a kwargs dict for the backend."""
        arg = self._optimizer_arg
        if arg is None:
            return {}
        if isinstance(arg, str):
            return {'method': arg}
        if isinstance(arg, dict):
            return dict(arg)
        raise TypeError(
            f"for backend={self.backend!r}, optimizer must be a method name (str), "
            f"an options dict, or None; got {type(arg).__name__}."
        )

    # -------------------------------------------------------------------- fit

    def fit(self, target=None, n_steps=100, *, verbose=False):
        """Run the optimization and return a :class:`FitResult`.

        Parameters
        ----------
        target : Any, optional
            The data the objective compares the prediction against. Unused when a
            ``loss_fn`` was supplied (it consumes data itself).
        n_steps : int, default 100
            Optimization iterations: optax steps (``grad``), generations
            (``nevergrad``), or random restarts (``scipy``). Must be >= 1.
        verbose : bool, default False
            Print per-iteration progress.

        Returns
        -------
        FitResult
            The best-seen loss, parameters, history, and the fitted model.
        """
        if int(n_steps) < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps!r}.")
        if self.backend == 'grad':
            return self._fit_grad(target, int(n_steps), verbose)
        if self.backend == 'nevergrad':
            return self._fit_nevergrad(target, int(n_steps), verbose)
        return self._fit_scipy(target, int(n_steps), verbose)

    # --------------------------------------------------------------- backends

    def _fit_grad(self, target, n_steps, verbose):
        weights = self.model.states(brainstate.ParamState)
        if len(weights) == 0:
            raise ValueError(
                "model has no trainable parameters (brainstate.ParamState); set "
                "fit=True on at least one Param to fit with backend='grad'."
            )
        optimizer = self._optimizer_arg
        if optimizer is None:
            optimizer = braintools.optim.Adam(lr=1e-2)
        elif not isinstance(optimizer, braintools.optim.OptaxOptimizer):
            raise TypeError(
                "backend='grad' requires a braintools.optim optimizer "
                f"(e.g. Adam(lr=...)) or None; got {type(optimizer).__name__}."
            )
        optimizer.register_trainable_weights(weights)
        self._optimizer = optimizer

        def f_loss():
            with self.model.param_precompute():
                return self._eval(self.model, target)

        @brainstate.transform.jit
        def f_train():
            f_grad = brainstate.transform.grad(
                f_loss, weights, has_aux=True, return_value=True
            )
            grads, loss, aux = f_grad()
            optimizer.step(grads)
            return loss, aux

        history = []
        best_loss = float('inf')
        best_snapshot = None
        best_pred = None
        step = 0
        for step in range(n_steps):
            loss, aux = f_train()
            loss_f = float(loss)
            history.append(loss_f)
            if loss_f < best_loss:
                best_loss = loss_f
                best_snapshot = {k: jnp.asarray(v.value) for k, v in weights.items()}
                best_pred = aux
            if verbose:
                print(f"[grad] step {step}: loss={loss_f:.6g} best={best_loss:.6g}")
            if self._run_callbacks(step, loss_f, best_loss):
                break

        # Restore the best-seen weights so the model ends at its best point.
        if best_snapshot is not None:
            for k, v in weights.items():
                v.value = best_snapshot[k]

        best_params = {name: p.value() for name, p in _trainable_params(self.model).items()}
        return FitResult(
            backend='grad',
            best_loss=best_loss,
            best_params=best_params,
            history=history,
            n_steps=step + 1,
            prediction=best_pred,
            optimizer=optimizer,
            raw=None,
            model=self.model,
        )

    def _fit_nevergrad(self, target, n_steps, verbose):
        params = _trainable_params(self.model)
        bounds = self._resolve_bounds(params)
        names = list(bounds)
        opts = self._derivfree_opts()
        method = opts.pop('method', 'DE')
        n_sample = int(opts.pop('n_sample', 8))

        def batched(**stacked):
            losses = []
            for i in range(n_sample):
                values = {name: stacked[name][i] for name in names}
                losses.append(self._eval_candidate(params, values, target))
            return jnp.asarray(losses)

        optimizer = braintools.optim.NevergradOptimizer(
            batched, bounds, n_sample=n_sample, method=method, **opts
        )
        self._optimizer = optimizer
        best = optimizer.minimize(n_iter=n_steps, verbose=verbose)

        errors = np.asarray(optimizer.errors, dtype=float)
        best_loss = float(np.nanmin(errors))
        self._apply_params(params, best)
        best_params = {name: p.value() for name, p in params.items()}
        best_pred = None if self.loss_fn is not None else self.predict(self.model)
        return FitResult(
            backend='nevergrad',
            best_loss=best_loss,
            best_params=best_params,
            history=[float(e) for e in errors],
            n_steps=n_steps,
            prediction=best_pred,
            optimizer=optimizer,
            raw=best,
            model=self.model,
        )

    def _fit_scipy(self, target, n_steps, verbose):
        params = _trainable_params(self.model)
        bounds = self._resolve_bounds(params)
        names = list(bounds)
        opts = self._derivfree_opts()
        method = opts.pop('method', None)

        def loss(**candidate):
            values = {name: candidate[name] for name in names}
            return self._eval_candidate(params, values, target)

        optimizer = braintools.optim.ScipyOptimizer(
            loss, bounds, method=method, **opts
        )
        self._optimizer = optimizer
        res = optimizer.minimize(n_iter=n_steps)

        best_loss = float(res.fun)
        self._apply_params(params, res.x)
        best_params = {name: p.value() for name, p in params.items()}
        best_pred = None if self.loss_fn is not None else self.predict(self.model)
        if verbose:
            print(f"[scipy] best loss={best_loss:.6g} success={getattr(res, 'success', None)}")
        return FitResult(
            backend='scipy',
            best_loss=best_loss,
            best_params=best_params,
            history=[best_loss],
            n_steps=n_steps,
            prediction=best_pred,
            optimizer=optimizer,
            raw=res,
            model=self.model,
        )

    def _resolve_bounds(self, params):
        """Build the derivative-free search bounds, erroring when nothing to search."""
        if len(params) == 0 and not self.search_space:
            raise ValueError(
                "model has no trainable parameters and no search_space was given; "
                f"backend={self.backend!r} needs at least one parameter to search."
            )
        return _build_search_space(params, self.search_space)
