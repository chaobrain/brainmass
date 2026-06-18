Parameter Fitting
==================

This tutorial fits a model's parameters to data with :class:`brainmass.Fitter` --
one object that wraps the *write-the-objective-once, swap-the-backend* workflow.
You build a ``predict`` closure and an objective once, then fit with gradient
descent, Nevergrad, or SciPy by changing a single ``backend`` argument.

.. note::

   Every code block on this page is executed as part of the documentation build
   (via ``sphinx.ext.doctest``), so the snippets are guaranteed to run against the
   current API. The integration time step ``dt`` is global state set through
   ``brainstate.environ`` -- it is **not** a model argument.


Overview
--------

The fitting workflow has four pieces:

1. A model with one or more **trainable parameters** -- a
   :class:`brainstate.nn.Param` with ``fit=True`` (the default). A transform such
   as :class:`~brainstate.nn.SigmoidT` keeps the parameter inside physical bounds.
2. A ``predict(model) -> prediction`` closure, usually a :class:`brainmass.Simulator`
   run that returns a ``(time, regions)`` trajectory.
3. An **objective** comparing the prediction to data, from
   :mod:`brainmass.objectives`.
4. A :class:`~brainmass.Fitter` that minimises the loss with the chosen backend.


A Model, a Prediction, and an Objective
---------------------------------------

We use a minimal differentiable model with a single bounded gain ``k``. The
``SigmoidT(0.1, 5.0)`` transform constrains ``k`` to ``(0.1, 5.0)`` -- and those
bounds are later reused, verbatim, as the search box for the gradient-free
backends. The synthetic target is generated with a "true" gain of ``2.0``:

.. testcode::

   import brainmass
   import braintools
   import brainstate
   import brainunit as u
   import jax.numpy as jnp
   from brainstate.nn import Param, SigmoidT

   brainstate.environ.set(dt=0.1 * u.ms)

   class GainModel(brainstate.nn.Module):
       def __init__(self, init_gain=1.0):
           super().__init__()
           # Trainable, bounded parameter; stored unconstrained, exposed in (0.1, 5.0).
           self.k = Param(init_gain, t=SigmoidT(0.1, 5.0))

       def update(self, x):
           return self.k.value() * x

   # A fixed 3-region drive and the target it should reproduce at the true gain k = 2.0.
   t = jnp.linspace(0.0, 6.28, 40)[:, None]
   drive = jnp.sin(t * jnp.arange(1, 4))        # (40, 3): three distinct channels
   target = 2.0 * drive

   # predict(model) -> (time, regions): one Simulator run over the drive.
   def predict(model):
       sim = brainmass.Simulator(model, dt=0.1 * u.ms)
       return sim.run(4.0 * u.ms, inputs=drive, monitors=None)['output']

   # Write the objective ONCE, then reuse it for every backend below.
   objective = brainmass.objectives.timeseries_rmse()


Gradient-Based (optax)
----------------------

The ``grad`` backend is the canonical loop: it registers the model's trainable
``ParamState`` weights with the :mod:`braintools.optim` optimizer, differentiates
the loss through the (differentiable) model, and steps the optimizer.
``model.reg_loss()`` is added automatically and parameter transforms keep ``k``
inside its bounds.

.. testcode::

   fitter = brainmass.Fitter(
       GainModel(),
       braintools.optim.Adam(lr=0.2),
       predict=predict,
       objective=objective,
       backend='grad',
   )
   result = fitter.fit(target=target, n_steps=60)
   print(f"grad: fitted k = {float(result.best_params['k']):.2f}")
   assert result.best_loss <= result.history[0]


Gradient-Free (Nevergrad)
-------------------------

Swap ``backend='nevergrad'`` to search the same parameter with an evolutionary
algorithm -- useful when the objective is non-differentiable. The search box is
**derived automatically** from the ``SigmoidT`` transform, so no bounds need to be
spelled out. For the derivative-free backends the ``optimizer`` argument is an
options dict (or a method-name string):

.. testcode::

   fitter = brainmass.Fitter(
       GainModel(),
       {'method': 'DE', 'n_sample': 8},
       predict=predict,
       objective=objective,
       backend='nevergrad',
   )
   result = fitter.fit(target=target, n_steps=20)
   print(f"nevergrad: fitted k = {float(result.best_params['k']):.1f}")


SciPy
-----

``backend='scipy'`` runs :class:`braintools.optim.ScipyOptimizer`. Gradient
methods (``L-BFGS-B``) differentiate through the parameter assignment;
gradient-free methods (``Nelder-Mead``) work too. ``n_steps`` is the number of
random restarts -- a few restarts make the search robust to a poor starting
point:

.. testcode::

   fitter = brainmass.Fitter(
       GainModel(),
       {'method': 'L-BFGS-B'},
       predict=predict,
       objective=objective,
       backend='scipy',
   )
   result = fitter.fit(target=target, n_steps=3)
   print(f"scipy: fitted k = {float(result.best_params['k']):.2f}")


Search Spaces from Transforms
-----------------------------

The gradient-free backends search a bounded box per parameter. The bounds are
derived from each parameter's transform where the transform defines a finite
interval:

- :class:`~brainstate.nn.SigmoidT` / ``TanhT`` / ``SoftsignT`` / ``ScaledSigmoidT``
  -> ``(lower, upper)``
- :class:`~brainstate.nn.ClipT` -> ``(lower, upper)``

Half-line or unbounded transforms (``ReluT``, ``SoftplusT``, ``PositiveT``,
``IdentityT``, ...) define no finite box, so you must pass an explicit
``search_space`` (which also overrides any derived bound):

.. testcode::

   from brainstate.nn import ReluT

   class PositiveGain(brainstate.nn.Module):
       def __init__(self):
           super().__init__()
           self.k = Param(1.0, t=ReluT(0.0))   # lower bound only -> no finite box

       def update(self, x):
           return self.k.value() * x

   fitter = brainmass.Fitter(
       PositiveGain(),
       {'method': 'L-BFGS-B'},
       predict=predict,
       objective=objective,
       backend='scipy',
       search_space={'k': (0.1, 5.0)},          # supply the box explicitly
   )
   result = fitter.fit(target=target, n_steps=1)
   assert 0.1 <= float(result.best_params['k']) <= 5.0


Custom Loss Functions
---------------------

When the loss is more than ``objective(prediction, target)`` -- multiple terms,
custom weighting, data captured in a closure -- pass ``loss_fn(model) ->
(scalar_loss, aux)`` instead of ``predict`` + ``objective``. It is the entire loss
(you own any regularization):

.. testcode::

   def loss_fn(model):
       pred = predict(model)
       rmse = brainmass.objectives.timeseries_rmse()
       cos = brainmass.objectives.cosine_sim(as_loss=True)
       loss = rmse(pred, target) + 0.1 * cos(pred, target)
       return loss, pred

   fitter = brainmass.Fitter(GainModel(), braintools.optim.Adam(lr=0.2), loss_fn=loss_fn)
   result = fitter.fit(n_steps=40)
   print(f"custom loss: fitted k = {float(result.best_params['k']):.1f}")


Reading the Result
------------------

:meth:`~brainmass.Fitter.fit` returns a :class:`~brainmass.FitResult`:

- ``best_loss`` -- the lowest loss seen,
- ``best_params`` -- ``{name: value}`` in constrained (physical) space,
- ``history`` -- per-iteration loss,
- ``prediction`` -- the model output at the best point,
- ``model`` -- the model, left holding the best-seen parameters,
- ``optimizer`` / ``raw`` -- the underlying optimizer and its native result.

Callbacks (``callbacks=[fn]``, each called with ``{'step', 'loss', 'best_loss',
'model'}``) can monitor progress and, by returning ``True``, stop the ``grad``
backend early.


Best Practices
--------------

1. **Constrain with transforms.** A ``SigmoidT`` / ``SoftplusT`` parameter cannot
   leave its physical range, and its bounds double as the gradient-free search box.
2. **Start simple.** Fit one or two scalar parameters before a full matrix; the
   gradient-free backends evaluate candidates one at a time and suit low dimensions.
3. **Normalise the loss.** Scale objective terms to similar magnitudes (see
   :func:`brainmass.objectives.combine`) so one term does not dominate.
4. **Discard transients.** Pass ``transient=<n_samples>`` to drop warm-up samples
   from the prediction before the objective is applied.
5. **Swap backends to escape local minima.** Try a gradient-free global search
   (Nevergrad) when gradient descent stalls, then refine with ``grad``.


See Also
--------

- :doc:`forward_modeling` - getting observable data out of models
- :doc:`building_networks` - building the whole-brain models you fit
- :class:`brainmass.Simulator` - the run loop behind ``predict``
- :mod:`brainmass.objectives` - the objective builders
