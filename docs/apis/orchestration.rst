Orchestration
=============

.. currentmodule:: brainmass

The orchestration layer sits *on top of* the neural mass models. It provides the
reusable run loop, connectome builder, and loss/score builders that earlier had to
be hand-written in every example and tutorial:

- :class:`Network` wires a node model into a delay-coupled whole-brain network.
- :class:`Simulator` drives any model (single node or whole-brain network) and
  collects monitored trajectories into a unit-aware result.
- :mod:`brainmass.objectives` composes :mod:`braintools.metric` into small,
  jit / grad / vmap-safe objective callables over those trajectories.
- :class:`Fitter` fits a model's trainable parameters to data behind one
  ``.fit`` call, swapping between gradient (optax), Nevergrad, and SciPy
  backends without rewriting the objective.


Network
-------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Network

.. autoclass:: Network
   :members:
   :special-members: __init__


Simulator
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Simulator

.. autoclass:: Simulator
   :members:
   :special-members: __init__


Fitter
------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Fitter
   FitResult

.. autoclass:: Fitter
   :members:
   :special-members: __init__

.. autoclass:: FitResult
   :members:


Objectives
----------

.. currentmodule:: brainmass.objectives

Each function is a *builder*: it takes configuration and returns a small
``callable(prediction, target)`` that wraps :mod:`braintools.metric` without
reimplementing any metric maths. The callables are designed to be composed via
:func:`combine` into the loss a fitter minimises.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   timeseries_rmse
   fc_corr
   fc_rmse
   cosine_sim
   fcd
   fcd_distribution
   ks_distance
   wasserstein_1d
   fcd_ks
   fcd_wasserstein
   combine

.. autofunction:: timeseries_rmse

.. autofunction:: fc_corr

.. autofunction:: fc_rmse

.. autofunction:: cosine_sim

.. autofunction:: fcd

.. autofunction:: combine


FCD-distribution objectives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The standard FCD fitting target is the *distribution* of the FCD matrix's
off-diagonal values, not the matrix correlation (:func:`fcd`). These builders
compare that distribution between prediction and target: each computes the FCD of
both, kernel-density-estimates the off-diagonal values onto a shared grid, and
returns a distributional distance.

:func:`wasserstein_1d` is **smooth and differentiable**, so :func:`fcd_wasserstein`
is the recommended FCD objective for gradient-based fitting; :func:`ks_distance`
(the literature-standard Kolmogorov-Smirnov statistic) is a non-smooth supremum, so
:func:`fcd_ks` is best for evaluation / reporting. A degenerate (constant,
zero-variance) input yields a singular KDE and a ``nan`` distance.

.. autofunction:: fcd_distribution

.. autofunction:: ks_distance

.. autofunction:: wasserstein_1d

.. autofunction:: fcd_ks

.. autofunction:: fcd_wasserstein


See Also
--------

- :doc:`models` - the neural mass models that :class:`Simulator` drives
- :doc:`coupling` - building multi-region networks to simulate
