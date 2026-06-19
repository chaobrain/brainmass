Datasets
========

.. currentmodule:: brainmass.datasets

``brainmass.datasets`` is an extensible registry of small, bundled example
datasets, so the tutorials and the gallery run with **no external download**.
Every built-in entry is either a tiny, license-clean, deterministic file shipped
under ``brainmass/_data/`` or a fully synthetic generator. Registering a new
dataset is a single :func:`register_dataset` call.

Registry API
------------

.. autosummary::
   :toctree: generated/

   register_dataset
   list_datasets
   load_dataset


Typed containers
----------------

.. autosummary::
   :toctree: generated/

   Connectome
   Signal


Synthetic task generator
-------------------------

.. autosummary::
   :toctree: generated/

   delayed_match_task


Built-in datasets
-----------------

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Name
     - Returns
     - Description
   * - ``example_connectome``
     - :class:`Connectome`
     - A small (N=8) synthetic structural connectome: symmetric, zero-diagonal
       ``weights`` in ``[0, 1]`` and unit-aware Euclidean ``distances`` (in
       :data:`brainunit.mm`).
   * - ``example_signal``
     - :class:`Signal`
     - A short multi-region target time series (with its functional connectivity
       and sampling ``dt`` in :data:`brainunit.ms`) for the fitting tutorials.
   * - ``delayed_match_task``
     - ``(inputs, targets)``
     - A synthetic delayed-match-to-sample task for the HORN training tutorial
       (no bundled binary; deterministic given a seed).


Examples
--------

Load the bundled connectome and signal:

.. doctest::

   >>> import brainmass
   >>> conn = brainmass.datasets.load_dataset('example_connectome')
   >>> conn.weights.shape
   (8, 8)
   >>> print(conn.distances.unit)
   mm
   >>> sig = brainmass.datasets.load_dataset('example_signal')
   >>> sig.signal.shape
   (500, 8)

Generate a synthetic task and register a new dataset:

.. doctest::

   >>> inputs, targets = brainmass.datasets.delayed_match_task(n_samples=16, seq_len=8)
   >>> inputs.shape
   (16, 8, 4)
   >>> brainmass.datasets.register_dataset('my_data', lambda: 42, description='demo')
   >>> brainmass.datasets.load_dataset('my_data')
   42
   >>> _ = brainmass.datasets._REGISTRY.pop('my_data')


See Also
--------

- :doc:`/reference/viz` -- plotting helpers for the data these loaders return.
- :doc:`/reference/utilities` -- :func:`brainmass.list_models` model catalogue.
