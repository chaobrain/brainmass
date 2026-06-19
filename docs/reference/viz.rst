Visualization
=============

.. currentmodule:: brainmass.viz

``brainmass.viz`` provides thin plotting helpers that keep the tutorials and the
gallery concise and visually consistent. Each helper surfaces a standard
matplotlib plot or a :mod:`braintools.metric` output (functional connectivity,
power spectrum) -- none reimplements a metric.

.. note::

   ``matplotlib`` is an **optional** dependency, imported **lazily inside each
   function**. ``import brainmass`` never imports matplotlib on behalf of
   ``viz``; calling a helper without matplotlib installed raises a clear
   ``ImportError`` pointing at the extra. Install it with::

       pip install brainmass[viz]

   Every helper accepts an optional ``ax=`` (returning the
   :class:`matplotlib.axes.Axes` it drew on) and tolerates unit-aware
   (:class:`brainunit.Quantity`) inputs.

Plotting helpers
----------------

.. autosummary::
   :toctree: generated/

   plot_timeseries
   plot_phase_portrait
   plot_connectivity
   plot_functional_connectivity
   plot_power_spectrum


Examples
--------

The doctest setup selects the Agg (headless) backend, so these run in CI:

.. doctest::

   >>> import brainmass
   >>> conn = brainmass.datasets.load_dataset('example_connectome')
   >>> ax = brainmass.viz.plot_connectivity(conn.weights, labels=conn.labels)
   >>> len(ax.images)
   1
   >>> sig = brainmass.datasets.load_dataset('example_signal')
   >>> ax = brainmass.viz.plot_functional_connectivity(sig.signal)
   >>> len(ax.images)
   1
   >>> ax = brainmass.viz.plot_power_spectrum(sig.signal[:, 0], dt=sig.dt)
   >>> ax.get_xlabel()
   'frequency'


See Also
--------

- :doc:`/reference/datasets` -- the example data these helpers visualise.
- :doc:`/reference/observation` -- mapping neural activity to measurable signals.
