API Reference
=============

.. currentmodule:: brainmass

This section provides a comprehensive reference for all public APIs in the ``brainmass`` package.
The package is organized into several main categories that follow the hierarchical structure
of whole-brain network modeling:

.. code-block::

    Structural Connectivity (DTI / Structural MRI)
       ↓
    Neural Mass Models (NMMs)
       ↓
    Biophysical Forward Model
       ↓
    Observed Signals (EEG, MEG, fMRI BOLD)


Package Contents
----------------

The ``brainmass`` package provides the following components:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Component
     - Description
   * - :doc:`models`
     - Neural mass models for phenomenological and physiological brain dynamics
   * - :doc:`noise`
     - Stochastic noise processes for adding realistic variability
   * - :doc:`coupling`
     - Coupling mechanisms for connecting regions in brain networks
   * - :doc:`forward`
     - Forward models for mapping neural activity to observed signals (BOLD, EEG/MEG)
   * - :doc:`horn`
     - Harmonic Oscillator Recurrent Networks (HORN) for sequence learning
   * - :doc:`utilities`
     - Utility functions for common operations
   * - :doc:`types`
     - Type aliases and parameter wrapper classes


Quick Navigation
----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Neural Mass Models
      :link: models
      :link-type: doc

      Phenomenological and physiological models for simulating neural population dynamics

   .. grid-item-card:: Noise Processes
      :link: noise
      :link-type: doc

      Stochastic processes including OU, Gaussian, colored noise, and Brownian motion

   .. grid-item-card:: Coupling Mechanisms
      :link: coupling
      :link-type: doc

      Diffusive and additive coupling for multi-region brain networks

   .. grid-item-card:: Forward Models
      :link: forward
      :link-type: doc

      BOLD hemodynamics and EEG/MEG lead-field models

   .. grid-item-card:: HORN Models
      :link: horn
      :link-type: doc

      Harmonic oscillator recurrent networks for temporal sequences

   .. grid-item-card:: Utilities & Types
      :link: utilities
      :link-type: doc

      Helper functions and type definitions


General API Conventions
-----------------------

All neural mass models and dynamics classes in ``brainmass`` follow these conventions:

**Initialization:**

.. code-block:: python

   model = brainmass.ModelName(in_size=..., **parameters)
   model.init_all_states(batch_size=None)

- ``in_size``: Shape of the input (number of regions/nodes)
- Parameters are keyword arguments with default values
- ``init_state()`` must be called before simulation

**Dynamics:**

.. code-block:: python

   output = model.update(**inputs)

- ``update()`` advances the model by one time step
- Returns observable(s) relevant to the model
- Internal states are accessible as ``.value`` attributes

**Units:**

All models are unit-aware via ``brainunit``:

- Time constants: typically ``u.ms`` or ``u.s``
- Firing rates: typically ``u.Hz``
- Membrane potentials: typically ``u.mV``
- Quantities can be passed as ``float`` (unitless) or ``brainunit.Quantity``


.. toctree::
   :maxdepth: 2
   :hidden:

   models
   noise
   coupling
   forward
   horn
   utilities
