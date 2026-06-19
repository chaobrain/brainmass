``brainmass`` documentation
============================

`brainmass <https://github.com/chaobrain/brainmass>`_ implements neural mass models with `brainstate <https://github.com/chaobrain/brainstate>`_,
enabling whole-brain modeling with **differentiable programming** and JAX.

Where other whole-brain toolkits run forward simulations and fit parameters with
grid or evolutionary search, brainmass backpropagates *through the ODE solve*: it
brings gradient-based fitting, high-dimensional parameter fields, GPU/TPU batching,
and the ability to **train** neural-mass-style networks on tasks — all unit-safe and
end-to-end from parameters to BOLD / EEG / MEG signals.


----

Features
^^^^^^^^

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Comprehensive Model Library
      :class-card: sd-border-0
      :shadow: md

      20+ neural mass models from phenomenological oscillators to physiological population
      models, covering EEG, MEG, and fMRI applications

   .. grid-item-card:: Differentiable Optimization
      :class-card: sd-border-0
      :shadow: md

      Fit model parameters to empirical data using gradient-based (JAX) or gradient-free
      (Nevergrad / SciPy) optimization through one ``Fitter``

   .. grid-item-card:: Forward Modeling
      :class-card: sd-border-0
      :shadow: md

      Built-in BOLD hemodynamics and EEG/MEG lead-field models for linking neural activity
      to neuroimaging signals

   .. grid-item-card:: Unit-Safe Computing
      :class-card: sd-border-0
      :shadow: md

      Automatic dimensional analysis with ``brainunit`` prevents unit errors in scientific
      computing


----

How brainmass compares
^^^^^^^^^^^^^^^^^^^^^^^

brainmass shares the neural-mass / whole-brain modeling space with
`The Virtual Brain <https://www.thevirtualbrain.org/>`_ (TVB) and
`neurolib <https://github.com/neurolib-dev/neurolib>`_. Its distinguishing design choice
is a fully **differentiable, JAX-native** core. The table below is a deliberately
conservative summary of capabilities at the time of writing; consult each project for
its current state.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Capability
     - brainmass
     - The Virtual Brain
     - neurolib
   * - Differentiable / gradient-based fitting (backprop through the solve)
     - Yes
     - No
     - No
   * - JAX backend with GPU / TPU acceleration
     - Yes
     - No
     - No
   * - In-package orchestration & fitting (``Simulator`` / ``Network`` / ``Fitter``)
     - Yes
     - Partial
     - Partial
   * - Unit-safe quantities (dimensional analysis)
     - Yes
     - No
     - No
   * - Next-generation / exact mean-field models (e.g. Montbrió-Pazó-Roxin, Coombes-Byrne)
     - Yes
     - Partial
     - Partial
   * - In-package BOLD + EEG/MEG forward models
     - Yes
     - Yes
     - Partial

The deeper rationale lives in :doc:`concepts/why_differentiable`.


----

Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainmass[cpu]

    .. tab-item:: GPU (CUDA 12)

       .. code-block:: bash

          pip install -U brainmass[cuda12]

    .. tab-item:: GPU (CUDA 13)

       .. code-block:: bash

          pip install -U brainmass[cuda13]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainmass[tpu]

See :doc:`getting_started/installation` for detailed instructions.


----

Choose your path
^^^^^^^^^^^^^^^^^

brainmass serves three kinds of users. Pick the on-ramp that fits you — each is a
signposted route through the documentation, detailed in :doc:`getting_started/learning_paths`.

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Beginner
      :link: getting_started/learning_paths
      :link-type: doc

      **New to neural mass models or brainmass.** Install, run a first simulation, and
      build the mental model, then explore the model zoo.

   .. grid-item-card:: Researcher
      :link: getting_started/learning_paths
      :link-type: doc

      **Have empirical data (EEG / MEG / fMRI).** Map models to signals, fit them to your
      data, analyze the results, and study the case studies.

   .. grid-item-card:: Modeler
      :link: getting_started/learning_paths
      :link-type: doc

      **Build and extend models.** Custom couplings and objectives, performance, and
      differentiable / data-driven workflows.


----

Data-Driven Modeling
^^^^^^^^^^^^^^^^^^^^^

The flagship of brainmass is **data-driven modeling** — constructing, fitting, and
training neural-mass networks against data. The :doc:`data_driven/index` hub curates a
guided path through the differentiable workflow, and its roadmap reserves homes for the
growth areas (model discovery / system identification, a task-shaped trainer, and
simulation-based inference).


----

BrainX Ecosystem
^^^^^^^^^^^^^^^^

``brainmass`` is one part of our `brain modeling ecosystem <https://brainx.chaobrain.com/>`_.



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Get Started

   getting_started/installation
   getting_started/quickstart
   getting_started/key_concepts
   getting_started/learning_paths

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Learn

   tutorials/index
   howto/index
   concepts/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Showcase

   data_driven/index
   gallery/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Reference

   reference/index
   developer/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Additional Resources

   faq
   changelog
