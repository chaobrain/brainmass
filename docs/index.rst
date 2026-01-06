``brainmass`` documentation
============================

`brainmass <https://github.com/chaobrain/brainmass>`_ implements neural mass models with `brainstate <https://github.com/chaobrain/brainstate>`_,
enabling whole-brain modeling with differentiable programming and JAX.


----

Features
^^^^^^^^

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Comprehensive Model Library
      :class-card: sd-border-0
      :shadow: md

      13+ neural mass models from phenomenological oscillators to physiological population models,
      covering EEG, MEG, and fMRI applications

   .. grid-item-card:: Differentiable Optimization
      :class-card: sd-border-0
      :shadow: md

      Fit model parameters to empirical data using gradient-based (JAX) or gradient-free (Nevergrad) optimization

   .. grid-item-card:: Forward Modeling
      :class-card: sd-border-0
      :shadow: md

      Built-in BOLD hemodynamics and EEG/MEG lead-field models for linking neural activity to neuroimaging signals

   .. grid-item-card:: Unit-Safe Computing
      :class-card: sd-border-0
      :shadow: md

      Automatic dimensional analysis with ``brainunit`` prevents unit errors in scientific computing


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

See :doc:`tutorials/installation` for detailed instructions.


----

Where to Start
^^^^^^^^^^^^^^

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: New to brainmass?
      :link: tutorials/quickstart
      :link-type: doc

      Start with the **Quickstart Tutorial** for a 5-minute introduction

   .. grid-item-card:: Looking for examples?
      :link: examples/index
      :link-type: doc

      Browse the **Examples Gallery** for practical applications

   .. grid-item-card:: Need specific functionality?
      :link: api/index
      :link-type: doc

      Check the **API Reference** for detailed documentation

   .. grid-item-card:: Want to contribute?
      :link: developer/index
      :link-type: doc

      Read the **Developer Guide** to get started


----

Documentation Structure
^^^^^^^^^^^^^^^^^^^^^^^

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Step-by-step guides for common tasks

   .. grid-item-card:: Examples
      :link: examples/index
      :link-type: doc

      Jupyter notebooks with practical applications

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc

      Complete API documentation

   .. grid-item-card:: Developer Guide
      :link: developer/index
      :link-type: doc

      Contributing and extending brainmass


----

BrainX Ecosystem
^^^^^^^^^^^^^^^^

``brainmass`` is part of the `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_:

- **brainstate**: State management for dynamical systems
- **brainunit**: Unit-aware array operations
- **braintools**: Utilities and transforms
- **brainmass**: Neural mass models (this package)


----

Citation
^^^^^^^^

If you use ``brainmass`` in your research, please cite:

.. code-block:: bibtex

   @software{brainmass2024,
     author = {{BrainX Ecosystem}},
     title = {BrainMass: Neural Mass Models for Whole-Brain Modeling},
     year = {2024},
     url = {https://github.com/chaobrain/brainmass}
   }



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials and Guides

   tutorials/index
   examples/index
   developer/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Additional Resources

   faq
   api/index
   changelog

