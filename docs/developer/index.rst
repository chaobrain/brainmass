Developer Guide
===============

Welcome to the ``brainmass`` developer documentation! This guide helps you contribute to the project.


Getting Started
---------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc

      How to contribute code, docs, or examples

   .. grid-item-card:: Architecture
      :link: architecture
      :link-type: doc

      Package design and organization

   .. grid-item-card:: Creating Models
      :link: creating_models
      :link-type: doc

      Build custom neural mass models

   .. grid-item-card:: Testing
      :link: testing
      :link-type: doc

      Testing guidelines and practices


Extending brainmass
-------------------

These guides document the **stable extension contracts** — the public APIs you
build custom pieces against, so they slot into the orchestration layer (and
future data-driven tooling) exactly like the built-ins.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Building a Data-Driven Workflow
      :link: building_a_data_driven_workflow
      :link-type: doc

      The extension playbook: trainable params → custom loss → Fitter / training loop → vmap

   .. grid-item-card:: Extending Noise
      :link: extending_noise
      :link-type: doc

      Implement custom noise processes

   .. grid-item-card:: Creating a Coupling
      :link: creating_a_coupling
      :link-type: doc

      Implement a new network coupling (prefetch/delay contract)

   .. grid-item-card:: Creating an Objective
      :link: creating_an_objective
      :link-type: doc

      Write a composable loss that works across all Fitter backends



Quick Start for Contributors
-----------------------------

1. **Fork and Clone:**

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/brainmass.git
      cd brainmass

2. **Install Development Dependencies:**

   .. code-block:: bash

      pip install -e .[dev,doc]

3. **Create a Branch:**

   .. code-block:: bash

      git checkout -b feature/my-new-feature

4. **Make Changes and Test:**

   .. code-block:: bash

      pytest tests/
      make -C docs html

5. **Submit Pull Request:**

   Push your branch and open a PR on GitHub


Project Links
-------------

- `GitHub Repository <https://github.com/chaobrain/brainmass>`_
- `Issue Tracker <https://github.com/chaobrain/brainmass/issues>`_
- `Pull Requests <https://github.com/chaobrain/brainmass/pulls>`_
- `Discussions <https://github.com/chaobrain/brainmass/discussions>`_


Code of Conduct
---------------

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

Key principles:

- Be kind and courteous
- Respect differing opinions
- Focus on constructive feedback
- Assume good intentions


Getting Help
------------

- **Questions:** Open a discussion on GitHub
- **Bugs:** Report issues on GitHub
- **Feature requests:** Open an issue with the "enhancement" label
- **Chat:** Join our community (link TBD)


See Also
--------

- :doc:`../tutorials/index` - User tutorials
- :doc:`../reference/index` - API reference
- :doc:`../gallery/index` - Example gallery




.. toctree::
   :hidden:
   :maxdepth: 2

   contributing
   architecture
   creating_models
   building_a_data_driven_workflow
   creating_a_coupling
   creating_an_objective
   extending_noise
   testing
   documentation


