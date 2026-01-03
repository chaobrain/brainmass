``brainmass`` documentation
============================

`brainmass <https://github.com/chaobrain/brainmass>`_ implements neural mass models with `brainstate <https://github.com/chaobrain/brainstate>`_,
enabling whole-brain brain modeling with differentiable programming.




----

Features
^^^^^^^^^

.. grid::


   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Intuitive Programming
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``brainmass`` provides simple interface to build complex neural mass models.



   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Differentiable Optimization
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-6

         .. div:: sd-font-normal

            ``brainmass`` supports differentiable optimizations to fit model parameters to empirical data.



----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U brainmass[cpu]

    .. tab-item:: GPU

       .. code-block:: bash

          pip install -U brainmass[cuda12]

          pip install -U brainmass[cuda13]

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U brainmass[tpu]

----


See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^


``brainmass`` is one part of our `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.



.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Examples

   examples/00-hopf-osillator.ipynb
   examples/01-wilsonwowan-osillator.ipynb
   examples/02-fhn-osillator.ipynb
   examples/03-jansenrit_single_node_simulation.ipynb
   examples/10-parameter-exploration.ipynb
   examples/11-nevergrad-optimization.ipynb
   examples/12-scipy-optimization.ipynb
   examples/Modeling_resting_state_MEG_data.ipynb


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   changelog.md
   api.rst

