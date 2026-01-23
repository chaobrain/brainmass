Installation
============

This guide covers installing ``brainmass`` for different hardware configurations.


Quick Installation
------------------

For most users, CPU installation is the simplest:

.. code-block:: bash

   pip install -U brainmass[cpu]


Hardware-Specific Installation
-------------------------------

CPU
^^^

For CPU-only systems (recommended for getting started):

.. tab-set::

   .. tab-item:: Linux/macOS

      .. code-block:: bash

         pip install -U brainmass[cpu]

   .. tab-item:: Windows

      .. code-block:: bash

         pip install -U brainmass[cpu]


GPU (CUDA)
^^^^^^^^^^

For NVIDIA GPUs with CUDA support:

.. tab-set::

   .. tab-item:: CUDA 12.x

      .. code-block:: bash

         pip install -U brainmass[cuda12]

   .. tab-item:: CUDA 13.x

      .. code-block:: bash

         pip install -U brainmass[cuda13]


**Requirements:**

- NVIDIA GPU with compute capability ≥ 3.5
- CUDA toolkit installed (12.x or 13.x)
- cuDNN library

**Troubleshooting CUDA:**

If GPU is not detected:

.. code-block:: python

   import jax
   print(jax.devices())  # Should show GPU devices

If only CPU is shown, reinstall JAX with CUDA support:

.. code-block:: bash

   pip install --upgrade jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


TPU
^^^

For Google Cloud TPU:

.. code-block:: bash

   pip install -U brainmass[tpu]

**Note:** TPU support requires running on Google Cloud Platform with TPU VMs.


Development Installation
-------------------------

To install from source for development:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/chaobrain/brainmass.git
   cd brainmass

   # Install in editable mode with development dependencies
   pip install -e .[dev,doc]

**Development dependencies include:**

- Testing: ``pytest``, ``pytest-cov``
- Documentation: ``sphinx``, ``sphinx-book-theme``, etc.
- Code quality: ``ruff``, ``mypy``


Verifying Installation
-----------------------

After installation, verify that everything works:

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u

   print(f"brainmass version: {brainmass.__version__}")

   # Test basic functionality
   model = brainmass.HopfOscillator(in_size=10, omega=10 * u.Hz)
   model.init_all_states()
   output = model.update()

   print(f"Model output shape: {output.shape}")
   print("✓ Installation successful!")


Dependencies
------------

``brainmass`` depends on:

**Core:**

- ``jax`` ≥ 0.4.0: For numerical computations and automatic differentiation
- ``brainstate`` ≥ 0.2.9: State management for dynamical systems
- ``brainunit``: Unit-aware array operations
- ``braintools``: Utility functions and transforms
- ``numpy``: NumPy array operations

**Optional:**

- ``matplotlib``: For visualization (examples)
- ``nevergrad``: Gradient-free optimization (parameter fitting)
- ``scipy``: SciPy optimizers (parameter fitting)

Install optional dependencies:

.. code-block:: bash

   pip install brainmass[opt]  # Optimization libraries
   pip install brainmass[viz]  # Visualization



Platform-Specific Notes
-----------------------

**macOS with Apple Silicon (M1/M2):**

- Use CPU installation
- JAX has limited GPU support for Apple Silicon
- Performance is still excellent on M1/M2 CPUs

**Windows:**

- GPU support requires WSL2 for best compatibility
- Native Windows GPU support is experimental

**Linux:**

- Recommended platform for GPU/TPU
- Most straightforward CUDA setup


Next Steps
----------

After installation:

1. Follow the :doc:`quickstart` tutorial for your first simulation
2. Read the :doc:`choosing_models` guide to select appropriate models
3. Explore the :doc:`../examples/index` for practical applications


See Also
--------

- `JAX Installation Guide <https://github.com/google/jax#installation>`_
- :doc:`../faq` for troubleshooting
- `GitHub Issues <https://github.com/chaobrain/brainmass/issues>`_ for bug reports
