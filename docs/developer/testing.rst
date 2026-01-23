Testing
=======

Testing guidelines for ``brainmass``.


Running Tests
-------------

Run all tests:

.. code-block:: bash

   pytest tests/

Run specific test file:

.. code-block:: bash

   pytest tests/test_models.py

Run with coverage:

.. code-block:: bash

   pytest --cov=brainmass tests/


Test Structure
--------------

.. code-block:: text

   tests/
   ├── test_models.py          # Neural mass model tests
   ├── test_noise.py           # Noise process tests
   ├── test_coupling.py        # Coupling mechanism tests
   ├── test_forward.py         # Forward model tests
   └── test_utils.py           # Utility function tests


Writing Tests
-------------

Basic Test
^^^^^^^^^^

.. code-block:: python

   import pytest
   import jax.numpy as jnp
   import brainmass

   def test_hopf_oscillator():
       # Create model
       model = brainmass.HopfOscillator(in_size=10, omega=10)

       # Initialize
       model.init_all_states()

       # Test update
       output = model.update()

       # Assertions
       assert output.shape == (10,)
       assert not jnp.any(jnp.isnan(output))


Parametrized Tests
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   @pytest.mark.parametrize("in_size", [1, 10, 100])
   def test_model_shapes(in_size):
       model = brainmass.WilsonCowanModel(in_size=in_size)
       model.init_all_states()

       output = model.update(rE_inp=0.5, rI_inp=0.2)
       assert output.shape == (in_size,)


Batch Testing
^^^^^^^^^^^^^

.. code-block:: python

   @pytest.mark.parametrize("batch_size", [None, 1, 32])
   def test_batching(batch_size):
       model = brainmass.HopfOscillator(in_size=5, omega=10)
       model.init_all_states(batch_size=batch_size)

       output = model.update()

       if batch_size is None:
           assert output.shape == (5,)
       else:
           assert output.shape == (batch_size, 5)


Test Best Practices
-------------------

1. **Test one thing per test**
2. **Use descriptive test names**
3. **Check shapes, values, and edge cases**
4. **Test with and without batch dimension**
5. **Test with units where applicable**


Coverage
--------

Aim for >80% code coverage:

.. code-block:: bash

   pytest --cov=brainmass --cov-report=html tests/
   # Open htmlcov/index.html


Continuous Integration
----------------------

Tests run automatically on:

- Every pull request
- Pushes to main branch
- Nightly builds

Ensure tests pass before submitting PR.


See Also
--------

- pytest documentation
- :doc:`contributing` - Contribution process
