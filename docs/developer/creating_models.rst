Creating Custom Models
======================

Guide to implementing custom neural mass models.


Model Template
--------------

.. code-block:: python

   import brainstate
   import brainunit as u
   import jax.numpy as jnp

   class MyCustomModel(brainstate.nn.Dynamics):
       """Custom neural mass model.

       Args:
           in_size: Number of regions/nodes
           tau: Time constant (ms)
           other_param: Description
       """

       def __init__(self, in_size, tau=10.*u.ms, other_param=1.0):
           super().__init__()

           # Store parameters
           self.in_size = in_size
           self.tau = tau
           self.other_param = other_param

           # Initialize state variables
           self.x = brainstate.HiddenState(
               brainstate.init.Constant(0.),
               sharding=brainstate.ShardingMode.REPLICATED
           )

       def init_state(self, batch_size=None):
           """Initialize state variables."""
           shape = (batch_size, *self.in_size) if batch_size else self.in_size
           self.x.value = jnp.zeros(shape)

       def update(self, external_input=0.):
           """Update dynamics by one time step.

           Args:
               external_input: External driving input

           Returns:
               Observable output
           """
           # Get current state
           x = self.x.value

           # Compute dynamics (example: dx/dt = -x/tau + input)
           dx_dt = -x / self.tau + external_input

           # Update state (Euler integration, dt assumed 1 ms)
           dt = 1 * u.ms
           self.x.value = x + dx_dt * dt

           # Return observable
           return self.x.value


Example: Custom Oscillator
---------------------------

.. code-block:: python

   import brainstate
   import brainunit as u
   import jax.numpy as jnp

   class DampedOscillator(brainstate.nn.Dynamics):
       """Damped harmonic oscillator.

       Equations:
           dx/dt = v
           dv/dt = -omega^2 * x - 2*zeta*omega*v + F_ext

       Args:
           in_size: Number of oscillators
           omega: Natural frequency (Hz)
           zeta: Damping ratio (dimensionless)
       """

       def __init__(self, in_size, omega=10*u.Hz, zeta=0.1):
           super().__init__()

           self.in_size = in_size
           self.omega = omega
           self.zeta = zeta

           # State variables: position and velocity
           self.x = brainstate.HiddenState(brainstate.init.Constant(0.))
           self.v = brainstate.HiddenState(brainstate.init.Constant(0.))

       def init_state(self, batch_size=None):
           shape = (batch_size, *self.in_size) if batch_size else self.in_size
           self.x.value = jnp.zeros(shape)
           self.v.value = jnp.zeros(shape)

       def update(self, F_ext=0.):
           x = self.x.value
           v = self.v.value

           # Dynamics
           dx_dt = v
           dv_dt = -(self.omega**2) * x - 2*self.zeta*self.omega*v + F_ext

           # Euler integration
           dt = 1 * u.ms
           self.x.value = x + dx_dt * dt
           self.v.value = v + dv_dt * dt

           return self.x.value


Adding Noise Support
---------------------

.. code-block:: python

   class MyModel(brainstate.nn.Dynamics):
       def __init__(self, in_size, noise=None):
           super().__init__()
           self.in_size = in_size
           self.noise = noise  # Optional noise source

           self.x = brainstate.HiddenState(brainstate.init.Constant(0.))

       def init_state(self, batch_size=None):
           shape = (batch_size, *self.in_size) if batch_size else self.in_size
           self.x.value = jnp.zeros(shape)

           # Initialize noise if present
           if self.noise is not None:
               self.noise.init_all_states(batch_size)

       def update(self, external_input=0.):
           x = self.x.value

           # Dynamics
           dx_dt = -x + external_input

           # Add noise if present
           if self.noise is not None:
               noise_value = self.noise.update()
               dx_dt += noise_value

           # Integrate
           dt = 1 * u.ms
           self.x.value = x + dx_dt * dt

           return self.x.value


Unit Handling
-------------

Ensure dimensional consistency:

.. code-block:: python

   import brainunit as u

   # Parameters with units
   tau = 10 * u.ms
   omega = 2 * jnp.pi * 10 * u.Hz

   # Computations preserve units
   frequency = 1 / tau  # Result: Hz
   period = 1 / omega  # Result: seconds


Testing Your Model
------------------

.. code-block:: python

   import pytest
   import jax.numpy as jnp

   def test_my_custom_model():
       # Create model
       model = MyCustomModel(in_size=10)
       model.init_all_states()

       # Test update
       output = model.update(external_input=1.0)

       # Check shape
       assert output.shape == (10,)

       # Test with batch
       model.init_all_states(batch_size=32)
       output = model.update()
       assert output.shape == (32, 10)


Documentation
-------------

Add comprehensive docstrings:

.. code-block:: python

   class MyModel(brainstate.nn.Dynamics):
       """One-line description.

       Detailed explanation of the model, including:
       - Mathematical equations
       - Biological interpretation
       - References to papers

       Mathematical Details:

       .. math::

           \\frac{dx}{dt} = f(x, t)

       Args:
           in_size: Shape of input/number of regions
           param1: Description with units
           param2: Description with default behavior

       Examples:

           >>> model = MyModel(in_size=10)
           >>> model.init_all_states()
           >>> output = model.update()

       References:
           Author et al. (2020). Paper title. Journal.
       """


Contributing Your Model
-----------------------

1. Implement model following template
2. Add tests in ``tests/``
3. Create example notebook in ``examples/``
4. Update API documentation
5. Submit pull request


See Also
--------

- :doc:`architecture` - Package structure
- :doc:`extending_noise` - Custom noise processes
- :doc:`contributing` - Contribution guidelines
- ``brainstate`` documentation
