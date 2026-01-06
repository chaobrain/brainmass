Extending Noise Processes
=========================

Guide to creating custom noise processes.


Noise Template
--------------

Stateless Noise
^^^^^^^^^^^^^^^

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import brainstate

   class MyStatelessNoise(brainstate.nn.Dynamics):
       """Custom stateless noise (i.i.d. samples).

       Args:
           in_size: Shape of noise output
           sigma: Noise amplitude
       """

       def __init__(self, in_size, sigma=1.0):
           super().__init__()
           self.in_size = in_size
           self.sigma = sigma

       def init_state(self, batch_size=None):
           # No state for stateless noise
           pass

       def update(self):
           """Generate noise sample."""
           key = jax.random.PRNGKey(0)  # Use proper key management
           noise = jax.random.normal(key, self.in_size) * self.sigma
           return noise


Stateful Noise
^^^^^^^^^^^^^^

.. code-block:: python

   class MyStatefulNoise(brainstate.nn.Dynamics):
       """Custom stateful noise process.

       Args:
           in_size: Shape of noise output
           sigma: Noise amplitude
           tau: Correlation time
       """

       def __init__(self, in_size, sigma=1.0, tau=10.0):
           super().__init__()
           self.in_size = in_size
           self.sigma = sigma
           self.tau = tau

           # Internal state
           self.x = brainstate.HiddenState(brainstate.init.Constant(0.))

       def init_state(self, batch_size=None):
           shape = (batch_size, *self.in_size) if batch_size else self.in_size
           self.x.value = jnp.zeros(shape)

       def update(self):
           x = self.x.value

           # Dynamics: dx/dt = -x/tau + sigma*sqrt(2/tau)*xi
           key = jax.random.PRNGKey(0)  # Use proper key
           xi = jax.random.normal(key, x.shape)

           dt = 1.0  # ms
           dx = (-x / self.tau + self.sigma * jnp.sqrt(2 / self.tau) * xi) * dt

           self.x.value = x + dx
           return self.x.value


Example: Exponential Noise
---------------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import brainstate

   class ExponentialNoise(brainstate.nn.Dynamics):
       """Exponentially distributed noise.

       Args:
           in_size: Output shape
           rate: Rate parameter (lambda)
       """

       def __init__(self, in_size, rate=1.0):
           super().__init__()
           self.in_size = in_size
           self.rate = rate

       def init_state(self, batch_size=None):
           pass

       def update(self):
           key = jax.random.PRNGKey(0)
           return jax.random.exponential(key, shape=self.in_size) / self.rate


Proper Key Management
----------------------

Use ``brainstate`` random key infrastructure:

.. code-block:: python

   class ProperNoise(brainstate.nn.Dynamics):
       def __init__(self, in_size, sigma=1.0):
           super().__init__()
           self.in_size = in_size
           self.sigma = sigma

       def init_state(self, batch_size=None):
           pass

       def update(self):
           # Use brainstate's random key management
           key = brainstate.random.split_key()
           noise = jax.random.normal(key, self.in_size) * self.sigma
           return noise


Unit-Aware Noise
----------------

.. code-block:: python

   import brainunit as u

   class UnitAwareNoise(brainstate.nn.Dynamics):
       def __init__(self, in_size, sigma=1.0*u.Hz):
           super().__init__()
           self.in_size = in_size
           self.sigma = sigma

       def update(self):
           key = brainstate.random.split_key()
           noise_unitless = jax.random.normal(key, self.in_size)
           return noise_unitless * self.sigma  # Preserves units


Testing
-------

.. code-block:: python

   def test_my_noise():
       noise = MyStatefulNoise(in_size=(10,), sigma=1.0, tau=10.0)
       noise.init_all_states()

       # Test output shape
       sample = noise.update()
       assert sample.shape == (10,)

       # Test batch
       noise.init_all_states(batch_size=32)
       sample = noise.update()
       assert sample.shape == (32, 10)


See Also
--------

- :doc:`creating_models` - Custom model creation
- :doc:`../api/noise` - Noise API reference
