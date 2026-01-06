Utility Functions
=================

.. currentmodule:: brainmass

``brainmass`` provides several utility functions for common operations in neural mass modeling.


API Reference
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   sys2nd
   sigmoid
   bounded_input
   process_sequence


sys2nd
^^^^^^

.. autofunction:: sys2nd

Converts a second-order differential equation to a first-order system.

**Mathematical Background:**

A second-order ODE:

.. math::

   \ddot{x} = f(x, \dot{x}, t)

can be rewritten as a first-order system:

.. math::

   \dot{x} &= v \\
   \dot{v} &= f(x, v, t)

**Example:**

.. code-block:: python

   import brainmass
   import jax.numpy as jnp

   # Harmonic oscillator: ẍ + ω²x = 0
   omega = 2.0

   x = 1.0  # position
   v = 0.0  # velocity

   # Acceleration from second-order equation
   acc = -omega**2 * x

   # Convert to first-order updates
   dx_dt, dv_dt = brainmass.sys2nd(v, acc)

   # Euler integration
   dt = 0.01
   x_new = x + dx_dt * dt
   v_new = v + dv_dt * dt


sigmoid
^^^^^^^

.. autofunction:: sigmoid

Standard sigmoid (logistic) activation function.

**Formula:**

.. math::

   \sigma(x) = \frac{1}{1 + e^{-x}}

**Properties:**
- Range: (0, 1)
- Smooth and differentiable everywhere
- Used in neural activation, probability mapping

**Example:**

.. code-block:: python

   import brainmass
   import jax.numpy as jnp

   x = jnp.array([-2, -1, 0, 1, 2])
   y = brainmass.sigmoid(x)
   # y ≈ [0.119, 0.269, 0.5, 0.731, 0.881]

   # Common in firing rate models
   def firing_rate(membrane_potential, threshold, gain):
       shifted = gain * (membrane_potential - threshold)
       return brainmass.sigmoid(shifted)


bounded_input
^^^^^^^^^^^^^

.. autofunction:: bounded_input

Bounds input values to a specified range, useful for ensuring physiological constraints.

**Example:**

.. code-block:: python

   import brainmass
   import jax.numpy as jnp

   # Bound firing rates to [0, 100] Hz
   rates = jnp.array([-10, 50, 150])
   bounded_rates = brainmass.bounded_input(rates, lower=0, upper=100)
   # bounded_rates = [0, 50, 100]

   # Bound membrane potentials to realistic range
   V = jnp.array([-80, -65, -40, 20])  # mV
   V_bounded = brainmass.bounded_input(V, lower=-80, upper=0)
   # V_bounded = [-80, -65, -40, 0]


process_sequence
^^^^^^^^^^^^^^^^

.. autofunction:: process_sequence

Processes a sequence through a model, handling state management automatically.

**Example:**

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainstate

   # Create model
   model = brainmass.HopfOscillator(in_size=5, omega=10)
   model.init_all_states()

   # Input sequence
   input_sequence = jnp.randn(100, 5)  # (time_steps, in_size)

   # Process sequence
   outputs = brainmass.process_sequence(model, input_sequence)
   # outputs has shape (100, output_dim)

   # Equivalent to:
   # outputs = brainstate.transform.for_loop(
   #     lambda t: model.update(input_sequence[t]),
   #     jnp.arange(100)
   # )


Common Usage Patterns
---------------------

Building Custom Activation Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp

   def threshold_linear(x, threshold=0.0, slope=1.0):
       """Rectified linear with custom threshold"""
       return brainmass.bounded_input(
           slope * (x - threshold),
           lower=0.0,
           upper=jnp.inf
       )

   def soft_threshold(x, threshold=0.0, gain=1.0):
       """Smooth threshold using sigmoid"""
       return brainmass.sigmoid(gain * (x - threshold))


Implementing Custom Second-Order Dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u

   class CustomOscillator:
       def __init__(self, omega, damping):
           self.omega = omega
           self.damping = damping
           self.x = 0.0
           self.v = 0.0

       def update(self, external_force):
           # Second-order equation: ẍ + 2ζωẋ + ω²x = F
           acc = external_force - 2 * self.damping * self.omega * self.v \
                 - self.omega**2 * self.x

           # Convert to first-order
           dx_dt, dv_dt = brainmass.sys2nd(self.v, acc)

           # Integrate (Euler)
           dt = 0.001
           self.x += dx_dt * dt
           self.v += dv_dt * dt

           return self.x


Bounded Sigmoid for Physiological Ranges
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def physiological_sigmoid(x, x_min, x_max, v_min, v_max):
       """Map input range [x_min, x_max] to output [v_min, v_max]"""
       # Normalize to [0, 1]
       x_norm = (x - x_min) / (x_max - x_min)

       # Apply sigmoid for smoothness
       y_norm = brainmass.sigmoid(10 * (x_norm - 0.5))  # steepness = 10

       # Scale to output range
       return v_min + (v_max - v_min) * y_norm

   # Example: membrane potential to firing rate
   V = jnp.array([-70, -60, -55, -50, -40])  # mV
   rates = physiological_sigmoid(V, x_min=-70, x_max=-40, v_min=0, v_max=100)


Tips and Best Practices
------------------------

**Numerical Stability:**
- Use ``bounded_input`` to prevent overflow/underflow
- Clip gradients for oscillator dynamics
- Check for NaN/Inf values in long simulations

**Unit Safety:**
- Utilities work with both unitless arrays and ``brainunit.Quantity``
- Ensure consistent units when combining utility outputs with model states

**Performance:**
- These utilities are JIT-compiled by JAX for efficiency
- Use them inside ``jax.jit`` decorated functions

**Debugging:**
- Use ``bounded_input`` to catch out-of-range values during development
- ``process_sequence`` simplifies debugging sequential models


See Also
--------

- :doc:`models` - Neural mass models that use these utilities
- :doc:`types` - Type aliases for function signatures
- JAX documentation for ``jax.numpy`` and ``jax.nn`` functions
