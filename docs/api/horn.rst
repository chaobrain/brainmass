HORN Models
===========

.. currentmodule:: brainmass

HORN (Harmonic Oscillator Recurrent Networks) are specialized recurrent neural networks based on
coupled harmonic oscillators. They excel at learning and generating temporal sequences with
complex dynamics.


Overview
--------

HORN models use harmonic oscillators as computational units, where the oscillatory dynamics
naturally encode temporal patterns. Unlike traditional RNNs with sigmoid or tanh activations,
HORN leverages the rich dynamics of coupled oscillators for sequence learning.

**Key Features:**
- Oscillator-based recurrent dynamics
- Natural handling of periodic and quasi-periodic sequences
- Interpretable dynamical systems approach
- Compatible with gradient-based optimization


Architecture
------------

HORN models consist of three main components:

1. **HORNStep**: Single time step update of coupled oscillators
2. **HORNSeqLayer**: Layer that processes sequential inputs
3. **HORNSeqNetwork**: Full network with multiple HORN layers


API Reference
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HORNStep
   HORNSeqLayer
   HORNSeqNetwork


HORNStep
^^^^^^^^

:class:`HORNStep` implements a single dynamics step for a population of coupled harmonic oscillators:

.. math::

   \ddot{x} + 2\zeta\omega\dot{x} + \omega^2 x = f(x, \text{input})

where :math:`\zeta` is damping, :math:`\omega` is natural frequency, and :math:`f` is a coupling function.

**Example:**

.. code-block:: python

   import brainmass
   import jax.numpy as jnp

   # Create HORN step with 10 oscillators
   horn_step = brainmass.HORNStep(
       in_size=5,    # input dimension
       num_osc=10,   # number of oscillators
       omega=1.0,    # natural frequency
       zeta=0.1,     # damping coefficient
   )
   horn_step.init_all_states()

   # Single step update
   x_input = jnp.randn(5)
   x_out = horn_step.update(x_input)


HORNSeqLayer
^^^^^^^^^^^^

:class:`HORNSeqLayer` wraps :class:`HORNStep` to process sequences:

.. code-block:: python

   import brainstate

   horn_layer = brainmass.HORNSeqLayer(
       in_size=5,
       num_osc=10,
       omega=1.0,
       zeta=0.1,
   )
   horn_layer.init_all_states()

   # Process sequence
   sequence = jnp.randn(100, 5)  # (time_steps, in_size)

   outputs = brainstate.transform.for_loop(
       lambda t: horn_layer.update(sequence[t]),
       jnp.arange(100)
   )


HORNSeqNetwork
^^^^^^^^^^^^^^

:class:`HORNSeqNetwork` stacks multiple HORN layers to create a deep recurrent network:

.. code-block:: python

   horn_net = brainmass.HORNSeqNetwork(
       in_size=5,
       hidden_sizes=[20, 20, 10],  # 3 HORN layers
       out_size=3,                  # output dimension
       omega=1.0,
       zeta=0.1,
   )
   horn_net.init_all_states()

   # Forward pass through network
   sequence = jnp.randn(100, 5)

   def forward_step(t):
       return horn_net.update(sequence[t])

   predictions = brainstate.transform.for_loop(forward_step, jnp.arange(100))


Use Cases
---------

Sequence Generation
^^^^^^^^^^^^^^^^^^^

HORN networks can learn to generate temporal sequences:

.. code-block:: python

   import brainmass
   import jax
   import jax.numpy as jnp
   import brainstate

   # Create generator network
   generator = brainmass.HORNSeqNetwork(
       in_size=1,           # seed input
       hidden_sizes=[50, 50],
       out_size=10,         # sequence dimension
       omega=2.0,
       zeta=0.05,
   )
   generator.init_all_states()

   # Generate sequence
   seed = jnp.array([1.0])

   generated_sequence = []
   for t in range(500):
       output = generator.update(seed)
       generated_sequence.append(output)
       seed = output[:1]  # feedback

   generated_sequence = jnp.stack(generated_sequence)


Time Series Prediction
^^^^^^^^^^^^^^^^^^^^^^^

Predict future values of a time series:

.. code-block:: python

   # Training data
   time_series = ...  # shape (T, D)

   predictor = brainmass.HORNSeqNetwork(
       in_size=time_series.shape[1],
       hidden_sizes=[100, 50],
       out_size=time_series.shape[1],
       omega=1.5,
       zeta=0.1,
   )
   predictor.init_all_states()

   # Training loop (simplified)
   def loss_fn(params, inputs, targets):
       # Forward pass with params
       predictions = ...
       return jnp.mean((predictions - targets) ** 2)

   # Optimize with JAX
   optimizer = ...
   for epoch in range(num_epochs):
       for batch_inputs, batch_targets in dataloader:
           grads = jax.grad(loss_fn)(params, batch_inputs, batch_targets)
           params = optimizer.update(grads, params)


Oscillatory Pattern Recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Classify temporal patterns with oscillatory structure:

.. code-block:: python

   # HORN classifier
   classifier = brainmass.HORNSeqNetwork(
       in_size=64,          # e.g., sensor channels
       hidden_sizes=[100],
       out_size=5,          # number of classes
       omega=3.0,           # match expected oscillation frequency
       zeta=0.2,
   )
   classifier.init_all_states()

   # Classification
   input_signal = ...  # shape (time_steps, 64)

   logits = []
   for t in range(input_signal.shape[0]):
       output = classifier.update(input_signal[t])
       logits.append(output)

   # Final classification from last output or pooling
   final_logits = logits[-1]
   predicted_class = jnp.argmax(final_logits)


Parameter Selection
-------------------

**Natural Frequency (ω):**
- Should match the characteristic frequency of the data
- For data with dominant frequency :math:`f`, set :math:`\omega \approx 2\pi f`
- Multiple oscillators can have different frequencies to capture multi-scale dynamics

**Damping (ζ):**
- Controls oscillation decay
- :math:`\zeta < 1`: Underdamped (oscillatory)
- :math:`\zeta = 1`: Critically damped
- :math:`\zeta > 1`: Overdamped (no oscillations)
- Typical values: 0.05 - 0.5 for learning temporal patterns

**Number of Oscillators:**
- More oscillators increase capacity but also parameters
- Start with 50-100 oscillators per layer
- Scale based on sequence complexity


Training Considerations
-----------------------

**Initialization:**

Proper initialization is important for oscillator-based networks:

.. code-block:: python

   horn_net.init_all_states(batch_size=32)  # for batched training

   # Custom initialization of oscillator states
   for layer in horn_net.layers:
       # Initialize positions and velocities
       layer.x.value = jax.random.normal(key, layer.x.value.shape) * 0.1
       layer.v.value = jax.random.normal(key, layer.v.value.shape) * 0.01


**Gradient Clipping:**

Oscillator dynamics can have large gradients; use gradient clipping:

.. code-block:: python

   grads = jax.grad(loss_fn)(params, inputs, targets)

   # Clip gradients
   clipped_grads = jax.tree_map(
       lambda g: jnp.clip(g, -1.0, 1.0),
       grads
   )


**Learning Rate:**

Start with smaller learning rates (1e-4 to 1e-3) due to oscillatory dynamics.


Advantages and Limitations
---------------------------

**Advantages:**
- Natural temporal dynamics without gating mechanisms
- Interpretable oscillator-based representation
- Effective for periodic and quasi-periodic patterns
- Differentiable and trainable with standard optimizers

**Limitations:**
- More parameters than vanilla RNNs for same hidden size
- Requires tuning of oscillator parameters (ω, ζ)
- May not outperform LSTMs/GRUs on all sequence tasks
- Less established than traditional RNN architectures


Comparison with Traditional RNNs
---------------------------------

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Aspect
     - HORN
     - LSTM/GRU
   * - Dynamics
     - Oscillatory (physics-based)
     - Gated activations
   * - Interpretability
     - High (oscillator states)
     - Low (hidden states)
   * - Best for
     - Periodic/oscillatory patterns
     - General sequences
   * - Parameters
     - More (oscillator equations)
     - Fewer (compact gates)
   * - Training
     - Gradient-based (may need clipping)
     - Gradient-based (stable)


See Also
--------

- :doc:`models` - Neural mass models also use oscillator dynamics
- :doc:`../examples/basic/index` - Example notebooks with HORN models
- :doc:`../developer/creating_models` - Creating custom dynamical models
