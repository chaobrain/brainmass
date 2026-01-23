Quickstart
==========

This 5-minute tutorial demonstrates the basic ``brainmass`` workflow: create a neural mass model,
add noise, simulate dynamics, and visualize results.


Your First Simulation
----------------------

Let's simulate a single brain region using the Hopf oscillator model:

.. code-block:: python

   import brainmass
   import brainunit as u
   import brainstate
   import jax.numpy as jnp
   import matplotlib.pyplot as plt

   # 1. Create a Hopf oscillator (10 Hz oscillation)
   model = brainmass.HopfOscillator(
       in_size=1,              # single region
       omega=2 * jnp.pi * 10 * u.Hz,  # 10 Hz frequency
       a=0.1,                  # bifurcation parameter (>0 for oscillations)
   )

   # 2. Initialize model state
   model.init_all_states()

   # 3. Run simulation for 1000 time steps
   def step(i):
       return model.update()

   time_series = brainstate.transform.for_loop(step, jnp.arange(1000))

   # 4. Visualize
   plt.figure(figsize=(10, 4))
   plt.plot(time_series[:, 0])  # plot x coordinate
   plt.xlabel('Time step')
   plt.ylabel('Activity')
   plt.title('Hopf Oscillator Dynamics')
   plt.show()


**What's happening:**

- We create a 10 Hz oscillator (``omega = 2π × 10``)
- ``a > 0`` puts the oscillator in the limit cycle regime
- ``for_loop`` efficiently runs 1000 simulation steps
- The output is a (1000, 1) array of the oscillator's x-coordinate


Adding Noise
------------

Real neural activity is noisy. Let's add stochastic fluctuations:

.. code-block:: python

   import brainmass
   import brainunit as u
   import brainstate
   import jax.numpy as jnp

   # Create oscillator
   model = brainmass.HopfOscillator(in_size=1, omega=2 * jnp.pi * 10 * u.Hz, a=0.1)

   # Add Ornstein-Uhlenbeck noise
   model.noise = brainmass.OUProcess(
       in_size=1,
       sigma=0.5 * u.Hz,  # noise amplitude
       tau=20. * u.ms,    # correlation time
   )

   # Initialize
   model.init_all_states()

   # Simulate with noise
   time_series = brainstate.transform.for_loop(
       lambda i: model.update(),
       jnp.arange(1000)
   )


The noise is automatically added during ``update()``, creating more realistic fluctuations.


Multi-Region Networks
----------------------

Simulate multiple brain regions simultaneously:

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   N_regions = 10

   # Create network of Wilson-Cowan models
   network = brainmass.WilsonCowanModel(
       in_size=N_regions,
       tau_E=10. * u.ms,
       tau_I=20. * u.ms,
   )

   # Add noise to each population
   network.noise_E = brainmass.OUProcess(
       in_size=N_regions,
       sigma=0.3 * u.Hz,
       tau=20. * u.ms
   )

   network.noise_I = brainmass.OUProcess(
       in_size=N_regions,
       sigma=0.2 * u.Hz,
       tau=30. * u.ms
   )

   # Initialize
   network.init_all_states()

   # Simulate
   def step(i):
       return network.update(rE_inp=0.5, rI_inp=0.2)

   exc_rates = brainstate.transform.for_loop(step, jnp.arange(2000))

   # exc_rates has shape (2000, 10) - excitatory rates for 10 regions


**Note:** ``in_size=N_regions`` creates N independent copies of the model (no coupling yet).


Adding Coupling
---------------

Connect regions with diffusive coupling:

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   N = 10

   # Create connectivity matrix (random for demo)
   W = jax.random.uniform(jax.random.PRNGKey(0), (N, N)) * 0.1
   W = W.at[jnp.diag_indices(N)].set(0.)  # no self-connections

   # Create models
   nodes = brainmass.HopfOscillator(in_size=N, omega=10 * u.Hz)
   coupling = brainmass.DiffusiveCoupling(conn=W, k=0.2)

   # Initialize
   nodes.init_all_states()
   coupling.init_all_states()

   # Network simulation with coupling
   def network_step(i):
       x = nodes.x.value
       coupled_input = coupling(x, x)

       output = nodes.update()
       nodes.x.value += coupled_input  # add coupling to state

       return output

   network_activity = brainstate.transform.for_loop(
       network_step,
       jnp.arange(1000)
   )


Forward Modeling (BOLD Signal)
-------------------------------

Map neural activity to fMRI BOLD signal:

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   N_regions = 5

   # Neural mass model
   nmm = brainmass.WilsonCowanModel(in_size=N_regions)
   nmm.init_all_states()

   # BOLD hemodynamic model
   bold_model = brainmass.BOLDSignal(in_size=N_regions)
   bold_model.init_all_states()

   # Simulate neural activity
   def sim_neural(i):
       return nmm.update(rE_inp=0.5, rI_inp=0.2)

   neural_activity = brainstate.transform.for_loop(
       sim_neural,
       jnp.arange(5000)  # 5000 time steps
   )

   # Generate BOLD signal from neural activity
   def sim_bold(z):
       bold_model.update(z=z)
       return bold_model.bold()

   bold_signal = brainstate.transform.for_loop(
       sim_bold,
       neural_activity
   )

   # bold_signal shape: (5000, 5)
   # Downsample to TR ~ 2s if needed: bold_signal[::2000]


Common Patterns
---------------

**Pattern 1: Batched Simulations**

Run multiple simulations in parallel:

.. code-block:: python

   batch_size = 32

   model = brainmass.HopfOscillator(in_size=10, omega=10 * u.Hz)
   model.init_all_states(batch_size=batch_size)

   output = model.update()  # shape: (32, 10)


**Pattern 2: Accessing Internal States**

.. code-block:: python

   model = brainmass.WilsonCowanModel(in_size=5)
   model.init_all_states()

   model.update(rE_inp=0.5, rI_inp=0.2)

   # Access internal state variables
   exc_rate = model.rE.value  # excitatory firing rate
   inh_rate = model.rI.value  # inhibitory firing rate


**Pattern 3: Custom Time Steps**

.. code-block:: python

   # Default dt is typically 1 ms
   # To change, set during model creation or manually integrate

   model = brainmass.HopfOscillator(in_size=1, omega=10 * u.Hz, dt=0.5 * u.ms)


Quick Reference
---------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Task
     - Code
   * - Create model
     - ``model = brainmass.ModelName(in_size=N, **params)``
   * - Initialize state
     - ``model.init_all_states(batch_size=None)``
   * - Single step
     - ``output = model.update(**inputs)``
   * - Simulate loop
     - ``brainstate.transform.for_loop(lambda i: model.update(), jnp.arange(T))``
   * - Add noise
     - ``model.noise = brainmass.OUProcess(in_size=N, sigma=..., tau=...)``
   * - Access state
     - ``value = model.state_var.value``
   * - Reset state
     - ``model.reset_state()``


Next Steps
----------

Now that you've run your first simulations:

1. :doc:`choosing_models` - Learn which model to use for your application
2. :doc:`building_networks` - Create realistic multi-region networks
3. :doc:`parameter_fitting` - Optimize parameters to match empirical data
4. :doc:`../examples/index` - Explore complete examples


Tips for Beginners
-------------------

- **Start simple:** Single region, no noise, no coupling
- **Visualize often:** Plot time series to understand dynamics
- **Check units:** Use ``brainunit`` quantities to avoid unit errors
- **Read docstrings:** Use ``help(brainmass.ModelName)`` for parameter details
- **Use examples:** The :doc:`../examples/index` are your best resource


See Also
--------

- :doc:`../api/models` - Complete model reference
- :doc:`../api/noise` - Noise processes documentation
- :doc:`../api/coupling` - Coupling mechanisms
- :doc:`../api/forward` - Forward models
