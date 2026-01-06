Building Multi-Region Networks
===============================

This tutorial covers creating and simulating large-scale brain networks with multiple regions.


Basic Network Setup
-------------------

A brain network consists of:

1. **Nodes**: Neural mass models representing brain regions
2. **Edges**: Structural connectivity between regions
3. **Coupling**: Mechanism for inter-regional communication

Simple Network Example
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   N_regions = 10

   # 1. Create node dynamics (uncoupled)
   nodes = brainmass.HopfOscillator(
       in_size=N_regions,
       omega=2 * jnp.pi * 10 * u.Hz,
       a=0.1,
   )

   # 2. Create structural connectivity
   W = jnp.ones((N_regions, N_regions)) * 0.05
   W = W.at[jnp.diag_indices(N_regions)].set(0.)  # no self-connections

   # 3. Create coupling
   coupling = brainmass.DiffusiveCoupling(conn=W, k=0.2)

   # 4. Initialize
   nodes.init_all_states()
   coupling.init_all_states()

   # 5. Simulation loop
   def network_step(i):
       x = nodes.x.value
       coupled_input = coupling(x, x)

       output = nodes.update()
       nodes.x.value += coupled_input

       return output

   network_activity = brainstate.transform.for_loop(
       network_step,
       jnp.arange(1000)
   )


Structural Connectivity
-----------------------

Loading from DTI
^^^^^^^^^^^^^^^^

.. code-block:: python

   import jax.numpy as jnp

   # Load structural connectivity from DTI tractography
   # Common formats: .npy, .mat, .txt
   SC = jnp.load('structural_connectivity.npy')  # shape (N, N)

   # Typically: SC[i, j] = fiber density from j → i

   # Normalize (common preprocessing)
   SC_norm = SC / SC.sum(axis=0, keepdims=True)  # column normalization

   # Or row normalization
   SC_norm = SC / SC.sum(axis=1, keepdims=True)


Creating Synthetic Networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Random Network:**

.. code-block:: python

   import jax

   key = jax.random.PRNGKey(0)
   N = 90

   # Random weights
   W_random = jax.random.uniform(key, (N, N)) * 0.1
   W_random = W_random.at[jnp.diag_indices(N)].set(0.)


**Small-World Network:**

.. code-block:: python

   # Simplified small-world (ring + random shortcuts)
   N = 90
   k = 4  # nearest neighbors

   # Ring lattice
   W = jnp.zeros((N, N))
   for i in range(N):
       for j in range(1, k//2 + 1):
           W = W.at[i, (i+j) % N].set(0.1)
           W = W.at[i, (i-j) % N].set(0.1)

   # Add random shortcuts (rewiring)
   p_rewire = 0.1
   key = jax.random.PRNGKey(42)
   # ... rewiring logic ...


**Hub Network:**

.. code-block:: python

   N = 90
   N_hubs = 5

   W = jnp.zeros((N, N))

   # Hubs connect to all
   W = W.at[:N_hubs, :].set(0.2)
   W = W.at[:, :N_hubs].set(0.2)

   # Remove self-connections
   W = W.at[jnp.diag_indices(N)].set(0.)


Realistic Brain Networks
-------------------------

Using Anatomical Atlases
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Example: AAL90 atlas
   N_AAL90 = 90

   # Load AAL90 connectivity
   SC_AAL90 = jnp.load('AAL90_SC.npy')  # from DTI

   # Create network
   nodes = brainmass.WilsonCowanModel(in_size=N_AAL90)
   coupling = brainmass.DiffusiveCoupling(conn=SC_AAL90, k=0.1)


**Common Atlases:**

- AAL (Automated Anatomical Labeling): 90/116 regions
- Desikan-Killiany: 68 cortical regions
- Destrieux: 148 cortical regions
- Schaefer: 100/200/400/etc. parcels


Distance-Dependent Delays
^^^^^^^^^^^^^^^^^^^^^^^^^^

For large-scale networks, account for axonal conduction delays:

.. code-block:: python

   # Distance matrix (mm)
   distances = jnp.load('region_distances.npy')  # shape (N, N)

   # Conduction velocity (m/s)
   velocity = 6.0  # typical: 3-9 m/s

   # Compute delays (ms)
   delays_ms = (distances / velocity).astype(int)  # in time steps

   # Implement with circular buffer (simplified)
   max_delay = delays_ms.max()
   history = jnp.zeros((max_delay, N_regions))

   def step_with_delay(i, hist):
       # Get delayed activity for each connection
       # ... (implementation depends on delay structure) ...
       pass


Heterogeneous Networks
-----------------------

Different Models per Region
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Thalamus: fast oscillators
   N_thal = 10
   thalamus = brainmass.HopfOscillator(
       in_size=N_thal,
       omega=2 * jnp.pi * 40 * u.Hz,  # 40 Hz
   )

   # Cortex: excitatory-inhibitory dynamics
   N_cort = 80
   cortex = brainmass.WilsonCowanModel(in_size=N_cort)

   # Coupling between subsystems
   W_thal_cort = jnp.ones((N_cort, N_thal)) * 0.1  # thalamus → cortex
   W_cort_thal = jnp.ones((N_thal, N_cort)) * 0.05  # cortex → thalamus

   def hetero_network_step(i):
       # Thalamus dynamics
       thal_out = thalamus.update()

       # Cortex receives thalamic input
       thal_drive = (W_thal_cort @ thal_out).mean()
       cort_out = cortex.update(rE_inp=thal_drive, rI_inp=0.)

       # Thalamus receives cortical feedback
       cort_feedback = (W_cort_thal @ cortex.rE.value).mean()
       thalamus.x.value += cort_feedback * 0.1

       return cort_out


Region-Specific Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Different parameters for each region
   N = 90

   # Example: heterogeneous excitability
   a_values = jax.random.uniform(jax.random.PRNGKey(0), (N,)) * 0.2  # 0-0.2 range

   # Manually apply per-region (requires custom implementation)
   # Or use batched models with different parameters


Network Analysis
----------------

Computing Functional Connectivity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Simulate network
   activity = brainstate.transform.for_loop(network_step, jnp.arange(10000))

   # Compute FC (Pearson correlation)
   activity_centered = activity - activity.mean(axis=0)
   FC = jnp.corrcoef(activity_centered.T)  # shape (N, N)

   # Visualize
   import matplotlib.pyplot as plt
   plt.imshow(FC, cmap='RdBu_r', vmin=-1, vmax=1)
   plt.colorbar()
   plt.title('Functional Connectivity')


Network Synchrony
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Kuramoto order parameter
   def kuramoto_order(phases):
       \"\"\"Measure global synchronization\"\"\"
       z = jnp.mean(jnp.exp(1j * phases))
       return jnp.abs(z)  # R ∈ [0, 1]

   # For Kuramoto network
   kuramoto_net = brainmass.KuramotoNetwork(in_size=100, omega_mean=10*u.Hz)
   kuramoto_net.init_all_states()

   order_params = []
   for i in range(1000):
       phases = kuramoto_net.theta.value
       R = kuramoto_order(phases)
       order_params.append(R)
       kuramoto_net.update()


Complete Network Example
-------------------------

Whole-Brain Resting-State Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   # Parameters
   N_regions = 90  # AAL90 atlas
   coupling_strength = 0.2
   simulation_time = 600  # seconds
   dt = 1 * u.ms
   T_steps = int((simulation_time * u.second / dt).magnitude)

   # Load structural connectivity
   SC = jnp.load('AAL90_SC_normalized.npy')

   # Create components
   nodes = brainmass.WongWangModel(in_size=N_regions)
   coupling = brainmass.DiffusiveCoupling(conn=SC, k=coupling_strength)

   # Add noise for spontaneous activity
   nodes.noise_E = brainmass.OUProcess(
       in_size=N_regions,
       sigma=0.01 * u.Hz,
       tau=100 * u.ms,
   )

   # BOLD forward model
   bold = brainmass.BOLDSignal(in_size=N_regions)

   # Initialize
   nodes.init_all_states()
   coupling.init_all_states()
   bold.init_all_states()

   # Simulate
   print("Running simulation...")
   neural_activity = []

   for t in range(T_steps):
       # Get synaptic activity
       S_E = nodes.S_E.value

       # Apply coupling
       coupled_input = coupling(S_E, S_E)

       # Update nodes
       output = nodes.update(S_E_ext=coupled_input)
       neural_activity.append(output)

       if t % 10000 == 0:
           print(f"Step {t}/{T_steps}")

   neural_activity = jnp.stack(neural_activity)

   # Generate BOLD
   print("Generating BOLD signal...")
   bold_ts = []
   for z in neural_activity:
       bold.update(z=z)
       bold_ts.append(bold.bold())

   bold_ts = jnp.stack(bold_ts)

   # Downsample to TR = 2s
   TR_steps = int((2 * u.second / dt).magnitude)
   bold_downsampled = bold_ts[::TR_steps]

   # Compute FC
   FC_sim = jnp.corrcoef(bold_downsampled.T)

   print(f"Simulated FC shape: {FC_sim.shape}")


Best Practices
--------------

1. **Start Small**: Test with N=10-20 regions before scaling to N=90+
2. **Normalize Connectivity**: Prevent unstable dynamics from unnormalized SC
3. **Monitor Dynamics**: Plot time series to check for explosions/collapse
4. **Use Noise**: Spontaneous fluctuations prevent fixed points
5. **Check Timescales**: Match dt to fastest dynamics in the network
6. **Profile Performance**: Use JAX profiling for large networks


Common Issues
-------------

**Exploding Activity:**

- Reduce coupling strength `k`
- Normalize connectivity matrix
- Check for positive feedback loops

**No Synchronization:**

- Increase coupling strength
- Check connectivity topology
- Ensure sufficient simulation time

**Slow Simulation:**

- Use JIT compilation: `@jax.jit`
- Reduce number of regions for testing
- Use simpler models (Hopf vs Jansen-Rit)


Next Steps
----------

- :doc:`adding_coupling` - Advanced coupling mechanisms
- :doc:`forward_modeling` - Map network activity to neuroimaging signals
- :doc:`parameter_fitting` - Optimize network parameters


See Also
--------

- :doc:`../api/coupling` - Coupling API reference
- :doc:`../api/models` - Node model options
- :doc:`../examples/index` - Network simulation examples
