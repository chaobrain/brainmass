Coupling Mechanisms
===================

.. currentmodule:: brainmass

Coupling mechanisms connect neural mass models across brain regions, enabling the simulation of
large-scale brain networks. ``brainmass`` provides both class-based and functional coupling APIs.


Overview
--------

Coupling maps activity from source nodes to target nodes via a structural connectivity matrix.
Two fundamental coupling forms are provided:

1. **Diffusive Coupling**: Proportional to the difference between connected nodes
2. **Additive Coupling**: Directly sums weighted inputs from connected nodes


Coupling Types
--------------

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Coupling Type
     - Mathematical Form
     - Description
   * - Diffusive
     - :math:`k \sum_j W_{ij} (x_j - x_i)`
     - Drives nodes toward neighbors' states
   * - Additive
     - :math:`k \sum_j W_{ij} x_j`
     - Weighted sum of neighbor inputs


Mathematical Details
--------------------

**Diffusive Coupling**

For a network with :math:`N` nodes and connectivity matrix :math:`\mathbf{W}`, the diffusive
coupling input to node :math:`i` is:

.. math::

   C_i = k \sum_{j=1}^{N} W_{ij} (x_j - x_i)

This can be rewritten using the graph Laplacian :math:`\mathbf{L}`:

.. math::

   \mathbf{C} = -k \mathbf{L} \mathbf{x}

where :math:`L_{ij} = \delta_{ij} \sum_k W_{ik} - W_{ij}`.

**Additive Coupling**

The additive coupling input is:

.. math::

   C_i = k \sum_{j=1}^{N} W_{ij} x_j = k (\mathbf{W}^T \mathbf{x})_i


Connectivity Matrix Conventions
--------------------------------

**Shape and Indexing:**
- Connectivity matrix ``W`` has shape ``(N_out, N_in)`` or ``(N_regions, N_regions)``
- ``W[i, j]`` represents the connection strength from source node ``j`` to target node ``i``
- For symmetric networks (undirected graphs), ``W`` is symmetric
- Row sums ``W[i, :].sum()`` give the total input weights to node ``i``

**Common Structures:**
- **Structural connectivity**: From DTI tractography (weighted by fiber density)
- **Functional connectivity**: From fMRI correlations (correlation matrices)
- **Random networks**: Erdős-Rényi, scale-free, small-world topologies


API Reference
-------------

Class-Based Coupling
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   DiffusiveCoupling
   AdditiveCoupling


**Example: Diffusive Coupling**

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   N = 10  # number of regions

   # Create structural connectivity
   W = jnp.ones((N, N)) * 0.1  # uniform weak coupling
   W = W.at[jnp.diag_indices(N)].set(0.)  # no self-connections

   # Create coupling module
   coupling = brainmass.DiffusiveCoupling(
       conn=W,
       k=0.5,  # global coupling strength
   )
   coupling.init_all_states()

   # Create node dynamics
   nodes = brainmass.HopfOscillator(in_size=N, omega=10 * u.Hz)
   nodes.init_all_states()

   # Simulation loop
   def step(i):
       local_activity = nodes.update()
       coupled_input = coupling(local_activity, local_activity)
       # Add coupling to node state
       nodes.x.value += coupled_input
       return local_activity

   outputs = brainstate.transform.for_loop(step, jnp.arange(1000))


Functional Coupling
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   diffusive_coupling
   additive_coupling


Functional APIs provide stateless coupling for imperative use:

.. code-block:: python

   # Functional diffusive coupling
   coupled_input = brainmass.diffusive_coupling(
       source=source_activity,  # shape (..., N)
       target=target_activity,  # shape (..., N)
       conn=W,                  # shape (N, N)
       k=0.5,
   )

   # Functional additive coupling
   coupled_input = brainmass.additive_coupling(
       source=source_activity,
       conn=W,
       k=0.5,
   )


Connectivity Utilities
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:

   laplacian_connectivity
   LaplacianConnParam


**Laplacian Matrix:**

The graph Laplacian is commonly used in diffusive coupling:

.. code-block:: python

   W = ...  # connectivity matrix

   # Compute unnormalized Laplacian
   L = brainmass.laplacian_connectivity(W, normalize=False)
   # L[i,i] = sum_j W[i,j], L[i,j] = -W[i,j] for i≠j

   # Compute normalized Laplacian (symmetric normalization)
   L_norm = brainmass.laplacian_connectivity(W, normalize=True)


Usage Patterns
--------------

Basic Network Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   # Parameters
   N_regions = 90
   coupling_strength = 0.1

   # Load structural connectivity (example: random)
   rng = jax.random.PRNGKey(0)
   W = jax.random.uniform(rng, (N_regions, N_regions)) * 0.1
   W = W.at[jnp.diag_indices(N_regions)].set(0.)

   # Create network components
   nodes = brainmass.WilsonCowanModel(in_size=N_regions)
   coupling = brainmass.DiffusiveCoupling(conn=W, k=coupling_strength)

   # Add noise
   nodes.noise_E = brainmass.OUProcess(
       in_size=N_regions,
       sigma=0.5 * u.Hz,
       tau=20. * u.ms
   )

   # Initialize
   nodes.init_all_states()
   coupling.init_all_states()

   # Simulate
   def network_step(i):
       rE = nodes.rE.value
       rI = nodes.rI.value

       # Update nodes with coupling
       coupling_E = coupling(rE, rE)
       output = nodes.update(rE_inp=coupling_E, rI_inp=0.)

       return output

   time_series = brainstate.transform.for_loop(
       network_step,
       jnp.arange(5000)
   )


Heterogeneous Coupling
^^^^^^^^^^^^^^^^^^^^^^^

Different regions can have different coupling parameters:

.. code-block:: python

   # Region-specific coupling strengths
   k_per_region = jnp.array([0.1, 0.2, 0.3, ...])  # shape (N,)

   # Apply in coupling
   coupled = brainmass.diffusive_coupling(source, target, conn=W, k=1.0)
   coupled = coupled * k_per_region[:, None]  # scale per-region


Multi-Modal Coupling
^^^^^^^^^^^^^^^^^^^^

Combine multiple coupling mechanisms:

.. code-block:: python

   # Diffusive coupling on excitatory population
   coupling_E = brainmass.DiffusiveCoupling(conn=W_excitatory, k=0.2)

   # Additive coupling on inhibitory population
   coupling_I = brainmass.AdditiveCoupling(conn=W_inhibitory, k=0.1)

   def step(i):
       rE = nodes.rE.value
       rI = nodes.rI.value

       coupled_E = coupling_E(rE, rE)
       coupled_I = coupling_I(rI)

       nodes.update(rE_inp=coupled_E, rI_inp=coupled_I)


Time-Delayed Coupling
^^^^^^^^^^^^^^^^^^^^^^

For long-range connections with transmission delays:

.. code-block:: python

   # Simple delay implementation with circular buffer
   delay_steps = 5  # time steps
   history_buffer = jnp.zeros((delay_steps, N_regions))

   def step_with_delay(i, buffer):
       current_activity = nodes.update()

       # Get delayed activity
       delayed_activity = buffer[0]  # oldest in buffer

       # Apply coupling with delay
       coupled = coupling(delayed_activity, current_activity)
       nodes.x.value += coupled

       # Update buffer (shift and append)
       buffer = jnp.roll(buffer, shift=-1, axis=0)
       buffer = buffer.at[-1].set(current_activity)

       return current_activity, buffer


Unit Handling
-------------

Coupling respects ``brainunit`` units:

.. code-block:: python

   # Activity in Hz
   activity = jnp.array([10., 20., 30.]) * u.Hz

   # Coupling returns Hz
   coupled = brainmass.diffusive_coupling(
       source=activity,
       target=activity,
       conn=W,
       k=0.5,  # unitless coupling strength
   )
   # coupled has units Hz


Performance Tips
----------------

1. **Pre-compute Laplacian**: For diffusive coupling, pre-compute the Laplacian if ``conn`` is static:

   .. code-block:: python

      L = brainmass.laplacian_connectivity(W, normalize=False)
      coupled = -k * (L @ source_activity)

2. **Sparse Connectivity**: For large sparse networks, consider sparse matrix operations (JAX BCOO)

3. **Batched Simulations**: All coupling functions support batched inputs:

   .. code-block:: python

      # Batch of 32 simulations
      source = jnp.zeros((32, N_regions)) * u.Hz
      coupled = coupling(source, source)  # shape (32, N_regions)


Common Issues
-------------

**Sign Convention:**
- Diffusive coupling typically has ``k > 0``, producing ``C_i = k * sum(x_j - x_i)``
- Negative ``k`` inverts the coupling (anti-phase synchronization)

**Normalization:**
- Connectivity matrices can be normalized by row sums, column sums, or total sum
- Normalization affects the effective coupling strength

**Self-Connections:**
- Diagonal elements ``W[i, i]`` represent self-connections
- Often set to zero to avoid redundant self-coupling


See Also
--------

- :doc:`models` - Neural mass models to couple
- :doc:`../tutorials/adding_coupling` - Tutorial on network coupling
- :doc:`../tutorials/building_networks` - Multi-region network tutorial
- :doc:`../examples/index` - Example notebooks with coupling
