Adding Coupling
===============

This tutorial focuses on coupling mechanisms for connecting brain regions in networks.

See :doc:`building_networks` for general network setup. This guide covers coupling details.


Coupling Types
--------------

Diffusive Coupling
^^^^^^^^^^^^^^^^^^

Drives nodes toward their neighbors' states:

.. math::

   C_i = k \sum_j W_{ij} (x_j - x_i)

.. code-block:: python

   import brainmass
   import jax.numpy as jnp

   W = jnp.ones((10, 10)) * 0.1  # connectivity matrix
   W = W.at[jnp.diag_indices(10)].set(0.)

   coupling = brainmass.DiffusiveCoupling(conn=W, k=0.5)
   coupling.init_all_states()

   # Use in network
   coupled_input = coupling(source_activity, target_activity)


Additive Coupling
^^^^^^^^^^^^^^^^^

Weighted sum of neighbor inputs:

.. math::

   C_i = k \sum_j W_{ij} x_j

.. code-block:: python

   coupling = brainmass.AdditiveCoupling(conn=W, k=0.3)
   coupled_input = coupling(source_activity)


When to Use Each
^^^^^^^^^^^^^^^^

- **Diffusive**: Most common for brain networks; models synchronization
- **Additive**: Direct input summation; simpler but less realistic


Coupling Strength
-----------------

The global coupling parameter ``k`` controls network integration:

.. code-block:: python

   # Weak coupling: independent regions
   coupling_weak = brainmass.DiffusiveCoupling(conn=W, k=0.01)

   # Moderate coupling: balanced
   coupling_mod = brainmass.DiffusiveCoupling(conn=W, k=0.2)

   # Strong coupling: synchronized
   coupling_strong = brainmass.DiffusiveCoupling(conn=W, k=1.0)


**Finding optimal k:**

- Start with k ~ 0.1
- Increase until network shows desired synchronization
- Too high → all regions synchronized (unrealistic)
- Too low → no functional connectivity


Multi-Modal Coupling
---------------------

Couple different state variables:

.. code-block:: python

   import brainmass
   import brainstate

   # Separate coupling for E and I populations
   coupling_E = brainmass.DiffusiveCoupling(conn=W_E, k=0.2)
   coupling_I = brainmass.DiffusiveCoupling(conn=W_I, k=0.1)

   def multi_modal_step(i):
       rE = nodes.rE.value
       rI = nodes.rI.value

       coupled_E = coupling_E(rE, rE)
       coupled_I = coupling_I(rI, rI)

       nodes.update(rE_inp=coupled_E, rI_inp=coupled_I)


Laplacian Coupling
------------------

Use graph Laplacian for diffusive coupling:

.. code-block:: python

   # Compute Laplacian
   L = brainmass.laplacian_connectivity(W, normalize=False)
   # L[i,i] = sum_j W[i,j], L[i,j] = -W[i,j] for i≠j

   # Diffusive coupling via Laplacian
   coupled = -k * (L @ activity)


Normalized Laplacian:

.. code-block:: python

   L_norm = brainmass.laplacian_connectivity(W, normalize=True)
   # Symmetric normalization: D^(-1/2) L D^(-1/2)


Time-Delayed Coupling
----------------------

Implement axonal transmission delays:

.. code-block:: python

   import jax.numpy as jnp

   max_delay = 10  # time steps
   N_regions = 90

   # Circular buffer for history
   history_buffer = jnp.zeros((max_delay, N_regions))

   def step_with_delay(i, buffer):
       # Current activity
       current = nodes.update()

       # Apply coupling with oldest buffer value (max delay)
       delayed_activity = buffer[0]
       coupled = coupling(delayed_activity, current)
       nodes.x.value += coupled

       # Update buffer
       buffer = jnp.roll(buffer, shift=-1, axis=0)
       buffer = buffer.at[-1].set(current)

       return current, buffer

   # Run with scan for stateful buffer
   def scan_fn(buffer, i):
       output, new_buffer = step_with_delay(i, buffer)
       return new_buffer, output

   final_buffer, outputs = jax.lax.scan(
       scan_fn,
       history_buffer,
       jnp.arange(1000)
   )


Custom Coupling Functions
--------------------------

Implement custom coupling rules:

.. code-block:: python

   def nonlinear_coupling(source, target, conn, k, threshold=0.5):
       \"\"\"Only couple if source activity exceeds threshold\"\"\"
       active_source = jnp.where(source > threshold, source, 0.)
       return k * (conn.T @ active_source)

   # Use in network
   def custom_network_step(i):
       activity = nodes.x.value
       coupled = nonlinear_coupling(activity, activity, W, k=0.2, threshold=0.3)
       nodes.update()
       nodes.x.value += coupled


Directional Coupling
--------------------

Different coupling for different directions:

.. code-block:: python

   # Asymmetric connectivity
   W_forward = jnp.load('feedforward_connections.npy')
   W_backward = jnp.load('feedback_connections.npy')

   coupling_ff = brainmass.AdditiveCoupling(conn=W_forward, k=0.3)
   coupling_fb = brainmass.AdditiveCoupling(conn=W_backward, k=0.1)

   def directional_step(i):
       activity = nodes.x.value

       # Feedforward coupling
       ff_input = coupling_ff(activity)

       # Feedback coupling (different strength)
       fb_input = coupling_fb(activity)

       # Combine
       total_coupling = ff_input + fb_input
       nodes.update()
       nodes.x.value += total_coupling


Optimization and Performance
-----------------------------

JIT Compilation
^^^^^^^^^^^^^^^

.. code-block:: python

   import jax

   @jax.jit
   def fast_network_step(state_x, state_y):
       # Move coupling inside JIT for speed
       coupled = coupling(state_x, state_x)
       # ... update logic ...
       return new_state_x, new_state_y


Sparse Connectivity
^^^^^^^^^^^^^^^^^^^

For large sparse networks, consider sparse representations:

.. code-block:: python

   # Dense: (N, N) array
   # Sparse: Use JAX BCOO (experimental)
   # Or threshold small weights
   W_sparse = jnp.where(W > 0.01, W, 0.)


Best Practices
--------------

1. **Start Simple**: Test with diffusive coupling before trying custom coupling
2. **Normalize SC**: Divide by max or row/column sums to prevent instability
3. **Tune k Systematically**: Scan coupling strengths to find regime
4. **Monitor FC**: Compare simulated functional connectivity to empirical data
5. **Check Units**: Ensure coupling input has correct units for the model


Troubleshooting
---------------

**Network Explodes:**

- Reduce coupling strength k
- Normalize connectivity matrix
- Add damping/noise

**No Synchronization:**

- Increase k
- Check connectivity topology (any isolated nodes?)
- Ensure sufficient simulation time

**Unphysical Dynamics:**

- Check coupling sign (diffusive should drive toward average)
- Verify connectivity matrix orientation (row vs column convention)


Next Steps
----------

- :doc:`forward_modeling` - Map coupled network to neuroimaging signals
- :doc:`parameter_fitting` - Optimize coupling parameters
- :doc:`../api/coupling` - Full coupling API reference


See Also
--------

- :doc:`building_networks` - Network construction
- :doc:`../examples/index` - Coupling examples
