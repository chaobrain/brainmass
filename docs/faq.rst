FAQ and Troubleshooting
=======================

Frequently asked questions and solutions to common problems.


Installation Issues
-------------------

**Q: ImportError: cannot import 'brainstate'**

A: Ensure brainstate is installed and up to date:

.. code-block:: bash

   pip install --upgrade brainstate>=0.2.9


**Q: JAX not detecting GPU**

A: Check CUDA installation and reinstall JAX with CUDA support:

.. code-block:: bash

   pip install --upgrade jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


**Q: ModuleNotFoundError for brainunit**

A: Install or update brainunit:

.. code-block:: bash

   pip install --upgrade brainunit


Usage Questions
---------------

**Q: Which model should I use for fMRI?**

A: Use :class:`WongWangModel` or :class:`WilsonCowanModel`. They have slow synaptic dynamics suitable for BOLD timescales. See :doc:`tutorials/choosing_models`.


**Q: How do I fit parameters to my data?**

A: See :doc:`tutorials/parameter_fitting` for complete workflow. Use Nevergrad for gradient-free or JAX+Optax for gradient-based optimization.


**Q: Can I use custom connectivity matrices?**

A: Yes! Load your structural connectivity (DTI) as a numpy/JAX array and pass to :class:`DiffusiveCoupling` or :class:`AdditiveCoupling`.


**Q: How do I add noise to models?**

A: Assign a noise object to model attributes:

.. code-block:: python

   model.noise = brainmass.OUProcess(in_size=10, sigma=0.5, tau=20)


Performance Issues
------------------

**Q: Simulations are slow**

A: Solutions:

- Use JIT compilation: Wrap simulation in ``@jax.jit``
- Use simpler models (Hopf vs Jansen-Rit)
- Reduce number of time steps
- Enable GPU if available


**Q: Out of memory errors**

A: Solutions:

- Reduce batch size
- Downsample time series before analysis
- Use CPU for very large networks
- Set ``XLA_PYTHON_CLIENT_PREALLOCATE=false``


**Q: Gradients are NaN**

A: Causes and fixes:

- Clip gradients: ``jnp.clip(grads, -1, 1)``
- Reduce learning rate
- Check for division by zero
- Normalize loss function


Model Behavior
--------------

**Q: Network activity explodes**

A: Solutions:

- Reduce coupling strength ``k``
- Normalize connectivity matrix
- Add noise for stability
- Check parameter ranges


**Q: No synchronization in network**

A: Solutions:

- Increase coupling strength
- Check connectivity (any isolated nodes?)
- Run longer simulation
- Verify diffusive vs additive coupling


**Q: BOLD signal doesn't stabilize**

A: Solutions:

- Run longer simulation (>60s)
- Check hemodynamic parameters
- Discard initial transient (~20s)


Unit Errors
-----------

**Q: DimensionMismatch error**

A: Ensure compatible units:

.. code-block:: python

   # WRONG:
   time_ms = 10 * u.ms
   rate_Hz = 5 * u.Hz
   result = time_ms + rate_Hz  # Error!

   # CORRECT:
   result = time_ms * rate_Hz  # Dimensionless


**Q: Lost units after JAX operations**

A: Some JAX functions strip units. Reattach:

.. code-block:: python

   x_with_units = 10 * u.ms
   y = jnp.exp(x_with_units.magnitude)  # Convert to magnitude
   y_with_units = y * u.dimensionless   # Reattach appropriate unit


**Q: How to convert units?**

A:

.. code-block:: python

   tau_ms = 10 * u.ms
   tau_s = tau_ms.to(u.second)  # 0.01 s


Data and Modeling
-----------------

**Q: Where can I get structural connectivity data?**

A: Sources:

- Human Connectome Project
- OpenNeuro
- Lab collaborations
- Synthetic networks for testing


**Q: What atlases are commonly used?**

A: Common choices:

- AAL (90/116 regions)
- Desikan-Killiany (68 regions)
- Schaefer (100-1000 parcels)
- Destrieux (148 regions)


**Q: How to compute functional connectivity?**

A:

.. code-block:: python

   # Pearson correlation
   FC = jnp.corrcoef(bold_timeseries.T)


**Q: How to validate my model?**

A: Compare:

- Simulated vs empirical FC
- Power spectra
- Phase locking value
- Dynamic FC


Common Errors
-------------

**Error: "Hidden state not initialized"**

Fix: Call ``model.init_all_states()`` before ``model.update()``

.. code-block:: python

   model = brainmass.HopfOscillator(in_size=10)
   model.init_all_states()  # Required!
   model.update()


**Error: "Shape mismatch in coupling"**

Fix: Ensure connectivity matrix matches node dimensions:

.. code-block:: python

   N = 90
   nodes = brainmass.HopfOscillator(in_size=N)
   W = jnp.ones((N, N))  # Must be (N, N)


**Error: "Module 'brainmass' has no attribute 'X'"**

Fix: Check spelling or see :doc:`api/index` for available models.


Advanced Topics
---------------

**Q: Can I use brainmass with TensorFlow/PyTorch?**

A: No, brainmass is built on JAX. For interoperability, convert arrays manually, but most functionality requires JAX.


**Q: How to implement custom models?**

A: See :doc:`developer/creating_models` for detailed guide.


**Q: Can I use different models for different regions?**

A: Yes! Create separate model instances and manually couple them. See :doc:`tutorials/building_networks`.


**Q: How to parallelize simulations?**

A: Use JAX's ``vmap`` for batching or ``pmap`` for multi-device:

.. code-block:: python

   # Batch dimension
   model.init_all_states(batch_size=32)

   # Or vmap over parameter sets
   vmap_simulate = jax.vmap(simulate_fn, in_axes=(0, None))


Getting Help
------------

If your question isn't answered here:

1. **Search Documentation:** Use search box (top-right)
2. **Check Examples:** :doc:`examples/index`
3. **GitHub Issues:** Report bugs at `github.com/chaobrain/brainmass/issues <https://github.com/chaobrain/brainmass/issues>`_
4. **Discussions:** Ask questions at GitHub Discussions
5. **Email:** Contact maintainers for urgent issues


See Also
--------

- :doc:`tutorials/index` - Tutorials for common tasks
- :doc:`api/index` - Complete API reference
- :doc:`examples/index` - Working examples
- :doc:`developer/index` - For contributors
