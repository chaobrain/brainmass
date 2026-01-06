Parameter Fitting
==================

This tutorial covers optimizing model parameters to match empirical data.


Overview
--------

Parameter fitting workflow:

1. Define a loss function comparing model output to data
2. Choose an optimization method
3. Run optimization to find best parameters
4. Validate fitted parameters


Loss Functions
--------------

Functional Connectivity Loss
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most common for fMRI data:

.. code-block:: python

   import jax.numpy as jnp

   def fc_loss(params, SC, FC_empirical):
       \"\"\"Loss based on functional connectivity matching\"\"\"

       # Run simulation with params
       bold_sim = simulate_network(params, SC)

       # Compute simulated FC
       FC_sim = jnp.corrcoef(bold_sim.T)

       # Compare to empirical FC
       # Option 1: Correlation
       FC_corr = jnp.corrcoef(FC_sim.flatten(), FC_empirical.flatten())[0, 1]
       loss = 1.0 - FC_corr  # minimize (1 - correlation)

       # Option 2: MSE
       # loss = jnp.mean((FC_sim - FC_empirical) ** 2)

       return loss


Time Series Loss
^^^^^^^^^^^^^^^^

For EEG/MEG or other time-domain data:

.. code-block:: python

   def timeseries_loss(params, data_empirical):
       \"\"\"Direct time series matching\"\"\"

       # Simulate
       data_sim = simulate_eeg(params)

       # MSE loss
       loss = jnp.mean((data_sim - data_empirical) ** 2)

       return loss


Power Spectrum Loss
^^^^^^^^^^^^^^^^^^^

Match frequency content:

.. code-block:: python

   from scipy import signal

   def psd_loss(params, psd_empirical, freqs_empirical):
       \"\"\"Match power spectral density\"\"\"

       # Simulate and compute PSD
       ts_sim = simulate(params)
       freqs_sim, psd_sim = signal.welch(ts_sim, fs=1000)

       # Interpolate to match empirical frequencies
       psd_sim_interp = jnp.interp(freqs_empirical, freqs_sim, psd_sim)

       # Log-space MSE (better for power spectra)
       loss = jnp.mean((jnp.log(psd_sim_interp) - jnp.log(psd_empirical)) ** 2)

       return loss


Optimization Methods
--------------------

Gradient-Free (Nevergrad)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Best for non-differentiable objectives:

.. code-block:: python

   import nevergrad as ng
   import brainmass
   import jax.numpy as jnp

   # Load data
   SC = jnp.load('SC.npy')
   FC_emp = jnp.load('FC_empirical.npy')

   # Define simulation function
   def simulate_network(coupling_strength):
       nodes = brainmass.WongWangModel(in_size=90)
       coupling = brainmass.DiffusiveCoupling(conn=SC, k=coupling_strength)
       bold = brainmass.BOLDSignal(in_size=90)

       nodes.init_all_states()
       coupling.init_all_states()
       bold.init_all_states()

       # Simulate (simplified)
       neural_ts = []
       for t in range(10000):
           S_E = nodes.S_E.value
           coupled = coupling(S_E, S_E)
           out = nodes.update(S_E_ext=coupled)
           neural_ts.append(out)

       neural_ts = jnp.stack(neural_ts)

       # Generate BOLD
       bold_ts = []
       for z in neural_ts:
           bold.update(z=z)
           bold_ts.append(bold.bold())

       return jnp.stack(bold_ts)[::2000]  # downsample

   # Define loss
   def objective(coupling_strength):
       bold_sim = simulate_network(coupling_strength)
       FC_sim = jnp.corrcoef(bold_sim.T)
       corr = jnp.corrcoef(FC_sim.flatten(), FC_emp.flatten())[0, 1]
       return 1.0 - corr  # minimize

   # Setup optimization
   instrum = ng.p.Scalar(init=0.2, lower=0.0, upper=1.0)
   optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=50)

   # Run optimization
   for i in range(50):
       x = optimizer.ask()
       loss = objective(x.value)
       optimizer.tell(x, loss)
       print(f"Iteration {i}: k={x.value:.3f}, loss={loss:.4f}")

   # Best parameters
   recommendation = optimizer.provide_recommendation()
   best_k = recommendation.value
   print(f"Best coupling strength: {best_k:.3f}")


Gradient-Based (JAX + Optax)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For differentiable models:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import optax
   import brainmass

   # Define differentiable simulation
   @jax.jit
   def simulate_and_loss(k, SC, FC_emp):
       # Simplified differentiable simulation
       # (full network simulation needs careful state management)

       # ... simulation code ...

       FC_sim = jnp.corrcoef(bold_sim.T)
       loss = jnp.mean((FC_sim - FC_emp) ** 2)

       return loss

   # Setup optimizer
   learning_rate = 1e-3
   optimizer = optax.adam(learning_rate)

   # Initial parameters
   params = {'k': 0.2}
   opt_state = optimizer.init(params)

   # Training loop
   for epoch in range(100):
       # Compute gradients
       loss, grads = jax.value_and_grad(simulate_and_loss)(
           params['k'], SC, FC_emp
       )

       # Update parameters
       updates, opt_state = optimizer.update(grads, opt_state)
       params = optax.apply_updates(params, updates)

       if epoch % 10 == 0:
           print(f"Epoch {epoch}: k={params['k']:.3f}, loss={loss:.4f}")


SciPy Optimizers
^^^^^^^^^^^^^^^^

Good for moderate-dimensional problems:

.. code-block:: python

   from scipy.optimize import minimize
   import jax.numpy as jnp

   def objective(params):
       k, noise_sigma = params
       # ... simulate ...
       loss = fc_loss(k, noise_sigma, FC_emp)
       return loss

   # Initial guess
   x0 = [0.2, 0.01]

   # Optimize
   result = minimize(
       objective,
       x0,
       method='Nelder-Mead',
       options={'maxiter': 100}
   )

   best_params = result.x
   print(f"Best parameters: k={best_params[0]:.3f}, sigma={best_params[1]:.4f}")


Multi-Parameter Optimization
-----------------------------

Optimizing Multiple Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nevergrad as ng

   # Define parameter space
   instrum = ng.p.Instrumentation(
       coupling_k=ng.p.Scalar(init=0.2, lower=0.0, upper=1.0),
       noise_sigma=ng.p.Scalar(init=0.01, lower=0.0, upper=0.1),
       tau_E=ng.p.Scalar(init=10.0, lower=5.0, upper=50.0),
       tau_I=ng.p.Scalar(init=20.0, lower=10.0, upper=100.0),
   )

   optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=200)

   def multi_param_objective(coupling_k, noise_sigma, tau_E, tau_I):
       # Simulate with all parameters
       # ...
       return loss

   # Optimize
   for i in range(200):
       x = optimizer.ask()
       loss = multi_param_objective(**x.kwargs)
       optimizer.tell(x, loss)


Constrained Parameters
^^^^^^^^^^^^^^^^^^^^^^

Use :class:`ArrayParam` for constraints:

.. code-block:: python

   import brainmass
   import braintools

   # Create constrained parameter (must be positive)
   tau_param = brainmass.ArrayParam(
       data=10.0,  # initial value
       transform=braintools.SoftplusTransform(),
   )

   # Optimize in unconstrained space
   def objective(tau_unconstrained):
       tau_param.value = tau_unconstrained
       tau_actual = tau_param.data  # always positive

       # Use tau_actual in simulation
       # ...
       return loss


Best Practices
--------------

1. **Start Simple**
   - Fit one parameter at a time initially
   - Add complexity gradually

2. **Use Multiple Initializations**
   - Run optimization from different starting points
   - Avoid local minima

3. **Normalize Loss**
   - Scale loss components to similar magnitude
   - Prevents one term dominating

4. **Validate on Hold-Out Data**
   - Test fitted parameters on unseen data
   - Check for overfitting

5. **Monitor Convergence**
   - Plot loss vs iteration
   - Stop when loss plateaus

6. **Set Reasonable Bounds**
   - Use prior knowledge for parameter ranges
   - Prevents unphysical values


Common Issues
-------------

**Optimization Stuck in Local Minimum:**

- Use global optimizer (Nevergrad)
- Try multiple initializations
- Increase budget

**Loss Not Decreasing:**

- Check loss function implementation
- Verify simulation is correct
- Reduce learning rate (gradient-based)

**Parameters Unrealistic:**

- Add constraints/bounds
- Use regularization
- Check units

**Slow Optimization:**

- Reduce simulation time
- Use simpler model for initial fits
- Parallelize evaluations (Nevergrad supports this)


Complete Example
----------------

Full FC-Based Parameter Fitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainstate
   import nevergrad as ng

   # Load data
   SC = jnp.load('SC_AAL90.npy')
   FC_emp = jnp.load('FC_empirical.npy')

   # Simulation function
   def simulate_bold_fc(coupling_k, noise_sigma):
       N = 90
       T = 600000  # 10 min at 1ms

       # Create network
       nodes = brainmass.WongWangModel(in_size=N)
       coupling = brainmass.DiffusiveCoupling(conn=SC, k=coupling_k)
       nodes.noise_E = brainmass.OUProcess(
           in_size=N,
           sigma=noise_sigma,
           tau=100.0,
       )
       bold = brainmass.BOLDSignal(in_size=N)

       # Initialize
       nodes.init_all_states()
       coupling.init_all_states()
       bold.init_all_states()

       # Simulate
       neural_ts = []
       for t in range(T):
           S_E = nodes.S_E.value
           coupled = coupling(S_E, S_E)
           out = nodes.update(S_E_ext=coupled)
           neural_ts.append(out)

           if t % 10000 == 0:
               print(f"  Sim: {t}/{T}")

       neural_ts = jnp.stack(neural_ts)

       # BOLD
       bold_ts = []
       for z in neural_ts:
           bold.update(z=z)
           bold_ts.append(bold.bold())

       bold_downsampled = jnp.stack(bold_ts)[::2000]

       # FC
       FC_sim = jnp.corrcoef(bold_downsampled.T)

       return FC_sim

   # Loss function
   def objective(coupling_k, noise_sigma):
       print(f"Trying k={coupling_k:.3f}, sigma={noise_sigma:.4f}")

       FC_sim = simulate_bold_fc(coupling_k, noise_sigma)
       corr = jnp.corrcoef(FC_sim.flatten(), FC_emp.flatten())[0, 1]
       loss = 1.0 - corr

       print(f"  Loss: {loss:.4f}, FC corr: {corr:.4f}")
       return loss

   # Optimization
   instrum = ng.p.Instrumentation(
       coupling_k=ng.p.Scalar(init=0.2, lower=0.0, upper=1.0),
       noise_sigma=ng.p.Scalar(init=0.01, lower=0.0, upper=0.1),
   )

   optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=50)

   for i in range(50):
       print(f"\n=== Iteration {i+1}/50 ===")
       x = optimizer.ask()
       loss = objective(**x.kwargs)
       optimizer.tell(x, loss)

   # Best result
   best = optimizer.provide_recommendation()
   print(f"\nBest parameters:")
   print(f"  Coupling k: {best.kwargs['coupling_k']:.3f}")
   print(f"  Noise sigma: {best.kwargs['noise_sigma']:.4f}")


Next Steps
----------

- Try parameter fitting on your own data
- Explore different loss functions
- Compare optimization methods
- :doc:`../examples/advanced/index` for advanced optimization examples


See Also
--------

- :doc:`forward_modeling` - Getting observable data from models
- :doc:`../api/types` - ArrayParam for constrained optimization
- Nevergrad documentation
- Optax documentation
