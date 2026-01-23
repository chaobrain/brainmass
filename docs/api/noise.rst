Noise Processes
================

.. currentmodule:: brainmass

Noise processes generate unit-safe stochastic drive that can be added to neural mass models
to capture the effects of unmodeled microscopic fluctuations, external inputs, or measurement noise.


Overview
--------

``brainmass`` provides several types of stochastic processes:

- **Stateless Noise**: Generates independent samples at each call (no internal state)
- **Stateful Noise**: Maintains internal state that evolves over time
- **Colored Noise**: Exhibits specific spectral properties (1/f^β noise)


Noise Comparison
----------------

.. list-table::
   :widths: 20 15 20 45
   :header-rows: 1

   * - Noise Type
     - Stateful
     - Spectral Property
     - Description
   * - :class:`GaussianNoise` / :class:`WhiteNoise`
     - No
     - Flat spectrum
     - Independent Gaussian samples, infinite bandwidth
   * - :class:`OUProcess`
     - Yes
     - Low-pass filtered
     - Ornstein-Uhlenbeck process with exponential autocorrelation
   * - :class:`BrownianNoise`
     - Yes
     - 1/f² spectrum
     - Integrated white noise, random walk
   * - :class:`ColoredNoise`
     - Yes
     - 1/f^β spectrum
     - Configurable spectral exponent β
   * - :class:`PinkNoise`
     - Yes
     - 1/f spectrum
     - β=1, commonly observed in neural activity
   * - :class:`BlueNoise`
     - Yes
     - f spectrum
     - β=-1, high-frequency emphasis
   * - :class:`VioletNoise`
     - Yes
     - f² spectrum
     - β=-2, derivative of white noise


Common Usage Patterns
----------------------

**Attaching Noise to Models**

The most common pattern is to attach noise directly to model attributes:

.. code-block:: python

   import brainmass
   import brainunit as u

   # Create model
   model = brainmass.WilsonCowanModel(in_size=10)

   # Attach noise to specific populations
   model.noise_E = brainmass.OUProcess(
       in_size=10,
       sigma=0.5 * u.Hz,  # noise amplitude
       tau=20. * u.ms,     # correlation time
   )
   model.noise_I = brainmass.OUProcess(
       in_size=10,
       sigma=0.3 * u.Hz,
       tau=30. * u.ms,
   )

   # Initialize (noise states are initialized automatically)
   model.init_all_states()

   # Noise is added internally during update()
   model.update(rE_inp=0.1, rI_inp=0.05)


**Direct Noise Generation**

For custom use cases, generate noise samples directly:

.. code-block:: python

   # Create noise source
   noise = brainmass.OUProcess(in_size=(100,), sigma=1.0 * u.Hz, tau=50. * u.ms)
   noise.init_all_states()

   # Generate samples
   import jax.numpy as jnp
   import brainstate

   samples = brainstate.transform.for_loop(
       lambda i: noise.update(),
       jnp.arange(10000)
   )


API Reference
-------------

Stateless Noise
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   GaussianNoise
   WhiteNoise


:class:`GaussianNoise` and :class:`WhiteNoise` are aliases that generate i.i.d. Gaussian samples
at each call. They have no internal state and produce infinite-bandwidth noise.

**Example:**

.. code-block:: python

   white = brainmass.WhiteNoise(in_size=10, mean=0., sigma=1.0 * u.Hz)

   # No need to initialize state (stateless)
   sample = white.update()  # shape (10,) with units Hz


Ornstein-Uhlenbeck Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   OUProcess


The Ornstein-Uhlenbeck (OU) process is a stateful, mean-reverting stochastic process defined by:

.. math::

   \tau \frac{dx}{dt} = -(x - \mu) + \sigma \sqrt{2\tau} \xi(t)

where :math:`\mu` is the mean, :math:`\tau` is the correlation time, :math:`\sigma` is the
noise amplitude, and :math:`\xi(t)` is white noise.

**Properties:**
- Exponential autocorrelation: :math:`\langle x(t)x(t+\Delta t) \rangle \propto e^{-\Delta t / \tau}`
- Stationary variance: :math:`\sigma^2`
- Low-pass filtered white noise with cutoff frequency :math:`f_c \sim 1/(2\pi\tau)`

**Example:**

.. code-block:: python

   ou = brainmass.OUProcess(
       in_size=50,
       mean=0.0,           # mean value
       sigma=0.5 * u.Hz,   # noise amplitude (stationary std)
       tau=20. * u.ms,     # correlation time
   )
   ou.init_all_states()

   # Generate correlated samples
   sample = ou.update()  # shape (50,) with exponential temporal correlation


Brownian Motion
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BrownianNoise


Brownian noise (also called red noise) is integrated white noise, producing a random walk:

.. math::

   x(t+dt) = x(t) + \sigma \sqrt{dt} \xi(t)

**Properties:**
- 1/f² power spectrum
- Variance grows linearly with time
- No mean reversion

**Example:**

.. code-block:: python

   brownian = brainmass.BrownianNoise(
       in_size=10,
       sigma=0.1 * u.Hz,  # diffusion coefficient
   )
   brownian.init_all_states()

   sample = brownian.update()  # random walk step


Colored Noise
^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   ColoredNoise
   PinkNoise
   BlueNoise
   VioletNoise


Colored noise processes exhibit specific power spectral densities of the form :math:`S(f) \propto 1/f^\beta`:

- **Pink noise** (:math:`\beta=1`): Commonly observed in neural activity and natural systems
- **Blue noise** (:math:`\beta=-1`): Emphasizes high frequencies
- **Violet noise** (:math:`\beta=-2`): Derivative of white noise

:class:`ColoredNoise` allows arbitrary spectral exponent β, while :class:`PinkNoise`,
:class:`BlueNoise`, and :class:`VioletNoise` are convenience classes with preset exponents.

**Example:**

.. code-block:: python

   # Pink noise (1/f spectrum)
   pink = brainmass.PinkNoise(
       in_size=100,
       sigma=1.0 * u.Hz,
   )
   pink.init_all_states()

   # Custom colored noise
   colored = brainmass.ColoredNoise(
       in_size=100,
       sigma=1.0 * u.Hz,
       beta=0.5,  # custom spectral exponent
   )
   colored.init_all_states()


Unit Handling
-------------

All noise processes are unit-aware. The ``sigma`` parameter determines the units of the output:

.. code-block:: python

   # Noise in Hz (for firing rates)
   noise_rate = brainmass.OUProcess(in_size=10, sigma=1.0 * u.Hz, tau=20. * u.ms)

   # Noise in mV (for membrane potentials)
   noise_voltage = brainmass.OUProcess(in_size=10, sigma=5.0 * u.mV, tau=10. * u.ms)

   # Unitless noise
   noise_dimensionless = brainmass.OUProcess(in_size=10, sigma=0.1, tau=15. * u.ms)


Performance Considerations
--------------------------

**Stateless vs Stateful:**
- Stateless noise (Gaussian/White) is faster as it has no state updates
- Use stateless noise when temporal correlations are not important
- Stateful noise (OU, Brownian, Colored) is necessary for realistic temporal structure

**Batch Simulations:**
All noise processes support batched simulations. Initialize with ``batch_size``:

.. code-block:: python

   noise = brainmass.OUProcess(in_size=10, sigma=0.5 * u.Hz, tau=20. * u.ms)
   noise.init_all_states(batch_size=32)  # 32 parallel simulations

   sample = noise.update()  # shape (32, 10)


Common Patterns
---------------

**Different Noise for Different Populations**

.. code-block:: python

   model = brainmass.WilsonCowanModel(in_size=10)

   # Stronger noise for excitatory population
   model.noise_E = brainmass.OUProcess(in_size=10, sigma=0.8 * u.Hz, tau=20. * u.ms)

   # Weaker noise for inhibitory population
   model.noise_I = brainmass.OUProcess(in_size=10, sigma=0.3 * u.Hz, tau=30. * u.ms)


**External Driving Signal**

.. code-block:: python

   # Create driving signal with pink noise (naturalistic)
   drive = brainmass.PinkNoise(in_size=1, sigma=0.5 * u.Hz)
   drive.init_all_states()

   model = brainmass.HopfOscillator(in_size=1)
   model.init_all_states()

   # Apply as external input
   def step(i):
       external_input = drive.update()
       return model.update(inp=external_input)


**Heterogeneous Noise Parameters**

For region-specific noise amplitudes:

.. code-block:: python

   import jax.numpy as jnp

   # Different sigma for each region
   sigmas = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5]) * u.Hz

   # Create noise with heterogeneous parameters
   # (Note: current API may require manual implementation for heterogeneous params)
   noise = brainmass.OUProcess(in_size=5, sigma=sigmas, tau=20. * u.ms)


See Also
--------

- :doc:`models` - Attaching noise to neural mass models
- :doc:`../tutorials/quickstart` - Basic noise usage examples
- :doc:`../examples/index` - Gallery with noise examples
