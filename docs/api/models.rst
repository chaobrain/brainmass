Neural Mass Models
==================

.. currentmodule:: brainmass

Neural mass models (NMMs) are mathematical descriptions of the average activity of large populations
of neurons. They provide a coarse-grained representation suitable for whole-brain network modeling,
capturing key dynamical features like oscillations, bistability, and excitability while remaining
computationally tractable.


Overview
--------

``brainmass`` provides two categories of neural mass models:

1. **Phenomenological Models**: Capture generic dynamical behaviors (oscillations, excitability)
   without direct physiological interpretation. Useful for studying synchronization, bifurcations,
   and other dynamical phenomena.

2. **Physiological Models**: Incorporate biophysical details like synaptic dynamics, membrane
   potentials, and ionic currents. Suitable for simulating realistic neural activity and linking
   to empirical data.


Model Comparison
----------------

The following table summarizes the key characteristics of each model:

.. list-table::
   :widths: 20 15 15 20 30
   :header-rows: 1

   * - Model
     - Category
     - Variables
     - Typical Use Cases
     - Key Features
   * - :class:`HopfStep`
     - Phenomenological
     - 2 (x, y)
     - Oscillation onset, rhythm generation
     - Supercritical Hopf bifurcation, normal form
   * - :class:`VanDerPolStep`
     - Phenomenological
     - 2 (x, y)
     - Nonlinear oscillations, relaxation
     - Self-sustained oscillations, limit cycle
   * - :class:`StuartLandauStep`
     - Phenomenological
     - 2 (x, y)
     - Oscillations with amplitude control
     - Complex amplitude, frequency control
   * - :class:`FitzHughNagumoStep`
     - Phenomenological
     - 2 (v, w)
     - Excitability, spike generation
     - Type II excitable system, fast-slow
   * - :class:`ThresholdLinearStep`
     - Phenomenological
     - 2 (rE, rI)
     - Fast dynamics, linear responses
     - Threshold activation, E-I populations
   * - :class:`KuramotoNetwork`
     - Phenomenological
     - 1 (θ)
     - Phase synchronization
     - Phase oscillators, order parameter
   * - :class:`JansenRitStep`
     - Physiological
     - 6 states
     - EEG generation, alpha rhythms
     - 3 populations (pyramidal, excitatory, inhibitory interneurons)
   * - :class:`WilsonCowanStep`
     - Physiological
     - 2 (rE, rI)
     - E-I dynamics, population rates
     - Classic two-population model, firing rates
   * - :class:`WongWangStep`
     - Physiological
     - 2 (S_E, S_I)
     - Decision making, resting state fMRI
     - Spiking network reduction, NMDA/GABA synapses
   * - :class:`MontbrioPazoRoxinStep`
     - Physiological
     - 2 (r, v)
     - Mean-field dynamics, theta neurons
     - Quadratic integrate-and-fire, exact reduction


Common API Pattern
------------------

All neural mass models follow a consistent interface:

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   # 1. Create model instance
   model = brainmass.WilsonCowanStep(
       in_size=10,  # 10 brain regions
       tau_E=10. * u.ms,
       tau_I=20. * u.ms,
       # ... other parameters
   )

   # 2. Initialize state variables
   model.init_all_states(batch_size=None)  # or batch_size=32 for batched simulations

   # 3. Run simulation
   def step(i):
       return model.update(rE_inp=0.1, rI_inp=0.05)

   outputs = brainstate.transform.for_loop(step, jnp.arange(1000))

   # 4. Access internal states
   excitatory_rate = model.rE.value  # current state
   inhibitory_rate = model.rI.value

**Key Methods:**

- ``__init__(in_size, **params)``: Initialize model with parameters
- ``init_state(batch_size=None)``: Initialize state variables before simulation
- ``update(**inputs)``: Advance dynamics by one time step, returns observable(s)
- ``.reset_state()``: Reset to initial conditions


Phenomenological Models
------------------------

Phenomenological models capture essential dynamical features without detailed biophysical mechanisms.


Oscillator Models
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HopfStep
   VanDerPolStep
   StuartLandauStep
   KuramotoNetwork


**Example: Hopf Oscillator**

The Hopf oscillator exhibits a supercritical Hopf bifurcation, transitioning from a fixed point
to limit cycle oscillations as the bifurcation parameter crosses zero:

.. code-block:: python

   hopf = brainmass.HopfStep(
       in_size=5,
       omega=2 * jnp.pi * 10 * u.Hz,  # 10 Hz oscillation
       a=0.1,  # bifurcation parameter > 0 for oscillations
   )
   hopf.init_all_states()

   # Simulate
   ts = brainstate.transform.for_loop(
       lambda i: hopf.update(),
       jnp.arange(1000)
   )


Excitable Models
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   FitzHughNagumoStep
   ThresholdLinearStep


**Example: FitzHugh-Nagumo**

The FitzHugh-Nagumo model is a two-variable simplification of the Hodgkin-Huxley model,
exhibiting excitability and spike generation:

.. code-block:: python

   fhn = brainmass.FitzHughNagumoStep(
       in_size=1,
       tau=12.5 * u.ms,
       a=0.7,
       b=0.8,
   )
   fhn.init_all_states()

   # Apply external input to trigger spike
   output = fhn.update(inp=0.5)


Physiological Models
--------------------

Physiological models incorporate biophysical details and are suitable for linking to empirical data.


Rate-Based Models
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   WilsonCowanStep
   JansenRitStep
   WongWangStep


**Example: Wilson-Cowan Model**

The Wilson-Cowan model describes the dynamics of excitatory and inhibitory populations:

.. code-block:: python

   wc = brainmass.WilsonCowanStep(
       in_size=(10,),  # 10 regions
       tau_E=10. * u.ms,
       tau_I=20. * u.ms,
       c_EE=12.0, c_EI=-4.0,
       c_IE=12.0, c_II=-3.0,
   )
   wc.init_all_states()

   # Add noise
   wc.noise_E = brainmass.OUProcess(in_size=(10,), sigma=0.3 * u.Hz, tau=20. * u.ms)
   wc.noise_I = brainmass.OUProcess(in_size=(10,), sigma=0.3 * u.Hz, tau=20. * u.ms)

   # Simulate
   ts = brainstate.transform.for_loop(
       lambda i: wc.update(rE_inp=0.5, rI_inp=0.2),
       jnp.arange(2000)
   )


**Example: Jansen-Rit Model**

The Jansen-Rit model simulates EEG-like signals from a cortical column with three neural populations:

.. code-block:: python

   jr = brainmass.JansenRitStep(in_size=1)
   jr.init_all_states()

   # Simulate and get EEG proxy
   eeg_signals = brainstate.transform.for_loop(
       lambda i: jr.update(p=220.),  # external input
       jnp.arange(5000)
   )

   # The output represents pyramidal membrane potential (EEG proxy)


Spiking Network Reductions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   MontbrioPazoRoxinStep


**Example: QIF Model (Montbrio-Pazo-Roxin)**

The QIF model is an exact mean-field reduction of networks of quadratic integrate-and-fire neurons:

.. code-block:: python

   qif = brainmass.MontbrioPazoRoxinStep(
       in_size=5,
       tau=10. * u.ms,
       v_rest=-65. * u.mV,
       v_th=-50. * u.mV,
   )
   qif.init_all_states()

   # r = population firing rate, v = mean membrane potential
   outputs = brainstate.transform.for_loop(
       lambda i: qif.update(I_ext=2.0 * u.nA),
       jnp.arange(1000)
   )


Adding Noise to Models
-----------------------

All models support adding stochastic noise through dedicated noise attributes:

.. code-block:: python

   # Create model
   model = brainmass.HopfStep(in_size=10, omega=10 * u.Hz)

   # Add Ornstein-Uhlenbeck noise
   model.noise = brainmass.OUProcess(
       in_size=10,
       sigma=0.1 * u.Hz,
       tau=50. * u.ms
   )

   model.init_all_states()

See :doc:`noise` for more details on noise processes.


Multi-Region Networks
----------------------

To create multi-region brain networks, combine models with coupling mechanisms:

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u

   N = 90  # number of regions

   # Create node dynamics
   nodes = brainmass.HopfStep(in_size=N, omega=10 * u.Hz)

   # Create coupling
   coupling = brainmass.DiffusiveCoupling(
       conn=connectivity_matrix,  # (N, N) structural connectivity
       k=0.1,  # coupling strength
   )

   # Initialize
   nodes.init_all_states()
   coupling.init_all_states()

   # Simulation loop
   def network_step(i):
       local_activity = nodes.update()
       coupled_input = coupling(local_activity)
       nodes.x.value += coupled_input  # add coupling to node state
       return local_activity

   brainstate.transform.for_loop(network_step, jnp.arange(1000))

See :doc:`coupling` for comprehensive coupling documentation.


See Also
--------

- :doc:`noise` - Stochastic processes for adding variability
- :doc:`coupling` - Coupling mechanisms for network connectivity
- :doc:`forward` - Forward models for mapping to observed signals
- :doc:`../tutorials/choosing_models` - Tutorial on selecting the right model
- :doc:`../tutorials/building_networks` - Tutorial on multi-region networks
