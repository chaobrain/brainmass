Forward Models
==============

.. currentmodule:: brainmass

Forward models map region-level neural activity to sensor-level observations, bridging the gap
between neural mass models and empirical neuroimaging data (fMRI BOLD, EEG, MEG).


Overview
--------

``brainmass`` provides forward models for the three major neuroimaging modalities:

1. **BOLD Signal**: Hemodynamic response mapping neural activity to fMRI BOLD signal
2. **EEG**: Electric field mapping via lead-field matrices (scalp electrodes)
3. **MEG**: Magnetic field mapping via lead-field matrices (magnetometers/gradiometers)


Forward Model Comparison
-------------------------

.. list-table::
   :widths: 20 25 25 30
   :header-rows: 1

   * - Modality
     - Input
     - Output
     - Temporal Resolution
   * - :class:`BOLDSignal`
     - Neural activity proxy (firing rate, synaptic activity)
     - BOLD signal (%)
     - ~1-2 seconds (hemodynamic lag)
   * - :class:`EEGLeadFieldModel`
     - Dipole moments (nA·m) or membrane potentials
     - Scalp potentials (μV)
     - Milliseconds
   * - :class:`MEGLeadFieldModel`
     - Dipole moments (nA·m) or membrane potentials
     - Magnetic field (fT)
     - Milliseconds


API Reference
-------------

BOLD Hemodynamics
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   BOLDSignal


The :class:`BOLDSignal` class implements the Balloon-Windkessel hemodynamic model
(Friston et al., 2003), mapping neural activity to BOLD percentage signal change.

**Model Description:**

The model consists of four coupled differential equations describing:
- Vasodilatory signal (:math:`s`)
- Blood flow (:math:`f`)
- Blood volume (:math:`v`)
- Deoxyhemoglobin content (:math:`q`)

The BOLD signal is then computed from volume and deoxyhemoglobin:

.. math::

   \text{BOLD}(t) = V_0 \left[ k_1(1-q) + k_2(1-q/v) + k_3(1-v) \right]

**Example Usage:**

.. code-block:: python

   import brainmass
   import brainunit as u
   import brainstate
   import jax.numpy as jnp

   N_regions = 10

   # Create neural mass model
   nmm = brainmass.WilsonCowanModel(in_size=N_regions)
   nmm.init_all_states()

   # Create BOLD forward model
   bold_model = brainmass.BOLDSignal(in_size=N_regions)
   bold_model.init_all_states()

   # Simulate neural activity
   def sim_step(i):
       rE = nmm.update(rE_inp=0.5, rI_inp=0.2)
       return rE

   neural_activity = brainstate.transform.for_loop(sim_step, jnp.arange(10000))

   # Map to BOLD signal
   def bold_step(z):
       bold_model.update(z=z)
       return bold_model.bold()

   bold_signal = brainstate.transform.for_loop(bold_step, neural_activity)

   # bold_signal has shape (10000, N_regions) with BOLD units (%)


**Parameters:**

Key hemodynamic parameters (see class documentation for full list):
- ``tau_s``: Signal decay time constant (~0.8 s)
- ``tau_f``: Autoregulation time constant (~0.4 s)
- ``alpha``: Grubb's exponent (~0.32)
- ``E_0``: Oxygen extraction fraction (~0.4)
- ``V_0``: Resting blood volume fraction (~0.02)

**Downsampling:**

BOLD signals are typically acquired at TR ~ 1-2 seconds. Downsample high-frequency neural activity:

.. code-block:: python

   dt_neural = 1 * u.ms  # neural simulation time step
   TR = 2 * u.s          # fMRI repetition time

   downsample_factor = int(TR / dt_neural)  # e.g., 2000

   # Downsample BOLD signal
   bold_downsampled = bold_signal[::downsample_factor]


EEG/MEG Lead-Field Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   LeadFieldModel
   EEGLeadFieldModel
   MEGLeadFieldModel


Lead-field models perform linear mapping from source dipoles to sensor measurements:

.. math::

   \mathbf{y}(t) = \mathbf{L} \mathbf{s}(t)

where:
- :math:`\mathbf{y}` is sensor data (M sensors)
- :math:`\mathbf{L}` is the lead-field matrix (M × R)
- :math:`\mathbf{s}` is source dipole moments (R regions)

**Lead-Field Matrix:**

The lead-field :math:`\mathbf{L}` encodes the physics of field propagation:
- **EEG**: Volume conduction through scalp, skull, CSF, brain
- **MEG**: Magnetic field from current dipoles (Biot-Savart law)

Lead-fields are typically computed from anatomical head models using software like:
- **FieldTrip** (MATLAB)
- **MNE-Python**
- **Brainstorm**
- **SimNIBS**

**Example: EEG Lead-Field**

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u

   N_regions = 90
   N_sensors = 64

   # Load or create lead-field matrix
   # Shape: (N_regions, N_sensors)
   # Units: V / (nA·m) for EEG
   L_eeg = jnp.load('leadfield_eeg.npy') * (u.volt / (u.nA * u.meter))

   # Create lead-field model
   eeg_model = brainmass.EEGLeadFieldModel(
       in_size=N_regions,
       out_size=N_sensors,
       L=L_eeg,
   )

   # Option 1: Pass dipole moments directly
   dipole_moments = jnp.randn(N_regions) * u.nA * u.meter
   eeg_signal = eeg_model(dipole_moments)  # shape (N_sensors,), units: V

   # Option 2: Convert membrane potentials to dipoles
   # Provide a scaling factor (nA·m / mV)
   eeg_model_with_scale = brainmass.EEGLeadFieldModel(
       in_size=N_regions,
       out_size=N_sensors,
       L=L_eeg,
       dipole_scale=1.5 * u.nA * u.meter / u.mV,  # biophysical conversion
   )

   membrane_potentials = nmm.V.value  # in mV
   eeg_signal = eeg_model_with_scale(membrane_potentials)


**Example: MEG Lead-Field**

.. code-block:: python

   # MEG lead-field units: T / (nA·m) or fT / (nA·m)
   L_meg = jnp.load('leadfield_meg.npy') * (u.fT / (u.nA * u.meter))

   meg_model = brainmass.MEGLeadFieldModel(
       in_size=N_regions,
       out_size=306,  # e.g., Neuromag system
       L=L_meg,
   )

   dipole_moments = ...
   meg_signal = meg_model(dipole_moments)  # shape (306,), units: fT


Complete Workflow Examples
---------------------------

fMRI BOLD Simulation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   N = 90  # AAL90 atlas
   T = 600  # 10 minutes at 1 ms resolution

   # 1. Create neural mass model
   nodes = brainmass.HopfOscillator(in_size=N, omega=10 * u.Hz, a=0.1)

   # 2. Add coupling
   W = jnp.load('structural_connectivity.npy')  # DTI tractography
   coupling = brainmass.DiffusiveCoupling(conn=W, k=0.2)

   # 3. Create BOLD forward model
   bold = brainmass.BOLDSignal(in_size=N)

   # 4. Initialize
   nodes.init_all_states()
   coupling.init_all_states()
   bold.init_all_states()

   # 5. Simulate neural activity
   neural_ts = []
   for i in range(T):
       x = nodes.x.value
       coupled_input = coupling(x, x)
       output = nodes.update()
       nodes.x.value += coupled_input
       neural_ts.append(output)

   neural_ts = jnp.stack(neural_ts)

   # 6. Generate BOLD signal
   bold_ts = []
   for z in neural_ts:
       bold.update(z=z)
       bold_ts.append(bold.bold())

   bold_ts = jnp.stack(bold_ts)  # shape (T, N)

   # 7. Downsample to TR = 2s
   bold_downsampled = bold_ts[::2000]  # assuming 1ms time step


EEG/MEG Source Space Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   N_sources = 90
   N_sensors_eeg = 64
   N_sensors_meg = 306

   # 1. Load lead-field matrices
   L_eeg = jnp.load('L_eeg.npy') * (u.volt / (u.nA * u.meter))
   L_meg = jnp.load('L_meg.npy') * (u.fT / (u.nA * u.meter))

   # 2. Create neural mass model (Jansen-Rit for EEG)
   nmm = brainmass.JansenRitModel(in_size=N_sources)
   nmm.init_all_states()

   # 3. Create forward models
   eeg_fwd = brainmass.EEGLeadFieldModel(
       in_size=N_sources,
       out_size=N_sensors_eeg,
       L=L_eeg,
       dipole_scale=2.0 * u.nA * u.meter / u.mV,  # pyramidal population scale
   )

   meg_fwd = brainmass.MEGLeadFieldModel(
       in_size=N_sources,
       out_size=N_sensors_meg,
       L=L_meg,
       dipole_scale=2.0 * u.nA * u.meter / u.mV,
   )

   # 4. Simulate
   def step(i):
       # Jansen-Rit output is pyramidal membrane potential (EEG proxy)
       v_pyramid = nmm.update(p=220.)
       return v_pyramid

   source_activity = brainstate.transform.for_loop(step, jnp.arange(10000))

   # 5. Project to sensors
   eeg_data = jax.vmap(eeg_fwd)(source_activity)  # shape (10000, 64)
   meg_data = jax.vmap(meg_fwd)(source_activity)  # shape (10000, 306)


Unit Handling
-------------

Forward models are fully unit-aware:

**BOLD:**
- Input: Neural activity (usually Hz or dimensionless)
- Output: BOLD percentage signal change (dimensionless or %)

**EEG:**
- Input: Dipole moments (nA·m) or membrane potentials (mV) with scale
- Lead-field: V / (nA·m)
- Output: Scalp potentials (V or μV)

**MEG:**
- Input: Dipole moments (nA·m) or membrane potentials (mV) with scale
- Lead-field: T / (nA·m) or fT / (nA·m)
- Output: Magnetic field (T or fT)


Performance Considerations
---------------------------

**Batched Forward Models:**

All forward models support batched inputs via ``vmap``:

.. code-block:: python

   # Time series: shape (T, N_regions)
   bold_timeseries = jax.vmap(bold_model)(neural_timeseries)

   # Batched simulations: shape (batch, N_regions)
   eeg_batch = jax.vmap(eeg_model)(dipole_batch)


**Memory vs Computation:**

For long simulations, consider:
1. Compute BOLD online (update at each step) vs offline (post-process entire time series)
2. Store neural activity and compute forward model afterward (saves memory for hemodynamics state)


Common Issues
-------------

**BOLD Signal Baseline:**

The BOLD signal typically starts from zero and takes ~10-20 seconds to reach steady state.
Discard initial transient or pre-initialize hemodynamic state.

**EEG/MEG Reference:**

- EEG typically uses average reference or other montages
- MEG is reference-free but may have sensor-specific scaling

**Dipole Orientation:**

- Lead-fields can be fixed-orientation (1D) or free-orientation (3D)
- ``brainmass`` assumes 1D (radial) lead-fields (common for cortical sources)


See Also
--------

- :doc:`models` - Neural mass models as inputs to forward models
- :doc:`../tutorials/forward_modeling` - Complete forward modeling tutorial
- :doc:`../examples/applications/index` - MEG/fMRI modeling examples
