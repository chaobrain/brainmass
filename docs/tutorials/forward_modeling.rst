Forward Modeling
================

Forward models map neural activity to observable neuroimaging signals (BOLD, EEG, MEG).


Overview
--------

.. code-block:: text

   Neural Mass Model → Forward Model → Neuroimaging Signal
   (hidden dynamics)   (biophysics)    (observable data)


Three main forward models:

1. **BOLD**: fMRI hemodynamic response
2. **EEG**: Electric scalp potentials
3. **MEG**: Magnetic field sensors


BOLD Signal Modeling
--------------------

Basic BOLD Workflow
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   N_regions = 90

   # 1. Create neural mass model
   nmm = brainmass.WongWangModel(in_size=N_regions)
   nmm.init_all_states()

   # 2. Create BOLD hemodynamic model
   bold = brainmass.BOLDSignal(in_size=N_regions)
   bold.init_all_states()

   # 3. Simulate neural activity
   def sim_neural(i):
       return nmm.update(S_E_ext=0.1)

   neural_activity = brainstate.transform.for_loop(
       sim_neural,
       jnp.arange(10000)  # 10 seconds at 1ms
   )

   # 4. Generate BOLD signal
   def sim_bold(z):
       bold.update(z=z)
       return bold.bold()

   bold_signal = brainstate.transform.for_loop(sim_bold, neural_activity)

   # 5. Downsample to TR
   TR_steps = 2000  # TR = 2s with dt=1ms
   bold_downsampled = bold_signal[::TR_steps]


Complete fMRI Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u
   import brainstate

   # Parameters
   N = 90
   T_seconds = 600  # 10 minutes
   dt = 1 * u.ms
   T_steps = int((T_seconds * u.second / dt).magnitude)

   # Load connectivity
   SC = jnp.load('SC_AAL90.npy')

   # Create network
   nodes = brainmass.WongWangModel(in_size=N)
   coupling = brainmass.DiffusiveCoupling(conn=SC, k=0.2)
   nodes.noise_E = brainmass.OUProcess(in_size=N, sigma=0.01*u.Hz, tau=100*u.ms)

   bold = brainmass.BOLDSignal(in_size=N)

   # Initialize
   nodes.init_all_states()
   coupling.init_all_states()
   bold.init_all_states()

   # Simulate
   neural_ts = []
   for t in range(T_steps):
       S_E = nodes.S_E.value
       coupled_input = coupling(S_E, S_E)
       output = nodes.update(S_E_ext=coupled_input)
       neural_ts.append(output)

       if t % 10000 == 0:
           print(f"{t}/{T_steps}")

   neural_ts = jnp.stack(neural_ts)

   # Generate BOLD
   bold_ts = []
   for z in neural_ts:
       bold.update(z=z)
       bold_ts.append(bold.bold())

   bold_ts = jnp.stack(bold_ts)

   # Downsample to TR=2s
   bold_fmri = bold_ts[::2000]

   # Compute FC
   FC_sim = jnp.corrcoef(bold_fmri.T)

   # Compare to empirical FC
   FC_emp = jnp.load('empirical_FC.npy')
   correlation = jnp.corrcoef(FC_sim.flatten(), FC_emp.flatten())[0, 1]
   print(f"FC correlation: {correlation:.3f}")


EEG/MEG Modeling
----------------

Lead-Field Setup
^^^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import jax.numpy as jnp
   import brainunit as u

   N_sources = 68  # cortical regions
   N_sensors = 64  # EEG electrodes

   # Load lead-field matrix from head model
   # L[i, j] = sensor i sensitivity to source j
   # Units: V / (nA·m)
   L_eeg = jnp.load('leadfield_eeg.npy') * (u.volt / (u.nA * u.meter))

   # Create EEG forward model
   eeg_model = brainmass.EEGLeadFieldModel(
       in_size=N_sources,
       out_size=N_sensors,
       L=L_eeg,
       dipole_scale=2.0 * u.nA * u.meter / u.mV,  # biophysical conversion
   )


EEG Simulation
^^^^^^^^^^^^^^

.. code-block:: python

   import brainmass
   import brainstate

   # Jansen-Rit model (generates EEG-like signals)
   nmm = brainmass.JansenRitModel(in_size=N_sources)
   nmm.init_all_states()

   # Simulate source activity
   def sim_sources(i):
       return nmm.update(p=220.)  # pyramidal membrane potential

   source_activity = brainstate.transform.for_loop(
       sim_sources,
       jnp.arange(10000)
   )

   # Project to sensors
   eeg_sensors = jax.vmap(eeg_model)(source_activity)
   # eeg_sensors shape: (10000, 64) with units V or μV


MEG Simulation
^^^^^^^^^^^^^^

.. code-block:: python

   # Load MEG lead-field
   # Units: fT / (nA·m) or T / (nA·m)
   L_meg = jnp.load('leadfield_meg.npy') * (u.fT / (u.nA * u.meter))

   meg_model = brainmass.MEGLeadFieldModel(
       in_size=N_sources,
       out_size=306,  # e.g., Neuromag system
       L=L_meg,
       dipole_scale=2.0 * u.nA * u.meter / u.mV,
   )

   # Project to MEG sensors
   meg_sensors = jax.vmap(meg_model)(source_activity)


Comparing Modalities
--------------------

Simultaneous EEG/MEG/fMRI
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use same neural activity for all modalities
   nmm = brainmass.JansenRitModel(in_size=68)
   nmm.init_all_states()

   # Forward models
   bold = brainmass.BOLDSignal(in_size=68)
   eeg_fwd = brainmass.EEGLeadFieldModel(in_size=68, out_size=64, L=L_eeg)
   meg_fwd = brainmass.MEGLeadFieldModel(in_size=68, out_size=306, L=L_meg)

   bold.init_all_states()

   # Simulate
   neural_ts = brainstate.transform.for_loop(
       lambda i: nmm.update(p=220.),
       jnp.arange(60000)  # 60 seconds
   )

   # Generate all modalities
   eeg_data = jax.vmap(eeg_fwd)(neural_ts)  # (60000, 64)
   meg_data = jax.vmap(meg_fwd)(neural_ts)  # (60000, 306)

   bold_ts = []
   for z in neural_ts:
       bold.update(z=z)
       bold_ts.append(bold.bold())
   bold_data = jnp.stack(bold_ts)[::2000]  # downsample to TR=2s


Advanced Topics
---------------

Handling Units
^^^^^^^^^^^^^^

.. code-block:: python

   # Ensure consistent units throughout pipeline

   # Neural activity in Hz
   neural_rate = nmm.rE.value  # Hz

   # BOLD input (dimensionless or Hz)
   bold.update(z=neural_rate)

   # EEG requires membrane potentials (mV) or dipoles (nA·m)
   V_pyramid = nmm.V_pyramid.value  # mV
   eeg_signal = eeg_model(V_pyramid)  # V or μV


Custom Forward Models
^^^^^^^^^^^^^^^^^^^^^

Implement custom observation functions:

.. code-block:: python

   def custom_bold_observation(neural_activity):
       \"\"\"Custom BOLD observation model\"\"\"
       # Apply nonlinear transformation
       transformed = jnp.tanh(neural_activity / 10.0)
       return transformed

   # Use in pipeline
   bold_custom = custom_bold_observation(neural_ts)


Model Validation
----------------

Comparing to Empirical Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Simulated BOLD FC
   FC_sim = jnp.corrcoef(bold_downsampled.T)

   # Empirical BOLD FC
   FC_emp = jnp.load('empirical_FC.npy')

   # Correlation
   FC_corr = jnp.corrcoef(FC_sim.flatten(), FC_emp.flatten())[0, 1]

   # Mean squared error
   FC_mse = jnp.mean((FC_sim - FC_emp) ** 2)

   print(f"FC correlation: {FC_corr:.3f}")
   print(f"FC MSE: {FC_mse:.3f}")


Power Spectral Density
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from scipy import signal

   # Compute PSD of simulated EEG
   freqs, psd = signal.welch(
       eeg_sensors[:, 0],  # channel 0
       fs=1000,  # 1 kHz sampling
       nperseg=1024
   )

   # Compare to empirical PSD
   import matplotlib.pyplot as plt
   plt.semilogy(freqs, psd, label='Simulated')
   plt.semilogy(freqs_emp, psd_emp, label='Empirical')
   plt.legend()
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('PSD')


Best Practices
--------------

1. **Match Timescales**: Use appropriate models for each modality (fast for EEG/MEG, slow for BOLD)
2. **Discard Transients**: Remove initial ~20s for BOLD to reach steady state
3. **Downsample Correctly**: Match sampling rate to modality (TR for fMRI, ms for EEG/MEG)
4. **Validate Lead-Fields**: Check that L has correct shape and units
5. **Use Realistic Parameters**: BOLD hemodynamics have well-established parameter ranges


Common Issues
-------------

**BOLD doesn't stabilize:**

- Run longer simulation (>60s)
- Check initial conditions
- Verify hemodynamic parameters

**EEG/MEG too small/large:**

- Check lead-field units and scaling
- Verify dipole_scale conversion factor
- Ensure biophysically realistic source activity

**Mismatch with empirical data:**

- Adjust coupling strength
- Check structural connectivity
- Tune model parameters (see :doc:`parameter_fitting`)


Next Steps
----------

- :doc:`parameter_fitting` - Optimize parameters to match empirical data
- :doc:`../api/forward` - Complete forward model API
- :doc:`../examples/applications/index` - Real data examples


See Also
--------

- :doc:`building_networks` - Network setup for forward modeling
- :doc:`../api/forward` - Forward model reference
