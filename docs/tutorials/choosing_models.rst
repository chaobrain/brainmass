Choosing Models
===============

This guide helps you select the appropriate neural mass model for your research question.


Decision Tree
-------------

.. code-block:: text

   Need physiological realism?
   ├─ YES → Physiological Models
   │  ├─ Modeling EEG/MEG? → JansenRitModel
   │  ├─ Modeling fMRI BOLD? → WongWangModel
   │  ├─ E-I population dynamics? → WilsonCowanModel
   │  └─ Mean-field spiking? → QIFModel (MontbrioPazoRoxin)
   │
   └─ NO → Phenomenological Models
      ├─ Studying oscillations? → HopfOscillator or StuartLandauOscillator
      ├─ Studying excitability? → FitzHughNagumoModel
      ├─ Phase synchronization? → KuramotoNetwork
      └─ Fast/simple dynamics? → ThresholdLinearModel


Model Categories
----------------

Phenomenological Models
^^^^^^^^^^^^^^^^^^^^^^^

**When to use:**

- Studying generic dynamical phenomena (bifurcations, synchronization)
- Computational efficiency is critical
- Detailed biophysical mechanisms are not required

**Hopf Oscillator**

- **Best for:** Onset of oscillations, rhythm generation
- **Key feature:** Supercritical Hopf bifurcation
- **Variables:** 2 (x, y)
- **Example use:** Studying how oscillatory activity emerges

.. code-block:: python

   hopf = brainmass.HopfOscillator(
       in_size=90,
       omega=2 * jnp.pi * 10 * u.Hz,  # 10 Hz alpha rhythm
       a=0.1,  # >0 for limit cycle
   )

**FitzHugh-Nagumo**

- **Best for:** Excitable systems, spike generation
- **Key feature:** Type II excitable, fast-slow dynamics
- **Variables:** 2 (v, w)
- **Example use:** Studying transitions from rest to spiking

**Kuramoto Network**

- **Best for:** Phase synchronization in oscillator networks
- **Key feature:** Order parameter for collective synchrony
- **Variables:** 1 (θ)
- **Example use:** Studying synchronization patterns


Physiological Models
^^^^^^^^^^^^^^^^^^^^

**When to use:**

- Linking models to empirical neuroimaging data
- Biophysical realism is important
- Modeling specific neural populations

**Jansen-Rit Model**

- **Best for:** EEG signal generation, alpha rhythms
- **Key feature:** 3 neural populations (pyramidal, excitatory, inhibitory)
- **Variables:** 6 states
- **Example use:** Simulating EEG from cortical columns

.. code-block:: python

   jr = brainmass.JansenRitModel(
       in_size=68,  # number of cortical sources
   )

**Wilson-Cowan Model**

- **Best for:** Excitatory-inhibitory population dynamics
- **Key feature:** Classic E-I model, firing rate equations
- **Variables:** 2 (rE, rI)
- **Example use:** Studying E-I balance, oscillations

**Wong-Wang Model**

- **Best for:** Resting-state fMRI, decision-making tasks
- **Key feature:** Reduced spiking network with NMDA/GABA synapses
- **Variables:** 2 (S_E, S_I)
- **Example use:** Whole-brain resting-state fMRI simulations

**QIF Model (Montbrio-Pazo-Roxin)**

- **Best for:** Mean-field reduction of spiking networks
- **Key feature:** Exact reduction of QIF neuron networks
- **Variables:** 2 (r, v)
- **Example use:** Linking spiking and rate dynamics


Model Comparison
----------------

.. list-table::
   :widths: 20 15 15 20 30
   :header-rows: 1

   * - Model
     - Complexity
     - Speed
     - Realism
     - Typical Application
   * - ThresholdLinear
     - Low
     - Very Fast
     - Low
     - Fast exploratory simulations
   * - Hopf
     - Low
     - Fast
     - Low
     - Oscillation emergence
   * - FitzHugh-Nagumo
     - Low
     - Fast
     - Medium
     - Excitability, spikes
   * - Kuramoto
     - Low
     - Fast
     - Low
     - Phase synchronization
   * - Wilson-Cowan
     - Medium
     - Medium
     - Medium
     - E-I dynamics, general purpose
   * - Jansen-Rit
     - High
     - Slow
     - High
     - EEG/MEG modeling
   * - Wong-Wang
     - High
     - Slow
     - High
     - fMRI BOLD, decision making
   * - QIF
     - Medium
     - Medium
     - High
     - Mean-field spiking networks


Use Case Examples
-----------------

Resting-State fMRI Study
^^^^^^^^^^^^^^^^^^^^^^^^

**Goal:** Simulate spontaneous BOLD fluctuations matching empirical FC

**Recommended:** :class:`WongWangModel` or :class:`WilsonCowanModel`

**Why:**
- Slow synaptic dynamics (NMDA) match BOLD timescales
- Captures E-I dynamics generating realistic fluctuations
- Well-validated for resting-state simulations

.. code-block:: python

   nodes = brainmass.WongWangModel(in_size=90)  # AAL90 atlas
   bold = brainmass.BOLDSignal(in_size=90)

   # Add structural coupling from DTI
   coupling = brainmass.DiffusiveCoupling(conn=SC_matrix, k=0.2)


EEG Source Modeling
^^^^^^^^^^^^^^^^^^^

**Goal:** Simulate EEG signals from cortical sources

**Recommended:** :class:`JansenRitModel`

**Why:**

- Explicitly models pyramidal neuron membrane potentials (EEG source)
- Validated against empirical EEG spectra
- Generates realistic alpha rhythms

.. code-block:: python

   jr = brainmass.JansenRitModel(in_size=68)  # cortical parcels
   eeg_fwd = brainmass.EEGLeadFieldModel(
       in_size=68,
       out_size=64,  # electrodes
       L=leadfield_matrix,
   )


Synchronization Study
^^^^^^^^^^^^^^^^^^^^^

**Goal:** Study emergence of network synchronization

**Recommended:** :class:`HopfOscillator` or :class:`KuramotoNetwork`

**Why:**

- Simple, interpretable dynamics
- Well-studied analytically
- Fast for large networks

.. code-block:: python

   # Kuramoto for phase-only
   kuramoto = brainmass.KuramotoNetwork(
       in_size=100,
       omega_mean=10 * u.Hz,
       omega_std=1 * u.Hz,
   )

   # Hopf for amplitude + phase
   hopf = brainmass.HopfOscillator(in_size=100, omega=10 * u.Hz, a=0.1)


Parameter Fitting Study
^^^^^^^^^^^^^^^^^^^^^^^

**Goal:** Fit model parameters to empirical data

**Recommended:** Start simple, then increase complexity

**Strategy:**

1. Start with :class:`HopfOscillator` or :class:`WilsonCowanModel` (fewer parameters)
2. Validate parameter estimates
3. If needed, move to :class:`JansenRitModel` or :class:`WongWangModel`

**Why:**

- Simpler models have fewer local minima
- Easier to interpret fitted parameters
- Can always add complexity later


Common Pitfalls
---------------

**Using Complex Models Too Early**

- More parameters = harder optimization
- Start simple, add complexity only if needed

**Ignoring Model Timescales**

- Jansen-Rit: τ ~ 10 ms → good for EEG (ms resolution)
- Wong-Wang: τ ~ 100 ms → good for fMRI (second resolution)
- Match model timescale to data modality

**Not Considering Computational Cost**

- Jansen-Rit: 6 variables per region → 6× slower than Hopf
- For exploratory work, use faster models

**Mismatched Observable**

- EEG: Use models with explicit membrane potentials (Jansen-Rit)
- fMRI BOLD: Use models with slow synaptic dynamics (Wong-Wang)
- MEG: Similar to EEG


Mixing Models
-------------

You can use different models for different regions:

.. code-block:: python

   # Thalamus: fast dynamics
   thalamus = brainmass.HopfOscillator(in_size=N_thal, omega=40 * u.Hz)

   # Cortex: slower dynamics
   cortex = brainmass.WilsonCowanModel(in_size=N_cort)

   # Couple them
   def network_step(i):
       thal_out = thalamus.update()
       cort_out = cortex.update(rE_inp=thal_out.mean())  # thalamic drive
       return cort_out


Model Extensions
----------------

All models support:

- **Noise:** Add stochastic fluctuations
- **Coupling:** Connect in networks
- **Batching:** Run multiple parameter sets in parallel
- **Optimization:** Fit parameters with gradient descent

See corresponding tutorials for details.


Summary Recommendations
-----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Your Goal
     - Recommended Model
   * - Learn brainmass basics
     - :class:`HopfOscillator` or :class:`WilsonCowanModel`
   * - EEG/MEG modeling
     - :class:`JansenRitModel`
   * - fMRI BOLD modeling
     - :class:`WongWangModel` or :class:`WilsonCowanModel`
   * - Fast exploratory work
     - :class:`Hopf Oscillator` or :class:`ThresholdLinearModel`
   * - Excitable dynamics
     - :class:`FitzHughNagumoModel`
   * - Spiking network reduction
     - :class:`QIFModel`
   * - Synchronization studies
     - :class:`KuramotoNetwork` or :class:`HopfOscillator`


Next Steps
----------

- :doc:`building_networks` - Create multi-region networks with your chosen model
- :doc:`../api/models` - Detailed documentation for each model
- :doc:`../examples/index` - See models in action


See Also
--------

- Deco et al. (2008). "The Dynamic Brain" for model selection guidance
- :doc:`parameter_fitting` for fitting model parameters
