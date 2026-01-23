Architecture
============

Overview of ``brainmass`` package structure and design.


Package Structure
-----------------

.. code-block:: text

   brainmass/
   ├── __init__.py          # Public API exports
   ├── _fhn.py              # FitzHugh-Nagumo model
   ├── _hopf.py             # Hopf oscillator
   ├── _jansen_rit.py       # Jansen-Rit model
   ├── _wilson_cowan.py     # Wilson-Cowan model
   ├── _wong_wang.py        # Wong-Wang model
   ├── _vdp.py              # Van der Pol oscillator
   ├── _sl.py               # Stuart-Landau oscillator
   ├── _qif.py              # QIF model
   ├── _linear.py           # Threshold-linear model
   ├── _kuramoto.py         # Kuramoto network
   ├── _xy_model.py         # XY oscillator base class
   ├── _noise.py            # Noise processes
   ├── _coupling.py         # Coupling mechanisms
   ├── _forward_model.py    # BOLD, EEG, MEG models
   ├── _horn.py             # HORN models
   ├── _utils.py            # Utility functions
   └── _typing.py           # Type aliases


Design Principles
-----------------

1. **Consistency:** All models follow same API pattern
2. **Modularity:** Components can be mixed and matched
3. **Unit Safety:** Use brainunit for dimensional analysis
4. **JAX First:** Designed for JIT compilation and autodiff
5. **Simplicity:** Clear, readable code over cleverness


Dependencies
------------

**Core:**

- ``brainstate``: State management for dynamical systems
- ``brainunit``: Unit-aware computations
- ``braintools``: Transforms and utilities
- ``jax``: Numerical computing backend

**Rationale:**

- Separates concerns (state, units, computation)
- Reusable across BrainX ecosystem
- Leverages JAX for performance


Model Base Classes
------------------

All models inherit from ``brainstate.nn.Dynamics``:

.. code-block:: python

   class MyModel(brainstate.nn.Dynamics):
       def __init__(self, in_size, **params):
           super().__init__()
           # Initialize parameters
           # Create state variables with HiddenState

       def init_state(self, batch_size=None):
           # Initialize state variables
           pass

       def update(self, **inputs):
           # Update dynamics by one time step
           # Return observable(s)
           pass


State Management
----------------

Uses ``brainstate.HiddenState`` for internal variables:

.. code-block:: python

   self.x = brainstate.HiddenState(
       brainstate.init.Constant(0.),
       sharding=brainstate.ShardingMode.REPLICATED
   )


Coupling Architecture
---------------------

Coupling as separate modules enables:

- Reusable across models
- Swappable coupling mechanisms
- Clear separation of node vs edge dynamics


Forward Models
--------------

Designed as transformations:

.. code-block:: text

   Neural Activity → Forward Model → Observable Signal
   (hidden)         (biophysics)    (measured)


Optimization Support
--------------------

- ``ArrayParam`` wraps parameters with transforms
- Compatible with JAX autodiff
- Works with gradient-free optimizers (Nevergrad)


Extension Points
----------------

Extend ``brainmass`` by:

1. **New models:** Inherit from ``Dynamics``
2. **New noise:** Inherit from noise base classes
3. **New coupling:** Implement coupling interface
4. **New forward models:** Follow BOLD/EEG pattern


Code Organization
-----------------

- Private modules: ``_module.py`` (implementation)
- Public API: ``__init__.py`` (exports)
- Tests: ``tests/test_module.py``
- Examples: ``examples/###-name.ipynb``


See Also
--------

- :doc:`creating_models` - Build custom models
- :doc:`extending_noise` - Custom noise processes
- ``brainstate`` documentation
- ``brainunit`` documentation
