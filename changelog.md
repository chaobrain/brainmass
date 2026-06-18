# Release Notes

## Unreleased

### New Features and Models

- **Model parity (TVB mean-field), sub-project F.** Added three literature-faithful complex
  mean-field models on the unified `NeuralMassDynamics` base, each with equations + numbered
  references in the docstring, `exp_euler`-default integration (with `braintools.quad`
  alternatives), unit/shape/dtype/batched support, gradient flow, and reference-regression
  tests (RHS-fidelity oracles + published-feature fallbacks, with `requires_tvb`/
  `requires_tvboptim`-gated live comparisons):
  - `EpileptorStep` — Jirsa et al. (2014). Six state variables `(x1, y1, z, x2, y2, g)` whose
    slow permittivity variable `z` autonomously drives seizure onset/offset; `x0` sets
    epileptogenicity. `lfp()` (`x2 - x1`) proxy.
  - `LarterBreakspearStep` — Breakspear, Terry & Friston (2003). Conductance-based `(V, W, Z)`
    mean field with Na/K/Ca channel gating; `d_V` selects the dynamical regime.
  - `CoombesByrneStep` — Coombes & Byrne (2019). Next-generation (exact) mean field of θ/QIF
    networks in `(r, v)` form; reduces to `MontbrioPazoRoxinStep` (`J = 0`) when `k = 0`.

## Version 0.0.6 (2026-06-18)

### Bug Fixes

- `coupling.py` — fixed a method/attribute name collision in `LaplacianConnParam` (the
  `normalize` precompute method shadowed the `normalize` mode string) that broke Laplacian
  connectivity normalization when it was actually exercised.
- `jansen_rit.py` `JansenRitTR.update` — fixed the time-resolution sub-step closure, which
  read `inp_M_tr`/`inp_E_tr`/`inp_I_tr` before they were assigned (it only worked by a
  late-binding accident).
- `noise.py` `BrownianNoise.update` — corrected the increment scaling so variance grows with
  `dt` (`sigma * sqrt(dt) * noise`) instead of collapsing to `sigma`.
- `noise.py` colored noise — corrected the spectral exponents of `BlueNoise` and
  `VioletNoise` to be consistent with the `1/f^β` definitions and their docstrings.
- `forward_model.py` `LeadFieldModel` — fixed docstring/shape drift and a dangling
  `self._noise_cov_q` reference in `_sample_noise` (the Cholesky factor `_noise_conv_Lc`).
- `jansen_rit.py` — corrected the `s_max` docstring (2.5 Hz → 5.0 Hz) to match the code and
  the literature value `2·e0` (Jansen & Rit 1995).

### Code Quality

- Deduplicated the `AdditiveConn`/`DelayedAdditiveConn` recurrent-connection classes into a
  single source of truth in `coupling.py`.
- Replaced a `brainmass.delay_index` self-import in `jansen_rit.py` with the local definition.
- Converted remaining Google-style docstrings to NumPy style (`utils.py`, `leadfield.py`,
  `wong_wang.py`) and documented when to use `LeadFieldModel` (unit-aware physical forward
  operator) versus `LeadfieldReadout` (lightweight trainable EEG head).

### Documentation

- Eliminated API drift across the documentation: every legacy `*Oscillator`/`*Model`/`QIF`
  class reference now uses its canonical `*Step` name (e.g. `HopfOscillator` → `HopfStep`,
  `WilsonCowanModel` → `WilsonCowanStep`, `QIF` → `MontbrioPazoRoxinStep`).
- Rewrote the Quickstart and other hero examples to run against the current API, and enabled
  `sphinx.ext.doctest` so docstring examples and key tutorial snippets are executed and
  verified at build time. Data-dependent tutorials are explicitly marked as non-executed.
- Merged the duplicated `autodoc_default_options` in `docs/conf.py` (the second definition
  had been silently disabling member documentation).
- Added a runnable Quick Start snippet to the README and corrected the citation version to
  match the package version.

### Tests

- Added regression tests (written test-first) covering the corrected coupling, Jansen-Rit,
  noise, and forward-model behavior.

## Version 0.0.5 (2026-01-07)

### Core Architecture

- Introduced unified `Dynamics` base class providing a consistent interface for all neural mass models, including standardized initialization methods and a `varshape` property for neuron population geometry specification.
- Refactored all model classes to adopt the "Step" suffix naming convention (e.g., `JansenRitStep`, `HopfStep`, `WilsonCowanStep`) for improved consistency throughout the codebase.
- Reorganized parameter handling to utilize the `Param` class for unit-safe initialization and validation across all dynamical models.
- Relocated `XY_Oscillator` to a dedicated module to improve code organization and maintainability.

### New Features and Models

- Implemented `HORNSeqLayer` with support for IO region handling in visual and motor task simulations.
- Replaced `QIFStep` with the more robust `MontbrioPazoRoxinStep` model for improved consistency with theoretical frameworks.
- Extended the Wilson-Cowan model with additional variants and enhanced input handling capabilities.
- Added `laplacian_connectivity` function with support for multiple connectivity types, including expanded documentation.
- Improved temporal accuracy in ODE step functions for neural mass model simulations.

### Documentation and Tutorials

- Published comprehensive tutorial series including:
  - Harmonic Oscillator Network Dynamics with parameter exploration and network simulations
  - Getting Started guide for new users
  - Stuart-Landau oscillator dynamics
  - Van der Pol oscillator dynamics
- Standardized tutorial structure by removing numerical prefixes from headers for improved navigation and consistency.
- Unified docstring formatting across all modules following NumPy documentation standards.
- Enhanced notebook structure with improved simulation descriptions and clearer organizational hierarchy.

### Dependencies and Infrastructure

- Updated `brainstate` dependency to version 0.2.9 for enhanced compatibility and performance.
- Migrated Pygments lexer from `ipython2` to `ipython3` for improved compatibility with modern Jupyter environments.
- Refactored sigmoid and other activation functions to leverage JAX primitives for GPU acceleration and automatic differentiation support.
- Updated GitHub Actions workflow dependencies (actions/checkout v5 → v6).
- Updated copyright year declarations throughout the project.

### Code Quality and Maintenance

- Removed deprecated code and unused functions to reduce technical debt.
- Systematically reorganized imports across the codebase for improved clarity and reduced circular dependencies.
- Cleaned up training loop implementations and removed obsolete state management methods.
- Standardized file naming conventions across notebooks and source files for improved discoverability.

## Version 0.0.4

- Introduced the shared `XY_Oscillator` base class and `ArrayParam` wrapper to unify initialization, transforms, and unit-safe parameter handling.
- Added new phenomenological models (`FitzHughNagumoModel`, `KuramotoNetwork`, `VanDerPolOscillator`, `StuartLandauOscillator`, `ThresholdLinearModel`, `QIF`) plus accompanying tests for the new models, Hopf dynamics, and parameter utilities.
- Published a refreshed example suite (Hopf, Wilson–Cowan, FitzHugh–Nagumo, parameter exploration, Nevergrad/SciPy optimization, MEG modeling) with notebook and script variants, including a KaggleHub-backed MEG workflow that downloads HCP sample data on demand.
- Refactored existing dynamics (Hopf, Wilson–Cowan, Jansen–Rit, Wong–Wang, noise processes, forward model) to rely on `braintools` initialization utilities, expand documentation, and tighten unit-aware behavior.
- Expanded project documentation with a reorganized API reference, refreshed landing page, new project logo, and an updated README covering installation badges, ecosystem packages, and citation details.
- Simplified CI by focusing on Python 3.13 runners and removing the multi-version JAX matrix from the daily and main workflows.
- Removed bundled example datasets and legacy notebooks/scripts; the MEG example now fetches data lazily to keep the repository lightweight.

## Version 0.0.1

The first release of the project.

