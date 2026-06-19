# Release Notes

## Version 0.1.1 (2026-06-19)

A compatibility release that unblocks installing brainmass alongside the rest of the
BrainX ecosystem (notably `brainpy` 2.8.0, which requires `braintools>=0.3.0`).

### Changed

- **Dependencies** — raised the `braintools` constraint from `<0.2.0` to `>=0.3.0`.
  brainmass 0.1.0 capped `braintools` below `0.2.0` because `braintools` 0.2.0
  introduced an `init.param` batched-initialization regression that double-prepended
  the batch axis (returning `(B, B, N)` instead of `(B, N)`). That regression was
  fixed upstream in `braintools` 0.3.0 (`init.param` no longer double-applies the
  batch dimension), so brainmass now requires the fixed release. This resolves the
  `braintools` version conflict between brainmass and `brainpy` 2.8.0, allowing the
  full ecosystem to co-install.

No source or public-API changes; behavior is identical to 0.1.0 on a correctly
resolved dependency set.

## Version 0.1.0 (2026-06-19)

This is a milestone release: brainmass grows from a collection of neural-mass *models* into a
complete, end-to-end **differentiable whole-brain modeling toolkit**. It adds a high-level
orchestration and fitting layer, reaches feature parity with The Virtual Brain's node /
coupling / observation libraries, and ships a fully restructured, persona-driven documentation
site. Because gradients flow through the entire *simulate → observe → score* pipeline, model
parameters can be recovered by gradient descent — not only by grid or evolutionary search.

### Highlights

- **High-level API** — `Simulator`, `Network`, and `Fitter` turn a model, a connectome, and a
  target into a few lines of code, with a single `Fitter` exposing three optimizer backends
  (gradient, gradient-free, and Bayesian).
- **Model parity with TVB** — seven new literature-faithful mean-field models, bringing the
  built-in catalogue to 17 model families.
- **Coupling & observation parity** — nonlinear couplings plus a convolution-based HRF-BOLD
  forward model with swappable haemodynamic kernels.
- **A new documentation site** — reorganized around the Diátaxis quadrants with on-ramps for
  beginners, experimentalists, and modelers, plus a Data-Driven Modeling showcase.

### Orchestration and Fitting

- **`Simulator`** — a JIT-friendly driver that rolls a model forward with `brainstate.transform`
  (no Python step loops), records the selected variables, and returns stacked trajectories. It is
  the single entry point used throughout the tutorials and gallery.
- **`Network`** — a connectome builder that wires neural-mass nodes through a coupling and an
  (optionally delayed) connectivity matrix, handling delay-buffer sizing and unit-safe
  broadcasting.
- **`Fitter` / `FitResult`** — one fitter, three interchangeable optimizer backends:
  gradient-based (Optax, backpropagating through the ODE solve), gradient-free (Nevergrad), and
  Bayesian (scikit-optimize). `FitResult` carries the recovered parameters, the loss history, and
  the fitted simulation for inspection.
- **`brainmass.objectives`** — composable, differentiable loss terms for data fitting:
  `timeseries_rmse`, `fc_corr` / `fc_rmse`, `cosine_sim`, functional-connectivity-dynamics
  distances (`fcd`, `fcd_distribution`, `fcd_ks`, `fcd_wasserstein`, `ks_distance`,
  `wasserstein_1d`), and a `combine` helper for weighted multi-term objectives.

### Models

- **Model parity with TVB (mean-field node library).** Brought brainmass to model parity with the
  tvboptim node library, adding seven literature-faithful models on the unified
  `NeuralMassDynamics` base. Each ships equations + numbered references in the docstring,
  `exp_euler`-default integration (with `braintools.quad` alternatives), unit/shape/dtype/
  batched support, gradient flow, and reference-regression tests: the RHS is checked against an
  embedded per-unit-time transcription of the upstream tvboptim equation (`rtol` 1e-6), the
  short-horizon RK4 trajectory against that same reference (correlation ≥ 0.99), plus
  always-on dynamical-feature assertions.

  Complex mean-field models:
  - `EpileptorStep` — Jirsa et al. (2014). Six state variables `(x1, y1, z, x2, y2, g)` whose
    slow permittivity variable `z` autonomously drives seizure onset/offset; `x0` sets
    epileptogenicity. `lfp()` (`x2 - x1`) proxy.
  - `LarterBreakspearStep` — Breakspear, Terry & Friston (2003). Conductance-based `(V, W, Z)`
    mean field with Na/K/Ca channel gating; `d_V` selects the dynamical regime.
  - `CoombesByrneStep` — Coombes & Byrne (2019). Next-generation (exact) mean field of θ/QIF
    networks in `(r, v)` form; reduces to `MontbrioPazoRoxinStep` (`J = 0`) when `k = 0`.

  Canonical / excitatory–inhibitory models:
  - `Generic2dOscillatorStep` — Sanz-Leon et al. (2015). TVB's flexible planar oscillator
    `(V, W)` whose polynomial-nullcline coefficients select the regime (excitable, bistable,
    Morris-Lecar-like).
  - `WongWangExcInhStep` — Deco et al. (2014). Two-population excitatory–inhibitory reduced
    Wong–Wang mean field `(S_E, S_I)`, the resting-state workhorse; distinct from the
    single-population decision-making `WongWangStep`. Exposes `H_e()` / `H_i()` population
    firing rates.
  - `LorenzStep` — Lorenz (1963). Chaotic `(x, y, z)` test fixture (use `dt = 0.01 * u.ms`).
  - `LinearStep` — TVB `Linear` node, `dx/dt = gamma*x + coupling` (distinct from the
    two-population `ThresholdLinearStep`).

### Coupling

- **Nonlinear coupling parity (TVB).** Added three differentiable nonlinear coupling functions
  alongside the existing `DiffusiveCoupling` / `AdditiveCoupling`, each available both as a
  `State`-carrying class and a pure functional helper: `SigmoidalCoupling` / `sigmoidal_coupling`,
  `HyperbolicTangentCoupling` / `hyperbolic_tangent_coupling`, and `SigmoidalJansenRitCoupling` /
  `sigmoidal_jansen_rit_coupling` (the pre/post-synaptic sigmoid used by Jansen–Rit networks).
  `AdditiveCoupling` gained an additive-bias term so node inputs can be offset directly.

### Observation and Forward Models

- **Convolution-based HRF-BOLD parity.** `HRFBold` convolves population activity with a
  haemodynamic response kernel to produce BOLD signals, with four swappable kernels behind a
  shared `HRFKernel` interface — `GammaHRFKernel`, `DoubleExponentialHRFKernel`,
  `MixtureOfGammasHRFKernel`, and `FirstOrderVolterraHRFKernel` (a Balloon–Windkessel
  linearization). This complements the existing physiological Balloon `BOLDSignal`.
- **`TemporalAverage`** — a lightweight observation operator that block-averages high-rate neural
  activity down to the imaging sampling rate (e.g. ms → TR), keeping fitting targets and
  simulations on a common time base.

### Architecture and Internals

- Unified the two-variable model classes onto a single shared base and split the larger
  `jansen_rit` and `horn` modules into packages, cutting duplication across the model zoo.
- Co-located tests as sibling `*_test.py` files next to the code under test, and added a
  differentiability / determinism / coverage safety net so gradient flow and reproducibility are
  exercised in CI.

### Documentation

- Converted the narrative guides (`tutorials/*` and `developer/*` plus the FAQ) from
  reStructuredText to executable Jupyter notebooks with thebe/live cells enabled. In the
  process, **repaired pre-existing API drift** in
  several guides that the doctest gate never caught (those examples were display-only
  `code-block`s): updated `WilsonCowanStep` recurrent-coupling kwargs (`c_EE` → `wEE`/`wEI`/…),
  the prefetch-based `DiffusiveCoupling`/`AdditiveCoupling` construction API, `KuramotoNetwork`
  (`omega`/`K` instead of `omega_mean`/`omega_std`), lead-field shapes/units, and the
  resting-state BOLD driver (`WongWangExcInhStep`/`WilsonCowanStep` instead of the reduced
  decision-making `WongWangStep`); replaced missing `.npy` connectivity loads with seeded
  inline synthetic matrices. Nearly every narrative guide now executes with embedded outputs (a
  couple ship unexecuted by design: intentional error-demo fragments and one advanced
  `brainstate` API not present in the pinned version).
- Documented the four new canonical/E-I models in `reference/models.rst` (comparison table,
  autosummary, and runnable usage examples), and disambiguated the `WongWangStep` (reduced
  decision-making) vs `WongWangExcInhStep` (two-population resting-state) rows.
- **Reorganized the documentation around the Diátaxis quadrants** with persona on-ramps and a
  data-driven showcase. The new top-level sections are **Getting Started** (installation,
  quickstart, key concepts, learning paths), **Tutorials** (a sequential 8-part path),
  **How-To Guides** (7 task recipes), **Concepts** (5 explanation pages), **Data-Driven
  Modeling** (the flagship guided path + roadmap), **Gallery** (a 17-model zoo + 5 case
  studies), **API Reference**, and **Developer**. Every model family now has a runnable demo,
  closing the long-standing gap where 9 models had no user-facing example. All narrative guides
  are executable Jupyter notebooks with real embedded outputs.
- **Path moves / redirects.** The API reference moved from `apis/` to `reference/`, the
  Quickstart moved from `tutorials/quickstart` to `getting_started/quickstart`, and the
  existing tutorial notebooks were split across `getting_started/`, `howto/`, and a renumbered
  `tutorials/01_…08_` sequence (all via `git mv`, history preserved). In-notebook and `:doc:`
  cross-references were rewritten to absolute docnames (leading slash) so they survive future
  moves. Bookmarks to the old `apis/index` and `tutorials/quickstart` paths should be updated
  to `reference/index` and `getting_started/quickstart` respectively.
- **Retired the legacy flat `examples/` notebook tree** (17 notebooks), now fully superseded by
  the in-tree `docs/gallery/` model zoo and case studies. The KaggleHub-backed MEG example was
  reworked as a `gallery/case_studies/resting_state_meg` demo that uses the bundled synthetic
  connectome instead of downloading external data.
- **New ergonomics surfaced in the API reference and used throughout the docs:**
  `brainmass.datasets` (a bundled synthetic connectome / signal / task plus an extensible
  dataset registry), `brainmass.viz` (thin matplotlib plotting helpers, behind the optional
  `[viz]` extra), and `brainmass.list_models()` (a curated model catalogue with `to_table()`).

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

