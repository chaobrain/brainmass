# Release Notes

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

