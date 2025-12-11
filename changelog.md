# Release Notes

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

