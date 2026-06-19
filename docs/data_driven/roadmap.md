# Data-Driven Modeling Roadmap

> **Skeleton** — reserves homes for the growth areas of the data-driven pillar. The
> deferred items below are **named and given a home now, not built in goal-13.** The
> `brainmass.datasets` registry, the objective extension pattern, and
> `developer/building_a_data_driven_workflow` are laid out so each slots in without
> restructuring the docs or the API.

The center of gravity of brainmass is **data-driven modeling** — constructing, fitting,
and training neural-mass networks against data. Today that means:

- **Construction** — `Network` builds a whole-brain model from a connectome.
- **Fitting** — `Fitter` fits parameters to a target with gradient-based or gradient-free
  backends.
- **Observation** — forward models map activity to BOLD / EEG / MEG signals to compare
  against data.

The roadmap below reserves homes for where this pillar grows next.

## Model discovery / system identification — *deferred*

Learning model **structure** or parameters from data, rather than fitting a fixed model.
A `data_driven/discovery/` section will ship as a stub index plus this roadmap entry; the
discovery track will register its own datasets through `brainmass.datasets`.

## Training a task-shaped `Trainer` — *deferred*

A task / dataset / epoch-shaped **`Trainer`**, distinct from `Fitter`'s target-fitting
shape, for training neural-mass-style networks (e.g. HORN) on cognitive tasks. It is
**named and given a home now but not implemented in goal-13.** The `datasets` registry,
the objectives extension pattern, and `developer/building_a_data_driven_workflow` are the
stable contract it will slot into.

## Simulation-based / amortized inference — *deferred*

Posterior inference over model parameters via simulation-based (likelihood-free) methods.
Reserved as a future track.
