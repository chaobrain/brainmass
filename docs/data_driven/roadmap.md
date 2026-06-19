# Data-Driven Modeling Roadmap

> **Forward-looking.** This page reserves homes for the growth areas of the data-driven
> pillar. Every item below is **named and given a home now, not built in goal-13.** Each
> has a documented *contract* — the existing public API it will slot into — so it can ship
> later without restructuring the docs or breaking the surface users build on today. The
> contract for all three is the extension playbook
> {doc}`/developer/building_a_data_driven_workflow`.

The center of gravity of brainmass is **data-driven modeling** — constructing, fitting,
and training neural-mass networks against data. What exists today, and what comes next:

| Verb | Today (shipped) | Next (this roadmap) |
| --- | --- | --- |
| **Construct** | {class}`brainmass.Network` builds a whole-brain model from a connectome. | Model **discovery** / system identification — learn the structure, not just tune it. |
| **Fit** | {class}`brainmass.Fitter` fits parameters to a *single fixed target*, gradient or gradient-free. | A task-shaped **`Trainer`** for `(inputs, targets)` datasets, and **amortized inference** over posteriors. |
| **Observe** | Forward models map activity → BOLD / EEG / MEG to compare against data. | — (complete for the current modalities). |

The three deferred tracks below all extend that picture.

---

## Model discovery / system identification — *deferred*

**What it is.** Today you pick a model (`HopfStep`, `WilsonCowanStep`, …) and fit its
parameters. *Discovery* learns the model **structure** — which terms, which couplings,
which latent populations — from data, rather than fitting a fixed equation. This is the
sparse-regression / symbolic-regression / latent-ODE family of methods adapted to
neural-mass dynamics.

**Why it fits brainmass.** The differentiable core is exactly what these methods need:
candidate structures are scored by backpropagating a data-fit loss through the solve,
and a sparsity penalty prunes terms. The machinery is the same one
{doc}`/developer/building_a_data_driven_workflow` documents — a trainable
`Param(value, fit=True)` per candidate term, a composable objective, and a gradient loop.

**Reserved home + contract.** A `data_driven/discovery/` section will ship as a stub
index plus this roadmap entry. The discovery track will:

- register its datasets through {mod}`brainmass.datasets` (`register_dataset` /
  `load_dataset`), exactly as the built-in `example_connectome` / `delayed_match_task`
  do, so a discovery benchmark is a first-class dataset;
- express each candidate term as a trainable parameter and its score as a
  `brainmass.objectives`-style `callable(prediction, target) -> scalar`, per the
  {doc}`/developer/building_a_data_driven_workflow` contract;
- compose with the existing {class}`brainmass.Simulator` / {class}`brainmass.Network`
  rather than introducing a parallel run loop.

---

## A task-shaped `Trainer` — *deferred*

**What it is.** A trainer for driving a network's parameters from a **task** — a dataset
of `(inputs, targets)` pairs — distinct from {class}`brainmass.Fitter`'s target-fitting
shape. This is how you'd train a `HORNSeqNetwork` (or any differentiable neural-mass net)
on a cognitive task: minibatch the data, loop over epochs, and track a held-out metric.

**The concrete gap it fills.** {class}`brainmass.Fitter` fits against **one fixed
target**: a single `predict(model) -> prediction` compared to a single `target`. Task
training is structurally different and `Fitter` exposes none of it:

- **no minibatching** — there is no `(inputs, targets)` dataset abstraction, only one
  target;
- **no epoch loop** — `fit(n_steps=...)` is optimiser steps against the fixed target, not
  passes over a dataset;
- **no held-out metric** — no train/validation split, no per-epoch evaluation hook.

So the tutorials and case studies that *do* train ({doc}`/tutorials/08_training_on_tasks`,
{doc}`/gallery/case_studies/horn_cognitive_task`) drop down to a **hand-written loop**:
`net.states(ParamState)` → `optimizer.register_trainable_weights(...)` → a jitted
`brainstate.transform.grad(loss, weights, has_aux=True, return_value=True)` →
`optimizer.step(grads)`, looped over epochs and minibatches. That loop is the `Trainer`'s
body, waiting to be wrapped.

**A second, sharper gap — batched state init.** Training needs batched hidden states (one
trajectory per example in the minibatch), but `HORNStep.init_state` **ignores its
`batch_size` argument** and allocates `x`, `y` at `self.in_size` only. So
`init_all_states(net, batch_size=B)` leaves the velocity state unbatched, and the scan
carry shape-mismatches `(B, 32)` vs `(32,)`. The current workaround in the training
notebooks is to **manually broadcast** the hidden states to batch shape before each
minibatch forward (`layer.horn.x.value = jnp.zeros((B,) + tuple(layer.horn.in_size))`,
likewise for `y`). A `Trainer` is the right home for that broadcast — and for fixing the
underlying `init_state` to honour `batch_size` — so callers never hand-roll it.

**Reserved home + contract.** The `Trainer` is **named and given a home now but not
implemented in goal-13.** It will build *against the exact contract* in
{doc}`/developer/building_a_data_driven_workflow`:

- it wraps the hand-written `grad` loop above (the playbook's section-3b loop) plus
  batched state initialisation;
- it registers task datasets through {mod}`brainmass.datasets`, like
  `delayed_match_task`;
- it reuses `brainmass.objectives`-style losses unchanged.

Authoring a custom model, objective, and loop the way the playbook shows is precisely
what makes a model `Trainer`-ready before the `Trainer` exists.

---

## Simulation-based / amortized inference — *deferred*

**What it is.** Where {class}`brainmass.Fitter` returns a **point estimate** of the best
parameters, simulation-based inference (SBI) returns a **posterior** — a distribution over
parameters consistent with the data — using likelihood-free / amortized methods (e.g.
neural posterior estimation). *Amortized* means the inference network is trained once on
simulations and then yields a posterior for new data in a single forward pass.

**Why it fits brainmass.** SBI is simulation-hungry, and brainmass simulations are JAX
programs: `vmap` and `jit` make generating the large simulation banks SBI needs cheap, and
GPU/TPU batching scales them. The differentiable core also enables gradient-informed SBI
variants that fixed-backend frameworks can't offer.

**Reserved home + contract.** Reserved as a future track under `data_driven/`. It will:

- draw its simulation banks from {class}`brainmass.Simulator` / {class}`brainmass.Network`
  under `vmap`-over-parameters (the parameter-batching idiom from
  {doc}`/howto/batch_and_accelerate`);
- score / summarise simulations with `brainmass.objectives`-style callables;
- follow the same dataset and objective contracts as the other tracks, per
  {doc}`/developer/building_a_data_driven_workflow`.

---

## The stable contract

All three deferred tracks build against the *same* public surface that exists today, so
each slots in without a breaking change:

- **Trainable parameters** — `Param(value, fit=True)` (constrained variants via a
  transform `t=...`).
- **Composable objectives** — `brainmass.objectives`-style
  `callable(prediction, target) -> scalar`, jit / grad / vmap-safe and unit-aware.
- **Datasets** — {mod}`brainmass.datasets` (`register_dataset` / `load_dataset`).
- **Run loop** — {class}`brainmass.Simulator` and {class}`brainmass.Network`, never a
  parallel implementation.

The full worked contract — exposing parameters, writing an objective, fitting both with
the high-level {class}`brainmass.Fitter` and a hand-written `grad` loop, and batching the
search — is {doc}`/developer/building_a_data_driven_workflow`. Read it before building any
new data-driven machinery.
