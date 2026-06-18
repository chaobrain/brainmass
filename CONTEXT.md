# CONTEXT — brainmass improvement program

Shared context + lessons ledger for the `/goal` program in `dev/superpowers/`.
Read this in full before starting any goal. Append a Lessons entry at the end of yours.

## Orientation

brainmass is a JAX-based, differentiable whole-brain / neural-mass modeling library in
the BrainX / chaobrain ecosystem. It ships **only models** and delegates everything else:

- `brainstate` — `Dynamics`/`Module` base classes, `HiddenState`/`ParamState`,
  `Param`/`Const`, the `environ` global (holds `dt`), constraint `Transform`s
  (`ReluT`, `SigmoidT`, …), `Regularization`s (`GaussianReg`, …), and the
  compile/transform layer (`for_loop`, `jit`, `grad`, `vmap`, `exp_euler_step`).
- `brainunit` (`u`) — physical units on every quantity.
- `braintools` — initializers (`braintools.init`), integrators
  (`braintools.quad.ode_*_step`), metrics (`braintools.metric.functional_connectivity`,
  `matrix_correlation`, `functional_connectivity_dynamics`), and optimizers
  (`braintools.optim`: optax / nevergrad / scipy wrappers).

A user cannot do a full study with brainmass alone — they must know the
`brainstate`/`braintools` APIs. The improvement program adds the missing in-package
orchestration (Simulator / Network / Fitter) on top of these primitives.

## Key facts every goal should know

- **Constrained params are already solved** by brainstate:
  `Param(value, t=Transform, reg=Regularization, fit=True)` + `Param.init(value, varshape)`
  + `module.reg_loss()` + `module.param_precompute()`. The value is stored in
  *unconstrained* space (`t.inverse`) and `Param.value()` returns the constrained value
  (`t.forward`). `fit=True` wraps it in a `ParamState` (trainable). Consume this; never
  reinvent bounds handling.
- **Model contract**: every model subclasses `brainstate.nn.Dynamics`; `__init__` wraps
  params via `Param.init`; `init_state(batch_size)` allocates `HiddenState`s; per-variable
  `dX` methods; a `derivative(state, t, *inps)` aggregator; `update(*inputs)` branches
  `exp_euler` (default) vs `getattr(braintools.quad, f'ode_{method}_step')`. Variations to
  respect: unit-carrying states/inputs (Jansen–Rit), derived-observable returns
  (`eeg()` = E − I), and `t` sourced as `0*u.ms` vs `environ.get('t', 0*u.ms)`.
- **Simulation idiom**: `environ.set(dt=…)` → `init_all_states()` →
  `brainstate.transform.for_loop(step_run, indices)` → collect `state.value`. Whole-brain
  nets put `step_run` as a method using `environ.context(i=, t=)`. `JansenRitTR.update`
  is the in-package precedent for a dt-substep-per-output-sample (TR) loop.
- **Coupling**: `DiffusiveCoupling(x_prefetch, y_prefetch, conn, k)` and
  `AdditiveCoupling(x_prefetch, conn, k)` consume `prefetch_delay(var, (delay, idx), init)`
  / `prefetch(var)`; `init_state` is `@call_order(2)` and calls `init_maybe_prefetch`.
- **FCD is unused**: `braintools.metric.functional_connectivity_dynamics` exists but has
  zero call sites — surface it in the objectives layer (goal-08).

## Known latent bugs (fix via TDD in goal-01 / goal-05)

- `coupling.py` `LaplacianConnParam.normalize` — method/attribute name collision
  (`precompute=self.normalize` then `self.normalize = normalize`; the method calls
  `self.normalize` internally). Broken when exercised.
- `jansen_rit.py` `JansenRitTR.update` — reads `inp_M_tr`/`inp_E_tr`/`inp_I_tr` before
  assignment (works only by Python late-binding accident).
- `noise.py` `BrownianNoise.update` — increment algebra cancels (`/dt_sqrt * dt_sqrt`) to
  `sigma` instead of `sigma*sqrt(dt)`.
- `noise.py` `BlueNoise`/`VioletNoise` — `beta` signs inconsistent with the `1/f^β` math
  and their own docstrings.
- `forward_model.py` `LeadFieldModel` — docstring/shape drift; `_sample_noise` references a
  nonexistent `self._noise_cov_q`.
- `jansen_rit.py` — `s_max` default mismatch (docstring 2.5 Hz vs code 5.0 Hz).

## Known drift / debt

- **Docs API drift**: `HopfOscillator` (×26, incl. Quickstart), `WilsonCowanModel` (×18),
  `WongWangModel`, `FitzHughNagumoModel`, `StuartLandauOscillator`,
  `VanDerPolOscillator`, `QIF` — all renamed to `*Step`; the imports won't run.
  Notebooks/tutorials are not executed at build (`jupyter_execute_notebooks = "off"`).
- `docs/conf.py` defines `autodoc_default_options` twice (the second silently wins).
- `Publish.yml` reads `brainmass/_version/__init__.py` (nonexistent; real file is
  `brainmass/_version.py`) → fails real releases.
- CI spans 4 Python versions, none being the declared floor 3.11.
- Cruft: untracked `nul`, `dev/replacement.py`, stray `[tool.flit.module]` in pyproject.
- `ModelFitting` is copy-pasted ~7× across `tutorials/`; whole-brain `Network` ~8× across
  `examples/`. The orchestration goals (06–08) consolidate these.

## Lessons ledger

> Append below, newest last. Format: `### <date> · goal-NN <slug>` then terse bullets:
> what changed, gotchas/surprises, and anything the next goal needs.

### 2026-06-18 · goal-00 bootstrap
- Created this ledger. No code changes. Next: goal-01 (code correctness, TDD on the bugs above).
- Gotcha: the worktree branched from `origin/main`, which at audit time was **two commits
  behind** local `main` — `docs/roadmap.md` (added in unpushed commit `3ca0bfa`) was absent
  from the worktree. Read it via `git show main:docs/roadmap.md`. If a future goal needs the
  roadmap and it's missing, check whether the roadmap commit has reached `origin/main` yet.
