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

### 2026-06-18 · goal-01 code-correctness
- **Bugs fixed (TDD red→green):**
  - `coupling.py` `LaplacianConnParam`: method/attr name collision — renamed the
    method `normalize`→`_laplacian`, passed as `precompute=self._laplacian`; the
    `normalize` mode string attribute now stands alone. (precompute is *lazy* — only
    called in `Param.value()`/`cache()` — so attr ordering after `super().__init__` is fine.)
  - `jansen_rit.py` `JansenRitTR.update`: `inp_*_tr` were referenced inside the sub-step
    closure before assignment; moved the `conn.update_tr()` + `k*input` lines above `def step`.
  - `noise.py` `BrownianNoise.update`: increment now `sigma*sqrt(dt)*noise` (variance scales
    with dt). Unit-safe form: `sigma * u.math.sqrt(dt/u.ms) * noise` (raw `nA + nA*sqrt(ms)` raises).
  - `noise.py` colored noise: `BlueNoise` beta -1→-2 (PSD∝f²), `VioletNoise` -2→-4 (PSD∝f⁴).
  - `forward_model.py` `LeadFieldModel`: fixed docstring/shape drift and the dangling
    `self._noise_cov_q` ref in `_sample_noise` (it's the Cholesky factor `_noise_conv_Lc`).
  - `jansen_rit.py` `s_max`: docstring said 2.5 Hz, code default 5.0 Hz. Literature value is
    `2·e0 = 5 s⁻¹` (Jansen & Rit 1995) → code was right, fixed the **docstring** only.
- **Low-risk debt:** deduped `AdditiveConn`/`DelayedAdditiveConn` into `coupling.py` (single
  source; both read state via `u.get_magnitude` → no-op on unitless HORN `'y'`, strips units
  from Jansen `'M'`; HORN/Jansen call sites pass `state=`). Replaced `brainmass.delay_index`
  self-import in `jansen_rit.py` with the local `delay_index`. Converted Google→NumPy docstrings
  in `utils.py`, `leadfield.py`, `wong_wang.py`.
- **Leadfield overlap decision:** keep BOTH. `LeadFieldModel` (forward_model.py) = unit-aware
  physical forward operator (units, vertex→region aggregation, noise cov). `LeadfieldReadout`
  (leadfield.py) = lightweight *trainable* EEG head on unitless arrays. Added reciprocal
  "use X when" + `See Also` notes to each.
- **New latent findings (NOT fixed — out of scope; flagged for maintainer):**
  - `LeadfieldReadout.normalize_leadfield` is labelled "L2" but computes `sum(sqrt(w²))=sum(|w|)`
    = **L1** norm. Left numerics unchanged; corrected the docstring/comment to say L1 and tested L1.
    Decide whether L2 (`sqrt(sum(w²))`) was intended.
  - `JansenRitNetwork` multi-layer is broken: a layer emits an mV `Quantity`, the next layer
    re-multiplies its input by `u.mV` → `UnitMismatchError`. Single-layer works. Marked the
    multi-layer test `xfail(strict)`; construction is still exercised.
- **Infra:** root `conftest.py` forces matplotlib `Agg` + autouse `plt.close("all")` so the
  `test_demo_*` (`plot=True`, `plt.show()`) no longer block pytest. Added test files:
  `horn_test.py`, `leadfield_test.py`, `utils_test.py` (HORN had none before).
- **Coverage:** all 8 changed source files ≥95% (forward_model/jansen_rit/leadfield/utils/
  wong_wang = 100%); suite 229 passed, 1 xfailed.
- **Gotchas for next goal:**
  - **`/mnt/d` (drvfs) + coverage's C/sysmon/pytrace tracer aborts (SIGABRT) on the `saiunit`
    C-extension import.** Workaround: a driver that imports `jax/brainunit/brainstate/braintools`
    FIRST (loads the C-ext uninstrumented), THEN `cov.start()` + `pytest.main([...])`. See the
    pattern used this goal.
  - **File-tool absolute paths go where you point them.** The session CWD is the worktree, but
    an absolute path to `/mnt/d/.../brainmass/brainmass/utils.py` writes to **main**, not the
    worktree (`.../.claude/worktrees/<wt>/brainmass/...`). Use the worktree path or relative-
    from-CWD. (Recovered with `git -C <main> checkout HEAD -- <file>`.)
  - Bumpy `Edit`-tool persistence was observed on drvfs earlier; `Write` (whole-file) and
    Bash read-modify-write (`.count()==1` guard) are reliable.
