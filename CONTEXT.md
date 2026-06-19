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
- `qif.py` `MontbrioPazoRoxinStep.update` (~line 260, non-`exp_euler` branch) — calls
  `ode_<method>_step((r, v), 0.*u.ms, r_inp, v_inp)` and **omits `self.derivative`** as the
  first positional arg, so `braintools.quad.ode_rk2_step`/`ode_rk4_step` get the state tuple
  where the vector field should be → `TypeError: input function should be callable`. Only the
  default `exp_euler` path works; `method='rk2'|'rk4'` is dead. Found in goal-04 (integrator
  sweep), parked as **2 strict `xfail`s** in `integrator_test.py` (so the fix flips them to
  pass). **RESOLVED in goal-05** — `MontbrioPazoRoxinStep` now routes through the shared
  `NeuralMassDynamics._solve_step`, which passes `self.derivative` correctly; the two `xfail`
  markers were removed and the cases pass.

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

### 2026-06-18 · goal-02 docs-integrity
- **API drift killed (grep over `docs/` + README now empty):** renamed every legacy class
  reference to its canonical `*Step` (`HopfOscillator`→`HopfStep`, `WilsonCowanModel`→
  `WilsonCowanStep`, `WongWangModel`→`WongWangStep`, `JansenRitModel`→`JansenRitStep`,
  `FitzHughNagumoModel`→`FitzHughNagumoStep`, `ThresholdLinearModel`→`ThresholdLinearStep`,
  `StuartLandauOscillator`→`StuartLandauStep`, `VanDerPolOscillator`→`VanDerPolStep`,
  `QIF`/`QIFModel`→`MontbrioPazoRoxinStep`). Left untouched: real classes (`LeadFieldModel`,
  EEG/MEG), pedagogical placeholders (`MyModel`/`CustomOscillator`), HORN's real `omega`
  param, Kuramoto's `omega_mean`/`omega_std`, and **historical changelog entries** (0.0.4/0.0.5
  legitimately shipped the old names — do NOT rewrite release notes; the DoD grep is scoped to
  the live `docs/` tree, not `changelog.md`).
- **`HopfStep` has no `omega`/`u.Hz` arg** — it takes a *dimensionless* `w` (normal-form angular
  frequency). All `HopfStep(omega=…*u.Hz)` calls in docs were never-runnable; fixed to `w=`.
- **Real `DiffusiveCoupling` API** (docs were full of the imagined `DiffusiveCoupling(conn=W,k=)`
  + `coupling(src,tgt)` form): construct from prefetch refs —
  `nodes.prefetch_delay('x', delays, src_idx, init=braintools.init.Constant(0.0))` (gives the
  `(N,N)` source read the impl REQUIRES) + `nodes.prefetch('x')` (the `(N,)` self read), then
  `DiffusiveCoupling(x_src, x_self, conn=W, k=…)`; drive with `coupling.update()` inside
  `environ.context(i=,t=)`. A plain `(N,)` source read raises `incompatible shape … expected
  (...,N,N)`. Verified runnable in quickstart + building_networks + adding_coupling.
- **Imagined whole-brain API still littering tutorials (NOT a real API):** `nodes.S_E.value`,
  `nodes.update(S_E_ext=…)`, `nodes.noise_E=…` assume a Wong-Wang/DMF that doesn't exist
  (`WongWangStep` has `S1`/`S2`, `update(coherence=)`, `noise_s1`/`noise_s2`). These live in the
  data-dependent tutorials (forward_modeling, parameter_fitting, building_networks whole-brain
  example) — marked **schematic / not executed** with a `.. note::` per the goal's allowance,
  rather than rewritten. A future orchestration goal (06–08) that adds a real `Network`/`Simulator`
  should revisit and make them runnable.
- **Docs are now self-checking via `sphinx.ext.doctest`** (lightest robust option chosen over
  notebook execution, which stays `nb_execution_mode="off"`). Key mechanics:
  - `.. testcode::` = execute-only (no output match); bare `>>>` napoleon blocks ARE checked
    for output because `doctest_test_doctest_blocks = "default"`. `.. code-block:: python` with
    `>>>` is NOT tested — only bare doctest blocks and `testcode` are.
  - `doctest_global_setup` imports ONLY `brainmass` (+ brainstate/u/jax/np/plt, `Agg`, `dt`).
    So docstring `>>>` examples MUST be self-contained: qualify `brainmass.HORNSeqNetwork`, not
    bare `HORNSeqNetwork`. **`doctest.DocTestFinder` over the package gives FALSE PASSES here** —
    it uses each module's own globals (bare class names resolve), so it missed `horn.py`'s bare
    `HORNSeqNetwork`. `sphinx-build -b doctest docs docs/_build/doctest` is the ONLY authoritative
    gate. Final: **48 tests, 0 failures**.
  - In bare `>>>` blocks, `init_all_states()`/`brainstate.nn.init_all_states(m)` RETURN the module
    (repr prints → doctest mismatch). Capture with `_ = …`. (In `.. testcode::` it's harmless.)
- **Deliverables:** README runnable Quick Start + citation `0.0.4`→`0.0.6`; `changelog.md` 0.0.6
  entry (goal-01 fixes + this docs work); `contributing.rst` corrected (NumPy not Google
  docstrings; no `tests/` dir — tests are co-located `brainmass/<m>_test.py`, `pytest brainmass/`;
  `.[dev,doc]` extras don't exist yet → install via `requirements-doc.txt`, consolidated extras
  are goal-03); `conf.py` single `autodoc_default_options`. `pytest brainmass/` still green
  (229 passed, 1 xfailed).
- **Gotcha for next goal:** the manual git worktree at `.claude/worktrees/goal-02-docs-integrity`
  must be committed-to early — a native/empty worktree got auto-cleaned ("unchanged") mid-session.
  And the same drvfs absolute-path trap as goal-01: file tools with an absolute path to the MAIN
  checkout write to main, not the worktree.

### 2026-06-18 · goal-04 testing-rigor
- **Headline delivered — gradients now tested.** New `gradient_test.py` (23 tests) proves AD flows
  through a `for_loop` sim for Hopf/WilsonCowan/JansenRit/Montbrio/FitzHughNagumo: grad-vs-FD,
  `jax.test_util.check_grads(modes=['rev'])`, vmap-batched grads, zero-noise-limit, and
  determinism-under-seeding. Net suite **229→311 passed (+3 xfail)**; coverage **77%→97.85%**,
  every module ≥ 91.8%.
- **Differentiable-test recipe (reuse this):** build the model *inside* the loss with the swept
  scalar as `Param(p, fit=True)`, `brainstate.random.seed(0)` FIRST (purity — unseeded `Uniform`
  init silently breaks check_grads/FD), run `for_loop`, reduce with `jnp.sum(u.get_magnitude(xs)**2)`
  (**must** strip units or grad carries mV²/Hz² and FD comparison explodes). Pick a param the loss
  is actually sensitive to: Montbrio grad wrt `J` at rest is ~1e-6 (FD noise drowns it) — swept
  `eta` with a `v_inp=1.0` drive instead (AD=FD=0.1378). FD tol `rel=2e-2, abs=1e-6`.
- **exp_euler is EXACT on linear ODEs** → use as analytic oracle (`integrator_test.py` asserts
  err<1e-5 on linear decay, plus convergence-order bands exp_euler≈1 / rk2≈2). rk4 on smooth
  nonlinear systems floors at float32 round-off ~8e-7, not 1e-12 — don't assert tighter.
- **Systemic seeding bug found across the suite:** helpers seeded `brainstate.random` but built data
  with `np.random.*` (uncontrolled). Both RNGs must be seeded. Fixed in noise/coupling/horn/
  jansen_rit/leadfield/wong_wang tests; conftest `seeded` fixture seeds both.
- **conftest extended (not replaced):** autouse `_isolate_environ_dt` snapshots/restores global
  `environ['dt']` — this leakage was real (`jansen_rit` `test_eeg_output_signal` had been passing
  only on a dt leaked from an earlier test). Plus `dt`/`seeded`/`connectome` fixtures, all
  `indirect=`-parametrizable. `matplotlib.use("Agg")` at import top makes `plt.show()` a no-op.
- **Dead tests repaired:** 5 commented `test_stability_long_simulation` (WC variants) uncommented +
  converted slow `for i in range(10000)` python loops → jitted `for_loop` (all stay bounded <0.5);
  Jansen/WongWang `plot=True` demos flipped to `plot=False` default (+ seeded WongWang) and given
  finiteness/transient/bounded-gating asserts instead of bare `return`. A wrong commented Jansen
  assertion (`out[-1] >= out[0]`; output is non-monotonic) was replaced with a responsiveness check.
- **Coverage gate is now standard `pytest-cov`** (`pyproject [tool.coverage] fail_under=90, branch,
  omit *_test.py`; CI Linux job runs `--cov=brainmass --cov-fail-under=90`; macOS/Win stay plain).
  **The goal-01 drvfs coverage-SIGABRT did NOT recur** — full suite under plain `pytest --cov` ran
  clean on `/mnt/d` (EXIT=0), so the `dev/superpowers/run_cov.py` C-ext-ordering workaround was
  needed only locally before and is NOT shipped. `pytest-cov`+`coverage` added to requirements-dev.
- **For goal-05:** see the new `qif.py` entry under *Known latent bugs* — the 2 strict `xfail`s in
  `integrator_test.py` flip to pass the moment `self.derivative` is restored.

### 2026-06-18 · goal-05 structure-refactor
- **Headline — unified the 2-variable model base + split the two dumping-ground modules, with
  PROVEN zero behaviour change.** New private `brainmass/_base.py::NeuralMassDynamics` owns the
  one duplicated integrator-selection branch (`exp_euler` per-variable `exp_euler_step` vs
  `getattr(braintools.quad, f'ode_{method}_step')(self.derivative, state, t, *inps)`). `XY_Oscillator`
  (→ hopf/sl/vdp), `FitzHughNagumoStep`, `MontbrioPazoRoxinStep` now subclass it and delegate via
  `self._solve_step(exp_euler_specs=…, ode_state=…, ode_inputs=…)`. Net suite **311→337 passed**,
  coverage **97.85%→98.17%** (`_base.py`, all `jansen_rit/*` and `horn/*` submodules 100%).
- **Migration scope discipline:** only models that *duplicated* the branch were migrated.
  `WongWang` (exp_euler-only, clamps S∈[0,1], no `method`/`derivative`) and `KuramotoNetwork`
  (single-var, exp_euler-only) have **no** branch → deliberately untouched. `WilsonCowan*` and
  `JansenRitStep` keep their *own* branches: WC already has an internal base hierarchy shared by
  ~10 variants, and Jansen is a 6-variable model — neither is a "2-variable oscillator", both are
  out of scope (a separate unification risk). Migrating only the genuine duplicates kept the diff
  small and the equivalence airtight.
- **Carve big modules mechanically, not by hand.** `jansen_rit.py` (991 L) → `jansen_rit/` and
  `horn.py` (773 L) → `horn/` were split with a throwaway Python script that maps **by top-level
  class name** (immune to line-number drift), reproduces the import header, rewrites intra-package
  relatives `.coupling`→`..coupling`, and rstrip-joins blocks. Hand-retyping ~1.8k lines would
  have been error-prone; the script made it deterministic. `orphaned `zeros()` helper dropped.
- **Non-breaking package split recipe (every old path must still resolve):**
  - Order submodule imports by dependency (step → connectivity → tr/network) to dodge cycles; the
    package's on-demand `from ..noise import …` works even though `brainmass/__init__` imports the
    package *before* its own `noise` line (the submodule loads independently).
  - The package `__init__.py` **re-exports every former module member** (incl. internal helpers
    `Identity`/`LaplacianConnectivity` and the coupling names `brainmass.horn.AdditiveConn` that a
    tutorial used) so `brainmass.<pkg>.<name>` keeps importing — this is also the path **pickle**
    uses for pre-split instances, so it must hold even for non-public classes.
  - `__module__` of moved classes shifts (`brainmass.jansen_rit`→`brainmass.jansen_rit.tr`) UNLESS
    hardcoded — `JansenRitStep`/`LaplacianConnectivity` keep `__module__='brainmass'`. Cosmetic and
    nothing asserts it; the re-exports cover pickle/back-compat regardless.
  - New `import_compat_test.py` locks all of this: `__all__` resolves, both old module paths
    resolve, top-level `is` package object (one class → one pytree registration), the Model→Step
    shim still warns, and moved classes still pytree-flatten.
- **Pre/post equivalence proof beats a self-referential snapshot.** Goldens were generated from a
  **throwaway `git worktree add --detach /tmp/bm_main origin/main`** (PYTHONPATH-pinned so the
  right `brainmass` imports past the site-packages copy), then compared to the refactored code:
  **all 7 touched models bit-identical** (not just within `rtol=1e-5`). The earlier Phase-B "1-ULP"
  scare was a capture artifact — same deterministic `exp_euler` run in two clean processes fuses
  identically in XLA. `characterization_test.py` pins those goldens (fixture
  `_characterization_goldens.json`, tol `rtol=1e-5/atol=1e-6` only to absorb future platform drift).
- **goal-01 debts closed:** (1) `LeadfieldReadout.normalize` — kept L1 as the back-compat default
  (`True`≡`'l1'`, the historical `sum(sqrt(w²))=sum(|w|)`), added opt-in `'l2'`
  (`sqrt(sum(w²))`), validated `_resolve_normalize` (bad value → `ValueError`); 4 new leadfield
  tests. (2) `JansenRitNetwork` multi-layer unit bug — fixed at the network level by
  `x = u.get_magnitude(layer(x))` between layers (single-layer output untouched); the strict
  `xfail` became a passing `test_network_multi_layer_runs`.
- **`np.asarray` on a unit-bearing `Quantity` RAISES** (refuses to silently drop the unit) — the
  pytree round-trip test compares the already-unit-stripped `jax.tree.leaves`, not the State values.
- **Gotcha confirmed again:** drvfs file-tool persistence is bumpy — `git rm` on a worktree-modified
  file needs `-f`; and a package dir shadows a same-named stale `.py` at import time, so delete the
  old module *before* trusting the smoke test.

### 2026-06-18 · goal-06 orchestration-simulator
- **Headline — the in-package run loop now exists.** New `brainmass/simulator.py::Simulator` (100%
  cov) and `brainmass/objectives.py` (100% cov) are the first orchestration layer (sub-project A1,
  06→07→08). Both exported from `__init__` under a `# Orchestration layer` block (`Simulator` as a
  class, `objectives` as a submodule). Suite **337→384 passed** (+47: 27 simulator + 20 objectives),
  coverage **98.17%→98.42%**. No model API touched — the layer is purely additive.
- **`Simulator(model, dt=None).run(duration, *, inputs, monitors, transient, sample_every,
  batch_size, init_states, jit)` → plain `dict`** of `{monitor_name: stacked_traj, 'ts': time_axis}`.
  It only *composes* the existing idiom (`environ.context(dt=)` → `init_all_states` →
  `for_loop(step_run, arange(n))` → collect `.value`); reimplements nothing. Returns a dict (not a
  Bunch) so it's a clean pytree through jit/grad/vmap. `sim.model` stays exposed.
- **Ordering that matters inside `run` (cost us 16 failures, then fixed):** `dt` must be live in
  `environ` *before* `init_all_states` because delay buffers size from `delay/dt` at init. BUT the
  `monitors` validation (`hasattr(model, name)`) must run *after* init — state attrs like `x`/`V`
  don't exist until `init_state`. So: resolve dt/steps/transient/sample_every and build `get_inputs`
  (input-length check, dt-independent) BEFORE the `environ.context(dt=dt)` block; do `init_all_states`
  then `_build_measure` INSIDE it. 07/08 closures that drive `.run` must respect this if they
  pre-validate monitors.
- **`monitors` polymorphism:** `None`→`{'output': update_return}`; `list[str]`→`getattr(model,name).value`
  each; `callable(model)`→`{'output': fn(model)}` (derived observables like `lambda m: m.eeg()`);
  `dict{name: str|callable}`. **`transient`** accepts a `Quantity` *or* an int step count;
  **`sample_every=k`** is a post-hoc stride `[n_tr::k]` (NOT a substep loop — `for_loop` materialises
  the whole trajectory anyway, so slicing is exact and ~free) and generalises the `JansenRitTR` TR
  loop. **`ts` convention:** `ts[k]` = time at the *end* of the k-th recorded step (state is
  post-`update`), i.e. `((arange(n_steps)+1)*dt)[n_tr::k]` → `ts[0]==dt`, `ts[-1]==duration`.
- **array `inputs` must be jnp, not numpy** — `arr[i]` with a traced loop counter `i` fails on a
  numpy array. Cast with `jnp.asarray` *unless* it's a `u.Quantity` (leave Quantities alone; casting
  drops units). Then `get_inputs(i,t) = (arr[i],)`.
- **Edge cases pinned (simulator_test.py, 27 tests):** dt unset→ValueError; non-integer
  `duration/dt`→floor+`warnings.warn`; zero/negative duration→raise; `transient>=duration`→raise;
  bad monitor name→raise (list *and* dict forms); wrong `inputs` length→raise; `sample_every<1`→raise.
  Behavioural proofs: single-node run **bit-identical** to a hand-written `for_loop` (seeded,
  `np.array_equal`); jit vs no-jit identical; units propagate (Montbrio `r`→Hz); `batch_size=4`→leading
  batch axis `(T,4,N)`; `init_states=False` continues from current state; **`jax.grad` through
  `run` ≈ FD** (rel 2e-2); whole-brain WilsonCowan+DiffusiveCoupling net matches a manual loop.
- **`objectives.py` = builders, not metrics.** Each fn returns a small `callable(pred, target)` that's
  jit/grad/vmap-safe and unit-aware, wrapping `braintools.metric` with zero re-implemented maths:
  `timeseries_rmse` (unit-CHECKED subtraction — mV vs Hz raises), `fc_corr(as_loss=)`,
  `fc_rmse`, `cosine_sim(as_loss=, epsilon=)`, `fcd(window_size=, step_size=, as_loss=)`, and
  `combine(*(weight, objective))`→weighted sum. **`fcd` finally surfaces
  `functional_connectivity_dynamics`** (was zero call sites — the long-flagged debt). `as_loss=True`
  returns `1-score` (minimise); default returns the raw score (maximise). `_mag = u.get_magnitude`
  strips units only where the metric is scale-invariant (FC/cosine), never on the RMSE subtraction.
  This is the loss surface the goal-08 fitter will minimise; goal-07 (connectome `Network`) feeds
  its trajectory straight into `Simulator.run` → `objectives.*`.
- **Doctest gate decision (carrying goal-02's lesson forward):** wrote all examples as **bare `>>>`
  napoleon blocks**, NOT `.. code-block:: python`. Only bare blocks (+`.. testcode::`) are executed
  by `sphinx.ext.doctest` (`doctest_test_doctest_blocks="default"`); a `code-block` with `>>>` is
  cosmetic and silently untested. CLAUDE.md asks for `.. code-block::` styling but the goal-02 ledger
  + the live doctest gate win here — examples that aren't gated rot. `combine`'s example uses two
  RMSE terms `(2.0+0.5)*1.0=2.5`, NOT an FC term: FC of an all-zeros (zero-variance) signal is `nan`.
- **Example migrated:** `examples/000-getting-started.ipynb` Step 5 now uses
  `brainmass.Simulator(node, dt).run(n_steps*dt, monitors=['x','y'], init_states=False)` instead of a
  hand-written `step_run`/`for_loop` — bit-identical trajectory (verified). Kept `n_steps`/`indices`
  defined in that cell because the downstream `visualization` cell and the `simulate_hope_model`
  exploration cell read the **global** `indices`/`t_ms`; dropping them would break later cells.
  (Notebooks are `nb_execution_mode="off"` — not run in CI — so verified the equivalence in a script.)

### 2026-06-18 · goal-07 orchestration-network
- **`brainmass.Network(node, *, conn, distance=None, speed=None, coupling='diffusive',
  coupled_var, k=1.0, delay_init=Uniform(0.,0.05), self_connection=False, noise=None)`** collapses the
  whole-brain wiring copy-pasted ~8× across examples into one `brainstate.nn.Module`. `update(*node_inputs)`
  is exactly the hand-written idiom: `current = self.coupling(); [current += self.noise()];
  return self.node(current, *node_inputs)`. Exposes `self.node`, `self.coupling`, `self.n_node`.
  The coupling current is always the node's **first** positional input, so it drives the coupled var
  (WilsonCowan `rE`, Hopf `x`, FHN `V` — all take that var's input first).
- **Parameterization:** `conn` plain array → diagonal zeroed (unless `self_connection=True`); accepts
  1-D flattened `(N*N,)` or 2-D `(N,N)`; a `Param`/`LaplacianConnParam` is passed through untouched (owns
  its structure, may train). `delays = distance/speed`: plain quotient → assumed **ms**, a unit-carrying
  quotient is used as-is; self-delay (diagonal) always zeroed; `distance is None or speed is None` →
  zero-delay (instantaneous) path. `coupling=` dispatches `DiffusiveCoupling` / `AdditiveCoupling` /
  `AdditiveCoupling(LaplacianConnParam(conn))`. `noise=` is a **network-level** process added to the
  coupling current — distinct from per-region node noise (`WilsonCowanStep(noise_E=...)`), which stays on
  the node. `k` may be a trainable `Param` (grad verified AD≈FD).
- **Bit-for-bit reasoning (for goal-08 validation):** the delay-buffer shape `(max_delay_steps, N)` is
  **independent** of whether `(delay, idx)` are passed flattened (examples/100) or as `(N,N)` matrices
  (Network). So a 2-D Network reproduces the flattened hand-wired example **exactly** given same N, same
  max delay, same seeded `Uniform` draw → proven via `np.array_equal` (seed-before-build + seed-before-run).
- **`coupled_var` validation:** prefetch does NOT validate eagerly (bad var → late AttributeError). Pre-init
  `node.states()` is empty; the node's state *names* are `{n for n,v in vars(node).items() if isinstance(v,
  brainstate.State)}`. Validate in `Network.init_state` at **`@call_order(1)`** (runs before the coupling's
  `call_order(2)` `init_maybe_prefetch`) → clean `ValueError` listing available states.
- **⚠ Fitter (goal-08) MUST NOT batch delayed coupling via `init_all_states(batch_size)`.** brainstate's
  `prefetch_delay` mis-orders the delayed `(N,N)` source read under a batch axis → `(N,N,B,N)` instead of
  `(B,N,N)` (rank-4, not salvageable by reshape); the target read batches fine `(B,N)`. Batch the fit by
  **vmap-over-parameters**, not batched initial states. `network_test.py::batched_delayed_coupling_unsupported`
  pins this with `pytest.raises`. (`Simulator.run(batch_size=)` is fine for a *single* node, not a delayed net.)
- **Merge gotcha:** PR #39 renamed `docs/api → docs/apis` but left ~16 inbound `:doc:`../api/...`` refs
  dangling; git rename-detection auto-applied my `orchestration.rst` edits to the new path (no conflict),
  but the stale refs needed a manual sweep. Build artifacts (`docs/_static/css|js`, `docs/_build`,
  `docs/api/generated`) are regenerated — don't commit them.
- **braintools deprecation sweep (library-wide, folded into this PR):** `braintools.init.ZeroInit()` warns
  **even bare** (it forwards `unit=u.UNITLESS`, which is `not None`, into the deprecated `Constant(unit=)`
  path) — so the warning is not limited to explicit `unit=` sites. Behaviour-preserving migration (verified
  output-identical): `ZeroInit()`→`Constant(0.0)`, `ZeroInit(unit=X)`→`Constant(0.0 * X)`,
  `Uniform(a,b,unit=X)`→`Uniform(a*X, b*X)`. Swept 229 sites across 33 files (every model's default
  initializer + tests + docstrings + notebooks); `pytest -W error::DeprecationWarning` now passes (was 232
  warnings / 26 collection errors). **Lesson:** when scoping a deprecation fix, reproduce the *actual*
  warnings and trace every trigger — don't estimate from the obvious parameter pattern alone.

### 2026-06-18 · goal-08 orchestration-fitter

- **`brainmass.Fitter(model, optimizer=None, *, loss_fn=None, objective=None, predict=None,
  backend='grad', callbacks=None, transient=None, search_space=None)`** + **`FitResult`** collapse the 7
  copy-pasted `ModelFitting` classes (in the untracked `tutorials/` scratch dir) into one object.
  `.fit(target=None, n_steps=100, *, verbose=False) -> FitResult`. "Write the objective once; swap
  backends." `FitResult` carries `backend, best_loss, best_params{name:constrained}, history, n_steps,
  prediction, optimizer, raw, model`.
- **`optimizer` is backend-polymorphic** (the key unification trick): for `backend='grad'` it is a
  `braintools.optim.OptaxOptimizer` instance (default `Adam(lr=1e-2)`); for `'nevergrad'`/`'scipy'` it is
  an **options dict** (`{'method':'DE','n_sample':8}` / `{'method':'L-BFGS-B'}`) or a method-name `str` or
  `None` — the real `NevergradOptimizer`/`ScipyOptimizer` is built *inside* `.fit()` (it needs the loss +
  bounds at construction) and exposed as `fitter.optimizer`. `n_steps` = optax steps / nevergrad
  generations / scipy random-restarts.
- **Two loss paths:** `loss_fn(model)->(scalar, aux)` is the *entire* loss (user owns reg); else
  `objective(predict(model)[transient:], target) + model.reg_loss()` with `objective` defaulting to
  `objectives.timeseries_rmse()`. `predict(model)->prediction` is a user-built `Simulator` closure (keeps
  the constructor free of duration/monitors). `transient` is an **int sample count** trimmed off axis 0.
- **grad backend = the canonical invariant, verbatim:** `weights = model.states(ParamState)` (empty →
  clear `ValueError`); `optimizer.register_trainable_weights(weights)`; jitted closure
  `grad(f_loss, weights, has_aux=True, return_value=True)` → `optimizer.step(grads)`; `f_loss` runs inside
  `with model.param_precompute():`. Track a **best-seen checkpoint** (snapshot `{path: state.value}` on
  improvement, restore at end) so the model ends at its best, not its last, point. Step-for-step it
  reproduces a hand-rolled `ModelFitting` loop (`fitter_test::test_grad_matches_handrolled` asserts
  `allclose` history).
- **search_space derivation from Param transforms** (`_finite_box`): only transforms with a *finite*
  interval auto-derive a box — `SigmoidT/TanhT/SoftsignT/ScaledSigmoidT` via `(.lower, .lower+.width)`,
  `ClipT` via `(.lower,.upper)`. Half-line/unbounded (`ReluT`(`.lower_bound`!), `SoftplusT/LogT/ExpT`
  (`.lower` only), `NegSoftplusT` (`.lower` stores upper), `PositiveT/NegativeT/IdentityT`) → **no box** →
  raise naming the param unless an explicit `search_space[name]` is given (which always overrides). Bounds
  are built at the param's value shape (`jnp.full(shape, lo/hi)`; scalars stay scalar) so candidate shape
  matches `set_value`.
- **⚠ optimizer gotcha for goals 09-10 — derivative-free candidates MUST be looped, not vmapped.**
  `brainstate.transform.vmap` over `Param.set_value` raises `BatchAxisError` (the written ParamState isn't
  in `out_states`) — same family as goal-07's batched-delay limitation. So the Nevergrad `batched_loss_fun`
  evaluates its `n_sample` candidates with a **Python `for` loop** (set_value→predict→objective, stacked);
  fine for the few scalar params derivative-free search suits, but it does NOT scale to matrix params.
  `Param.set_value(constrained)` stores `t.inverse(constrained)` and `value()` returns `t.forward` — they
  round-trip, so the search box lives in constrained/physical space and the transform constraint is
  redundant-but-harmless there.
- **Backend availability:** `scipy` is a hard `braintools` dep (always present); **`nevergrad` is only a
  braintools `testing` extra** — added to `requirements-dev.txt` so CI installs it, and the nevergrad tests
  `pytest.importorskip('nevergrad')` for robustness. SciPy *gradient* methods (`L-BFGS-B`) DO differentiate
  through `set_value` (verified); a `timeseries_rmse` loss is V-shaped (`|k−k*|·const`) with a sqrt kink at
  the optimum, so a single restart can stall — use a few restarts (`n_steps≈3`) for derivative-free robustness.
- **callbacks/early-stop** are a **grad-backend feature** (the Fitter owns that loop): each
  `callback({'step','loss','best_loss','model'})` fires per step and returning `True` stops early.
  Derivative-free backends run `.minimize(n_iter)` to completion (their loop is internal), so callbacks
  there fire once at the end — documented, not silently dropped.
- **Migration:** rewrote `docs/tutorials/parameter_fitting.rst` around `Fitter` (was schematic
  hand-rolled snippets + a nonexistent `brainmass.ArrayParam`) — all blocks are runnable `.. testcode::`
  (3-region drive so `fc_corr`/`cosine_sim` are non-degenerate). Added `Fitter`/`FitResult` autoclass to
  `docs/apis/orchestration.rst`. Real nodes (`HopfStep`, `WilsonCowanStep`) expose **no** trainable
  `ParamState` by default — fitting requires wrapping the target parameter in a `Param(..., t=...)` (as the
  `tutorials/` models do, e.g. `JansenRitTR(k=Param(5.5, t=ReluT(0.5)))`).
- **Local coverage is unmeasurable in this WSL env:** `pytest --cov` (and `COVERAGE_CORE=sysmon|pytrace`)
  **SIGABRT** with JAX imported — reproduces on pre-existing test files too, so it is environmental, not a
  code bug. CI (Linux) measures the `--cov-fail-under=90` gate fine. Audit branch coverage by reading the
  code + tests instead; full suite is **439 passed** (403 + 36 new) locally without `--cov`.

### 2026-06-19 · goal-09 tvb-models-complex

- **Three complex TVB mean-field models added** (sub-project F), all on the goal-05 unified
  `NeuralMassDynamics` base, dimensionless-state + `×1/u.ms` FHN convention (RHS methods return
  `value / u.ms` so `exp_euler` with dt-in-ms is consistent), `exp_euler` default with
  `braintools.quad.ode_*` alternatives via `_solve_step`:
  - **`CoombesByrneStep`** `(r, v)` — Coombes & Byrne (2019), next-gen exact mean field of θ/QIF.
    `g = k·π·r`; `dr = (Δ/π + 2vr − gr)/ms`; `dv = (v² − (πr)² + η + (v_syn−v)g)/ms`. Defaults
    `Δ=1, η=2, k=1, v_syn=−4`; init `r=0.1, v=0`. `update` returns `r`.
  - **`LarterBreakspearStep`** `(V, W, Z)` — Breakspear, Terry & Friston (2003), conductance-based
    Morris-Lecar mean field with Na/K/Ca gating. 32 params set in a loop via
    `setattr(self, name, Param.init(value, varshape))` from a verbatim `DEFAULT_PARAMS` dict; long-range
    coupling enters `V` only (`c_inst=0`). `update` returns `V`.
  - **`EpileptorStep`** 6-var `(x1, y1, z, x2, y2, g)` — Jirsa et al. (2014). Slow permittivity `z`
    drives seizure onset/offset; `x0` = epileptogenicity. `lfp() = x2 − x1` is what `update` returns.
- **⚠ Coombes-Byrne ↔ MPR for goal-10:** `CoombesByrneStep` with **`k = 0`** is *exactly*
  `MontbrioPazoRoxinStep` with `J = 0` (drop the synaptic-conductance term `(v_syn−v)·g` and the `−gr`
  term; both reduce to `dr=(Δ/π+2vr)/ms`, `dv=(v²−(πr)²+η)/ms`). `coombes_byrne_test::test_cb_reduces_to_mpr`
  asserts this (corr≥0.999 over an rk4 rollout). CB's `g=k·π·r` is the conductance generalization;
  MPR's `J` is a *current* coupling, CB's `k`/`v_syn` a *conductance* coupling — same QIF backbone,
  different synapse model. Mirror state names lowercase (`r`,`v`) so the two read as siblings.
- **Validation = embedded-reference-oracle pattern (proven across all 3).** TVB/tvboptim are NOT
  importable here, so each `*_test.py` embeds a verbatim transcription of tvboptim's JAX RHS as
  `_<model>_reference` + a `_P` defaults dict, and validates four ways: (1) **RHS-fidelity** — our
  `d<var>` vs the oracle at random states, **`rtol=1e-6`** (the literature-faithfulness gate; the real
  test); (2) **rk4 trajectory regression** corr≥0.99 vs an embedded `jax.lax.scan` rk4 of the oracle;
  (3) **always-on published-feature fallback**; (4) **gated live** `@requires_tvb`/`@requires_tvboptim`
  (`pytest.mark.skipif(not find_spec(...))` — *skipif objects assigned to a local, applied as `@requires_tvb`*;
  **no** `[tool.pytest] markers` registration needed, no `PytestUnknownMarkWarning`). RHS-fidelity tests
  must drive **non-trivial coupling gains** (Epileptor parametrized over `modification∈{0,1}` AND
  `Kvf=1.5, Kf=0.7, Ks=2.0`) or the coupling-port terms go unverified.
- **Feature fallbacks must be empirically probed, not guessed (regime boundaries are sharp):**
  - LB: `d_V=0.57` → limit cycle (std≈0.09); `d_V=0.50` → genuine fixed point (std≈1.5e-8). `d_V=0.40`
    looked like a fixed point but was actually **divergent** (std≈2.77) — probe before asserting.
  - Epileptor: `x0=−1.6` → epileptogenic (windowed-std detects ictal bursts + transitions, `x1.max()>0`)
    vs `x0=−2.4` → healthy (`x1.max()<0`). Needs **~30000 steps** to span onset+offset.
- **⚠ Epileptor noise routing — additive `add` term, NOT the coupling port.** First pass routed noise
  through `x1_inp`/`x2_inp`, but those are scaled by `Kvf`/`Kf` (=0 by default) → noise was *inert*. Fix:
  `dx1(self, x1, y1, z, x2, x1_inp, add=0.0)` returns `(det + add)/u.ms` where `det` includes the
  `Kvf·x1_inp` coupling-gain term; noise is injected via the **direct additive `add`** (TVB-faithful
  additive noise on the fast variables). Same for `dx2` with `Kf`. Lesson: trace where a stochastic term
  actually lands through default-zero gains before trusting a noise test.
- **dt/integrator sensitivity:** CB & LB are fine at `dt=0.1 ms` with `exp_euler`. Epileptor's slow `z`
  (rate `r≈4e-4`) is stiff-ish but *stable* — 20000 steps at default `r` keep `z` bounded ~2–5; no
  small-dt requirement, the slow manifold is attracting. rk4 oracle used only for the regression test.
- **Coverage finding refines goal-08's:** the SIGABRT is **submodule-scoped** — `--cov=brainmass.epileptor`
  aborts, but **`--cov=brainmass` (full-package) works locally**. Confirmed **100.00%** (stmts + branch)
  on all three new modules; full suite **471 passed, 3 skipped** (live-TVB gated). The only
  `sphinx -b doctest` failures (4) are pre-existing in goal-08's `tutorials/parameter_fitting.rst`
  (non-deterministic `fitter.fit()` print with no expected output) — out of scope here; left untouched.
  Our docstring `Examples` (21 `>>>` lines across the 3 classes) pass via `DocTestFinder` (0 failures).
  NB `doctest.testmod(mod)`/`pytest --doctest-modules` reported 0 — they mis-filter re-exported classes by
  `__module__`; `DocTestFinder().find(cls)` is the reliable local doctest check.

### 2026-06-19 · goal-10 tvb-models-canonical

- **Closed model parity with tvboptim** by adding four canonical/E-I nodes on the unified
  `NeuralMassDynamics` base, same embedded-reference-oracle pattern as goal-09:
  - **`Generic2dOscillatorStep`** `(V, W)` — Sanz-Leon et al. (2015). 12 params via setattr loop;
    cubic-nullcline `dV = d·tau·(−f·V³+e·V²+g·V+α·W+γ·(I+V_inp))`, `dW = (d/tau)·(a+b·V+c·V²−β·W)`.
    Validated against the analytic bistable fixed point (`d=1`, `b=0`, `c=−5`, `e=3`, `f=1` →
    `V*=(−1−√5)/2≈−1.618`, `W*=V*³−3V*²`) + excitable input-integration shift; **regime
    boundaries are sharp** — the excitable `I=2.0` probe failed to settle at sped-up `d=1.0`
    (Hopf at high drive), so the test uses `d=0.2` with `I=0` vs `I=3`.
  - **`WongWangExcInhStep`** `(S_E, S_I)` — Deco et al. (2014). **Distinct from `WongWangStep`**
    (the reduced *decision-making* `(S1,S2)` model): resolved by reading `wong_wang.py` first.
    Naive TVB transfer `H(x)=(a·x−b)/(1−exp(−d·(a·x−b)))` has a removable singularity at
    `a·x−b=0`, but the physiological regime sits at `x_e_scaled≈−7` throughout, so the naive
    form is safe; resting FP ≈ 3 Hz excitatory rate matches Deco 2014. `H_e()`/`H_i()` expose rates.
  - **`LorenzStep`** `(x, y, z)` — Lorenz (1963). Chaos test asserts positive Lyapunov / exponential
    divergence. **float32 gotcha:** perturbation `1e-8` is *below* float32 epsilon (~1e-7) so both
    trajectories were bit-identical (separation 0.0) — use `eps=1e-3`. Docstring notes
    `dt=0.01*u.ms` ≡ classic dimensionless `dt=0.01`.
  - **`LinearStep`** `(x)` — TVB `Linear`, `dx/dt=γx+coupling`. Validated against closed-form
    `x(t)=x0·exp(γt)` (exp-Euler is *exact* for linear systems) + forced FP `−c/γ`. Distinct from
    the two-population `ThresholdLinearStep`. **Simulator gotcha:** constant input must be a
    callable `inputs=lambda i,t: c` — a bare `(c,)` is read as an array of shape `(n_steps,…)`.
- **⚠ User mid-session override: "remove TVB TVBOPTIM comparison tests."** Deleted the gated
  `@requires_tvb`/`@requires_tvboptim` placeholder tests **and** the gating machinery
  (`importlib.util.find_spec`, `_HAS_TVB`, `requires_*`) from goal-09's
  `coombes_byrne_test.py`/`epileptor_test.py`/`larter_breakspear_test.py` too (this worktree merges
  to main). The embedded-oracle RHS-fidelity tests STAY (they compare against transcriptions, not
  live TVB). So goal-09's lesson above about `@requires_tvb` is now historical — those gates are gone.
- **goal-10b (folded into this PR per user): converted the narrative RST guides → Jupyter notebooks.**
  Scope (user-chosen): `tutorials/*` + `developer/*` + `faq` only; `apis/*` stay autosummary RST;
  index hubs stay. Wrote a tuned `rst2nb.py` (dev/, gitignored): top-level `.. testcode::` /
  `.. code-block:: python` → code cells; prose/`note`/`math`/`list-table`/`tab-set` → MyST
  colon-fence/`$$`/`{role}` markdown; bash/text code-blocks → fenced markdown; prepend a
  `remove-cell`-tagged setup cell (`%matplotlib inline` + the old `doctest_global_setup` imports + `dt`).
  Toctrees/`:doc:` resolve by **extension-less docname**, so deleting `quickstart.rst` + adding
  `quickstart.ipynb` auto-resolved — **no toctree edits needed**.
- **⚠ Notebook execution must use the worktree brainmass, not the kernel's installed one.**
  `jupyter nbconvert --execute` runs the `python3` kernel with cwd = the notebook's dir, which
  imported a *stale* site-packages brainmass (pre-goal-06: no `objectives`/`Network`). Fix:
  `PYTHONPATH=<worktree-root> jupyter nbconvert …`. (sphinx `-b doctest` is unaffected — conf.py puts
  the package on `sys.path`.)
- **Executing the notebooks surfaced — and we then repaired — real pre-existing doc drift** the
  doctest gate never caught (those examples were `.. code-block::`, not `.. testcode::`, so were never
  run). First pass: 7/14 executed clean. The other 7 had genuine API drift / missing-`.npy` /
  intentional fragments. **User chose "repair drift now"** → fixed 5 in parallel (one subagent per
  notebook, *execution as the correctness gate* — each must `nbconvert --execute` to exit 0): updated
  `WilsonCowanStep` kwargs (`c_EE`→`wEE/wEI/…`), the prefetch `DiffusiveCoupling`/`AdditiveCoupling`
  API, `KuramotoNetwork(omega,K)` (not `omega_mean/std`), lead-field shapes+units, and the BOLD driver
  (`WongWangExcInhStep`/`WilsonCowanStep`, since `WongWangStep` is the reduced `S1/S2` decision model
  with no `S_E`); replaced `.npy` loads with seeded inline matrices. **12/14 now execute** with embedded
  outputs; `faq` (intentional error-demo fragments) and `developer/architecture`
  (`brainstate.ShardingMode`, absent in the pinned version) stay unexecuted.
- **⚠ Subagent-repair gotchas worth keeping:** (a) `jupyter nbconvert --execute` uses the kernel's
  *installed* brainmass unless you pass `PYTHONPATH=<worktree>` — stale package = phantom failures.
  (b) A *deterministic* Wong-Wang network collapses all regions to one fixed point → zero-variance BOLD
  → `nan` FC; fix with per-region `OUProcess` noise **and** an explicit `brainstate.random.seed(...)`.
  (c) `DiffusiveCoupling` needs the source prefetch to yield `(N, N)` (each target reads every source),
  so `prefetch_delay` with a tiled `src_idx` is required — a plain `prefetch('x')` raises a shape error.
  (d) `KuramotoNetwork.omega` is **dimensionless** (passing `u.Hz` raises `UnitMismatchError` — it
  divides by `u.ms` internally). (e) lead-field `L` shape is `(in_size, out_size)` carrying unit
  `sensor_unit / dipole_unit`.
- **Doctest drive-by:** before the conversion, fixed goal-06's 4 `parameter_fitting.rst` `.. testcode::`
  failures with `.. testoutput:: :options: +ELLIPSIS` (stochastic `fitted k = …`); that file is now a
  notebook so the fix lives only in history, but the technique (ELLIPSIS testoutput for non-deterministic
  prints) is the right tool when a `testcode` block must print.
- **Parity table (brainmass ⊇ tvboptim node library):** FitzHugh-Nagumo, Hopf, Stuart-Landau,
  Van der Pol, Wilson-Cowan family, Jansen-Rit, Montbrió-Pazó-Roxin, Kuramoto (canonical, pre-existing) ·
  Epileptor, Larter-Breakspear, Coombes-Byrne (complex, goal-09) · Generic2dOscillator,
  WongWang-ExcInh, Lorenz, Linear (canonical/E-I, goal-10). brainmass additionally has HORN + forward
  models (BOLD/EEG/MEG lead fields) with no tvboptim equivalent.

### 2026-06-19 · goal-11 coupling-parity

- **Closed coupling parity with tvboptim** — three nonlinear `function + Module` pairs added to
  `coupling.py`, mirroring the existing kernel+thin-Module pattern exactly (no new base class):
  - **`sigmoidal_coupling` / `SigmoidalCoupling`** (TVB *Sigmoidal*, **post**-nonlinearity):
    `c = k·σ(slope·(a·Σ_j w_ij x_j + b − midpoint))`, defaults `k=1,a=1,b=0,slope=1,midpoint=0`.
  - **`hyperbolic_tangent_coupling` / `HyperbolicTangentCoupling`** (TVB *HyperbolicTangent*, **post**):
    `c = k·tanh(scale·Σ_j w_ij x_j)`, defaults `k=0.5,scale=2.0`; saturates to `±k`.
  - **`sigmoidal_jansen_rit_coupling` / `SigmoidalJansenRitCoupling`** (TVB *SigmoidalJansenRit*,
    **pre**-nonlinearity): `c = k·Σ_j w_ij·σ_JR(x_j)`, `σ_JR(x)=cmin+(cmax−cmin)/(1+e^{r(midpoint−x)})`,
    defaults `k=1,cmin=0,cmax=0.005,midpoint=6.0,r=0.56`. Source `x_j` is whatever the caller
    prefetches (e.g. JR `y1−y2`).
  - **Linear bias** added to `additive_coupling`/`AdditiveCoupling`: `c = k·Σ_j w_ij x_j + b`
    (default `b=0.0`).
- **`G ≡ k` convention:** brainmass keeps `k` for the global strength; tvboptim/TVB call it `G`.
  Documented `G ≡ k` in every new docstring + the docs page note. New trainable scalars
  (`slope`,`midpoint`,`cmin`,`cmax`,`r`,`a`,`b`) all go through `Param.init`.
- **pre- vs post-nonlinearity split is the key design axis:** post-forms (sigmoid/tanh) apply the
  saturating fn to the *summed* network input → output carries `k`'s unit; the JR pre-form applies
  σ_JR to each *source* before weighting → output carries `k·conn`'s unit (σ is dimensionless).
- **Back-compat win — `Param.init(float)` returns a `Const`, NOT a trainable `ParamState`.** So the
  new default `b=0.0` (and all new default scalars) add **zero** trainable params — goal-08 Fitter's
  `states(ParamState)` is unaffected. Trainable only when the caller passes an explicit `Param(...)`.
  Confirmed: `m.states(ParamState)` lists only explicit `Param`s, never `Param.init(float)` Consts.
- **`Quantity(nA) + 0.0` does NOT raise in brainunit** (treats `+0.0` as the additive identity),
  so `additive_coupling`'s new `+ b` is **bit-for-bit** identical for both dimensionless and
  unit-carrying coupling at the default — no gating needed. Verified `np.array_equal` (b=0 == old).
- **Nonlinear kernels strip units for the transcendental** via `u.get_magnitude` on the σ/tanh
  argument (`u.math.tanh`/`sigmoid` raise on a unitful Quantity) — same house style as Jansen-Rit.
  Output units come back via the `k *` (post) / `k * (conn * σ)` (pre) multiply.
- **conn handling for the new (additive-style, no target `y`) couplings:** shared helper
  `_coupling_conn_xmat` accepts 2-D `(N_out,N_in)` **or** a 1-D **square**-flattened `(N*N,)` conn
  (infers `n=isqrt(size)`; non-square 1-D raises — there's no `y` to disambiguate `N_out`, unlike
  `diffusive_coupling`). `additive_coupling` kept its 2-D-only body untouched (back-compat); only
  appended `+ b`.
- **Network integration:** extended `Network`'s `coupling=` dispatch with `'sigmoidal'`/`'tanh'`/
  `'sigmoidal_jansen_rit'` (each plugs the delayed `(N,N)` `src` read into the new Module exactly
  like `AdditiveCoupling`). 2-node `HopfStep` nets run clean under `Simulator` (instantaneous + with
  `distance/speed` delays). ⚠ `Network.__init__` calls `prefetch_delay`, which reads `dt` from
  `environ` **at construction time** — tests must `environ.set(dt=…)` *before* building the Network
  (the docstring example relies on `doctest_global_setup`'s `dt`).
- **No tvboptim reference-regression test was added.** tvboptim isn't importable in this env, and the
  goal-10 user override ("remove TVB/TVBOPTIM comparison tests") removed the gating machinery from
  earlier goals; tvboptim's coupling math is plain-array identical to ours, so the **closed-form
  oracles are the gate** (sigmoid@0 input `=k·σ(−slope·midpoint)`; tanh→`±k`; σ_JR@midpoint
  `=k·Σw·(cmin+cmax)/2`, far-limits → `cmin`/`cmax`). If a future goal wants live regression: match
  weights/params and `allclose` (tvboptim `Bunch(G,a,b,slope,midpoint)` / `Bunch(G,cmin,cmax,
  midpoint,r)` map 1:1 to our kwargs, `G→k`).
- **Coverage:** `coupling.py` 95.93% / `network.py` 100% (the new helper + 3 kernels + 3 Modules +
  bias are **100%** covered; the 6 residual `coupling.py` misses are *pre-existing* diffusive/additive
  guards + laplacian's unknown-normalize raise). Full suite **552 → 558 passed** (24 new coupling
  tests incl. closed-form, gradient-FD on k/slope/midpoint/b, units, batched, 1-D-flattened-conn &
  flattened-x, delayed-source-via-Network, constrained-`Param` round-trip). Local doctest via
  `DocTestFinder().find()` = 24/24 (the reliable check; `--doctest-modules` mis-filters re-exports).
- **Deferred (candidate follow-ons):** `KuramotoCoupling` (phase coupling `k·Σ w_ij·sin(θ_j−θ_i)`)
  and the hierarchical `SubspaceCoupling` (per-subpopulation coupling) — both out of scope here;
  observation models are goal-12. No new `Delayed*` twins (delays stay upstream via `prefetch_delay`).

### 2026-06-19 · goal-12 observation-parity

- **Closed the observation gap vs tvboptim** (sub-project F, observation slice). New module
  `brainmass/observation.py` (+ `observation_test.py`, **100 %** cov): `HRFKernel` base + 4 closed-form
  HRF kernels (`FirstOrderVolterraHRFKernel`, `GammaHRFKernel`, `DoubleExponentialHRFKernel`,
  `MixtureOfGammasHRFKernel`), the convolution `HRFBold` observation, and the standalone
  `TemporalAverage` monitor. Extended `objectives.py` (**100 %** cov) with the FCD-*distribution*
  objectives. All exported top-level (`# Observation models` block in `__init__`); existing
  `BOLDSignal` (Balloon ODE) and EEG/MEG lead fields **untouched**. Suite **586 → 607 passed**
  (50 observation + 14 objectives + 2 import_compat; no regressions).
- **Convolution-vs-Balloon BOLD guidance (documented in both classes + a forward.rst table):**
  `HRFBold` = a single linear convolution `BOLD = k1·V0·(h * y_ds − 1)` (downsample→convolve→decimate)
  — fast, simple, **differentiable in its scalar params** → the choice for *fitting*. `BOLDSignal` =
  the 4-state Balloon-Windkessel ODE → the choice when biophysical realism matters. Empirically they
  **track each other**: best-lag correlation **0.98** on a slow drive (zero-lag only 0.73 — the two
  hemodynamics have different intrinsic latencies, so compare BOLD waveforms over a ±few-TR lag, the
  standard way; a naive zero-lag corr under-measures agreement).
- **Delegation audit (what braintools already had vs what we added):**
  - **Reused** `braintools.metric.functional_connectivity_dynamics` (FCD matrix) — same call as goal-08's `fcd`.
  - **Delegated to jax:** KDE → `jax.scipy.stats.gaussian_kde`; FFT conv → `jax.scipy.signal.fftconvolve`;
    direct conv → `jnp.convolve`. **braintools.metric has no KDE/KS/Wasserstein/conv.**
  - **Hand-implemented (the genuine gap, tiny cumsum-based):** `ks_distance(p,q)` = `max|cdf_p−cdf_q|`,
    `wasserstein_1d(p,q,x)` = `Σ|cdf_p−cdf_q|·dx`. Both normalise cumsums internally so they accept
    densities *or* histograms.
- **`TemporalAverage` ↔ `Simulator`:** kept it a **standalone thin observation** (`TemporalAverage(period)
  (signal, dt)`), NOT a `Simulator` change. It is the *averaging* (anti-aliased) complement to the
  point-decimation `Simulator(sample_every=k)` — apply it as a post-transform on a run trajectory; it
  is also the downsampler `HRFBold` uses internally. `y[k]=mean(signal[k·w:(k+1)·w])`, `w=round(period/dt)`,
  `n_win=T//w`, trailing partial window dropped; `round(period/dt)<1` raises. Preserves units.
- **FCD-distribution objective API** (the standard FCD fitting target = the *distribution* of FCD
  off-diagonal values, not the matrix correlation `fcd` already had): primitives `fcd_distribution`
  (KDE of the upper-triangle on a `[-0.99,0.99]` grid, normalised to ∫=1), `ks_distance`, `wasserstein_1d`;
  builders `fcd_wasserstein(window_size,step_size,midpoints,bw_method,n_diag)` and `fcd_ks(...)` →
  `loss(prediction,target)` (0 on identity). **KS vs Wasserstein differentiability:** KS is a `max`
  (non-smooth — its grad is the indicator at the argmax) → use `fcd_ks` for *evaluation/reporting*;
  `wasserstein_1d` is smooth/grad-friendly → `fcd_wasserstein` is the recommended *fitting* loss
  (grad verified finite). Degenerate (constant, zero-variance) input → singular KDE → **`nan`** distance
  (documented + tested).
- **Validation = embedded-reference-oracle, NOT `requires_tvboptim` gating.** The goal-12 prompt asked
  for a `requires_tvboptim` marker, but goal-10's standing user override deleted all tvboptim test-gating
  machinery (this branch merges to main). Reconciled by following the goal-09/10 pattern: kernels checked
  against **verbatim embedded transcriptions** of tvboptim's closed forms (always-on, no live import) +
  analytic props (`h(0)=0`, peak=`a`, real underdamped `ω`); `HRFBold` against an **independent numpy +
  `scipy.signal.fftconvolve`** reproduction (genuinely different libs, not a copy of the impl). scipy
  cross-checks are always-on (scipy is a hard braintools dep, added to requirements-dev explicitly):
  `ks_distance` is **exact** vs `scipy.stats.ks_2samp` (build per-value histograms on the pooled-sorted
  unique grid so the normalised cumsums ARE the empirical CDFs scipy maximises); `wasserstein_1d` ≈
  `scipy.stats.wasserstein_distance` on a fine grid (rtol 2e-2, it's a Riemann sum of ∫|ΔCDF|dx).
- **Gotchas:** (a) `BOLDSignal` is dimensionless and needs a **unitless** `dt` in `environ` (its RK2 does
  `t+dt`; a `Quantity` dt → `ms + 1` UnitMismatch) — drive it with `dt=0.01` while `HRFBold` gets the
  matching real-time `10*u.ms`. (b) HRF kernels take `t` as a time `Quantity` (→ seconds) or a plain array
  (assumed **ms**, the TVB convention); they return a **dimensionless** array; grad flows w.r.t. continuous
  params (`tau_s`/`tau`/`λ`) when the kernel is built *inside* the differentiated fn (`n`/`duration` are
  int-cast → non-diff, fine). (c) docstring Examples use `.. code-block:: python` + `>>>` (CLAUDE.md +
  goal-09/10 model convention, validated by `DocTestFinder`); RST doc Examples use bare `>>>` (sphinx
  `-b doctest` gate, 159 tests / 0 fail). No sphinx-doctest in CI (deploy-docs runs `-b html` only).
- **brainmass now meets/exceeds tvboptim's observation set:** convolution BOLD + HRF-kernel family +
  `TemporalAverage` + FCD-distribution objectives (this goal) · Balloon-Windkessel ODE BOLD + FC/FCD-matrix
  objectives + `Simulator(sample_every=)` subsampling (pre-existing). **EEG/MEG lead-field forwards remain a
  brainmass-only advantage** (tvboptim has none); sEEG/iEEG noted as future (tvboptim lacks them too).

### 2026-06-19 · goal-13a docs-ia-skeleton

- **Headline — the entire new `docs/` navigation skeleton is in place** (Diátaxis quadrants +
  persona on-ramps + a data-driven showcase hub), so the ~10 content goals (13b–13m) each only
  **replace their own placeholder files in place and never edit a shared toctree.** This is the
  *nav-skeleton-first / replace-only-your-placeholders* rule — honor it to keep the parallel
  content PRs conflict-free (spec §5 "Nav-conflict rule"). Docs-only goal: **zero `brainmass/`
  source diff** (`git diff --stat origin/main -- brainmass/` empty); `import brainmass` unchanged.
- **Final `docs/` tree (top-level dirs):** `getting_started/` (installation, quickstart,
  key_concepts, learning_paths) · `tutorials/` (01–08) · `howto/` (7 recipes) · `concepts/` (5) ·
  `data_driven/` (index hub + roadmap.md) · `gallery/` (`model_zoo/` 17 demos + `case_studies/` 5) ·
  `reference/` (was `apis/`, +observation/datasets/viz placeholders) · `developer/` (unchanged) ·
  `faq` · `changelog` · `roadmap.md` (internal, excluded). Landing `index.rst` gained the
  conservative brainmass-vs-TVB-vs-neurolib comparison table + 3 persona on-ramp cards.
- **Placeholder convention (every net-new file):** valid nbformat-v4 `.ipynb`, **one markdown
  cell** = `# <Human Title>` + `> **Placeholder** — authored in goal-13<x>. Reserves its
  navigation slot.`, **no code cells**, stable `kernelspec=python3` metadata so myst-nb parses it.
  Stamped out with `dev/make_placeholders.py` (gitignored — the whole `dev/` tree is, per CLAUDE.md
  rule 7; helper not committed). 38 placeholders + 3 `reference/*.rst` stubs.
- **File→goal ownership map (who replaces what):** 13b api-ergonomics (datasets/viz/list_models +
  `reference/{datasets,viz,observation}` real pages) · 13c getting_started (installation, quickstart,
  key_concepts) · 13d tutorials 01–04 · 13e tutorials 05–08 · 13f all 7 howto · 13g all 5 concepts ·
  13h `gallery/model_zoo/*` (the 17 demos; closes the 9-model gap) · 13i `gallery/case_studies/*` (5) ·
  13j reference-polish · 13k developer expansion · 13l `data_driven/index` narrative · 13m integration.
- **Relocations were `git mv` (history preserved):** `apis/`→`reference/`; the 8 existing tutorial
  notebooks split into `getting_started/`, `howto/`, and renumbered `tutorials/0N_*`. **Gotcha:**
  moving a notebook changes how its *bare-docname* `{doc}` / `:doc:` cross-refs resolve (MyST/Sphinx
  resolve a bare name **relative to the referencing doc's directory**). Splitting siblings across
  dirs broke 24 in-notebook refs (`{doc}`quickstart``, `building_networks`, …). **Fix = rewrite to
  absolute docnames with a leading slash** (`{doc}`/getting_started/quickstart``) — location-
  independent, survives any future move. Did the whole apis→reference + moved-notebook + examples→
  gallery rewrite with small Python regex scripts over `*.rst/*.ipynb/*.md`.
- **conf.py / build gotchas:**
  - **examples→gallery rewiring:** removed `shutil.copytree('../examples', './examples')` and the
    `../examples` lexer wiring; the gallery now lives in-tree under `docs/gallery/`. Kept the
    `changelog.md` copy, `nb_execution_mode="off"`, doctest setup, theme, extensions. The
    `fix_ipython2_lexer_in_notebooks` helper **globs one directory non-recursively**, so it's now
    called once per gallery sub-dir (`gallery/model_zoo`, `gallery/case_studies`); goal-13h drops
    the real notebooks there.
  - **autosummary `generated/` is a build artifact — do NOT commit it.** The reorg created
    `docs/reference/generated/`; the existing `.gitignore` only ignored `docs/apis/generated`
    (and `*/generated` does NOT match a 3-segment path — gitignore `*` never crosses `/`). Added
    `docs/reference/generated` + `**/jupyter_execute` to `.gitignore`. A stray `git add -A` had
    staged 72 generated stubs — unstage + add the ignore rule.
  - **`docs/roadmap.md` is the internal program roadmap, not public nav** — it was a pre-existing
    toctree-orphan WARNING on `origin/main`; added it to `exclude_patterns` (the *pillar* roadmap
    `data_driven/roadmap.md` IS published, linked from the `data_driven/index` placeholder via a
    MyST `{toctree}`).
- **Build evidence:** `sphinx-build -b html . _build/html` → **EXIT 0, "build succeeded", 0
  unknown-document (broken `:doc:`) warnings, 0 toctree-orphan warnings.** The residual ~86–97
  warnings (`duplicate object description`, docutils field-list/footnote formatting,
  forward-reference, one myst deprecation) and the 11 docutils `ERROR:` lines are **all pre-existing
  and emitted from `brainmass/` source docstrings + autosummary** — byte-for-byte the same set as a
  baseline `origin/main` build (90 warnings / 11 ERRORs there too; goal-13a actually *reduced* the
  broken-ref + orphan count by fixing `roadmap.md`, `../examples/examples/index`, and a dangling
  `reference/utilities.rst → 'types'` ref). Cleaning those source-docstring warnings is goal-13j/13m
  territory, not this docs-only goal.
