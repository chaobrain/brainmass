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
