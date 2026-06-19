# brainmass Improvement Roadmap

Improvement program derived from a comparative audit of **brainmass** against the
competitor framework **tvboptim** (Charité / The Virtual Brain group). The work is
decomposed into five independent sub-projects, one per quality dimension the audit
examined: functionality, usability, maintainability, testing, and documentation.

Each sub-project below is a self-contained TODO list. They can be shipped in any order,
though the suggested sequencing (see *Dependencies* at the end) builds the safety net
(C, D, B) before the large feature (A).

> **Note on references.** File/line citations point at `main` at the time of the audit
> (2026-06-18) and may drift as the code changes. Treat them as starting pointers, not
> guarantees.

## Overview

| # | Sub-project | Dimension | Effort | Status |
|---|-------------|-----------|--------|--------|
| **A** | Fitting & orchestration layer | functionality + usability | Large | In design |
| **B** | Testing rigor | testing | Medium | Planned |
| **C** | CI / tooling & release hygiene | maintainability | Small–Medium | Planned |
| **D** | Documentation integrity | documentation | Small–Medium | Done (goal-02) |
| **E** | Code correctness & structure | maintainability | Medium | Planned |
| **F** | Model parity (TVB mean-field) | functionality | Medium | Done (goal-09 + goal-10) |

### What tvboptim does that we can learn from

- **End-to-end gradient fitting as a first-class, in-package workflow** — a reusable
  optimizer (`OptaxOptimizer.run`), transparent `Parameter` wrapping with constraint
  transforms, and `Space`/`Axes` for parallel parameter sweeps. *brainmass already has
  the constrained-parameter primitive (`Param(t=Transform, reg=, fit=True)` from
  brainstate) but no fitter, `Simulator`, or `Network` builder that consumes it.*
- **Differentiable observation/metric models in-package** (BOLD, FC, FCD incl. a
  differentiable Wasserstein FCD surrogate) plus a **local identifiability / sloppiness
  analysis**.
- **Tests that validate gradients & physics** — AD-vs-finite-difference checks,
  analytic-oracle tests, convergence-order tests, and reference-regression against
  upstream TVB (corr ≥ 0.99).
- **Docs whose code cells execute in CI** (so they can't rot), a one-source
  `.qmd → tested-HTML + clean-Colab .ipynb` pipeline, OIDC PyPI publishing, ruff in CI.

---

## A. Fitting & orchestration layer

**Dimension:** functionality + usability · **Effort:** Large · **Status:** In design

The headline gap. brainmass ships only *models*; every study re-hand-rolls the
simulation loop, the whole-brain wiring, and the fit loop. The `ModelFitting` class is
copy-pasted ~7× across `tutorials/` and the whole-brain `Network` class ~8× across
`examples/`. This sub-project provides three thin, brainstate-native wrappers plus
objective helpers that consume brainstate/braintools rather than reimplement them.

**MVP (the "core trio"):**

- [ ] **`Simulator`** — wrap the canonical run pattern (`environ.set(dt)` →
      `init_all_states` → `brainstate.transform.for_loop(step_run, indices)` → collect
      `.value`). Responsibilities: `duration → n_steps`; monitor selection (which
      states / derived observables to record); transient/warm-up discard; optional jit;
      batching via `brainstate.nn.Map` / `vmap`; unit-aware structured output. Generalize
      `JansenRitTR.update` (`jansen_rit.py:580`) as the dt-substep-per-output-sample (TR
      downsampling) precedent.
- [ ] **`Network` / connectome builder** — from `node + conn (Cmat) + distance (Dmat) +
      speed`: zero the SC diagonal, compute `dist/speed` delays (ms), build neighbour
      indices, wire `DiffusiveCoupling`/`AdditiveCoupling` via `prefetch_delay`/`prefetch`
      on a chosen `coupled_var`. Parameters: coupling type, coupled variable name, `k`,
      `speed`, per-node noise. Replaces the ~8 copy-pasted `Network` classes in
      `examples/` and `docs/examples/`.
- [ ] **`Fitter`** — one reusable fitter unifying the three optimizer backends behind a
      single `.fit()`:
  - [ ] gradient / optax (`brainstate.transform.grad(loss, weights, has_aux=True,
        return_value=True)` → `optimizer.step(grads)`),
  - [ ] gradient-free Nevergrad (`bounds={'k': [lo, hi]}`, `initialize()` + `minimize()`),
  - [ ] Scipy (`bounds=[(lo, hi)]`, single-array loss).
  - [ ] Consume `Param(fit=True)` for trainable discovery, `reg_loss()` for
        regularization, and `param_precompute()` for in-loop performance.
  - [ ] Opt-in history, callbacks, best-seen checkpoint, transient handling.
  - [ ] Consolidate the 7 `ModelFitting` copies (`tutorials/301`, `302`,
        `303-EEG-fitting`, `303-EEG-JansenRit-fitting`, `303-MEG-EEG-fitting`,
        `303-MEG-EEG-fitting-fc`, `EEG-HORN-fitting`).
- [ ] **Objective helpers** — composable loss builders over `braintools.metric`:
      `fc_corr`, `fc_rmse`, `timeseries_rmse`, `cosine_sim`, and **`fcd`** (surface
      `braintools.metric.functional_connectivity_dynamics`, currently unused anywhere).
- [ ] **Tests** — gradient flows through `Simulator`; `Network` output matches the
      hand-wired baseline; `Fitter` reduces loss on a toy problem across all three
      backends; seeded RNG throughout.
- [ ] **Docs** — a tutorial migrating one hand-rolled notebook to the new API; API
      reference pages.

**Roadmap follow-ons (deferred from MVP):**

- [ ] Parameter sweeps (`Space`-like → tidy DataFrame), generalizing
      `examples/200-parameter-exploration`.
- [ ] Local identifiability / sloppiness analysis (Fisher/Hessian eigenspectrum).

---

## B. Testing rigor

**Dimension:** testing · **Effort:** Medium · **Status:** Planned

The headline "differentiable" feature has **zero** gradient tests; stochastic tests are
unseeded; whole modules are uncovered. Borrow tvboptim's validation patterns.

- [ ] Add **gradient / differentiability tests** (`value_and_grad`,
      `jax.test_util.check_grads`, AD-vs-finite-difference) for representative models.
- [ ] **Seed all stochastic tests** (`brainstate.random` seed / `PRNGKey`); tighten the
      loose tolerances that currently mask broken noise processes.
- [ ] **Analytic-oracle / reference-regression tests** — linear systems with known
      fixed points / Lyapunov exponents; where feasible, published reference values
      (mirror tvboptim's TVB corr ≥ 0.99 regression registry).
- [ ] Add **`conftest.py`** with shared fixtures: `dt` setup/teardown (fix the global
      `environ` dt leakage across tests), seeded RNG, a small connectome.
- [ ] Use **`@pytest.mark.parametrize`** for model / variant / integrator sweeps
      (currently never used).
- [ ] **Cover untested modules:** `horn.py` (810 LOC, 0 coverage), `kuramoto.py`,
      `utils.py`, `leadfield.py`, `_xy_model.py`; the 6 untested Wilson–Cowan variants;
      the 4 functional coupling helpers; `JansenRitTR`.
- [ ] **Remove/replace dead tests:** the 5 commented-out `test_stability_long_simulation`
      in `wilson_cowan_test.py`; commented assertions at `jansen_rit_test.py:267`;
      convert `plot=True` / `plt.show()` demo "tests" (which assert nothing) into real
      assertions or `skip`/`mark` them.
- [ ] Add **`pytest-cov` + coverage gate** (`--cov-fail-under`, target >90% per
      CLAUDE.md) and wire it into CI.
- [ ] Reconcile `docs/developer/testing.rst` (describes a nonexistent `tests/` layout)
      with reality (co-located `*_test.py`, run via `pytest brainmass/`).

---

## C. CI / tooling & release hygiene

**Dimension:** maintainability · **Effort:** Small–Medium · **Status:** Planned

CI runs only `pytest`; there is no lint/type/coverage/doctest, the publish workflow will
fail on a real release, and the declared minimum Python is never tested.

- [ ] Add **ruff** (lint + format check) to CI; add a `[tool.ruff]` config; retire the
      stale flake8-only `.pre-commit-config.yaml` and run pre-commit in CI.
- [ ] Add a **mypy / pyright** type-check job (start lenient); add a `[tool.mypy]` config.
- [ ] Add **coverage** reporting + gate to CI (ties to B).
- [ ] Add **doctest** execution in CI (CLAUDE.md mandates doctestable `Examples`).
- [ ] **Fix `Publish.yml`** — the tag-vs-version step reads
      `Path('brainmass/_version/__init__.py')`, which does not exist (the real file is
      `brainmass/_version.py`); a real `release` event raises `FileNotFoundError`. Fix the
      path and test the guard.
- [ ] **Test the declared minimum Python (3.11).** Today the pipeline spans four
      versions, none being the floor: CI 3.13, docs build 3.14, publish 3.13, RTD 3.12.
      Unify, and add a small version matrix that includes 3.11.
- [ ] Adopt **PyPI Trusted Publishing (OIDC)** instead of a stored token (mirror
      tvboptim's `publish-pypi.yml`).
- [ ] **Remove repo cruft:** the untracked `nul` file; `dev/replacement.py` (one-off);
      the stray `[tool.flit.module]` block in `pyproject.toml`.
- [ ] Add **`dev` / `doc` extras** to `pyproject.toml` — docs tell users
      `pip install -e .[dev,doc]` but those extras do not exist.

---

## D. Documentation integrity

**Dimension:** documentation · **Effort:** Small–Medium · **Status:** Done (goal-02);
CI gating deferred to sub-project C (goal-03)

After the `*Step` rename, the prose and tutorials referenced symbols that no longer
existed — the Quickstart example did not even import — and nothing caught it because
notebooks/tutorials are not executed at build time. Addressed by goal-02.

- [x] **Fixed the pervasive API drift.** Replaced the legacy ``*Oscillator`` / ``*Model``
      class aliases (Hopf, Wilson–Cowan, Wong–Wang, FitzHugh–Nagumo, Stuart–Landau, Van der
      Pol, and the Montbrió–Pazó–Roxin mean-field model) with their canonical ``*Step`` names
      across prose and tutorials; grep-verified that none remain (goal-02).
- [x] **Made docs self-checking (content + config).** Enabled `sphinx.ext.doctest`: every
      docstring `>>>` example is doctest-clean and the Quickstart / units guide run as
      executable `.. testcode::` blocks. Notebooks stay non-executed (large embedded outputs;
      HCP data dependency) and are documented as such. *Wiring `make doctest` into CI is
      sub-project C (goal-03).*
- [x] **Fixed `docs/conf.py`** — merged the duplicated `autodoc_default_options` (the second
      silently disabled `members` / `special-members` / `undoc-members`) into one (goal-02).
- [x] Added a **usage code snippet to `README.md`** and fixed the stale citation version
      (`0.0.4` → `0.0.6`) (goal-02).
- [x] Added the **`0.0.6` changelog entry** (goal-02).
- [x] **Fixed `developer/contributing.rst` contradictions** — NumPy-doc (not Google-style),
      co-located `*_test.py` (not a `tests/` dir), and the dev/doc install instructions
      (goal-02).
- [ ] Evaluate the **one-source → tested-HTML + clean-Colab** tutorial pipeline from
      tvboptim.

---

## E. Code correctness & structure

**Dimension:** maintainability · **Effort:** Medium · **Status:** Planned

Concrete latent bugs (all currently untested) plus structural cleanup. Per CLAUDE.md
rule 4, each bug fix starts with a reproducing test.

**Bugs — reproduce, then fix:**

- [ ] `coupling.py:485-503` — `LaplacianConnParam.normalize`: method/attribute name
      collision. `precompute=self.normalize` captures the bound method, then
      `self.normalize = normalize` reassigns it to the string arg, and the `normalize()`
      method calls `self.normalize` internally → broken when exercised.
- [ ] `jansen_rit.py:657-667` — `JansenRitTR.update` reads `inp_M_tr` / `inp_E_tr` /
      `inp_I_tr` before they are assigned (works only by Python late-binding accident).
- [ ] `noise.py:107` — `BrownianNoise.update` increment algebra cancels
      (`sigma / dt_sqrt * dt_sqrt`) to `sigma` instead of the intended `sigma*sqrt(dt)`.
- [ ] `noise.py:178-195` — `BlueNoise` / `VioletNoise` `beta` signs are inconsistent
      with the `1/f^β` spectral math and with their own docstrings.
- [ ] `forward_model.py` — `LeadFieldModel` docstring/shape drift (`(T,R)→(T,M)` vs
      `(R,T)→(M,T)`); `_sample_noise` references a nonexistent `self._noise_cov_q`.
- [ ] `jansen_rit.py:168` vs `:215` — `s_max` default mismatch (docstring 2.5 Hz vs
      code 5.0 Hz).

**Structure:**

- [ ] **Unify base classes.** `XY_Oscillator` covers only 3 of ~6 two-variable
      oscillators; `FitzHughNagumoStep`, `WongWangStep`, `KuramotoNetwork`, and
      Wilson–Cowan each reinvent the `Dynamics` contract. Introduce a coherent shared
      base (the 0.0.5 changelog's "unified `Dynamics` base class" is aspirational).
- [ ] **Split the dumping grounds.** `jansen_rit.py` (1027 LOC) and `horn.py` (810 LOC)
      hold 8 / 6 internal network/layer classes that are undocumented, untested, and
      partly experimental.
- [ ] Resolve the **duplicate `AdditiveConn` / `DelayedAdditiveConn`** class definitions
      present in both `horn.py` and `jansen_rit.py`.
- [ ] Consolidate the **two overlapping observation abstractions**:
      `forward_model.LeadFieldModel` vs `leadfield.LeadfieldReadout`.
- [ ] Convert **Google-style docstrings to NumPy-doc** in `utils.py`, `leadfield.py`,
      `wong_wang.py` (per CLAUDE.md).
- [ ] Replace the **`brainmass` self-import** in `jansen_rit.py` (it calls
      `brainmass.delay_index` instead of the local `delay_index` it already imports).

---

## F. Model parity (TVB mean-field)

**Dimension:** functionality · **Effort:** Medium · **Status:** Done (goal-09 + goal-10)

brainmass ships a solid canonical-oscillator library (FitzHugh–Nagumo, Hopf, Stuart–Landau,
Van der Pol, Wilson–Cowan family, Jansen–Rit, Montbrió–Pazó–Roxin), but lacked several of the
*complex* mean-field models that The Virtual Brain / tvboptim use for whole-brain studies of
seizures, conductance-based dynamics, and next-generation (exact) mean fields, as well as a
handful of canonical TVB nodes. This sub-project closes that gap, porting each model
literature-faithfully onto the unified ``NeuralMassDynamics`` base (goal-05) with equations +
numbered references in the docstring, ``exp_euler``-default integration with
``braintools.quad`` alternatives, and **reference-regression tests**: each model's RHS is
checked against an embedded, per-unit-time transcription of the upstream tvboptim equation
(``rtol`` 1e-6), its short-horizon RK4 trajectory against that same reference (correlation
≥ 0.99), plus always-on dynamical-feature assertions (analytic fixed points, bistability,
chaos / positive Lyapunov sign, AD-vs-FD gradients).

**Complex mean-field models (goal-09):**

- [x] **`EpileptorStep`** — Jirsa, Stacey, Quilichini, Ivanov & Bernard (2014). Six state
      variables ``(x1, y1, z, x2, y2, g)`` with coupled fast/slow subsystems and a slow
      permittivity variable ``z`` that autonomously drives seizure onset and offset; ``x0``
      sets epileptogenicity. Validated via the seizure onset/offset feature fallback and an
      RHS-fidelity oracle (with non-zero ``Kvf``/``Kf``/``Ks`` coupling gains).
- [x] **`LarterBreakspearStep`** — Larter & Breakspear; Breakspear, Terry & Friston (2003).
      Conductance-based ``(V, W, Z)`` mean field with Na/K/Ca channel gating; the pyramidal
      threshold variance ``d_V`` selects the dynamical regime. Validated via the
      limit-cycle-vs-fixed-point feature fallback and an RHS-fidelity oracle.
- [x] **`CoombesByrneStep`** — Coombes & Byrne (2019), a next-generation neural mass (exact
      mean field of θ/QIF networks) in ``(r, v)`` form. Reduces to
      :class:`~brainmass.MontbrioPazoRoxinStep` (with ``J = 0``) when the synaptic
      conductance scale ``k = 0`` — validated by that agreement plus an RHS-fidelity oracle.

**Canonical / simple models (goal-10):**

- [x] **`Generic2dOscillatorStep`** — Sanz-Leon, Knock, Spiegler & Jirsa (2015). TVB's flexible
      planar oscillator ``(V, W)`` whose polynomial-nullcline coefficients
      ``(a, b, c, d, e, f, g, alpha, beta, gamma, I, tau)`` select the regime (excitable,
      bistable, Morris-Lecar-like). Validated against an analytic cubic-nullcline fixed point,
      excitable input integration, and an RHS-fidelity oracle across three regimes.
- [x] **`WongWangExcInhStep`** — Deco, Ponce-Alvarez, Hagmann, Romani, Mantini & Corbetta
      (2014). Two-population excitatory–inhibitory reduced Wong–Wang mean field ``(S_E, S_I)``,
      the resting-state workhorse; distinct from the existing single-population decision-making
      :class:`~brainmass.WongWangStep`. Validated via the resting fixed point (≈3 Hz excitatory
      rate), monostability from low/high initial conditions, and an RHS-fidelity oracle.
- [x] **`LorenzStep`** — Lorenz (1963). Chaotic ``(x, y, z)`` test fixture; validated by a
      short-horizon RK4 trajectory regression, a positive-Lyapunov / exponential-divergence
      check, and an RHS-fidelity oracle.
- [x] **`LinearStep`** — TVB ``Linear`` node, ``dx/dt = gamma*x + coupling`` (distinct from the
      two-population :class:`~brainmass.ThresholdLinearStep`). Validated against the closed-form
      ``x(t) = x0 * exp(gamma t)`` decay (exp-Euler is exact for linear systems), the forced
      fixed point ``-c/gamma``, and an RHS-fidelity oracle.

With goal-10 merged, brainmass is at **model parity** with the tvboptim node library across
the complex mean-field and canonical-oscillator families (see the comparison table in
``docs/apis/models.rst``).

---

## Dependencies & suggested sequencing

```
C (CI gates) ─┐
D (doc fixes) ─┼─► makes the repo trustworthy ──► A (orchestration layer)
B (test rigor)─┘
E (correctness) ── independent; can run any time, ideally before A touches coupling/JR
```

- **A** is the highest-value, highest-effort item and benefits from **C + D + B** being
  in place first (working docs, CI gates, gradient tests) so it is built on solid ground.
- **E** is independent and low-risk; doing it before **A** is helpful because **A**'s
  `Network` builder will touch `coupling.py` / Jansen–Rit, where several bugs live.
- **C** and **D** are the cheapest, highest-ROI "stop the bleeding" fixes (the Quickstart
  won't import; the publish workflow will fail a real release).
