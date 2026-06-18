# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Integrator (ODE solver) sweeps and convergence-order tests.

Every model exposes a ``method`` argument selecting the integration scheme
(``exp_euler`` by default, or any ``braintools.quad.ode_<method>_step``). These
tests sweep ``method`` across representative models -- exercising the non-default
``derivative`` code paths that were otherwise untested -- and verify the schemes
are not just finite but *correct*, using analytic oracles and measured
convergence orders.
"""

import numpy as np
import pytest

import braintools
import brainstate
import brainunit as u

import brainmass

METHODS = ["exp_euler", "rk2", "rk4"]


def _simulate_last(model, drive, read, n_steps):
    """Advance ``model`` for ``n_steps`` and return the final read value (NumPy)."""
    def step(i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            drive(model)
            return read(model)

    xs = brainstate.transform.for_loop(step, np.arange(n_steps))
    return np.asarray(u.get_magnitude(xs[-1])).reshape(-1)


# Build/drive/read recipes per model, parametrized by integration ``method``.
SWEEP = {
    "hopf": dict(
        build=lambda method: brainmass.HopfStep(2, a=-0.2, method=method),
        drive=lambda m: m.update(0.1, 0.0),
        read=lambda m: m.x.value,
    ),
    "fhn": dict(
        build=lambda method: brainmass.FitzHughNagumoStep(
            2, method=method,
            init_V=braintools.init.Constant(0.1), init_w=braintools.init.ZeroInit()),
        drive=lambda m: m.update(V_inp=0.3),
        read=lambda m: m.V.value,
    ),
    "montbrio": dict(
        build=lambda method: brainmass.MontbrioPazoRoxinStep(
            2, method=method,
            init_r=braintools.init.ZeroInit(unit=u.Hz), init_v=braintools.init.ZeroInit()),
        drive=lambda m: m.update(v_inp=0.5),
        read=lambda m: m.r.value,
    ),
    "wilson_cowan": dict(
        build=lambda method: brainmass.WilsonCowanStep(2, method=method),
        drive=lambda m: m.update(rE_inp=1.0),
        read=lambda m: m.rE.value,
    ),
}


_MONTBRIO_RK_XFAIL = pytest.mark.xfail(
    reason="LATENT BUG (deferred to goal-05): qif.py MontbrioPazoRoxinStep.update "
           "non-exp_euler branch calls ode_<method>_step((r, v), ...) without passing "
           "self.derivative as the first (function) argument -- cf. the correct calls in "
           "_xy_model.py and fhn.py. So method in {rk2, rk4} raises 'input function should "
           "be callable'.",
    strict=True,
    raises=AssertionError,
)


def _sweep_cases():
    """Build (model, method) cases; mark the known-broken montbrio rk2/rk4 xfail."""
    cases = []
    for model_name in SWEEP:
        for method in METHODS:
            marks = (_MONTBRIO_RK_XFAIL
                     if model_name == "montbrio" and method in ("rk2", "rk4") else ())
            cases.append(pytest.param(model_name, method, marks=marks))
    return cases


@pytest.mark.parametrize("model_name, method", _sweep_cases())
def test_method_runs_and_is_finite(model_name, method, dt):
    """Each integrator produces a finite trajectory of the right shape."""
    spec = SWEEP[model_name]
    model = spec["build"](method)
    brainstate.nn.init_all_states(model)
    last = _simulate_last(model, spec["drive"], spec["read"], n_steps=50)
    assert last.shape == (2,)
    assert np.all(np.isfinite(last))


def _hopf_linear_decay_final(method, dt_ms, a=-1.0, x0=1.0, T_ms=5.0):
    """Integrate the *linear* Hopf reduction ``dx/dt = a*x`` and return ``x(T)``.

    With ``w=0`` and ``beta=0`` the Hopf node reduces to scalar linear decay,
    whose exact solution is ``x(T) = x0 * exp(a * T)``. This is an analytic
    oracle for the integrators.
    """
    brainstate.environ.set(dt=dt_ms * u.ms)
    m = brainmass.HopfStep(
        1, a=a, w=0.0, beta=0.0, method=method,
        init_x=braintools.init.Constant(x0), init_y=braintools.init.ZeroInit())
    brainstate.nn.init_all_states(m)
    n = int(round(T_ms / dt_ms))
    out = _simulate_last(m, lambda mm: mm.update(0.0, 0.0), lambda mm: mm.x.value, n)
    return float(out[0])


def test_exp_euler_is_exact_on_linear_decay():
    """Exponential Euler integrates a linear ODE exactly (analytic oracle).

    For ``dx/dt = a*x`` the exponential-Euler update is the exact propagator, so
    the numerical solution must match ``exp(a*T)`` to round-off. ``rk4`` should be
    near-exact, while the lower-order ``rk2`` carries a visibly larger error.
    """
    exact = np.exp(-5.0)  # a = -1, T = 5 ms
    err_exp = abs(_hopf_linear_decay_final("exp_euler", 0.1) - exact)
    err_rk4 = abs(_hopf_linear_decay_final("rk4", 0.1) - exact)
    err_rk2 = abs(_hopf_linear_decay_final("rk2", 0.1) - exact)

    assert err_exp < 1e-5, f"exp_euler not exact on linear decay: err={err_exp:.2e}"
    assert err_rk4 < 1e-4, f"rk4 inaccurate on linear decay: err={err_rk4:.2e}"
    assert err_rk2 > err_rk4, "rk2 should be less accurate than rk4 here"


def _fhn_final(method, dt_ms, T_ms=4.0):
    """Final ``V`` of a smooth nonlinear FitzHugh-Nagumo run (for order estimation)."""
    brainstate.environ.set(dt=dt_ms * u.ms)
    m = brainmass.FitzHughNagumoStep(
        1, method=method,
        init_V=braintools.init.Constant(0.5), init_w=braintools.init.Constant(0.1))
    brainstate.nn.init_all_states(m)
    n = int(round(T_ms / dt_ms))
    out = _simulate_last(m, lambda mm: mm.update(V_inp=0.3), lambda mm: mm.V.value, n)
    return float(out[0])


@pytest.mark.parametrize(
    "method, lo, hi",
    [("exp_euler", 0.8, 1.3), ("rk2", 1.7, 2.3)],
)
def test_convergence_order(method, lo, hi):
    """Global error shrinks at the scheme's theoretical order under dt-halving.

    Measured against a fine ``rk4`` reference on a smooth nonlinear system,
    ``exp_euler`` converges at first order and ``rk2`` at second order. (``rk4``
    is excluded: at float32 it reaches the round-off floor before the asymptotic
    order can be measured -- its accuracy is checked separately below.)
    """
    ref = _fhn_final("rk4", 0.001)
    errs = [abs(_fhn_final(method, dt) - ref) for dt in (0.2, 0.1, 0.05)]
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    mean_order = float(np.mean(orders))
    assert lo <= mean_order <= hi, (
        f"{method}: measured order {mean_order:.2f} not in [{lo}, {hi}] (errs={errs})"
    )


def test_rk4_more_accurate_than_rk2():
    """At a fixed step, the higher-order scheme is strictly more accurate."""
    ref = _fhn_final("rk4", 0.001)
    err_rk2 = abs(_fhn_final("rk2", 0.2) - ref)
    err_rk4 = abs(_fhn_final("rk4", 0.2) - ref)
    assert err_rk4 < err_rk2
    assert err_rk4 < 1e-4
