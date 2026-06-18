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

"""Differentiability / gradient tests for the representative neural-mass models.

``brainmass`` advertises itself as a *differentiable* whole-brain modelling
library, yet historically shipped **zero** gradient tests. This module installs
that safety net: for each representative model it asserts that a gradient flows
through a short :func:`brainstate.transform.for_loop` simulation with respect to
a trainable :class:`brainstate.nn.Param`, cross-checks reverse-mode autodiff
against finite differences, and runs :func:`jax.test_util.check_grads`.

Notes
-----
Two facts make these tests work and are easy to get wrong:

1. Model parameters are **non-trainable by default** -- ``Param.init(scalar, shape)``
   wraps the value in a ``Const`` (``fit=False``). To obtain a trainable
   ``ParamState`` the model must be constructed with ``Param(value, fit=True)``;
   only then does ``model.states(brainstate.ParamState)`` find the parameter.
2. The loss function must be **deterministic**. Several models default to random
   (``Uniform``) state initializers; without seeding, each evaluation of the loss
   starts from a different state and the finite-difference / ``check_grads``
   cross-checks fail spuriously. Every loss below seeds the RNG first.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import brainstate
import brainunit as u
from brainstate.nn import Param

import brainmass

# A short horizon keeps the tests fast while still exercising recurrence.
N_STEP = 40
IN_SIZE = 2
SEED = 0


def _scalar_loss(model, drive, read):
    """Run a short deterministic simulation and return a scalar loss.

    Parameters
    ----------
    model : brainstate.nn.Dynamics
        An already-initialised model instance.
    drive : callable
        ``drive(model)`` advances the model by one step (calls ``model.update``).
    read : callable
        ``read(model)`` returns the state quantity to accumulate into the loss.

    Returns
    -------
    jax.Array
        ``sum(read(model) ** 2)`` over the trajectory, with physical units
        stripped so the result is a plain scalar.
    """
    def step(i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            drive(model)
            return read(model)

    xs = brainstate.transform.for_loop(step, np.arange(N_STEP))
    return jnp.sum(u.get_magnitude(xs) ** 2)


# Each spec builds a model whose named parameter is trainable, drives it, and
# reads a state. ``p0`` is the operating point for the gradient checks.
# ``strong`` flags models whose gradient is robustly far from zero (used for the
# extra "gradient is non-trivial" assertion); models with legitimately tiny
# gradients are still validated by the AD-vs-FD and check_grads cross-checks.
def _hopf(p):
    return brainmass.HopfStep(IN_SIZE, a=Param(p, fit=True))


def _wilson_cowan(p):
    return brainmass.WilsonCowanStep(IN_SIZE, wEE=Param(p, fit=True))


def _jansen_rit(p):
    return brainmass.JansenRitStep(IN_SIZE, C=Param(p, fit=True))


def _montbrio(p):
    # Differentiate w.r.t. the excitability ``eta`` while driving the membrane
    # potential, so the population is in an active regime where the parameter
    # genuinely influences the firing rate (the gradient w.r.t. ``J`` at rest is
    # ~1e-6 and dominated by round-off, which makes a meaningful check impossible).
    return brainmass.MontbrioPazoRoxinStep(IN_SIZE, eta=Param(p, fit=True))


def _fhn(p):
    return brainmass.FitzHughNagumoStep(IN_SIZE, alpha=Param(p, fit=True))


SPECS = {
    "hopf": dict(build=_hopf, drive=lambda m: m.update(0.1, 0.0),
                 read=lambda m: m.x.value, p0=0.3, strong=True),
    "wilson_cowan": dict(build=_wilson_cowan, drive=lambda m: m.update(rE_inp=1.0),
                         read=lambda m: m.rE.value, p0=10.0, strong=True),
    "jansen_rit": dict(build=_jansen_rit, drive=lambda m: m.update(M_inp=3.0 * u.mV),
                       read=lambda m: m.E.value - m.I.value, p0=135.0, strong=False),
    "montbrio": dict(build=_montbrio, drive=lambda m: m.update(v_inp=1.0),
                     read=lambda m: m.r.value, p0=-5.0, strong=True),
    "fhn": dict(build=_fhn, drive=lambda m: m.update(V_inp=0.5),
                read=lambda m: m.V.value, p0=3.0, strong=True),
}
MODELS = list(SPECS)


def _pure_loss(spec):
    """Return a pure ``f(pval) -> scalar`` for a model spec.

    The returned function seeds the RNG, builds the model with a trainable
    parameter set to ``pval``, initialises its states, and returns the scalar
    simulation loss. Seeding makes ``f`` a genuine (deterministic) function of
    ``pval`` even for models with random default initializers.
    """
    def f(pval):
        brainstate.random.seed(SEED)
        model = spec["build"](pval)
        brainstate.nn.init_all_states(model)
        return _scalar_loss(model, spec["drive"], spec["read"])

    return f


@pytest.fixture(autouse=True)
def _dt(dt):
    """Install a known time step for every gradient test (uses the ``dt`` fixture)."""
    return dt


@pytest.mark.parametrize("name", MODELS)
def test_grad_flows_through_for_loop(name):
    """A gradient w.r.t. a trainable ``Param`` flows through a ``for_loop`` sim.

    Uses the ``brainstate.transform.grad`` idiom over ``model.states(ParamState)``
    -- the same path a user fitting the model would take.
    """
    spec = SPECS[name]
    brainstate.random.seed(SEED)
    model = spec["build"](jnp.asarray(spec["p0"], dtype=float))
    brainstate.nn.init_all_states(model)

    weights = model.states(brainstate.ParamState)
    assert len(weights) == 1, "exactly one trainable Param expected"

    def loss():
        return _scalar_loss(model, spec["drive"], spec["read"])

    grads, value = brainstate.transform.grad(loss, weights, return_value=True)()

    leaves = jax.tree.leaves(grads)
    assert leaves, "grad tree is empty -- no ParamState was differentiated"
    for g in leaves:
        assert jnp.all(jnp.isfinite(g)), f"{name}: non-finite gradient"
    assert jnp.isfinite(value)


@pytest.mark.parametrize("name", MODELS)
def test_value_and_grad(name):
    """``jax.value_and_grad`` over the pure loss returns a finite value+grad."""
    spec = SPECS[name]
    f = _pure_loss(spec)
    value, grad = jax.value_and_grad(f)(jnp.asarray(spec["p0"], dtype=float))
    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)
    if spec["strong"]:
        assert abs(float(grad)) > 1e-3, (
            f"{name}: gradient ~0 ({float(grad):.2e}); flow likely blocked"
        )


@pytest.mark.parametrize("name", MODELS)
def test_ad_matches_finite_difference(name):
    """Reverse-mode AD agrees with a central finite difference.

    This is the core *correctness* check: it validates the gradient value (not
    merely that something finite came back), and works even for models whose
    gradient is legitimately small.
    """
    spec = SPECS[name]
    f = _pure_loss(spec)
    p0 = float(spec["p0"])
    ad = float(jax.grad(f)(jnp.asarray(p0)))

    eps = 1e-3 * max(1.0, abs(p0))
    fd = (float(f(jnp.asarray(p0 + eps))) - float(f(jnp.asarray(p0 - eps)))) / (2 * eps)

    assert ad == pytest.approx(fd, rel=2e-2, abs=1e-6), (
        f"{name}: AD={ad:.6g} disagrees with finite-difference={fd:.6g}"
    )


@pytest.mark.parametrize("name", MODELS)
def test_check_grads(name):
    """``jax.test_util.check_grads`` passes (reverse mode, first order)."""
    spec = SPECS[name]
    f = _pure_loss(spec)
    check_grads(f, (jnp.asarray(float(spec["p0"])),), order=1, modes=["rev"],
                rtol=2e-2, atol=2e-2)


def test_grad_through_stochastic_model_is_deterministic():
    """Gradient through a *stochastic* model is finite and reproducible when seeded.

    Seeding the RNG before building the model makes the noise realisation -- and
    therefore the gradient -- bit-for-bit reproducible, which is what allows
    stochastic models to be fitted with gradient methods at all.
    """
    def grad_once():
        brainstate.random.seed(123)
        noise = brainmass.GaussianNoise(IN_SIZE, sigma=0.3)  # unitless to match Hopf x
        model = brainmass.HopfStep(IN_SIZE, a=Param(jnp.asarray(0.3), fit=True),
                                   noise_x=noise)
        brainstate.nn.init_all_states(model)
        weights = model.states(brainstate.ParamState)

        def loss():
            return _scalar_loss(model, lambda m: m.update(0.1, 0.0), lambda m: m.x.value)

        grads, value = brainstate.transform.grad(loss, weights, return_value=True)()
        return float(jax.tree.leaves(grads)[0]), float(value)

    g1, v1 = grad_once()
    g2, v2 = grad_once()
    assert np.isfinite(g1) and np.isfinite(v1)
    assert g1 == g2, "seeded stochastic gradient is not reproducible"
    assert v1 == v2


def test_vmap_batched_gradients():
    """Batched (``vmap``) gradients over a sweep of parameter values are finite."""
    f = _pure_loss(SPECS["hopf"])
    p_values = jnp.array([0.1, 0.2, 0.3, 0.4])
    batched = jax.vmap(jax.grad(f))(p_values)
    assert batched.shape == (4,)
    assert jnp.all(jnp.isfinite(batched))
    # Different operating points should give different gradients.
    assert float(jnp.std(batched)) > 1e-3


def test_zero_noise_limit_recovers_deterministic_gradient():
    """With zero-amplitude noise the stochastic path equals the noise-free path."""
    def grad_with(noise):
        brainstate.random.seed(7)
        model = brainmass.HopfStep(IN_SIZE, a=Param(jnp.asarray(0.3), fit=True),
                                   noise_x=noise)
        brainstate.nn.init_all_states(model)
        weights = model.states(brainstate.ParamState)
        grads = brainstate.transform.grad(
            lambda: _scalar_loss(model, lambda m: m.update(0.1, 0.0),
                                 lambda m: m.x.value),
            weights,
        )()
        return float(jax.tree.leaves(grads)[0])

    g_zero_noise = grad_with(brainmass.GaussianNoise(IN_SIZE, sigma=0.0))
    g_no_noise = grad_with(None)
    assert g_zero_noise == pytest.approx(g_no_noise, rel=1e-5, abs=1e-6)
