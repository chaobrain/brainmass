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

"""Tests for :class:`brainmass.Fitter` and :class:`brainmass.FitResult`.

Covers the documented contract for all three backends (``grad`` / ``nevergrad`` /
``scipy``), the ``search_space`` derivation from ``Param`` transforms, the loss
paths (objective vs custom ``loss_fn``), constrained round-tripping, callbacks /
early-stop, every enumerated edge-case error, and the "definition of done"
equivalence: the ``grad`` backend reproduces a hand-rolled ``ModelFitting`` loop
step for step.
"""

import numpy as np
import pytest

import braintools
import brainstate
import brainunit as u
import jax.numpy as jnp
from brainstate.nn import Param, Const, SigmoidT, ReluT, ClipT, IdentityT

import brainmass
from brainmass import Fitter, FitResult
from brainmass.fitter import (
    _trainable_params,
    _finite_box,
    _build_search_space,
)

DT = 1.0 * u.ms
TARGET = jnp.asarray(2.0)


# --------------------------------------------------------------------------- #
# Toy models + predict                                                        #
# --------------------------------------------------------------------------- #

class ScalarToy(brainstate.nn.Module):
    """A single bounded scalar ``k`` whose ``update`` returns ``k``."""

    def __init__(self, init=1.0, t=None):
        super().__init__()
        self.k = Param(init, t=SigmoidT(0.5, 3.0) if t is None else t)

    def update(self):
        return self.k.value()


class MixedToy(brainstate.nn.Module):
    """A trainable bounded ``k`` plus a fixed ``Const`` offset (non-trainable)."""

    def __init__(self):
        super().__init__()
        self.k = Param(1.0, t=SigmoidT(0.5, 3.0))
        self.offset = Const(0.0)

    def update(self):
        return self.k.value() + self.offset.value()


class ConstOnly(brainstate.nn.Module):
    """No trainable parameters at all."""

    def __init__(self):
        super().__init__()
        self.c = Const(1.0)

    def update(self):
        return self.c.value()


def scalar_predict(model):
    """Mean of a short noise-free run -> a scalar that equals ``k``."""
    sim = brainmass.Simulator(model, dt=DT)
    return jnp.mean(sim.run(3.0 * DT, monitors=None)['output'])


@pytest.fixture(autouse=True)
def _dt():
    brainstate.environ.set(dt=DT)
    yield
    brainstate.environ.pop('dt', None)


# --------------------------------------------------------------------------- #
# Exports / FitResult                                                         #
# --------------------------------------------------------------------------- #

def test_exports():
    assert brainmass.Fitter is Fitter
    assert brainmass.FitResult is FitResult
    assert 'Fitter' in brainmass.__all__
    assert 'FitResult' in brainmass.__all__


def test_fitresult_repr():
    r = FitResult(
        backend='grad', best_loss=0.125, best_params={'k': jnp.asarray(2.0)},
        history=[1.0, 0.5, 0.125], n_steps=3,
    )
    text = repr(r)
    assert 'grad' in text and 'k' in text and '0.125' in text


# --------------------------------------------------------------------------- #
# search-space derivation helpers                                             #
# --------------------------------------------------------------------------- #

def test_trainable_params_excludes_const():
    params = _trainable_params(MixedToy())
    assert set(params) == {'k'}  # offset (Const) excluded


def test_finite_box_sigmoid():
    lo, hi = _finite_box(SigmoidT(0.5, 3.0))
    assert (float(lo), float(hi)) == (0.5, 3.0)


def test_finite_box_clip():
    lo, hi = _finite_box(ClipT(-1.0, 4.0))
    assert (float(lo), float(hi)) == (-1.0, 4.0)


def test_finite_box_unbounded_returns_none():
    assert _finite_box(ReluT(0.0)) is None
    assert _finite_box(IdentityT()) is None


def test_build_search_space_autoderive():
    bounds = _build_search_space(_trainable_params(ScalarToy()), None)
    assert set(bounds) == {'k'}
    lo, hi = bounds['k']
    assert (float(lo), float(hi)) == (0.5, 3.0)


def test_build_search_space_user_override():
    bounds = _build_search_space(_trainable_params(ScalarToy()), {'k': (1.0, 2.0)})
    lo, hi = bounds['k']
    assert (float(lo), float(hi)) == (1.0, 2.0)


def test_build_search_space_unbounded_requires_explicit():
    params = _trainable_params(ScalarToy(t=ReluT(0.0)))
    with pytest.raises(ValueError, match="cannot derive search bounds"):
        _build_search_space(params, None)
    # ... but works once given explicitly.
    bounds = _build_search_space(params, {'k': (0.5, 3.0)})
    assert set(bounds) == {'k'}


def test_build_search_space_unknown_name():
    with pytest.raises(ValueError, match="unknown parameter"):
        _build_search_space(_trainable_params(ScalarToy()), {'nope': (0.0, 1.0)})


def test_build_search_space_array_param_shape():
    class VecToy(brainstate.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = Param(jnp.ones(4), t=SigmoidT(0.0, 1.0))

    bounds = _build_search_space(_trainable_params(VecToy()), None)
    lo, hi = bounds['w']
    assert jnp.shape(lo) == (4,) and jnp.shape(hi) == (4,)


# --------------------------------------------------------------------------- #
# Backends reduce loss                                                        #
# --------------------------------------------------------------------------- #

def test_grad_reduces_loss():
    fitter = Fitter(ScalarToy(), braintools.optim.Adam(lr=0.1), predict=scalar_predict)
    r = fitter.fit(target=TARGET, n_steps=40)
    assert r.backend == 'grad'
    assert r.best_loss <= r.history[0]
    assert r.best_loss < 0.05
    assert abs(float(r.best_params['k']) - 2.0) < 0.1
    # the model holds the best-seen parameters after fitting
    assert abs(float(fitter.model.k.value()) - 2.0) < 0.1


def test_grad_default_optimizer():
    fitter = Fitter(ScalarToy(), predict=scalar_predict)  # optimizer=None -> Adam
    r = fitter.fit(target=TARGET, n_steps=30)
    assert isinstance(fitter.optimizer, braintools.optim.OptaxOptimizer)
    assert r.best_loss <= r.history[0]


def test_nevergrad_reduces_loss(seeded):
    pytest.importorskip('nevergrad')
    fitter = Fitter(ScalarToy(), {'method': 'DE', 'n_sample': 6},
                    predict=scalar_predict, backend='nevergrad')
    r = fitter.fit(target=TARGET, n_steps=15)
    assert r.backend == 'nevergrad'
    assert r.best_loss < 0.05
    assert abs(float(r.best_params['k']) - 2.0) < 0.1
    assert len(r.history) > 0


def test_scipy_reduces_loss(seeded):
    fitter = Fitter(ScalarToy(), 'Nelder-Mead', predict=scalar_predict, backend='scipy')
    r = fitter.fit(target=TARGET, n_steps=1, verbose=True)
    assert r.backend == 'scipy'
    assert r.best_loss < 0.05
    assert abs(float(r.best_params['k']) - 2.0) < 0.1
    # raw SciPy OptimizeResult is exposed
    assert hasattr(r.raw, 'x')


def test_nevergrad_default_options(seeded):
    pytest.importorskip('nevergrad')
    # optimizer=None -> default DE method + default n_sample.
    fitter = Fitter(ScalarToy(), predict=scalar_predict, backend='nevergrad')
    r = fitter.fit(target=TARGET, n_steps=5)
    assert np.isfinite(r.best_loss)


def test_optimizer_property_before_and_after_fit():
    opt = braintools.optim.Adam(lr=0.1)
    fitter = Fitter(ScalarToy(), opt, predict=scalar_predict)
    assert fitter.optimizer is opt           # before fit: returns the argument
    fitter.fit(target=TARGET, n_steps=2)
    assert fitter.optimizer is opt           # after fit: the registered optimizer


def test_scipy_gradient_method(seeded):
    # L-BFGS-B (a gradient method) traces through Param.set_value.
    fitter = Fitter(ScalarToy(), {'method': 'L-BFGS-B'},
                    predict=scalar_predict, backend='scipy')
    r = fitter.fit(target=TARGET, n_steps=1)
    assert abs(float(r.best_params['k']) - 2.0) < 0.1


# --------------------------------------------------------------------------- #
# DoD: grad reproduces a hand-rolled ModelFitting loop                        #
# --------------------------------------------------------------------------- #

def test_grad_matches_handrolled_modelfitting():
    n_steps = 25

    # --- Fitter (grad backend) ---
    fitter = Fitter(ScalarToy(), braintools.optim.Adam(lr=0.1), predict=scalar_predict)
    res = fitter.fit(target=TARGET, n_steps=n_steps)

    # --- hand-rolled canonical ModelFitting loop on an identical model ---
    model = ScalarToy()
    weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=0.1)
    opt.register_trainable_weights(weights)

    def f_loss():
        with model.param_precompute():
            pred = scalar_predict(model)
            loss = jnp.sqrt(jnp.mean((pred - TARGET) ** 2)) + model.reg_loss()
        return loss, pred

    @brainstate.transform.jit
    def f_train():
        fg = brainstate.transform.grad(f_loss, weights, has_aux=True, return_value=True)
        grads, loss, pred = fg()
        opt.step(grads)
        return loss

    hand_history = [float(f_train()) for _ in range(n_steps)]

    # Step-for-step identical numerics (same init, optimizer, objective).
    np.testing.assert_allclose(res.history, hand_history, rtol=1e-4, atol=1e-5)
    assert abs(res.history[-1] - hand_history[-1]) < 1e-4


# --------------------------------------------------------------------------- #
# FCD objective works through the Fitter                                      #
# --------------------------------------------------------------------------- #

def test_fcd_objective_runs():
    # A deterministic, time-varying multi-region prediction so FCD is well-defined.
    t = jnp.linspace(0.0, 6.28, 60)[:, None]
    base = jnp.sin(t * jnp.arange(1, 4)[None, :])  # (60, 3)
    target = jnp.cos(t * jnp.arange(1, 4)[None, :])

    def predict_ts(model):
        return base * model.k.value()

    fitter = Fitter(
        ScalarToy(), braintools.optim.Adam(lr=0.05),
        predict=predict_ts,
        objective=brainmass.objectives.fcd(window_size=20, step_size=5, as_loss=True),
    )
    r = fitter.fit(target=target, n_steps=3)
    assert np.isfinite(r.best_loss)


# --------------------------------------------------------------------------- #
# Loss paths                                                                  #
# --------------------------------------------------------------------------- #

def test_loss_fn_path():
    def loss_fn(model):
        pred = scalar_predict(model)
        return (pred - TARGET) ** 2, pred

    fitter = Fitter(ScalarToy(), braintools.optim.Adam(lr=0.1), loss_fn=loss_fn)
    r = fitter.fit(n_steps=40)
    assert r.best_loss < 0.05
    assert abs(float(r.best_params['k']) - 2.0) < 0.1


def test_loss_fn_path_nevergrad(seeded):
    pytest.importorskip('nevergrad')

    def loss_fn(model):
        pred = scalar_predict(model)
        return (pred - TARGET) ** 2, pred

    fitter = Fitter(ScalarToy(), {'n_sample': 6}, loss_fn=loss_fn, backend='nevergrad')
    r = fitter.fit(n_steps=12)
    assert r.prediction is None  # loss_fn path does not re-run predict
    assert r.best_loss < 0.05


def test_transient_trims_prediction():
    captured = {}

    def predict_ts(model):
        return jnp.ones((10, 2)) * model.k.value()

    def objective(pred, target):
        captured['len'] = pred.shape[0]
        return jnp.mean((pred - target) ** 2)

    fitter = Fitter(ScalarToy(), braintools.optim.Adam(lr=0.05),
                    predict=predict_ts, objective=objective, transient=4)
    fitter.fit(target=jnp.ones((6, 2)) * 2.0, n_steps=1)
    assert captured['len'] == 6  # 10 - 4 transient samples


# --------------------------------------------------------------------------- #
# Callbacks / early-stop                                                      #
# --------------------------------------------------------------------------- #

def test_callbacks_called():
    seen = []

    def cb(info):
        seen.append(info['step'])

    fitter = Fitter(ScalarToy(), braintools.optim.Adam(lr=0.1),
                    predict=scalar_predict, callbacks=[cb])
    fitter.fit(target=TARGET, n_steps=5)
    assert seen == [0, 1, 2, 3, 4]


def test_early_stop():
    def cb(info):
        return info['step'] >= 2  # stop after the 3rd step

    fitter = Fitter(ScalarToy(), braintools.optim.Adam(lr=0.1),
                    predict=scalar_predict, callbacks=[cb])
    r = fitter.fit(target=TARGET, n_steps=50)
    assert r.n_steps == 3
    assert len(r.history) == 3


# --------------------------------------------------------------------------- #
# Constrained round-trip                                                      #
# --------------------------------------------------------------------------- #

def test_constrained_roundtrip():
    p = Param(1.5, t=SigmoidT(0.5, 3.0))
    p.set_value(jnp.asarray(2.25))
    assert abs(float(p.value()) - 2.25) < 1e-4  # value round-trips through transform


def test_fit_respects_bounds(seeded):
    # An optimum (4.0) outside the (0.5, 3.0) box must be clamped to the bound.
    fitter = Fitter(ScalarToy(), 'Nelder-Mead', predict=scalar_predict, backend='scipy')
    r = fitter.fit(target=jnp.asarray(4.0), n_steps=2)
    assert float(r.best_params['k']) <= 3.0 + 1e-4


# --------------------------------------------------------------------------- #
# Edge-case errors                                                            #
# --------------------------------------------------------------------------- #

def test_unknown_backend():
    with pytest.raises(ValueError, match="backend must be one of"):
        Fitter(ScalarToy(), predict=scalar_predict, backend='nope')


def test_requires_loss_or_predict():
    with pytest.raises(ValueError, match="loss_fn.*or.*predict"):
        Fitter(ScalarToy())


def test_loss_and_predict_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        Fitter(ScalarToy(), loss_fn=lambda m: (0.0, None), predict=scalar_predict)


def test_grad_no_trainable_params():
    fitter = Fitter(ConstOnly(), braintools.optim.Adam(lr=0.1), predict=scalar_predict)
    with pytest.raises(ValueError, match="no trainable parameters"):
        fitter.fit(target=TARGET, n_steps=3)


def test_derivfree_no_trainable_params():
    fitter = Fitter(ConstOnly(), 'Nelder-Mead', predict=scalar_predict, backend='scipy')
    with pytest.raises(ValueError, match="no trainable parameters"):
        fitter.fit(target=TARGET, n_steps=1)


def test_grad_wrong_optimizer_type():
    fitter = Fitter(ScalarToy(), {'method': 'DE'}, predict=scalar_predict)  # dict for grad
    with pytest.raises(TypeError, match="braintools.optim optimizer"):
        fitter.fit(target=TARGET, n_steps=2)


def test_derivfree_bad_optimizer_type():
    fitter = Fitter(ScalarToy(), 123, predict=scalar_predict, backend='scipy')
    with pytest.raises(TypeError, match="method name"):
        fitter.fit(target=TARGET, n_steps=1)


def test_n_steps_must_be_positive():
    fitter = Fitter(ScalarToy(), braintools.optim.Adam(lr=0.1), predict=scalar_predict)
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        fitter.fit(target=TARGET, n_steps=0)


def test_verbose_runs(capsys):
    fitter = Fitter(ScalarToy(), braintools.optim.Adam(lr=0.1), predict=scalar_predict)
    fitter.fit(target=TARGET, n_steps=2, verbose=True)
    assert '[grad]' in capsys.readouterr().out
