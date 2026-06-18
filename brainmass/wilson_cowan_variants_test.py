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

"""Coverage + behaviour tests for the less-common Wilson-Cowan variants.

The base :class:`brainmass.WilsonCowanStep` and a few simple variants were tested
already; this module covers the previously-untested variants -- divisive,
divisive-input, delayed, adaptive and three-population -- plus the sequence
network components (``WilsonCowanSeqLayer`` / ``WilsonCowanSeqNetwork``).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import brainstate
import brainunit as u
from brainstate.nn import Param

import brainmass
from brainmass import wilson_cowan as wc


def _run(model, n_steps, **inp):
    """Advance ``model`` for ``n_steps`` and return the stacked output trajectory."""
    def step(i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            return model.update(**inp)

    return brainstate.transform.for_loop(step, np.arange(n_steps))


# All single-node Wilson-Cowan step variants that were previously untested.
STEP_VARIANTS = {
    "divisive": brainmass.WilsonCowanDivisiveStep,
    "divisive_input": brainmass.WilsonCowanDivisiveInputStep,
    "delayed": brainmass.WilsonCowanDelayedStep,
    "adaptive": brainmass.WilsonCowanAdaptiveStep,
    "threepop": brainmass.WilsonCowanThreePopulationStep,
}


@pytest.mark.parametrize("name", list(STEP_VARIANTS))
def test_variant_simulates_finitely(name, dt):
    """Each variant constructs, initialises and integrates to a finite trajectory."""
    model = STEP_VARIANTS[name](2)
    brainstate.nn.init_all_states(model)
    out = _run(model, 100, rE_inp=0.5)
    assert out.shape == (100, 2)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("name", list(STEP_VARIANTS))
def test_variant_rk4_integration(name, dt):
    """Each variant integrates finitely under a higher-order solver (rk4 path).

    This exercises every variant's ``derivative`` aggregator (used only by the
    non-``exp_euler`` branch), which the default-method tests never reach.
    """
    model = STEP_VARIANTS[name](2, method="rk4")
    brainstate.nn.init_all_states(model)
    out = _run(model, 50, rE_inp=0.5)
    assert out.shape == (50, 2)
    assert jnp.all(jnp.isfinite(out))


def test_divisive_semisaturation_keeps_output_bounded(dt):
    """Divisive normalisation with positive semisaturation stays finite/bounded."""
    model = brainmass.WilsonCowanDivisiveStep(1, sigma_E=1.0, sigma_I=1.0)
    brainstate.nn.init_all_states(model)
    out = _run(model, 200, rE_inp=2.0)
    assert jnp.all(jnp.isfinite(out))


def test_adaptive_has_four_states_and_suppresses_activity(dt):
    """Adaptive variant tracks adaptation currents and damps sustained activity."""
    model = brainmass.WilsonCowanAdaptiveStep(1, b_E=0.5, tau_aE=20.0 * u.ms)
    brainstate.nn.init_all_states(model)
    # Four state variables: rE, rI, aE, aI.
    states = model.states(brainstate.HiddenState)
    names = {k[-1] if isinstance(k, tuple) else k for k in states.keys()}
    assert {"rE", "rI", "aE", "aI"} <= names

    out = np.asarray(u.get_magnitude(_run(model, 2000, rE_inp=3.0)))
    early_peak = out[:200].max()
    late = out[-200:].mean()
    assert late < early_peak, "adaptation should reduce activity over time"


def test_adaptive_adaptation_derivatives(dt):
    """The adaptation derivative helpers return finite ``1/ms`` rates."""
    model = brainmass.WilsonCowanAdaptiveStep(3)
    brainstate.nn.init_all_states(model)
    daE = model.daE(model.aE.value, model.rE.value)
    daI = model.daI(model.aI.value, model.rI.value)
    assert u.get_unit(daE) == u.get_unit(1.0 / u.ms)
    assert jnp.all(jnp.isfinite(u.get_magnitude(daE)))
    assert jnp.all(jnp.isfinite(u.get_magnitude(daI)))


def test_threepopulation_has_modulatory_state(dt):
    """The three-population variant adds a modulatory state ``rM`` and ``drM``."""
    model = brainmass.WilsonCowanThreePopulationStep(2)
    brainstate.nn.init_all_states(model)
    states = model.states(brainstate.HiddenState)
    names = {k[-1] if isinstance(k, tuple) else k for k in states.keys()}
    assert {"rE", "rI", "rM"} <= names
    drM = model.drM(model.rM.value, model.rE.value, model.rI.value, 0.0)
    assert jnp.all(jnp.isfinite(u.get_magnitude(drM)))
    out = _run(model, 100, rE_inp=1.0, rI_inp=0.5, rM_inp=0.2)
    assert jnp.all(jnp.isfinite(out))


def test_threepop_base_is_abstract(dt):
    """``WilsonCowanThreePopBase`` is abstract: its RHS methods are unimplemented."""
    base = wc.WilsonCowanThreePopBase(2)
    brainstate.nn.init_all_states(base)
    with pytest.raises(NotImplementedError):
        with brainstate.environ.context(i=0, t=0.0 * u.ms):
            base.update(0.5)


def test_variant_gradient_flows(dt):
    """A gradient flows through a divisive variant w.r.t. a trainable weight."""
    def loss(w):
        brainstate.random.seed(0)
        model = brainmass.WilsonCowanDivisiveStep(2, wEE=Param(w, fit=True))
        brainstate.nn.init_all_states(model)
        out = _run(model, 40, rE_inp=1.0)
        return jnp.sum(u.get_magnitude(out) ** 2)

    value, grad = jax.value_and_grad(loss)(jnp.asarray(6.0))
    assert jnp.isfinite(value) and jnp.isfinite(grad)


# --------------------------------------------------------------------------- #
# Sequence network components
# --------------------------------------------------------------------------- #

def _seq(n_steps, n_features, seed=0):
    """Deterministic ``(n_steps, n_features)`` input sequence (seeds numpy's RNG)."""
    brainstate.random.seed(seed)
    np.random.seed(seed)  # the sequence below uses numpy's RNG, not brainstate's
    return jnp.asarray(np.random.randn(n_steps, n_features).astype("float32"))


def test_seq_layer_processes_sequence(dt):
    """``WilsonCowanSeqLayer`` maps a ``(T, n_input)`` sequence to ``(T, n_hidden)``."""
    layer = wc.WilsonCowanSeqLayer(3, 5)
    brainstate.nn.init_all_states(layer)
    out = layer.update(_seq(10, 3))
    assert out.shape == (10, 5)
    assert jnp.all(jnp.isfinite(out))


def test_seq_layer_record_state(dt):
    """``record_state=True`` returns the per-step internal states alongside output."""
    layer = wc.WilsonCowanSeqLayer(3, 5)
    brainstate.nn.init_all_states(layer)
    result = layer.update(_seq(8, 3), record_state=True)
    assert isinstance(result, tuple) and len(result) == 2


def test_seq_layer_with_delay(dt):
    """The delayed recurrent path (``DelayedAdditiveConn``) runs over a sequence."""
    brainstate.random.seed(0)
    np.random.seed(0)  # deterministic delay matrix (numpy's RNG)
    delay = jnp.asarray(np.random.randint(1, 4, size=(5, 5)).astype("float32")) * u.ms
    layer = wc.WilsonCowanSeqLayer(3, 5, delay=delay)
    brainstate.nn.init_all_states(layer)
    out = layer.update(_seq(30, 3))
    assert out.shape == (30, 5)
    assert jnp.all(jnp.isfinite(out))


def test_seq_network_single_layer(dt):
    """``WilsonCowanSeqNetwork`` reduces a sequence to an ``(n_output,)`` readout."""
    net = wc.WilsonCowanSeqNetwork(4, 6, 2)
    brainstate.nn.init_all_states(net)
    out = net.update(_seq(12, 4))
    assert out.shape == (2,)
    assert jnp.all(jnp.isfinite(out))


def test_seq_network_multi_layer_and_hidden_activation(dt):
    """A multi-layer network stacks layers and exposes per-layer activations."""
    net = wc.WilsonCowanSeqNetwork(4, [6, 8], 3)
    brainstate.nn.init_all_states(net)
    inp = _seq(12, 4)
    out = net.update(inp)
    assert out.shape == (3,)
    hidden = net.hidden_activation(inp)
    assert isinstance(hidden, list) and len(hidden) == 2
    assert hidden[0].shape == (12, 6)
    assert hidden[1].shape == (12, 8)
