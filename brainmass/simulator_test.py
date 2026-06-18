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

"""Tests for :class:`brainmass.Simulator`.

Covers the documented contract (shapes, time axis, monitors, inputs, transient,
sample_every, batching, jit) and every enumerated edge case, plus the two
"definition of done" equivalences: a single-node run reproduces the hand-written
``for_loop`` output exactly, and a whole-brain run matches the example-100-style
hand-wired loop.
"""

import numpy as np
import pytest

import braintools
import brainstate
import brainunit as u

import brainmass
from brainmass import Simulator

DT = 0.1 * u.ms


# --------------------------------------------------------------------------- #
# Basic contract                                                              #
# --------------------------------------------------------------------------- #

def test_model_attribute_exposed():
    node = brainmass.HopfStep(2, a=-0.2)
    sim = Simulator(node, dt=DT)
    assert sim.model is node


def test_run_shapes_and_time_axis(seeded):
    sim = Simulator(brainmass.HopfStep(3, a=-0.2), dt=DT)
    res = sim.run(10.0 * u.ms, monitors=['x'])
    assert res['x'].shape == (100, 3)
    assert res['ts'].shape == (100,)
    # ts is the time at the END of each step: first = dt, last = duration.
    assert u.math.allclose(res['ts'][0], DT)
    assert u.math.allclose(res['ts'][-1], 10.0 * u.ms)


def test_dt_read_from_environ(dt):
    # No dt passed to the Simulator -> it must read the environ dt (set by the
    # ``dt`` fixture) at run time.
    sim = Simulator(brainmass.HopfStep(2, a=-0.2))
    res = sim.run(5.0 * u.ms, monitors=['x'])
    assert res['x'].shape == (50, 2)


def test_dt_unset_raises():
    brainstate.environ.pop('dt', None)  # restored by the autouse fixture
    sim = Simulator(brainmass.HopfStep(2, a=-0.2))
    with pytest.raises(ValueError, match='dt is not set'):
        sim.run(5.0 * u.ms)


# --------------------------------------------------------------------------- #
# Definition of done: exact reproduction of the manual loop                   #
# --------------------------------------------------------------------------- #

def test_exact_reproduces_manual_for_loop():
    """monitors=['x'] reproduces a hand-written for_loop bit-for-bit (seeded)."""
    brainstate.environ.set(dt=DT)

    brainstate.random.seed(7)
    manual_model = brainmass.HopfStep(2, a=-0.2)
    brainstate.nn.init_all_states(manual_model)

    def step(i):
        manual_model.update()
        return manual_model.x.value

    manual = brainstate.transform.for_loop(step, np.arange(100))

    brainstate.random.seed(7)
    res = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(
        10.0 * u.ms, monitors=['x'], jit=False
    )
    assert np.array_equal(np.asarray(manual), np.asarray(res['x']))


def test_jit_matches_nojit(seeded):
    brainstate.random.seed(3)
    a = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(5.0 * u.ms, monitors=['x'], jit=True)
    brainstate.random.seed(3)
    b = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(5.0 * u.ms, monitors=['x'], jit=False)
    assert np.allclose(np.asarray(a['x']), np.asarray(b['x']), rtol=1e-5, atol=1e-6)


# --------------------------------------------------------------------------- #
# Monitors                                                                    #
# --------------------------------------------------------------------------- #

def test_monitors_none_records_update_return(seeded):
    res = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(2.0 * u.ms, monitors=None)
    # HopfStep.update() returns the (x, y) tuple; for_loop stacks each leaf.
    assert set(res) == {'output', 'ts'}
    x_out, y_out = res['output']
    assert x_out.shape == (20, 2) and y_out.shape == (20, 2)


def test_monitors_callable_derived_observable(seeded):
    # eeg() = E - I, a derived observable not stored as a single state.
    res = Simulator(brainmass.JansenRitStep(in_size=1), dt=DT).run(
        5.0 * u.ms, monitors=lambda m: m.eeg()
    )
    assert res['output'].shape == (50, 1)
    assert u.get_unit(res['output']) == u.mV


def test_monitors_dict_mixed(seeded):
    res = Simulator(brainmass.JansenRitStep(in_size=1), dt=DT).run(
        3.0 * u.ms, monitors={'exc': 'E', 'eeg': lambda m: m.eeg()}
    )
    assert set(res) == {'exc', 'eeg', 'ts'}
    assert res['exc'].shape == (30, 1)
    assert res['eeg'].shape == (30, 1)


def test_monitor_missing_attr_raises():
    sim = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT)
    with pytest.raises(ValueError, match='not an attribute'):
        sim.run(2.0 * u.ms, monitors=['does_not_exist'])


def test_monitor_dict_missing_attr_raises():
    sim = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT)
    with pytest.raises(ValueError, match='not an attribute'):
        sim.run(2.0 * u.ms, monitors={'bad': 'nope'})


# --------------------------------------------------------------------------- #
# Inputs                                                                      #
# --------------------------------------------------------------------------- #

def test_inputs_array_drives_update(seeded):
    n_steps = 50
    drive = np.linspace(0.0, 1.0, n_steps).astype('float32')
    res = Simulator(brainmass.FitzHughNagumoStep(1), dt=DT).run(
        5.0 * u.ms, inputs=drive, monitors=['V']
    )
    assert res['V'].shape == (50, 1)


def test_inputs_array_wrong_length_raises():
    sim = Simulator(brainmass.FitzHughNagumoStep(1), dt=DT)
    with pytest.raises(ValueError, match='length'):
        sim.run(5.0 * u.ms, inputs=np.zeros(7, dtype='float32'), monitors=['V'])


def test_inputs_callable_tuple_and_scalar(seeded):
    # Hopf.update(x_inp, y_inp): callable returns a tuple, splatted into update.
    res = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(
        2.0 * u.ms, inputs=lambda i, t: (0.1, 0.0), monitors=['x']
    )
    assert res['x'].shape == (20, 2)

    # FHN.update(V_inp): callable returns a single value, wrapped into a 1-tuple.
    res2 = Simulator(brainmass.FitzHughNagumoStep(1), dt=DT).run(
        2.0 * u.ms, inputs=lambda i, t: 0.3, monitors=['V']
    )
    assert res2['V'].shape == (20, 1)


# --------------------------------------------------------------------------- #
# Transient / sample_every                                                    #
# --------------------------------------------------------------------------- #

def test_transient_quantity_and_int_discard_leading(seeded):
    full = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(10.0 * u.ms, monitors=['x'])
    assert full['x'].shape == (100, 2)

    res_q = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(
        10.0 * u.ms, monitors=['x'], transient=2.0 * u.ms
    )
    assert res_q['x'].shape == (80, 2)
    # 20 steps discarded; first kept step is index 20, ending at 2.1 ms.
    assert u.math.allclose(res_q['ts'][0], 2.1 * u.ms)

    res_i = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(
        10.0 * u.ms, monitors=['x'], transient=20
    )
    assert res_i['x'].shape == (80, 2)


def test_transient_ge_duration_raises():
    sim = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT)
    with pytest.raises(ValueError, match='shorter than the run'):
        sim.run(5.0 * u.ms, monitors=['x'], transient=5.0 * u.ms)


def test_transient_negative_raises():
    sim = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT)
    with pytest.raises(ValueError, match='>= 0'):
        sim.run(5.0 * u.ms, monitors=['x'], transient=-1)


def test_sample_every_downsamples(seeded):
    res = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(
        10.0 * u.ms, monitors=['x'], sample_every=5
    )
    assert res['x'].shape == (20, 2)
    assert res['ts'].shape == (20,)
    # stride of 5: samples at steps 0, 5, 10, ... -> first ends at 0.1 ms,
    # consecutive samples spaced 5*dt apart.
    assert u.math.allclose(res['ts'][0], 0.1 * u.ms)
    assert u.math.allclose(res['ts'][1] - res['ts'][0], 5 * DT)


def test_sample_every_with_transient(seeded):
    res = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT).run(
        10.0 * u.ms, monitors=['x'], transient=2.0 * u.ms, sample_every=4
    )
    # (100 - 20) steps, every 4th -> 20 samples.
    assert res['x'].shape == (20, 2)


def test_sample_every_invalid_raises():
    sim = Simulator(brainmass.HopfStep(2, a=-0.2), dt=DT)
    with pytest.raises(ValueError, match='sample_every'):
        sim.run(2.0 * u.ms, monitors=['x'], sample_every=0)


# --------------------------------------------------------------------------- #
# Steps resolution                                                            #
# --------------------------------------------------------------------------- #

def test_non_integer_multiple_warns_and_floors(seeded):
    with pytest.warns(UserWarning, match='not an integer multiple'):
        res = Simulator(brainmass.HopfStep(1, a=-0.2), dt=DT).run(1.05 * u.ms, monitors=['x'])
    assert res['x'].shape == (10, 1)  # floor(10.5)


def test_zero_duration_raises():
    sim = Simulator(brainmass.HopfStep(1, a=-0.2), dt=DT)
    with pytest.raises(ValueError, match='must be >= 1 dt'):
        sim.run(0.0 * u.ms)


# --------------------------------------------------------------------------- #
# Units / batching / init                                                     #
# --------------------------------------------------------------------------- #

def test_units_propagate_through_outputs(seeded):
    res = Simulator(brainmass.MontbrioPazoRoxinStep(2), dt=DT).run(2.0 * u.ms, monitors=['r'])
    assert u.get_unit(res['r']) == u.Hz


def test_batched_run(seeded):
    res = Simulator(brainmass.HopfStep(3, a=-0.2), dt=DT).run(
        2.0 * u.ms, monitors=['x'], batch_size=4
    )
    assert res['x'].shape == (20, 4, 3)


def test_init_states_false_continues(seeded):
    # Non-zero IC so the (decaying) trajectory actually moves; the default zero
    # IC sits exactly on the fixed point and would make the test degenerate.
    make = lambda: brainmass.HopfStep(
        2, a=-0.2,
        init_x=braintools.init.Constant(0.5),
        init_y=braintools.init.Constant(0.5),
    )
    sim = Simulator(make(), dt=DT)
    first = sim.run(2.0 * u.ms, monitors=['x'])
    last_state = np.asarray(sim.model.x.value)
    # Continue WITHOUT re-initialising; the run must pick up from where it left,
    # so its first sample differs from the first run's first sample.
    cont = sim.run(2.0 * u.ms, monitors=['x'], init_states=False)
    assert not np.allclose(np.asarray(cont['x'][0]), np.asarray(first['x'][0]))
    assert np.all(np.isfinite(np.asarray(cont['x'])))
    assert last_state.shape == (2,)


# --------------------------------------------------------------------------- #
# Definition of done: gradients flow through run                              #
# --------------------------------------------------------------------------- #

def test_gradient_flows_through_run():
    import jax
    brainstate.environ.set(dt=DT)

    def loss(a):
        model = brainmass.HopfStep(
            1, a=a, w=0.2, beta=1.0,
            init_x=braintools.init.Constant(0.5),
            init_y=braintools.init.Constant(0.5),
        )
        res = Simulator(model, dt=DT).run(2.0 * u.ms, monitors=['x'], jit=False)
        return u.math.sum(u.get_magnitude(res['x']) ** 2)

    g = float(jax.grad(loss)(-0.2))
    fd = float((loss(-0.2 + 1e-3) - loss(-0.2 - 1e-3)) / 2e-3)
    assert np.isfinite(g)
    assert np.isclose(g, fd, rtol=2e-2, atol=1e-5)


# --------------------------------------------------------------------------- #
# Definition of done: whole-brain run matches hand-wired loop                 #
# --------------------------------------------------------------------------- #

class _WholeBrain(brainstate.nn.Module):
    """Minimal example-100-style network: WilsonCowan nodes + diffusive coupling."""

    def __init__(self, sc, delay, k=1.0):
        super().__init__()
        n = sc.shape[0]
        indices = np.tile(np.arange(n), n)
        self.node = brainmass.WilsonCowanStep(n)
        self.coupling = brainmass.DiffusiveCoupling(
            self.node.prefetch_delay(
                'rE', (delay.flatten(), indices),
                init=braintools.init.Uniform(0.0, 0.05),
            ),
            self.node.prefetch('rE'),
            sc,
            k=k,
        )

    def update(self):
        current = self.coupling()
        return self.node(current)

    def step_run(self, i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            return self.update()


def test_whole_brain_matches_manual_loop(connectome):
    sc, delay = connectome['SC'], connectome['delay']
    brainstate.environ.set(dt=DT)
    n_steps = 60

    brainstate.random.seed(11)
    manual_net = _WholeBrain(sc, delay)
    brainstate.nn.init_all_states(manual_net)
    manual = brainstate.transform.for_loop(manual_net.step_run, np.arange(n_steps))

    brainstate.random.seed(11)
    sim_net = _WholeBrain(sc, delay)
    res = Simulator(sim_net, dt=DT).run(n_steps * DT, monitors=None, jit=False)

    assert np.allclose(np.asarray(manual), np.asarray(res['output']), rtol=1e-5, atol=1e-6)
