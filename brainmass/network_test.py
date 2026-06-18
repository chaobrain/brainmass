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

"""Tests for :class:`brainmass.Network`.

The headline test reproduces a hand-wired ``examples/100``-style network
bit-for-bit (seeded). The rest enumerate the edge cases from the goal spec:
non-symmetric SC, instantaneous vs delayed coupling, the ``self_connection``
toggle, a clear error for an unknown ``coupled_var``, 1-D vs 2-D connectivity,
delay-unit handling, the (unsupported) batched-coupling limitation, a gradient
through ``k``, the additive / Laplacian paths, and running under ``Simulator``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import braintools
import brainstate
import brainunit as u
from brainstate.nn import Param

import brainmass


DT = 0.1 * u.ms


def _det_hopf(n, **kw):
    """A deterministic (noise-free, fixed-IC) Hopf node for exact comparisons."""
    return brainmass.HopfStep(
        n, a=0.1, w=0.2,
        init_x=braintools.init.Constant(0.2),
        init_y=braintools.init.Constant(0.1),
        **kw,
    )


def _run(model, n_steps, dt=DT, **run_kw):
    """Drive ``model`` with the simulator, monitoring the node's ``x`` state."""
    sim = brainmass.Simulator(model, dt=dt)
    return sim.run(n_steps * dt, monitors=lambda m: m.node.x.value, **run_kw)


# --------------------------------------------------------------------------- #
# construction / exposure                                                     #
# --------------------------------------------------------------------------- #

def test_node_and_n_node_exposed(dt, connectome):
    node = _det_hopf(connectome["n"])
    net = brainmass.Network(node, conn=connectome["SC"], coupled_var="x")
    assert net.node is node
    assert net.n_node == connectome["n"]


def test_runs_under_simulator(dt, connectome):
    n = connectome["n"]
    net = brainmass.Network(_det_hopf(n), conn=connectome["SC"], coupled_var="x", k=0.3)
    res = _run(net, 50)
    assert res["output"].shape == (50, n)
    # time axis: post-step convention, ts[0] == dt, ts[-1] == duration.
    assert u.math.allclose(res["ts"][0], DT)
    assert u.math.allclose(res["ts"][-1], 50 * DT)


def test_simulator_array_input_forwarded_to_node(dt, connectome):
    # An extra per-step input from the simulator is forwarded after the coupling
    # current (drives the node's second input, e.g. rI_inp).
    n = connectome["n"]
    node = brainmass.WilsonCowanStep(n)
    net = brainmass.Network(node, conn=connectome["SC"], coupled_var="rE", k=0.3)
    sim = brainmass.Simulator(net, dt=DT)
    drive = jnp.ones((20, n)) * 0.1
    res = sim.run(20 * DT, inputs=drive, monitors=lambda m: m.node.rE.value)
    assert res["output"].shape == (20, n)


# --------------------------------------------------------------------------- #
# bit-for-bit vs a hand-wired (examples/100-style) baseline                   #
# --------------------------------------------------------------------------- #

class _RefNet(brainstate.nn.Module):
    """Hand-wired whole-brain net mirroring the ``examples/100`` pattern.

    Uses the *flattened* ``(delay, index)`` form of the examples to prove the
    2-D form used by :class:`brainmass.Network` is equivalent bit-for-bit.
    """

    def __init__(self, conn, dist, speed, k, coupled_var, delay_init):
        super().__init__()
        n = conn.shape[0]
        cw = np.asarray(conn).copy()
        np.fill_diagonal(cw, 0.0)
        dly = np.asarray(dist).copy() / speed
        np.fill_diagonal(dly, 0.0)
        dly = dly * u.ms
        idx = np.tile(np.arange(n), n)  # flattened, example-100 style
        self.node = brainmass.WilsonCowanStep(
            n, noise_E=brainmass.OUProcess(n, sigma=0.01)
        )
        self.coupling = brainmass.DiffusiveCoupling(
            self.node.prefetch_delay(coupled_var, (dly.flatten(), idx), init=delay_init),
            self.node.prefetch(coupled_var),
            cw,
            k=k,
        )

    def update(self):
        return self.node(self.coupling())


def _make_net(conn, dist, speed, k, coupled_var, delay_init):
    node = brainmass.WilsonCowanStep(
        conn.shape[0], noise_E=brainmass.OUProcess(conn.shape[0], sigma=0.01)
    )
    return brainmass.Network(
        node, conn=conn, distance=dist, speed=speed,
        coupled_var=coupled_var, k=k, delay_init=delay_init,
    )


def _run_rE(model, n_steps, dt=DT):
    def step(i):
        with brainstate.environ.context(i=i, t=i * dt):
            model.update()
            return model.node.rE.value
    return brainstate.transform.for_loop(step, np.arange(n_steps))


def test_bit_for_bit_matches_handwired_baseline(dt, connectome):
    conn = np.asarray(connectome["SC"])
    dist = np.asarray(connectome["dist"])
    speed, k = 2.0, 0.4
    build_seed, run_seed = 7, 11
    delay_init = braintools.init.Uniform(0.0, 0.05)

    brainstate.random.seed(build_seed)
    ref = _RefNet(conn, dist, speed, k, "rE", delay_init)
    brainstate.nn.init_all_states(ref)
    brainstate.random.seed(run_seed)
    ref_out = np.asarray(_run_rE(ref, 40))

    brainstate.random.seed(build_seed)
    net = _make_net(conn, dist, speed, k, "rE", delay_init)
    brainstate.nn.init_all_states(net)
    brainstate.random.seed(run_seed)
    net_out = np.asarray(_run_rE(net, 40))

    assert np.array_equal(ref_out, net_out)


# --------------------------------------------------------------------------- #
# connectivity handling                                                       #
# --------------------------------------------------------------------------- #

def test_self_connection_toggle_zeros_diagonal(dt, connectome):
    n = connectome["n"]
    full = np.ones((n, n), dtype=np.float32) * 0.3  # nonzero diagonal on purpose

    net_off = brainmass.Network(_det_hopf(n), conn=full, coupled_var="x")
    diag_off = np.diag(np.asarray(u.get_magnitude(net_off.coupling.conn.value())))
    assert np.allclose(diag_off, 0.0)

    net_on = brainmass.Network(
        _det_hopf(n), conn=full, coupled_var="x", self_connection=True
    )
    diag_on = np.diag(np.asarray(u.get_magnitude(net_on.coupling.conn.value())))
    assert np.allclose(diag_on, 0.3)


def test_conn_1d_flattened_matches_2d(dt, connectome):
    n = connectome["n"]
    w2 = np.asarray(connectome["SC"]).copy()
    np.fill_diagonal(w2, 0.0)
    w1 = w2.flatten()

    def run(conn):
        brainstate.random.seed(1)
        net = brainmass.Network(
            _det_hopf(n), conn=conn, coupled_var="x", k=0.3,
            delay_init=braintools.init.ZeroInit(),
        )
        return np.asarray(_run(net, 20, jit=False)["output"])

    assert np.array_equal(run(w1), run(w2))


def test_nonsymmetric_sc_respected(dt, connectome):
    n = connectome["n"]
    asym = np.asarray(connectome["SC"]).copy()  # connectome SC is symmetric...
    asym[0, 1] = 0.9
    asym[1, 0] = 0.0  # ...break the symmetry

    def run(conn):
        net = brainmass.Network(
            _det_hopf(n), conn=conn, coupled_var="x", k=0.5,
            delay_init=braintools.init.ZeroInit(),
        )
        return np.asarray(_run(net, 30, jit=False)["output"])

    # A directed SC and its transpose must drive different dynamics.
    assert not np.allclose(run(asym), run(asym.T))


def test_conn_wrong_size_raises(dt, connectome):
    n = connectome["n"]
    with pytest.raises(ValueError, match="flattened conn"):
        brainmass.Network(_det_hopf(n), conn=np.ones(n * n + 1), coupled_var="x")
    with pytest.raises(ValueError, match="conn has shape"):
        brainmass.Network(_det_hopf(n), conn=np.ones((n + 1, n + 1)), coupled_var="x")
    with pytest.raises(ValueError, match="1-D .* or 2-D"):
        brainmass.Network(_det_hopf(n), conn=np.ones((n, n, 1)), coupled_var="x")


# --------------------------------------------------------------------------- #
# delays                                                                      #
# --------------------------------------------------------------------------- #

def test_instantaneous_differs_from_delayed(dt, connectome):
    n = connectome["n"]
    conn = connectome["SC"]
    dist = connectome["dist"]

    def run(distance, speed):
        brainstate.random.seed(2)
        net = brainmass.Network(
            _det_hopf(n), conn=conn, distance=distance, speed=speed,
            coupled_var="x", k=0.6, delay_init=braintools.init.ZeroInit(),
        )
        return np.asarray(_run(net, 60, jit=False)["output"])

    delayed = run(dist, 0.5)        # slow speed -> sizeable delays
    instant = run(None, None)       # delay-free path
    assert not np.allclose(delayed, instant)


def test_faster_speed_is_closer_to_instantaneous(dt, connectome):
    n = connectome["n"]
    conn = connectome["SC"]
    dist = connectome["dist"]

    def run(distance, speed):
        brainstate.random.seed(3)
        net = brainmass.Network(
            _det_hopf(n), conn=conn, distance=distance, speed=speed,
            coupled_var="x", k=0.6, delay_init=braintools.init.ZeroInit(),
        )
        return np.asarray(_run(net, 60, jit=False)["output"])

    # delay = distance / speed: a faster conduction speed shrinks every delay,
    # so the trajectory moves monotonically toward the delay-free baseline.
    instant = run(None, None)
    slow = run(dist, 0.3)
    fast = run(dist, 50.0)
    d_slow = np.max(np.abs(slow - instant))
    d_fast = np.max(np.abs(fast - instant))
    assert d_fast < d_slow


def test_delay_units_plain_equals_quantity(dt, connectome):
    n = connectome["n"]
    conn = connectome["SC"]
    dist = np.asarray(connectome["dist"])

    def run(distance, speed):
        brainstate.random.seed(4)
        net = brainmass.Network(
            _det_hopf(n), conn=conn, distance=distance, speed=speed,
            coupled_var="x", k=0.5, delay_init=braintools.init.ZeroInit(),
        )
        return np.asarray(_run(net, 30, jit=False)["output"])

    # plain (dist / 2 -> ms) vs unit-carrying (mm / (mm/ms) -> ms): same delays.
    plain = run(dist, 2.0)
    quant = run(dist * u.mm, 2.0 * u.mm / u.ms)
    assert np.array_equal(plain, quant)


# --------------------------------------------------------------------------- #
# coupled_var validation                                                      #
# --------------------------------------------------------------------------- #

def test_coupled_var_not_a_state_raises_clear_error(dt, connectome):
    n = connectome["n"]
    net = brainmass.Network(_det_hopf(n), conn=connectome["SC"], coupled_var="bogus")
    with pytest.raises(ValueError, match="coupled_var 'bogus' is not a state"):
        brainstate.nn.init_all_states(net)


# --------------------------------------------------------------------------- #
# coupling kinds                                                              #
# --------------------------------------------------------------------------- #

def test_additive_coupling_runs(dt, connectome):
    n = connectome["n"]
    net = brainmass.Network(
        _det_hopf(n), conn=connectome["SC"], coupling="additive",
        coupled_var="x", k=0.3,
    )
    assert _run(net, 20, jit=False)["output"].shape == (20, n)


def test_laplacian_coupling_runs_and_grad(dt, connectome):
    n = connectome["n"]

    def loss(kval):
        net = brainmass.Network(
            _det_hopf(n), conn=connectome["SC"], coupling="laplacian",
            coupled_var="x", k=Param(kval), delay_init=braintools.init.ZeroInit(),
        )
        xs = _run(net, 25, jit=False)["output"]
        return jnp.mean(u.get_magnitude(xs) ** 2)

    val = loss(0.3)
    assert np.isfinite(float(val))
    g = jax.grad(loss)(0.3)
    assert np.isfinite(float(g))


def test_invalid_coupling_raises(dt, connectome):
    with pytest.raises(ValueError, match="coupling must be"):
        brainmass.Network(
            _det_hopf(connectome["n"]), conn=connectome["SC"],
            coupling="nonsense", coupled_var="x",
        )


def test_trainable_conn_param_passthrough(dt, connectome):
    n = connectome["n"]
    w = np.asarray(connectome["SC"]).copy()
    np.fill_diagonal(w, 0.0)

    def loss(scale):
        # A Param conn is passed through untouched (and is trainable).
        net = brainmass.Network(
            _det_hopf(n), conn=Param(w * scale), coupled_var="x", k=0.5,
            delay_init=braintools.init.ZeroInit(),
        )
        xs = _run(net, 25, jit=False)["output"]
        return jnp.mean(u.get_magnitude(xs) ** 2)

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))


def test_prebuilt_laplacian_conn_param(dt, connectome):
    n = connectome["n"]
    w = np.asarray(connectome["SC"]).copy()
    np.fill_diagonal(w, 0.0)
    # A pre-built LaplacianConnParam is used as-is by the 'laplacian' path.
    lap = brainmass.LaplacianConnParam(w, fit=False)
    net = brainmass.Network(
        _det_hopf(n), conn=lap, coupling="laplacian", coupled_var="x", k=0.3,
        delay_init=braintools.init.ZeroInit(),
    )
    assert _run(net, 20, jit=False)["output"].shape == (20, n)


# --------------------------------------------------------------------------- #
# noise                                                                       #
# --------------------------------------------------------------------------- #

def test_network_level_noise_changes_trajectory(dt, connectome):
    n = connectome["n"]
    conn = connectome["SC"]

    brainstate.random.seed(5)
    quiet = brainmass.Network(
        _det_hopf(n), conn=conn, coupled_var="x", k=0.3,
        delay_init=braintools.init.ZeroInit(),
    )
    quiet_out = np.asarray(_run(quiet, 30, jit=False)["output"])

    brainstate.random.seed(5)
    noisy = brainmass.Network(
        _det_hopf(n), conn=conn, coupled_var="x", k=0.3,
        delay_init=braintools.init.ZeroInit(),
        noise=brainmass.OUProcess(n, sigma=0.05),
    )
    noisy_out = np.asarray(_run(noisy, 30, jit=False)["output"])

    assert not np.allclose(quiet_out, noisy_out)


# --------------------------------------------------------------------------- #
# gradient                                                                    #
# --------------------------------------------------------------------------- #

def test_gradient_through_k(dt, connectome):
    n = connectome["n"]
    conn = connectome["SC"]

    def loss(kval):
        net = brainmass.Network(
            _det_hopf(n), conn=conn, coupled_var="x", k=Param(kval),
            delay_init=braintools.init.ZeroInit(),
        )
        xs = _run(net, 30, jit=False)["output"]
        return jnp.mean(u.get_magnitude(xs) ** 2)

    g = jax.grad(loss)(0.3)
    fd = (loss(0.3 + 1e-3) - loss(0.3 - 1e-3)) / 2e-3
    assert np.isfinite(float(g))
    assert np.allclose(float(g), float(fd), rtol=2e-2, atol=1e-6)


# --------------------------------------------------------------------------- #
# batched-coupling limitation (documented)                                    #
# --------------------------------------------------------------------------- #

def test_batched_delayed_coupling_unsupported(dt, connectome):
    """Batched whole-brain coupling is a brainstate ``prefetch_delay`` limitation.

    A batched delayed source read mis-orders the batch axis (``(N, N, B, N)``
    instead of ``(B, N, N)``), so ``DiffusiveCoupling`` rejects it. Batch a
    whole-brain network via vmap-over-parameters at the fit level instead of
    ``init_all_states(batch_size=...)``.
    """
    n = connectome["n"]
    net = brainmass.Network(
        _det_hopf(n), conn=connectome["SC"], coupled_var="x", k=0.3,
    )
    with pytest.raises(Exception):
        _run(net, 10, batch_size=2)
