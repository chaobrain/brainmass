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

"""Tests for :mod:`brainmass.observation`.

The HRF kernels are validated against *embedded* verbatim transcriptions of
tvboptim's closed-form kernel maths (the embedded-reference-oracle pattern from
goals 09/10 -- always-on, no live ``tvboptim`` import), plus the analytic
properties each kernel must satisfy (``h(0) == 0``, peak normalisation, real
underdamped frequency), unit-awareness, gradient-safety, and vmap-safety.
"""

import math

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

import brainunit as u

from brainmass import observation


# --------------------------------------------------------------------------- #
# Embedded tvboptim kernel oracles (verbatim closed forms; t in milliseconds)  #
# --------------------------------------------------------------------------- #

def _ref_first_order_volterra(t_ms, tau_s=0.8, tau_f=0.4, scaling=1.0 / 3.0):
    t_s = t_ms / 1000.0
    omega = jnp.sqrt(1.0 / tau_f - 1.0 / (4.0 * tau_s ** 2))
    return scaling * jnp.exp(-0.5 * (t_s / tau_s)) * jnp.sin(omega * t_s) / omega


def _ref_gamma(t_ms, tau=1.08, n=3.0, a=0.1):
    t_s = t_ms / 1000.0
    factorial = math.factorial(int(n) - 1)
    kernel = ((t_s / tau) ** (n - 1) * jnp.exp(-(t_s / tau))) / (tau * factorial)
    peak = jnp.max(kernel)
    peak = jnp.where(peak > 0, peak, 1.0)
    return kernel / peak * a


def _ref_double_exp(t_ms, tau_1=7.22, tau_2=7.4, f_1=0.03, f_2=0.12,
                    amp_1=0.1, amp_2=0.1, a=0.1):
    t_s = t_ms / 1000.0
    kernel = (amp_1 * jnp.exp(-t_s / tau_1) * jnp.sin(2 * math.pi * f_1 * t_s)
              - amp_2 * jnp.exp(-t_s / tau_2) * jnp.sin(2 * math.pi * f_2 * t_s))
    peak = jnp.max(kernel)
    peak = jnp.where(peak > 0, peak, 1.0)
    return kernel / peak * a


def _ref_mixture_gammas(t_ms, a_1=6.0, a_2=13.0, l=1.0, c=0.4):
    t_s = t_ms / 1000.0
    g1 = jsp.special.gamma(a_1)
    g2 = jsp.special.gamma(a_2)
    return ((l * t_s) ** (a_1 - 1) * jnp.exp(-l * t_s) / g1
            - c * (l * t_s) ** (a_2 - 1) * jnp.exp(-l * t_s) / g2)


@pytest.fixture
def grid_ms():
    """A fixed time grid in milliseconds spanning a 20 s kernel support."""
    return jnp.linspace(0.0, 20_000.0, 256)


_KERNELS = [
    (observation.FirstOrderVolterraHRFKernel, _ref_first_order_volterra),
    (observation.GammaHRFKernel, _ref_gamma),
    (observation.DoubleExponentialHRFKernel, _ref_double_exp),
    (observation.MixtureOfGammasHRFKernel, _ref_mixture_gammas),
]


# --------------------------------------------------------------------------- #
# Closed-form / embedded-oracle equality                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('kernel_cls, ref', _KERNELS)
def test_kernel_matches_embedded_oracle(kernel_cls, ref, grid_ms):
    got = kernel_cls()(grid_ms * u.ms)
    expected = ref(grid_ms)
    assert np.allclose(np.asarray(got), np.asarray(expected), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('kernel_cls, ref', _KERNELS)
def test_kernel_returns_dimensionless(kernel_cls, ref, grid_ms):
    got = kernel_cls()(grid_ms * u.ms)
    # h(t) is a dimensionless array, not a Quantity.
    assert not isinstance(got, u.Quantity)
    assert np.asarray(got).shape == (grid_ms.shape[0],)


@pytest.mark.parametrize('kernel_cls, ref', _KERNELS)
def test_kernel_is_zero_at_origin(kernel_cls, ref, grid_ms):
    got = kernel_cls()(grid_ms * u.ms)
    assert np.isclose(float(np.asarray(got)[0]), 0.0, atol=1e-8)


def test_gamma_peak_equals_amplitude(grid_ms):
    k = observation.GammaHRFKernel(a=0.1)
    got = np.asarray(k(grid_ms * u.ms))
    assert np.isclose(got.max(), 0.1, rtol=1e-6)


def test_double_exp_peak_equals_amplitude(grid_ms):
    k = observation.DoubleExponentialHRFKernel(a=0.1)
    got = np.asarray(k(grid_ms * u.ms))
    assert np.isclose(got.max(), 0.1, rtol=1e-6)


def test_first_order_volterra_omega_is_real():
    # Defaults satisfy 4*tau_s**2 > tau_f so the underdamped omega is real.
    k = observation.FirstOrderVolterraHRFKernel()
    omega = math.sqrt(1.0 / k.tau_f - 1.0 / (4.0 * k.tau_s ** 2))
    assert omega > 0


# --------------------------------------------------------------------------- #
# Unit awareness                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('kernel_cls, ref', _KERNELS)
def test_kernel_unit_aware_ms_vs_seconds(kernel_cls, ref, grid_ms):
    k = kernel_cls()
    h_ms = np.asarray(k(grid_ms * u.ms))
    h_s = np.asarray(k((grid_ms / 1000.0) * u.second))
    assert np.allclose(h_ms, h_s, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize('kernel_cls, ref', _KERNELS)
def test_kernel_plain_array_is_milliseconds(kernel_cls, ref, grid_ms):
    k = kernel_cls()
    h_plain = np.asarray(k(grid_ms))          # plain array assumed ms
    h_quant = np.asarray(k(grid_ms * u.ms))
    assert np.allclose(h_plain, h_quant, rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------- #
# Gradient- and vmap-safety                                                     #
# --------------------------------------------------------------------------- #

def test_first_order_volterra_grad_through_tau_s(grid_ms):
    def loss(tau_s):
        k = observation.FirstOrderVolterraHRFKernel(tau_s=tau_s)
        return jnp.sum(k(grid_ms * u.ms))

    g = jax.grad(loss)(0.8)
    assert np.isfinite(float(g))
    assert float(g) != 0.0


def test_gamma_grad_through_tau(grid_ms):
    def loss(tau):
        return jnp.sum(observation.GammaHRFKernel(tau=tau)(grid_ms * u.ms))

    g = jax.grad(loss)(1.08)
    assert np.isfinite(float(g))


def test_mixture_gammas_grad_through_lambda(grid_ms):
    def loss(l):
        return jnp.sum(observation.MixtureOfGammasHRFKernel(l=l)(grid_ms * u.ms))

    g = jax.grad(loss)(1.0)
    assert np.isfinite(float(g))


@pytest.mark.parametrize('kernel_cls, ref', _KERNELS)
def test_kernel_vmap_over_grids(kernel_cls, ref, grid_ms):
    k = kernel_cls()
    grids = jnp.stack([grid_ms, grid_ms * 0.5]) * u.ms
    out = jax.vmap(k)(grids)
    assert np.asarray(out).shape == (2, grid_ms.shape[0])
    assert np.allclose(np.asarray(out)[0], np.asarray(k(grid_ms * u.ms)),
                       rtol=1e-6, atol=1e-8)


def test_base_kernel_is_abstract(grid_ms):
    with pytest.raises(NotImplementedError):
        observation.HRFKernel()(grid_ms * u.ms)


# --------------------------------------------------------------------------- #
# TemporalAverage                                                              #
# --------------------------------------------------------------------------- #

def test_temporal_average_matches_manual_window_mean():
    rng = np.random.default_rng(0)
    signal = jnp.asarray(rng.standard_normal((40, 3)))
    avg = observation.TemporalAverage(period=4. * u.ms)(signal, dt=1. * u.ms)
    # w = 4, n_win = 10; y[k] = mean(signal[4k:4k+4]).
    expected = np.asarray(signal).reshape(10, 4, 3).mean(axis=1)
    assert np.asarray(avg).shape == (10, 3)
    assert np.allclose(np.asarray(avg), expected)


def test_temporal_average_drops_trailing_partial_window():
    signal = jnp.arange(43. * 2).reshape(43, 2)
    avg = observation.TemporalAverage(period=4. * u.ms)(signal, dt=1. * u.ms)
    # 43 // 4 = 10 complete windows; the final 3 samples are dropped.
    assert np.asarray(avg).shape == (10, 2)
    expected = np.asarray(signal)[:40].reshape(10, 4, 2).mean(axis=1)
    assert np.allclose(np.asarray(avg), expected)


def test_temporal_average_non_integer_period_rounds():
    signal = jnp.arange(100.).reshape(100, 1)
    # period/dt = 4 / 0.3 = 13.33 -> w = round(13.33) = 13; n_win = 100 // 13 = 7.
    avg = observation.TemporalAverage(period=4. * u.ms)(signal, dt=0.3 * u.ms)
    assert np.asarray(avg).shape == (7, 1)
    expected = np.asarray(signal)[:91].reshape(7, 13, 1).mean(axis=1)
    assert np.allclose(np.asarray(avg), expected)


def test_temporal_average_preserves_units():
    signal = jnp.ones((20, 2)) * u.mV
    avg = observation.TemporalAverage(period=5. * u.ms)(signal, dt=1. * u.ms)
    assert isinstance(avg, u.Quantity)
    assert avg.unit == u.mV
    assert np.allclose(np.asarray(u.get_magnitude(avg)), 1.0)


def test_temporal_average_1d_signal():
    signal = jnp.arange(20.)
    avg = observation.TemporalAverage(period=5. * u.ms)(signal, dt=1. * u.ms)
    assert np.asarray(avg).shape == (4,)
    expected = np.asarray(signal).reshape(4, 5).mean(axis=1)
    assert np.allclose(np.asarray(avg), expected)


def test_temporal_average_batched_trailing_dims():
    signal = jnp.ones((30, 4, 5))
    avg = observation.TemporalAverage(period=3. * u.ms)(signal, dt=1. * u.ms)
    assert np.asarray(avg).shape == (10, 4, 5)


def test_temporal_average_jit_safe():
    signal = jnp.ones((30, 2))
    ta = observation.TemporalAverage(period=3. * u.ms)
    fn = jax.jit(lambda s: ta(s, dt=1. * u.ms))
    assert np.asarray(fn(signal)).shape == (10, 2)


def test_temporal_average_period_shorter_than_dt_raises():
    # round(period/dt) = round(0.5) = 0 -> a window narrower than one sample.
    signal = jnp.ones((10, 1))
    with pytest.raises(ValueError):
        observation.TemporalAverage(period=1. * u.ms)(signal, dt=2. * u.ms)


# --------------------------------------------------------------------------- #
# HRFBold -- independent numpy/scipy oracle                                    #
# --------------------------------------------------------------------------- #

from scipy.signal import fftconvolve as _scipy_fftconvolve  # noqa: E402


def _ref_hrfbold(signal, dt_ms, kernel, k_1=5.6, V_0=0.02, period_ms=1000.,
                 downsample_ms=4., mode='valid', history=None):
    """Independent numpy/scipy reproduction of the HRFBold pipeline."""
    sig = np.asarray(u.get_magnitude(signal), dtype=np.float64)
    rest = sig.shape[1:]
    # 1) temporal-average downsample dt -> downsample_period
    w = int(round(downsample_ms / dt_ms))
    n_win = sig.shape[0] // w
    ys = sig[:n_win * w].reshape((n_win, w) + rest).mean(axis=1)
    # 2) HRF kernel on the downsample grid
    dur_ms = float(kernel.duration.to_decimal(u.ms))
    ksamp = int(math.ceil(dur_ms / downsample_ms))
    ktime = np.linspace(0.0, dur_ms, ksamp)
    hrf = np.asarray(kernel(ktime), dtype=np.float64)
    # 3) prepend history (zeros by default)
    if history is None:
        pad = np.zeros((ksamp,) + rest, dtype=np.float64)
    else:
        pad = np.asarray(u.get_magnitude(history), dtype=np.float64)
    ysh = np.concatenate([pad, ys], axis=0)
    # 4) convolve each column independently with scipy
    cols = ysh.reshape(ysh.shape[0], -1)
    out = np.stack(
        [_scipy_fftconvolve(cols[:, j], hrf, mode=mode) for j in range(cols.shape[1])],
        axis=1,
    )
    # 5) BOLD scaling
    bold = k_1 * V_0 * (out - 1.0)
    # 6) decimate to TR
    step = int(round(period_ms / downsample_ms))
    idx = np.arange(step, bold.shape[0], step)
    return bold[idx].reshape((-1,) + rest)


@pytest.fixture
def neural_signal():
    """A slow multi-region neural drive sampled at dt = 1 ms."""
    t = jnp.arange(2000.)            # 2000 ms at dt=1ms
    phases = jnp.asarray([0.0, 0.6, 1.2])
    sig = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * t[:, None] / 800.0 + phases[None, :])
    return sig                       # (2000, 3), hovers around 1


@pytest.fixture
def small_kernel():
    return observation.FirstOrderVolterraHRFKernel(duration=400. * u.ms)


def test_hrfbold_matches_independent_oracle(neural_signal, small_kernel):
    bold = observation.HRFBold(
        period=200. * u.ms, downsample_period=4. * u.ms, kernel=small_kernel,
    )
    got = bold(neural_signal, dt=1. * u.ms)
    expected = _ref_hrfbold(neural_signal, 1.0, small_kernel,
                            period_ms=200., downsample_ms=4., mode='valid')
    assert np.asarray(got).shape == expected.shape
    assert np.allclose(np.asarray(got), expected, rtol=1e-4, atol=1e-5)


def test_hrfbold_fft_equals_direct(neural_signal, small_kernel):
    bold_fft = observation.HRFBold(period=200. * u.ms, downsample_period=4. * u.ms,
                                   kernel=small_kernel, method='fft')
    bold_dir = observation.HRFBold(period=200. * u.ms, downsample_period=4. * u.ms,
                                   kernel=small_kernel, method='direct')
    a = np.asarray(bold_fft(neural_signal, dt=1. * u.ms))
    b = np.asarray(bold_dir(neural_signal, dt=1. * u.ms))
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('mode', ['valid', 'same', 'full'])
def test_hrfbold_conv_modes_run(neural_signal, small_kernel, mode):
    bold = observation.HRFBold(period=200. * u.ms, downsample_period=4. * u.ms,
                               kernel=small_kernel, convolution_mode=mode)
    got = np.asarray(bold(neural_signal, dt=1. * u.ms))
    expected = _ref_hrfbold(neural_signal, 1.0, small_kernel,
                            period_ms=200., downsample_ms=4., mode=mode)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_hrfbold_preserves_region_dimension(neural_signal, small_kernel):
    bold = observation.HRFBold(period=200. * u.ms, downsample_period=4. * u.ms,
                               kernel=small_kernel)
    got = np.asarray(bold(neural_signal, dt=1. * u.ms))
    assert got.ndim == 2
    assert got.shape[1] == 3          # regions preserved


def test_hrfbold_grad_through_k1(neural_signal, small_kernel):
    def loss(k_1):
        bold = observation.HRFBold(k_1=k_1, period=200. * u.ms,
                                   downsample_period=4. * u.ms, kernel=small_kernel)
        return jnp.sum(bold(neural_signal, dt=1. * u.ms))

    g = jax.grad(loss)(5.6)
    assert np.isfinite(float(g))
    assert float(g) != 0.0


def test_hrfbold_grad_through_kernel_param(neural_signal):
    def loss(tau_s):
        kernel = observation.FirstOrderVolterraHRFKernel(tau_s=tau_s,
                                                         duration=400. * u.ms)
        bold = observation.HRFBold(period=200. * u.ms, downsample_period=4. * u.ms,
                                   kernel=kernel)
        return jnp.sum(bold(neural_signal, dt=1. * u.ms))

    g = jax.grad(loss)(0.8)
    assert np.isfinite(float(g))


def test_hrfbold_batched_via_vmap(neural_signal, small_kernel):
    batch = jnp.stack([neural_signal, neural_signal * 1.1])  # (2, 2000, 3)
    bold = observation.HRFBold(period=200. * u.ms, downsample_period=4. * u.ms,
                               kernel=small_kernel)
    out = jax.vmap(lambda s: bold(s, dt=1. * u.ms))(batch)
    single = np.asarray(bold(neural_signal, dt=1. * u.ms))
    assert np.asarray(out).shape == (2,) + single.shape
    assert np.allclose(np.asarray(out)[0], single, rtol=1e-4, atol=1e-5)


def test_hrfbold_history_warm_start_changes_output(neural_signal, small_kernel):
    ksamp = int(math.ceil(400. / 4.))            # kernel samples on 4 ms grid
    history = jnp.ones((ksamp, 3))               # non-zero warm-start
    bold_hist = observation.HRFBold(period=200. * u.ms, downsample_period=4. * u.ms,
                                    kernel=small_kernel, history=history)
    bold_zero = observation.HRFBold(period=200. * u.ms, downsample_period=4. * u.ms,
                                    kernel=small_kernel)
    a = np.asarray(bold_hist(neural_signal, dt=1. * u.ms))
    b = np.asarray(bold_zero(neural_signal, dt=1. * u.ms))
    expected = _ref_hrfbold(neural_signal, 1.0, small_kernel, period_ms=200.,
                            downsample_ms=4., mode='valid', history=history)
    assert a.shape == expected.shape
    assert np.allclose(a, expected, rtol=1e-4, atol=1e-5)
    assert not np.allclose(a, b)                 # history changes the result


def test_hrfbold_invalid_convolution_mode_raises():
    with pytest.raises(ValueError):
        observation.HRFBold(convolution_mode='nope')


def test_hrfbold_invalid_method_raises():
    with pytest.raises(ValueError):
        observation.HRFBold(method='nope')


def test_hrfbold_default_kernel_is_first_order_volterra():
    bold = observation.HRFBold()
    assert isinstance(bold.kernel, observation.FirstOrderVolterraHRFKernel)


def test_hrfbold_downsample_period_not_multiple_of_dt(neural_signal, small_kernel):
    # downsample_period/dt = 4 / 0.3 = 13.33 -> rounds to 13; must still run.
    bold = observation.HRFBold(period=100. * u.ms, downsample_period=4. * u.ms,
                               kernel=small_kernel)
    got = np.asarray(bold(neural_signal, dt=0.3 * u.ms))
    expected = _ref_hrfbold(neural_signal, 0.3, small_kernel, period_ms=100.,
                            downsample_ms=4., mode='valid')
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_hrfbold_correlates_with_balloon_windkessel():
    """Qualitative: convolution BOLD tracks the Balloon-Windkessel ODE BOLD.

    The Balloon-Windkessel ODE is dimensionless and integrated with a *unitless*
    ``dt`` (seconds); ``HRFBold`` receives the matching real-time step as a
    Quantity (0.01 s == 10 ms). Both are driven by the same slow neural drive.
    """
    import brainmass
    import brainstate

    brainstate.random.seed(0)
    dt_bw = 0.01                        # seconds, unitless (BW ODE is dimensionless)
    n_steps = 6000                      # 60 s
    t_s = jnp.arange(n_steps) * dt_bw   # seconds
    # slow neural oscillation (period 20 s) hovering around 1
    z = 1.0 + 0.4 * jnp.sin(2 * jnp.pi * t_s / 20.0)
    z = jnp.broadcast_to(z[:, None], (n_steps, 1))

    # Balloon-Windkessel ODE BOLD (unitless dt)
    balloon = brainmass.BOLDSignal(in_size=1)
    with brainstate.environ.context(dt=dt_bw):
        brainstate.nn.init_all_states(balloon)

        def bw_step(i):
            balloon.update(z[i])
            return balloon.bold()

        balloon_bold = brainstate.transform.for_loop(bw_step, jnp.arange(n_steps))
    balloon_bold = np.asarray(u.get_magnitude(balloon_bold)).reshape(-1)

    # convolution BOLD on the same drive, real-time dt = 10 ms, TR = 1 s
    dt_q = 10. * u.ms
    hrf = observation.HRFBold(
        period=1000. * u.ms, downsample_period=20. * u.ms,
        kernel=observation.FirstOrderVolterraHRFKernel(duration=20000. * u.ms),
    )
    hrf_bold = np.asarray(hrf(z, dt=dt_q)).reshape(-1)

    # downsample the balloon BOLD to the same TR and align lengths
    balloon_tr = np.asarray(
        observation.TemporalAverage(period=1000. * u.ms)(
            jnp.asarray(balloon_bold)[:, None], dt=dt_q
        )
    ).reshape(-1)
    n = min(len(balloon_tr), len(hrf_bold))
    # drop the first few TRs (hemodynamic warm-up) before correlating
    a, b = balloon_tr[5:n], hrf_bold[5:n]
    # The two hemodynamic models have different intrinsic latencies, so compare
    # them up to a hemodynamic lag (best correlation over +/- 6 TR shifts) -- the
    # standard way to compare BOLD waveforms with differing peak delays.
    best = max(
        np.corrcoef(a[lag:], b[:len(b) - lag])[0, 1] if lag >= 0
        else np.corrcoef(a[:len(a) + lag], b[-lag:])[0, 1]
        for lag in range(-6, 7)
    )
    assert best > 0.9
