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

"""Convolution-based observation models (HRF kernels, HRF BOLD, temporal averaging).

This module adds the *convolution* fMRI BOLD path -- a hemodynamic response
function (HRF) kernel convolved with downsampled neural activity -- alongside the
physiologically detailed Balloon-Windkessel ODE in
:class:`brainmass.BOLDSignal`. It also provides :class:`TemporalAverage`, a
windowed-mean downsampling observation that complements the point-decimation
``Simulator(sample_every=)`` subsampling.

When to use which BOLD path
---------------------------
- :class:`HRFBold` (this module) -- a linear convolution of neural activity with
  a closed-form HRF kernel. Fast, simple, fully differentiable in its handful of
  scalar parameters; the natural choice for **fitting** and for quickly turning a
  long neural trajectory into a BOLD time series.
- :class:`brainmass.BOLDSignal` -- the four-state Balloon-Windkessel ODE
  (Friston et al. 2003). Physiologically detailed (vasodilatory signal, blood
  flow, volume, deoxyhemoglobin); prefer it when biophysical realism matters.

Both are kept; neither replaces the other.
"""

import math

import brainunit as u
import jax
import jax.numpy as jnp
import jax.scipy as jsp

__all__ = [
    'HRFKernel',
    'FirstOrderVolterraHRFKernel',
    'GammaHRFKernel',
    'DoubleExponentialHRFKernel',
    'MixtureOfGammasHRFKernel',
    'TemporalAverage',
    'HRFBold',
]


def _steps_per_period(period, dt):
    """Return ``round(period / dt)`` as a concrete positive Python ``int``."""
    ratio = float(u.get_magnitude(period / dt))
    w = int(round(ratio))
    if w < 1:
        raise ValueError(
            f"period {period} is shorter than dt {dt}; round(period/dt) = {w} "
            "must be >= 1."
        )
    return w


def _to_seconds(t):
    """Return the time grid ``t`` as a bare magnitude in **seconds**.

    A :class:`brainunit.Quantity` is converted with full unit checking; a plain
    array is assumed to be in **milliseconds** (the tvboptim / TVB convention)
    and divided by 1000.
    """
    if isinstance(t, u.Quantity):
        return t.to_decimal(u.second)
    return jnp.asarray(t) / 1000.0


class HRFKernel:
    r"""Base class for hemodynamic response function (HRF) kernels.

    A kernel is a closed-form function of time :math:`h(t)` defining the
    hemodynamic impulse response convolved with neural activity to form the BOLD
    signal (see :class:`HRFBold`). Subclasses implement :meth:`__call__`.

    Parameters
    ----------
    duration : brainunit.Quantity
        Temporal support of the kernel (how far out :math:`h(t)` is evaluated
        when building a convolution grid). Default ``20 * u.second``.

    Notes
    -----
    Calling a kernel evaluates :math:`h(t)` on a time grid. The grid may be a
    :class:`brainunit.Quantity` (any time unit) or a plain array, which is
    interpreted as **milliseconds**. The returned :math:`h(t)` is a dimensionless
    array.

    See Also
    --------
    HRFBold : convolution BOLD observation that consumes a kernel.
    """
    __module__ = 'brainmass'

    def __init__(self, duration=20. * u.second):
        self.duration = duration

    def __call__(self, t):
        """Evaluate :math:`h(t)` on the time grid ``t`` (abstract)."""
        raise NotImplementedError


class FirstOrderVolterraHRFKernel(HRFKernel):
    r"""First-order Volterra kernel of the hemodynamic system (TVB canonical).

    The canonical damped-oscillator HRF -- the first-order Volterra kernel of the
    Balloon/Windkessel hemodynamics (Friston et al. 2000) [1]_, ported from TVB's
    ``FirstOrderVolterra`` equation:

    .. math::

        h(t) = s \, e^{-t / (2\tau_s)} \, \frac{\sin(\omega t)}{\omega},
        \qquad
        \omega = \sqrt{\frac{1}{\tau_f} - \frac{1}{4\tau_s^2}},

    where :math:`t` is in **seconds**, :math:`\tau_s` is the signal-decay time
    constant, :math:`\tau_f` the feedback time constant and :math:`s` an amplitude
    scaling. This is the *underdamped* solution: :math:`\omega` is real only when
    :math:`4\tau_s^2 > \tau_f` (the defaults satisfy this); violating it makes
    :math:`\omega` -- and the whole kernel -- ``NaN``.

    Despite the shared name of the mathematician Vito Volterra, this is unrelated
    to Lotka-Volterra (predator-prey) dynamics.

    Parameters
    ----------
    tau_s : float
        Signal decay time constant in **seconds** (default ``0.8``).
    tau_f : float
        Feedback time constant in **seconds** (default ``0.4``).
    scaling : float
        Kernel amplitude scaling factor (default ``1 / 3``).
    duration : brainunit.Quantity
        Kernel support (default ``20 * u.second``).

    References
    ----------
    .. [1] Friston KJ, Mechelli A, Turner R, Price CJ (2000). Nonlinear responses
           in fMRI: the Balloon model, Volterra kernels, and other hemodynamics.
           NeuroImage 12(4): 466-477.

    Examples
    --------
    .. code-block:: python

        >>> import brainmass
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> k = brainmass.FirstOrderVolterraHRFKernel()
        >>> t = jnp.linspace(0., 20000., 5) * u.ms
        >>> h = k(t)
        >>> h.shape
        (5,)
        >>> float(h[0])
        0.0
    """
    __module__ = 'brainmass'

    def __init__(self, tau_s=0.8, tau_f=0.4, scaling=1.0 / 3.0,
                 duration=20. * u.second):
        super().__init__(duration=duration)
        self.tau_s = tau_s
        self.tau_f = tau_f
        self.scaling = scaling

    def __call__(self, t):
        t_s = _to_seconds(t)
        omega = jnp.sqrt(1.0 / self.tau_f - 1.0 / (4.0 * self.tau_s ** 2))
        return (self.scaling
                * jnp.exp(-0.5 * (t_s / self.tau_s))
                * jnp.sin(omega * t_s) / omega)


class GammaHRFKernel(HRFKernel):
    r"""Gamma HRF kernel (Boynton et al. 1996).

    A peak-normalised gamma probability density [1]_:

    .. math::

        h(t) \propto \frac{(t/\tau)^{n-1} e^{-t/\tau}}{\tau\,(n-1)!},

    rescaled so its peak equals the amplitude factor :math:`a` (matching TVB's
    ``Gamma`` equation). :math:`t` is in **seconds**.

    Parameters
    ----------
    tau : float
        Exponential time constant in **seconds** (default ``1.08``).
    n : float
        Phase-delay / shape parameter (default ``3.0``).
    a : float
        Amplitude after peak-normalisation (default ``0.1``).
    duration : brainunit.Quantity
        Kernel support (default ``20 * u.second``).

    References
    ----------
    .. [1] Boynton GM, Engel SA, Glover GH, Heeger DJ (1996). Linear systems
           analysis of functional magnetic resonance imaging in human V1.
           J Neurosci 16(13): 4207-4221.

    Examples
    --------
    .. code-block:: python

        >>> import brainmass
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> k = brainmass.GammaHRFKernel()
        >>> h = k(jnp.linspace(0., 20000., 256) * u.ms)
        >>> bool(jnp.isclose(h.max(), 0.1, rtol=1e-5))
        True
    """
    __module__ = 'brainmass'

    def __init__(self, tau=1.08, n=3.0, a=0.1, duration=20. * u.second):
        super().__init__(duration=duration)
        self.tau = tau
        self.n = n
        self.a = a

    def __call__(self, t):
        t_s = _to_seconds(t)
        factorial = math.factorial(int(self.n) - 1)
        kernel = ((t_s / self.tau) ** (self.n - 1)
                  * jnp.exp(-(t_s / self.tau))) / (self.tau * factorial)
        peak = jnp.max(kernel)
        peak = jnp.where(peak > 0, peak, 1.0)
        return kernel / peak * self.a


class DoubleExponentialHRFKernel(HRFKernel):
    r"""Double-exponential (damped-oscillation difference) HRF kernel.

    A difference of two damped sinusoids (Polonsky et al. 2000) [1]_:

    .. math::

        h(t) \propto a_1 e^{-t/\tau_1}\sin(2\pi f_1 t)
                    - a_2 e^{-t/\tau_2}\sin(2\pi f_2 t),

    peak-normalised and rescaled to amplitude :math:`a` (matching TVB's
    ``DoubleExponential`` equation). :math:`t` is in **seconds**.

    Parameters
    ----------
    tau_1, tau_2 : float
        Time constants of the two exponentials in **seconds**
        (defaults ``7.22``, ``7.4``).
    f_1, f_2 : float
        Frequencies of the two sinusoids in **Hz** (defaults ``0.03``, ``0.12``).
    amp_1, amp_2 : float
        Amplitudes of the two terms (defaults ``0.1``, ``0.1``).
    a : float
        Amplitude after peak-normalisation (default ``0.1``).
    duration : brainunit.Quantity
        Kernel support (default ``40 * u.second``).

    References
    ----------
    .. [1] Polonsky A, Blake R, Braun J, Heeger DJ (2000). Neuronal activity in
           human primary visual cortex correlates with perception during
           binocular rivalry. Nat Neurosci 3(11): 1153-1159.

    Examples
    --------
    .. code-block:: python

        >>> import brainmass
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> k = brainmass.DoubleExponentialHRFKernel()
        >>> h = k(jnp.linspace(0., 40000., 512) * u.ms)
        >>> bool(jnp.isclose(h.max(), 0.1, rtol=1e-5))
        True
    """
    __module__ = 'brainmass'

    def __init__(self, tau_1=7.22, tau_2=7.4, f_1=0.03, f_2=0.12,
                 amp_1=0.1, amp_2=0.1, a=0.1, duration=40. * u.second):
        super().__init__(duration=duration)
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.f_1 = f_1
        self.f_2 = f_2
        self.amp_1 = amp_1
        self.amp_2 = amp_2
        self.a = a

    def __call__(self, t):
        t_s = _to_seconds(t)
        kernel = (self.amp_1 * jnp.exp(-t_s / self.tau_1)
                  * jnp.sin(2 * math.pi * self.f_1 * t_s)
                  - self.amp_2 * jnp.exp(-t_s / self.tau_2)
                  * jnp.sin(2 * math.pi * self.f_2 * t_s))
        peak = jnp.max(kernel)
        peak = jnp.where(peak > 0, peak, 1.0)
        return kernel / peak * self.a


class MixtureOfGammasHRFKernel(HRFKernel):
    r"""Mixture-of-gammas HRF kernel (Glover 1999).

    A difference of two gamma densities -- the canonical SPM-style HRF [1]_:

    .. math::

        h(t) = \frac{(\lambda t)^{a_1 - 1} e^{-\lambda t}}{\Gamma(a_1)}
             - c\,\frac{(\lambda t)^{a_2 - 1} e^{-\lambda t}}{\Gamma(a_2)},

    with :math:`t` in **seconds**. The first gamma models the positive BOLD peak;
    the second, scaled by :math:`c`, the post-stimulus undershoot.

    Parameters
    ----------
    a_1 : float
        Shape parameter of the first (peak) gamma (default ``6.0``).
    a_2 : float
        Shape parameter of the second (undershoot) gamma (default ``13.0``).
    l : float
        Rate / inverse-scale parameter :math:`\lambda` (default ``1.0``).
    c : float
        Relative amplitude of the undershoot gamma (default ``0.4``).
    duration : brainunit.Quantity
        Kernel support (default ``20 * u.second``).

    References
    ----------
    .. [1] Glover GH (1999). Deconvolution of impulse response in event-related
           BOLD fMRI. NeuroImage 9(4): 416-429.

    Examples
    --------
    .. code-block:: python

        >>> import brainmass
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> k = brainmass.MixtureOfGammasHRFKernel()
        >>> h = k(jnp.linspace(0., 20000., 256) * u.ms)
        >>> h.shape
        (256,)
    """
    __module__ = 'brainmass'

    def __init__(self, a_1=6.0, a_2=13.0, l=1.0, c=0.4, duration=20. * u.second):
        super().__init__(duration=duration)
        self.a_1 = a_1
        self.a_2 = a_2
        self.l = l
        self.c = c

    def __call__(self, t):
        t_s = _to_seconds(t)
        g1 = jsp.special.gamma(self.a_1)
        g2 = jsp.special.gamma(self.a_2)
        return ((self.l * t_s) ** (self.a_1 - 1) * jnp.exp(-self.l * t_s) / g1
                - self.c * (self.l * t_s) ** (self.a_2 - 1)
                * jnp.exp(-self.l * t_s) / g2)


class TemporalAverage:
    r"""Downsample a trajectory by averaging over non-overlapping time windows.

    For a window of ``w = round(period / dt)`` samples, the ``k``-th output is the
    mean of the ``k``-th block:

    .. math::

        y_k = \frac{1}{w}\sum_{j=0}^{w-1} y[k\,w + j],
        \qquad k = 0, \dots, \lfloor T / w \rfloor - 1.

    Trailing samples that do not fill a complete window are dropped. The output is
    sampled at ``period`` (i.e. ``n_win = T // w`` rows).

    This is a thin, standalone observation -- it does **not** modify
    :class:`brainmass.Simulator`. It is the *averaging* complement to the
    point-decimation ``Simulator(sample_every=k)`` subsampling: ``sample_every``
    keeps every ``k``-th sample, whereas :class:`TemporalAverage` averages each
    window (smoother, anti-aliased). Apply it as a post-transform on a run
    trajectory (``ta(res['output'], dt)``); it is also used internally by
    :class:`HRFBold` to reduce neural activity to the convolution grid.

    Parameters
    ----------
    period : brainunit.Quantity
        Averaging-window length (default ``4 * u.ms``).

    See Also
    --------
    brainmass.Simulator : ``sample_every=`` does point-decimation subsampling.
    HRFBold : uses :class:`TemporalAverage` to build its convolution grid.

    Examples
    --------
    .. code-block:: python

        >>> import brainmass
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> signal = jnp.arange(20.).reshape(20, 1)
        >>> ta = brainmass.TemporalAverage(period=5. * u.ms)
        >>> y = ta(signal, dt=1. * u.ms)
        >>> y.shape
        (4, 1)
        >>> [float(v) for v in y[:, 0]]
        [2.0, 7.0, 12.0, 17.0]
    """
    __module__ = 'brainmass'

    def __init__(self, period=4. * u.ms):
        self.period = period

    def __call__(self, signal, dt):
        r"""Average ``signal`` over non-overlapping ``period``-wide windows.

        Parameters
        ----------
        signal : array-like or brainunit.Quantity
            Trajectory with time on axis 0 and arbitrary trailing dimensions,
            shape ``(T, *rest)``.
        dt : brainunit.Quantity
            Sampling step of ``signal`` (the simulation time step).

        Returns
        -------
        array-like or brainunit.Quantity
            Windowed means, shape ``(T // w, *rest)`` where
            ``w = round(period / dt)``. Units are preserved.
        """
        w = _steps_per_period(self.period, dt)
        unit = u.get_unit(signal) if isinstance(signal, u.Quantity) else None
        mag = u.get_magnitude(signal)
        n_win = mag.shape[0] // w
        trimmed = mag[:n_win * w]
        windows = trimmed.reshape((n_win, w) + mag.shape[1:])
        avg = jnp.mean(windows, axis=1)
        return avg * unit if unit is not None else avg


def _convolve_columns(sig2d, hrf, mode, method):
    """Convolve each column of ``sig2d`` (shape ``(L, M)``) with ``hrf``."""
    if method == 'fft':
        def conv(col):
            return jsp.signal.fftconvolve(col, hrf, mode=mode)
    else:
        def conv(col):
            return jnp.convolve(col, hrf, mode=mode)
    return jax.vmap(conv, in_axes=1, out_axes=1)(sig2d)


class HRFBold:
    r"""Convolution-based fMRI BOLD observation (HRF kernel).

    Forms the BOLD signal by convolving downsampled neural activity with a
    closed-form hemodynamic response function (HRF) kernel, then decimating to the
    repetition time (TR). The pipeline is:

    1. **Downsample** neural activity from the simulation step ``dt`` to
       ``downsample_period`` with a :class:`TemporalAverage`.
    2. **Convolve** each region's series (after prepending a warm-up ``history``,
       zeros by default) with ``kernel`` evaluated on the ``downsample_period``
       grid.
    3. **Scale** to BOLD: :math:`\mathrm{BOLD} = k_1 V_0 (c - 1)`, where :math:`c`
       is the convolution output (the ``-1`` removes the unit baseline, matching
       the TVB convention).
    4. **Decimate** to the TR by taking every ``round(period / downsample_period)``
       sample.

    Unlike the four-state Balloon-Windkessel ODE in :class:`brainmass.BOLDSignal`,
    this is a single linear convolution -- fast, simple and fully differentiable in
    its scalar parameters, which makes it the natural choice for **fitting**.
    Prefer :class:`brainmass.BOLDSignal` when biophysical realism matters. Both are
    kept.

    Parameters
    ----------
    k_1 : float
        BOLD signal scaling factor (default ``5.6``).
    V_0 : float
        Resting blood volume fraction (default ``0.02``).
    period : brainunit.Quantity
        Final BOLD sampling period / TR (default ``1000 * u.ms``).
    downsample_period : brainunit.Quantity
        Intermediate downsampling period the kernel is sampled on
        (default ``4 * u.ms``).
    kernel : HRFKernel
        HRF kernel to convolve with (default
        :class:`FirstOrderVolterraHRFKernel`).
    convolution_mode : {'valid', 'same', 'full'}
        Convolution edge mode (default ``'valid'``; with the zero ``history``
        prepend this yields a causal output of length ``T_ds + 1``).
    method : {'fft', 'direct'}
        ``'fft'`` uses :func:`jax.scipy.signal.fftconvolve`; ``'direct'`` uses
        :func:`jax.numpy.convolve`. They agree numerically (default ``'fft'``).
    history : array-like, optional
        Warm-up buffer prepended before convolution, shape
        ``(round(duration / downsample_period), *regions)``. ``None`` (default)
        prepends zeros.

    See Also
    --------
    brainmass.BOLDSignal : the physiologically detailed Balloon-Windkessel ODE BOLD.
    TemporalAverage : the windowed-mean downsampler used internally.

    Examples
    --------
    .. code-block:: python

        >>> import brainmass
        >>> import brainunit as u
        >>> import jax.numpy as jnp
        >>> t = jnp.arange(2000.)
        >>> z = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * t[:, None] / 800.0)
        >>> bold = brainmass.HRFBold(
        ...     period=200. * u.ms, downsample_period=4. * u.ms,
        ...     kernel=brainmass.FirstOrderVolterraHRFKernel(duration=400. * u.ms),
        ... )
        >>> y = bold(z, dt=1. * u.ms)
        >>> y.shape[1]
        1
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        k_1=5.6,
        V_0=0.02,
        period=1000. * u.ms,
        downsample_period=4. * u.ms,
        kernel=None,
        convolution_mode='valid',
        method='fft',
        history=None,
    ):
        if convolution_mode not in ('valid', 'same', 'full'):
            raise ValueError(
                f"convolution_mode must be 'valid', 'same' or 'full', got "
                f"{convolution_mode!r}."
            )
        if method not in ('fft', 'direct'):
            raise ValueError(f"method must be 'fft' or 'direct', got {method!r}.")
        self.k_1 = k_1
        self.V_0 = V_0
        self.period = period
        self.downsample_period = downsample_period
        self.kernel = kernel if kernel is not None else FirstOrderVolterraHRFKernel()
        self.convolution_mode = convolution_mode
        self.method = method
        self.history = history
        self._downsampler = TemporalAverage(period=downsample_period)

    def __call__(self, signal, dt):
        r"""Map a neural-activity trajectory to a BOLD time series.

        Parameters
        ----------
        signal : array-like or brainunit.Quantity
            Neural-activity proxy with time on axis 0, shape ``(T, *regions)``.
            Only its magnitude is used (BOLD is a normalised observable).
        dt : brainunit.Quantity
            Simulation time step of ``signal``.

        Returns
        -------
        jax.Array
            Dimensionless BOLD signal sampled at ``period``, shape
            ``(T_bold, *regions)``.
        """
        # 1) downsample neural activity to the convolution grid (magnitude only)
        ys = u.get_magnitude(self._downsampler(signal, dt))
        rest = ys.shape[1:]

        # 2) HRF kernel evaluated on the downsample grid
        dur_ms = float(self.kernel.duration.to_decimal(u.ms))
        ksamp = int(math.ceil(
            float(u.get_magnitude(self.kernel.duration / self.downsample_period))
        ))
        kernel_time = jnp.linspace(0.0, dur_ms, ksamp) * u.ms
        hrf = self.kernel(kernel_time)

        # 3) prepend warm-up history (zeros by default)
        if self.history is None:
            pad = jnp.zeros((ksamp,) + rest, dtype=ys.dtype)
        else:
            pad = u.get_magnitude(self.history)
        ys_h = jnp.concatenate([pad, ys], axis=0)

        # 4) convolve each region column with the kernel
        sig2d = ys_h.reshape(ys_h.shape[0], -1)
        conv = _convolve_columns(sig2d, hrf, self.convolution_mode, self.method)

        # 5) BOLD scaling (the -1 removes the unit baseline, TVB convention)
        bold = self.k_1 * self.V_0 * (conv - 1.0)

        # 6) decimate to the TR
        step = _steps_per_period(self.period, self.downsample_period)
        idx = jnp.arange(step, bold.shape[0], step)
        bold = bold[idx]
        return bold.reshape((bold.shape[0],) + rest)
