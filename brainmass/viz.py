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

"""Thin, lazily-imported plotting helpers for brainmass results.

These helpers keep the tutorials and the gallery concise and visually
consistent. They are deliberately **thin**: each one surfaces a standard
matplotlib plot or a :mod:`braintools.metric` output (functional connectivity,
power spectrum) -- none reimplements a metric.

``matplotlib`` is an **optional** dependency, imported **lazily inside each
function**. Importing :mod:`brainmass` (or this module) therefore never imports
matplotlib, and ``import brainmass`` works in an environment without it.
Install the plotting extra with ``pip install brainmass[viz]``. Every function
accepts an optional ``ax=`` and returns the :class:`matplotlib.axes.Axes` it
drew on, and tolerates unit-aware (:class:`brainunit.Quantity`) inputs by
stripping the magnitude for plotting.

See Also
--------
brainmass.datasets : the example data these helpers visualise.
braintools.metric : the connectivity / spectral metrics surfaced here.
"""

import brainunit as u
import numpy as np

__all__ = [
    'plot_timeseries',
    'plot_phase_portrait',
    'plot_connectivity',
    'plot_functional_connectivity',
    'plot_power_spectrum',
]


def _import_pyplot():
    """Import :mod:`matplotlib.pyplot` lazily with a helpful error.

    Returns
    -------
    module
        The :mod:`matplotlib.pyplot` module.

    Raises
    ------
    ImportError
        If matplotlib is not installed, with a message pointing at the
        ``brainmass[viz]`` extra.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "brainmass.viz requires matplotlib, which is not installed. "
            "Install the plotting extra with `pip install brainmass[viz]`."
        ) from exc
    return plt


def _mag(x):
    """Strip units from ``x`` for plotting (a no-op for plain arrays)."""
    return np.asarray(u.get_magnitude(x))


def _resolve_ax(ax):
    """Return ``ax`` or a freshly created one."""
    if ax is None:
        plt = _import_pyplot()
        _, ax = plt.subplots()
    return ax


def plot_timeseries(signal, ts=None, *, labels=None, ax=None, **kwargs):
    """Plot one or more region time series.

    Parameters
    ----------
    signal : array_like or brainunit.Quantity
        ``(time,)`` or ``(time, region)`` data. Units are stripped for plotting.
    ts : array_like or brainunit.Quantity, optional
        The time axis (length ``time``). Defaults to sample indices.
    labels : sequence of str, optional
        Per-region legend labels; a legend is added when given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created when ``None``.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    See Also
    --------
    plot_phase_portrait : plot one variable against another.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainmass
        >>> t = np.linspace(0, 6, 100)
        >>> ax = brainmass.viz.plot_timeseries(np.sin(t), ts=t)
        >>> ax.get_xlabel()
        'time'
    """
    ax = _resolve_ax(ax)
    data = _mag(signal)
    if data.ndim == 1:
        data = data[:, None]
    x = np.arange(data.shape[0]) if ts is None else _mag(ts)
    for j in range(data.shape[1]):
        label = labels[j] if labels is not None else None
        ax.plot(x, data[:, j], label=label, **kwargs)
    ax.set_xlabel('time')
    ax.set_ylabel('signal')
    if labels is not None:
        ax.legend()
    return ax


def plot_phase_portrait(x, y, *, ax=None, **kwargs):
    """Plot a phase portrait of one variable against another.

    Parameters
    ----------
    x, y : array_like or brainunit.Quantity
        The two trajectories (same length). Units are stripped for plotting.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created when ``None``.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    See Also
    --------
    plot_timeseries : plot variables against time.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainmass
        >>> t = np.linspace(0, 6, 100)
        >>> ax = brainmass.viz.plot_phase_portrait(np.sin(t), np.cos(t))
        >>> ax.get_xlabel()
        'x'
    """
    ax = _resolve_ax(ax)
    ax.plot(_mag(x), _mag(y), **kwargs)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return ax


def plot_connectivity(matrix, *, labels=None, ax=None, cmap='viridis', colorbar=True, **kwargs):
    """Plot a connectivity (or any square) matrix as a heatmap.

    Parameters
    ----------
    matrix : array_like or brainunit.Quantity
        ``(N, N)`` matrix. Units are stripped for plotting.
    labels : sequence of str, optional
        Tick labels for both axes.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created when ``None``.
    cmap : str, optional
        Colormap name. Default ``'viridis'``.
    colorbar : bool, optional
        Whether to attach a colorbar. Default ``True``.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.imshow`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    See Also
    --------
    plot_functional_connectivity : connectivity derived from a time series.

    Examples
    --------
    .. code-block:: python

        >>> import brainmass
        >>> conn = brainmass.datasets.load_dataset('example_connectome')
        >>> ax = brainmass.viz.plot_connectivity(conn.weights)
        >>> len(ax.images)
        1
    """
    ax = _resolve_ax(ax)
    data = _mag(matrix)
    im = ax.imshow(data, cmap=cmap, **kwargs)
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
    if colorbar:
        ax.figure.colorbar(im, ax=ax)
    return ax


def plot_functional_connectivity(data, *, is_matrix=False, labels=None, ax=None, **kwargs):
    """Plot a functional-connectivity matrix.

    When ``data`` is a ``(time, region)`` time series (the default), the
    functional connectivity is computed via
    :func:`braintools.metric.functional_connectivity` and plotted. When
    ``is_matrix`` is ``True``, ``data`` is treated as an already-computed
    ``(region, region)`` FC matrix.

    Parameters
    ----------
    data : array_like or brainunit.Quantity
        Either a ``(time, region)`` time series or a ``(region, region)`` FC
        matrix (see ``is_matrix``). Units are stripped before computing FC.
    is_matrix : bool, optional
        If ``True``, ``data`` is a precomputed FC matrix. Default ``False``.
    labels : sequence of str, optional
        Tick labels for both axes.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created when ``None``.
    **kwargs
        Forwarded to :func:`plot_connectivity`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    See Also
    --------
    plot_connectivity : the underlying heatmap helper.
    braintools.metric.functional_connectivity : the surfaced metric.

    Examples
    --------
    .. code-block:: python

        >>> import brainmass
        >>> sig = brainmass.datasets.load_dataset('example_signal')
        >>> ax = brainmass.viz.plot_functional_connectivity(sig.signal)
        >>> len(ax.images)
        1
    """
    import braintools

    if is_matrix:
        fc = _mag(data)
    else:
        fc = braintools.metric.functional_connectivity(_mag(data))
    return plot_connectivity(fc, labels=labels, ax=ax, **kwargs)


def plot_power_spectrum(signal, dt, *, ax=None, loglog=True, **kwargs):
    """Plot the power spectral density of a 1-D signal.

    The PSD is computed via
    :func:`braintools.metric.power_spectral_density`.

    Parameters
    ----------
    signal : array_like or brainunit.Quantity
        ``(time,)`` signal. Units are stripped before the transform.
    dt : float or brainunit.Quantity
        The sampling step. A :class:`brainunit.Quantity` is converted to
        milliseconds; a plain float is used as-is.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure/axes is created when ``None``.
    loglog : bool, optional
        Use log-log axes. Default ``True``.
    **kwargs
        Forwarded to :meth:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    See Also
    --------
    braintools.metric.power_spectral_density : the surfaced metric.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import brainunit as u
        >>> import brainmass
        >>> t = np.linspace(0, 1, 200)
        >>> sig = np.sin(2 * np.pi * 10 * t)
        >>> ax = brainmass.viz.plot_power_spectrum(sig, dt=5.0 * u.ms)
        >>> ax.get_xlabel()
        'frequency'
    """
    import braintools

    ax = _resolve_ax(ax)
    dt_val = float(u.Quantity(dt).to(u.ms).mantissa) if isinstance(dt, u.Quantity) else float(dt)
    freqs, psd = braintools.metric.power_spectral_density(_mag(signal), dt_val)
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    if loglog:
        ax.loglog(freqs, psd, **kwargs)
    else:
        ax.plot(freqs, psd, **kwargs)
    ax.set_xlabel('frequency')
    ax.set_ylabel('power')
    return ax
