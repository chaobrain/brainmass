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

import builtins

import numpy as np
import brainunit as u
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import brainmass
from brainmass import viz


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def signal():
    t = np.linspace(0, 4 * np.pi, 200)
    return np.stack([np.sin(t), np.cos(t), np.sin(2 * t)], axis=1)


# ---------------------------------------------------------------------------
# plot_timeseries
# ---------------------------------------------------------------------------

def test_plot_timeseries_returns_axes(signal):
    ax = viz.plot_timeseries(signal)
    assert isinstance(ax, plt.Axes)
    # one line per region
    assert len(ax.lines) == signal.shape[1]


def test_plot_timeseries_accepts_existing_ax(signal):
    fig, ax = plt.subplots()
    out = viz.plot_timeseries(signal, ax=ax)
    assert out is ax


def test_plot_timeseries_with_explicit_times(signal):
    ts = np.arange(signal.shape[0]) * 0.5
    ax = viz.plot_timeseries(signal, ts=ts)
    assert isinstance(ax, plt.Axes)


def test_plot_timeseries_unit_aware_inputs(signal):
    sig = signal * u.mV
    ts = np.arange(signal.shape[0]) * u.ms
    ax = viz.plot_timeseries(sig, ts=ts)
    assert isinstance(ax, plt.Axes)


def test_plot_timeseries_1d_signal():
    ax = viz.plot_timeseries(np.sin(np.linspace(0, 6, 50)))
    assert len(ax.lines) == 1


def test_plot_timeseries_labels(signal):
    ax = viz.plot_timeseries(signal, labels=['a', 'b', 'c'])
    legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_texts == ['a', 'b', 'c']


# ---------------------------------------------------------------------------
# plot_phase_portrait
# ---------------------------------------------------------------------------

def test_plot_phase_portrait_returns_axes(signal):
    ax = viz.plot_phase_portrait(signal[:, 0], signal[:, 1])
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) >= 1


def test_plot_phase_portrait_with_ax(signal):
    fig, ax = plt.subplots()
    out = viz.plot_phase_portrait(signal[:, 0], signal[:, 1], ax=ax)
    assert out is ax


def test_plot_phase_portrait_unit_aware(signal):
    ax = viz.plot_phase_portrait(signal[:, 0] * u.mV, signal[:, 1] * u.mV)
    assert isinstance(ax, plt.Axes)


# ---------------------------------------------------------------------------
# plot_connectivity
# ---------------------------------------------------------------------------

def test_plot_connectivity_returns_axes():
    w = np.abs(np.random.RandomState(0).randn(6, 6))
    ax = viz.plot_connectivity(w)
    assert isinstance(ax, plt.Axes)
    assert len(ax.images) == 1


def test_plot_connectivity_with_labels_and_ax():
    w = np.abs(np.random.RandomState(0).randn(4, 4))
    fig, ax = plt.subplots()
    out = viz.plot_connectivity(w, labels=['a', 'b', 'c', 'd'], ax=ax)
    assert out is ax


def test_plot_connectivity_unit_aware():
    w = np.abs(np.random.RandomState(0).randn(5, 5)) * u.mm
    ax = viz.plot_connectivity(w)
    assert isinstance(ax, plt.Axes)


def test_plot_connectivity_no_colorbar():
    w = np.abs(np.random.RandomState(0).randn(4, 4))
    fig, ax = plt.subplots()
    n_initial = len(fig.axes)
    viz.plot_connectivity(w, ax=ax, colorbar=False)
    # No extra colorbar axes were added.
    assert len(fig.axes) == n_initial


# ---------------------------------------------------------------------------
# plot_functional_connectivity
# ---------------------------------------------------------------------------

def test_plot_functional_connectivity_from_timeseries(signal):
    ax = viz.plot_functional_connectivity(signal)
    assert isinstance(ax, plt.Axes)
    assert len(ax.images) == 1


def test_plot_functional_connectivity_precomputed_matrix():
    fc = np.eye(4)
    ax = viz.plot_functional_connectivity(fc, is_matrix=True)
    assert isinstance(ax, plt.Axes)


def test_plot_functional_connectivity_with_ax(signal):
    fig, ax = plt.subplots()
    out = viz.plot_functional_connectivity(signal, ax=ax)
    assert out is ax


# ---------------------------------------------------------------------------
# plot_power_spectrum
# ---------------------------------------------------------------------------

def test_plot_power_spectrum_returns_axes(signal):
    ax = viz.plot_power_spectrum(signal[:, 0], dt=1.0 * u.ms)
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) >= 1


def test_plot_power_spectrum_plain_dt(signal):
    ax = viz.plot_power_spectrum(signal[:, 0], dt=1.0)
    assert isinstance(ax, plt.Axes)


def test_plot_power_spectrum_with_ax(signal):
    fig, ax = plt.subplots()
    out = viz.plot_power_spectrum(signal[:, 0], dt=1.0, ax=ax)
    assert out is ax


def test_plot_power_spectrum_unit_aware_signal(signal):
    ax = viz.plot_power_spectrum(signal[:, 0] * u.mV, dt=1.0 * u.ms)
    assert isinstance(ax, plt.Axes)


def test_plot_power_spectrum_linear_axes(signal):
    ax = viz.plot_power_spectrum(signal[:, 0], dt=1.0, loglog=False)
    assert ax.get_xscale() == 'linear'


# ---------------------------------------------------------------------------
# Lazy import behaviour
# ---------------------------------------------------------------------------

def test_viz_module_does_not_import_pyplot_at_load():
    # The module itself must not have imported matplotlib at import time. We
    # check by inspecting the module's own namespace for a bound pyplot.
    assert not hasattr(viz, 'plt')
    assert not hasattr(viz, 'matplotlib')


def test_missing_matplotlib_raises_helpful_error(signal, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == 'matplotlib' or name.startswith('matplotlib.'):
            raise ImportError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ImportError) as exc:
        viz.plot_timeseries(signal)
    assert 'viz' in str(exc.value)


def test_import_brainmass_does_not_import_pyplot_lazily():
    # Importing brainmass.viz must not pull in matplotlib.pyplot at module load.
    import importlib
    import sys
    # Whatever the current state, re-importing the module must not require pyplot.
    mod = importlib.import_module('brainmass.viz')
    src = mod.__dict__
    assert 'pyplot' not in {k for k in src if 'pyplot' in k}


# ---------------------------------------------------------------------------
# Top-level exposure
# ---------------------------------------------------------------------------

def test_viz_exposed_on_package():
    assert hasattr(brainmass, 'viz')
    assert brainmass.viz is viz
