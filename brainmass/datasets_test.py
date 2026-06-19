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

import numpy as np
import brainunit as u
import pytest

import brainmass
from brainmass import datasets


# ---------------------------------------------------------------------------
# Registry mechanics
# ---------------------------------------------------------------------------

def test_builtin_datasets_listed():
    names = [name for name, _ in datasets.list_datasets()]
    assert 'example_connectome' in names
    assert 'example_signal' in names
    assert 'delayed_match_task' in names


def test_list_datasets_returns_name_description_pairs():
    listing = datasets.list_datasets()
    assert isinstance(listing, list)
    for entry in listing:
        assert isinstance(entry, tuple)
        assert len(entry) == 2
        name, description = entry
        assert isinstance(name, str)
        assert isinstance(description, str)


def test_list_datasets_sorted_by_name():
    names = [name for name, _ in datasets.list_datasets()]
    assert names == sorted(names)


def test_register_and_load_roundtrip():
    sentinel = object()
    datasets.register_dataset('temp_demo_ds', lambda: sentinel, description='temp')
    try:
        assert datasets.load_dataset('temp_demo_ds') is sentinel
        assert ('temp_demo_ds', 'temp') in datasets.list_datasets()
    finally:
        datasets._REGISTRY.pop('temp_demo_ds', None)


def test_register_dataset_default_description():
    datasets.register_dataset('temp_no_desc', lambda: 1)
    try:
        descs = dict(datasets.list_datasets())
        assert descs['temp_no_desc'] == ''
    finally:
        datasets._REGISTRY.pop('temp_no_desc', None)


def test_register_non_callable_raises():
    with pytest.raises(TypeError):
        datasets.register_dataset('bad', 123)


def test_register_duplicate_raises():
    datasets.register_dataset('dup_demo', lambda: 1)
    try:
        with pytest.raises(ValueError):
            datasets.register_dataset('dup_demo', lambda: 2)
    finally:
        datasets._REGISTRY.pop('dup_demo', None)


def test_load_unknown_dataset_raises_with_available_names():
    with pytest.raises(KeyError) as exc:
        datasets.load_dataset('does_not_exist')
    msg = str(exc.value)
    # The error must list the available names to be helpful.
    assert 'example_connectome' in msg
    assert 'does_not_exist' in msg


# ---------------------------------------------------------------------------
# example_connectome
# ---------------------------------------------------------------------------

def test_example_connectome_container_fields():
    conn = datasets.load_dataset('example_connectome')
    assert isinstance(conn, datasets.Connectome)
    assert conn.weights.shape == (8, 8)
    assert conn.distances.shape == (8, 8)
    assert len(conn.labels) == 8


def test_example_connectome_weights_symmetric_zero_diagonal():
    conn = datasets.load_dataset('example_connectome')
    w = np.asarray(conn.weights)
    assert np.allclose(w, w.T)
    assert np.allclose(np.diag(w), 0.0)
    assert w.min() >= 0.0
    assert w.max() <= 1.0


def test_example_connectome_distances_unit_aware_mm():
    conn = datasets.load_dataset('example_connectome')
    assert u.get_unit(conn.distances) == u.mm
    d = u.get_magnitude(conn.distances)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert d.min() >= 0.0


def test_example_connectome_deterministic_across_calls():
    a = datasets.load_dataset('example_connectome')
    b = datasets.load_dataset('example_connectome')
    assert np.array_equal(np.asarray(a.weights), np.asarray(b.weights))
    assert np.array_equal(u.get_magnitude(a.distances), u.get_magnitude(b.distances))


# ---------------------------------------------------------------------------
# example_signal
# ---------------------------------------------------------------------------

def test_example_signal_container_fields():
    sig = datasets.load_dataset('example_signal')
    assert isinstance(sig, datasets.Signal)
    assert sig.signal.ndim == 2
    n_time, n_region = sig.signal.shape
    assert n_region == 8
    assert n_time > 1
    assert sig.fc.shape == (8, 8)
    assert len(sig.labels) == 8


def test_example_signal_dt_is_unit_aware():
    sig = datasets.load_dataset('example_signal')
    assert u.get_unit(sig.dt) == u.ms
    assert u.get_magnitude(sig.dt) > 0.0


def test_example_signal_fc_symmetric_unit_diagonal():
    sig = datasets.load_dataset('example_signal')
    fc = np.asarray(sig.fc)
    assert np.allclose(fc, fc.T)
    assert np.allclose(np.diag(fc), 1.0)


def test_example_signal_deterministic():
    a = datasets.load_dataset('example_signal')
    b = datasets.load_dataset('example_signal')
    assert np.array_equal(np.asarray(a.signal), np.asarray(b.signal))


# ---------------------------------------------------------------------------
# delayed_match_task (synthetic generator)
# ---------------------------------------------------------------------------

def test_delayed_match_task_shapes():
    inputs, targets = datasets.delayed_match_task(n_samples=16, seq_len=12)
    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    assert inputs.shape[0] == 16
    assert inputs.shape[1] == 12
    assert targets.shape[0] == 16


def test_delayed_match_task_deterministic_with_seed():
    a_in, a_tg = datasets.delayed_match_task(n_samples=8, seed=0)
    b_in, b_tg = datasets.delayed_match_task(n_samples=8, seed=0)
    assert np.array_equal(np.asarray(a_in), np.asarray(b_in))
    assert np.array_equal(np.asarray(a_tg), np.asarray(b_tg))


def test_delayed_match_task_different_seeds_differ():
    a_in, _ = datasets.delayed_match_task(n_samples=8, seed=0)
    b_in, _ = datasets.delayed_match_task(n_samples=8, seed=1)
    assert not np.array_equal(np.asarray(a_in), np.asarray(b_in))


def test_delayed_match_task_targets_are_binary_match_labels():
    inputs, targets = datasets.delayed_match_task(n_samples=64, seed=2)
    targets = np.asarray(targets)
    uniq = set(np.unique(targets).tolist())
    assert uniq.issubset({0, 1})
    # A balanced-ish generator should produce both classes.
    assert len(uniq) == 2


def test_delayed_match_task_registered_loader_returns_callable_default():
    # The registered loader returns the default task (so load_dataset works too).
    inputs, targets = datasets.load_dataset('delayed_match_task')
    assert np.asarray(inputs).shape[0] > 0
    assert np.asarray(targets).shape[0] == np.asarray(inputs).shape[0]


def test_delayed_match_task_n_classes():
    inputs, targets = datasets.delayed_match_task(n_samples=32, n_symbols=5, seq_len=10, seed=3)
    inputs = np.asarray(inputs)
    # one-hot symbol depth == n_symbols
    assert inputs.shape[-1] == 5


def test_delayed_match_task_rejects_short_sequences():
    with pytest.raises(ValueError):
        datasets.delayed_match_task(n_samples=4, seq_len=1)


def test_delayed_match_task_rejects_too_few_symbols():
    with pytest.raises(ValueError):
        datasets.delayed_match_task(n_samples=4, n_symbols=1)


# ---------------------------------------------------------------------------
# Top-level exposure
# ---------------------------------------------------------------------------

def test_datasets_exposed_on_package():
    assert hasattr(brainmass, 'datasets')
    assert brainmass.datasets is datasets
