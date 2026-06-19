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

import brainmass
from brainmass.list_models import ModelInfo, list_models, _MODEL_REGISTRY


# The set of public neural-mass models documented in docs/reference/models.rst,
# plus the HORN model family. ``list_models()`` must cover exactly these.
_EXPECTED_NODE_MODELS = {
    'HopfStep', 'VanDerPolStep', 'StuartLandauStep', 'FitzHughNagumoStep',
    'ThresholdLinearStep', 'KuramotoNetwork', 'JansenRitStep', 'WilsonCowanStep',
    'WongWangStep', 'MontbrioPazoRoxinStep', 'CoombesByrneStep',
    'LarterBreakspearStep', 'EpileptorStep', 'Generic2dOscillatorStep',
    'WongWangExcInhStep', 'LorenzStep', 'LinearStep',
}
_EXPECTED_HORN_MODELS = {'HORNStep', 'HORNSeqLayer', 'HORNSeqNetwork'}
_EXPECTED = _EXPECTED_NODE_MODELS | _EXPECTED_HORN_MODELS


def test_list_models_returns_modelinfo_records():
    models = list_models()
    assert len(models) > 0
    for m in models:
        assert isinstance(m, ModelInfo)
        assert isinstance(m.name, str)
        assert isinstance(m.category, str)
        assert isinstance(m.n_state_vars, int)
        assert isinstance(m.use_case, str)


def test_list_models_covers_expected_set_exactly():
    names = [m.name for m in list_models()]
    assert set(names) == _EXPECTED


def test_list_models_no_duplicates():
    names = [m.name for m in list_models()]
    assert len(names) == len(set(names))


def test_every_listed_model_is_a_real_public_attribute():
    for m in list_models():
        assert hasattr(brainmass, m.name), m.name
        assert m.name in brainmass.__all__, m.name


def test_categories_match_models_rst():
    by_name = {m.name: m for m in list_models()}
    # Spot-check the canonical phenomenological / physiological split.
    assert by_name['HopfStep'].category == 'phenomenological'
    assert by_name['LorenzStep'].category == 'phenomenological'
    assert by_name['LinearStep'].category == 'phenomenological'
    assert by_name['JansenRitStep'].category == 'physiological'
    assert by_name['WilsonCowanStep'].category == 'physiological'
    assert by_name['EpileptorStep'].category == 'physiological'


def test_only_known_categories_used():
    cats = {m.category for m in list_models()}
    assert cats <= {'phenomenological', 'physiological', 'network'}


def test_state_var_counts_are_positive():
    for m in list_models():
        assert m.n_state_vars >= 1


def test_known_state_var_counts():
    by_name = {m.name: m for m in list_models()}
    assert by_name['HopfStep'].n_state_vars == 2
    assert by_name['JansenRitStep'].n_state_vars == 6
    assert by_name['EpileptorStep'].n_state_vars == 6
    assert by_name['LinearStep'].n_state_vars == 1
    assert by_name['LarterBreakspearStep'].n_state_vars == 3


def test_modelinfo_repr_is_readable():
    m = list_models()[0]
    r = repr(m)
    assert m.name in r
    assert 'ModelInfo' in r


def test_to_table_returns_str_with_all_models():
    table = list_models.to_table()
    assert isinstance(table, str)
    for name in _EXPECTED:
        assert name in table
    # header columns present
    assert 'category' in table.lower()


def test_list_models_exposed_on_package():
    assert hasattr(brainmass, 'list_models')
    assert brainmass.list_models is list_models


def test_registry_is_sorted_and_consistent():
    # The public list reflects the internal registry exactly.
    reg_names = [info.name for info in _MODEL_REGISTRY]
    assert reg_names == [m.name for m in list_models()]
