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

"""A dependency-free catalogue of the public neural-mass models.

:func:`list_models` returns a list of typed :class:`ModelInfo` records -- one per
user-facing model -- giving the model name, its category (phenomenological /
physiological / network), the number of state variables, and a one-line typical
use case. It powers the ``howto/choose_a_model`` guide and is pleasant to scan in
the REPL (``brainmass.list_models.to_table()``).

The catalogue is a curated registry kept in sync with the model table in
``docs/reference/models.rst`` (a co-located test asserts it covers exactly the
documented model set plus the HORN family). Adding a model is one
:class:`ModelInfo` entry.
"""

from typing import List, NamedTuple

__all__ = [
    'ModelInfo',
    'list_models',
]


class ModelInfo(NamedTuple):
    """A typed catalogue record describing one public model.

    Attributes
    ----------
    name : str
        The public class name (resolvable as ``brainmass.<name>``).
    category : str
        ``'phenomenological'``, ``'physiological'``, or ``'network'``.
    n_state_vars : int
        Number of hidden state variables the model integrates.
    use_case : str
        A short typical-use-case description.
    """

    name: str
    category: str
    n_state_vars: int
    use_case: str

    def __repr__(self) -> str:  # pragma: no cover - trivial formatting
        return (
            f"ModelInfo(name={self.name!r}, category={self.category!r}, "
            f"n_state_vars={self.n_state_vars}, use_case={self.use_case!r})"
        )


# Curated catalogue, kept in sync with docs/reference/models.rst. Ordered by
# category then by the documentation table.
_MODEL_REGISTRY: List[ModelInfo] = [
    # -- Phenomenological node models --
    ModelInfo('HopfStep', 'phenomenological', 2, 'Oscillation onset, rhythm generation'),
    ModelInfo('VanDerPolStep', 'phenomenological', 2, 'Nonlinear relaxation oscillations'),
    ModelInfo('StuartLandauStep', 'phenomenological', 2, 'Amplitude-controlled oscillations'),
    ModelInfo('FitzHughNagumoStep', 'phenomenological', 2, 'Excitability, spike generation'),
    ModelInfo('ThresholdLinearStep', 'phenomenological', 2, 'Fast linear E-I responses'),
    ModelInfo('Generic2dOscillatorStep', 'phenomenological', 2, 'Flexible planar dynamics (TVB)'),
    ModelInfo('LorenzStep', 'phenomenological', 3, 'Chaos, coupling test fixture'),
    ModelInfo('LinearStep', 'phenomenological', 1, 'Baseline node, coupling sanity checks'),
    # -- Physiological node models --
    ModelInfo('WilsonCowanStep', 'physiological', 2, 'E-I population firing-rate dynamics'),
    ModelInfo('JansenRitStep', 'physiological', 6, 'EEG generation, alpha rhythms'),
    ModelInfo('WongWangStep', 'physiological', 2, 'Decision making (perceptual choice)'),
    ModelInfo('WongWangExcInhStep', 'physiological', 2, 'Resting-state BOLD/FC, E-I balance'),
    ModelInfo('MontbrioPazoRoxinStep', 'physiological', 2, 'Exact QIF mean-field (theta neurons)'),
    ModelInfo('CoombesByrneStep', 'physiological', 2, 'Next-gen mean-field, conductance synapses'),
    ModelInfo('LarterBreakspearStep', 'physiological', 3, 'Conductance-based limit cycles / chaos'),
    ModelInfo('EpileptorStep', 'physiological', 6, 'Seizure onset/offset, epilepsy'),
    # -- Network / phase models --
    ModelInfo('KuramotoNetwork', 'network', 1, 'Phase synchronization'),
    # -- HORN family (oscillator recurrent networks) --
    ModelInfo('HORNStep', 'network', 2, 'Single coupled-oscillator step'),
    ModelInfo('HORNSeqLayer', 'network', 2, 'Sequential HORN layer'),
    ModelInfo('HORNSeqNetwork', 'network', 2, 'Multi-layer HORN sequence network'),
]


def _to_table() -> str:
    """Render the model catalogue as a plain-text aligned table.

    Returns
    -------
    str
        A human-readable, fixed-width table of every model.
    """
    header = ('name', 'category', '#states', 'use_case')
    rows = [
        (m.name, m.category, str(m.n_state_vars), m.use_case)
        for m in _MODEL_REGISTRY
    ]
    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows))
        for i in range(len(header))
    ]
    fmt = '  '.join('{:<' + str(w) + '}' for w in widths)
    lines = [fmt.format(*header), fmt.format(*('-' * w for w in widths))]
    lines += [fmt.format(*r) for r in rows]
    return '\n'.join(lines)


def list_models() -> List[ModelInfo]:
    """Return the catalogue of public neural-mass models.

    Returns
    -------
    list of ModelInfo
        One typed record per user-facing model, giving its name, category,
        number of state variables, and a one-line typical use case.

    See Also
    --------
    brainmass.list_models.to_table : a plain-text rendering of the same data.

    Examples
    --------
    .. code-block:: python

        >>> import brainmass
        >>> models = brainmass.list_models()
        >>> {m.name for m in models} >= {'HopfStep', 'JansenRitStep'}
        True
        >>> next(m for m in models if m.name == 'JansenRitStep').n_state_vars
        6
    """
    return list(_MODEL_REGISTRY)


# Attach the table renderer so callers can do ``brainmass.list_models.to_table()``.
list_models.to_table = _to_table
