# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

"""EEG readout modules using a leadfield matrix.

Transforms neural-mass-model source activity to sensor-space EEG signals.
"""

import brainunit as u
import jax
from brainstate.nn import Module, Param

from .typing import Array, Parameter

__all__ = [
    'LeadfieldReadout',
]


class LeadfieldReadout(Module):
    r"""Trainable leadfield-matrix EEG readout.

    Transforms a source-activity vector (e.g. ``E - I``) to EEG sensor space
    using a leadfield matrix:

    .. math::

        \mathrm{EEG} = c_{y0}\, (\hat{L}\, x) - y_0 ,

    where :math:`\hat{L}` is the (optionally row-normalised, demeaned) leadfield
    matrix and ``x`` is the source-activity vector.

    Parameters
    ----------
    lm : Array
        Leadfield matrix of shape ``(output_size, node_size)``. Wrapped in a
        trainable :class:`~brainstate.nn.Param`.
    y0 : Parameter
        Output bias parameter, subtracted from every channel.
    cy0 : Parameter
        Global scaling coefficient applied to the projected activity.
    normalize : bool, default True
        If ``True``, scale each leadfield row by its L1 norm (``sum(|w|)``,
        the sum of absolute values). Note: historically labelled "L2".
    demean : bool, default True
        If ``True``, remove the per-channel mean across sensors (columns).

    See Also
    --------
    brainmass.forward_model.LeadFieldModel
        The heavier, unit-aware forward operator.

    Notes
    -----
    Use :class:`LeadfieldReadout` when you want a lightweight, *trainable* EEG
    readout head — the leadfield is a fitted parameter with optional row
    normalisation / demeaning, and inputs/outputs are plain (unitless) arrays.
    Use :class:`brainmass.forward_model.LeadFieldModel` instead when you need a
    physically faithful forward model: explicit physical units, vertex→region
    aggregation, and additive measurement noise with a configurable covariance.
    """
    __module__ = 'brainmass'

    def __init__(
        self,
        lm: Array,
        y0: Parameter,
        cy0: Parameter,
        normalize: bool = True,
        demean: bool = True,
    ):
        super().__init__()

        self.lm = Param.init(lm)
        self.lm.precompute = self.normalize_leadfield
        self.y0 = y0
        self.cy0 = cy0
        self.normalize = normalize
        self.demean = demean

    def normalize_leadfield(self, lm: Array) -> Array:
        """Row-normalise and/or demean the leadfield matrix.

        Applied lazily as the ``precompute`` hook of ``self.lm`` whenever the
        parameter value is read.

        Parameters
        ----------
        lm : Array
            Leadfield matrix of shape ``(output_size, node_size)``.

        Returns
        -------
        Array
            The leadfield matrix after optional L1 row normalisation
            (``normalize``; scales each row by ``sum(|w|)``) and optional
            per-channel demeaning (``demean``).
        """

        if self.normalize:
            # Scale each row by its L1 norm: sum(|w|). (Comment historically
            # said "L2"; the code computes sum(sqrt(w**2)) == sum(|w|).)
            row_norms = u.math.sum(u.math.sqrt(lm ** 2), axis=1, keepdims=True)
            lm = lm / row_norms

        # Optional: remove mean across channels (columns)
        if self.demean:
            lm = lm - u.math.mean(lm, axis=0, keepdims=True)

        return lm

    def update(self, x: Array):
        """Project source activity to EEG sensor space.

        Parameters
        ----------
        x : Array
            Source-activity array whose last axis has length ``node_size``.
            Leading axes (batch/time) are mapped over automatically.

        Returns
        -------
        Array
            EEG signals with the last axis replaced by ``output_size``.
        """
        lm = self.lm.value()
        y0 = self.y0.value()
        cy0 = self.cy0.value()
        fn = lambda xx: cy0 * (lm @ xx) - y0
        for _ in range(x.ndim - 1):
            fn = jax.vmap(fn)
        return fn(x)
