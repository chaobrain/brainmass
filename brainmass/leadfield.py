"""
EEG readout modules using leadfield matrix.

Transforms neural mass model source activity to sensor-space EEG signals.
"""

import brainunit as u
import jax
from brainstate.nn import Module, Param

from .typing import Array, Parameter

__all__ = [
    'LeadfieldReadout',
]


class LeadfieldReadout(Module):
    """
    Leadfield matrix EEG readout.

    Transforms source activity (E - I) to EEG sensor space using
    a leadfield matrix.

    EEG = cy0 * lm_normalized @ (E - I) - y0

    Args:
        lm: Leadfield matrix parameter (output_size, node_size).
        y0: Output bias parameter.
        cy0: Scaling coefficient parameter.
        normalize: Whether to L2-normalize leadfield rows.
        demean: Whether to remove mean across channels.
    """

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
        """
        Normalize leadfield matrix.

        Args:
            lm: (output_size, node_size) leadfield matrix.

        Returns:
            Normalized leadfield matrix.
        """

        if self.normalize:
            # L2 normalize each row (each sensor's weights)
            row_norms = u.math.sum(u.math.sqrt(lm ** 2), axis=1, keepdims=True)
            lm = lm / row_norms

        # Optional: remove mean across channels (columns)
        if self.demean:
            lm = lm - u.math.mean(lm, axis=0, keepdims=True)

        return lm

    def update(self, x: Array):
        lm = self.lm.value()
        y0 = self.y0.value()
        cy0 = self.cy0.value()
        fn = lambda xx: cy0 * (lm @ xx) - y0
        for _ in range(x.ndim - 1):
            fn = jax.vmap(fn)
        return fn(x)
