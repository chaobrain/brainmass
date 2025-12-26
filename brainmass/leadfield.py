"""
EEG readout modules using leadfield matrix.

Transforms neural mass model source activity to sensor-space EEG signals.
"""

import brainunit as u

from braintools.param import Param, Data
from .typing import Array
from .dynamics import Dynamics

__all__ = [
    'LeadfieldReadout',
]


class LeadfieldReadout(Dynamics):
    """
    Leadfield matrix EEG readout.

    Transforms source activity (E - I) to EEG sensor space using
    a leadfield matrix.

    EEG = cy0 * lm_normalized @ (E - I) - y0
    """

    def __init__(
        self,
        lm: Param,
        y0: Param,
        cy0: Param,
        normalize: bool = True,
        demean: bool = True,
    ):
        """
        Initialize leadfield readout.

        Args:
            lm: Leadfield matrix parameter (output_size, node_size).
            y0: Output bias parameter.
            cy0: Scaling coefficient parameter.
            normalize: Whether to L2-normalize leadfield rows.
            demean: Whether to remove mean across channels.
            hook_manager: Optional hook manager.
        """
        super().__init__()

        self.lm = lm
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

    def create_initial_state(self, *args, **kwargs) -> Data:
        return Data()

    def retrieve_params(self, *args, **kwargs) -> Data:
        return Data(
            lm=self.normalize_leadfield(self.lm.value()),
            y0=self.y0.value(),
            cy0=self.cy0.value(),
        )

    def update(
        self,
        state: None,
        param: Data,
        x: Array,
    ):
        eeg = param.cy0 * (param.lm @ x) - param.y0
        return state, eeg
