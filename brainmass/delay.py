"""
Delay processing module for neural mass models.

Provides delay dynamics classes that inherit from the Dynamics base class:
    - Delay: Low-level delay buffer component
    - SingleStepDelay: Wraps dynamics with single-step integration
    - MultiStepDelay: Wraps dynamics with multi-step integration and aggregation
"""

from typing import Any, Callable, Optional, Tuple

import brainstate
import brainunit as u
import jax
import numpy as np

import braintools.init
from braintools.param import Data
from .dynamics import Dynamics, I, O
from .functions import slice_data, process_sequence
from .hooks import HookManager
from .typing import Array

__all__ = [
    'OutputExtractor',
    'Delay',
    'SingleStepDelay',
    'MultiStepDelay',
]


class OutputExtractor:
    """
    Callable for extracting specific values from output for delay buffer.

    Can extract values using:
        - Custom function
        - Dictionary key
        - Tuple/list index
        - Array index on last dimension

    Example:
        # Extract 'P' key from dict output
        extractor = OutputExtractor(key='P')

        # Extract first element from tuple
        extractor = OutputExtractor(index=0)

        # Custom extraction
        extractor = OutputExtractor(func=lambda x: x.E - x.I)
    """

    def __init__(
        self,
        func: Optional[Callable[[Any], Array]] = None,
        key: Optional[str] = None,
        index: Optional[int] = None
    ):
        """
        Initialize output extractor.

        Args:
            func: Custom extraction function.
            key: If output is dict, extract this key.
            index: If output is tuple/list, extract this index.
                   If output is tensor, extract this index on last dim.
        """
        self._func = func
        self._key = key
        self._index = index

    def __call__(self, output: Any) -> Array:
        """
        Extract tensor from output.

        Args:
            output: Output from dynamics forward().

        Returns:
            Extracted tensor for delay buffer.

        Raises:
            ValueError: If cannot extract tensor from output type.
        """
        if self._func is not None:
            return self._func(output)

        if self._key is not None and isinstance(output, dict):
            return output[self._key]

        if self._index is not None:
            if isinstance(output, (tuple, list)):
                return output[self._index]
            if isinstance(output, (jax.Array, u.Quantity, np.ndarray)) and output.ndim >= 2:
                return output[..., self._index]

        # Default: return as-is (must be tensor)
        if isinstance(output, (jax.Array, u.Quantity, np.ndarray)):
            return output

        raise ValueError(
            f"Cannot extract tensor from output of type {type(output)}. "
            "Provide an extractor with func, key, or index."
        )


class Delay(Dynamics):
    """
    Delay buffer as a Dynamics component.

    Inherits from Dynamics base class with:
        State: DelayState (buffer + delay indices)
        Param: None (no trainable parameters)
        Output: Delayed activity matrix (node_size, node_size)

    This is the low-level delay buffer component used internally by
    SingleStepDelay and MultiStepDelay.

    Example:
        delay = Delay(
            node_size=68,
            delay_idx=delay_tensor,
            output_extractor=OutputExtractor(key='P')
        )

        # In dynamics loop
        delay_state, delayed = delay.forward(
            delay_state, None,
            {'new_values': dynamics_output}
        )
    """

    def __init__(
        self,
        node_size: int,
        delay_idx: Array,
        init: Callable = braintools.init.ZeroInit(),
        output_extractor: OutputExtractor = OutputExtractor(),
        hook_manager: Optional[HookManager] = None,
        children_cache: bool = True
    ):
        """
        Initialize delay buffer.

        Args:
            node_size: Number of brain regions.
            delay_idx: (node_size, node_size) delay indices tensor.
            output_extractor: How to extract values from input for buffering.
            hook_manager: Optional HookManager.
            children_cache: Whether to cache child dynamics discovery results.
        """
        super().__init__(hook_manager, children_cache)

        self.node_size = node_size
        self.output_extractor = output_extractor
        self.init = init

        # Clamp and register delay indices
        self.delay_idx = np.asarray(delay_idx, dtype=brainstate.environ.ditype())
        self.delays_max = self.delay_idx.max() + 1

        # Node index matrix for advanced indexing
        self.node_idx = np.tile(np.expand_dims(np.arange(self.node_size), axis=1), (1, self.node_size))

    def update(
        self,
        state: Data,
        param: None,
        inputs: I
    ) -> Tuple[Data, O]:
        # inputs = self.output_extractor(inputs)
        new_buffer = u.math.concatenate([u.math.expand_dims(inputs, axis=0), state.buffer[:-1]], axis=0)
        return Data(buffer=new_buffer), None

    def get_delayed_value(self, state: Data) -> Array:
        """
        Get delayed activity matrix from state.

        Computes delayed[i, j] = buffer[delay_idx[i, j], j], which is the
        delayed activity from region j arriving at region i.

        Args:
            state: Current delay state.

        Returns:
            (node_size, node_size) delayed activity matrix.
        """
        return state.buffer[self.delay_idx, self.node_idx]

    def retrieve_params(self, *args, **kwargs) -> None:
        """Delay has no parameters."""
        return None

    def create_initial_state(self, *args, **kwargs) -> Data:
        """
        Create initial delay state.

        Returns:
            Initialized DelayState.
        """
        return Data(buffer=self.init((self.delays_max, self.node_size)))


class SingleStepDelay(Dynamics):
    """
    Single-step delay with dynamics integration.

    Wraps a dynamics instance and handles delay buffer updates for
    single-step integration.

    This is similar to MultiStepDelay but runs only one integration step
    per forward call.

    Example:
        dynamics = JRStepModel(...)
        model = SingleStepDelay(
            dynamics=dynamics,
            node_size=68,
            delay_idx=delay_tensor,
        )

        # state is Data with 'dynamics' and 'delay'
        state = model.create_initial_state(node_size=68)
        param = model.get_param()

        new_state, outputs = model.forward(
            state, param,
            {'external': ext_input}
        )
        # outputs['delayed']: (68, 68) delayed activity
        # outputs['output']: dynamics output


    Args:
        dynamics: Single-step dynamics to wrap.
        node_size: Number of brain regions.
        delays: (node_size, node_size) delay indices tensor.
        output_extractor: How to extract values from output for buffering.
        hook_manager: Optional HookManager.
        children_cache: Whether to cache child dynamics discovery results.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        node_size: int,
        delays: Array,
        delay_init: Optional[Callable] = None,
        output_extractor: Optional[OutputExtractor] = None,
        hook_manager: Optional['HookManager'] = None,
        children_cache: bool = True
    ):
        super().__init__(hook_manager, children_cache)

        self.dynamics = dynamics
        self.node_size = node_size

        # Create internal delay buffer
        self.delay = Delay(
            node_size=node_size,
            delay_idx=delays,
            init=delay_init,
            output_extractor=output_extractor
        )

    def update(
        self,
        state: Data,
        param: Data,
        inputs: I
    ) -> Tuple[Data, O]:
        """
        Run single-step dynamics and update delay buffer.

        Args:
            state: Data with 'dynamics' and 'delay' children.
            param: Parameters for wrapped dynamics.
            inputs: Inputs for wrapped dynamics (any type).

        Returns:
            Tuple of (new_state, outputs):
                - new_state: Data with updated dynamics and delay
                - outputs: Dict with 'delayed' and 'output' keys
        """
        # Run single-step dynamics
        dyn_state, output = self.dynamics.update(state.dynamics, param.dynamics, inputs)
        delay_state, _ = self.delay.update(state.delay, param.delay, output)

        # Compose new state
        new_state = Data(dynamics=dyn_state, delay=delay_state)

        return new_state, output

    def get_delayed_value(self, state: Data) -> Array:
        """
        Get delayed activity from composed state.

        Args:
            state: Data with 'delay' child.

        Returns:
            (node_size, node_size) delayed activity matrix.
        """
        return self.delay.get_delayed_value(state.delay)


class MultiStepDelay(Dynamics):
    """
    Multi-step delay with aggregation.

    Inherits from MultiStepDynamics, wraps a dynamics instance and handles
    delay buffer updates with aggregation.

    This is the typical choice for TR-level dynamics where multiple
    integration steps are run per TR.

    Example:
        dynamics = JRStepModel(...)
        model = MultiStepDelay(
            dynamics=dynamics,
            steps=10,
            node_size=68,
            delay_idx=delay_tensor,
            aggregation='last'
        )

        # state is Data with 'dynamics' and 'delay'
        state = model.create_initial_state(node_size=68)
        param = model.get_param()

        new_state, outputs = model.forward(
            state, param,
            {'external': ext_input}
        )
        # outputs['delayed']: (68, 68) delayed activity
        # outputs['history']: (10, 68) P activity history


    Args:
        dynamics: Single-step dynamics to wrap.
        steps: Number of steps per forward call.
        node_size: Number of brain regions.
        delays: (node_size, node_size) delay indices tensor.
        output_extractor: How to extract values from output for buffering.
        aggregation: How to aggregate multi-step output:

                    'last' - use last step value
                    'mean' - use temporal mean
                    'max' - use maximum value
        hook_manager: Optional HookManager.
        children_cache: Whether to cache child dynamics discovery results.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        steps: int,
        node_size: int,
        delays: Array,
        delay_init: Callable = braintools.init.ZeroInit(),
        output_extractor: OutputExtractor = OutputExtractor(),
        aggregation: str = 'last',
        hook_manager: Optional['HookManager'] = None,
        children_cache: bool = True
    ):
        super().__init__(hook_manager, children_cache)
        self.dynamics = dynamics
        self.steps = steps
        # Create internal delay buffer
        self.delay = Delay(
            node_size=node_size,
            delay_idx=delays,
            init=delay_init,
            output_extractor=output_extractor
        )
        self.aggregation = aggregation

    def update(
        self,
        state: Data,
        param: Data,
        inputs: I
    ) -> Tuple[Data, O]:
        dyn_state = state.dynamics
        dyn_param = param.dynamics
        history = []
        for step_idx in range(self.steps):
            # Slice inputs that have step dimension
            step_inputs = slice_data(inputs, lambda x: x[step_idx])
            dyn_state, dyn_output = self.dynamics.update(dyn_state, dyn_param, step_inputs)
            history.append(dyn_output)
        aggregated = process_sequence(history, mode=self.aggregation)

        # Aggregate and update delay buffer
        delay_state, _ = self.delay.update(state.delay, param.delay, aggregated)

        # Compose new state
        new_state = Data(dynamics=dyn_state, delay=delay_state)

        return new_state, history

    def get_delayed_value(self, state: Data) -> Array:
        return self.delay.get_delayed_value(state.delay)
