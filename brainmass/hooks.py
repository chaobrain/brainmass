"""
Hook mechanism for NMM models.

Provides a flexible hook system for:

- Debugging and visualization (recording intermediate states)
- Training intervention (modifying states during forward pass)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import brainunit as u

from .typing import Array

__all__ = [
    'Hook',
    'HookContext',
    'HookManager',
    'HookPoint',
    'hook',
    'StateRecorderHook',
    'StateSaturationHook',
    'NaNDetectorHook',
]


class HookPoint(Enum):
    """Hook trigger points during model execution."""

    # Dynamics hooks
    PRE_STEP = auto()  # Before single step update
    POST_STEP = auto()  # After single step update
    PRE_TR = auto()  # Before TR update
    POST_TR = auto()  # After TR update

    # Delay hooks
    PRE_DELAY_UPDATE = auto()  # Before delay buffer update
    POST_DELAY_UPDATE = auto()  # After delay buffer update

    # Readout hooks
    PRE_READOUT = auto()  # Before EEG readout
    POST_READOUT = auto()  # After EEG readout


@dataclass
class HookContext:
    """Context passed to hook functions."""

    hook_point: HookPoint
    step_idx: int = 0
    tr_idx: int = 0
    states: Dict[str, Array] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


class Hook(ABC):
    """Abstract base class for hooks."""

    @property
    @abstractmethod
    def hook_points(self) -> List[HookPoint]:
        """Return list of hook points this hook listens to."""
        pass

    @abstractmethod
    def __call__(self, context: HookContext) -> Optional[Dict[str, Array]]:
        """
        Execute the hook.

        Args:
            context: Hook context with current state information.

        Returns:
            Optional dict of state modifications. Return None for no changes.
        """
        pass


class HookManager:
    """Manages registration and triggering of hooks."""

    def __init__(self):
        self._hooks: Dict[HookPoint, List[Hook]] = {hp: [] for hp in HookPoint}
        self._enabled: bool = True

    def register(self, hook: Hook) -> None:
        """Register a hook for its specified hook points."""
        for hp in hook.hook_points:
            if hook not in self._hooks[hp]:
                self._hooks[hp].append(hook)

    def unregister(self, hook: Hook) -> None:
        """Unregister a hook from all its hook points."""
        for hp in hook.hook_points:
            if hook in self._hooks[hp]:
                self._hooks[hp].remove(hook)

    def trigger(
        self,
        hook_point: HookPoint,
        context: HookContext
    ) -> Dict[str, Array]:
        """
        Trigger all hooks registered for a hook point.

        Args:
            hook_point: The hook point to trigger.
            context: Context to pass to hooks.

        Returns:
            Merged dict of all state modifications from hooks.
        """
        if not self._enabled:
            return {}

        modifications = {}
        for hook in self._hooks[hook_point]:
            result = hook(context)
            if result is not None:
                modifications.update(result)
        return modifications

    def enable(self) -> None:
        """Enable hook triggering."""
        self._enabled = True

    def disable(self) -> None:
        """Disable hook triggering (for fast inference)."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if hooks are enabled."""
        return self._enabled

    def clear(self) -> None:
        """Remove all registered hooks."""
        for hp in HookPoint:
            self._hooks[hp].clear()


def hook(hook_points: List[HookPoint]):
    """
    Decorator to convert a function into a Hook.

    Usage:
        @hook([HookPoint.POST_STEP])
        def my_hook(context: HookContext) -> Optional[Dict[str, Array]]:
            print(f"Step {context.step_idx}: P = {context.states['P'].mean()}")
            return None
    """

    def decorator(func: Callable[[HookContext], Optional[Dict[str, Array]]]):
        class FunctionHook(Hook):
            @property
            def hook_points(self) -> List[HookPoint]:
                return hook_points

            def __call__(self, context: HookContext) -> Optional[Dict[str, Array]]:
                return func(context)

        return FunctionHook()

    return decorator


# =============================================================================
# Predefined Hooks
# =============================================================================

class StateRecorderHook(Hook):
    """
    Records states at specified hook points for debugging/visualization.

    Usage:
        recorder = StateRecorderHook([HookPoint.POST_STEP])
        model.hook_manager.register(recorder)
        # ... run model ...
        recorded = recorder.to_dict()  # {'P': (num_steps, node_size), ...}
    """

    def __init__(
        self,
        record_points: Optional[List[HookPoint]] = None,
        state_keys: Optional[List[str]] = None
    ):
        """
        Args:
            record_points: Hook points to record at. Default: [POST_STEP].
            state_keys: Which state keys to record. Default: all.
        """
        self._record_points = record_points or [HookPoint.POST_STEP]
        self._state_keys = state_keys
        self.records: List[HookContext] = []

    @property
    def hook_points(self) -> List[HookPoint]:
        return self._record_points

    def __call__(self, context: HookContext) -> None:
        # Deep copy states to avoid reference issues
        states_to_record = {}
        keys = self._state_keys or context.states.keys()
        for k in keys:
            if k in context.states:
                states_to_record[k] = context.states[k].clone().detach()

        recorded = HookContext(
            hook_point=context.hook_point,
            step_idx=context.step_idx,
            tr_idx=context.tr_idx,
            states=states_to_record,
            extra=context.extra.copy()
        )
        self.records.append(recorded)
        return None

    def clear(self) -> None:
        """Clear all recorded states."""
        self.records.clear()

    def to_dict(self) -> Dict[str, Array]:
        """
        Convert records to stacked tensors.

        Returns:
            Dict mapping state name to (num_records, ...) tensor.
        """
        if not self.records:
            return {}

        result = {}
        keys = self.records[0].states.keys()
        for key in keys:
            tensors = [r.states[key] for r in self.records if key in r.states]
            if tensors:
                result[key] = u.math.stack(tensors, axis=0)
        return result


class StateSaturationHook(Hook):
    """
    Applies tanh saturation to states for training stability.

    Usage:
        saturation = StateSaturationHook(bound=1000.0)
        model.hook_manager.register(saturation)
    """

    def __init__(
        self,
        bound: float = 1000.0,
        state_keys: Optional[List[str]] = None
    ):
        """
        Args:
            bound: Saturation bound (applies tanh(x/bound) * bound).
            state_keys: Which states to saturate. Default: all.
        """
        self.bound = bound
        self._state_keys = state_keys

    @property
    def hook_points(self) -> List[HookPoint]:
        return [HookPoint.POST_STEP]

    def __call__(self, context: HookContext) -> Optional[Dict[str, Array]]:
        modifications = {}
        keys = self._state_keys or context.states.keys()

        for key in keys:
            if key in context.states:
                state = context.states[key]
                saturated = self.bound * u.math.tanh(state / self.bound)
                modifications[key] = saturated

        return modifications if modifications else None


class NaNDetectorHook(Hook):
    """
    Detects NaN values in states and optionally replaces them.

    Usage:
        detector = NaNDetectorHook(replace_with=0.0)
        model.hook_manager.register(detector)
    """

    def __init__(
        self,
        replace_with: Optional[float] = None,
        raise_error: bool = False
    ):
        """
        Args:
            replace_with: Value to replace NaNs with. None means no replacement.
            raise_error: Whether to raise an error when NaN is detected.
        """
        self.replace_with = replace_with
        self.raise_error = raise_error
        self.nan_detected: List[tuple] = []  # (tr_idx, step_idx, key)

    @property
    def hook_points(self) -> List[HookPoint]:
        return [HookPoint.POST_STEP]

    def __call__(self, context: HookContext) -> Optional[Dict[str, Array]]:
        modifications = {}

        for key, val in context.states.items():
            if u.math.any(u.math.isnan(val)):
                self.nan_detected.append((context.tr_idx, context.step_idx, key))

                if self.raise_error:
                    raise ValueError(
                        f"NaN detected in {key} at TR {context.tr_idx}, "
                        f"step {context.step_idx}"
                    )

                if self.replace_with is not None:
                    modifications[key] = u.math.where(
                        u.math.isnan(val),
                        u.math.full_like(val, self.replace_with),
                        val
                    )

        return modifications if modifications else None

    def clear(self) -> None:
        """Clear NaN detection history."""
        self.nan_detected.clear()
