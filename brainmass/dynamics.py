"""
Base classes for neural mass dynamics.

Provides abstract base classes for dynamics with unified interface:
    forward(state, param, inputs) -> (new_state, output)
"""

from typing import Dict, Optional, Tuple, TypeVar

import brainstate

from braintools.param import Param, Data
from .typing import Array
from .hooks import HookManager

__all__ = [
    'Dynamics',
    'MultiStepDynamics',
]

# Type variables for generic typing
I = TypeVar('I')  # input type
O = TypeVar('O')  # Output type


class Dynamics(brainstate.nn.Module):
    """
    Abstract base class for all dynamics.

    Defines the unified interface for dynamics models:
        forward(state, param, inputs) -> (new_state, output)

    Type Parameters:
        S: State type
        P: Parameter type (can be None for stateless dynamics)
        O: Output type (what gets passed to delay/coupling)

    Features:

    - Auto-discovery of child dynamics from nn.Module children
    - Auto-composition of get_param() and create_initial_state()
    - Hook support for debugging and monitoring
    """

    def __init__(
        self,
        hook_manager: Optional[HookManager] = None,
        children_cache: bool = True
    ):
        """
        Initialize dynamics module.

        Args:
            hook_manager: Optional HookManager for hook support.
                         If None, hooks will be disabled.
            children_cache: Whether to cache child dynamics discovery results.
                           Default True for performance. Set False for debugging
                           or when dynamically adding children.
        """
        super().__init__()
        self._hook_manager = hook_manager
        self._child_dynamics_cache = None
        self._children_cache_enabled = children_cache

    def update(
        self,
        state: Data,
        param: Data,
        inputs: I,
    ) -> Tuple[Data, O]:
        """
        Perform one step of dynamics.

        This is the core method that implements the differential equations
        and performs one integration step.

        Returns:
            Tuple of (new_state, output):
                - new_state: Updated state after one step
                - output: Observable output for delay/coupling
        """
        raise NotImplementedError

    def retrieve_params(self, *args, **kwargs) -> Data:
        """
        Retrieve all dynamics parameters.

        Default behavior: auto-compose from child dynamics.
        Override this method for leaf dynamics or custom behavior.

        Returns:
            Parameter object. For hierarchical dynamics, returns
            ComposedData containing all child parameters.

        Raises:
            NotImplementedError: If this is a leaf dynamics without
                                child dynamics and method is not overridden.
        """
        children = self._get_child_dynamics()
        if not children:
            raise NotImplementedError(
                f"{self.__class__.__name__} is a leaf dynamics and must "
                "implement get_param()"
            )
        return Data(children={
            name: child.retrieve_params(*args, **kwargs)
            for name, child in children.items()
        })

    def create_initial_state(self, *args, **kwargs) -> Data:
        """
        Create initial state for this dynamics model.

        Default behavior: auto-compose from child dynamics.
        Override this method for leaf dynamics or custom behavior.

        Args:
            *args: Arguments
            **kwargs: keyword arguments.

        Returns:
            Initialized state object. For hierarchical dynamics, returns
            ComposedData containing all child states.

        Raises:
            NotImplementedError: If this is a leaf dynamics without
                                child dynamics and method is not overridden.
        """
        children = self._get_child_dynamics()
        if not children:
            raise NotImplementedError(
                f"{self.__class__.__name__} is a leaf dynamics and must "
                "implement create_initial_state()"
            )
        return Data(children={
            name: child.create_initial_state(*args, **kwargs)
            for name, child in children.items()
        })

    def _get_child_dynamics(self) -> Dict[str, 'Dynamics']:
        """
        Discover child dynamics from nn.Module children.

        Uses PyTorch's module registration to find child Dynamics.
        Results are cached for performance if children_cache is enabled.

        Returns:
            Dict mapping child names to Dynamics instances.
        """
        # Check cache first (if enabled)
        if self._children_cache_enabled and self._child_dynamics_cache is not None:
            return self._child_dynamics_cache

        # Cache miss or disabled - discover children
        children = {}
        for name, module in self.nodes(allowed_hierarchy=(1, 1)).items():
            if isinstance(module, Dynamics):
                assert len(name) == 1
                children[name[0]] = module

        # Populate cache only if enabled
        if self._children_cache_enabled:
            self._child_dynamics_cache = children

        return children

    def _invalidate_child_cache(self) -> None:
        """Invalidate child dynamics cache (call after adding children)."""
        self._child_dynamics_cache = None

    @property
    def hook_manager(self) -> Optional[HookManager]:
        """Get the hook manager."""
        return self._hook_manager

    @hook_manager.setter
    def hook_manager(self, manager: Optional[HookManager]) -> None:
        """Set the hook manager and propagate to children."""
        self._hook_manager = manager
        # Propagate to children
        for child in self._get_child_dynamics().values():
            child.hook_manager = manager

    def reg_loss(self) -> Array:
        """
        Compute total regularization loss for all parameters.

        Returns:
            Scalar tensor with total regularization loss.
        """
        total_loss = 0.0
        for module in self.nodes(Param).values():
            total_loss = total_loss + module.reg_loss()
        return total_loss


class MultiStepDynamics(Dynamics):
    """
    Multi-step dynamics wrapper.

    Wraps any Dynamics and runs it for N steps, automatically stacking
    outputs on a new dimension.

    This is useful for:
        - TR-level integration (running multiple steps per TR)
        - Collecting state history for delay buffer updates
        - Any scenario requiring multiple sequential dynamics steps

    Type Parameters:
        S: State type (from wrapped dynamics)
        P: Parameter type (from wrapped dynamics)
        O: Output type of wrapped dynamics (will be stacked)

    """

    def __init__(
        self,
        dynamics: Dynamics,
        hook_manager: Optional[HookManager] = None,
        children_cache: bool = True
    ):
        """
        Initialize multistep dynamics wrapper.

        Args:
            dynamics: Single-step dynamics module to wrap.
            hook_manager: Optional HookManager. If None, uses the
                         wrapped dynamics' hook_manager.
            children_cache: Whether to cache child dynamics discovery results.
        """
        super().__init__(hook_manager or dynamics.hook_manager, children_cache)
        self.dynamics = dynamics

    def update(
        self,
        state: Data,
        param: Data,
        inputs: I,
    ) -> Tuple[Data, O]:
        fn = lambda s, i: self.dynamics.update(s, param, i)
        return brainstate.transform.scan(fn, state, inputs)

    def retrieve_params(self, *args, **kwargs) -> Data:
        """Get parameters from the wrapped dynamics."""
        return self.dynamics.retrieve_params(*args, **kwargs)

    def create_initial_state(self, *args, **kwargs) -> Data:
        """Create initial state using the wrapped dynamics."""
        return self.dynamics.create_initial_state(*args, **kwargs)
