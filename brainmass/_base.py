"""
Base classes for neural mass dynamics.

Provides abstract base classes for dynamics with unified interface:
    forward(state, param, inputs) -> (new_state, output)
"""

import numbers
from functools import partial
from typing import Dict, Tuple, TypeVar, Any, Optional, Union, Callable

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

import brainstate
from braintools.param import Data
from ._typing import Initializer

PyTree = brainstate.typing.PyTree

__all__ = [
    'DynamoProtocol',
    'Dynamics',
    'Module',
    'Vmap',
    'Delay',
    'DelayAccess',
]

# Type variables for generic typing
I = TypeVar('I')  # input type
O = TypeVar('O')  # Output type

_INTERP_LINEAR = 'linear_interp'
_INTERP_ROUND = 'round'


class DynamoProtocol(brainstate.mixin.Mixin):
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

    def define_params(self, *args, **kwargs) -> Data:
        """
        Define all dynamics parameters.

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
        return Data(
            children={
                name: child.define_params(*args, **kwargs)
                for name, child in children.items()
            },
            name=f'{self.__class__.__name__}_Param'
        )

    def define_states(self, *args, **kwargs) -> Data:
        """
        Define initial state for this dynamics model.

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
        return Data(
            children={
                name: child.define_states(*args, **kwargs)
                for name, child in children.items()
            },
            name=f'{self.__class__.__name__}_State'
        )

    def _get_child_dynamics(self) -> Dict[str, 'DynamoProtocol']:
        """
        Discover child dynamics from nn.Module children.

        Uses PyTorch's module registration to find child DynamoProtocol.
        Results are cached for performance if children_cache is enabled.

        Returns:
            Dict mapping child names to DynamoProtocol instances.
        """
        # Cache miss or disabled - discover children
        children = {}
        assert isinstance(self, brainstate.nn.Module), f'This class must be a nn.Module, but {type(self)}.'
        for name, module in self.nodes(allowed_hierarchy=(1, 1)).items():
            if isinstance(module, DynamoProtocol):
                assert len(name) == 1
                children[name[0]] = module
        return children


class Module(brainstate.nn.Module, DynamoProtocol):
    def register_delay(self, ):
        pass


class Vmap(Module):
    def __init__(
        self,
        module: DynamoProtocol,
        state_axis: int | None = 0,
        param_axis: int | None = None,
        input_axis: Any = 0,
        output_axis: Any = 0,
    ) -> None:
        super().__init__()
        assert isinstance(module, DynamoProtocol)
        self.module = module
        self.state_axis = state_axis
        self.param_axis = param_axis
        self.input_axis = input_axis
        self.output_axis = output_axis

    def define_states(self, batch_size=None, *args, **kwargs) -> Data:
        if self.state_axis is not None:
            if len(args) == len(kwargs) == 0:
                if batch_size is None:
                    raise ValueError('Batch size must be provided when no arguments are given.')
            return brainstate.transform.vmap(
                self.module.define_states,
                axis_size=batch_size,
                out_axes=self.state_axis,
            )(*args, **kwargs)
        else:
            return self.module.define_states(*args, **kwargs)

    def define_params(self, batch_size=None, *args, **kwargs) -> Data:
        if self.param_axis is not None:
            if len(args) == len(kwargs) == 0:
                if batch_size is None:
                    raise ValueError('Batch size must be provided when no arguments are given.')
            return brainstate.transform.vmap(
                self.module.define_params,
                axis_size=batch_size,
                out_axes=self.param_axis,
            )(*args, **kwargs)
        else:
            return self.module.define_params(*args, **kwargs)

    def update(self, states, params, inputs):
        states, output = brainstate.transform.vmap(
            self.module,
            in_axes=(
                self.state_axis,
                self.param_axis,
                self.input_axis,
            ),
            out_axes=(
                self.state_axis,
                self.output_axis,
            )
        )(states, params, inputs)
        return states, output

    def run(self, inputs, return_states=False, return_params=False):
        batch_sizes = jax.tree.map(lambda x: x.shape[0], jax.tree.leaves(inputs))
        batch_sizes = set(batch_sizes)
        assert len(batch_sizes) == 1, "All input leaves must have the same batch size."
        batch_size = batch_sizes.pop()
        states = self.define_states(batch_size=batch_size)
        params = self.define_params(batch_size=batch_size)
        states, output = self.update(states, params, inputs)
        returns = []
        if return_states:
            returns.append(states)
        if return_params:
            returns.append(params)
        returns.append(output)
        if len(returns) == 1:
            return returns[0]
        else:
            return returns


def _get_delay(delay_time):
    if delay_time is None:
        return 0. * brainstate.environ.get_dt(), 0
    delay_step = delay_time / brainstate.environ.get_dt()
    assert u.get_dim(delay_step) == u.DIMENSIONLESS
    delay_step = jnp.ceil(delay_step).astype(brainstate.environ.ditype())
    return delay_time, delay_step


class DelayAccess(brainstate.graph.Node):
    """
    Accessor node for a registered entry in a Delay instance.

    This node holds a reference to a Delay and a named entry that was
    registered on that Delay. It is used by graphs to query delayed
    values by delegating to the underlying Delay instance.

    Args:
        delay: The delay instance.
        *time_and_index: The delay time.
        entry: The delay entry.
    """

    __module__ = 'brainmass'

    def __init__(
        self,
        delay: 'Delay',
        *time_and_index,
        entry: str,
    ):
        super().__init__()
        self.delay = delay
        assert isinstance(delay, Delay), 'The input delay should be an instance of Delay.'
        self._delay_entry = entry
        self.delay_info = delay.register_entry(self._delay_entry, *time_and_index)

    def update(self):
        return self.delay.at(self._delay_entry)


class Delay(Module):
    """
    Delay variable for storing short-term history data.

    The data in this delay variable is arranged as::

         delay = 0             [ data
         delay = 1               data
         delay = 2               data
         ...                     ....
         ...                     ....
         delay = length-1        data
         delay = length          data ]

    Args:
      time: int, float. The delay time.
      init: Any. The delay data. It can be a Python number, like float, int, boolean values.
        It can also be arrays. Or a callable function or instance of ``Connector``.
        Note that ``initial_delay_data`` should be arranged as the following way::

           delay = 1             [ data
           delay = 2               data
           ...                     ....
           ...                     ....
           delay = length-1        data
           delay = length          data ]
      entries: optional, dict. The delay access entries.
    """

    __module__ = 'brainmass'

    max_time: float  #
    max_length: int

    def __init__(
        self,
        target_info: PyTree,
        time: Optional[Union[int, float, u.Quantity]] = None,  # delay time
        init: Optional[Initializer] = None,  # delay data before t0
        entries: Optional[Dict] = None,  # delay access entry
        interp_method: str = _INTERP_LINEAR,  # interpolation method
        take_aware_unit: bool = False
    ):
        super().__init__()

        # target information
        self.target_info = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), target_info)

        # interp method
        assert interp_method in [_INTERP_LINEAR, _INTERP_ROUND], (
            f'Un-supported interpolation method {interp_method}. '
            f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}'
        )
        self.interp_method = interp_method

        # delay length and time
        with jax.ensure_compile_time_eval():
            self.max_time, delay_length = _get_delay(time)
            self.max_length = delay_length + 1

        # delay data
        if init is not None:
            if not isinstance(init, (numbers.Number, jax.Array, np.ndarray, Callable)):
                raise TypeError(f'init should be Array, Callable, or None. But got {init}')
        self._init = init

        # other info
        self._registered_entries = dict()

        # other info
        if entries is not None:
            for entry, delay_time in entries.items():
                if isinstance(delay_time, (tuple, list)):
                    self.register_entry(entry, *delay_time)
                else:
                    self.register_entry(entry, delay_time)

        self.take_aware_unit = take_aware_unit
        self._unit = None

    def _f_to_init(self, a, batch_size):
        shape = list(a.shape)
        if batch_size is not None:
            shape.insert(0, batch_size)
        shape.insert(0, self.max_length)
        if isinstance(self._init, (jax.Array, np.ndarray, numbers.Number)):
            data = jnp.broadcast_to(jnp.asarray(self._init, a.dtype), shape)
        elif callable(self._init):
            data = self._init(shape, dtype=a.dtype)
        else:
            assert self._init is None, f'init should be Array, Callable, or None. but got {self._init}'
            data = jnp.zeros(shape, dtype=a.dtype)
        return data

    def define_states(self, *args, batch_size: int = None, **kwargs) -> Data:
        fun = partial(self._f_to_init, batch_size=batch_size)
        return Data(name='Delay', buffer=jax.tree.map(fun, self.target_info))

    def define_params(self, *args, **kwargs) -> None:
        return None

    def register_delay(self, *time_and_index):
        """
        Register delay times and update the maximum delay configuration.

        This method processes one or more delay times, validates their format and consistency,
        and updates the delay buffer size if necessary. It handles both scalar and vector
        delay times, ensuring all vector delays have the same size.

        Args:
            *time_and_index: Variable number of delay time arguments. The first argument should be
                the primary delay time (float, int, or array-like). Additional arguments are
                treated as indices or secondary delay parameters. All delay times should be
                non-negative numbers or arrays of the same size.

        Returns:
            tuple or None: If time_and_index[0] is None, returns None. Otherwise, returns a tuple
                containing (delay_step, *time_and_index[1:]) where delay_step is the computed
                delay step in integer time units, and the remaining elements are the
                additional delay parameters passed in.

        Raises:
            AssertionError: If no delay time is provided (empty time_and_index).
            ValueError: If delay times have inconsistent sizes when using vector delays,
                or if delay times are not scalar or 1D arrays.

        Note:
            - The method updates self.max_time and self.max_length if the new delay
              requires a larger buffer size.
            - Delay steps are computed using the current environment time step (dt).
            - All delay indices (time_and_index[1:]) must be integers.
            - Vector delays must all have the same size as the first delay time.

        Example:
            >>> delay_obj.register_delay(5.0)  # Register 5ms delay
            >>> delay_obj.register_delay(jnp.array([2.0, 3.0]), 0, 1)  # Vector delay with indices
        """
        if len(time_and_index) == 0:
            return None
        for index in time_and_index[1:]:
            if not jnp.issubdtype(u.math.get_dtype(index), jnp.integer):
                raise TypeError(f'The index should be integer. But got {index}.')
        if time_and_index[0] is None:
            return None
        with jax.ensure_compile_time_eval():
            time, delay_step = _get_delay(time_and_index[0])
            max_delay_step = jnp.max(delay_step)
            self.max_time = u.math.max(time)

            # delay variable
            if self.max_length <= max_delay_step + 1:
                self.max_length = max_delay_step + 1
            return delay_step, *time_and_index[1:]

    def register_entry(self, entry: str, *time_and_index) -> 'Delay':
        """
        Register an entry to access the delay data.

        Args:
            entry: str. The entry to access the delay data.
            time_and_index: The delay time of the entry, the first element is the delay time,
                the second and later element is the index.
        """
        if entry in self._registered_entries:
            raise KeyError(
                f'Entry {entry} has been registered. '
                f'The existing delay for the key {entry} is {self._registered_entries[entry]}. '
                f'The new delay for the key {entry} is {time_and_index}. '
                f'You can use another key. '
            )
        delay_info = self.register_delay(*time_and_index)
        self._registered_entries[entry] = delay_info
        return delay_info

    def access(self, entry: str, *time_and_index) -> DelayAccess:
        """
        Create a DelayAccess object for a specific delay entry and delay time.

        Args:
            entry (str): The name of the delay entry to access.
            time_and_index (Sequence): The delay time or parameters associated with the entry.

        Returns:
            DelayAccess: An object that provides access to the delay data for the specified entry and time.
        """
        return DelayAccess(self, *time_and_index, entry=entry)

    def at(self, entry: str) -> PyTree:
        """
        Get the data at the given entry.

        Args:
          entry: str. The entry to access the data.

        Returns:
          The data.
        """
        assert isinstance(entry, str), (f'entry should be a string for describing the '
                                        f'entry of the delay data. But we got {entry}.')
        if entry not in self._registered_entries:
            raise KeyError(f'Does not find delay entry "{entry}".')
        delay_step = self._registered_entries[entry]
        if delay_step is None:
            delay_step = (0,)
        return self.retrieve_at_step(*delay_step)

    def retrieve_at_step(self, state: Data, delay_step, *indices) -> PyTree:
        """
        Retrieve the delay data at the given delay time step (the integer to indicate the time step).

        Parameters
        ----------
        delay_step: int_like
          Retrieve the data at the given time step.
        indices: tuple
          The indices to slice the data.

        Returns
        -------
        delay_data: The delay data at the given delay step.

        """
        assert delay_step is not None, 'The delay step should be given.'

        if brainstate.environ.get(brainstate.environ.JIT_ERROR_CHECK, False):
            def _check_delay(delay_len):
                raise ValueError(
                    f'The request delay length should be less than the '
                    f'maximum delay {self.max_length - 1}. But we got {delay_len}'
                )

            brainstate.transform.jit_error_if(
                delay_step >= self.max_length,
                _check_delay,
                delay_step
            )

        # rotation method
        with jax.ensure_compile_time_eval():
            delay_idx = delay_step

            # the delay index
            if hasattr(delay_idx, 'dtype') and not jnp.issubdtype(delay_idx.dtype, jnp.integer):
                raise ValueError(f'"delay_len" must be integer, but we got {delay_idx}')
            indices = (delay_idx,) + indices

            # the delay data
            if self._unit is None:
                return jax.tree.map(lambda a: a[indices], state.buffer)
            else:
                return jax.tree.map(
                    lambda hist, unit: u.maybe_decimal(hist[indices] * unit),
                    state.buffer,
                    self._unit
                )

    def retrieve_at_time(self, state: Data, delay_time, *indices) -> PyTree:
        """
        Retrieve the delay data at the given delay time step (the integer to indicate the time step).

        Parameters
        ----------
        delay_time: float
          Retrieve the data at the given time.
        indices: tuple
          The indices to slice the data.

        Returns
        -------
        delay_data: The delay data at the given delay step.

        """
        assert delay_time is not None, 'The delay time should be given.'

        current_time = brainstate.environ.get(brainstate.environ.T, desc='The current time.')
        dt = brainstate.environ.get_dt()

        if brainstate.environ.get(brainstate.environ.JIT_ERROR_CHECK, False):
            def _check_delay(t_now, t_delay):
                raise ValueError(
                    f'The request delay time should be within '
                    f'[{t_now - self.max_time - dt}, {t_now}], '
                    f'but we got {t_delay}'
                )

            brainstate.transform.jit_error_if(
                jnp.logical_or(
                    delay_time > current_time,
                    delay_time < current_time - self.max_time - dt
                ),
                _check_delay,
                current_time,
                delay_time
            )

        with jax.ensure_compile_time_eval():
            diff = current_time - delay_time
            float_time_step = diff / dt

            if self.interp_method == _INTERP_LINEAR:  # "linear" interpolation
                index_t0 = jnp.asarray(jnp.floor(float_time_step), dtype=jnp.int32)
                index_t1 = jnp.asarray(jnp.ceil(float_time_step), dtype=jnp.int32)
                data_at_t0 = self.retrieve_at_step(state, index_t0, *indices)
                data_at_t1 = self.retrieve_at_step(state, index_t1, *indices)
                t_diff = float_time_step - jnp.floor(float_time_step)
                return jax.tree.map(lambda a, b: a * (1 - t_diff) + b * t_diff, data_at_t0, data_at_t1)

            elif self.interp_method == _INTERP_ROUND:  # "round" interpolation
                index_t = jnp.asarray(jnp.round(float_time_step), dtype=jnp.int32)
                return self.retrieve_at_step(state, index_t, *indices)

            else:  # raise error
                raise ValueError(
                    f'Un-supported interpolation method {self.interp_method}, '
                    f'we only support: {[_INTERP_LINEAR, _INTERP_ROUND]}'
                )

    def update(self, state: Data, param: Data, current: PyTree) -> Tuple[Data, None]:
        with jax.ensure_compile_time_eval():
            if self.take_aware_unit and self._unit is None:
                self._unit = jax.tree.map(lambda x: u.get_unit(x), current, is_leaf=u.math.is_quantity)

            current = jax.tree.map(lambda a: jnp.expand_dims(a, 0), current)
            if self.max_length > 1:
                buffer = jax.tree.map(
                    lambda hist, cur: jnp.concatenate([cur, hist[:-1]], axis=0),
                    state.buffer,
                    current
                )
            else:
                buffer = current
        return state.replace(buffer=buffer), None


class Dynamics(Module):
    def __init__(self, in_size: brainstate.typing.Size, name: Optional[str] = None):
        # initialize
        super().__init__(name=name)

        # geometry size of neuron population
        if isinstance(in_size, (list, tuple)):
            if len(in_size) <= 0:
                raise ValueError(f'"in_size" must be int, or a tuple/list of int. But we got {type(in_size)}')
            if not isinstance(in_size[0], (int, np.integer)):
                raise ValueError(f'"in_size" must be int, or a tuple/list of int. But we got {type(in_size)}')
            in_size = tuple(in_size)
        elif isinstance(in_size, (int, np.integer)):
            in_size = (in_size,)
        else:
            raise ValueError(f'"in_size" must be int, or a tuple/list of int. But we got {type(in_size)}')
        self.in_size = in_size

        # in-/out- size of neuron population
        self.out_size = self.in_size

    @property
    def varshape(self):
        """
        Get the shape of variables in the neuron group.

        This property provides access to the geometry (shape) of the neuron population,
        which determines how variables and states are structured.

        Returns
        -------
        tuple
            A tuple representing the dimensional shape of the neuron group,
            matching the in_size parameter provided during initialization.

        See Also
        --------
        in_size : The input geometry specification for the neuron group
        """
        return self.in_size
