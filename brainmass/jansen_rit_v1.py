"""
Complete Jansen-Rit neural mass model.

Combines all submodules (delay, dynamics, connectivity, readout) into
a unified model for brain simulation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Callable

import numpy as np
import torch

from .data import Data, ComposedData
from .delay import OutputExtractor, Delay
from .dynamics import Dynamics, O, S
from .functions import bounded_input, sigmoid, bounded_input, sys2nd, slice_data, process_sequence
from .leadfield import LeadfieldReadout
from braintools.param import Param

__all__ = [
    'JRState',
    'JRParam',

    # model at single step
    'JRStepModel',

    # model at single TR
    'JRTRModel',
    'MultiGainJRTRModel',
    'SingleGainJRTRModel',

    # model across multiple TR
    'JRWindowModel',

    # Laplacian-based long-range connectivity
    'LaplacianConnectivity',
    'ThreePathwayConnectivity',
]


@dataclass
class LaplacianParam(Data):
    W_norm: torch.Tensor
    D: torch.Tensor
    g: torch.Tensor


class LaplacianConnectivity(Dynamics):
    """
    Base Laplacian connectivity module.

    Computes weighted long-range connectivity using graph Laplacian encoding.
    The Laplacian form ensures that the total input to each node is balanced
    by self-inhibition (diagonal term).
    """

    def __init__(
        self,
        sc: torch.Tensor,
        gain: Optional[Param] = None,
        g: Optional[Param] = None,
        mask: Optional[torch.Tensor] = None,
        symmetrize: bool = False
    ):
        """
        Initialize Laplacian connectivity.

        Args:
            sc: (node_size, node_size) structural connectivity matrix.
            mask: Optional (node_size, node_size) binary mask for connections.
            symmetrize: Whether to symmetrize the weight matrix.
        """
        super().__init__()

        self.gain = gain
        self.g = g
        self.node_size = sc.shape[0]
        self.symmetrize = symmetrize

        # Register buffers
        self.register_buffer('sc', torch.tensor(sc, dtype=torch.float32))

        if mask is not None:
            self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))
        else:
            self.register_buffer('mask', None)

    def create_initial_state(self, *args, **kwargs) -> S:
        return None

    def retrieve_params(self, *args, **kwargs) -> LaplacianParam:
        # Get weight parameter
        w = torch.tensor(0., dtype=torch.float32) if self.gain is None else self.gain.value()

        # Apply exponential gain
        W = torch.exp(w) * self.sc

        # Optional symmetrization
        if self.symmetrize:
            W = 0.5 * (W + W.T)

        # Normalize and apply mask
        # W_norm = W / (torch.linalg.norm(W) + 1e-8)
        W_norm = W / torch.linalg.norm(W)
        if self.mask is not None:
            W_norm = W_norm * self.mask

        # Compute Laplacian diagonal: D = -diag(sum(W, dim=1))
        D = -torch.diag(torch.sum(W_norm, dim=1))

        # Get gain parameter
        g = torch.tensor(1.0, dtype=torch.float32) if self.g is None else self.g.value()

        return LaplacianParam(W_norm=W_norm, D=D, g=g)

    def update(
        self,
        state: None,
        param: LaplacianParam,
        inputs: Tuple
    ):
        delayed_activity, dyn_state = inputs

        # Long-range delayed input: sum_j W[i,j] * Ed[i,j]
        # delayed_activity is (node_size, node_size), need to transpose for proper sum
        long_range = torch.sum(param.W_norm * delayed_activity.T, dim=1)

        # Self-feedback through Laplacian diagonal
        self_feedback = param.D @ dyn_state.P

        return state, {'E': param.g * (long_range + self_feedback), 'P': 0., 'I': 0.}


class ThreePathwayConnectivity(Dynamics):
    """
    Three-pathway connectivity (ismail2025/momi2025 style).

    Implements three independent connection pathways:
    - Lateral (l): P -> P (symmetric)
    - Forward (f): P -> E
    - Backward (b): P -> I

    Args:
        sc: Structural connectivity matrix.
        w_ll: Lateral weight parameter.
        w_ff: Forward weight parameter.
        w_bb: Backward weight parameter.
        g_l: Lateral gain parameter.
        g_f: Forward gain parameter.
        g_b: Backward gain parameter.
        mask: Optional connection mask.
    """

    def __init__(
        self,
        sc: torch.Tensor,
        w_ll: Optional[Param] = None,
        w_ff: Optional[Param] = None,
        w_bb: Optional[Param] = None,
        g_l: Optional[Param] = None,
        g_f: Optional[Param] = None,
        g_b: Optional[Param] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Three Laplacian modules for different pathways
        self.w_bb = LaplacianConnectivity(sc, gain=w_bb, g=g_b, mask=mask, symmetrize=False)
        self.w_ff = LaplacianConnectivity(sc, gain=w_ff, g=g_f, mask=mask, symmetrize=False)
        self.w_ll = LaplacianConnectivity(sc, gain=w_ll, g=g_l, mask=mask, symmetrize=True)

    def update(
        self,
        state: ComposedData,
        param: ComposedData,
        inputs: Tuple,
    ):
        delayed_activity, dyn_state = inputs
        E_minus_I = dyn_state.E - dyn_state.I

        _, out_ll = self.w_ll.update(None, param['w_ll'], (delayed_activity, dyn_state))
        dyn_state = dyn_state.replace(P=E_minus_I)
        _, out_ff = self.w_ff.update(None, param['w_ff'], (delayed_activity, dyn_state))
        _, out_bb = self.w_bb.update(None, param['w_bb'], (delayed_activity, dyn_state))

        target_dict = dict()
        target_dict['P'] = out_ll['E']
        target_dict['E'] = out_ff['E']
        target_dict['I'] = -out_bb['E']
        return state, target_dict


@dataclass
class JRState(Data):
    """
    State container for Jansen-Rit neural mass model.

    Contains 6 state variables:

    - P, E, I: Position states (currents) for Pyramidal, Excitatory, Inhibitory
    - Pv, Ev, Iv: Velocity states (voltages) for each population

    All tensors have shape (node_size,) for single time point,
    or (batch_size, node_size) for batched operations.

    Inherits from Data, implementing the standard interface.
    """

    P: torch.Tensor  # Pyramidal current
    E: torch.Tensor  # Excitatory current
    I: torch.Tensor  # Inhibitory current
    Pv: torch.Tensor  # Pyramidal voltage
    Ev: torch.Tensor  # Excitatory voltage
    Iv: torch.Tensor  # Inhibitory voltage

    @property
    def node_size(self) -> Tuple[int, ...]:
        """Return the number of nodes."""
        return tuple(self.P.shape)

    @classmethod
    def init(
        cls,
        fn,
        node_size,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        kwargs = {}
        if device is not None:
            kwargs['device'] = device
        if dtype is not None:
            kwargs['dtype'] = dtype
        return cls(
            P=fn(node_size, **kwargs),
            E=fn(node_size, **kwargs),
            I=fn(node_size, **kwargs),
            Pv=fn(node_size, **kwargs),
            Ev=fn(node_size, **kwargs),
            Iv=fn(node_size, **kwargs)
        )


@dataclass
class JRParam(Data):
    """Parameters for Jansen-Rit dynamics."""
    A: torch.Tensor
    a: torch.Tensor
    B: torch.Tensor
    b: torch.Tensor
    vmax: torch.Tensor
    v0: torch.Tensor
    r: torch.Tensor
    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor
    std_in: torch.Tensor
    k: torch.Tensor
    ki: torch.Tensor
    kE: torch.Tensor
    kI: torch.Tensor


class JRStepModel(Dynamics[JRState, JRParam, Dict]):
    """
    Jansen-Rit single-step dynamics.

    Implements one integration step of the Jansen-Rit model.
    Inherits from Dynamics base class with new unified interface:
        forward(state, param, inputs) -> (new_state, output)

    Output is the P (pyramidal) activity for delay coupling.
    """

    def __init__(
        self,
        node_size: int,
        step_size: float,
        # dynamics parameters
        A: Param,
        a: Param,
        B: Param,
        b: Param,
        vmax: Param,
        v0: Param,
        r: Param,
        c1: Param,
        c2: Param,
        c3: Param,
        c4: Param,
        std_in: Param,
        k: Param,
        ki: Param,
        kE: Param,
        kI: Param,
        # other parameters
        state_saturation: bool = True,
        input_saturation: bool = True,
        state_init: Optional[Callable] = None,
    ):
        super().__init__()

        self.input_dict = dict()

        self.node_size = node_size
        self.step_size = step_size
        self.state_saturation = state_saturation
        self.input_saturation = input_saturation
        self.state_init = state_init

        self.A = A
        self.a = a
        self.B = B
        self.b = b
        self.vmax = vmax
        self.v0 = v0
        self.r = r
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.std_in = std_in
        self.k = k
        self.ki = ki
        self.kE = kE
        self.kI = kI

    def create_initial_state(
        self, device: Optional[torch.device] = None
    ) -> JRState:
        """Create randomly initialized state."""
        return JRState.init(self.state_init, self.node_size, device=device)

    def retrieve_params(self) -> JRParam:
        """Get all dynamics parameters as JRParam dataclass."""
        return JRParam(
            A=self.A.value(),
            B=self.B.value(),
            a=self.a.value(),
            b=self.b.value(),
            vmax=self.vmax.value(),
            v0=self.v0.value(),
            r=self.r.value(),
            c1=self.c1.value(),
            c2=self.c2.value(),
            c3=self.c3.value(),
            c4=self.c4.value(),
            std_in=self.std_in.value(),
            k=self.k.value(),
            ki=self.ki.value(),
            kE=self.kE.value(),
            kI=self.kI.value(),
        )

    def compute_firing_rates(
        self,
        state: JRState,
        param: JRParam,
        external_input: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        vmax = param.vmax
        v0 = param.v0
        r = param.r
        c1 = param.c1
        c2 = param.c2
        c3 = param.c3
        c4 = param.c4
        std_in = param.std_in
        k = param.k
        ki = param.ki

        # Baseline inputs for E and I populations (default 0)
        kE = param.kE
        kI = param.kI

        # Handle both single-pathway and three-pathway models
        ext = k * ki * external_input

        # noise
        noise_P = torch.randn(state.P.shape) * std_in
        noise_E = torch.randn(state.E.shape) * std_in
        noise_I = torch.randn(state.I.shape) * std_in

        # Pyramidal population input
        rP = (
            ext +
            noise_P +
            self.input_dict['P'] +
            sigmoid(state.E - state.I, vmax, v0, r)
        )

        # Excitatory population input
        rE = (
            kE +
            noise_E +
            self.input_dict['E'] +
            c2 * sigmoid(c1 * state.P, vmax, v0, r)
        )

        # Inhibitory population input
        rI = (
            kI +
            noise_I +
            self.input_dict['I'] +
            c4 * sigmoid(c3 * state.P, vmax, v0, r)
        )
        self.input_dict.clear()

        return {'rP': rP, 'rE': rE, 'rI': rI}

    def euler_step(
        self,
        state: JRState,
        rates: Dict[str, torch.Tensor],
        param: JRParam,
    ) -> JRState:
        """
        Perform Euler integration step.

        Args:
            state: Current state.
            rates: Firing rates {rP, rE, rI}.
            param: Parameters {A, a, B, b}.

        Returns:
            Updated state after one step.
        """
        A = param.A
        B = param.B
        a = param.a
        b = param.b
        dt = self.step_size

        # Apply input bounding
        if self.input_saturation:
            rP_bd = bounded_input(rates['rP'], 500.)
            rE_bd = bounded_input(rates['rE'], 500.)
            rI_bd = bounded_input(rates['rI'], 500.)
        else:
            rP_bd = rates['rP']
            rE_bd = rates['rE']
            rI_bd = rates['rI']

        # Position update: x_new = x + dt * v
        P_new = state.P + dt * state.Pv
        E_new = state.E + dt * state.Ev
        I_new = state.I + dt * state.Iv

        # Velocity update: v_new = v + dt * sys2nd(...)
        Pv_new = state.Pv + dt * sys2nd(A, a, rP_bd, state.P, state.Pv)
        Ev_new = state.Ev + dt * sys2nd(A, a, rE_bd, state.E, state.Ev)
        Iv_new = state.Iv + dt * sys2nd(B, b, rI_bd, state.I, state.Iv)

        state = JRState(
            P=P_new,
            E=E_new,
            I=I_new,
            Pv=Pv_new,
            Ev=Ev_new,
            Iv=Iv_new
        )
        if not self.state_saturation:
            return state

        return JRState(
            P=bounded_input(state.P, 1000.),
            E=bounded_input(state.E, 1000.),
            I=bounded_input(state.I, 1000.),
            Pv=bounded_input(state.Pv, 1000.),
            Ev=bounded_input(state.Ev, 1000.),
            Iv=bounded_input(state.Iv, 1000.)
        )

    def update(
        self,
        state: JRState,
        param: JRParam,
        inputs: torch.Tensor
    ) -> Tuple[JRState, O]:
        # Compute firing rates
        rates = self.compute_firing_rates(state, param, inputs)

        # Euler integration
        new_state = self.euler_step(state, rates, param)

        # Output is P (pyramidal) activity for delay coupling
        return new_state, new_state.to_dict()


class JRTRModel(Dynamics):
    """
    Modular Jansen-Rit neural mass model.

    Combines:

    - SingleStepDelay: Time-delayed long-range connectivity
    - JRStepModel/MultiStep: Neural mass dynamics
    - Connectivity: Local and long-range interactions
    - Readout: EEG signal generation

    Uses Data to unify dynamics state and delay state.

    """

    def __init__(
        self,
        node_size: int,  # Number of brain regions
        output_size: int,  # Number of EEG channels
        # Dynamics parameters (Param modules)
        A: Param,
        a: Param,
        B: Param,
        b: Param,
        vmax: Param,
        v0: Param,
        r: Param,
        c1: Param,
        c2: Param,
        c3: Param,
        c4: Param,
        std_in: Param,
        k: Param,
        ki: Param,
        kE: Param,
        kI: Param,
        state_init: Callable,
        # Connectivity parameters
        connectivity: torch.nn.Module,
        # Structural data
        dist: Union[np.ndarray, torch.Tensor],
        mu: Param,
        delays_max: int,
        delays_init: Callable,
        # Readout parameters
        lm: Param,
        y0: Param,
        cy0: Param,
        # config
        tr: float = 0.001,  # TR duration (seconds)
        step_size: float = 0.0001,  # Integration step (seconds)
        delay_type: str = 'last',  # Delay type: 'single_step', 'last', 'avg', 'max'
        state_saturation: bool = True,
        input_saturation: bool = True,
    ):
        super().__init__()

        self.node_size = node_size
        self.output_size = output_size
        self.steps_per_TR = int(tr / step_size)
        self.delay_type = delay_type

        # 1. Dynamics
        self.dynamics = JRStepModel(
            node_size,
            step_size,
            A=A,
            a=a,
            B=B,
            b=b,
            vmax=vmax,
            v0=v0,
            r=r,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            std_in=std_in,
            k=k,
            ki=ki,
            kE=kE,
            kI=kI,
            state_saturation=state_saturation,
            input_saturation=input_saturation,
            state_init=state_init,
        )

        # 2. Delay
        mu_val = mu.value()
        conduct_velocity = 1.5 + mu_val
        if isinstance(dist, np.ndarray):
            dist = torch.from_numpy(dist).float()
        delay_idx = dist / conduct_velocity
        delay_idx = torch.clamp(delay_idx.long(), 0, delays_max)
        self.delay = Delay(node_size, delay_idx, delays_init, output_extractor=OutputExtractor(key='P'))

        # 3. Connectivity
        if not isinstance(connectivity, (LaplacianConnectivity, ThreePathwayConnectivity)):
            raise TypeError(
                f'connectivity should be an instance of '
                f'SinglePathwayConnectivity or ThreePathwayConnectivity. '
                f'But we got {type(connectivity)} instead.'
            )
        self.connectivity = connectivity

        # 4. Readout
        self.readout = LeadfieldReadout(lm=lm, y0=y0, cy0=cy0)

    def update(
        self,
        state: ComposedData,
        param: ComposedData,
        inputs: torch.Tensor,
    ):
        dyn_state = state['dynamics']
        dyn_param = param['dynamics']
        conn_param = param['connectivity']
        lm_param = param['readout']
        delay_state = state['delay']

        # 1. Get delayed activity
        delayed_activity = self.delay.get_delayed_value(delay_state)

        # history = []
        for i_step in range(self.steps_per_TR):
            conn_out = self.connectivity.update(None, conn_param, (delayed_activity, dyn_state))[-1]
            self.dynamics.input_dict.update(conn_out)
            dyn_state, out = self.dynamics.update(dyn_state, dyn_param, inputs[i_step])
            # history.append(out)
        # delay_state, _ = self.delay.forward(delay_state, None, process_sequence(history, self.delay_type))
        delay_state, _ = self.delay.update(delay_state, None, dyn_state.P)

        # 4. Compute readout
        _, eeg = self.readout(None, lm_param, dyn_state.E - dyn_state.I)

        return state.replace(dynamics=dyn_state, delay=delay_state), eeg


class MultiGainJRTRModel(JRTRModel):
    """
    Jansen-Rit model with three-pathway connectivity.

    Uses ThreePathwayConnectivity with separate gains for
    P->P (lateral), P->E (forward), and P->I (backward) pathways.
    """

    def __init__(
        self,
        node_size: int,  # Number of brain regions
        output_size: int,  # Number of EEG channels
        # Dynamics parameters (Param modules)
        A: Param,
        a: Param,
        B: Param,
        b: Param,
        vmax: Param,
        v0: Param,
        r: Param,
        c1: Param,
        c2: Param,
        c3: Param,
        c4: Param,
        std_in: Param,
        k: Param,
        ki: Param,
        kE: Param,
        kI: Param,
        state_init: Callable,
        # Connectivity parameters
        g: Param,
        g_f: Param,
        g_b: Param,
        w_ll: Param,
        w_ff: Param,
        w_bb: Param,
        # Conduction velocity
        dist: Union[np.ndarray, torch.Tensor],
        mu: Param,
        delays_max: int,
        delays_init: Callable,
        # Readout parameters
        lm: Param,
        y0: Param,
        cy0: Param,
        # Structural data
        sc: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        # config
        tr: float = 0.001,  # TR duration (seconds)
        step_size: float = 0.0001,  # Integration step (seconds)
        delay_type: str = 'last',  # Delay type: 'single_step', 'last', 'avg', 'max'
        state_saturation: bool = True,
        input_saturation: bool = True,
    ):
        connectivity = ThreePathwayConnectivity(
            sc=sc,
            mask=mask,
            w_ll=w_ll,
            w_ff=w_ff,
            w_bb=w_bb,
            g_l=g,
            g_f=g_f,
            g_b=g_b
        )

        super().__init__(
            node_size,
            output_size,
            A=A,
            a=a,
            B=B,
            b=b,
            vmax=vmax,
            v0=v0,
            r=r,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            std_in=std_in,
            k=k,
            ki=ki,
            kE=kE,
            kI=kI,
            state_init=state_init,
            connectivity=connectivity,
            dist=dist,
            mu=mu,
            delays_max=delays_max,
            delays_init=delays_init,
            lm=lm,
            y0=y0,
            cy0=cy0,
            tr=tr,
            step_size=step_size,
            delay_type=delay_type,
            input_saturation=input_saturation,
            state_saturation=state_saturation,
        )


class SingleGainJRTRModel(JRTRModel):
    """
    Jansen-Rit model with single-pathway connectivity.

    Uses SinglePathwayConnectivity with a single gain parameter.
    """

    def __init__(
        self,
        node_size: int,  # Number of brain regions
        output_size: int,  # Number of EEG channels
        # Dynamics parameters (Param modules)
        A: Param,
        a: Param,
        B: Param,
        b: Param,
        vmax: Param,
        v0: Param,
        r: Param,
        c1: Param,
        c2: Param,
        c3: Param,
        c4: Param,
        std_in: Param,
        k: Param,
        ki: Param,
        kE: Param,
        kI: Param,
        state_init: Callable,
        # Connectivity parameters
        g: Param,
        w_bb: Param,
        # Conduction velocity
        dist: Union[np.ndarray, torch.Tensor],
        mu: Param,
        delays_max: int,
        delays_init: Callable,
        # Readout parameters
        lm: Param,
        y0: Param,
        cy0: Param,
        # Structural data
        sc: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        # config
        tr: float = 0.001,  # TR duration (seconds)
        step_size: float = 0.0001,  # Integration step (seconds)
        delay_type: str = 'last',  # Delay type: 'single_step', 'last', 'avg', 'max'
        state_saturation: bool = True,
        input_saturation: bool = True,
    ):
        connectivity = LaplacianConnectivity(sc=sc, mask=mask, gain=w_bb, g=g, symmetrize=True)

        super().__init__(
            node_size,
            output_size,
            A=A,
            a=a,
            B=B,
            b=b,
            vmax=vmax,
            v0=v0,
            r=r,
            c1=c1,
            c2=c2,
            c3=c3,
            c4=c4,
            std_in=std_in,
            k=k,
            ki=ki,
            kE=kE,
            kI=kI,
            state_init=state_init,
            connectivity=connectivity,
            dist=dist,
            mu=mu,
            delays_max=delays_max,
            delays_init=delays_init,
            lm=lm,
            y0=y0,
            cy0=cy0,
            tr=tr,
            step_size=step_size,
            delay_type=delay_type,
            input_saturation=input_saturation,
            state_saturation=state_saturation,
        )


class JRWindowModel(torch.nn.Module):
    def __init__(
        self,
        tr_model: JRTRModel,
    ):
        super().__init__()
        assert isinstance(tr_model, JRTRModel)
        self.tr_model = tr_model

    def create_initial_state(self):
        return self.tr_model.create_initial_state()

    def forward(
        self,
        states: ComposedData,
        multi_tr_inputs: torch.Tensor,
        TRs_per_window: int,
    ):
        eeg_window = []
        state_window = []
        params = self.tr_model.retrieve_params()
        for i_tr in range(TRs_per_window):
            states, eeg = self.tr_model.update(states, params, multi_tr_inputs[i_tr])
            eeg_window.append(eeg)
            state_window.append(states['dynamics'].to_dict())
        return states, (process_sequence(eeg_window), process_sequence(state_window, 'stack'))
