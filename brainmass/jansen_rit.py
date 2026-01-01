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

from typing import Callable, Optional

import brainstate.environ
import braintools.init
import brainunit as u
import numpy as np
from braintools.param import Param, Data

from ._delay import OutputDelay
from ._base import Dynamics, MultiStepDynamics
from ._common import sys2nd, sigmoid, bounded_input
from ._leadfield import LeadfieldReadout
from ._typing import Array

__all__ = [
    "JansenRitStep",
    "JansenRitTR",
    "JansenRitWindow",
]


class JansenRitStep(Dynamics):
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
        state_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__()

        self.node_size = node_size
        self.step_size = step_size
        self.state_saturation = state_saturation
        self.input_saturation = input_saturation
        self.state_init = state_init
        self.u_2ndsys_ub = 500.

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

    def get_states(self, *args, **kwargs) -> Data:
        state = self.state_init((6, self.node_size))
        return Data(
            P=state[0],
            E=state[1],
            I=state[2],
            Pv=state[3],
            Ev=state[4],
            Iv=state[5],
        )

    def get_params(self, *args, **kwargs) -> Data:
        return Data(
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

    def update(self, state: Data, param: Data, input: Array):
        pi1 = getattr(param, "pi1", None)
        pi2 = getattr(param, "pi2", None)
        ei1 = getattr(param, "ei1", None)
        ei2 = getattr(param, "ei2", None)
        ii1 = getattr(param, "ii1", None)
        ii2 = getattr(param, "ii2", None)

        P = state.P
        E = state.E
        I = state.I
        Pv = state.Pv
        Ev = state.Ev
        Iv = state.Iv

        ext = param.k * param.ki * input  # (node_size,)
        node_size = ext.shape[0]

        # 计算各群体的发放率
        rP = (
            ext +
            param.std_in * brainstate.random.randn(node_size) +  # (node_size,)
            sigmoid(state.E - I, param.vmax, param.v0, param.r)  # (node_size,) firing rate for Main population
        )
        if pi1 is not None:
            rP += pi1
        if pi2 is not None:
            rP += u.math.matmul(pi2, P)

        rE = (
            param.kE +
            param.std_in * brainstate.random.randn(node_size) +  # (node_size,)
            param.c2 * sigmoid(param.c1 * P, param.vmax, param.v0, param.r)
        )
        if ei1 is not None:
            rE += ei1
        if ei2 is not None:
            rE += u.math.matmul(ei2, E - I)

        rI = (
            param.kI +
            param.std_in * brainstate.random.randn(node_size) +  # (node_size,)
            param.c4 * sigmoid(param.c3 * P, param.vmax, param.v0, param.r)
        )
        if ii1 is not None:
            rI += ii1
        if ii2 is not None:
            rI += u.math.matmul(ii2, E - I)

        # Update the states by step-size.
        dt = brainstate.environ.get_dt()
        ddP = P + dt * Pv
        ddE = E + dt * Ev
        ddI = I + dt * Iv
        ddPv = Pv + dt * sys2nd(param.A, param.a, bounded_input(rP, self.u_2ndsys_ub), P, Pv)
        ddEv = Ev + dt * sys2nd(param.A, param.a, bounded_input(rE, self.u_2ndsys_ub), E, Ev)
        ddIv = Iv + dt * sys2nd(param.B, param.b, bounded_input(rI, self.u_2ndsys_ub), I, Iv)

        # Calculate the saturation for model states (for stability and gradient calculation).
        E = bounded_input(ddE, 1e3)
        I = bounded_input(ddI, 1e3)
        P = bounded_input(ddP, 1e3)
        Ev = bounded_input(ddEv, 1e3)
        Iv = bounded_input(ddIv, 1e3)
        Pv = bounded_input(ddPv, 1e3)
        state = Data(E=E, I=I, P=P, Ev=Ev, Iv=Iv, Pv=Pv)
        return state, state.to_dict()


class LaplacianConnectivity(Dynamics):
    def __init__(
        self,
        sc: Array,
        w_ll: Param,
        w_ff: Param,
        w_bb: Param,
        g_l: Param,
        g_f: Param,
        g_b: Param,
        mask: Optional[Array] = None,
    ):
        super().__init__()

        # Three Laplacian modules for different pathways
        self.sc = sc
        self.w_ff = w_ff
        self.w_ll = w_ll
        self.w_bb = w_bb
        self.g_l = g_l
        self.g_f = g_f
        self.g_b = g_b
        self.mask = mask

    def get_params(self, *args, **kwargs) -> Data:
        w_bb = self.w_bb.value()
        w_ff = self.w_ff.value()
        w_ll = self.w_ll.value()

        gb = self.g_b.value()
        gl = self.g_l.value()
        gf = self.g_f.value()

        # Update the Laplacian based on the updated connection gains w_bb.
        w_b = u.math.exp(w_bb) * self.sc
        w_n_b = w_b / u.math.linalg.norm(w_b)
        if self.mask is not None:
            w_n_b = w_n_b * self.mask
        dg_b = -u.math.diag(u.math.sum(w_n_b, axis=1))

        # Update the Laplacian based on the updated connection gains w_ff.
        w_f = u.math.exp(w_ff) * self.sc
        w_n_f = w_f / u.linalg.norm(w_f)
        if self.mask is not None:
            w_n_f = w_n_f * self.mask
        dg_f = -u.math.diag(u.math.sum(w_n_f, axis=1))

        # Update the Laplacian based on the updated connection gains w_ll.
        w = u.math.exp(w_ll) * self.sc
        w = 0.5 * (w + u.math.transpose(w, (0, 1)))
        w_n_l = w / u.linalg.norm(w)
        if self.mask is not None:
            w_n_l = w_n_l * self.mask
        dg_l = -u.math.diag(u.math.sum(w_n_l, axis=1))

        return Data(
            w_n_b=w_n_b * gb,
            w_n_f=w_n_f * gf,
            w_n_l=w_n_l * gl,
            dg_b=dg_b * gb,
            dg_f=dg_f * gf,
            dg_l=dg_l * gl,
        )

    def get_states(self, *args, **kwargs) -> Data:
        return Data()

    def update(self, state: Data, param: Data, Ed: Array):
        LEd_b = u.math.sum(param.w_n_b * Ed.T, axis=1)
        LEd_f = u.math.sum(param.w_n_f * Ed.T, axis=1)
        LEd_l = u.math.sum(param.w_n_l * Ed.T, axis=1)

        return state, dict(
            pi1=LEd_l,
            pi2=param.dg_l,
            ei1=LEd_f,
            ei2=param.dg_f,
            ii1=-LEd_b,
            ii2=-param.dg_b,
        )


class JansenRitTR(Dynamics):
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

        # distance parameters
        mu: Array,
        dist: Array,

        # structural parameters
        sc: Array,
        w_ll: Param,
        w_ff: Param,
        w_bb: Param,
        g_l: Param,
        g_f: Param,
        g_b: Param,

        # leadfield parameters
        cy0: Param,
        y0: Param,
        lm: Param,

        # other parameters
        mask: Optional[Array] = None,
        state_saturation: bool = True,
        input_saturation: bool = True,
        state_init: Callable = braintools.init.ZeroInit(),
        delay_init: Callable = braintools.init.ZeroInit(),
    ):
        super().__init__()
        self.node_size = node_size
        self.step_size = step_size

        self.step = JansenRitStep(
            node_size=node_size,
            step_size=step_size,
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
        self.delay = OutputDelay(
            node_size=node_size,
            delay_idx=np.asarray(dist / mu, dtype=brainstate.environ.ditype()),
            init=delay_init,
        )
        self.conn = LaplacianConnectivity(
            sc=sc,
            w_ll=w_ll,
            w_ff=w_ff,
            w_bb=w_bb,
            g_l=g_l,
            g_f=g_f,
            g_b=g_b,
            mask=mask,
        )
        self.leadfield = LeadfieldReadout(
            lm=lm,
            y0=y0,
            cy0=cy0,
        )

    def update(self, state: Data, param: Data, inputs: Array):
        assert inputs.ndim == 2, f''

        # connection
        Ed = self.delay.get_delayed_value(state.delay)
        conn_out = self.conn(None, param.conn, Ed)[1]

        # for loop step model
        step_param = param.step.add(conn_out)
        step_state = state.step
        step_state, _ = brainstate.transform.scan(lambda s, i: self.step(s, step_param, i), step_state, inputs)

        # delay
        delay_state, _ = self.delay(state.delay, None, step_state.P)

        # leadfield
        _, eeg = self.leadfield(None, param.leadfield, step_state.E - step_state.I)

        return state.replace(step=step_state, delay=delay_state), (eeg, step_state)


class JansenRitWindow(MultiStepDynamics):
    """
    Jansen-Rit neural mass model for EEG simulation.

    A module for forward model (JansenRit) to simulate a batch of EEG signals.

    """

    def __init__(
        self,
        node_size: int,
        step_size: float,
        output_size: int,
        tr: float,
        sc: np.ndarray,
        dist: np.ndarray,
        # Model parameters using Param API
        A: Param,
        a: Param,
        B: Param,
        b: Param,
        g: Param,
        g_f: Param,
        g_b: Param,
        c1: Param,
        c2: Param,
        c3: Param,
        c4: Param,
        std_in: Param,
        vmax: Param,
        v0: Param,
        r: Param,
        y0: Param,
        mu,
        k: Param,
        kE: Param,
        kI: Param,
        cy0: Param,
        ki: Param,
        lm: Param,
        w_bb: Param,
        w_ff: Param,
        w_ll: Param,
        state_init: Callable,
        delay_init: Callable,
        mask=None,
    ):
        dynamics = JansenRitTR(
            node_size=node_size,
            step_size=step_size,

            # dynamics parameters
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

            # distance parameters
            mu=mu,
            dist=dist,

            # structural parameters
            sc=sc,
            w_ll=w_ll,
            w_ff=w_ff,
            w_bb=w_bb,
            g_l=g,
            g_f=g_f,
            g_b=g_b,

            # leadfield parameters
            cy0=cy0,
            y0=y0,
            lm=lm,

            # other parameters
            mask=mask,
            state_init=state_init,
            delay_init=delay_init,
        )

        super(JansenRitWindow, self).__init__(dynamics)
        self.tr = tr  # tr ms (integration step 0.1 ms)
        self.step_size = step_size  # integration step 0.1 ms
        self.steps_per_TR = int(tr / step_size)
        self.node_size = node_size  # num of ROI
        self.output_size = output_size  # num of M/EEG channels

        brainstate.environ.set(dt=step_size)
