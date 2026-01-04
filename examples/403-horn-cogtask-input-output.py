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

import math
import os.path
from typing import Callable, Optional

import braintools
import brainunit as u
import jax.numpy as jnp
import kagglehub
import matplotlib.pyplot as plt
import numpy as np

import brainstate
import cogtask
from brainmass import Parameter, Initializer, HORNStep, AdditiveCoupling, LaplacianConnParam
from brainstate import DelayState
from get_AAL2_io_regions import get_io_region_indices, IOIndexConfig


class HORNSeqLayer(brainstate.nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        input_indices: np.ndarray,
        output_indices: np.ndarray,
        alpha: Parameter = 0.04,  # excitability
        omega: Parameter = 2. * math.pi / 28.,  # natural frequency
        gamma: Parameter = 0.01,  # damping
        v: Parameter = 0.0,  # feedback
        h: brainstate.typing.ArrayLike = 1.0,  # integration step size
        state_init: Callable = braintools.init.ZeroInit(),
        delay: Initializer = None,
        conn: Initializer = None,
        rec_w_init: Initializer = braintools.init.KaimingNormal(),
        rec_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
        inp_w_init: Initializer = braintools.init.KaimingNormal(),
        inp_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
        out_w_init: Initializer = braintools.init.KaimingNormal(),
        out_b_init: Optional[Initializer] = braintools.init.ZeroInit(),
    ):
        super().__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.rec_w_init = rec_w_init
        self.rec_b_init = rec_b_init
        self.inp_w_init = inp_w_init
        self.inp_b_init = inp_b_init
        self.out_w_init = out_w_init
        self.out_b_init = out_b_init

        self.input_indices = input_indices
        self.output_indices = output_indices

        self.horn = HORNStep(n_hidden, alpha=alpha, omega=omega, gamma=gamma, v=v, h=h, state_init=state_init)
        self.i2h = brainstate.nn.Linear(self.n_input, len(input_indices), w_init=inp_w_init, b_init=inp_b_init)
        assert delay is not None, "Delay initializer must be provided for HORNSeqLayer."
        assert conn is not None, "Connection initializer must be provided for HORNSeqLayer."
        delay_time = braintools.init.param(delay, (n_hidden, n_hidden))
        neuron_idx = np.tile(np.expand_dims(np.arange(n_hidden), axis=0), (n_hidden, 1))
        self.h2h = AdditiveCoupling(
            self.horn.prefetch_delay('y', delay_time, neuron_idx, init=braintools.init.ZeroInit()),
            LaplacianConnParam(conn),
        )
        self.horn.recurrent_fn = self.h2h
        self.h2o = brainstate.nn.Linear(len(output_indices), self.n_output, w_init=out_w_init, b_init=out_b_init)

    def update(self, inputs, record_state: bool = False):
        def step(inp):
            input_ = jnp.zeros((self.n_hidden,))
            input_ = input_.at[self.input_indices].set(inp)
            out = self.horn(input_)[self.output_indices]
            st = dict(x=self.horn.x.value, y=self.horn.y.value)
            return (st, out) if record_state else out

        output = brainstate.transform.for_loop(step, self.i2h(inputs))
        if record_state:
            states, output = output
            return self.h2o(output[-1]), states
        else:
            return self.h2o(output[-1])


def try_train_with_visual_input_and_motor_output():
    cfg = IOIndexConfig(
        visual="core+extended",
        motor="core+extended",
        hemisphere="both",
    )
    # cfg = IOIndexConfig(
    #     visual="core",
    #     motor="core",
    #     hemisphere="both",
    # )
    io_info = get_io_region_indices(cfg)
    print('Input regions: ', io_info['input_names'])
    print('Output regions: ', io_info['output_names'])

    brainstate.environ.set(dt=10. * u.ms)

    task = cogtask.DelayComparison()

    path = kagglehub.dataset_download("oujago/hcp-gw-data-samples")
    data = braintools.file.msgpack_load(os.path.join(path, "hcp-data-sample.msgpack"))
    connection = data['Cmat']
    distance = data['Dmat']
    signal_speed = 2.

    batch_size = 16
    n_hidden = connection.shape[0]
    alpha = 0.04
    omega = 0.224
    gamma = 0.01
    v = 0.0

    model = HORNSeqLayer(
        task.num_inputs,
        n_hidden,
        task.num_outputs,
        input_indices=io_info['input_idx'],
        output_indices=io_info['output_idx'],
        alpha=alpha,
        omega=omega,
        gamma=gamma,
        v=v,
        delay=distance / signal_speed * u.ms,
        conn=connection,
    )

    # Adam optimizer
    trainable_weights = model.states(brainstate.ParamState)
    opt = braintools.optim.Adam(lr=1e-2)
    opt.register_trainable_weights(trainable_weights)

    def f_loss(xs, ys):
        vmap_module = brainstate.nn.ModuleMapper(
            model,
            init_map_size=xs.shape[0],
            init_state_axes={1: DelayState},
            behavior='vmap',
        )
        vmap_module.init_all_states()
        predictions = vmap_module(xs)
        predictions = u.math.flatten(predictions, end_axis=-2)
        ys = ys[:, -1]  # last time step labels
        loss_ = braintools.metric.softmax_cross_entropy_with_integer_labels(predictions, ys).mean()
        acc_ = u.math.asarray(predictions.argmax(1) == ys, dtype=u.math.float32).mean()
        return loss_, acc_

    @brainstate.transform.jit
    def train(xs, ys):
        f_grad = brainstate.transform.grad(f_loss, grad_states=trainable_weights, return_value=True, has_aux=True)
        grads, loss_, acc = f_grad(xs, ys)
        opt.update(grads)
        return loss_, acc

    def visualize_connectivity():
        fig, gs = braintools.visualize.get_figure(1, 1, 6., 6.)
        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(model.h2h.conn.value(), cmap='viridis', aspect='auto')
        fig.colorbar(im, ax=ax, label='Connectivity Strength')
        ax.set_title('Connectivity Matrix')
        ax.set_xlabel('Target Region Index')
        ax.set_ylabel('Source Region Index')
        plt.show()

    for i_epoch in range(80):

        if i_epoch % 10 == 0:
            visualize_connectivity()

        running_acc = []
        running_loss = []
        for _ in range(50):
            X, Y = task.batch_sample(batch_size, time_first=False)
            loss, acc = train(X, Y)
            running_loss.append(loss)
            running_acc.append(acc)
        print(f'Epoch {i_epoch + 1}, Loss {np.mean(running_loss):0.4f}, Acc {np.mean(running_acc):0.3f}')


if __name__ == '__main__':
    try_train_with_visual_input_and_motor_output()
