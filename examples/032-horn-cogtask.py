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

import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np
from brainstate import DelayState
from sklearn.decomposition import PCA

import cogtask
from brainmass import HORNSeqNetwork

brainstate.environ.set(dt=10. * u.ms)

task = cogtask.DelayComparison()

batch_size = 16
hidden_size = 128
tau = 400. * u.ms
alpha = 0.04
omega = 0.224
gamma = 0.01
v = 0.0

model = HORNSeqNetwork(
    task.num_inputs,
    hidden_size,
    task.num_outputs,
    alpha=alpha,
    omega=omega,
    gamma=gamma,
    v=v,
    # delay=braintools.init.Uniform(1 * u.ms, 40 * u.ms),
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


@brainstate.transform.jit
def predict(xs):
    mapmodule = brainstate.nn.ModuleMapper(model, xs.shape[0], init_state_axes={1: DelayState})
    mapmodule.init_all_states()
    predictions = mapmodule.map('hidden_activation')(xs)
    return predictions


for i_epoch in range(80):
    running_acc = []
    running_loss = []
    for _ in range(50):
        X, Y = task.batch_sample(batch_size, time_first=False)
        loss, acc = train(X, Y)
        running_loss.append(loss)
        running_acc.append(acc)
    print(f'Epoch {i_epoch + 1}, Loss {np.mean(running_loss):0.4f}, Acc {np.mean(running_acc):0.3f}')


def run(num_trial=1):
    inputs, targets = task.batch_sample(batch_size, time_first=False)
    rnn_activity = predict(inputs)[0]

    # Concatenate activity for PCA
    pca = PCA(n_components=2)
    activity = rnn_activity.reshape(-1, rnn_activity.shape[-1])
    pca.fit(activity)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(6, 3))
    for i in range(num_trial):
        activity_pc = pca.transform(rnn_activity[i])
        color = 'red' if targets[i, -1] == 1 else 'blue'
        _ = ax1.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)
        if i < 3:
            _ = ax2.plot(activity_pc[:, 0], activity_pc[:, 1], 'o-', color=color)
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    plt.show()


run(num_trial=16)
