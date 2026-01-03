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

import argparse

import brainunit as u
import braintools
import jax.numpy as jnp
import numpy as np
import torch
import torchvision

import brainstate
from brainmass import HORNSeqNetwork

# command line arguments
parser = argparse.ArgumentParser(description='HORN training script')
parser.add_argument('--num-hidden', type=int, default=32, help='number of units')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--shuffle', action='store_true', help='whether to shuffle stimulus time steps')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--h', type=float, default=1.0, help='microscopic time constant h (default: 1)')
parser.add_argument('--alpha', type=float, default=0.04, help='excitability coefficient alpha')
parser.add_argument('--omega', type=float, default=0.224, help='natural frequency omega')  # 2 * pi / 28 for sMNIST
parser.add_argument('--gamma', type=float, default=0.01, help='damping coefficient gamma')
parser.add_argument('--v', type=float, default=0., help='feedback coefficient v')

args = parser.parse_args()

brainstate.environ.set(dt=1. * u.ms)

# sMNIST as 1-dim time series
dim_input = 1

# 10 MNIST classes
dim_output = 10

# batch size of the test set
batch_size_train = args.batch_size
batch_size_test = 1000

# to shuffle mnist digits
if args.shuffle:
    perm = brainstate.random.shuffle(np.arange(784))

# load dataset
size_validation = 1000  # size of validation dataset
train_set = torchvision.datasets.MNIST(
    root='data', train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_set = torchvision.datasets.MNIST(
    root='data', train=False, transform=torchvision.transforms.ToTensor(), download=True
)
train_set, valid_set = torch.utils.data.random_split(train_set, [len(train_set) - size_validation, size_validation])

# data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size_test, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)

# instantiate homogeneous HORN model
# see dynamics.py for an example of a heterogeneous HORN
model = HORNSeqNetwork(
    dim_input,
    args.num_hidden,
    dim_output,
    alpha=args.alpha,
    omega=args.omega,
    gamma=args.gamma,
    v=args.v,
    delay=braintools.init.Uniform(1 * u.ms, 40 * u.ms)
)
weights = model.states(brainstate.ParamState)

# bce loss and optimizer for training
optimizer = braintools.optim.AdamW(lr=args.lr)
optimizer.register_trainable_weights(weights)


def batch_run(xs):
    batch_size = xs.shape[0]
    mapmodule = brainstate.nn.ModuleMapper(model, init_map_size=batch_size)
    mapmodule.init_all_states()
    mapmodule.param_precompute()
    out = mapmodule(xs)
    return out


def f_loss(xs, ys):
    prediction = batch_run(xs)
    loss_ = braintools.metric.softmax_cross_entropy_with_integer_labels(prediction, ys).mean()
    return loss_, prediction


@brainstate.transform.jit
def f_train(xs, ys):
    grads, loss_, preds = brainstate.transform.grad(f_loss, weights, return_value=True, has_aux=True)(xs, ys)
    optimizer.step(grads)
    acc = (preds.argmax(axis=-1) == ys).mean()
    return loss_, acc


@brainstate.transform.jit
def f_predict(xs, ys):
    loss_, prediction = f_loss(xs, ys)
    acc_num = (prediction.argmax(axis=-1) == ys).sum()
    return loss_, acc_num


def data_processing(images_, labels_):
    images_ = jnp.asarray(images_)
    images_ = images_.reshape(images_.shape[0], 1, 784)
    images_ = jnp.transpose(images_, (0, 2, 1))
    if args.shuffle:
        images_ = images_[:, perm]
    return images_, jnp.asarray(labels_)


# run inference on test set
def evaluate_model(data_loader):
    correct = 0
    test_loss = 0

    # loop over batches in data loader
    for i, (images, labels) in enumerate(data_loader):
        images, labels = data_processing(images, labels)

        # run model inference - record true returns dynamics
        loss, acc_num = f_predict(images, labels)

        # compute loss + number of correct predictions
        test_loss += float(loss)
        correct += acc_num

    # compute loss and accuracy
    test_loss /= len(data_loader)
    accuracy = 100. * correct / len(data_loader.dataset)

    return float(accuracy)


# training loop
best_eval = 0.
for epoch in range(args.epochs):
    # loop over batches for one epoch
    for batch, (images, labels) in enumerate(train_loader):
        images, labels = data_processing(images, labels)
        loss, acc = f_train(images, labels)
        if batch % 100 == 0:
            test_acc = evaluate_model(test_loader)
            print(f'epoch {epoch} batch {batch}: test acc {test_acc:.2f}')

    # compute validation and test accuracy
    valid_acc = evaluate_model(valid_loader)
    test_acc = evaluate_model(test_loader)
    if valid_acc > best_eval:
        best_eval = valid_acc
        final_test_acc = test_acc

    # log accuracy
    print(f'val: {valid_acc:.4f}, test: {test_acc:.4f}')
