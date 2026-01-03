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

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import brainstate
import braintools
from brainmass._horn import HORNSeqLayer

brainstate.environ.set(dt=1.0)  # set global time step

# 1-dim time series as input
num_input = 1
num_hidden = 32

# hyperparameters
alpha = 0.04  # excitability
omega_base = 2. * math.pi / 28.  # natural frequency
gamma_base = 0.01  # damping

# oscillation parameters - varying across units
# uncomment to enable
omega_min = 0.5 * omega_base
omega_max = 2.0 * omega_base
gamma_min = 0.5 * gamma_base
gamma_max = 2.0 * gamma_base

# construct heterogeneous HORN
model = HORNSeqLayer(
    num_input,
    num_hidden,
    alpha=alpha,
    omega=braintools.init.Uniform(omega_min, omega_max),
    gamma=braintools.init.Uniform(gamma_min, gamma_max),
    v=0.1,
    inp_w_init=braintools.init.Normal(0, 1.),
    rec_w_init=braintools.init.Normal(0, 0.1),
    state_init=braintools.init.Normal(0., 1.),
)

# compute dynamics for 1000 time steps
domain = jnp.arange(1000)

# stimulus shape: input length x batch x input dimension, i.e. 1000 x 1 x 1
stimulus = np.expand_dims(np.sin(domain * np.pi * 2 / 60), axis=1)
stimulus[500:] = 0  # not stimulus for last 500 time steps

# run model dynamics
model.init_all_states()
model.param_precompute()
x_t = model.update(stimulus)

fig, gs = braintools.visualize.get_figure(1, 2, 4., 5.)
# plot weight matrix
ax = fig.add_subplot(gs[0, 0])
ax.matshow(model.h2h.weight.value['weight'])
ax.set_title('W_hh')

# show amplitude dynamics
ax = fig.add_subplot(gs[0, 1])
for i in range(num_hidden):
    ax.plot(domain, x_t[:, i])
ax.plot(domain, stimulus[:], color='k', linewidth=2)
ax.set_xlabel('time')
ax.set_ylabel('amplitude')
plt.show()
