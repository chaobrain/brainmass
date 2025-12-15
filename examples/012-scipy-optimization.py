#%% md
# # Parameter Optimization with ``Scipy``
# 
# [scipy](https://docs.scipy.org/doc/scipy/tutorial/optimize.html) has provided many excellent optimization algorithms for decades. Here we show how to use them to fit a whole-brain network model to empirical functional connectivity (FC).
#%% md
# 
# This notebook demonstrates gradient-based (or gradient-free) parameter optimization with SciPy to fit a whole-brain network to empirical functional connectivity (FC).
# 
# - Goal: tune global coupling k and noise sigma of a Wilson–Cowan network so simulated FC matches a target FC.
# - Loss: 1 - corr(FC_target, FC_model).
# - We JIT-compile the loss for speed (optional) and call a SciPy optimizer.
# 
#%%
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

import brainmass
import braintools

#%%
brainstate.environ.set(dt=0.1 * u.ms)
#%%
import os.path
import kagglehub

path = kagglehub.dataset_download("oujago/hcp-gw-data-samples")
data = braintools.file.msgpack_load(os.path.join(path, "hcp-data-sample.msgpack"))

target_fc = [braintools.metric.functional_connectivity(x.T) for x in data['BOLDs']]
target_fc = jnp.mean(jnp.asarray(target_fc), axis=0)


#%% md
# ## Data and Target FC
# 
# We download a small HCP sample via kagglehub, providing structural connectivity (Cmat) and distances (Dmat). For each BOLD time series, we compute FC and then average across scans to obtain target_fc.
# 
# - Cmat: weights used for coupling.
# - Dmat: distances converted to delays using a signal speed.
# - target_fc: average empirical FC used by the loss.
# 
# If fetching fails, point to a local msgpack file, or replace with your own FC target.
# 
#%%
class Network(brainstate.nn.Module):
    def __init__(self, signal_speed=2., k=1., sigma=0.01):
        super().__init__()

        conn_weight = data['Cmat'].copy()
        np.fill_diagonal(conn_weight, 0)
        delay_time = data['Dmat'].copy() / signal_speed
        np.fill_diagonal(delay_time, 0)
        indices_ = np.arange(conn_weight.shape[1])
        indices_ = np.tile(np.expand_dims(indices_, axis=0), (conn_weight.shape[0], 1))

        self.node = brainmass.WilsonCowanModel(
            80,
            noise_E=brainmass.OUProcess(80, sigma=sigma, init=braintools.init.ZeroInit()),
            noise_I=brainmass.OUProcess(80, sigma=sigma, init=braintools.init.ZeroInit()),
        )
        self.coupling = brainmass.DiffusiveCoupling(
            self.node.prefetch_delay('rE', (delay_time * u.ms, indices_), init=braintools.init.Uniform(0, 0.05)),
            self.node.prefetch('rE'),
            conn_weight,
            k=k
        )

    def update(self):
        current = self.coupling()
        rE = self.node(current)
        return rE

    def step_run(self, i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            return self.update()


#%%
def simulation(k, sigma):
    net = Network(k=k, sigma=sigma)
    brainstate.nn.init_all_states(net)
    indices = np.arange(0, 1e3 * u.ms // brainstate.environ.get_dt())
    exes = brainstate.transform.for_loop(net.step_run, indices)
    fc = braintools.metric.functional_connectivity(exes)
    return braintools.metric.matrix_correlation(target_fc, fc)


#%% md
# ## Model and Coupling
# 
# We simulate 80 Wilson–Cowan nodes with OU noise on E and I. Diffusive coupling is applied on rE via DiffusiveCoupling:
# 
# - Global gain k scales Cmat.
# - Delays are Dmat / signal_speed and handled with prefetch_delay.
# - The module's update returns rE, used for FC computation.
# 
#%%
@brainstate.transform.jit
def loss_fn(arr):
    k, sigma = arr
    return 1 - simulation(k, sigma)


#%% md
# ## Simulation and Loss
# 
# The simulation runs the network for a short window, computes FC from excitatory activity, then returns correlation with target_fc. The loss is 1 - correlation so that lower is better. We wrap the two parameters (k, sigma) into a single array for SciPy.'s API and optionally JIT-compile for speed.
# 
#%%
opt = braintools.optim.ScipyOptimizer(
    loss_fn, bounds=[(0.5, 3.0), (0.0, 1.)], method='L-BFGS-B'
)
best_r = opt.minimize(n_iter=1)
print(best_r)

#%% md
# ## SciPy Optimizer Setup
# 
# We use braintools.optim.ScipyOptimizer as a thin wrapper around scipy.optimize.minimize. Bounds and method (e.g., L-BFGS-B, Nelder-Mead) can be customized. Increase n_iter or switch methods for robustness.
# 
# Tips:
# - Start with short simulations for fast iterations, then refine.
# - Consider multiple random restarts.
# - If gradients are unreliable (stochastic noise), prefer derivative-free methods (Nelder-Mead, Powell).
# 
#%% md
# ## Notes
# 
# - Ensure JAX backend is configured to benefit from jit.
# - Set seeds or fix noise initial states for reproducibility when comparing runs.
# - You can extend the loss to multi-objective (e.g., also matching power spectra).
# 