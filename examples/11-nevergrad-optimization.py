#%% md
# # Parameter Optimization with ``Nevergrad``
# 
# [nevergrad](https://github.com/facebookresearch/nevergrad) is a Python toolbox for performing gradient-free optimization.
#%% md
#  This notebook demonstrates gradient-free parameter optimization of a whole-brain network using Nevergrad.
#  
#  - Objective: tune the global coupling `k` of a Wilson–Cowan network so that simulated functional connectivity (FC) matches an empirical target FC.
#  - Approach: define a similarity-based loss `1 - corr(FC_target, FC_model)`, then minimize it with a Nevergrad optimizer.
#  - Acceleration: JIT-compile and `vmap` to evaluate multiple candidates in parallel.
# 
#%%
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np

import brainmass
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
#  We load a sample HCP dataset via `kagglehub` which provides structural (`Cmat`) and distance (`Dmat`) connectivity. For each BOLD time series, we compute FC and then average across scans.
#  
#  - `Cmat`: connection weights (used for coupling).
#  - `Dmat`: inter-node distances (used to build delays given a signal speed).
#  - `target_fc`: mean empirical FC used as optimization target.
#  
#  If dataset access fails, provide a local path and load the same `msgpack` file, or substitute your own target FC.
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
            noise_E=brainmass.OUProcess(80, sigma=sigma, init=brainstate.init.ZeroInit()),
            noise_I=brainmass.OUProcess(80, sigma=sigma, init=brainstate.init.ZeroInit()),
        )
        self.coupling = brainmass.DiffusiveCoupling(
            self.node.prefetch_delay('rE', (delay_time * u.ms, indices_), init=brainstate.init.Uniform(0, 0.05)),
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
    indices = np.arange(0, 6e3 * u.ms // brainstate.environ.get_dt())
    exes = brainstate.transform.for_loop(net.step_run, indices)
    fc = braintools.metric.functional_connectivity(exes)
    return braintools.metric.matrix_correlation(target_fc, fc)
#%% md
# ## Model and Coupling
#  
#  We simulate 80 Wilson–Cowan nodes with OU noise on both E and I. Diffusive coupling is applied on `rE` via `DiffusiveCoupling`:
#  
#  - Global gain `k` scales `Cmat`. 
#  - Delays derive from `Dmat / signal_speed` and are handled with `prefetch_delay`. 
#  - The update returns the current excitatory activity `rE`, which forms the time series used for FC.
# 
#%% md
# ## Simulation and Loss
#  
#  Given `k` (and fixed `sigma` here), we simulate ~6 seconds, compute model FC from the excitatory time series, then compute a similarity score:
#  
#  - `functional_connectivity(X)`: FC from time-by-node array `X`.
#  - `matrix_correlation(A,B)`: Pearson correlation between vectorized FCs.
#  - Loss: `1 - correlation` (maximize match ⇔ minimize loss).
#  
#  We wrap the evaluation to enable `vmap` over a batch of `k` values, and JIT-compile it for speed.
# 
#%%
@brainstate.transform.jit
def vmap_loss_fn(k):
    return 1 - brainstate.transform.vmap(lambda x: simulation(x, sigma=0.05))(k)
#%%
opt = braintools.optim.NevergradOptimizer(
    vmap_loss_fn, method='DE', n_sample=4, bounds={'k': [0.5, 3.0]}
)
opt.initialize()
opt.minimize(n_iter=10)

#%% md
# ## Practical Tips and Extensions
#  
#  - Increase `n_iter` and adjust `method` (e.g., CMA, PSO, DE) for better searches.
#  - Vectorize multiple parameters: extend `vmap_loss_fn` and `bounds` to include, e.g., `sigma` or `signal_speed`.
#  - Start with shorter simulations for quick iterations, then refine with longer runs.
#  - Set random seeds and consider fixed noise initial states for reproducibility.
#  - Ensure JAX backend (CPU/GPU) is configured to benefit from `jit`/`vmap`.
# 