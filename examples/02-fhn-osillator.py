#%% md
# # The Fitz-Hugh Nagumo oscillator
# 
# The Fitz-Hugh Nagumo (FHN) model is a simplified representation of neuronal activity, capturing essential features of excitability and oscillations. It consists of two coupled differential equations representing a fast activator variable and a slow recovery variable. The model is often used to study the dynamics of single neurons or neural populations.
# 
#%%
import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import numpy as np

import brainmass
#%%
brainstate.environ.set(dt=0.1 * u.ms)
#%% md
# ## Single node simulation
# 
# 
# We use the polynomial FitzHugh-Nagumo form implemented in BrainMass:
# 
# $$
# \begin{aligned}
# \dot V &= -\alpha V^{3} + \beta V^{2} + \gamma V - w + I_V(t), \
# \tau \dot w &= V - \delta - \epsilon w + I_w(t).
# \end{aligned}
# $$
# 
# - $V$: activator (membrane-potential-like), $w$: slow recovery.
# - $\alpha, \beta, \gamma, \delta, \epsilon$: dimensionless shape parameters; $\tau$: time constant.
# - Units: right-hand sides have unit $1/\mathrm{ms}$ in this implementation (consistent with brainunit).
# 
#%% md
# We instantiate one FHN node with OU noise on both E and I. The `step_run(i)` helper advances the model one time step with a proper `(i, t)` context. The node returns the current excitatory rate `rE`.
#%%
node = brainmass.FitzHughNagumoModel(
    1,
    noise_V=brainmass.OUProcess(1, sigma=0.01),
    noise_w=brainmass.OUProcess(1, sigma=0.01),
)
brainstate.nn.init_all_states(node)


def step_run(i):
    with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
        node.update()
        return node.V.value, node.w.value


indices = np.arange(10000)
vs, ws = brainstate.transform.for_loop(step_run, indices)
#%% md
# The trace below shows `rE(t)` over time. With small noise and constant drive, the oscillator may settle to a fixed point or a limit cycle depending on parameters.
#%%
plt.plot(indices * brainstate.environ.get_dt(), vs, lw=2, label='V')
plt.plot(indices * brainstate.environ.get_dt(), ws, lw=2, label='W')
plt.xlabel("t [ms]")
plt.ylabel("Activity")
plt.legend()
plt.show()
#%% md
# ## Bifurcation diagram
# 
# Let's draw a simple one-dimensional bifurcation diagram of this model to orient ourselves in the parameter space.
# 
#%%
# these are the different input values that we want to scan
exc_inputs = np.linspace(0, 2, 50)
#%%
nodes = brainmass.FitzHughNagumoModel(exc_inputs.size)
brainstate.nn.init_all_states(nodes)


def step_run(i):
    with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
        return nodes.update(exc_inputs)


indices = np.arange(10000)
exec_activity = brainstate.transform.for_loop(step_run, indices)
exec_activity = exec_activity[1000:]  # discard initial transient
#%%
max_exc = exec_activity.max(axis=0)
min_exc = exec_activity.min(axis=0)
#%% md
# The resulting diagram shows how steady‑state activity or oscillation amplitude varies with the external excitatory drive.
#%%
plt.plot(exc_inputs, max_exc, c='k', lw=2)
plt.plot(exc_inputs, min_exc, c='k', lw=2)
plt.title("Bifurcation diagram of the Wilson-Cowan model")
plt.xlabel("Input to exc")
plt.ylabel("Min / max exc")
plt.show()
#%% md
# ## Load HCP connectome data
# 
# Let's load a sample of Human connectome data from Kaggle. This includes structural connectivity (`Cmat`) and distance (`Dmat`) matrices for 80 brain regions.
#%%
import os.path
import kagglehub

path = kagglehub.dataset_download("oujago/hcp-gw-data-samples")
data = braintools.file.msgpack_load(os.path.join(path, "hcp-data-sample.msgpack"))
#%% md
# ## Brain network
# 
# 
# For $N$ nodes with states $(V_i, w_i)$, a common diffusive coupling on $V$ is:
# 
# $$
# \begin{aligned}
# \dot V_i &= f(V_i, w_i) + K \sum_j W_{ij}(V_j - V_i) + I_i^V(t), \
# \tau \dot w_i &= g(V_i, w_i) + I_i^w(t),
# \end{aligned}
# $$
# 
# Here $W$ is the connectivity and $K$ a global gain. In BrainMass, you can assemble this using `brainmass.diffusive_coupling` with prefetches for the $V$ variables, or the module `DiffusiveCoupling`.
# 
# Relation to phase models: near a stable limit cycle (if parameters permit), the dynamics can be approximated by phase-only coupling (Kuramoto-type).
# 
#%%
class Network(brainstate.nn.Module):
    def __init__(self, signal_speed=2., k=1.):
        super().__init__()

        conn_weight = data['Cmat'].copy()
        np.fill_diagonal(conn_weight, 0)

        delay_time = data['Dmat'].copy() / signal_speed
        np.fill_diagonal(delay_time, 0)
        indices_ = np.arange(conn_weight.shape[1])
        indices_ = np.tile(np.expand_dims(indices_, axis=0), (conn_weight.shape[0], 1))

        n_node = conn_weight.shape[0]

        self.node = brainmass.FitzHughNagumoModel(
            n_node,
            noise_V=brainmass.OUProcess(n_node, sigma=0.01),
            noise_w=brainmass.OUProcess(n_node, sigma=0.01),
        )
        self.coupling = brainmass.DiffusiveCoupling(
            self.node.prefetch_delay('V', (delay_time * u.ms, indices_), init=braintools.init.Uniform(0, 0.05)),
            self.node.prefetch('V'),
            conn_weight,
            k=k  # set the global coupling strength of the brain network
        )

    def update(self):
        current = self.coupling()
        # let's put all nodes close to the limit cycle such that
        # noise can kick them in and out of the oscillation
        # all nodes get the same constant input
        rE = self.node(V_inp=current + 0.72)
        return rE

    def step_run(self, i):
        with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
            return self.update()
#%%
dt = brainstate.environ.get_dt()

net = Network()
brainstate.nn.init_all_states(net)
indices = np.arange(0, int(6 * u.second / dt))
exes = brainstate.transform.for_loop(net.step_run, indices)
#%% md
# We compute functional connectivity (pairwise correlation) from the simulated `rE` matrix and plot it alongside sample time series. Stronger structure‑function correspondence typically emerges around specific `k` and delay regimes.
#%%
fig, gs = braintools.visualize.get_figure(1, 2, 4, 6)
ax1 = fig.add_subplot(gs[0, 0])
fc = braintools.metric.functional_connectivity(exes)
ax = ax1.imshow(fc)
plt.colorbar(ax, ax=ax1)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(indices, exes[:, ::5], alpha=0.8)
plt.show()
#%%
