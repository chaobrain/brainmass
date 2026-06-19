# BrainMass: whole-brain modeling with differentiable neural mass models

<p align="center">
  	<img alt="Header image of braintrace." src="https://brainx.chaobrain.com/images/brainmass.webp" width=40%>
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Version](https://img.shields.io/pypi/v/brainmass.svg)](https://pypi.org/project/brainmass/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/brainmass.svg)](https://pypi.org/project/brainmass/)
[![CI](https://github.com/chaobrain/brainmass/actions/workflows/CI.yml/badge.svg)](https://github.com/chaobrain/brainmass/actions/workflows/CI.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://readthedocs.org/projects/brainmass/badge/?version=latest)](https://brainx.chaobrain.com/brainmass/)

BrainMass is a Python library for whole-brain computational modeling using differentiable neural mass models. Built on
JAX for high-performance computing, it provides tools for simulating brain dynamics, fitting neural signal data, and
training cognitive tasks.

## Installation

### From PyPI (recommended)

```bash
pip install brainmass
```

### From Source

```bash
git clone https://github.com/chaobrain/brainmass.git
cd brainmass
pip install -e .
```

### GPU Support

For CUDA support:

```bash
pip install brainmass[cuda12]
pip install brainmass[cuda13]
```

For TPU support:

```bash
pip install brainmass[tpu]
```

### Ecosystem

For whole brain modeling ecosystem:

```bash
pip install BrainX 

# GPU support
pip install BrainX[cuda12]
pip install BrainX[cuda13]

# TPU support
pip install BrainX[tpu]
```


## Quick Start

Simulate a single brain region with the Hopf oscillator model. The high-level
``Simulator`` drives the compiled run loop and collects the monitored trajectories into a
unit-aware result dict -- no hand-written stepping required. The integration time step
``dt`` is supplied to the ``Simulator``:

```python
import brainmass
import brainunit as u

# Create a single-region Hopf oscillator in the limit-cycle regime (a > 0)
node = brainmass.HopfStep(in_size=1, a=0.25, w=0.3)

# Run for 200 ms, dropping the first 20 ms transient, recording x and y
sim = brainmass.Simulator(node, dt=0.1 * u.ms)
res = sim.run(200 * u.ms, monitors=['x', 'y'], transient=20 * u.ms)

print(res['x'].shape)  # (1800, 1)

# Plot the limit cycle (matplotlib via the optional [viz] extra)
brainmass.viz.plot_timeseries(res['x'], ts=res['ts'])
```

Wiring a whole-brain network and fitting a parameter are just as direct:

```python
import brainmass
import brainunit as u
from brainstate.nn import Param

# A delay-coupled network from a bundled example connectome
conn = brainmass.datasets.load_dataset('example_connectome')
net = brainmass.Network(
    brainmass.HopfStep(in_size=conn.weights.shape[0], a=0.1, w=0.3),
    conn=conn.weights, distance=conn.distances, speed=10 * u.mm / u.ms,
    coupling='diffusive', coupled_var='x', k=0.5,
)

# Fit a model parameter to data with gradients (or swap to Nevergrad / SciPy)
node = brainmass.HopfStep(in_size=1, a=Param(0.1, fit=True), w=0.3)
fitter = brainmass.Fitter(node, loss_fn=my_loss)   # backend='grad' | 'nevergrad' | 'scipy'
result = fitter.fit(n_steps=50)
```

### Documentation

The full documentation is organized for different goals:

- **[Getting Started](https://brainx.chaobrain.com/brainmass/getting_started/index.html)** —
  installation, a five-minute [quickstart](https://brainx.chaobrain.com/brainmass/getting_started/quickstart.html),
  key concepts, and persona learning paths.
- **[Tutorials](https://brainx.chaobrain.com/brainmass/tutorials/index.html)** — a sequential
  path from a first simulation to building networks, forward modeling, fitting, and training.
- **[How-To Guides](https://brainx.chaobrain.com/brainmass/howto/index.html)** — task-focused
  recipes (choose a model, work with units, accelerate, custom coupling/objective, sweeps, analysis).
- **[Concepts](https://brainx.chaobrain.com/brainmass/concepts/index.html)** — the *why*:
  neural mass models, differentiable programming, architecture, coupling/delays, forward models.
- **[Data-Driven Modeling](https://brainx.chaobrain.com/brainmass/data_driven/index.html)** —
  the flagship guided path through the differentiable, data-driven workflow.
- **[Gallery](https://brainx.chaobrain.com/brainmass/gallery/index.html)** — a runnable model
  zoo (one demo per model family) and end-to-end case studies.
- **[API Reference](https://brainx.chaobrain.com/brainmass/reference/index.html)** — every
  public symbol, organized by category.
- **[Developer Guide](https://brainx.chaobrain.com/brainmass/developer/index.html)** —
  contributing and the extension playbooks (custom models, couplings, objectives, workflows).


## Citation

If you use BrainMass in your research, please cite:

```bibtex
@software{brainmass,
  title={BrainMass: Whole-brain modeling with differentiable neural mass models},
  author={BrainMass Developers},
  url={https://github.com/chaobrain/brainmass},
  version={0.1.0},
  year={2026}
}
```

## License

BrainMass is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## See also the ecosystem

[BrainMass](https://github.com/chaobrain/brainmass) is one of our brain simulation ecosystem: https://brainx.chaobrain.com/
