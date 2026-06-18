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

Simulate a single brain region with the Hopf oscillator model. The integration time step
``dt`` is global state set once through ``brainstate.environ`` -- it is **not** a model argument:

```python
import brainmass
import brainstate
import brainunit as u
import numpy as np

# dt is a global, set once through the environment (not a model argument)
brainstate.environ.set(dt=0.1 * u.ms)

# Create a single-region Hopf oscillator in the limit-cycle regime (a > 0)
model = brainmass.HopfStep(in_size=1, a=0.25, w=0.2)
model.init_all_states()

# Advance the model one step at a time, recording the x-coordinate
def step(i):
    with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
        model.update()
        return model.x.value

x = brainstate.transform.for_loop(step, np.arange(1000))
print(x.shape)  # (1000, 1)
```

See the [Quickstart tutorial](https://brainx.chaobrain.com/brainmass/tutorials/quickstart.html)
for noise, multi-region networks, coupling, and forward (BOLD/EEG/MEG) modeling.


## Citation

If you use BrainMass in your research, please cite:

```bibtex
@software{brainmass,
  title={BrainMass: Whole-brain modeling with differentiable neural mass models},
  author={BrainMass Developers},
  url={https://github.com/chaobrain/brainmass},
  version={0.0.6},
  year={2025}
}
```

## License

BrainMass is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## See also the ecosystem

[BrainMass](https://github.com/chaobrain/brainmass) is one of our brain simulation ecosystem: https://brainx.chaobrain.com/
