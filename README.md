# BrainMass: whole-brain modeling with differentiable neural mass models

<p align="center">
  	<img alt="Header image of braintrace." src="https://raw.githubusercontent.com/chaobrain/brainmass/main/docs/_static/brainmass.png" width=40%>
</p>

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Version](https://img.shields.io/pypi/v/brainmass.svg)](https://pypi.org/project/brainmass/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/brainmass.svg)](https://pypi.org/project/brainmass/)
[![CI](https://github.com/chaobrain/brainmass/actions/workflows/CI.yml/badge.svg)](https://github.com/chaobrain/brainmass/actions/workflows/CI.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://readthedocs.org/projects/brainmass/badge/?version=latest)](https://brainmass.readthedocs.io/)

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


## Citation

If you use BrainMass in your research, please cite:

```bibtex
@software{brainmass,
  title={BrainMass: Whole-brain modeling with differentiable neural mass models},
  author={BrainMass Developers},
  url={https://github.com/chaobrain/brainmass},
  version={0.0.4},
  year={2025}
}
```

## License

BrainMass is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## See also the ecosystem

[BrainMass](https://github.com/chaobrain/brainmass) is one of our brain simulation ecosystem: https://brainmodeling.readthedocs.io/
