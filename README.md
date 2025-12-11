# BrainMass

**Whole-brain modeling with differentiable neural mass models**

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

For CUDA 12 support:

```bash
pip install brainmass[cuda12]
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

# TPU support
pip install BrainX[tpu]
```

## Dependencies

Core dependencies:

- `jax`: High-performance computing and automatic differentiation
- `numpy`: Numerical computations
- `brainstate`: State management and neural dynamics
- `brainunit`: Unit system for neuroscience
- `brainscale`: Online learning support
- `braintools`: Additional analysis tools

  Optional dependencies:
- `matplotlib`: Plotting and visualization
- `nevergrad`: Parameter optimization

## Documentation

Full documentation is available at [brainmass.readthedocs.io](https://brainmass.readthedocs.io/).

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

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

## Support

- **Issues**: [GitHub Issues](https://github.com/chaobrain/brainmass/issues)
- **Documentation**: [ReadTheDocs](https://brainmass.readthedocs.io/)
- **Contact**: chao.brain@qq.com

## Ehe brain modeling ecosystem

See also the brain simulation ecosystem: https://brainmodeling.readthedocs.io/



---

**Keywords**: neural mass model, brain modeling, computational neuroscience, JAX, differentiable programming
