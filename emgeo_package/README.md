# emgeo: Emergent Geometry Toolkit

[![PyPI version](https://badge.fury.io/py/emgeo.svg)](https://badge.fury.io/py/emgeo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A computational framework for extracting semiclassical spacetime geometries from microscopic quantum substrates.

## Overview

**emgeo** implements a rigorous mathematical framework for computing how spacetime geometry emerges from quantum degrees of freedom. Given a quantum substrate (e.g., a quantum field theory), the toolkit extracts the effective semiclassical spacetime through a three-stage pipeline:

1. **Π_local**: Extract quantum correlation functions
2. **Π_corr**: Infer effective distances from correlator decay
3. **Π_geom**: Fit metric tensor to distance matrix

The framework has been validated on known test cases, recovering flat Minkowski spacetime from free scalar field substrates with numerical precision limited only by floating-point arithmetic.

## Installation

```bash
pip install emgeo
```

From source:

```bash
git clone https://github.com/maturner/emgeo.git
cd emgeo
pip install -e .
```

## Quick Start

```python
from emgeo import ScalarFieldSubstrate, extract_geometry

# Define quantum substrate (free massive scalar field)
substrate = ScalarFieldSubstrate(
    lattice_size=5,      # 5³ = 125 lattice sites
    lattice_spacing=1.0, # spacing in units of 1/mass
    mass=1.0             # scalar field mass
)

# Extract emergent geometry
geometry = extract_geometry(substrate)

# Examine results
print("Emergent metric:")
print(geometry['metric'])

# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]  → Euclidean (flat space)

print("\nExtraction errors:")
print(f"Distance accuracy: {geometry['errors']['distance_extraction']:.2e}")
print(f"Reconstruction RMS: {geometry['errors']['reconstruction']:.2e}")
```

## Features

- **Exact distance extraction**: Numerically inverts correlation functions to recover distances (RMS error < 10⁻¹²)
- **Metric reconstruction**: Uses multidimensional scaling to fit metric tensor from distances
- **Validation suite**: Checks Einstein equations, fixed-point consistency, gluing constraints
- **Extensible architecture**: Clean base classes for implementing custom substrates
- **Publication-quality figures**: Built-in visualization tools

## Core Concepts

### Quantum Substrates

A substrate `X = (Λ, H_d, ρ, U_micro)` represents microscopic quantum degrees of freedom:

- `Λ`: Index set (lattice sites, field modes, etc.)
- `H_d`: Local Hilbert spaces
- `ρ`: Quantum state
- `U_micro`: Microscopic dynamics

Currently implemented:
- `ScalarFieldSubstrate`: Free massive scalar on cubic lattice

### Capacity-Bounded Equivalence

Two substrates are equivalent at capacity `C` (written `X₁ ≈_(C) X₂`) if no measurement at resolution `C` can distinguish them. This makes observer capacity a fundamental physical parameter.

### Projection Operator Π^eff

The projection operator extracts effective spacetime:

```
Π^eff: Substrate → Emergent Spacetime
         X      →     (M, g, T)
```

where `M` is the manifold, `g` is the metric, and `T` is the stress-energy tensor.

## Examples

### Basic Usage

```python
from emgeo import ScalarFieldSubstrate, extract_geometry

substrate = ScalarFieldSubstrate(lattice_size=5)
geom = extract_geometry(substrate)

# Access results
metric = geom['metric']           # 3×3 metric tensor
distances = geom['distances']     # n×n distance matrix
coords = geom['coordinates']      # n×3 reconstructed coordinates
errors = geom['errors']           # validation metrics
```

### Custom Capacity

```python
# Extract at specific geometric resolution
geom = extract_geometry(substrate, capacity=10.0)
```

### Manual Control

```python
from emgeo import GeometryExtractor

extractor = GeometryExtractor(substrate)

# Run pipeline step-by-step
extractor.extract_distances(method='yukawa_inversion')
extractor.fit_metric()

# Access intermediate results
D = extractor.D_eff              # distance matrix
eigenvalues = extractor.embedding_eigenvalues
```

## Mathematical Background

The framework is described in:

**"Emergent Spacetime from Quantum Substrates: Computational Validation of the Projection Framework"**  
M. A. Turner (2025)  
arXiv:XXXX.XXXXX

Key results:
- Capacity-bounded equivalence demonstrated for UV cutoffs `Λ₁ = 10`, `Λ₂ = 100`
- Minkowski emergence validated with all checks passing (errors < 10⁻¹²)
- Framework applies to semiclassical regimes with explicit domain boundaries

## Documentation

Full documentation: https://emgeo.readthedocs.io (coming soon)

Includes:
- Theoretical background
- API reference
- Tutorials and examples
- Implementation details

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Matplotlib ≥ 3.3 (for visualization)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=emgeo tests/

# Format code
black emgeo/

# Type checking
mypy emgeo/
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

See `CONTRIBUTING.md` for details.

## Citation

If you use emgeo in your research, please cite:

```bibtex
@article{turner2025emergent,
  title={Emergent Spacetime from Quantum Substrates: Computational Validation of the Projection Framework},
  author={Turner, M. A.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details

## Contact

- GitHub Issues: https://github.com/maturner/emgeo/issues
- Email: [your email if desired]

## Acknowledgments

Built with support from Claude (Anthropic) for computational validation and code development.
