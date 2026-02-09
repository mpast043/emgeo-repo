"""
Scalar field substrate implementation.

Free massive scalar field on a cubic lattice.

Notes on scaling:
- Full lattice_size=64 has 64^3 = 262,144 sites.
- Dense pairwise distances / dense G is not feasible at that size.
- Use subsample_stride or sampling to reduce to a manageable n_sites.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .base import QuantumSubstrate


class ScalarFieldSubstrate(QuantumSubstrate):
    def __init__(
        self,
        lattice_size: int = 5,
        subsample_stride: int = 1,
        lattice_spacing: float = 1.0,
        mass: float = 1.0,
        dtype: np.dtype = np.float64,
        name: str = "scalar_field",
        max_dense_sites: int = 6000,
    ):
        super().__init__(name=name)

        self.N = int(lattice_size)
        self.stride = max(int(subsample_stride), 1)
        self.a = float(lattice_spacing)
        self.m = float(mass)
        self.dtype = dtype
        self.max_dense_sites = int(max_dense_sites)

        self._construct_lattice()
        self._compute_correlators()

    def _construct_lattice(self) -> None:
        idx = np.arange(0, self.N, self.stride, dtype=np.int32)
        x = idx * self.a
        y = idx * self.a
        z = idx * self.a

        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        self.sites = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float64, copy=False)

        self.n_sites = int(self.sites.shape[0])
        self.n_sites_total = int(self.N ** 3)

        if self.n_sites > self.max_dense_sites:
            raise MemoryError(
                f"Selected n_sites={self.n_sites} exceeds max_dense_sites={self.max_dense_sites}. "
                "Increase subsample_stride or lower lattice_size for dense runs."
            )

    def _compute_correlators(self) -> None:
        distances = squareform(pdist(self.sites, metric="euclidean")).astype(np.float64, copy=False)
        r = np.where(distances > 1e-12, distances, 1e-12)

        log_prefactor = np.log(self.m / (4.0 * np.pi))
        self.log_G = (log_prefactor - np.log(r) - self.m * r).astype(np.float64, copy=False)

        diag_log_G = log_prefactor - np.log(self.a)
        np.fill_diagonal(self.log_G, diag_log_G)

        self.G = np.exp(self.log_G).astype(self.dtype, copy=False)
        np.fill_diagonal(self.G, self.m / (4.0 * np.pi * self.a))

    @property
    def correlation_length(self) -> float:
        return 1.0 / self.m

    def __repr__(self) -> str:
        return (
            f"ScalarFieldSubstrate(N={self.N}, a={self.a}, m={self.m}, "
            f"n_sites={self.n_sites}/{self.n_sites_total}, ξ={self.correlation_length:.3f})"
        )