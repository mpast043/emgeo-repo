"""
Scalar field substrate implementation.

Free massive scalar field on a lattice, used for testing emergent geometry.

Important:
For lattice_size=64, the full lattice has 64^3 = 262,144 sites.
Any code that forms all pairwise distances (pdist) or a dense correlator G
will require hundreds of GiB. This implementation adds safe subsampling.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .base import QuantumSubstrate


class ScalarFieldSubstrate(QuantumSubstrate):
    """
    Free massive scalar field on a cubic lattice.

    Parameters
    ----------
    lattice_size : int
        Number of sites per dimension (lattice_size^3 total sites).
    lattice_spacing : float
        Physical spacing between sites.
    mass : float
        Scalar field mass (correlation length ξ = 1/m).
    name : str
        Identifier.

    subsample_stride : int | None
        Keep only sites where i,j,k are multiples of stride.
        Example: lattice_size=64, stride=4 -> (64/4)^3 = 4096 sites.

    sample_sites : int | None
        Randomly sample this many sites from the full lattice.
        Use this if you want coverage without regular structure.
        Mutually exclusive with subsample_stride.

    seed : int
        RNG seed for sample_sites.

    dtype : np.dtype
        dtype for G. float32 recommended.

    max_dense_sites : int
        Guard rail. Raise early if selected sites exceed this.
    """

    def __init__(
        self,
        lattice_size: int = 5,
        lattice_spacing: float = 1.0,
        mass: float = 1.0,
        name: str = "scalar_field",
        subsample_stride: int | None = None,
        sample_sites: int | None = None,
        seed: int = 0,
        dtype: np.dtype = np.float32,
        max_dense_sites: int = 6000,
    ):
        super().__init__(name=name)

        if subsample_stride is not None and sample_sites is not None:
            raise ValueError("Use only one of subsample_stride or sample_sites, not both.")

        self.N = int(lattice_size)
        self.a = float(lattice_spacing)
        self.m = float(mass)

        self.subsample_stride = subsample_stride
        self.sample_sites = sample_sites
        self.seed = int(seed)
        self.dtype = dtype
        self.max_dense_sites = int(max_dense_sites)

        self._construct_lattice()
        self._compute_correlators()

    def _construct_lattice(self) -> None:
        grid = np.indices((self.N, self.N, self.N), dtype=np.int32).reshape(3, -1).T  # (N^3, 3)

        if self.subsample_stride is not None:
            s = int(self.subsample_stride)
            if s <= 0:
                raise ValueError("subsample_stride must be positive.")
            mask = (grid[:, 0] % s == 0) & (grid[:, 1] % s == 0) & (grid[:, 2] % s == 0)
            grid = grid[mask]

        if self.sample_sites is not None:
            n_total = grid.shape[0]
            n_keep = int(self.sample_sites)
            if n_keep <= 0 or n_keep > n_total:
                raise ValueError(f"sample_sites must be in [1, {n_total}]")
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n_total, size=n_keep, replace=False)
            grid = grid[idx]

        self.grid_ijk = grid
        self.sites = (grid.astype(np.float64) * self.a)
        self.n_sites = int(self.sites.shape[0])

        self.n_sites_total = self.N ** 3  # for reporting

    def _compute_correlators(self) -> None:
        if self.n_sites > self.max_dense_sites:
            raise MemoryError(
                f"Selected n_sites={self.n_sites} exceeds max_dense_sites={self.max_dense_sites}. "
                "Use subsample_stride or sample_sites to reduce the dense problem size."
            )

        distances = squareform(pdist(self.sites, metric="euclidean")).astype(np.float64, copy=False)

        r = np.where(distances > 1e-10, distances, 1e-10)
        G = (self.m / (4.0 * np.pi * r)) * np.exp(-self.m * r)

        np.fill_diagonal(G, self.m / (4.0 * np.pi * self.a))
        self.G = G.astype(self.dtype, copy=False)

    @property
    def correlation_length(self) -> float:
        return 1.0 / self.m

    def __repr__(self) -> str:
        return (
            f"ScalarFieldSubstrate(N={self.N}, a={self.a}, m={self.m}, "
            f"n_sites={self.n_sites}/{self.n_sites_total}, ξ={self.correlation_length:.3f})"
        )