"""
Scalar field substrate implementation.

Free massive scalar field on a lattice, the simplest non-trivial
quantum substrate for testing emergent geometry.
"""

import numpy as np
from .base import QuantumSubstrate


class ScalarFieldSubstrate(QuantumSubstrate):
    """
    Free massive scalar field on a cubic lattice.
    
    This represents a quantum field theory with:
    - Equation of motion: (∂² - m²)φ = 0
    - State: Ground state (Gaussian)
    - Correlator: Yukawa propagator G(r) = (m/4πr)exp(-mr)
    
    Parameters
    ----------
    lattice_size : int
        Number of sites per dimension (creates lattice_size³ total sites)
    lattice_spacing : float
        Physical spacing between sites (units of 1/mass)
    mass : float
        Scalar field mass (sets correlation length ξ = 1/m)
    name : str, optional
        Human-readable identifier
        
    Attributes
    ----------
    sites : np.ndarray, shape (n_sites, 3)
        3D coordinates of all lattice sites
    G : np.ndarray, shape (n_sites, n_sites)
        Correlation matrix G[i,j] = ⟨0|φ(i)φ(j)|0⟩
    """
    
    def __init__(
        self,
        lattice_size: int = 5,
        lattice_spacing: float = 1.0,
        mass: float = 1.0,
        name: str = "scalar_field"
    ):
        super().__init__(name=name)
        
        self.N = lattice_size
        self.a = lattice_spacing
        self.m = mass
        
        # Build substrate
        self._construct_lattice()
        self._compute_correlators()
        
    def _construct_lattice(self) -> None:
        """Construct 3D cubic lattice."""
        # Create 3D grid
        x = np.arange(self.N) * self.a
        y = np.arange(self.N) * self.a
        z = np.arange(self.N) * self.a
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten to list of sites
        self.sites = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
        self.n_sites = len(self.sites)
        
    def _compute_correlators(self) -> None:
        """
        Compute ground state two-point function.
        
        For free massive scalar in 3D:
        G(r) = (m / 4πr) * exp(-mr)  (Yukawa propagator)
        
        With UV regularization at lattice spacing a.
        """
        from scipy.spatial.distance import pdist, squareform
        
        # Pairwise distances
        distances = squareform(pdist(self.sites, metric='euclidean'))
        
        # Avoid r=0 singularity
        r = np.where(distances > 1e-10, distances, 1e-10)
        
        # Yukawa propagator
        self.G = (self.m / (4 * np.pi * r)) * np.exp(-self.m * r)
        
        # Set diagonal (r→0 limit) using UV cutoff
        np.fill_diagonal(self.G, self.m / (4 * np.pi * self.a))
    
    @property
    def correlation_length(self) -> float:
        """Correlation length ξ = 1/m."""
        return 1.0 / self.m
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"ScalarFieldSubstrate(N={self.N}, a={self.a}, m={self.m}, "
                f"n_sites={self.n_sites}, ξ={self.correlation_length:.3f})")
