"""
Base classes for quantum substrates.

A substrate represents the microscopic quantum degrees of freedom
from which spacetime geometry emerges.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class QuantumSubstrate(ABC):
    """
    Abstract base class for quantum substrates.
    
    A substrate X = (Λ, H_d, ρ, U_micro) consists of:
    - Λ: index set (sites, modes, degrees of freedom)
    - H_d: local Hilbert spaces
    - ρ: global quantum state
    - U_micro: microscopic dynamics
    
    Concrete implementations must define:
    - _construct_lattice(): Build spatial structure
    - _compute_correlators(): Calculate quantum correlation functions
    """
    
    def __init__(self, name: str = "substrate"):
        """
        Initialize substrate base.
        
        Parameters
        ----------
        name : str
            Human-readable name for this substrate
        """
        self.name = name
        self.sites = None
        self.n_sites = 0
        self.G = None  # Correlation matrix
        
    @abstractmethod
    def _construct_lattice(self) -> None:
        """Construct spatial lattice structure."""
        pass
    
    @abstractmethod
    def _compute_correlators(self) -> None:
        """Compute quantum correlation functions."""
        pass
    
    def get_correlator(self, i: int, j: int) -> float:
        """
        Get correlation between sites i and j.
        
        Parameters
        ----------
        i, j : int
            Site indices
            
        Returns
        -------
        float
            Correlation G(i,j) = ⟨0|φ(i)φ(j)|0⟩
        """
        if self.G is None:
            raise ValueError("Correlators not computed. Call _compute_correlators() first.")
        return self.G[i, j]
    
    def get_local_correlators(
        self, 
        site_idx: int, 
        max_separation: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get correlators in neighborhood of a site.
        
        Parameters
        ----------
        site_idx : int
            Index of central site
        max_separation : float, optional
            Maximum distance to include. If None, includes all sites.
            
        Returns
        -------
        separations : np.ndarray
            Distances to neighbors
        correlations : np.ndarray
            Corresponding correlation values
        """
        if self.sites is None:
            raise ValueError("Lattice not constructed.")
        if self.G is None:
            raise ValueError("Correlators not computed.")
            
        # Distances from this site to all others
        r = np.linalg.norm(self.sites - self.sites[site_idx], axis=1)
        
        if max_separation is not None:
            mask = r <= max_separation
            r = r[mask]
            G_local = self.G[site_idx, mask]
        else:
            G_local = self.G[site_idx, :]
        
        # Sort by distance
        sort_idx = np.argsort(r)
        return r[sort_idx], G_local[sort_idx]
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', n_sites={self.n_sites})"
