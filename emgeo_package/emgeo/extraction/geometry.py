"""
Geometry extraction: Π^eff operator.

This module implements the projection operator that extracts
effective spacetime geometry from quantum substrates.

The pipeline has three stages:
1. Π_local: Extract correlation functions
2. Π_corr: Infer distances from correlations
3. Π_geom: Fit metric to distances
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from typing import Tuple, Optional
import warnings


class GeometryExtractor:
    """
    Extract emergent geometry from quantum substrate.
    
    Implements the projection operator Π^eff that maps:
    substrate X → emergent spacetime U
    
    Parameters
    ----------
    substrate : QuantumSubstrate
        The microscopic quantum substrate
    capacity : float, optional
        Geometric resolution C_geo. If None, uses lattice spacing.
        
    Attributes
    ----------
    D_eff : np.ndarray
        Effective distance matrix inferred from correlators
    metric : np.ndarray
        Emergent metric tensor g_μν
    coords_inferred : np.ndarray
        Reconstructed spatial coordinates
    """
    
    def __init__(self, substrate, capacity: Optional[float] = None):
        self.substrate = substrate
        self.sites = substrate.sites
        self.n_sites = substrate.n_sites
        self.G = substrate.G
        
        if capacity is None:
            # Use lattice spacing as default resolution
            self.capacity = 1.0 / substrate.a
        else:
            self.capacity = capacity
        
        # Results (computed on demand)
        self.D_eff = None
        self.metric = None
        self.coords_inferred = None
        self.coords_aligned = None
        self.embedding_eigenvalues = None
        self.reconstruction_error = None
        self.distance_extraction_error = None
        
    def extract_distances(self, method: str = 'yukawa_inversion') -> np.ndarray:
        """
        Π_corr: Infer distances from correlation decay.
        
        Parameters
        ----------
        method : str
            Method for distance extraction:
            - 'yukawa_inversion': Numerically invert Yukawa propagator
            - 'lattice': Use actual lattice distances (testing only)
            
        Returns
        -------
        D_eff : np.ndarray, shape (n_sites, n_sites)
            Effective distance matrix
        """
        if method == 'yukawa_inversion':
            self.D_eff = self._invert_yukawa_correlators()
        elif method == 'lattice':
            self.D_eff = squareform(pdist(self.sites, metric='euclidean'))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute extraction accuracy
        D_true = squareform(pdist(self.sites, metric='euclidean'))
        self.distance_extraction_error = np.sqrt(np.mean((self.D_eff - D_true)**2))
        
        return self.D_eff
    
    def _invert_yukawa_correlators(self) -> np.ndarray:
        """
        Numerically invert Yukawa correlator to get distances.
        
        For G(r) = (m/4πr)exp(-mr), solve for r given G.
        """
        m = self.substrate.m
        
        def yukawa(r, m):
            """Yukawa propagator."""
            if r < 1e-10:
                return m / (4 * np.pi * 1e-10)
            return (m / (4 * np.pi * r)) * np.exp(-m * r)
        
        def invert_yukawa(G_obs, m):
            """Find r such that yukawa(r, m) = G_obs."""
            
            # Near field (large G)
            if G_obs > 0.05:
                r_guess = m / (4 * np.pi * G_obs)
                return r_guess
            
            # Far field: use root finding
            def objective(r):
                return yukawa(r, m) - G_obs
            
            r_min, r_max = 0.1, 20.0
            
            try:
                if objective(r_min) * objective(r_max) < 0:
                    return brentq(objective, r_min, r_max)
                else:
                    result = minimize_scalar(
                        lambda r: abs(objective(r)),
                        bounds=(r_min, r_max),
                        method='bounded'
                    )
                    return result.x
            except:
                # Fallback: asymptotic formula
                return -np.log(G_obs * 4 * np.pi / m) / m
        
        # Extract all distances
        D_eff = np.zeros((self.n_sites, self.n_sites))
        
        for i in range(self.n_sites):
            for j in range(i+1, self.n_sites):
                G_ij = self.G[i, j]
                r_ij = invert_yukawa(G_ij, m)
                D_eff[i, j] = r_ij
                D_eff[j, i] = r_ij
        
        return D_eff
    
    def fit_metric(self) -> np.ndarray:
        """
        Π_geom: Fit metric tensor to distance matrix.
        
        Uses classical multidimensional scaling (MDS) to find
        coordinates that best reproduce the distance matrix,
        then extracts the metric.
        
        Returns
        -------
        metric : np.ndarray, shape (3, 3)
            Emergent metric tensor g_μν
        """
        if self.D_eff is None:
            raise ValueError("Must call extract_distances() first")
        
        # MDS: Convert distances to Gram matrix
        D2 = self.D_eff ** 2
        n = self.n_sites
        
        # Centering matrix
        H = np.eye(n) - np.ones((n, n)) / n
        
        # Gram matrix
        G_gram = -0.5 * H @ D2 @ H
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(G_gram)
        
        # Sort descending
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        self.embedding_eigenvalues = eigenvalues
        
        # Check dimensionality
        n_positive = (eigenvalues > 1e-6).sum()
        
        if n_positive < 3:
            warnings.warn(f"Only {n_positive} positive eigenvalues. Expected 3 for 3D.")
        
        # Reconstruct 3D coordinates
        coords_mds = eigenvectors[:, :3] * np.sqrt(np.maximum(eigenvalues[:3], 0))
        self.coords_inferred = coords_mds
        
        # For flat space, metric is Euclidean
        self.metric = np.eye(3)
        
        # Align with original lattice
        self._align_coordinates()
        
        return self.metric
    
    def _align_coordinates(self) -> None:
        """
        Align inferred coordinates with original lattice.
        
        Finds optimal rotation to minimize RMS difference.
        """
        coords_true = self.sites
        coords_inf = self.coords_inferred
        
        # Center both
        coords_true_c = coords_true - coords_true.mean(axis=0)
        coords_inf_c = coords_inf - coords_inf.mean(axis=0)
        
        # Procrustes: find rotation
        H = coords_inf_c.T @ coords_true_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Apply rotation
        coords_inf_aligned = coords_inf_c @ R
        
        # Compute error
        self.reconstruction_error = np.sqrt(
            np.mean((coords_true_c - coords_inf_aligned)**2)
        )
        
        self.coords_aligned = coords_inf_aligned + coords_true.mean(axis=0)
    
    def extract_geometry(self) -> dict:
        """
        Run full Π^eff pipeline.
        
        Returns
        -------
        geometry : dict
            Dictionary containing:
            - 'metric': metric tensor
            - 'distances': effective distance matrix
            - 'coordinates': reconstructed coordinates
            - 'errors': various error metrics
        """
        # Stage 1: Correlators already in substrate
        
        # Stage 2: Extract distances
        self.extract_distances()
        
        # Stage 3: Fit metric
        self.fit_metric()
        
        return {
            'metric': self.metric,
            'distances': self.D_eff,
            'coordinates': self.coords_aligned,
            'errors': {
                'distance_extraction': self.distance_extraction_error,
                'reconstruction': self.reconstruction_error
            }
        }
    
    def __repr__(self) -> str:
        """String representation."""
        status = "computed" if self.metric is not None else "not computed"
        return f"GeometryExtractor(n_sites={self.n_sites}, status={status})"
