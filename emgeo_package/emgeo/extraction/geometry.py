"""
Geometry extraction: Π^eff operator.

Pipeline stages:
1) Π_local: Extract correlation functions (already in substrate)
2) Π_corr: Infer effective distances from correlators
3) Π_geom: Fit coordinates (MDS) and return a metric surrogate
"""

import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from typing import Optional
import warnings


class GeometryExtractor:
    """
    Extract emergent geometry from quantum substrate.

    Expected substrate fields
    substrate.sites: (n_sites, d_true) array of site coordinates (optional but recommended)
    substrate.n_sites: int
    substrate.G: (n_sites, n_sites) correlator matrix
    substrate.m: mass parameter (optional, only for yukawa_inversion)

    Notes
    This implementation prioritizes numerical robustness:
    - Default Π_corr uses a monotone map D = -log(|G| + eps)
    - Yukawa inversion is available but guarded and uses adaptive bracketing
    """

    def __init__(self, substrate, capacity: Optional[float] = None, embed_dim: int = 2):
        self.substrate = substrate

        self.sites = getattr(substrate, "sites", None)
        self.n_sites = int(getattr(substrate, "n_sites"))
        self.G = np.asarray(getattr(substrate, "G"), dtype=float)

        if self.G.shape != (self.n_sites, self.n_sites):
            raise ValueError("substrate.G must be shape (n_sites, n_sites)")
        if not np.allclose(self.G, self.G.T, atol=1e-10):
            raise ValueError("substrate.G must be symmetric")

        self.embed_dim = int(embed_dim)
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")

        a = getattr(substrate, "a", None)
        if capacity is None:
            self.capacity = 1.0 / float(a) if a is not None else 1.0
        else:
            self.capacity = float(capacity)

        self.D_eff = None
        self.metric = None
        self.coords_inferred = None
        self.coords_aligned = None
        self.embedding_eigenvalues = None
        self.reconstruction_error = None
        self.distance_extraction_error = None

    def extract_distances(self, method: str = "log_map", eps: float = 1e-12) -> np.ndarray:
        """
        Π_corr: Infer distances from correlation decay.

        Methods
        log_map:
            D_ij = -log(|G_ij| + eps). Always defined, robust.
        yukawa_inversion:
            Invert a Yukawa-like propagator. Only use if you know the correlator matches it.
        lattice:
            Uses true lattice distances from substrate.sites (testing only).
        """
        if method == "log_map":
            self.D_eff = self._log_map_distances(eps=eps)
        elif method == "yukawa_inversion":
            self.D_eff = self._invert_yukawa_correlators(eps=eps)
        elif method == "lattice":
            self._require_sites()
            self.D_eff = squareform(pdist(self.sites, metric="euclidean"))
        else:
            raise ValueError(f"Unknown method: {method}")

        if self.sites is not None:
            D_true = squareform(pdist(self.sites, metric="euclidean"))
            self.distance_extraction_error = float(np.sqrt(np.mean((self.D_eff - D_true) ** 2)))
        else:
            self.distance_extraction_error = None

        return self.D_eff

    def _log_map_distances(self, eps: float) -> np.ndarray:
        G = np.abs(self.G) + float(eps)
        D = -np.log(G)
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)

        if not np.all(np.isfinite(D)):
            raise ValueError("Non-finite distances produced by log_map; check correlator values")

        D[D < 0.0] = 0.0
        return D

    def _invert_yukawa_correlators(self, eps: float) -> np.ndarray:
        """
        Invert a Yukawa correlator to get distances.

        Model:
        G(r) = (m / (4πr)) exp(-m r)   (3D Yukawa-like form)

        Guardrails:
        - clamps G_obs to positive
        - adaptive bracketing for brentq
        - returns finite r even in extreme tails
        """
        m = float(getattr(self.substrate, "m", 1.0))
        if m <= 0:
            raise ValueError("substrate.m must be positive for yukawa_inversion")

        def yukawa(r: float) -> float:
            r = max(r, 1e-12)
            return (m / (4.0 * np.pi * r)) * np.exp(-m * r)

        def invert_one(G_obs: float) -> float:
            g = float(np.abs(G_obs))
            g = max(g, float(eps))

            # If g is larger than the near-field value at tiny r, clamp to small distance
            g0 = yukawa(1e-12)
            if g >= g0:
                return 1e-12

            # Find bracket [lo, hi] such that yukawa(lo) >= g >= yukawa(hi)
            lo = 1e-12
            hi = 1.0 / m
            if hi <= lo:
                hi = lo * 10.0

            # Expand hi until yukawa(hi) <= g or hi is huge
            for _ in range(80):
                if yukawa(hi) <= g:
                    break
                hi *= 2.0

            # If we never got below g, then g is extremely tiny; use asymptotic estimate
            if yukawa(hi) > g:
                # For large r: yukawa(r) ~ (m/(4πr)) e^{-mr}
                # Solve approximately: mr + log r = log(m/(4πg))
                # Use one-step approximation ignoring log r then refine once.
                target = np.log(m / (4.0 * np.pi * g))
                r = max(target / m, 1e-12)
                r = max((target - np.log(max(r, 1e-12))) / m, 1e-12)
                return r

            # Now solve yukawa(r) - g = 0 on [lo, hi]
            def f(r: float) -> float:
                return yukawa(r) - g

            try:
                return float(brentq(f, lo, hi, maxiter=200))
            except Exception:
                # Fallback to monotone log-map distance scaled by 1/m
                return float(max(-np.log(g) / m, 1e-12))

        D_eff = np.zeros((self.n_sites, self.n_sites), dtype=float)
        for i in range(self.n_sites):
            for j in range(i + 1, self.n_sites):
                r_ij = invert_one(self.G[i, j])
                D_eff[i, j] = r_ij
                D_eff[j, i] = r_ij

        return D_eff

    def fit_metric(self) -> np.ndarray:
        """
        Π_geom: Fit coordinates using classical MDS.

        Returns a metric surrogate:
        inverse covariance of embedded coordinates (regularized)
        This is not curvature, but a stable diagnostic object.
        """
        if self.D_eff is None:
            raise ValueError("Must call extract_distances() first")

        D = np.asarray(self.D_eff, dtype=float)
        if not np.all(np.isfinite(D)):
            raise ValueError("D_eff contains non-finite values")

        D2 = D ** 2
        n = self.n_sites

        H = np.eye(n) - np.ones((n, n)) / n
        G_gram = -0.5 * H @ D2 @ H

        eigenvalues, eigenvectors = eigh(G_gram)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.embedding_eigenvalues = eigenvalues

        n_positive = int((eigenvalues > 1e-10).sum())
        if n_positive < self.embed_dim:
            warnings.warn(
                f"Only {n_positive} positive eigenvalues. "
                f"Reducing embed_dim from {self.embed_dim} to {max(n_positive, 1)}."
            )
            d = max(n_positive, 1)
        else:
            d = self.embed_dim

        coords = eigenvectors[:, :d] * np.sqrt(np.maximum(eigenvalues[:d], 0.0))
        self.coords_inferred = coords

        X = coords - coords.mean(axis=0, keepdims=True)
        cov = (X.T @ X) / max(X.shape[0] - 1, 1)
        cov = cov + 1e-9 * np.eye(d)
        self.metric = np.linalg.inv(cov)

        if self.sites is not None and self.sites.shape[0] == self.n_sites:
            self._align_coordinates()
        else:
            self.coords_aligned = self.coords_inferred
            self.reconstruction_error = None

        return self.metric

    def _align_coordinates(self) -> None:
        self._require_sites()
        coords_true = np.asarray(self.sites, dtype=float)

        coords_inf = np.asarray(self.coords_inferred, dtype=float)
        if coords_true.shape[0] != coords_inf.shape[0]:
            raise ValueError("sites and inferred coords must have same number of points")

        coords_true_c = coords_true - coords_true.mean(axis=0, keepdims=True)
        coords_inf_c = coords_inf - coords_inf.mean(axis=0, keepdims=True)

        # If inferred dimension differs from true dimension, align in the smaller space
        d_true = coords_true_c.shape[1]
        d_inf = coords_inf_c.shape[1]
        d = min(d_true, d_inf)

        A = coords_inf_c[:, :d]
        B = coords_true_c[:, :d]

        M = A.T @ B
        U, _, Vt = np.linalg.svd(M)
        R = Vt.T @ U.T

        # Fix improper rotation (reflection) if needed
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        aligned = A @ R
        self.reconstruction_error = float(np.sqrt(np.mean((B - aligned) ** 2)))

        out = coords_inf_c.copy()
        out[:, :d] = aligned
        self.coords_aligned = out + coords_true.mean(axis=0, keepdims=True)

    def _require_sites(self) -> None:
        if self.sites is None:
            raise ValueError("substrate.sites is required for this operation")

    def extract_geometry(self, corr_method: str = "log_map") -> dict:
        self.extract_distances(method=corr_method)
        self.fit_metric()
        return {
            "metric": self.metric,
            "distances": self.D_eff,
            "coordinates": self.coords_aligned,
            "errors": {
                "distance_extraction": self.distance_extraction_error,
                "reconstruction": self.reconstruction_error,
            },
        }

    def __repr__(self) -> str:
        status = "computed" if self.metric is not None else "not computed"
        return f"GeometryExtractor(n_sites={self.n_sites}, status={status})"
