"""
Geometry extraction: Π^eff operator.

Pipeline stages:
1) Π_local: Extract correlation functions (already in substrate)
2) Π_corr: Infer effective distances from correlators
3) Π_geom: Classical MDS (double-centered squared distances)

Notes
This module returns geometry objects that are natural for classical MDS:
- coordinates (embedding)
- Gram matrix B (inner products)
Optionally returns a metric surrogate:
- euclidean: identity in embedding space
- mahalanobis: inverse covariance of embedded coordinates (whitening metric)
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, Any, Literal

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.special import lambertw


CorrMethod = Literal["log_map", "yukawa_inversion", "lattice"]
MetricKind = Literal["euclidean", "mahalanobis", "none"]
NegPolicy = Literal["require_positive", "abs"]


class GeometryExtractor:
    """
    Extract emergent geometry from a quantum substrate.

    Expected substrate fields:
      substrate.n_sites: int
      substrate.G: (n_sites, n_sites) correlator matrix (symmetric)
      substrate.sites: (n_sites, d_true) coordinates (optional, used for diagnostics)
      substrate.m: mass parameter (used for yukawa_inversion)
      substrate.a: lattice spacing (optional, used for default capacity)
    """

    def __init__(self, substrate, capacity: Optional[float] = None, embed_dim: int = 3):
        self.substrate = substrate

        self.sites = getattr(substrate, "sites", None)
        self.n_sites = int(getattr(substrate, "n_sites"))
        self.G = np.asarray(getattr(substrate, "G"))

        if self.G.shape != (self.n_sites, self.n_sites):
            raise ValueError("substrate.G must be shape (n_sites, n_sites)")
        if not np.allclose(self.G, self.G.T, atol=1e-10):
            raise ValueError("substrate.G must be symmetric")

        self.embed_dim = int(embed_dim)
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")

        a = getattr(substrate, "a", None)
        self.capacity = float(capacity) if capacity is not None else (1.0 / float(a) if a is not None else 1.0)

        self.D_eff: Optional[np.ndarray] = None
        self.gram_B: Optional[np.ndarray] = None

        self.coords_inferred: Optional[np.ndarray] = None
        self.coords_aligned: Optional[np.ndarray] = None

        self.embedding_eigenvalues: Optional[np.ndarray] = None

        self.metric_kind: Optional[str] = None
        self.metric: Optional[np.ndarray] = None

        self.distance_extraction_error: Optional[float] = None
        self.reconstruction_error: Optional[float] = None
        self.mds_stress: Optional[float] = None

    def extract_distances(
        self,
        method: CorrMethod = "log_map",
        eps: float = 1e-12,
        neg_policy: NegPolicy = "require_positive",
    ) -> np.ndarray:
        """
        Π_corr: Infer distances from correlation decay.

        log_map:
          D_ij = -log(|G_ij| + eps) (robust, but scale is arbitrary)

        yukawa_inversion:
          Invert 3D Yukawa-like correlator:
            G(r) = (m / (4πr)) exp(-m r)
          Uses analytic inversion with Lambert W on principal branch.

        lattice:
          D_ij = ||x_i - x_j|| from substrate.sites (testing only)
        """
        if method == "log_map":
            self.D_eff = self._log_map_distances(eps=eps)
        elif method == "yukawa_inversion":
            self.D_eff = self._invert_yukawa_correlators_lambertw(eps=eps, neg_policy=neg_policy)
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
        Gabs = np.abs(self.G).astype(np.float64, copy=False) + float(eps)
        D = -np.log(Gabs)
        D = (D + D.T) / 2.0
        np.fill_diagonal(D, 0.0)

        if not np.all(np.isfinite(D)):
            raise ValueError("Non-finite distances produced by log_map; check correlator values")

        D[D < 0.0] = 0.0
        return D

    def _invert_yukawa_correlators_lambertw(self, eps: float, neg_policy: NegPolicy) -> np.ndarray:
        """
        Invert 3D Yukawa-like correlator:
          G(r) = (m / (4πr)) exp(-m r)
        => r = W(m^2 / (4π G)) / m   (principal branch k=0)

        neg_policy:
          require_positive: raise if any off-diagonal G <= 0
          abs: use |G| (changes physics, but tolerates sign/noise)
        """
        m = float(getattr(self.substrate, "m", 1.0))
        if m <= 0:
            raise ValueError("substrate.m must be positive for yukawa_inversion")

        G = self.G.astype(np.float64, copy=False)

        if neg_policy == "require_positive":
            off = G.copy()
            np.fill_diagonal(off, 1.0)
            if np.any(off <= 0.0):
                raise ValueError("Negative/zero off-diagonal correlators not supported by yukawa_inversion. "
                                 "Use neg_policy='abs' if you accept that modeling choice.")
            g = off
        elif neg_policy == "abs":
            g = np.abs(G)
            np.fill_diagonal(g, 1.0)
        else:
            raise ValueError(f"Unknown neg_policy: {neg_policy}")

        g = np.maximum(g, float(eps))

        arg = (m * m) / (4.0 * np.pi * g)
        np.fill_diagonal(arg, 1.0)

        w = lambertw(arg, k=0)
        r = (np.real(w) / m).astype(np.float64, copy=False)

        r[~np.isfinite(r)] = 0.0
        r[r < 0.0] = 0.0
        np.fill_diagonal(r, 0.0)

        r = (r + r.T) / 2.0
        return r

    def fit_geometry(
        self,
        metric_kind: MetricKind = "mahalanobis",
        tol_pos: float = 1e-10,
        use_truncated_above: int = 2000,
        procrustes_scale: bool = True,
        stress: bool = True,
    ) -> Dict[str, Any]:
        """
        Π_geom: Classical MDS using double-centering.

        Returns:
          coords: inferred coordinates (n, d)
          gram_B: Gram matrix (n, n)
          eigenvalues: embedding eigenvalues
          metric: optional metric surrogate in embedding space

        metric_kind:
          none: do not compute a metric surrogate
          euclidean: identity (dxd)
          mahalanobis: inverse covariance of embedded coords (dxd)
        """
        if self.D_eff is None:
            raise ValueError("Must call extract_distances() first")

        D = np.asarray(self.D_eff, dtype=np.float64)
        if not np.all(np.isfinite(D)):
            raise ValueError("D_eff contains non-finite values")
        if D.shape[0] != self.n_sites:
            raise ValueError("D_eff must be shape (n_sites, n_sites)")

        n = D.shape[0]
        D2 = D * D

        row_mean = D2.mean(axis=1, keepdims=True)
        col_mean = D2.mean(axis=0, keepdims=True)
        total_mean = float(D2.mean())
        B = -0.5 * (D2 - row_mean - col_mean + total_mean)
        B = (B + B.T) / 2.0

        self.gram_B = B

        if n > use_truncated_above:
            k = min(n - 1, max(self.embed_dim + 8, 12))
            try:
                vals, vecs = eigsh(B, k=k, which="LA")
                idx = np.argsort(vals)[::-1]
                eigenvalues = vals[idx]
                eigenvectors = vecs[:, idx]
            except Exception:
                eigenvalues, eigenvectors = eigh(B)
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
        else:
            eigenvalues, eigenvectors = eigh(B)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        self.embedding_eigenvalues = eigenvalues

        n_positive = int((eigenvalues > tol_pos).sum())
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

        if stress:
            dhat = squareform(pdist(self.coords_inferred, metric="euclidean"))
            self.mds_stress = float(np.sqrt(np.mean((dhat - D) ** 2)))
        else:
            self.mds_stress = None

        self.metric_kind = metric_kind
        if metric_kind == "none":
            self.metric = None
        elif metric_kind == "euclidean":
            self.metric = np.eye(d, dtype=np.float64)
        elif metric_kind == "mahalanobis":
            X = coords - coords.mean(axis=0, keepdims=True)
            cov = (X.T @ X) / max(X.shape[0] - 1, 1)
            cov = cov + 1e-9 * np.eye(d)
            self.metric = np.linalg.inv(cov)
        else:
            raise ValueError(f"Unknown metric_kind: {metric_kind}")

        if self.sites is not None and self.sites.shape[0] == n:
            self._align_coordinates(scale=procrustes_scale)
        else:
            self.coords_aligned = self.coords_inferred
            self.reconstruction_error = None

        return {
            "coordinates": self.coords_aligned,
            "coordinates_raw": self.coords_inferred,
            "gram": self.gram_B,
            "eigenvalues": self.embedding_eigenvalues,
            "metric_kind": self.metric_kind,
            "metric": self.metric,
            "errors": {
                "distance_extraction": self.distance_extraction_error,
                "reconstruction": self.reconstruction_error,
                "mds_stress": self.mds_stress,
            },
        }

    def _align_coordinates(self, scale: bool = True) -> None:
        self._require_sites()
        coords_true = np.asarray(self.sites, dtype=np.float64)

        coords_inf = np.asarray(self.coords_inferred, dtype=np.float64)
        if coords_true.shape[0] != coords_inf.shape[0]:
            raise ValueError("sites and inferred coords must have same number of points")

        coords_true_c = coords_true - coords_true.mean(axis=0, keepdims=True)
        coords_inf_c = coords_inf - coords_inf.mean(axis=0, keepdims=True)

        d = min(coords_true_c.shape[1], coords_inf_c.shape[1])
        A = coords_inf_c[:, :d]
        B = coords_true_c[:, :d]

        M = A.T @ B
        U, _, Vt = np.linalg.svd(M)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        aligned = A @ R

        if scale:
            denom = float(np.trace(aligned.T @ aligned))
            if denom > 1e-12:
                s = float(np.trace(aligned.T @ B) / denom)
                aligned = s * aligned

        self.reconstruction_error = float(np.sqrt(np.mean((B - aligned) ** 2)))

        out = coords_inf_c.copy()
        out[:, :d] = aligned
        self.coords_aligned = out + coords_true.mean(axis=0, keepdims=True)

    def _require_sites(self) -> None:
        if self.sites is None:
            raise ValueError("substrate.sites is required for this operation")

    def extract_geometry(
        self,
        corr_method: CorrMethod = "yukawa_inversion",
        metric_kind: MetricKind = "mahalanobis",
        neg_policy: NegPolicy = "require_positive",
        eps: float = 1e-12,
        procrustes_scale: bool = True,
    ) -> Dict[str, Any]:
        self.extract_distances(method=corr_method, eps=eps, neg_policy=neg_policy)
        return self.fit_geometry(metric_kind=metric_kind, procrustes_scale=procrustes_scale)

    def __repr__(self) -> str:
        status = "computed" if self.coords_inferred is not None else "not computed"
        return f"GeometryExtractor(n_sites={self.n_sites}, status={status})"