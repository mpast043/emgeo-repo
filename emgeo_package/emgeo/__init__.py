"""
emgeo public API
"""

from .substrates.scalar_field import ScalarFieldSubstrate
from .extraction.geometry import GeometryExtractor


def extract_geometry(substrate, **kwargs):
    embed_dim = kwargs.pop("embed_dim", 3)
    return GeometryExtractor(substrate, embed_dim=embed_dim).extract_geometry(**kwargs)


__all__ = [
    "ScalarFieldSubstrate",
    "GeometryExtractor",
    "extract_geometry",
]
