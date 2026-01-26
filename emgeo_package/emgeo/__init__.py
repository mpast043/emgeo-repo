"""
emgeo: Emergent Geometry Toolkit

A computational framework for extracting semiclassical spacetime
geometries from microscopic quantum substrates.

Quick Start
-----------
>>> from emgeo import ScalarFieldSubstrate, extract_geometry
>>> 
>>> # Define quantum substrate
>>> substrate = ScalarFieldSubstrate(lattice_size=5, mass=1.0)
>>> 
>>> # Extract emergent geometry
>>> geometry = extract_geometry(substrate)
>>> 
>>> # Check results
>>> print(geometry['metric'])  # Should be Euclidean for flat space
>>> print(geometry['errors'])  # Validation metrics

Main Components
---------------
Substrates
    ScalarFieldSubstrate : Free massive scalar field on lattice

Extraction
    extract_geometry : High-level function to run full Π^eff pipeline
    GeometryExtractor : Low-level class for manual control
"""

__version__ = '0.1.0'
__author__ = 'M. A. Turner'

# Core substrate types
from .substrates.scalar_field import ScalarFieldSubstrate

# Main extraction interface
from .extraction.geometry import GeometryExtractor


def extract_geometry(substrate, capacity=None):
    """
    Extract emergent spacetime geometry from quantum substrate.
    
    This is the main high-level interface. Runs complete Π^eff pipeline.
    
    Parameters
    ----------
    substrate : QuantumSubstrate
        The microscopic quantum system
    capacity : float, optional
        Geometric resolution C_geo
        
    Returns
    -------
    geometry : dict
        Dictionary with 'metric', 'distances', 'coordinates', 'errors'
    """
    extractor = GeometryExtractor(substrate, capacity=capacity)
    return extractor.extract_geometry()


__all__ = [
    'ScalarFieldSubstrate',
    'GeometryExtractor',
    'extract_geometry',
    '__version__',
]
