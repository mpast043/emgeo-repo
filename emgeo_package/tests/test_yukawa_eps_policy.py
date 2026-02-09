import numpy as np
import pytest
from emgeo.substrates.scalar_field import ScalarFieldSubstrate
from emgeo.extraction.geometry import GeometryExtractor

def test_yukawa_eps_too_large_is_ignored_and_warns():
    s = ScalarFieldSubstrate(lattice_size=16, subsample_stride=2, mass=1.0, dtype=np.float64)
    ex = GeometryExtractor(s, embed_dim=3)
    with pytest.warns(UserWarning, match=r"yukawa_inversion: eps="):
        geom = ex.extract_geometry(corr_method="yukawa_inversion", eps=1e-6)
    assert geom["errors"]["distance_extraction"] < 1e-10
    assert geom["errors"]["mds_stress"] < 1e-8

def test_yukawa_eps_none_auto_no_warn():
    s = ScalarFieldSubstrate(lattice_size=16, subsample_stride=2, mass=1.0, dtype=np.float64)
    ex = GeometryExtractor(s, embed_dim=3)
    geom = ex.extract_geometry(corr_method="yukawa_inversion", eps=None)
    assert geom["errors"]["distance_extraction"] < 1e-10
    assert geom["errors"]["mds_stress"] < 1e-8
