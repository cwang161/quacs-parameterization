import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from quacs.biogenic.megan.src.MEGCAN import canopy_layer_quadrature
from quacs.biogenic.megan.biogenic_emission_megan_v3 import MeganSettings


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 9])
def test_gaussian_quadrature(n):
    x, w = canopy_layer_quadrature(n)
    raw_x, raw_w = leggauss(n)
    np.testing.assert_allclose(x, (raw_x + 1) / 2)
    np.testing.assert_allclose(w, raw_w / 2)
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(np.dot(w, x), 0.5)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 9])
def test_uniform_midpoints(n):
    x, w = canopy_layer_quadrature(n, "uniform")
    np.testing.assert_allclose(x, (np.arange(n) + 0.5) / n)
    np.testing.assert_allclose(w, np.ones(n) / n)


def test_invalid_options():
    for n in [0, -1, 1.5, True]:
        with pytest.raises(ValueError):
            canopy_layer_quadrature(n)
    with pytest.raises(ValueError):
        canopy_layer_quadrature(5, "other")


def test_default_setting():
    assert MeganSettings().canopy_layer_method == "gaussian"
