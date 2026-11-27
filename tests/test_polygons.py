from __future__ import annotations

import numpy as np

from pta.core import surrogate_density
from pta.polygons import density_to_polygons, polygon_area, simplify_closed


def test_polygons_are_inside_the_design_box(cfg):
    polygons = density_to_polygons(surrogate_density(cfg), cfg)
    assert len(polygons) > 5
    tolerance = 1e-6
    for points in polygons:
        array = np.asarray(points)
        assert array[:, 0].min() >= -cfg.design_lx / 2 - tolerance
        assert array[:, 0].max() <= cfg.design_lx / 2 + tolerance
        assert array[:, 1].min() >= -cfg.design_ly / 2 - tolerance
        assert array[:, 1].max() <= cfg.design_ly / 2 + tolerance


def test_fill_fraction_is_physical(cfg):
    polygons = density_to_polygons(surrogate_density(cfg), cfg)
    fraction = polygon_area(polygons) / (cfg.design_lx * cfg.design_ly)
    assert 0.2 < fraction < 0.8


def test_deterministic(cfg):
    first = density_to_polygons(surrogate_density(cfg), cfg)
    second = density_to_polygons(surrogate_density(cfg), cfg)
    assert len(first) == len(second)
    np.testing.assert_allclose(np.asarray(first[0]), np.asarray(second[0]))


def test_simplify_keeps_closed_shape():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    kept = simplify_closed(square, 0.01)
    assert len(kept) >= 3
    assert kept[0].tolist() == [0, 0]
    assert kept[-1].tolist() == [0, 0]
