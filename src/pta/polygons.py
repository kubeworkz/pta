"""Density-to-GDS polygon conversion with foundry DRC filtering.

The inverse-designed density map is turned into manufacturable geometry in four steps:

1. Gaussian smoothing removes sub-pixel staircase edges.
2. Morphological opening then closing removes islands and pinholes below the minimum
   feature size.
3. Contours are extracted at the 0.5 level, with holes handled as a boolean difference so
   the exported polygons carry real voids instead of filled ones.
4. Vertices are simplified to a 5 nm tolerance and clipped to the design box.

The result is a list of exterior boundaries in micrometres, ready for
:class:`photonforge.Polygon`.
"""

from __future__ import annotations

import gdstk
import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter

from .config import AcceleratorConfig

PRECISION = 1e-3  # 1 nm GDS grid


def _disk(radius_px: int) -> np.ndarray:
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (x * x + y * y) <= radius_px * radius_px


def clean_density(density: np.ndarray, cfg: AcceleratorConfig) -> np.ndarray:
    """Smooth and morphologically clean a density map; returns a boolean material mask."""
    smoothed = gaussian_filter(np.asarray(density, dtype=float), sigma=1.2, mode="constant")
    mask = smoothed >= 0.5

    radius_px = max(1, int(np.ceil(0.5 * cfg.drc_min_feature / cfg.dl_opt)))
    disk = _disk(radius_px)
    mask = binary_opening(mask, structure=disk)
    mask = binary_closing(mask, structure=disk)
    return mask


def _contours_to_um(
    mask: np.ndarray,
    cfg: AcceleratorConfig,
    pad_px: int,
) -> list[list[tuple[float, float]]]:
    """Extract contours of ``mask`` and map array indices to chip coordinates (um)."""
    dx = cfg.design_lx / cfg.nx
    dy = cfg.design_ly / cfg.ny
    # gdstk indexes contours as [column, row]; transpose so x runs along the design x axis.
    outlines = gdstk.contour(mask.astype(float).T, level=0.5)
    result = []
    for polygon in outlines:
        points = polygon.points.copy()
        points[:, 0] = (points[:, 0] - pad_px) * dx - cfg.design_lx / 2.0
        points[:, 1] = (points[:, 1] - pad_px) * dy - cfg.design_ly / 2.0
        result.append([(float(x), float(y)) for x, y in points])
    return result


def _to_gdstk(boundaries: list[list[tuple[float, float]]]) -> list[gdstk.Polygon]:
    return [gdstk.Polygon(points) for points in boundaries if len(points) >= 3]


def simplify_closed(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker simplification of a closed polygon, implemented iteratively."""
    points = np.asarray(points, dtype=float)
    if len(points) < 5 or tolerance <= 0.0:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = points[end] - points[start]
        norm = float(np.hypot(*segment))
        if norm < 1e-12:
            distances = np.hypot(*(points[start + 1 : end] - points[start]).T)
        else:
            offsets = points[start + 1 : end] - points[start]
            distances = np.abs(segment[0] * offsets[:, 1] - segment[1] * offsets[:, 0]) / norm
        index = int(np.argmax(distances))
        if float(distances[index]) > tolerance:
            split = start + 1 + index
            keep[split] = True
            stack.extend([(start, split), (split, end)])
    kept = points[keep]
    return kept if len(kept) >= 3 else points


def _area(polygons: list[gdstk.Polygon]) -> float:
    return float(sum(polygon.area() for polygon in polygons))


def density_to_polygons(
    density: np.ndarray,
    cfg: AcceleratorConfig,
    simplify_um: float = 0.005,
) -> list[list[tuple[float, float]]]:
    """Convert a density map into DRC-filtered, hole-aware polygons in micrometres."""
    density = np.asarray(density, dtype=float)
    if density.shape != (cfg.nx, cfg.ny):
        raise ValueError(f"expected density of shape {(cfg.nx, cfg.ny)}, got {density.shape}")

    pad_px = 4
    padded = np.pad(density, pad_width=pad_px, mode="constant", constant_values=0.0)
    mask = clean_density(padded, cfg)

    material = _to_gdstk(_contours_to_um(mask, cfg, pad_px))
    voids = _to_gdstk(_contours_to_um(~mask, cfg, pad_px))
    if not material:
        return []

    merged = gdstk.boolean(material, voids, "not", precision=PRECISION) if voids else material
    box = gdstk.rectangle(
        (-cfg.design_lx / 2.0, -cfg.design_ly / 2.0),
        (cfg.design_lx / 2.0, cfg.design_ly / 2.0),
    )
    clipped = gdstk.boolean(merged, [box], "and", precision=PRECISION)

    result: list[list[tuple[float, float]]] = []
    for polygon in clipped:
        if polygon.area() <= 1e-6:
            continue
        points = simplify_closed(polygon.points, simplify_um)
        if len(points) < 3:
            continue
        result.append([(float(x), float(y)) for x, y in points])
    return result


def polygon_area(boundaries: list[list[tuple[float, float]]]) -> float:
    return _area(_to_gdstk(boundaries))
