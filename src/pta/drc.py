"""Design-rule checks with no severity allowances.

The previous version of this module had one number that decided everything: a 0.35 µm
"notch allowance" below which a measured defect was declared noise. That number was larger
than the 0.25 µm gap rule and hid real geometry errors. Nothing here has such a knob.
Every drawn primitive is checked on its own geometry by :func:`check_gds`:

* **width** — the narrowest component of each primitive is measured by the small root of
  ``w² - (P/2) w + A = 0``, exact for a strip of any aspect ratio. The measurement is stable:
  it does not sit on the knife edge of a boolean operation the way an erosion by exactly
  ``w/2`` does, which leaves a 4 nm ribbon that fragments at the 1 nm grid.
* **local uniformity** — the same primitive erodes by ``0.45 w`` (leaving an 80 nm ribbon,
  numerically solid) and must stay in one piece, so a point drawn below 0.9 w is reported
  even when the mean width of the primitive looks fine.
* **gap** — dilating by ``g/2`` must not merge two distinct components.
* **cross-reference spacing** — the smallest Euclidean distance between the material of two
  different cell references. A gap rule measures a cell in isolation, so two routes drawn
  in different cells can miss each other by 100 nm and still pass every per-cell check.
  References that butt together (distance zero, shared edge) are only accepted when the
  layout declares a connection between them *at that coordinate* and all of their contact
  material lies within half the joint width of that port. One predicate covers every shape
  a wrong connection can take: an undeclared edge-to-edge contact is geometrically
  identical to a joint, while a declared pair whose contact reaches past the joint cap - a
  joint in the wrong place, a second touch elsewhere, a route that keeps running along the
  block edge - is reported as contact outside the port. A positive-area overlap is
  reported as a short instead.
* **opening** — the freeform core is tested with a disk-shaped opening at ``w/2``. Unlike
  erosion, an opening removes a sub-rule spur even when the rest of the component survives,
  which is exactly the case the survival test cannot see. Convex corners are removed too, so
  each residual is compared with the corner cap its own interior angle can produce: a cap
  may not reach further than ``r / tan(theta/2)`` from its vertex. Material beyond every
  such cap is a real protrusion narrower than ``w``.
* **shorts** — two distinct cell references may not share material. Routes butt into the
  blocks they serve (a shared edge, zero area), so any positive overlap is a crossing,
  which neither a width nor a gap rule can see.
* **connectivity** — a cell that should hold a known number of nets must keep that count.
  A ring 0.1 µm inside its bus leaves one thick band with no gap anywhere: no width or gap
  rule can report it, while the count drops from 2K+1 to 2K.

The freeform core is arbitrary geometry rather than a drawn band, so it uses erosion for the
mean width (``width_method="erosion"``) and the opening test for local features. The
erosion only reports a whole component vanishing or splitting; a spur that disappears from
the edge of a surviving island leaves no trace, which is why the opening test is required
there and not on the routed layer (whose smooth bends would turn every corner into a cap).
The generated surrogate clears the rule: 0.252 um narrowest island, 0.281 um smallest gap
and no opening defect at the 0.140 um rule. The first run of that check also exposed that
the core GDS had never been inspected at all - its polygons sit in the top cell rather than
behind a reference, and the old instance reader returned an empty list for it.

``RESOLUTION`` (4 nm, 0.5 % of the width) is the resolution of the *measurement*: the curve
sampler costs 1.5-2.2 nm of width on a 0.8 µm waveguide and the boolean grid adds 1 nm. It
is reported next to every measurement so the margin is visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gdstk
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import AcceleratorConfig

GRID = 1e-3  # 1 nm: the boolean grid of the polygon engine
# Resolution of the measurement, not a design allowance. PhotonForge samples curves with a
# polygon tolerance that costs 1.5-2.2 nm of width on a 0.8 um waveguide (measured: 0.7985
# on a straight route, 0.7978 on the hybrid's tightest arcs), and the boolean grid adds
# 1 nm. The rule is therefore enforced to 4 nm, i.e. 0.5 % of the width. A 6 nm loss -- the
# default circle sampling before this pass -- still fails, and so does the 0.1 um ring/bus
# short that the previous threshold-based DRC hid.
RESOLUTION = 4 * GRID


def _instances(gds_path: str | Path, layer: tuple[int, int]) -> list[dict[str, Any]]:
    """Per-reference polygons of one layer, in top-cell coordinates."""
    library = gdstk.read_gds(str(gds_path))
    top_level = library.top_level()
    if not top_level:
        raise ValueError(f"{gds_path} contains no top-level cell")
    instances: list[dict[str, Any]] = []
    # A standalone component (the freeform core GDS) is written with its polygons in the top
    # cell, not behind a reference. Without this branch the checker saw zero instances and
    # every rule below passed vacuously on an empty set.
    own = top_level[0].get_polygons(
        layer=layer[0], datatype=layer[1], depth=0
    )
    if own:
        instances.append(
            {"name": top_level[0].name, "polygons": own, "bounds": _bounds(own)}
        )
    for reference in top_level[0].references:
        if reference.rotation or reference.x_reflection or reference.magnification != 1.0:
            raise NotImplementedError("only translated references are supported")
        origin = (float(reference.origin[0]), float(reference.origin[1]))
        polygons = [
            polygon.translate(origin)
            for polygon in reference.cell.get_polygons(
                layer=layer[0], datatype=layer[1], depth=None
            )
        ]
        if not polygons:
            continue
        bounds = _bounds(polygons)
        instances.append({"name": reference.cell.name, "polygons": polygons, "bounds": bounds})
    return instances


def _offset(polygons: list[gdstk.Polygon], distance: float) -> list[gdstk.Polygon]:
    """Morphological offset with a disk-shaped structuring element."""
    if not polygons or abs(distance) < 1e-12:
        return list(polygons)
    return gdstk.offset(
        polygons, distance, join="round", tolerance=1e-4, precision=GRID, use_union=False
    )


def merge(polygons: list[gdstk.Polygon]) -> list[gdstk.Polygon]:
    """Boolean union of a layer: the physical material, with overlaps resolved."""
    if not polygons:
        return []
    return gdstk.boolean(polygons, [], "or", precision=GRID) or []


def _area(polygons: list[gdstk.Polygon] | gdstk.Polygon) -> float:
    if isinstance(polygons, gdstk.Polygon):
        return float(polygons.area())
    return float(sum(polygon.area() for polygon in polygons))


def _bounds(polygons: list[gdstk.Polygon]) -> tuple[float, float, float, float]:
    boxes = [polygon.bounding_box() for polygon in polygons]
    return (
        min(box[0][0] for box in boxes),
        min(box[0][1] for box in boxes),
        max(box[1][0] for box in boxes),
        max(box[1][1] for box in boxes),
    )


# -- raster morphology (used to generate the freeform pattern) --------------


def raster_open(pixels: np.ndarray, radius_um: float, step_um: float) -> np.ndarray:
    """Opening with a Euclidean disk of ``radius_um`` (sub-pixel accurate)."""
    # edt measures centre-to-centre to the nearest empty pixel, half a pixel more than the
    # distance to the boundary; without the correction every straight edge loses one row.
    inside = np.maximum(ndimage.distance_transform_edt(pixels) - 0.5, 0.0) * step_um
    core = inside >= radius_um
    outside = np.maximum(ndimage.distance_transform_edt(~core) - 0.5, 0.0) * step_um
    # The edge pixel of a straight feature sits exactly radius_um from the eroded set; with a
    # strict comparison every edge would be reported as a 1-pixel defect. Clipping restores
    # the defining property of an opening (it is a subset of the original set), which the
    # half-pixel distance correction can otherwise violate at the border.
    return (outside <= radius_um + 1e-9) & pixels


def raster_close(pixels: np.ndarray, radius_um: float, step_um: float) -> np.ndarray:
    """Closing with a Euclidean disk of ``radius_um`` (sub-pixel accurate)."""
    outside = np.maximum(ndimage.distance_transform_edt(~pixels) - 0.5, 0.0) * step_um
    grown = outside < radius_um
    inside = np.maximum(ndimage.distance_transform_edt(grown) - 0.5, 0.0) * step_um
    return (inside >= radius_um - 1e-9) | pixels


# -- routed layer: topology, not area ---------------------------------------


def _erode_report(
    parents: list[gdstk.Polygon], radius: float
) -> tuple[list[gdstk.Polygon], list[dict[str, Any]], list[dict[str, Any]]]:
    """Erode the merged layer, then describe components that vanished or split."""
    # Offset each parent on its own: gdstk unions overlapping offset results even with
    # use_union=False, which would hide the very merge the bridge check is looking for.
    children = merge([piece for parent in parents for piece in _offset([parent], -radius)])
    vanished: list[dict[str, Any]] = []
    pinched: list[dict[str, Any]] = []
    for parent in parents:
        box = parent.bounding_box()
        parent_box = (box[0][0], box[0][1], box[1][0], box[1][1])
        # Assign by material containment, not by bounding box: a ring's bounding box also
        # covers its hole and its neighbours' space.
        members = [
            child
            for child in children
            if gdstk.inside([(float(child.points[0][0]), float(child.points[0][1]))], [parent])[0]
        ]
        if not members:
            vanished.append({"bbox_um": [float(v) for v in parent_box], "area_um2": _area(parent)})
        elif len(members) > 1:
            for first, second in zip(members, members[1:]):
                neck = gdstk.boolean(_offset([first], radius), _offset([second], radius), "and", precision=GRID)
                if neck:
                    pinched.append(
                        {"bbox_um": [float(v) for v in (*neck[0].bounding_box()[0], *neck[0].bounding_box()[1])],
                         "area_um2": _area(neck)}
                    )
    return children, vanished, pinched


def _dilate_report(
    parents: list[gdstk.Polygon], radius: float
) -> tuple[list[gdstk.Polygon], list[dict[str, Any]]]:
    """Dilate the merged layer, then describe components that merged."""
    grown = [piece for parent in parents for piece in _offset([parent], radius)]
    merged = merge(grown)
    bridges: list[dict[str, Any]] = []
    if len(merged) >= len(parents):
        return merged, bridges
    for first_index, first in enumerate(grown):
        for second in grown[first_index + 1 :]:
            shared = gdstk.boolean([first], [second], "and", precision=GRID)
            if shared and _area(shared) > 4 * GRID * GRID:
                bridges.append(
                    {"bbox_um": [float(v) for v in (*shared[0].bounding_box()[0], *shared[0].bounding_box()[1])],
                     "area_um2": _area(shared)}
                )
    return merged, bridges


def band_width(polygons: list[gdstk.Polygon]) -> float:
    """Width of a drawn band as ``2 * area / perimeter``.

    Exact for a rectangle, and accurate to O(w/L) for a curved strip: the area is ``w L`` and
    the perimeter is ``2 L`` plus the two end caps, so the estimate is the *mean* width of the
    piece. This is the right measurement for the routed layer, which is drawn from 800 nm
    paths: a survival predicate at exactly ``w/2`` leaves a 4 nm ribbon, which fragments at
    the 1 nm boolean grid and reports defects that are purely numerical.
    """
    # Per component, solve the rectangle relation exactly: with s = P/2 = L + w and A = L w,
    # the width is the small root of w^2 - s w + A = 0. The naive 2A/P is the large-L limit
    # of this and reads low for short pieces such as the hybrid's 140 um arms. The estimate
    # is per component because the relation does not compose over several pieces; the
    # narrowest component is what the width rule cares about.
    widths: list[float] = []
    for polygon in polygons:
        half = 0.5 * polygon.perimeter()
        discriminant = half * half - 4.0 * polygon.area()
        widths.append(0.5 * (half - float(np.sqrt(max(discriminant, 0.0)))))
    return min(widths) if widths else 0.0


def _measure_width(parents: list[gdstk.Polygon], ceiling: float) -> float:
    """Largest erosion radius that leaves every component present and in one piece."""
    low, high = 0.0, ceiling
    for _ in range(13):
        middle = 0.5 * (low + high)
        _, vanished, pinched = _erode_report(parents, middle)
        if vanished or pinched:
            high = middle
        else:
            low = middle
    return 2.0 * low


def _measure_gap(parents: list[gdstk.Polygon], ceiling: float) -> float:
    """Smallest half-gap between distinct components.

    Bisects on the component count, not on the reported bridge polygons: two components
    exactly ``2 * radius`` apart touch after dilation and merge without producing a
    positive-area overlap.
    """
    low, high = 0.0, ceiling
    for _ in range(13):
        middle = 0.5 * (low + high)
        grown = merge([piece for parent in parents for piece in _offset([parent], middle)])
        if len(grown) < len(parents):
            high = middle
        else:
            low = middle
    return 2.0 * low


def _boundary_segments(
    polygons: list[gdstk.Polygon],
) -> tuple[np.ndarray, np.ndarray]:
    """Edge list of a polygon set as ``(starts, ends)`` arrays."""
    starts: list[np.ndarray] = []
    for polygon in polygons:
        points = np.asarray(polygon.points, dtype=float)
        if len(points) < 3:
            continue
        starts.append(points)
    if not starts:
        return np.empty((0, 2)), np.empty((0, 2))
    points = np.concatenate(starts)
    return points, _shifted(starts)


def _shifted(blocks: list[np.ndarray]) -> np.ndarray:
    """Successor of every vertex inside its own polygon, concatenated."""
    return np.concatenate([np.roll(block, -1, axis=0) for block in blocks])


def _point_segment_distance(
    points: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to each segment and the closest point on the segment."""
    edges = ends - starts
    offsets = points - starts
    length2 = (edges * edges).sum(-1)
    t = np.clip(
        (offsets * edges).sum(-1) / np.where(length2 > 0.0, length2, 1.0), 0.0, 1.0
    )
    closest = starts + t[..., None] * edges
    delta = points - closest
    return np.sqrt((delta * delta).sum(-1)), closest


def polygon_distance(
    first: list[gdstk.Polygon], second: list[gdstk.Polygon]
) -> tuple[float, list[float] | None]:
    """Smallest distance between the boundaries of two polygon sets, and its midpoint.

    The closest pair of points between two disjoint polygons is always an endpoint of one
    of two edges and its projection on the other edge, so four point-to-segment sweeps per
    edge pair are exact. Crossing boundaries are excluded by the caller (a positive-area
    overlap is a short); a shared edge returns zero and is treated as a joint.
    """
    first_start, first_end = _boundary_segments(first)
    second_start, second_end = _boundary_segments(second)
    if not len(first_start) or not len(second_start):
        return float("inf"), None

    best = float("inf")
    location: list[float] | None = None
    chunk = 64
    for offset in range(0, len(first_start), chunk):
        a0 = first_start[offset : offset + chunk, None, :]
        a1 = first_end[offset : offset + chunk, None, :]
        b0 = second_start[None, :, :]
        b1 = second_end[None, :, :]
        to_second_start, closest_second = _point_segment_distance(a0, b0, b1)
        to_second_end, closest_second_end = _point_segment_distance(a1, b0, b1)
        from_second_start, closest_first = _point_segment_distance(b0, a0, a1)
        from_second_end, closest_first_end = _point_segment_distance(b1, a0, a1)
        distances = np.stack(
            [to_second_start, to_second_end, from_second_start, from_second_end]
        )
        index = np.unravel_index(int(np.argmin(distances)), distances.shape)
        distance = float(distances[index])
        if distance >= best:
            continue
        layer, row, column = index
        if layer == 0:
            point, partner = first_start[offset + row], closest_second[row, column]
        elif layer == 1:
            point, partner = first_end[offset + row], closest_second_end[row, column]
        elif layer == 2:
            point, partner = closest_first[row, column], second_start[column]
        else:
            point, partner = closest_first_end[row, column], second_end[column]
        best = distance
        location = [float(0.5 * (point[0] + partner[0])), float(0.5 * (point[1] + partner[1]))]
    return best, location


def _boxes_overlap(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> bool:
    return not (
        first[2] < second[0] or second[2] < first[0] or first[3] < second[1] or second[3] < first[1]
    )


def _contact_regions(
    first: list[gdstk.Polygon], second: list[gdstk.Polygon], radius: float
) -> list[gdstk.Polygon]:
    """Connected regions where two references touch to within ``radius``.

    A butt joint has no overlap area, so the material intersection is empty. Growing both
    sides by the measurement resolution and intersecting instead turns every contact into a
    thin band, one connected component per distinct touch. That is what separates the
    declared joint at a port from a second, accidental touch elsewhere on the same pair.
    """
    grown_first = merge(_offset(first, radius))
    grown_second = merge(_offset(second, radius))
    return gdstk.boolean(grown_first, grown_second, "and", precision=GRID) or []


def _contact_extent(
    regions: list[gdstk.Polygon], points: list[tuple[float, float]]
) -> float:
    """Farthest contact point from the nearest declared port.

    A butt joint's band is the dilated end cap, so it reaches ``w / 2 + resolution`` from
    the port. A declared joint in the wrong place, a second touch, and a route that keeps
    running along the block edge all reach further than that; they differ in shape, not in
    whether they stay inside the cap.
    """
    if not regions:
        return float("inf")
    return min(
        max(
            float(np.hypot(x - point[0], y - point[1]))
            for region in regions
            for x, y in region.points
        )
        for point in points
    )


def check_spacing(
    instances: list[dict[str, Any]],
    min_gap: float,
    declared: dict[tuple[str, str], list[tuple[float, float]]] | None = None,
    joint_width: float = 0.8,
    resolution: float = RESOLUTION,
) -> dict[str, Any]:
    """Distance between the material of different cell references.

    The per-cell gap rule cannot see a near miss between two cells: a route belongs to its
    own cell reference and each reference is checked alone. This rule measures the actual
    Euclidean clearance between every pair of references whose bounding boxes are within
    the gap rule. Two references that touch (``d <= resolution``) form a butt joint, the
    normal way a route entering a block is drawn - but only the layout knows whether that
    contact was intended, and where. A joint is accepted only when ``declared`` maps the
    pair to a port coordinate and every point of the contact lies within
    ``joint_width / 2`` of it. That one predicate covers every shape a wrong connection can
    take: an undeclared pair is an undeclared contact, and a declared pair whose contact
    reaches past the joint cap - a joint in the wrong place, a second touch, or a route
    that keeps running along the block edge - is contact outside the port. A positive-area
    overlap is left to :func:`check_shorts`.
    """
    declared_points: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for (first, second), points in (declared or {}).items():
        key = tuple(sorted((str(first), str(second))))
        declared_points.setdefault(key, []).extend(
            (float(point[0]), float(point[1])) for point in points
        )
    failures: list[dict[str, Any]] = []
    measured = float("inf")
    abutments = 0
    undeclared = 0
    outside = 0
    extent_max = 0.0
    joint_limit = 0.5 * joint_width + 3.0 * resolution
    pairs = 0
    for index, first in enumerate(instances):
        first_bounds = first["bounds"]
        for second in instances[index + 1 :]:
            second_bounds = second["bounds"]
            if (
                first_bounds[0] - second_bounds[2] > min_gap
                or second_bounds[0] - first_bounds[2] > min_gap
                or first_bounds[1] - second_bounds[3] > min_gap
                or second_bounds[1] - first_bounds[3] > min_gap
            ):
                continue
            pairs += 1
            # A positive-area overlap has no meaningful spacing: the two nets are joined.
            if _boxes_overlap(first_bounds, second_bounds):
                shared = gdstk.boolean(
                    first["merged"], second["merged"], "and", precision=GRID
                )
                if shared and _area(shared) > 4 * GRID * GRID:
                    continue
            distance, location = polygon_distance(first["merged"], second["merged"])
            if distance <= resolution:
                pair = tuple(sorted((first["name"], second["name"])))
                points = declared_points.get(pair)
                if points is None:
                    undeclared += 1
                    failures.append(
                        {
                            "kind": "undeclared_contact",
                            "a": first["name"],
                            "b": second["name"],
                            "gap_um": 0.0,
                            "location_um": location,
                        }
                    )
                else:
                    # One predicate for every wrong shape: all the contact material has to
                    # sit within the joint cap at the declared port. A joint somewhere else,
                    # a second touch, and a route that keeps running along the edge all
                    # leave contact beyond that cap.
                    regions = _contact_regions(
                        first["merged"], second["merged"], resolution
                    )
                    extent = _contact_extent(regions, points)
                    if extent > joint_limit:
                        outside += 1
                        failures.append(
                            {
                                "kind": "contact_outside_port",
                                "a": first["name"],
                                "b": second["name"],
                                "gap_um": 0.0,
                                "location_um": location,
                                "declared_um": [[float(x), float(y)] for x, y in points],
                                "extent_um": extent if np.isfinite(extent) else None,
                                "limit_um": joint_limit,
                                "contacts": len(regions),
                            }
                        )
                    else:
                        abutments += 1
                        extent_max = max(extent_max, extent)
                continue
            measured = min(measured, distance)
            if distance < min_gap - resolution:
                failures.append(
                    {
                        "kind": "clearance",
                        "a": first["name"],
                        "b": second["name"],
                        "gap_um": distance,
                        "location_um": location,
                    }
                )
    return {
        "rule_um": min_gap,
        "pairs_checked": pairs,
        "abutments": abutments,
        "undeclared_joints": undeclared,
        "outside_port": outside,
        "joint_extent_max_um": extent_max,
        "joint_limit_um": joint_limit,
        "measured_min_um": None if measured == float("inf") else measured,
        "defects": sorted(failures, key=lambda item: item["gap_um"]),
        "count": len(failures),
    }


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1])
    )


def corner_caps(
    polygons: list[gdstk.Polygon], radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convex vertices of a polygon set and how far their opening cap can reach.

    For a wedge with half-angle ``alpha``, the opening's residual is bounded by the two
    tangent points at ``radius / tan(alpha)`` from the apex: no disk of the structuring
    element covers the corner before that distance. Reflex corners lose no material, so
    their reach is zero. Processing the residual against these caps keeps the rule from
    reporting the rounding that any sharp corner of a rectilinear layout produces.
    """
    vertices: list[np.ndarray] = []
    reaches: list[np.ndarray] = []
    for polygon in polygons:
        points = np.asarray(polygon.points, dtype=float)
        if len(points) < 3:
            continue
        if _signed_area(points) < 0.0:
            points = points[::-1]
        previous = points - np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0) - points
        cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
        dot = (previous * following).sum(1)
        angle = np.arctan2(cross, dot)
        convex = angle > 1e-12
        reach = np.zeros_like(angle)
        reach[convex] = radius / np.tan(0.5 * angle[convex])
        vertices.append(points)
        reaches.append(reach)
    if not vertices:
        return np.empty((0, 2)), np.empty(0)
    return np.concatenate(vertices), np.concatenate(reaches)


def _sample_boundary(polygon: gdstk.Polygon, step: float) -> np.ndarray:
    """Points spaced at most ``step`` apart along the closed boundary."""
    points = np.asarray(polygon.points, dtype=float)
    samples: list[np.ndarray] = []
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        length = float(np.hypot(*(end - start)))
        count = max(1, int(np.ceil(length / step)))
        fraction = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
        samples.append(start + fraction * (end - start))
    return np.concatenate(samples) if samples else points


def opening_report(
    polygons: list[gdstk.Polygon], radius: float, resolution: float = RESOLUTION
) -> dict[str, Any]:
    """Opening test: material that no inscribed disk of ``radius`` can cover.

    The residual of ``opening(M, r)`` is the set of points that no disk of radius ``r``
    inside the material contains. That is the morphological definition of a feature
    narrower than ``2 r``, and unlike the erosion survival test it does not need the
    component to vanish or split. Convex corners fail the same covering test by geometry,
    so each residual is first matched against the cap reach of the corners it touches.
    """
    material = merge(polygons)
    opened = merge(_offset(_offset(material, -radius), radius))
    residual = gdstk.boolean(material, opened, "not", precision=GRID) or []
    if not residual:
        return {
            "radius_um": radius,
            "residual_area_um2": 0.0,
            "residual_components": 0,
            "defects": [],
            "count": 0,
        }

    vertices, reach = corner_caps(material, radius)
    tree = cKDTree(vertices)
    neighbours = min(16, len(vertices))
    # Sample the residual rather than its vertices: a long thin sliver has vertices only at
    # its ends, and those ends are usually the concave junctions where it meets the body.
    step = 4.0 * resolution
    defects: list[dict[str, Any]] = []
    for component in residual:
        samples = _sample_boundary(component, step)
        distance, index = tree.query(samples, k=neighbours)
        if index.ndim == 1:
            index = index[:, None]
            distance = distance[:, None]
        beyond = float(np.min(distance - reach[index], axis=1).max())
        if beyond > resolution:
            box = component.bounding_box()
            defects.append(
                {
                    "beyond_corner_um": beyond,
                    "area_um2": _area(component),
                    "bbox_um": [float(v) for v in (*box[0], *box[1])],
                }
            )
    return {
        "radius_um": radius,
        "residual_area_um2": _area(residual),
        "residual_components": len(residual),
        "defects": sorted(defects, key=lambda item: -item["beyond_corner_um"]),
        "count": len(defects),
    }


def check_shorts(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Positive-area overlap between two distinct cell references."""
    shorts: list[dict[str, Any]] = []
    for index, first in enumerate(instances):
        for second in instances[index + 1 :]:
            a, b = first["bounds"], second["bounds"]
            if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
                continue
            shared = gdstk.boolean(first["polygons"], second["polygons"], "and", precision=GRID)
            if not shared:
                continue
            area = _area(shared)
            if area <= 4 * GRID * GRID:
                continue
            shorts.append(
                {
                    "a": first["name"],
                    "b": second["name"],
                    "area_um2": float(area),
                    "bbox_um": [float(v) for v in (*shared[0].bounding_box()[0], *shared[0].bounding_box()[1])],
                }
            )
    return sorted(shorts, key=lambda item: -item["area_um2"])


def check_gds(
    gds_path: str | Path,
    cfg: AcceleratorConfig,
    min_width: float | None = None,
    min_gap: float | None = None,
    rules: str = "waveguide",
    exclude_freeform: bool = False,
    freeform_cell: str = "FREEFORM_CORE",
    width_method: str = "band",
    expected_components: dict[str, int] | None = None,
    declared_joints: dict[tuple[str, str], list[tuple[float, float]]] | None = None,
    joint_width: float | None = None,
) -> dict[str, Any]:
    """Width, gap, spacing, opening, short and connectivity check of a GDS layer.

    The width and gap rules run per cell reference, not on the merged chip. Merging the whole
    layer fuses every butt joint into one polygon and turns sub-nanometre port offsets into
    apparent pinches, which is how a morphological DRC starts needing a tolerance.     Each drawn primitive is checked on its own geometry; the space *between* references is covered by
    the spacing rule, and unintended contact by the short check. Spacing covers every
    reference, including the freeform cell: ``exclude_freeform`` keeps the core out of the
    width, gap, opening and connectivity rules, whose 800 nm limits do not apply inside it,
    but its port joints are checked like any other. A butt joint between two
    references is only accepted if ``declared_joints`` lists that pair and all of the
    contact lies within half the joint width of the declared port - so an unrelated
    edge-to-edge contact, a route touching the wrong part of its block, a second touch, and
    a route that keeps running along the block edge are all reported. The opening test runs
    when ``width_method`` is not ``"band"``, i.e. on the freeform core.
    """
    min_width = cfg.drc_min_width if min_width is None else min_width
    min_gap = cfg.drc_min_gap if min_gap is None else min_gap
    layer = cfg.layer("sin_core")
    width_radius = 0.5 * min_width - RESOLUTION
    gap_radius = 0.5 * min_gap - RESOLUTION

    instances = _instances(gds_path, layer)
    for instance in instances:
        instance["merged"] = merge(instance["polygons"])
    checked = [
        instance
        for instance in instances
        if not (exclude_freeform and instance["name"] == freeform_cell)
    ]
    width_failures: list[dict[str, Any]] = []
    gap_failures: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    measured_width = float("inf")
    measured_gap = float("inf")
    for instance in checked:
        merged = instance["merged"]
        counts[instance["name"]] = len(merged)
        if width_method == "band":
            drawn = band_width(merged)
            if drawn < min_width - RESOLUTION:
                box = instance["bounds"]
                width_failures.append(
                    {
                        "component": instance["name"],
                        "measured_um": drawn,
                        "bbox_um": [float(v) for v in box],
                    }
                )
            # Local uniformity. The mean width above cannot see a fused junction, and a
            # survival test at w/2 cannot see anything through the 4 nm ribbon it leaves.
            # Eroding at 0.9 w / 2 keeps an 80 nm ribbon (stable at the 1 nm grid) and still
            # fails any point drawn below 0.9 w.
            _, flat_vanished, flat_pinched = _erode_report(merged, 0.45 * min_width - RESOLUTION)
            for defect in flat_vanished + flat_pinched:
                width_failures.append({"component": instance["name"], "rule": "local", **defect})
        else:
            _, vanished, pinched = _erode_report(merged, width_radius)
            for defect in vanished + pinched:
                width_failures.append({"component": instance["name"], **defect})
        _, bridges = _dilate_report(merged, gap_radius)
        for defect in bridges:
            gap_failures.append({"component": instance["name"], **defect})
        measured_width = min(
            measured_width,
            band_width(merged) if width_method == "band" else _measure_width(merged, 0.9 * min_width),
        )
        measured_gap = min(measured_gap, _measure_gap(merged, 4.0 * min_gap))

    # Connectivity. No width or gap rule can see two nets fused into one piece of material:
    # the union reads as a single fat band and there is no gap left to measure. A cell that
    # is supposed to hold a known set of nets must keep its component count, so a short that
    # joins two neighbouring structures is caught by the count dropping.
    connectivity_failures: list[dict[str, Any]] = []
    if expected_components:
        for cell_name, expected in sorted(expected_components.items()):
            drawn = counts.get(cell_name)
            if drawn != expected:
                connectivity_failures.append(
                    {"cell": cell_name, "components": drawn, "expected": expected}
                )

    # Spacing covers every reference, including the freeform cell. The routes butt the core
    # at its eight declared ports, and those joints need the same declaration check as the
    # routed ones; only the width, gap, opening and connectivity rules are restricted to
    # ``checked``, because the 800 nm routing rules do not apply inside the core.
    spacing = check_spacing(
        instances, min_gap, declared_joints,
        joint_width=cfg.wg_width if joint_width is None else joint_width,
    )

    opening = None
    if width_method != "band":
        reports = [opening_report(instance["merged"], 0.5 * min_width) for instance in checked]
        defects = [
            {"component": instance["name"], **defect}
            for instance, report in zip(checked, reports)
            for defect in report["defects"]
        ]
        opening = {
            "radius_um": 0.5 * min_width,
            "residual_area_um2": sum(report["residual_area_um2"] for report in reports),
            "residual_components": sum(report["residual_components"] for report in reports),
            "defects": sorted(defects, key=lambda item: -item["beyond_corner_um"])[:100],
            "count": len(defects),
        }

    shorts = check_shorts(instances)
    return {
        "file": str(gds_path),
        "rules": rules,
        "grid_um": GRID,
        "resolution_um": RESOLUTION,
        "min_width_rule_um": min_width,
        "min_gap_rule_um": min_gap,
        "components_checked": len(checked),
        "width_method": "band (2A/P)" if width_method == "band" else "erosion",
        "passes": not width_failures
        and not gap_failures
        and not spacing["defects"]
        and not (opening and opening["count"])
        and not shorts
        and not connectivity_failures,
        "width": {
            "measured_min_um": None if measured_width == float("inf") else measured_width,
            "defects": width_failures,
            "count": len(width_failures),
        },
        "gap": {
            "measured_min_um": None if measured_gap == float("inf") else measured_gap,
            "defects": gap_failures,
            "count": len(gap_failures),
        },
        "spacing": spacing,
        "opening": opening,
        "shorts": shorts,
        "connectivity": {
            "checked": len(expected_components or {}),
            "components": counts,
            "failures": connectivity_failures,
        },
    }


def cell_names(gds_path: str | Path, layer: tuple[int, int]) -> list[str]:
    """Cell names referenced by the top cell, for building a connectivity expectation."""
    return sorted({instance["name"] for instance in _instances(gds_path, layer)})


def summarize(report: dict[str, Any]) -> str:
    """One-line summary for the CLI."""
    width = report["width"]
    gap = report["gap"]
    spacing = report.get("spacing") or {}
    opening = report.get("opening")
    verdict = "PASS" if report["passes"] else "FAIL"

    def number(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    parts = [
        f"{verdict} {report['rules']:<9}",
        f"width {number(width.get('measured_min_um'))}",
        f"(rule {report['min_width_rule_um']:.3f})",
        f"gap {number(gap.get('measured_min_um'))}",
        f"(rule {report['min_gap_rule_um']:.3f})",
        f"spacing {number(spacing.get('measured_min_um'))}",
        f"joints {spacing.get('abutments', 0)}",
    ]
    if spacing.get("undeclared_joints"):
        parts.append(f"undeclared {spacing['undeclared_joints']}")
    if spacing.get("outside_port"):
        parts.append(f"outside port {spacing['outside_port']}")
    if opening is not None:
        parts.append(f"opening {opening['count']}")
    parts.append(f"shorts {len(report['shorts'])}")
    return " ".join(parts)
