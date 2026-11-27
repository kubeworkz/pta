from __future__ import annotations

import gdstk
import numpy as np
import pytest

from pta import drc
from pta.drc import band_width, check_gds, merge, raster_open


def _write_gds(tmp_path, cells, placements, name="TOP"):
    """Write a GDS with one top cell referencing the given cells."""
    library = gdstk.Library()
    created = {}
    for cell_name, polygons in cells.items():
        cell = library.new_cell(cell_name)
        for polygon in polygons:
            # the checker reads the SiN core layer
            cell.add(gdstk.Polygon(np.asarray(polygon.points), layer=1, datatype=0))
        created[cell_name] = cell
    top = library.new_cell(name)
    for cell_name, origin in placements:
        top.add(gdstk.Reference(created[cell_name], origin=origin))
    path = tmp_path / "drc.gds"
    library.write_gds(str(path))
    return path


def _bar(width, length=200.0, y=0.0, x=0.0):
    return gdstk.rectangle((x, y - width / 2), (x + length, y + width / 2))


def _ring(center, radius=12.0, width=0.8):
    outer = gdstk.ellipse(center, radius + width / 2, tolerance=1e-4)
    inner = gdstk.ellipse(center, radius - width / 2, tolerance=1e-4)
    return gdstk.boolean([outer], [inner], "not", precision=1e-4)[0]


def test_band_width_of_a_strip():
    assert band_width([_bar(0.8)]) == pytest.approx(0.8, abs=1e-9)
    assert band_width([_bar(0.5)]) == pytest.approx(0.5, abs=1e-9)


def test_band_width_is_aspect_ratio_independent():
    for length in (20.0, 200.0, 5000.0):
        assert band_width([_bar(0.8, length)]) == pytest.approx(0.8, abs=1e-9)


def test_legal_pair_passes(tmp_path, cfg):
    path = _write_gds(tmp_path, {"A": [_bar(0.8), _bar(0.8, y=1.5)]}, [("A", (0.0, 0.0))])
    report = check_gds(path, cfg, min_width=0.8, min_gap=0.25)
    assert report["passes"] is True
    assert report["width"]["measured_min_um"] == pytest.approx(0.8, abs=1e-6)
    assert report["gap"]["measured_min_um"] == pytest.approx(0.7, abs=2e-3)


def test_narrow_bar_is_flagged(tmp_path, cfg):
    path = _write_gds(tmp_path, {"A": [_bar(0.5)]}, [("A", (0.0, 0.0))])
    report = check_gds(path, cfg, min_width=0.8, min_gap=0.25)
    assert report["passes"] is False
    assert report["width"]["count"] >= 1
    assert report["width"]["measured_min_um"] < 0.8


def test_close_pair_is_flagged(tmp_path, cfg):
    path = _write_gds(tmp_path, {"A": [_bar(0.8), _bar(0.8, y=1.0)]}, [("A", (0.0, 0.0))])  # 0.2 um gap
    report = check_gds(path, cfg, min_width=0.8, min_gap=0.25)
    assert report["passes"] is False
    assert report["gap"]["count"] >= 1


def test_crossing_references_are_flagged(tmp_path, cfg):
    vertical = [gdstk.rectangle((-0.4, -50), (0.4, 50))]
    path = _write_gds(
        tmp_path,
        {"HORIZONTAL": [_bar(0.8)], "VERTICAL": vertical},
        [("HORIZONTAL", (0.0, 0.0)), ("VERTICAL", (100.0, 0.0))],
    )
    report = check_gds(path, cfg, min_width=0.8, min_gap=0.25)
    assert report["passes"] is False
    assert report["shorts"], "a crossing between two references must be reported"


def test_fused_ring_bus_is_caught_by_the_component_count(tmp_path, cfg):
    """A ring fused to its bus is invisible to width and gap rules; connectivity sees it.

    The width rule reads the fused pair as one thick band and the gap rule sees no gap, so
    the short that a missing half-width term once produced has to be caught by counting the
    separate pieces. A mux cell must hold one bus, K rings and K drop waveguides.
    """
    from pta import blocks, pdk

    pdk.install(cfg)
    mux = blocks.wdm_mux(cfg, "WDM_MUX_TEST", cfg.freqs_hz)
    path = tmp_path / "mux.gds"
    mux.write_gds(str(path))
    top = gdstk.read_gds(str(path)).top_level()[0]
    polygons = top.get_polygons(layer=1, datatype=0, depth=None)
    assert len(merge(polygons)) == 2 * cfg.K + 1


def test_healthy_ring_bus_passes(tmp_path, cfg):
    bus = _bar(0.8, length=100.0, y=0.0, x=0.0)
    ring = _ring((50.0, -13.1))
    path = _write_gds(tmp_path, {"MUX": [bus, ring]}, [("MUX", (0.0, 0.0))])
    report = check_gds(path, cfg, min_width=0.8, min_gap=0.25)
    assert report["passes"] is True
    assert report["gap"]["measured_min_um"] == pytest.approx(0.3, abs=3e-3)


def test_raster_open_keeps_a_straight_edge():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 5:35] = True
    opened = raster_open(mask, 0.07, 0.02)
    # A straight edge survives exactly; only the four convex corners are rounded, which is a
    # property of an opening, not a defect.
    assert opened[20, 5:35].all()
    assert opened.sum() >= mask.sum() - 20


def test_merge_resolves_overlap():
    merged = merge([_bar(0.8, length=100.0), _bar(0.8, length=100.0, y=0.4)])
    assert len(merged) == 1
    assert band_width(merged) > 0.8


def test_cross_reference_near_miss_is_flagged(tmp_path, cfg):
    """Two legal cells can still miss each other; only a between-reference rule sees it.

    Each bar is a clean 0.8 um waveguide with no neighbour inside its own cell, so the
    width and gap rules report nothing. The 0.2 um clearance exists only across the two
    references, which is where the previous version of the checker stopped looking.
    """
    path = _write_gds(
        tmp_path,
        {"LEFT": [_bar(0.8, length=100.0)], "RIGHT": [_bar(0.8, length=100.0, x=100.2)]},
        [("LEFT", (0.0, 0.0)), ("RIGHT", (0.0, 0.0))],
    )
    report = check_gds(path, cfg, min_width=0.8, min_gap=0.25)
    assert report["gap"]["count"] == 0
    assert report["spacing"]["count"] == 1
    assert report["spacing"]["defects"][0]["gap_um"] == pytest.approx(0.2, abs=2e-3)
    assert report["passes"] is False


def test_declared_abutment_is_a_joint(tmp_path, cfg):
    """A route butts into the block it serves; the layout declares that connection."""
    path = _write_gds(
        tmp_path,
        {"A": [_bar(0.8, length=100.0)], "B": [_bar(0.8, length=100.0, x=100.0)]},
        [("A", (0.0, 0.0)), ("B", (0.0, 0.0))],
    )
    report = check_gds(
        path, cfg, min_width=0.8, min_gap=0.25, declared_joints={("A", "B"): [(100.0, 0.0)]}
    )
    assert report["passes"] is True
    assert report["spacing"]["abutments"] == 1
    assert report["spacing"]["undeclared_joints"] == 0
    assert report["spacing"]["outside_port"] == 0
    assert report["spacing"]["measured_min_um"] is None
    assert report["shorts"] == []


def test_declared_joint_must_touch_at_the_declared_port(tmp_path, cfg):
    """Pairing the right names is not enough: the joint has to be at the declared port.

    These two references do touch at x = 100, but the declared port is (150, 0), which is
    inside the second bar only. Naming the pair used to be sufficient; now all of the
    contact has to sit at the declared coordinate, so a route that brushes its block
    somewhere other than its port is reported instead of counted as a connection.
    """
    path = _write_gds(
        tmp_path,
        {"A": [_bar(0.8, length=100.0)], "B": [_bar(0.8, length=100.0, x=100.0)]},
        [("A", (0.0, 0.0)), ("B", (0.0, 0.0))],
    )
    report = check_gds(
        path, cfg, min_width=0.8, min_gap=0.25,
        declared_joints={("A", "B"): [(150.0, 0.0)]},
    )
    assert report["shorts"] == []
    assert report["spacing"]["abutments"] == 0
    assert report["spacing"]["outside_port"] == 1
    assert report["spacing"]["defects"][0]["kind"] == "contact_outside_port"
    assert report["spacing"]["defects"][0]["declared_um"] == [[150.0, 0.0]]
    assert report["passes"] is False


def test_extra_contact_away_from_the_declared_port_is_reported(tmp_path, cfg):
    """Landing on the declared port does not license a second touch somewhere else.

    Cell ROUTE holds the bar butted to BLOCK at the declared port (100, 0), plus a second
    bar whose left edge lies on BLOCK's right edge between y = 2 and 10. Both contacts have
    zero area, so neither the short rule nor the gap rule can see the second one; the
    containment test can, because the second band reaches far beyond the port cap.
    """
    block = gdstk.rectangle((0.0, -10.0), (100.0, 10.0))
    route = _bar(0.8, length=100.0, x=100.0)
    spur = gdstk.rectangle((100.0, 2.0), (100.8, 20.0))
    path = _write_gds(
        tmp_path,
        {"BLOCK": [block], "ROUTE": [route, spur]},
        [("BLOCK", (0.0, 0.0)), ("ROUTE", (0.0, 0.0))],
    )
    report = check_gds(
        path, cfg, min_width=0.8, min_gap=0.25,
        declared_joints={("BLOCK", "ROUTE"): [(100.0, 0.0)]},
    )
    assert report["shorts"] == []
    assert report["spacing"]["abutments"] == 0
    assert report["spacing"]["outside_port"] == 1
    defect = report["spacing"]["defects"][0]
    assert defect["kind"] == "contact_outside_port"
    assert defect["contacts"] == 2
    assert defect["extent_um"] > 5.0
    assert report["passes"] is False


def test_contact_band_running_along_the_block_edge_is_reported(tmp_path, cfg):
    """Landing on the port is not enough if the contact keeps going along the edge.

    ROUTE is one L-shaped piece: a bar butted at the declared port (100, 0), plus a leg
    whose left edge also lies on BLOCK's right edge up to y = 5. The touches are
    contiguous, so this is a single contact band that contains the port; only its reach
    past the port cap shows that it is not a waveguide-width joint.
    """
    block = gdstk.rectangle((0.0, -10.0), (100.0, 10.0))
    bar = _bar(0.8, length=100.0, x=100.0)
    leg = gdstk.rectangle((100.0, 0.3), (100.8, 5.0))
    route = merge([bar, leg])[0]
    path = _write_gds(
        tmp_path,
        {"BLOCK": [block], "ROUTE": [route]},
        [("BLOCK", (0.0, 0.0)), ("ROUTE", (0.0, 0.0))],
    )
    report = check_gds(
        path, cfg, min_width=0.8, min_gap=0.25,
        declared_joints={("BLOCK", "ROUTE"): [(100.0, 0.0)]},
    )
    assert report["shorts"] == []
    assert report["spacing"]["abutments"] == 0
    assert report["spacing"]["outside_port"] == 1
    defect = report["spacing"]["defects"][0]
    assert defect["kind"] == "contact_outside_port"
    assert defect["extent_um"] == pytest.approx(5.0, abs=0.02)
    assert defect["limit_um"] == pytest.approx(0.4 + 3 * report["resolution_um"], abs=1e-9)
    assert defect["contacts"] == 1
    assert report["passes"] is False


def test_undeclared_abutment_is_a_failure(tmp_path, cfg):
    """Two unrelated nets edge to edge are geometrically identical to a joint.

    A width, gap or short rule cannot tell the difference: the shared edge has zero area
    and no gap. Only the layout's connection list can, so a butt that is not declared there
    has to fail rather than be assumed intended.
    """
    path = _write_gds(
        tmp_path,
        {"LEFT": [_bar(0.8, length=100.0)], "RIGHT": [_bar(0.8, length=100.0, x=100.0)]},
        [("LEFT", (0.0, 0.0)), ("RIGHT", (0.0, 0.0))],
    )
    report = check_gds(path, cfg, min_width=0.8, min_gap=0.25)
    assert report["shorts"] == []
    assert report["gap"]["count"] == 0
    assert report["spacing"]["abutments"] == 0
    assert report["spacing"]["undeclared_joints"] == 1
    assert report["spacing"]["defects"][0]["kind"] == "undeclared_contact"
    assert {report["spacing"]["defects"][0]["a"], report["spacing"]["defects"][0]["b"]} == {
        "LEFT",
        "RIGHT",
    }
    assert report["passes"] is False


def test_freeform_cell_is_joint_checked_but_not_width_checked(tmp_path, cfg):
    """The core is outside the 800 nm routing rules, not outside the joint rule.

    The core stub is 0.4 um wide, which the routing width rule would reject if the core
    went through it. It is deliberately excluded from width and gap; its route at the
    declared port still has to pass the declared-joint check, so the core joints are
    covered instead of being skipped along with the cell.
    """
    path = _write_gds(
        tmp_path,
        {
            "FREEFORM_CORE": [_bar(0.4, length=100.0)],
            "ROUTE": [_bar(0.8, length=100.0, x=100.0)],
        },
        [("FREEFORM_CORE", (0.0, 0.0)), ("ROUTE", (0.0, 0.0))],
    )
    report = check_gds(
        path, cfg, min_width=0.8, min_gap=0.25, exclude_freeform=True,
        declared_joints={("FREEFORM_CORE", "ROUTE"): [(100.0, 0.0)]},
    )
    assert report["width"]["count"] == 0, "the core stays outside the routing width rule"
    assert report["spacing"]["abutments"] == 1
    assert report["spacing"]["undeclared_joints"] == 0
    assert report["spacing"]["outside_port"] == 0
    assert report["passes"] is True


def test_freeform_joint_that_stops_short_is_a_clearance_failure(tmp_path, cfg):
    """Including the core in spacing means a route that misses its stub is reported."""
    path = _write_gds(
        tmp_path,
        {
            "FREEFORM_CORE": [_bar(0.4, length=100.0)],
            "ROUTE": [_bar(0.8, length=100.0, x=100.2)],
        },
        [("FREEFORM_CORE", (0.0, 0.0)), ("ROUTE", (0.0, 0.0))],
    )
    report = check_gds(
        path, cfg, min_width=0.8, min_gap=0.25, exclude_freeform=True,
        declared_joints={("FREEFORM_CORE", "ROUTE"): [(100.0, 0.0)]},
    )
    assert report["spacing"]["count"] == 1
    assert report["spacing"]["defects"][0]["kind"] == "clearance"
    assert report["spacing"]["defects"][0]["gap_um"] == pytest.approx(0.2, abs=2e-3)
    assert {report["spacing"]["defects"][0]["a"], report["spacing"]["defects"][0]["b"]} == {
        "FREEFORM_CORE",
        "ROUTE",
    }
    assert report["passes"] is False


def test_declaration_does_not_excuse_a_clearance_violation(tmp_path, cfg):
    """A declared connection is a licence to touch, not to stop 0.2 um short."""
    path = _write_gds(
        tmp_path,
        {"LEFT": [_bar(0.8, length=100.0)], "RIGHT": [_bar(0.8, length=100.0, x=100.2)]},
        [("LEFT", (0.0, 0.0)), ("RIGHT", (0.0, 0.0))],
    )
    report = check_gds(
        path, cfg, min_width=0.8, min_gap=0.25,
        declared_joints={("LEFT", "RIGHT"): [(100.0, 0.0)]},
    )
    assert report["spacing"]["count"] == 1
    assert report["spacing"]["defects"][0]["kind"] == "clearance"
    assert report["spacing"]["defects"][0]["gap_um"] == pytest.approx(0.2, abs=2e-3)
    assert report["passes"] is False


def test_opening_flags_a_sub_rule_tip_the_erosion_test_misses(tmp_path, cfg):
    """A 40 nm spur does not vanish as a component, so only the opening test sees it."""
    bar = gdstk.rectangle((0.0, -0.4), (2000.0, 0.4))
    spur = gdstk.rectangle((900.0, 0.4), (900.04, 0.8))
    path = _write_gds(tmp_path, {"CORE": [bar, spur]}, [("CORE", (0.0, 0.0))])
    report = check_gds(
        path, cfg, min_width=0.14, min_gap=0.14, width_method="erosion", rules="freeform"
    )
    assert report["width"]["count"] == 0, "the body survives erosion in one piece"
    assert report["opening"]["count"] >= 1
    assert report["opening"]["defects"][0]["beyond_corner_um"] > 0.05
    assert report["passes"] is False


def test_opening_does_not_flag_right_angle_corners(tmp_path, cfg):
    """The opening removes every convex corner; that rounding is not a width defect."""
    path = _write_gds(tmp_path, {"CORE": [_bar(0.8, length=200.0)]}, [("CORE", (0.0, 0.0))])
    report = check_gds(
        path, cfg, min_width=0.14, min_gap=0.14, width_method="erosion", rules="freeform"
    )
    assert report["opening"]["residual_components"] > 0
    assert report["opening"]["count"] == 0
    assert report["passes"] is True


def test_generated_core_clears_the_freeform_rule(tmp_path, cfg):
    """The surrogate must pass the rule it is checked against, not just on average.

    This locks the correlation length that makes the thresholded field manufacturable: a
    0.24 um field produced 85-120 nm gaps between large islands, which the polygon-level
    checker (now that it actually reads the core GDS) reports as real defects.
    """
    from pta import core, polygons

    boundaries = polygons.density_to_polygons(core.surrogate_density(cfg), cfg)
    library = gdstk.Library()
    cell = library.new_cell("CORE")
    for points in boundaries:
        cell.add(gdstk.Polygon(np.asarray(points), layer=1, datatype=0))
    path = tmp_path / "core.gds"
    library.write_gds(str(path))

    report = check_gds(
        path, cfg, min_width=cfg.drc_freeform_um, min_gap=cfg.drc_freeform_um,
        width_method="erosion", rules="freeform",
    )
    limit = cfg.drc_freeform_um - report["resolution_um"]
    assert report["components_checked"] == 1
    assert report["width"]["measured_min_um"] >= limit
    assert report["gap"]["measured_min_um"] >= limit
    assert report["opening"]["count"] == 0
    assert report["passes"] is True
