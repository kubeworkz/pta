"""Physical building blocks of the accelerator, built as PhotonForge components.

Every block carries:

* real GDS geometry on the configured foundry layers,
* optical ports on the 800 nm SiN strip specification,
* a :class:`photonforge.DataModel` S-matrix from :mod:`pta.smodels`.

The freeform core is the only block whose S-matrix comes from a full electromagnetic
solve. The passive mux, MZM, hybrid and connectors use analytic models and are labelled as
such in the layout manifest so tapeout readiness is explicit.
"""

from __future__ import annotations

import gdstk
import numpy as np
import photonforge as pf

from . import pdk, smodels
from .config import AcceleratorConfig

# PhotonForge samples a curve adaptively. The default leaves vertices ~5 um apart on a
# 300 um bend, which cuts up to 12 nm off the inner edge of an 800 nm waveguide -- enough to
# trip a width rule that is exact to 1 nm. Ask for a dense sampling so the drawn polygon is
# the prescribed geometry, not a coarse chord of it.
PATH_EVALS = 200

# The drop waveguides leave each ring, run left at their own port height, then climb to the
# drop plane. The first ring sits this far from the cell edge so every climb gets 200 um of
# horizontal run; 35 um (the old value) made the outer drops self-intersect.
MUX_X0 = 380.0
MUX_LEAD = 240.0


def mux_bus_span(cfg: AcceleratorConfig) -> tuple[float, float]:
    """x extent of the through bus inside the mux cell, for the floorplan."""
    pitch = 2.0 * cfg.ring_radius_um + 120.0
    return MUX_X0 - 80.0, MUX_X0 + (cfg.K - 1) * pitch + 60.0


MIN_STUB_JOIN_UM2 = 0.1
STUB_LANDING_STEP_UM = 0.1


def stub_landing_limit(cfg: AcceleratorConfig, core) -> float:
    """Furthest an access stub may extend into the design region.

    The analytic surrogate is a random fill whose islands can sit microns from the
    boundary, so it gets the loose limit. An ingested Tidy3D core is a real optimized
    design: if its material does not reach the ports, the design is wrong and the build
    must fail rather than hide the gap behind a multi-micron lead.
    """
    if core.backend == "analytic":
        return cfg.stub_landing_max_um
    return cfg.stub_landing_em_max_um


def _stub_landing(
    cfg: AcceleratorConfig,
    islands: list[gdstk.Polygon],
    y: float,
    side: float,
    port_name: str,
    max_extension_um: float,
) -> tuple[gdstk.Polygon, float]:
    """Extend an access stub from the design boundary until it joins an island.

    The thresholded surrogate has a void in front of some ports, and the nearest material
    can be a thin corner, so a fixed overlap cannot guarantee a connection. The stub walks
    inward until it shares at least ``MIN_STUB_JOIN_UM2`` with the pattern; a port that
    reaches ``max_extension_um`` without meeting material fails the build instead of
    shipping a floating waveguide.
    """
    half = 0.5 * cfg.wg_width
    boundary = side * 0.5 * cfg.design_lx
    start = boundary + side * cfg.stub_len
    depth = STUB_LANDING_STEP_UM
    while depth <= max_extension_um:
        tip = boundary - side * depth
        stub = gdstk.rectangle((min(start, tip), y - half), (max(start, tip), y + half))
        shared = gdstk.boolean([stub], islands, "and", precision=1e-4) or []
        join = sum(float(polygon.area()) for polygon in shared)
        if join >= MIN_STUB_JOIN_UM2:
            return stub, depth
        depth += STUB_LANDING_STEP_UM
    raise ValueError(
        f"core port {port_name} finds no island within {max_extension_um:.1f} um; "
        "the stub would be floating"
    )


def freeform_core(
    cfg: AcceleratorConfig,
    core,
    polygons: list[list[tuple[float, float]]],
    freqs_hz: np.ndarray,
) -> tuple[pf.Component, dict[str, float]]:
    """Inverse-designed SiN scatterer with access waveguides and the bonded TFLN slab.

    The eight stubs are access waveguides: each runs from its chip port into the design
    region until it shares at least ``MIN_STUB_JOIN_UM2`` with an island. The landing depth
    per port is returned for the core report. The extension limit is 6 um for the analytic
    surrogate but 0.6 um for an ingested Tidy3D core, where a multi-micron lead would mean
    the design is not port-connected; a port with no island within the limit fails the
    build instead of shipping a floating waveguide.
    """
    comp = pf.Component("FREEFORM_CORE")
    layer = cfg.layer("sin_core")
    for points in polygons:
        comp.add(layer, pf.Polygon(points))

    # The stub must reach into the design region until it shares real area with an island.
    # A fixed 0.3 um overlap left four of the eight ports floating: the pattern has a void
    # in front of them, and the old "0.190 um minimum gap" was the stub-to-island gap, not
    # a pattern gap. The landing is now part of the build assertion, so a port cannot be
    # shipped unless its stub actually joins material.
    islands = [gdstk.Polygon(points) for points in polygons]
    if not islands:
        raise ValueError("the freeform core has no pattern polygons")
    landings: dict[str, float] = {}
    limit = stub_landing_limit(cfg, core)
    for i in range(cfg.N):
        y = cfg.port_y(i)
        for sign, port_name, angle in ((-1.0, f"IN_{i}", 0.0), (1.0, f"OUT_{i}", 180.0)):
            stub, depth = _stub_landing(cfg, islands, y, sign, port_name, limit)
            landings[port_name] = depth
            (x0, _), (x1, _) = stub.bounding_box()
            comp.add(
                layer,
                pf.Rectangle(
                    center=(0.5 * (x0 + x1), y), size=(x1 - x0, cfg.wg_width)
                ),
            )
            comp.add_port(
                pdk.port(cfg, (sign * (cfg.design_lx / 2 + cfg.stub_len), y), angle),
                port_name=port_name,
            )

    comp.add(
        cfg.layer("tfln_slab"),
        pf.Rectangle(center=(0.0, 0.0), size=(cfg.design_lx + 6.0, cfg.design_ly + 6.0)),
    )
    # Device marker: the freeform region is governed by the 140 nm scatterer rule, not the
    # 800 nm routing rule, so the chip-level waveguide DRC must skip it.
    comp.add(
        cfg.layer("exclude"),
        pf.Rectangle(
            center=(0.0, 0.0),
            size=(cfg.design_lx + 2 * cfg.stub_len + 1.0, cfg.design_ly + 1.0),
        ),
    )
    comp.add_model(smodels.core_model(core, freqs_hz), model_name="tidy3d_core")
    return comp, landings


def tfln_mzm(cfg: AcceleratorConfig, name: str, transmission: np.ndarray, freqs_hz: np.ndarray) -> pf.Component:
    """Push-pull TFLN Mach-Zehnder modulator with GSG travelling-wave electrodes."""
    comp = pf.Component(name)
    layer = cfg.layer("sin_core")
    width = cfg.wg_width
    length = cfg.mod_length_um
    arm_offset = 5.0          # push-pull arm separation / 2
    slab_width = 12.0         # 1x2 MMI splitter width
    slab_half = 15.0          # splitter / combiner slab length / 2
    taper_len = 20.0

    # Splitter: input taper into a multimode slab, then two tapered arms. Compared with a
    # point-to-point Y, the slab never narrows to zero, so the geometry is DRC clean.
    in_taper = pf.Path(origin=(0.0, 0.0), width=width)
    in_taper.segment(endpoint=(taper_len, 0.0), width=3.0)
    comp.add(layer, in_taper)
    comp.add(
        layer,
        pf.Rectangle(center=(taper_len + slab_half, 0.0), size=(2 * slab_half, slab_width)),
    )
    split_x = taper_len + 2 * slab_half
    for sign in (1.0, -1.0):
        out_taper = pf.Path(origin=(split_x, sign * arm_offset), width=3.0)
        out_taper.segment(endpoint=(split_x + taper_len, sign * arm_offset), width=width)
        comp.add(layer, out_taper)
        arm = pf.Path(origin=(split_x + taper_len, sign * arm_offset), width=width)
        arm.segment(endpoint=(length - split_x - taper_len, sign * arm_offset))
        comp.add(layer, arm)
        in_taper_out = pf.Path(origin=(length - split_x - taper_len, sign * arm_offset), width=width)
        in_taper_out.segment(endpoint=(length - split_x, sign * arm_offset), width=3.0)
        comp.add(layer, in_taper_out)

    comp.add(
        layer,
        pf.Rectangle(center=(length - taper_len - slab_half, 0.0), size=(2 * slab_half, slab_width)),
    )
    out_taper = pf.Path(origin=(length - taper_len, 0.0), width=3.0)
    out_taper.segment(endpoint=(length, 0.0), width=width)
    comp.add(layer, out_taper)

    # Bonded TFLN slab over the active section.
    comp.add(
        cfg.layer("tfln_slab"),
        pf.Rectangle(center=(length / 2, 0.0), size=(length - 2 * split_x, 2 * arm_offset + 24.0)),
    )

    # GSG electrodes: signal trace between the arms (push-pull field), grounds outside.
    comp.add(
        cfg.layer("rf_metal"),
        pf.Rectangle(center=(length / 2, 0.0), size=(length - 2 * split_x, 6.0)),
    )
    for sign in (1.0, -1.0):
        comp.add(
            cfg.layer("rf_metal"),
            pf.Rectangle(center=(length / 2, sign * 18.0), size=(length - 2 * split_x, 12.0)),
        )

    comp.add_port(pdk.port(cfg, (0.0, 0.0), 0.0), port_name="OPT_IN")
    comp.add_port(pdk.port(cfg, (length, 0.0), 180.0), port_name="OPT_OUT")
    comp.add_model(smodels.two_port("OPT_IN", "OPT_OUT", transmission, freqs_hz), model_name="mzm")
    return comp


def wdm_mux(cfg: AcceleratorConfig, name: str, freqs_hz: np.ndarray) -> pf.Component:
    """Add-drop microring WDM multiplexer, one ring per comb line.

    Rings sit below a straight through bus. Each drop waveguide leaves the ring plane
    downwards and exits on the left edge, with the ports ordered so the leftmost ring owns
    the highest port. That ordering keeps every drop path planar with no crossings.
    """
    comp = pf.Component(name)
    layer = cfg.layer("sin_core")
    width = cfg.wg_width
    gap = cfg.drc_min_gap + 0.05  # 50 nm margin over the foundry rule
    radius = cfg.ring_radius_um
    pitch_x = 2.0 * radius + 120.0
    x0 = MUX_X0
    # Distance from a ring centre to the centre of the adjacent bus: the ring's outer
    # radius, the coupler gap, and the bus's own half-width. Omitting the last term puts
    # the bus 0.1 um inside the ring, which shorts the resonator to both buses.
    y_ring = -(radius + gap + width)
    y_drop = y_ring - (radius + gap + width)

    bus_x0 = x0 - 80.0
    bus_x1 = x0 + (cfg.K - 1) * pitch_x + 60.0
    comp.add(
        layer,
        pf.Rectangle(center=((bus_x0 + bus_x1) / 2, 0.0), size=(bus_x1 - bus_x0, width)),
    )
    comp.add_port(pdk.port(cfg, (bus_x1, 0.0), 180.0), port_name="BUS_OUT")

    for k in range(cfg.K):
        x_center = x0 + k * pitch_x
        comp.add(
            layer,
            pf.Circle(
                center=(x_center, y_ring),
                radius=radius + width / 2,
                inner_radius=radius - width / 2,
                min_evals=PATH_EVALS,
            ),
        )
        # TiN heater over the top of the ring for resonance trimming onto the ITU grid.
        comp.add(
            cfg.layer("heater"),
            pf.Rectangle(center=(x_center, y_ring + radius + 2.0), size=(1.4 * radius, 2.4)),
        )

        # Ports descend with k so no drop path crosses another: the leftmost ring owns the
        # highest port and each path rises to the drop plane to the right of the previous
        # ring's coupling section.
        y_port = y_drop - 40.0 - k * 40.0
        drop = pf.Path(origin=(0.0, y_port), width=width)
        drop.segment(endpoint=(x_center - MUX_LEAD, y_port))
        drop.s_bend(endpoint=(x_center - 40.0, y_drop), euler_fraction=0.5, min_evals=PATH_EVALS)
        drop.segment(endpoint=(x_center + 35.0, y_drop))
        comp.add(layer, drop)
        comp.add_port(pdk.port(cfg, (0.0, y_port), 0.0), port_name=f"CH_IN_{k}")

    comp.add_model(smodels.mux_model(cfg, freqs_hz), model_name="ring_mux")
    return comp


def homodyne_hybrid(cfg: AcceleratorConfig, name: str, freqs_hz: np.ndarray) -> pf.Component:
    """2x2 3 dB directional coupler acting as a 180 degree balanced homodyne hybrid."""
    comp = pf.Component(name)
    layer = cfg.layer("sin_core")
    width = cfg.wg_width
    center_sep = width + cfg.drc_min_gap + 0.05
    port_pitch = 12.0
    coupler_half = 20.0
    half_length = 70.0  # 50 um per transition: the 20 um version bent at R ~ 10 um

    for sign in (1.0, -1.0):
        path = pf.Path(origin=(-half_length, sign * port_pitch / 2), width=width)
        path.s_bend(endpoint=(-coupler_half, sign * center_sep / 2), euler_fraction=0.5, min_evals=PATH_EVALS)
        path.segment(endpoint=(coupler_half, sign * center_sep / 2))
        path.s_bend(endpoint=(half_length, sign * port_pitch / 2), euler_fraction=0.5, min_evals=PATH_EVALS)
        comp.add(layer, path)

    # Thermo-optic phase trimmer on the LO arm to lock the homodyne reference phase.
    comp.add(
        cfg.layer("heater"),
        pf.Rectangle(center=(-45.0, -port_pitch / 2), size=(20.0, 2.4)),
    )

    comp.add_port(pdk.port(cfg, (-half_length, port_pitch / 2), 0.0), port_name="SIG_IN")
    comp.add_port(pdk.port(cfg, (-half_length, -port_pitch / 2), 0.0), port_name="LO_IN")
    comp.add_port(pdk.port(cfg, (half_length, port_pitch / 2), 180.0), port_name="PD_POS")
    comp.add_port(pdk.port(cfg, (half_length, -port_pitch / 2), 180.0), port_name="PD_NEG")
    comp.add_model(smodels.hybrid_model(cfg, freqs_hz), model_name="hybrid")
    return comp


def lo_splitter(cfg: AcceleratorConfig, name: str, count: int, freqs_hz: np.ndarray) -> pf.Component:
    """1 x N MMI star coupler used to fan a single LO comb out to every hybrid.

    Kept as a standalone component in this revision: the golden-build chip takes the LO
    comb through per-hybrid fibre ports, which keeps the photonic layer crossing free. The
    splitter is DRC-checked and exported for the next revision.
    """
    comp = pf.Component(name)
    layer = cfg.layer("sin_core")
    width = cfg.wg_width
    taper_len = 20.0
    mmi_len = 120.0
    pitch = 12.0
    slab_width = (count - 1) * pitch + 3.2
    x_slab0 = taper_len
    x_slab1 = taper_len + mmi_len

    in_taper = pf.Path(origin=(0.0, 0.0), width=width)
    in_taper.segment(endpoint=(taper_len, 0.0), width=3.2)
    comp.add(layer, in_taper)
    comp.add(
        layer,
        pf.Rectangle(center=((x_slab0 + x_slab1) / 2, 0.0), size=(mmi_len, slab_width)),
    )

    for i in range(count):
        y = (i - (count - 1) / 2) * pitch
        taper = pf.Path(origin=(x_slab1, y), width=3.2)
        taper.segment(endpoint=(x_slab1 + taper_len, y), width=width)
        comp.add(layer, taper)
        comp.add_port(pdk.port(cfg, (x_slab1 + taper_len, y), 180.0), port_name=f"LO_OUT_{i}")

    comp.add_port(pdk.port(cfg, (0.0, 0.0), 0.0), port_name="LO_IN")
    comp.add_model(smodels.lo_splitter_model(cfg, count, freqs_hz), model_name="mmi_star")
    return comp


def s_connector(cfg: AcceleratorConfig, name: str, dx: float, dy: float, freqs_hz: np.ndarray) -> pf.Component:
    """Planar connector between two collinear-facing ports: straight or Euler S-bend.

    Port ``A`` sits at the origin facing ``+x`` and port ``B`` at ``(dx, dy)`` facing ``-x``,
    so placing the reference at the start port lands ``B`` exactly on the target.
    """
    comp = pf.Component(name)
    lead_in = min(60.0, 0.5 * max(dx, 1.0))
    path = pf.Path(origin=(0.0, 0.0), width=cfg.wg_width)
    path.segment(endpoint=(lead_in, 0.0))
    if abs(dy) < 1e-9:
        path.segment(endpoint=(dx, 0.0))
    else:
        path.s_bend(endpoint=(dx, dy), euler_fraction=0.5, min_evals=PATH_EVALS)
    comp.add(cfg.layer("sin_core"), path)
    # TiN phase trimmer on the straight lead-in: every lane needs an independent phase
    # reference at the core input, because lithographic path-length errors are far larger
    # than a wavelength even when the layout is nominally matched.
    comp.add(
        cfg.layer("heater"),
        pf.Rectangle(center=(0.5 * lead_in, 0.0), size=(0.8 * lead_in, 2.4)),
    )
    comp.add_port(pdk.port(cfg, (0.0, 0.0), 0.0), port_name="A")
    comp.add_port(pdk.port(cfg, (dx, dy), 180.0), port_name="B")
    length = float(path.length())
    comp.add_model(smodels.route_model(cfg, length, freqs_hz, ports=("A", "B")), model_name="route")
    return comp
