"""Full-chip assembly: floorplan, connections, DRC and GDSII export.

The floorplan is a left-to-right dataflow:

``COMB_SIG_IN_i_k -> TFLN MZM -> ring WDM mux -> freeform core -> homodyne hybrid -> PD``

with the LO comb delivered to each hybrid's ``COMB_LO_IN_i`` port. Every connection between
blocks is an explicit Euler S-bend of known length, so the layout has no crossings, no
overlaps between unrelated waveguides and a deterministic S-matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import photonforge as pf

from . import blocks, pdk, smodels
from .config import AcceleratorConfig


@dataclass
class LayoutResult:
    component: pf.Component
    blocks: list[dict[str, Any]] = field(default_factory=list)
    connections: list[dict[str, Any]] = field(default_factory=list)
    ports: dict[str, dict[str, Any]] = field(default_factory=dict)
    floorplan: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return _bounds_of(self.component)

    def declared_joints(self) -> dict[tuple[str, str], list[tuple[float, float]]]:
        """Reference pairs the floorplan declares jointed, and where.

        A connection record names the route cell, the two blocks it joins, and the port
        coordinates at both ends. The material joints are route-to-block at those ports, so
        each pair is declared together with the coordinate it is allowed to touch at. The
        DRC uses this to reject two unrelated nets drawn edge to edge *and* a declared route
        that happens to touch its block somewhere other than the port it was routed to.
        """
        joints: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for connection in self.connections:
            route = str(connection["name"])
            source, _, _ = str(connection["from"]).partition(".")
            target, _, _ = str(connection["to"]).partition(".")
            source_point = tuple(float(v) for v in connection["source_um"])
            target_point = tuple(float(v) for v in connection["target_um"])
            for first, second, point in (
                (route, source, source_point),
                (route, target, target_point),
            ):
                joints.setdefault(tuple(sorted((first, second))), []).append(point)  # type: ignore[arg-type]
        return joints

    def manifest(self) -> dict[str, Any]:
        x0, y0, x1, y1 = self.bounds
        return {
            "chip": self.component.name,
            "bounds_um": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1,
                          "width": x1 - x0, "height": y1 - y0},
            "floorplan_um": self.floorplan,
            "blocks": self.blocks,
            "connections": self.connections,
            "ports": self.ports,
            "warnings": self.warnings,
        }


def _bounds_of(component: pf.Component) -> tuple[float, float, float, float]:
    values = np.asarray(component.bounds(), dtype=float).reshape(-1)
    if values.size != 4:
        raise ValueError(f"unexpected bounds shape from PhotonForge: {values}")
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _describe(component: pf.Component, kind: str, model: str, status: str) -> dict[str, Any]:
    x0, y0, x1, y1 = _bounds_of(component)
    return {
        "name": component.name,
        "kind": kind,
        "model": model,
        "status": status,
        "ports": {name: {"center": [float(c) for c in port.center],
                         "direction": float(port.input_direction)}
                  for name, port in component.ports.items()},
        "bounds_um": [x0, y0, x1, y1],
    }


def build_chip(
    cfg: AcceleratorConfig,
    core,
    polygons: list[list[tuple[float, float]]],
    freqs_hz: np.ndarray,
) -> LayoutResult:
    """Assemble the full accelerator chip.

    Every MZM is biased at ``V = V_pi`` (unit field transmission) so the solved S-matrix is
    the linear kernel of the machine; the operand is applied as a signed modulation depth
    during verification. That keeps one layout valid for every input vector.

    Args:
        cfg: build configuration.
        core: :class:`pta.core.CoreModel` for the freeform core.
        polygons: DRC-clean freeform polygons in micrometres.
        freqs_hz: comb frequencies used for every model.
    """
    pdk.install(cfg)

    result = LayoutResult(component=pf.Component("PTA_GOLDEN_BUILD"))
    top = result.component

    core_comp, _ = blocks.freeform_core(cfg, core, polygons, freqs_hz)
    hybrid_comp = blocks.homodyne_hybrid(cfg, "HOMODYNE_HYBRID", freqs_hz)
    result.blocks.append(_describe(core_comp, "freeform_core", "tidy3d", "em_verified"))
    result.blocks.append(_describe(hybrid_comp, "homodyne_hybrid", "ideal_2x2", "analytic"))

    # -- floorplan ---------------------------------------------------------
    mzm_x = 0.0
    mzm_len = cfg.mod_length_um
    mux_x = mzm_len + 600.0
    mux_bus_x1 = blocks.mux_bus_span(cfg)[1]
    # The lane fan-in has to absorb up to (N-1)/2 lane pitches of vertical offset. Keep the
    # horizontal run long enough that the S-bends stay gentle: steep S-bends fold their own
    # inner edge and violate the width rule.
    core_in_x = mux_x + mux_bus_x1 + 3200.0
    core_center_x = core_in_x + cfg.design_lx / 2 + cfg.stub_len
    core_out_x = core_center_x + cfg.design_lx / 2 + cfg.stub_len
    hybrid_x = core_out_x + 2200.0
    result.floorplan = {
        "mzm_x": mzm_x,
        "mzm_length": mzm_len,
        "mux_x": mux_x,
        "core_in_x": core_in_x,
        "core_center_x": core_center_x,
        "core_out_x": core_out_x,
        "hybrid_x": hybrid_x,
        "lane_pitch": cfg.lane_pitch,
    }

    core_ref = top.add_reference(core_comp).translate((core_center_x, 0.0))

    def connect(
        ref_a: pf.Reference,
        port_a: str,
        ref_b: pf.Reference,
        port_b: str,
        name: str,
    ) -> None:
        """Route from ``ref_a.port_a`` to ``ref_b.port_b`` and record the declared joint.

        The target is resolved from the destination port, not passed as bare coordinates, so
        the manifest names both references and the DRC can verify that every butt joint on
        the chip is one the floorplan intended.
        """
        port = ref_a[port_a]
        target_port = ref_b[port_b]
        source = (float(port.center[0]), float(port.center[1]))
        target = (float(target_port.center[0]), float(target_port.center[1]))
        dx = target[0] - source[0]
        dy = target[1] - source[1]
        if dx <= 1.0:
            raise ValueError(f"{name}: no horizontal room to route ({dx:.1f} um)")
        connector = blocks.s_connector(cfg, name, dx, dy, freqs_hz)
        top.add_reference(connector).translate(port.center)
        result.connections.append(
            {
                "name": name,
                "from": f"{ref_a.component_name}.{port_a}",
                "to": f"{ref_b.component_name}.{port_b}",
                "source_um": [source[0], source[1]],
                "target_um": [target[0], target[1]],
                "dx_um": dx,
                "dy_um": dy,
            }
        )

    for i in range(cfg.N):
        lane_y = (i - (cfg.N - 1) / 2.0) * cfg.lane_pitch

        mux = blocks.wdm_mux(cfg, f"WDM_MUX_L{i}", freqs_hz)
        mux_ref = top.add_reference(mux).translate((mux_x, lane_y))
        result.blocks.append(_describe(mux, "wdm_mux", "ring_lorentzian", "analytic"))

        # -- input lanes: K modulators, one per comb line -------------------
        # Rows descend with k because the mux drops descend with k (each drop must pass
        # below the coupling section of the ring to its left). Ascending rows here made the
        # three connectors cross each other in every lane.
        for k in range(cfg.K):
            mzm_y = lane_y - (k - (cfg.K - 1) / 2.0) * cfg.fiber_pitch
            transfer = smodels.mzm_transfer(cfg, np.full(cfg.K, cfg.v_pi), freqs_hz)
            mzm = blocks.tfln_mzm(cfg, f"MZM_L{i}_C{k}", transfer, freqs_hz)
            mzm_ref = top.add_reference(mzm).translate((mzm_x, mzm_y))
            top.add_port(mzm_ref["OPT_IN"], port_name=f"COMB_SIG_IN_{i}_{k}")
            result.blocks.append(_describe(mzm, "tfln_mzm", "analytic_push_pull", "analytic"))

            connect(mzm_ref, "OPT_OUT", mux_ref, f"CH_IN_{k}", f"ROUTE_MZM_L{i}_C{k}")

        # -- mux bus into core input --------------------------------------
        connect(mux_ref, "BUS_OUT", core_ref, f"IN_{i}", f"ROUTE_IN_{i}")

        # -- core output into homodyne hybrid -----------------------------
        hybrid_ref = top.add_reference(hybrid_comp).translate((hybrid_x, lane_y))
        connect(core_ref, f"OUT_{i}", hybrid_ref, "SIG_IN", f"ROUTE_OUT_{i}")

        top.add_port(hybrid_ref["LO_IN"], port_name=f"COMB_LO_IN_{i}")
        top.add_port(hybrid_ref["PD_POS"], port_name=f"PD_POS_{i}")
        top.add_port(hybrid_ref["PD_NEG"], port_name=f"PD_NEG_{i}")

    top.add_model(pf.CircuitModel(), model_name="accelerator_circuit")

    for name, port in top.ports.items():
        result.ports[name] = {
            "center": [float(c) for c in port.center],
            "direction": float(port.input_direction),
        }
    return result


def export_gds(component: pf.Component, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    component.write_gds(str(path))
    return path


def save_manifest(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path
