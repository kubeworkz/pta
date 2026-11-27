"""End-to-end pipeline: design -> layout -> verify -> metrics -> portal artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import blocks, core as core_module, drc, io, layout, metrics, pdk, polygons, smodels, t3d, verify
from .config import AcceleratorConfig


class CoreIngestError(ValueError):
    """The ``--core`` file could not be read or does not describe a usable core."""


class CoreDensityError(CoreIngestError):
    """An ingested core has no density map, so its geometry cannot be built."""


@dataclass
class PipelineResult:
    cfg: AcceleratorConfig
    core: core_module.CoreModel
    layout_result: layout.LayoutResult
    smatrix: Any
    verification: dict[str, Any]
    metrics: dict[str, Any]
    drc_core: dict[str, Any]
    drc_chip: dict[str, Any]
    artifacts: dict[str, Path] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        x0, y0, x1, y1 = self.layout_result.bounds
        return {
            "chip_size_mm": [round((x1 - x0) / 1000.0, 3), round((y1 - y0) / 1000.0, 3)],
            "ports": len(self.layout_result.ports),
            "blocks": len(self.layout_result.blocks),
            "connections": len(self.layout_result.connections),
            "drc_core_passes": self.drc_core["passes"],
            "drc_chip_passes": self.drc_chip["passes"],
            "mvm_worst_relative_error": self.verification["worst_relative_error"],
            "mvm_passes": self.verification["passes"],
            "tops": self.metrics["tops"],
            "tops_per_w": self.metrics["tops_per_w"],
            "enob_bits": self.metrics["enob_bits"],
        }


def default_inputs(cfg: AcceleratorConfig, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng((cfg.seed if seed is None else seed) + 11)
    return rng.uniform(-1.0, 1.0, size=(cfg.N, cfg.K))


def run_pipeline(
    cfg: AcceleratorConfig,
    inputs: np.ndarray | None = None,
    core_path: str | Path | None = None,
    run_drc: bool = True,
) -> PipelineResult:
    """Run every stage and write the portal-ready artifacts into ``cfg.artifacts_dir``."""
    pdk.install(cfg)
    artifacts: dict[str, Path] = {}

    # -- 1. core model -----------------------------------------------------
    if core_path is not None:
        try:
            core = core_module.CoreModel.from_npz(core_path)
        except FileNotFoundError as exc:
            raise CoreIngestError(f"core file not found: {core_path}") from exc
        except Exception as exc:  # malformed npz: bad zip, wrong keys, bad shapes, ...
            raise CoreIngestError(f"cannot read core {core_path}: not a valid npz") from exc
    else:
        core = core_module.build_core(cfg)

    if core.density is None:
        if core.backend != "analytic":
            raise CoreDensityError(
                f"ingested core {core_path} has no density map: an S-matrix alone cannot "
                "build the layout for a non-analytic core, and the analytic surrogate is "
                "not a substitute for the real geometry. Re-export the core with its "
                "density (CoreModel.save stores it) or run the analytic build."
            )
        core.density = core_module.surrogate_density(cfg)
    artifacts["core_npz"] = core.save(cfg.path("core_npz"))
    polygons_um = polygons.density_to_polygons(core.density, cfg)
    if not polygons_um:
        raise RuntimeError("the freeform density map produced no polygons")

    freqs_hz = core.freqs_hz if core.freqs_hz.shape == (cfg.K,) else cfg.freqs_hz
    inputs = default_inputs(cfg) if inputs is None else np.asarray(inputs, dtype=float)

    # -- 2. standalone core GDS + DRC --------------------------------------
    core_component, core_landings = blocks.freeform_core(cfg, core, polygons_um, freqs_hz)
    artifacts["core_gds"] = layout.export_gds(core_component, cfg.path("core_gds"))
    drc_core = (
        drc.check_gds(
            artifacts["core_gds"],
            cfg,
            rules="freeform",
            min_width=cfg.drc_freeform_um,
            min_gap=cfg.drc_freeform_um,
            width_method="erosion",
        )
        if run_drc
        else _skipped_drc("core")
    )

    # -- 3. LO splitter component -----------------------------------------
    splitter = blocks.lo_splitter(cfg, "LO_MMI_SPLITTER", cfg.N, freqs_hz)
    artifacts["lo_gds"] = layout.export_gds(splitter, cfg.path("lo_gds"))

    # -- 4. full chip ------------------------------------------------------
    layout_result = layout.build_chip(cfg, core, polygons_um, freqs_hz)
    artifacts["chip_gds"] = layout.export_gds(layout_result.component, cfg.path("chip_gds"))
    drc_chip = (
        drc.check_gds(
            artifacts["chip_gds"],
            cfg,
            rules="waveguide",
            exclude_freeform=True,
            expected_components=_expected_nets(cfg, artifacts["chip_gds"]),
            declared_joints=layout_result.declared_joints(),
        )
        if run_drc
        else _skipped_drc("chip")
    )

    drc_core["stub_landing_um"] = core_landings
    drc_core["stub_landing_limit_um"] = blocks.stub_landing_limit(cfg, core)

    layout_payload = layout_result.manifest()
    layout_payload["drc"] = drc_chip
    layout_payload["inputs"] = inputs.tolist()
    layout_payload["core_backend"] = core.backend
    layout_payload["model_status"] = _model_status(layout_result)
    artifacts["layout"] = layout.save_manifest(layout_payload, cfg.path("layout"))
    artifacts["drc_core"] = layout.save_manifest(drc_core, cfg.path("drc_core"))
    artifacts["drc_chip"] = layout.save_manifest(drc_chip, cfg.path("drc_chip"))

    # -- 5. circuit solve --------------------------------------------------
    smatrix = layout_result.component.s_matrix(freqs_hz, show_progress=False)
    artifacts["chip_npz"] = io.save_smatrix(
        cfg.path("chip_npz"),
        smatrix,
        freqs_hz,
        meta={
            "chip": layout_result.component.name,
            "core_backend": core.backend,
            "ports": list(smatrix.ports),
            "N": cfg.N,
            "K": cfg.K,
        },
    )

    # -- 6. verification and system model ----------------------------------
    verification = verify.verify_mvm(smatrix, cfg, core, inputs)
    verification["core_backend"] = core.backend
    artifacts["verify"] = layout.save_manifest(verification, cfg.path("verify"))

    system = metrics.evaluate(cfg)
    system["verification"] = {
        "worst_relative_error": verification["worst_relative_error"],
        "passes": verification["passes"],
    }
    system["drc"] = {"core": drc_core["passes"], "chip": drc_chip["passes"]}
    artifacts["metrics"] = layout.save_manifest(system, cfg.path("metrics"))

    # -- 7. Tidy3D hand-off ------------------------------------------------
    artifacts["t3d_design"] = t3d.write_design_spec(cfg, core.weights, cfg.path("t3d_design"))
    verify_specs = t3d.write_verification_specs(cfg, artifacts["core_gds"], Path(cfg.artifacts_dir) / "tidy3d")
    for index, path in enumerate(verify_specs):
        artifacts[f"t3d_verify_in{index}"] = path
    artifacts["t3d_validate"] = t3d.write_straight_validation_spec(
        cfg, Path(cfg.artifacts_dir) / "tidy3d" / "tidy3d_validate_straight.json"
    )

    return PipelineResult(
        cfg=cfg,
        core=core,
        layout_result=layout_result,
        smatrix=smatrix,
        verification=verification,
        metrics=system,
        drc_core=drc_core,
        drc_chip=drc_chip,
        artifacts=artifacts,
    )


def _expected_nets(cfg: AcceleratorConfig, chip_gds: Path) -> dict[str, int]:
    """Connectivity expectation for every drawn cell.

    Every cell on the chip is one connected net except the mux (bus + K rings + K drops)
    and the hybrid (two independent arms). The freeform core is a density map of many
    islands, not a net, so it is excluded from the connectivity rule.
    """
    expected = {name: 1 for name in drc.cell_names(chip_gds, cfg.layer("sin_core"))}
    expected.pop("FREEFORM_CORE", None)
    for index in range(cfg.N):
        expected[f"WDM_MUX_L{index}"] = 2 * cfg.K + 1
    expected["HOMODYNE_HYBRID"] = 2
    return expected


def _skipped_drc(name: str) -> dict[str, Any]:
    return {"file": name, "passes": None, "skipped": True, "layers": {}}


def _model_status(layout_result: layout.LayoutResult) -> dict[str, str]:
    status: dict[str, str] = {}
    for block in layout_result.blocks:
        status.setdefault(block["kind"], block["status"])
    return status
