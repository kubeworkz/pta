"""Portal upload bundle: manifest with checksums plus a single zip archive."""

from __future__ import annotations

import hashlib
import json
import platform
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import AcceleratorConfig

UPLOAD_README = """# PTA golden build - portal upload

## Tidy3D

Before the core jobs, upload `tidy3d_validate_straight.json` and run
`python -m pta.validate_straight <downloaded>.hdf5`. A straight guide must show forward
power near 1 and weak backward/reflection power; the full checklist is in
`docs/tidy3d-validation.md`. Only then spend credits on the core runs.

1. `tidy3d_design_spec.json` describes the inverse-design problem (stack, design region,
   comb lines, target matrices, filter and projection schedule). Submit it to the
   inverse-design service to produce the freeform mask.
2. `pta_core.gds` is the freeform core geometry (layer {sin_layer} SiN, layer {tfln_layer}
   TFLN mesa).
3. `tidy3d/tidy3d_verify_in*.json` are complete 3D FDTD simulations of that GDS, one per
   input port. Upload each, export the output-monitor amplitudes as
   `verify_in<i>.npz` (key `amplitudes`, shape `(K, N)`), then run
   `pta import-em --density <mask.npy> --monitors <dir>` to write a reciprocal, passive
   `core_smatrix.npz` carrying the density map. With the downloaded `SimulationData`
   files, `pta import-em --design-data design.hdf5 --verify-data verify_in0.hdf5 ...`
   extracts and imports them in one command.
4. Run `pta all --core artifacts/core_smatrix.npz` and report the Frobenius error against
   the analytic kernel and the worst back-reflection.

## PhotonForge

1. `pta_chip.gds` is the full tapeout layout. Import it as a library and check the layer map
   in `layout_manifest.json`.
2. `chip_smatrix.npz` holds the solved circuit S-matrix at every comb line, with the port
   order listed in the file, for cross-checking the imported layout.
3. `layout_manifest.json` lists every block, its model class and its verification status, so
   the portal can flag blocks that still need an EM sweep.
4. `drc_core.json` and `drc_chip.json` are the polygon-level DRC reports for the SiN and
   TFLN layers.

## Sign-off checklist

- [ ] DRC clean on both layers (see `drc_chip.json`)
- [ ] Signed MVM relative error below the pass threshold (`verification.json`)
- [ ] Freeform core S-matrix replaced by the Tidy3D extraction (`core_smatrix.npz`)
- [ ] Mux, MZM and hybrid models replaced by EM/EO sweeps before tapeout
- [ ] Layer map confirmed against the foundry DRM
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_bundle(
    cfg: AcceleratorConfig,
    files: Iterable[str | Path],
    manifest_extra: dict[str, Any] | None = None,
    out_dir: str | Path = "dist",
) -> Path:
    """Write ``portal_manifest.json`` and a zip of every artifact beside it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    resolved: list[Path] = []
    for item in files:
        path = Path(item)
        if not path.exists():
            continue
        resolved.append(path)
        entries.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})

    manifest = {
        "schema": "pta.portal_bundle.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chip": "PTA_GOLDEN_BUILD",
        "configuration": {
            "N": cfg.N,
            "K": cfg.K,
            "wavelengths_nm": [round(w * 1e3, 3) for w in cfg.wavelengths_um],
            "layers": {name: list(layer) for name, layer in cfg.layers.items()},
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "files": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }
    if manifest_extra:
        manifest.update(manifest_extra)

    manifest_path = out_dir / "portal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")

    readme_path = out_dir / "README_UPLOAD.md"
    readme_path.write_text(
        UPLOAD_README.format(
            sin_layer=cfg.layer("sin_core"), tfln_layer=cfg.layer("tfln_slab")
        ),
        encoding="utf-8",
    )

    bundle_path = out_dir / "pta_portal_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in resolved:
            archive.write(path, arcname=path.name)
        archive.write(manifest_path, arcname=manifest_path.name)
        archive.write(readme_path, arcname=readme_path.name)
    return bundle_path
