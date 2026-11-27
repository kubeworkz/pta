"""Command line interface: ``pta design | layout | verify | metrics | bundle | all``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from . import config as config_module


class CliInputError(ValueError):
    """A CLI-supplied config file, override, or inputs file is missing or invalid."""


def _parse_overrides(values: Sequence[str] | None) -> dict[str, Any]:
    known = {item.name for item in fields(config_module.AcceleratorConfig)}
    overrides: dict[str, Any] = {}
    for item in values or ():
        key, separator, raw = item.partition("=")
        if not separator:
            raise CliInputError(f"--set expects key=value, got {item!r}")
        key = key.strip()
        if key not in known:
            raise CliInputError(f"--set {item}: unknown configuration key {key!r}")
        try:
            overrides[key] = yaml.safe_load(raw.strip())
        except yaml.YAMLError as exc:
            raise CliInputError(f"--set value {item!r} is not valid YAML") from exc
    return overrides


def _load_config(path: str | None, overrides: dict[str, Any]) -> config_module.AcceleratorConfig:
    """Load the configuration, turning user-input failures into a CLI error."""
    source = path or "default configuration"
    try:
        config = config_module.load_config(path)
    except FileNotFoundError as exc:
        raise CliInputError(str(exc)) from exc
    except yaml.YAMLError as exc:
        raise CliInputError(f"cannot read config {source}: not valid YAML") from exc
    except OSError as exc:  # directory, permission denied, unreadable file, ...
        raise CliInputError(f"cannot read config {source}: not a readable file") from exc
    except (TypeError, ValueError) as exc:
        raise CliInputError(f"invalid config {source}: {exc}") from exc

    if not overrides:
        return config

    base = config
    for key, value in overrides.items():
        try:
            config = config.with_overrides(**{key: value})
        except (TypeError, ValueError) as exc:
            raise CliInputError(f"--set {key}={value}: {exc}") from exc

    try:
        config.validate()
    except (TypeError, ValueError) as exc:
        # Name a single bad override when there is one; cross-field failures that need the
        # whole set are reported against every override instead of blaming the first.
        for key, value in overrides.items():
            try:
                base.with_overrides(**{key: value}).validate()
            except (TypeError, ValueError):
                raise CliInputError(f"--set {key}={value}: {exc}") from exc
        joined = ", ".join(f"--set {key}={value}" for key, value in overrides.items())
        raise CliInputError(f"{joined}: {exc}") from exc
    return config


def _load_inputs(path: str | None, cfg: config_module.AcceleratorConfig) -> np.ndarray | None:
    if path is None:
        return None
    try:
        data = np.load(path, allow_pickle=False)
        key = "inputs" if "inputs" in data else next(iter(data.files))
        inputs = np.asarray(data[key], dtype=float)
    except FileNotFoundError as exc:
        raise CliInputError(f"inputs file not found: {path}") from exc
    except Exception as exc:  # malformed npz: bad zip, no arrays, non-numeric data, ...
        raise CliInputError(f"cannot read inputs {path}: not a valid npz") from exc
    if inputs.shape != (cfg.N, cfg.K):
        raise CliInputError(f"inputs must have shape {(cfg.N, cfg.K)}, got {inputs.shape}")
    return inputs


def _print_summary(result) -> None:
    summary = result.summary()
    m = result.metrics
    print("")
    print(f"  chip            {summary['chip_size_mm'][0]:.3f} x {summary['chip_size_mm'][1]:.3f} mm")
    print(f"  blocks/ports    {summary['blocks']} / {summary['ports']}")
    print(f"  DRC core/chip   {summary['drc_core_passes']} / {summary['drc_chip_passes']}")
    print(f"  signed MVM      {summary['mvm_worst_relative_error']*100:.2f} % worst relative error")
    print(f"  throughput      {m['tops']:.1f} TOPS at {m['baud_ghz']:.0f} GBd")
    print(f"  efficiency      {m['tops_per_w']:.2f} TOPS/W  ({m['energy_per_mac_fj']:.1f} fJ/MAC)")
    print(f"  receiver        {m['snr_db']:.1f} dB SNR, {m['enob_bits']:.2f} bits ENOB")
    print("")
    for line in result.verification["lines"]:
        print(
            f"  {line['wavelength_nm']:.0f} nm | error {line['relative_error']*100:5.2f} % | "
            f"measured {np.array2string(np.asarray(line['measured']), precision=3)}"
        )
    print("")
    for name, path in result.artifacts.items():
        print(f"  {name:16s} {path}")


def _common_parser() -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand.

    Defaults are suppressed so that a value given before the subcommand is not clobbered by
    the subparser's default.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=str, default=argparse.SUPPRESS, help="YAML configuration file")
    common.add_argument(
        "--set",
        action="append",
        default=argparse.SUPPRESS,
        metavar="KEY=VALUE",
        help="override a configuration entry (repeatable)",
    )
    common.add_argument("--out", type=str, default=argparse.SUPPRESS, help="artifacts directory")
    common.add_argument("--inputs", type=str, default=argparse.SUPPRESS, help="npz with an (N, K) input vector")
    common.add_argument(
        "--core",
        type=str,
        default=argparse.SUPPRESS,
        help="npz with a Tidy3D core S-matrix and its density map (required to build the layout)",
    )
    common.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="override the configuration seed")
    common.add_argument("--no-drc", action="store_true", default=argparse.SUPPRESS, help="skip the polygon DRC")
    return common


def main(argv: Sequence[str] | None = None) -> int:
    common = _common_parser()
    parser = argparse.ArgumentParser(
        prog="pta",
        description="Coherent frequency-comb photonic tensor accelerator: design, layout, verify, bundle.",
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("all", help="run every stage and build the portal bundle", parents=[common])
    subparsers.add_parser("design", help="freeform core model and Tidy3D hand-off", parents=[common])
    subparsers.add_parser("layout", help="assemble the chip, run DRC and export GDSII", parents=[common])
    subparsers.add_parser("verify", help="circuit-level signed MVM verification", parents=[common])
    metrics_parser = subparsers.add_parser("metrics", help="TOPS/W, SNR and ENOB", parents=[common])
    metrics_parser.add_argument("--pareto", action="store_true", help="also run the design sweep")
    subparsers.add_parser("bundle", help="package existing artifacts for the portals", parents=[common])
    import_parser = subparsers.add_parser(
        "import-em",
        help="assemble a core S-matrix from Tidy3D density and verify monitors",
        parents=[common],
    )
    import_parser.add_argument(
        "--density",
        default=None,
        help="optimized density map (.npy, or .npz with a 'density' array)",
    )
    import_parser.add_argument(
        "--monitors",
        default=None,
        help="directory with verify_in<i>.npz monitor amplitudes, one per input port",
    )
    import_parser.add_argument(
        "--design-data",
        default=None,
        help="Tidy3D inverse-design SimulationData file carrying the optimized density",
    )
    import_parser.add_argument(
        "--verify-data",
        nargs="+",
        default=None,
        help="Tidy3D verify SimulationData files, one per input port, in input order",
    )
    import_parser.add_argument(
        "--structure-name",
        default=None,
        help="CustomMedium structure holding the density (if the design data has several)",
    )
    import_parser.add_argument(
        "--core-out",
        default=None,
        help="output core npz (default: artifacts/core_smatrix.npz)",
    )

    args = parser.parse_args(argv)
    try:
        overrides = _parse_overrides(getattr(args, "set", None))
    except CliInputError as exc:
        print(f"pta: error: {exc}", file=sys.stderr)
        return 1
    try:
        cfg = _load_config(getattr(args, "config", None), overrides)
    except CliInputError as exc:
        print(f"pta: error: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "out", None):
        cfg = cfg.with_overrides(artifacts_dir=args.out)
    if getattr(args, "seed", None) is not None:
        cfg = cfg.with_overrides(seed=args.seed)

    if args.command == "import-em":
        from . import em_import

        try:
            data_mode = bool(args.design_data or args.verify_data)
            archive_mode = bool(args.density or args.monitors)
            if data_mode and archive_mode:
                raise em_import.EmImportError(
                    "use either --density/--monitors or --design-data/--verify-data, not both"
                )
            if data_mode:
                if not args.design_data or not args.verify_data:
                    raise em_import.EmImportError(
                        "--design-data and --verify-data must be given together"
                    )
                if len(args.verify_data) != cfg.N:
                    raise em_import.EmImportError(
                        f"expected {cfg.N} --verify-data files (one per input), "
                        f"got {len(args.verify_data)}"
                    )
                design = em_import.load_simulation_data(args.design_data)
                verify = [
                    em_import.load_simulation_data(
                        path,
                        monitor_names=[f"out_{index}" for index in range(cfg.N)],
                    )
                    for path in args.verify_data
                ]
                imported = em_import.import_from_simulation_data(
                    cfg,
                    verify,
                    design,
                    args.core_out,
                    structure_name=args.structure_name,
                )
            else:
                if not args.density or not args.monitors:
                    raise em_import.EmImportError(
                        "--density and --monitors are required unless --design-data "
                        "and --verify-data are given"
                    )
                imported = em_import.import_core(
                    cfg, Path(args.density), Path(args.monitors), args.core_out
                )
        except em_import.EmImportError as exc:
            print(f"pta: error: {exc}", file=sys.stderr)
            return 1
        worst = max(line["max_singular_value"] for line in imported.report["lines"])
        print(f"core: {imported.path}")
        print(
            f"  backend {imported.core.backend} | {imported.core.N}x{imported.core.N} | "
            f"{imported.core.K} lines | max singular {worst:.4f}"
        )
        print(
            f"  density {imported.core.density.shape[0]}x{imported.core.density.shape[1]} "
            f"| reciprocal | passive"
        )
        return 0

    try:
        inputs = _load_inputs(getattr(args, "inputs", None), cfg)
    except CliInputError as exc:
        print(f"pta: error: {exc}", file=sys.stderr)
        return 1
    run_drc = not getattr(args, "no_drc", False)
    core_path = getattr(args, "core", None)

    from . import bundle as bundle_module
    from . import layout as layout_module
    from . import metrics as metrics_module
    from . import pipeline as pipeline_module

    if args.command == "metrics":
        system = metrics_module.evaluate(cfg)
        path = layout_module.save_manifest(system, cfg.path("metrics"))
        print(json.dumps({key: system[key] for key in ("tops", "tops_per_w", "energy_per_mac_fj", "snr_db", "enob_bits")}, indent=2))
        print(f"written: {path}")
        if args.pareto:
            points = metrics_module.pareto_sweep(cfg)
            payload = {"points": points, "front": metrics_module.pareto_front(points)}
            pareto_path = layout_module.save_manifest(payload, cfg.path("metrics_pareto"))
            print(f"pareto points: {len(points)} -> {pareto_path}")
        return 0

    if args.command == "bundle":
        files = [cfg.path(name) for name in ("core_gds", "chip_gds", "lo_gds", "core_npz", "chip_npz",
                                             "layout", "drc_core", "drc_chip", "verify", "metrics",
                                             "t3d_design")]
        files += sorted(Path(cfg.artifacts_dir).glob("tidy3d/*.json"))
        path = bundle_module.build_bundle(cfg, files, out_dir="dist")
        print(f"bundle: {path}")
        return 0

    try:
        result = pipeline_module.run_pipeline(
            cfg, inputs=inputs, core_path=core_path, run_drc=run_drc
        )
    except pipeline_module.CoreIngestError as exc:
        print(f"pta: error: {exc}", file=sys.stderr)
        return 1
    _print_summary(result)

    if args.command == "all":
        files = list(result.artifacts.values())
        path = bundle_module.build_bundle(
            cfg,
            files,
            manifest_extra={"summary": result.summary()},
            out_dir="dist",
        )
        print(f"\n  portal bundle   {path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
