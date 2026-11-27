"""Tidy3D hand-off: the verification simulation and the inverse-design job description.

Two different artifacts are produced, because Tidy3D is used for two different things:

``tidy3d_verify_in<i>.json``
    A complete 3D FDTD :class:`tidy3d.Simulation` of the exported freeform core GDS. It is
    built and validated locally (no cloud credits, no API key) and can be uploaded to the
    portal as-is to extract the S-matrix that replaces the analytic surrogate.

``tidy3d_design_spec.json``
    The inverse-design problem statement: stack, design region, grid, comb lines, target
    matrices, filter radius, projection schedule and objective. This is the job description
    for the adjoint run that produces the mask the core spec verifies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import AcceleratorConfig


def _media(cfg: AcceleratorConfig):
    import tidy3d as td

    sio2 = td.Medium(permittivity=cfg.n_sio2**2, name="SiO2")
    sin = td.Medium(permittivity=cfg.n_sin**2, name="SiN")
    tfln = td.AnisotropicMedium(
        xx=td.Medium(permittivity=cfg.n_tfln_o**2),
        yy=td.Medium(permittivity=cfg.n_tfln_o**2),
        zz=td.Medium(permittivity=cfg.n_tfln_e**2),
        name="TFLN",
    )
    return sio2, sin, tfln


def _mode_source(cfg: AcceleratorConfig, x: float, y: float, freq: float | None = None):
    """The core mode source, shared so the straight validation cannot drift from it.

    ``freq`` narrows the pulse to one comb line; the verification and straight-validation
    simulations leave it unset and launch the full comb from its centre frequency.
    """
    import tidy3d as td

    freq0 = float(np.mean(cfg.freqs_hz)) if freq is None else float(freq)
    return td.ModeSource(
        center=(x, y, 0.0),
        size=(0.0, 2.5, cfg.t_sin + 1.5),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=0.15 * freq0),
        direction="+",
        mode_spec=td.ModeSpec(num_modes=1),
        name="input",
    )


def _mode_monitor(
    cfg: AcceleratorConfig, name: str, x: float, y: float, freq: float | None = None
):
    """The core output/reflection monitor, shared with the straight validation."""
    import tidy3d as td

    return td.ModeMonitor(
        center=(x, y, 0.0),
        size=(0.0, 2.0, cfg.t_sin + 1.5),
        freqs=cfg.freqs_hz if freq is None else [float(freq)],
        mode_spec=td.ModeSpec(num_modes=1),
        name=name,
    )


def build_verification_sim(cfg: AcceleratorConfig, gds_path: str | Path, input_index: int):
    """Build the 3D FDTD verification simulation of the exported core."""
    import gdstk

    import tidy3d as td

    sio2, sin, tfln = _media(cfg)
    library = gdstk.read_gds(str(gds_path))
    cell = next(iter(library.cells))
    sin_layer = cfg.layer("sin_core")
    tfln_layer = cfg.layer("tfln_slab")

    geometry_sin = td.Geometry.from_gds(
        gds_cell=cell,
        gds_layer=sin_layer[0],
        gds_dtype=sin_layer[1],
        axis=2,
        slab_bounds=(-cfg.t_sin / 2, cfg.t_sin / 2),
    )
    geometry_tfln = td.Geometry.from_gds(
        gds_cell=cell,
        gds_layer=tfln_layer[0],
        gds_dtype=tfln_layer[1],
        axis=2,
        slab_bounds=(cfg.t_sin / 2 + cfg.t_gap, cfg.t_sin / 2 + cfg.t_gap + cfg.t_tfln),
    )

    structures = [
        td.Structure(geometry=geometry_sin, medium=sin, name="SiN"),
        td.Structure(geometry=geometry_tfln, medium=tfln, name="TFLN"),
    ]

    freq0 = float(np.mean(cfg.freqs_hz))
    x_source = -cfg.design_lx / 2 - 1.0
    x_monitor = cfg.design_lx / 2 + 1.0

    source = _mode_source(cfg, x_source, cfg.port_y(input_index))
    monitors = [
        _mode_monitor(cfg, f"out_{index}", x_monitor, cfg.port_y(index))
        for index in range(cfg.N)
    ]
    reflection = [
        _mode_monitor(cfg, f"refl_{index}", x_source - 0.4, cfg.port_y(index))
        for index in range(cfg.N)
    ]

    sim = td.Simulation(
        size=(cfg.design_lx + 2 * cfg.stub_len + 3.0, cfg.design_ly + 8.0, 2.8),
        run_time=500 / freq0,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=1.55),
        structures=structures,
        sources=[source],
        monitors=monitors + reflection,
        medium=sio2,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        # ASCII-only unit string: the default is the Greek mu, which round-trips badly
        # through portal upload forms and older Windows locales.
        plot_length_units="um",
    )
    return sim


def build_straight_validation_sim(cfg: AcceleratorConfig):
    """Minimal straight SiN waveguide for validating monitor direction and scale.

    Uses the same stack, mode source, comb frequencies and output/reflection monitors as
    :func:`build_verification_sim`, but only on lane 0: the other core lanes have no
    waveguide in this minimal geometry. The guide extends through both mode planes, with
    the same 0.5 um lead to the facet as the core access stubs, so the source launches and
    the monitor receives the guided SiN mode rather than a cladding mode.
    """
    import tidy3d as td

    sio2, sin, tfln = _media(cfg)
    y0 = cfg.port_y(0)
    x_source = -cfg.design_lx / 2 - 1.0
    x_monitor = cfg.design_lx / 2 + 1.0
    # Match the core stubs, which end at design_lx/2 + stub_len. Keep at least 0.5 um
    # beyond each mode plane even if stub_len is reduced below 1.5 um.
    guide_half = cfg.design_lx / 2 + max(cfg.stub_len, 1.5)
    guide_length = 2.0 * guide_half

    # The TFLN slab carries the TM-horizontal guided mode whose evanescent tails
    # extend well beyond the 2.0 um output monitor's y-extent. A 2.0 um monitor
    # therefore under-reports the guided power (we observed ~0.826 vs ~0.931 for
    # an 8.0 um monitor on an otherwise identical straight guide). Use a monitor
    # y-extent of 8.0 um and match the TFLN slab y-extent so the slab is not
    # extruded through the monitor plane and the guided mode is fully captured.
    monitor_y = 8.0
    slab_y = 8.0

    structures = [
        td.Structure(
            geometry=td.Box(
                center=(0.0, y0, 0.0),
                size=(guide_length, cfg.wg_width, cfg.t_sin),
            ),
            medium=sin,
            name="SiN",
        ),
        td.Structure(
            geometry=td.Box(
                center=(0.0, y0, cfg.t_sin / 2 + cfg.t_gap + cfg.t_tfln / 2),
                size=(max(cfg.design_lx + 6.0, guide_length), slab_y, cfg.t_tfln),
            ),
            medium=tfln,
            name="TFLN",
        ),
    ]

    freq0 = float(np.mean(cfg.freqs_hz))
    source = _mode_source(cfg, x_source, y0)
    output = td.ModeMonitor(
        center=(x_monitor, y0, 0.0),
        size=(0.0, monitor_y, cfg.t_sin + 1.5),
        freqs=cfg.freqs_hz,
        mode_spec=td.ModeSpec(num_modes=1),
        name="out_0",
    )
    reflection = td.ModeMonitor(
        center=(x_source - 0.4, y0, 0.0),
        size=(0.0, monitor_y, cfg.t_sin + 1.5),
        freqs=cfg.freqs_hz,
        mode_spec=td.ModeSpec(num_modes=1),
        name="refl_0",
    )

    return td.Simulation(
        center=(0.0, y0, 0.0),
        size=(2.0 * (guide_half + 1.5), 8.0, 2.8),
        run_time=500 / freq0,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=1.55),
        structures=structures,
        sources=[source],
        monitors=[output, reflection],
        medium=sio2,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        plot_length_units="um",
    )


DESIGN_TASK_PREFIX = "pta_core"

#: Depth of the SiN landing seeded at every port before the adjoint starts (um).
DESIGN_PORT_PAD_UM = 0.4

#: Conic filter radius and tanh projection schedule advertised by the design spec.
DESIGN_BETA_START = 2.0
DESIGN_BETA_GROWTH = 1.08
DESIGN_EPOCHS = 40

#: Throughput floor: each input must transmit this fraction of the target line norm.
DESIGN_THROUGHPUT_FLOOR = 0.5
DESIGN_THROUGHPUT_WEIGHT = 10.0

#: Erosion/dilation penalty that keeps the mask binarised while it is optimised.
DESIGN_PENALTY_WEIGHT = 1.0


def design_beta(step: int) -> float:
    """Projection steepness for optimizer step ``step`` (1-based), as the spec advertises."""
    return DESIGN_BETA_START * DESIGN_BETA_GROWTH**step


def design_initial_density(cfg: AcceleratorConfig) -> np.ndarray:
    """Surrogate fill with a solid SiN landing at every port of the design region.

    The design region starts at the stub tip, and an adjoint run started from a random fill
    typically leaves a void in front of a port within the first steps. An ingested core is
    rejected when a port finds no island within ``stub_landing_em_max_um``, so the landing is
    seeded (not rescued later, and not by loosening the rule). ``DESIGN_PORT_PAD_UM`` is well
    inside that limit.
    """
    from .core import surrogate_density

    density = surrogate_density(cfg)
    pad = max(int(round(DESIGN_PORT_PAD_UM / cfg.dl_opt)), 1)
    half_width = max(int(round(0.5 * cfg.wg_width / cfg.dl_opt)), 1)
    for index in range(cfg.N):
        row = int(round((cfg.port_y(index) + cfg.design_ly / 2) / cfg.dl_opt))
        low, high = max(row - half_width, 0), min(row + half_width, cfg.ny)
        density[:pad, low:high] = 1.0
        density[cfg.nx - pad :, low:high] = 1.0
    return density


def design_tasks(cfg: AcceleratorConfig) -> list[tuple[int, int]]:
    """``(comb line, input port)`` pairs; each pair is one inverse-design simulation."""
    return [(line, index) for line in range(cfg.K) for index in range(cfg.N)]


def build_design_simulation(
    cfg: AcceleratorConfig, line: int, input_index: int
) -> "object":
    """One comb line and one input port of the inverse-design problem.

    The installed ``tidy3d`` inverse-design plugin accepts only a single frequency per job
    (``InverseDesign._validate_metric_f``), so the comb is a set of these simulations joined
    by ``InverseDesignMulti``. Each one carries its own mode source at its line frequency --
    a source-free design simulation is rejected at upload -- and the SiN access stubs from
    every port up to the design-region edge, so the optimised region is port-connected.
    """
    import tidy3d as td

    sio2, sin, tfln = _media(cfg)
    freq = float(cfg.freqs_hz[line])
    x_source = -cfg.design_lx / 2 - 1.0
    x_monitor = cfg.design_lx / 2 + 1.0
    stub_center = (cfg.design_lx + cfg.stub_len) / 2.0

    structures = []
    for index in range(cfg.N):
        y = cfg.port_y(index)
        for sign, side in ((-1.0, "in"), (1.0, "out")):
            structures.append(
                td.Structure(
                    geometry=td.Box(
                        center=(sign * stub_center, y, 0.0),
                        size=(cfg.stub_len, cfg.wg_width, cfg.t_sin),
                    ),
                    medium=sin,
                    name=f"SiN_stub_{index}_{side}",
                )
            )
    structures.append(
        td.Structure(
            geometry=td.Box(
                center=(0.0, 0.0, cfg.t_sin / 2 + cfg.t_gap + cfg.t_tfln / 2),
                size=(cfg.design_lx + 6.0, cfg.design_ly + 6.0, cfg.t_tfln),
            ),
            medium=tfln,
            name="TFLN",
        )
    )

    return td.Simulation(
        size=(cfg.design_lx + 2 * cfg.stub_len + 3.0, cfg.design_ly + 8.0, 2.8),
        run_time=500.0 / freq,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10, wavelength=1.55),
        structures=structures,
        sources=[_mode_source(cfg, x_source, cfg.port_y(input_index), freq=freq)],
        monitors=[
            _mode_monitor(cfg, f"out_{index}", x_monitor, cfg.port_y(index), freq=freq)
            for index in range(cfg.N)
        ],
        medium=sio2,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        plot_length_units="um",
    )


def design_objective(
    cfg: AcceleratorConfig,
    weights: np.ndarray,
    *,
    tasks: list[tuple[int, int]],
    task_names: list[str],
):
    """Phase- and gain-invariant overlap over every ``(line, input)`` simulation.

    Verification fits a pilot gain and per-lane phases before it scores the kernel, so the
    design objective must be blind to the same two quantities: a per-element error against an
    absolute ``alpha * W * exp(i phi_ref)`` spends gradient on a constant the receiver
    calibrates away. Each task contributes the normalised overlap

        |<g, t>|^2 / (||g||^2 ||t||^2),   g = W[k][:, i] * exp(i phi_ref[k])

    minus a throughput shortfall, so a mask that transmits nothing cannot score merely by
    being self-consistent. It is differentiable, and one gradient step updates the density
    shared by every comb line.
    """
    import autograd.numpy as anp

    from tidy3d.plugins.expressions.metrics import ModeAmp

    targets = np.asarray(weights, dtype=float) * np.exp(
        1j * np.asarray(cfg.phi_ref, dtype=float)
    )[:, None, None]
    amps = {
        (line, index, out): ModeAmp(
            monitor_name=f"out_{out}",
            freqs=float(cfg.freqs_hz[line]),
            direction="+",
            mode_index=0,
        )
        for line, index in tasks
        for out in range(cfg.N)
    }

    def post_process_fn(batch) -> object:
        total = anp.zeros(())
        for (line, index), task_name in zip(tasks, task_names):
            data = batch[task_name]
            measured = anp.stack(
                [amps[(line, index, out)].evaluate(data) for out in range(cfg.N)]
            )
            target = targets[line][:, index]
            target_norm2 = float(np.sum(np.abs(target) ** 2))
            measured_norm2 = anp.sum(anp.abs(measured) ** 2)
            inner = anp.sum(anp.conj(target) * measured)
            total = total + anp.abs(inner) ** 2 / (target_norm2 * measured_norm2 + 1e-30)

            reference = cfg.alpha_scale * float(np.sqrt(target_norm2))
            shortfall = anp.maximum(
                0.0,
                DESIGN_THROUGHPUT_FLOOR * reference - anp.sqrt(measured_norm2 + 1e-30),
            )
            total = total - DESIGN_THROUGHPUT_WEIGHT * shortfall**2 / (reference**2 + 1e-30)
        return total

    return post_process_fn


def build_inverse_design(
    cfg: AcceleratorConfig, weights: np.ndarray, *, beta: float | None = None
):
    """Build the multi-line, multi-input inverse-design problem and its objective.

    The design region carries the conic filter and tanh projection the design spec advertises
    (``filter_radius_um = cfg.drc_min_feature``, ``projection = ramp_tanh``); ``beta`` is the
    projection steepness for this run's step of :func:`design_beta`.
    """
    import tidy3d.plugins.invdes as tdi

    tasks = design_tasks(cfg)
    beta = design_beta(1) if beta is None else float(beta)
    design = tdi.InverseDesignMulti(
        design_region=tdi.TopologyDesignRegion(
            size=(cfg.design_lx, cfg.design_ly, cfg.t_sin),
            center=(0.0, 0.0, 0.0),
            eps_bounds=(cfg.n_sio2**2, cfg.n_sin**2),
            pixel_size=cfg.dl_opt,
            transformations=(
                tdi.FilterProject(radius=cfg.drc_min_feature, beta=beta),
            ),
            penalties=(
                tdi.ErosionDilationPenalty(
                    length_scale=cfg.drc_freeform_um, weight=DESIGN_PENALTY_WEIGHT
                ),
            ),
        ),
        simulations=tuple(build_design_simulation(cfg, line, index) for line, index in tasks),
        task_name=DESIGN_TASK_PREFIX,
        output_monitor_names=tuple(
            tuple(f"out_{out}" for out in range(cfg.N)) for _ in tasks
        ),
    )
    objective = design_objective(
        cfg, weights, tasks=tasks, task_names=list(design.task_names)
    )
    return design, objective


def validate_design(design) -> None:
    """Upload-validate every simulation in a design, one at a time.

    ``Optimizer.validate_pre_upload`` dereferences ``design.simulation``, which does not
    exist on ``InverseDesignMulti`` and also skips the remaining simulations. Validate each
    base simulation directly instead, naming the task that failed.
    """
    simulations = getattr(design, "simulations", None)
    if simulations is None:
        simulations = (design.simulation,)
    names = list(getattr(design, "task_names", None) or [design.task_name])
    for name, simulation in zip(names, simulations):
        try:
            simulation.validate_pre_upload()
        except Exception as exc:
            raise ValueError(
                f"inverse-design simulation {name!r} is not uploadable: {exc}"
            ) from exc


def design_density(cfg: AcceleratorConfig, result, design=None) -> np.ndarray:
    """The optimised ``(cfg.nx, cfg.ny)`` density map from an optimizer history.

    Pass ``design`` to apply the design region's filter and projection: the raw parameters
    are not the mask the simulation solved, and shipping them would report a density the
    optimization never evaluated.
    """
    params = np.asarray(result.get_last("params"), dtype=float)
    if design is not None:
        params = np.asarray(design.design_region.material_density(params), dtype=float)
    density = params[..., 0] if params.ndim == 3 else params
    if density.shape != (cfg.nx, cfg.ny):
        raise ValueError(
            f"design result has density shape {density.shape}, expected {(cfg.nx, cfg.ny)}"
        )
    return np.clip(density, 0.0, 1.0)


def save_epoch_results(sim_data, directory: str | Path) -> list[Path]:
    """Write one optimizer epoch's per-task ``SimulationData`` to disk.

    The optimizer's ``BatchData`` exists only in memory for the duration of a step, and the
    per-task results behind it are not retrievable from Tidy3D afterwards (the adjoint tasks
    in particular return 404 once the run is over). The optimizer callback hands them back
    through ``aux_data["sim_data"]``, so they are written here instead of being dropped.

    ``sim_data`` is either the ``{task_name: SimulationData}`` mapping the callback provides
    or a single ``SimulationData``. Returns the files written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    items = list(sim_data.items()) if hasattr(sim_data, "items") else [(None, sim_data)]
    written: list[Path] = []
    for name, data in items:
        path = directory / (f"{name}.hdf5" if name else "simulation.hdf5")
        data.to_file(str(path))
        written.append(path)
    return written


def write_design_data(cfg: AcceleratorConfig, design, result, path: str | Path) -> Path:
    """Write the optimised design as a ``SimulationData`` for ``pta import-em``.

    The density is carried by the design region's custom medium, which is what the importer
    reads; no extra cloud run is needed to produce the file.
    """
    import tidy3d as td

    params = np.asarray(result.get_last("params"), dtype=float)
    simulations = design.to_simulation(params)
    if isinstance(simulations, dict):
        simulation = next(iter(simulations.values()))
    else:
        simulation = simulations
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    td.SimulationData(simulation=simulation, data=()).to_file(str(path))
    return path


def write_straight_validation_spec(cfg: AcceleratorConfig, path: str | Path) -> Path:
    """Write the straight-waveguide validation simulation as a Tidy3D JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    simulation = build_straight_validation_sim(cfg)
    path.write_text(simulation.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_verification_specs(
    cfg: AcceleratorConfig, gds_path: str | Path, out_dir: str | Path
) -> list[Path]:
    """Write one simulation JSON per input port and return the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(cfg.N):
        sim = build_verification_sim(cfg, gds_path, index)
        path = out_dir / f"tidy3d_verify_in{index}.json"
        path.write_text(sim.model_dump_json(indent=2), encoding="utf-8")
        written.append(path)
    return written


def design_spec(cfg: AcceleratorConfig, weights: np.ndarray) -> dict[str, Any]:
    """Job description for the inverse-design run that generates the freeform mask."""
    weights = np.asarray(weights, dtype=float)
    return {
        "schema": "pta.tidy3d.inverse_design.v1",
        "objective": {
            "type": "phase_invariant_overlap",
            "description": (
                "Maximise, for every comb line and input port, the phase- and gain-invariant "
                "overlap |<alpha * W[k] * exp(i phi_ref[k]), T>|^2 / (norms), with a floor on "
                "the transmitted norm so a dark mask cannot score."
            ),
            "throughput_floor": DESIGN_THROUGHPUT_FLOOR,
            "throughput_weight": DESIGN_THROUGHPUT_WEIGHT,
            "alpha_scale": cfg.alpha_scale,
            "target_matrices": weights.tolist(),
            "phi_ref_rad": [float(value) for value in cfg.phi_ref],
        },
        "geometry": {
            "design_region_um": [cfg.design_lx, cfg.design_ly, cfg.t_sin],
            "grid_um": cfg.dl_opt,
            "port_pitch_um": cfg.port_pitch,
            "stub_length_um": cfg.stub_len,
            "port_width_um": cfg.wg_width,
            "n_inputs": cfg.N,
            "n_outputs": cfg.N,
        },
        "stack": {
            "t_sin_um": cfg.t_sin,
            "t_sio2_gap_um": cfg.t_gap,
            "t_tfln_um": cfg.t_tfln,
            "n_sio2": cfg.n_sio2,
            "n_sin": cfg.n_sin,
            "n_tfln_o": cfg.n_tfln_o,
            "n_tfln_e": cfg.n_tfln_e,
        },
        "wavelengths_um": list(cfg.wavelengths_um),
        "optimizer": {
            "method": "adjoint",
            "filter": "conic",
            "filter_radius_um": cfg.drc_min_feature,
            "projection": "ramp_tanh",
            "beta_schedule": [
                design_beta(step) for step in range(1, DESIGN_EPOCHS + 1)
            ],
            "penalty": {
                "type": "erosion_dilation",
                "length_scale_um": cfg.drc_freeform_um,
                "weight": DESIGN_PENALTY_WEIGHT,
            },
            "learning_rate": 0.05,
            "adam_betas": [0.9, 0.999],
            "epochs": DESIGN_EPOCHS,
        },
        "post_processing": {
            "threshold": 0.5,
            "min_feature_um": cfg.drc_min_feature,
            "simplify_tolerance_um": 0.005,
            "gds_layers": {
                "sin_core": list(cfg.layer("sin_core")),
                "tfln_slab": list(cfg.layer("tfln_slab")),
            },
        },
    }


def write_design_spec(cfg: AcceleratorConfig, weights: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(design_spec(cfg, weights), indent=2) + "\n", encoding="utf-8")
    return path
