"""Import a Tidy3D-solved core into the PTA core interchange format.

The four verification simulations written by :mod:`pta.t3d` each excite one input port and
record one mode monitor per output port. Extract the complex transmission of each output
mode at every comb line from the solved ``SimulationData`` and save it per input as::

    monitors/verify_in0.npz     # amplitudes[k, o] = IN_0 -> OUT_o at line k
    monitors/verify_in1.npz
    ...

Each monitor archive holds ``amplitudes`` with shape ``(K, N)`` (complex) and may hold
``freqs_hz`` with shape ``(K,)``; when present the frequencies must match the configured
comb. A plain ``.npy`` amplitude array is accepted too. The adjoint mask is a
``(cfg.nx, cfg.ny)`` density map in ``[0, 1]`` saved as ``.npy``, or as ``.npz`` with key
``density``.

The assembled core is reciprocal by construction: the transmission columns from the
forward runs form the lower-left block and its transpose the upper-right block. Reflection
is not imported; the standard physics report is run on the result, and a non-passive
S-matrix (a monitor normalization error) is rejected instead of clipped.

With the solved :class:`tidy3d.SimulationData` objects in hand, skip the manual archive
step: :func:`extract_monitor_amplitudes` reads the output mode monitors,
:func:`extract_density` maps an optimised custom medium (isotropic or anisotropic) back to
a density using the design region's ``eps_bounds`` (or an explicit array), and
:func:`extract_em_results` writes both in the format consumed by :func:`import_core`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import core as core_module
from .config import AcceleratorConfig

#: Complex field tolerance when checking the monitor comb against the configured one.
FREQ_TOLERANCE_HZ = 1e6

MONITOR_TEMPLATE = "verify_in{index}.npz"


class EmImportError(ValueError):
    """The Tidy3D density or monitor export could not be turned into a core."""


@dataclass
class EmImportResult:
    """Assembled core, the file it was written to, and its physics report."""

    core: core_module.CoreModel
    path: Path
    report: dict[str, Any]


def _load(path: str | Path, what: str):
    try:
        return np.load(Path(path), allow_pickle=False)
    except FileNotFoundError as exc:
        raise EmImportError(f"{what} file not found: {path}") from exc
    except Exception as exc:  # malformed npy/npz
        raise EmImportError(f"cannot read {what} {path}: not a valid npy/npz") from exc


def load_simulation_data(
    path: str | Path, *, monitor_names: Sequence[str] | None = None
) -> Any:
    """Load a downloaded Tidy3D ``SimulationData`` file.

    ``monitor_names`` limits the loaded monitor data, which keeps the four verification
    result files from materialising fields the importer does not need.
    """
    import tidy3d as td

    try:
        return td.SimulationData.from_file(Path(path), monitor_names=monitor_names)
    except FileNotFoundError as exc:
        raise EmImportError(f"SimulationData file not found: {path}") from exc
    except Exception as exc:
        raise EmImportError(
            f"cannot read SimulationData {path}: not a valid Tidy3D result"
        ) from exc


def _close(loaded) -> None:
    if isinstance(loaded, np.lib.npyio.NpzFile):
        loaded.close()


def validate_density(
    cfg: AcceleratorConfig, density: np.ndarray, *, source: str = "density"
) -> np.ndarray:
    """Check the density grid and value range; returns a clipped copy."""
    density = np.asarray(density, dtype=float)
    expected = (cfg.nx, cfg.ny)
    if density.shape != expected:
        raise EmImportError(
            f"{source} has shape {density.shape}, expected {expected} "
            f"({cfg.design_lx:g} x {cfg.design_ly:g} um at {cfg.dl_opt:g} um)"
        )
    if not np.all(np.isfinite(density)):
        raise EmImportError(f"{source} contains non-finite values")
    low = float(np.min(density))
    high = float(np.max(density))
    if low < -1e-6 or high > 1.0 + 1e-6:
        raise EmImportError(f"{source} must be in [0, 1], got [{low:.4g}, {high:.4g}]")
    return np.clip(density, 0.0, 1.0)


def load_density(path: str | Path, cfg: AcceleratorConfig) -> np.ndarray:
    """Read a ``.npy`` density map or an ``.npz`` archive with a ``density`` key."""
    loaded = _load(path, "density")
    try:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "density" not in loaded.files:
                raise EmImportError(f"density archive {path} has no 'density' array")
            values = loaded["density"]
        else:
            values = loaded
        try:
            density = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise EmImportError(f"density {path} is not a numeric array") from exc
    finally:
        _close(loaded)
    return validate_density(cfg, density, source=f"density {path}")


def load_amplitudes(path: str | Path, cfg: AcceleratorConfig) -> np.ndarray:
    """Read one input's ``(K, N)`` complex transmission matrix from a monitor export."""
    loaded = _load(path, "monitor")
    freqs: np.ndarray | None = None
    try:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "amplitudes" not in loaded.files:
                raise EmImportError(f"monitor {path} has no 'amplitudes' array")
            values = loaded["amplitudes"]
            if "freqs_hz" in loaded.files:
                freqs = np.asarray(loaded["freqs_hz"], dtype=float)
        else:
            values = loaded
        try:
            amplitudes = np.asarray(values, dtype=complex)
        except (TypeError, ValueError) as exc:
            raise EmImportError(f"monitor {path} is not a complex amplitude array") from exc
    finally:
        _close(loaded)

    expected = (cfg.K, cfg.N)
    if amplitudes.shape != expected:
        raise EmImportError(
            f"monitor {path} has shape {amplitudes.shape}, expected {expected} (K lines x N outputs)"
        )
    if not np.all(np.isfinite(amplitudes)):
        raise EmImportError(f"monitor {path} contains non-finite amplitudes")
    expected_freqs = cfg.freqs_hz
    if freqs is not None and (
        freqs.shape != expected_freqs.shape
        or not np.allclose(freqs, expected_freqs, rtol=0.0, atol=FREQ_TOLERANCE_HZ)
    ):
        raise EmImportError(f"monitor {path} is for a different comb grid")
    return amplitudes


def load_monitors(directory: str | Path, cfg: AcceleratorConfig) -> np.ndarray:
    """Read ``verify_in0..N-1`` and return ``(N, K, N)`` as ``[input, line, output]``."""
    directory = Path(directory)
    if not directory.exists():
        raise EmImportError(f"monitor directory not found: {directory}")
    if not directory.is_dir():
        raise EmImportError(f"monitor path is not a directory: {directory}")
    columns = []
    for index in range(cfg.N):
        candidates = [
            directory / MONITOR_TEMPLATE.format(index=index),
            directory / f"verify_in{index}.npy",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise EmImportError(
                f"monitor for input {index} not found in {directory} "
                f"(expected verify_in{index}.npz)"
            )
        columns.append(load_amplitudes(path, cfg))
    return np.stack(columns)


def assemble_core(
    cfg: AcceleratorConfig,
    density: np.ndarray,
    columns: np.ndarray,
    *,
    backend: str = "tidy3d",
    note: str | None = None,
) -> core_module.CoreModel:
    """Assemble and check a reciprocal, passive core from monitor amplitudes."""
    density = validate_density(cfg, density)
    columns = np.asarray(columns, dtype=complex)
    expected = (cfg.N, cfg.K, cfg.N)
    if columns.shape != expected:
        raise EmImportError(
            f"monitor amplitudes must have shape {expected} (input, line, output), "
            f"got {columns.shape}"
        )
    if not np.all(np.isfinite(columns)):
        raise EmImportError("monitor amplitudes contain non-finite values")

    transmission = np.transpose(columns, (1, 2, 0))  # (K, out, in)
    n = cfg.N
    s_matrix = np.zeros((cfg.K, 2 * n, 2 * n), dtype=complex)
    s_matrix[:, n:, :n] = transmission
    s_matrix[:, :n, n:] = np.swapaxes(transmission, 1, 2)

    from .weights import target_matrices

    assembled = core_module.CoreModel(
        freqs_hz=cfg.freqs_hz,
        s_matrix=s_matrix,
        weights=target_matrices(cfg),
        phi_ref=cfg.phi_ref,
        alpha=cfg.alpha_scale,
        backend=backend,
        density=density,
        note=note
        or (
            "S-matrix assembled from Tidy3D forward runs: output-port mode monitor "
            "amplitudes form the transmission block, which is reflected into a "
            "reciprocal scatterer; reflection is not imported."
        ),
    )
    report = core_module.physics_report(assembled)
    if not report["reciprocal"]:
        raise EmImportError("assembled S-matrix is not reciprocal")
    if not report["passive"]:
        worst = max(line["max_singular_value"] for line in report["lines"])
        raise EmImportError(
            f"monitor amplitudes are not passive: max singular value {worst:.6g} exceeds 1; "
            "check the mode-monitor normalization"
        )
    return assembled


def import_core(
    cfg: AcceleratorConfig,
    density_path: str | Path,
    monitor_dir: str | Path,
    out_path: str | Path | None = None,
    *,
    backend: str = "tidy3d",
    note: str | None = None,
) -> EmImportResult:
    """Build the core from a Tidy3D density map and monitor directory, then save it."""
    density = load_density(density_path, cfg)
    columns = load_monitors(monitor_dir, cfg)
    assembled = assemble_core(cfg, density, columns, backend=backend, note=note)
    path = Path(out_path) if out_path is not None else cfg.path("core_npz")
    assembled.save(path)
    return EmImportResult(
        core=assembled, path=path, report=core_module.physics_report(assembled)
    )


# -- extraction from Tidy3D SimulationData ------------------------------------


def extract_monitor_amplitudes(
    sim_data: Any, cfg: AcceleratorConfig, *, direction: str = "+"
) -> np.ndarray:
    """Return one verify run's ``(K, N)`` output amplitudes from its Tidy3D data.

    ``sim_data`` is the :class:`tidy3d.SimulationData` of one ``tidy3d_verify_in<i>``
    simulation. The ``out_0..out_{N-1}`` mode monitors carry the transmission into each
    output; the forward direction is ``"+"`` for monitors on the output facet.
    """
    amplitudes = np.zeros((cfg.K, cfg.N), dtype=complex)
    for out_index in range(cfg.N):
        name = f"out_{out_index}"
        try:
            monitor_data = sim_data[name]
        except Exception as exc:
            available = [
                monitor.name
                for monitor in getattr(
                    getattr(sim_data, "simulation", None), "monitors", ()
                )
                or ()
            ]
            raise EmImportError(
                f"verify data has no monitor {name!r} (available: {available})"
            ) from exc

        amps = monitor_data.amps
        if "direction" in amps.dims:
            directions = [str(value) for value in amps.coords["direction"].values]
            if direction not in directions:
                raise EmImportError(
                    f"monitor {name!r} has directions {directions}, expected {direction!r}"
                )
            amps = amps.sel(direction=direction)
        if "mode_index" in amps.dims:
            modes = [int(value) for value in amps.coords["mode_index"].values]
            if 0 not in modes:
                raise EmImportError(f"monitor {name!r} has no fundamental mode")
            amps = amps.sel(mode_index=0)
        if set(amps.dims) != {"f"}:
            raise EmImportError(
                f"monitor {name!r} amplitudes have dimensions {tuple(amps.dims)}, "
                "expected (f,) after selecting direction and mode 0"
            )

        freqs = np.asarray(amps.coords["f"].values, dtype=float)
        values = np.asarray(amps.values, dtype=complex)
        if freqs.shape != (cfg.K,):
            raise EmImportError(
                f"monitor {name!r} has {freqs.size} frequencies, expected {cfg.K}"
            )
        for line, target in enumerate(cfg.freqs_hz):
            nearest = int(np.argmin(np.abs(freqs - target)))
            if abs(float(freqs[nearest]) - float(target)) > FREQ_TOLERANCE_HZ:
                raise EmImportError(f"monitor {name!r} is for a different comb grid")
            amplitudes[line, out_index] = values[nearest]
    return amplitudes


def _select_plane(data: Any, *, z: float, frequency: float | None = None) -> np.ndarray:
    """Select one z plane (and frequency, if present) and return real values as (x, y)."""
    if "z" in data.dims:
        data = data.sel(z=z, method="nearest")
    if frequency is not None and "f" in data.dims:
        data = data.sel(f=frequency, method="nearest")
    if set(data.dims) != {"x", "y"}:
        raise EmImportError(
            f"density component has dimensions {tuple(data.dims)}, expected (x, y); "
            "pass a regular x-y-z permittivity grid"
        )
    data = data.transpose("x", "y")
    return np.real(np.asarray(data.values, dtype=complex))


def _resolve_simulation(design_data: Any) -> Any:
    """Find the structures in a SimulationData, InverseDesignResult or Simulation."""
    candidate = getattr(design_data, "sim_last", design_data)
    if isinstance(candidate, dict):
        if len(candidate) != 1:
            raise EmImportError(
                "design result contains several simulations; pass one design or a density array"
            )
        candidate = next(iter(candidate.values()))
    simulation = getattr(candidate, "simulation", candidate)
    if not hasattr(simulation, "structures"):
        raise EmImportError(
            "design data does not expose a Tidy3D Simulation; pass a density array instead"
        )
    return simulation


def _design_eps_bounds(design_data: Any) -> tuple[float, float] | None:
    """Read ``eps_bounds`` from an inverse-design result, if it carries them."""
    for holder in (design_data, getattr(design_data, "design", None)):
        if holder is None:
            continue
        bounds = getattr(holder, "eps_bounds", None)
        if bounds is None:
            region = getattr(holder, "design_region", None)
            bounds = getattr(region, "eps_bounds", None)
        if bounds is not None:
            return tuple(float(value) for value in bounds)
    return None


def _medium_permittivity(
    medium: Any, component: str, *, z: float, frequency: float
) -> np.ndarray:
    """Return one medium component's real permittivity as an (x, y) grid."""
    import tidy3d as td

    if isinstance(medium, td.CustomMedium):
        permittivity = getattr(medium, "permittivity", None)
        if permittivity is not None:
            if not hasattr(permittivity, "sel"):
                raise EmImportError(
                    "the optimised permittivity is not a regular grid; pass the density array"
                )
            return _select_plane(permittivity, z=z)
        dataset = getattr(medium, "eps_dataset", None)
        if dataset is None:
            raise EmImportError("the optimised CustomMedium has no permittivity data")
        return _select_plane(dataset.eps_xx, z=z, frequency=frequency)

    if isinstance(medium, td.CustomAnisotropicMedium):
        part = getattr(medium, component, None)
        if not isinstance(part, td.CustomMedium):
            raise EmImportError(
                f"anisotropic component {component!r} has no spatial permittivity; "
                "pass the density array"
            )
        return _medium_permittivity(part, component, z=z, frequency=frequency)

    raise EmImportError("the optimised structure is not a CustomMedium")


def extract_density(
    design_data: Any,
    cfg: AcceleratorConfig,
    *,
    structure_name: str | None = None,
    eps_bounds: tuple[float, float] | None = None,
    component: str = "xx",
    z: float = 0.0,
) -> np.ndarray:
    """Recover the optimised density from a Tidy3D design result.

    ``design_data`` may be an explicit density array, a ``SimulationData``, a
    ``Simulation`` or an ``InverseDesignResult`` (use ``sim_last``). The design structure
    may carry a ``CustomMedium`` or a ``CustomAnisotropicMedium``; the selected
    ``component`` is mapped back to ``[0, 1]``. Bounds come from ``eps_bounds``, then the
    result's ``design_region.eps_bounds``, then the SiO2/SiN values.
    """
    if isinstance(design_data, np.ndarray):
        return validate_density(cfg, design_data, source="density array")

    import tidy3d as td

    simulation = _resolve_simulation(design_data)
    structures = list(getattr(simulation, "structures", ()) or ())
    candidates = [
        structure
        for structure in structures
        if isinstance(
            getattr(structure, "medium", None),
            (td.CustomMedium, td.CustomAnisotropicMedium),
        )
    ]
    if structure_name is not None:
        candidates = [s for s in candidates if s.name == structure_name]
        if not candidates:
            raise EmImportError(
                "design data has no CustomMedium or CustomAnisotropicMedium structure "
                f"named {structure_name!r}"
            )
    elif len(candidates) > 1:
        names = [structure.name for structure in candidates]
        raise EmImportError(
            f"design data has several custom-medium structures {names}; pass structure_name"
        )
    if not candidates:
        raise EmImportError(
            "design data has no custom-medium structure carrying the optimised density; "
            "pass the density array instead"
        )
    structure = candidates[0]
    values = _medium_permittivity(
        structure.medium,
        component,
        z=z,
        frequency=float(np.mean(cfg.freqs_hz)),
    )

    bounds = eps_bounds
    if bounds is None:
        bounds = _design_eps_bounds(design_data)
    if bounds is None:
        bounds = (cfg.n_sio2**2, cfg.n_sin**2)
    try:
        low, high = (float(value) for value in bounds)
    except (TypeError, ValueError) as exc:
        raise EmImportError("eps_bounds must be a (min, max) pair") from exc
    if high <= low:
        raise EmImportError("eps_bounds max must be greater than min")

    density = (values - low) / (high - low)
    if float(np.min(density)) < -0.05 or float(np.max(density)) > 1.05:
        raise EmImportError(
            "permittivity lies outside the supplied eps_bounds; pass the design region's "
            "eps_bounds or the density array"
        )
    return validate_density(
        cfg,
        np.clip(density, 0.0, 1.0),
        source=f"density from structure {structure.name!r}",
    )


@dataclass
class ExtractedEm:
    """Paths of the density and monitor archives written from SimulationData."""

    density_path: Path
    monitors_dir: Path


def extract_em_results(
    cfg: AcceleratorConfig,
    verify_data: Sequence[Any],
    design_data: Any,
    out_dir: str | Path,
    *,
    structure_name: str | None = None,
    eps_bounds: tuple[float, float] | None = None,
    component: str = "xx",
    direction: str = "+",
) -> ExtractedEm:
    """Write the importer's density and monitor archives from solved Tidy3D data.

    ``verify_data`` holds one :class:`tidy3d.SimulationData` per input port, in input order;
    ``design_data`` is the inverse-design result whose ``CustomMedium`` carries the optimised
    mask. The files are placed under ``out_dir`` in the format :func:`import_core` reads.
    """
    verify_data = list(verify_data)
    if len(verify_data) != cfg.N:
        raise EmImportError(
            f"expected {cfg.N} verify simulations (one per input), got {len(verify_data)}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    density = extract_density(
        design_data,
        cfg,
        structure_name=structure_name,
        eps_bounds=eps_bounds,
        component=component,
    )
    density_path = out / "optimized_density.npz"
    np.savez(density_path, density=density)

    monitors_dir = out / "monitors"
    monitors_dir.mkdir(exist_ok=True)
    for index, sim_data in enumerate(verify_data):
        amplitudes = extract_monitor_amplitudes(sim_data, cfg, direction=direction)
        np.savez(
            monitors_dir / f"verify_in{index}.npz",
            amplitudes=amplitudes,
            freqs_hz=cfg.freqs_hz,
        )
    return ExtractedEm(density_path=density_path, monitors_dir=monitors_dir)


def import_from_simulation_data(
    cfg: AcceleratorConfig,
    verify_data: Sequence[Any],
    design_data: Any,
    out_path: str | Path | None = None,
    *,
    work_dir: str | Path | None = None,
    structure_name: str | None = None,
    eps_bounds: tuple[float, float] | None = None,
    component: str = "xx",
    direction: str = "+",
) -> EmImportResult:
    """Extract solved Tidy3D data and build the core in one call."""
    work = (
        Path(work_dir)
        if work_dir is not None
        else Path(cfg.artifacts_dir) / "tidy3d_results"
    )
    extracted = extract_em_results(
        cfg,
        verify_data,
        design_data,
        work,
        structure_name=structure_name,
        eps_bounds=eps_bounds,
        component=component,
        direction=direction,
    )
    return import_core(cfg, extracted.density_path, extracted.monitors_dir, out_path)
