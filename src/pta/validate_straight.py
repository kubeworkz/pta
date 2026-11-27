"""Report forward/backward power, reflection and group index for the straight test.

The straight-waveguide validation spec is written by
:func:`pta.t3d.write_straight_validation_spec`. Download its ``SimulationData`` and run::

    python -m pta.validate_straight tidy3d_validate_straight.hdf5

The forward ``out_0`` amplitude should be close to unity, the backward amplitude and the
reflection monitor should be weak, and the phase slope gives the group index. The absolute
phase is arbitrary; only its slope across the comb is meaningful. The phase length is taken
from the source and output-monitor centres in the loaded simulation, not assumed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import C_0


class StraightValidationError(ValueError):
    """The SimulationData does not contain the expected validation monitors."""


@dataclass
class StraightReport:
    forward_power: np.ndarray
    backward_power: np.ndarray
    reflection_power: np.ndarray
    group_index: float
    distance_um: float


def _mode_amplitudes(
    data: Any, monitor_name: str, direction: str
) -> tuple[np.ndarray, np.ndarray]:
    try:
        monitor_data = data[monitor_name]
    except Exception as exc:
        raise StraightValidationError(
            f"SimulationData has no monitor {monitor_name!r}"
        ) from exc
    amps = monitor_data.amps
    if "direction" not in amps.dims:
        raise StraightValidationError(
            f"monitor {monitor_name!r} has no direction dimension"
        )
    directions = [str(value) for value in amps.coords["direction"].values]
    if direction not in directions:
        raise StraightValidationError(
            f"monitor {monitor_name!r} has directions {directions}, "
            f"expected {direction!r}"
        )
    if "mode_index" not in amps.dims:
        raise StraightValidationError(
            f"monitor {monitor_name!r} has no mode_index dimension"
        )
    if 0 not in [int(value) for value in amps.coords["mode_index"].values]:
        raise StraightValidationError(f"monitor {monitor_name!r} has no mode 0")
    amps = amps.sel(direction=direction, mode_index=0)
    if set(amps.dims) != {"f"}:
        raise StraightValidationError(
            f"monitor {monitor_name!r} amplitudes have dimensions {tuple(amps.dims)}, "
            "expected (f,) after selecting direction and mode 0"
        )
    return (
        np.asarray(amps.values, dtype=complex),
        np.asarray(amps.coords["f"].values, dtype=float),
    )


def _propagation_distance(data: Any, output_monitor: str) -> float:
    simulation = getattr(data, "simulation", None)
    if simulation is None:
        raise StraightValidationError(
            "SimulationData has no simulation; cannot derive the source-to-monitor distance"
        )
    sources = [source for source in simulation.sources if getattr(source, "name", None) == "input"]
    monitors = [
        monitor
        for monitor in simulation.monitors
        if getattr(monitor, "name", None) == output_monitor
    ]
    if len(sources) != 1:
        raise StraightValidationError("SimulationData must contain exactly one 'input' source")
    if len(monitors) != 1:
        raise StraightValidationError(
            f"SimulationData must contain exactly one {output_monitor!r} monitor"
        )
    distance = abs(float(monitors[0].center[0]) - float(sources[0].center[0]))
    if distance <= 0.0:
        raise StraightValidationError("source and output monitor are at the same x position")
    return distance


def evaluate(
    data: Any,
    *,
    output_monitor: str = "out_0",
    reflection_monitor: str = "refl_0",
) -> StraightReport:
    """Extract the validation powers and fit the group index from the phase slope."""
    forward, frequencies = _mode_amplitudes(data, output_monitor, "+")
    backward, backward_freqs = _mode_amplitudes(data, output_monitor, "-")
    reflection, reflection_freqs = _mode_amplitudes(data, reflection_monitor, "-")
    if (
        frequencies.shape != backward_freqs.shape
        or frequencies.shape != reflection_freqs.shape
    ):
        raise StraightValidationError("validation monitors use different frequency grids")
    if not (
        np.allclose(frequencies, backward_freqs, rtol=0.0, atol=1e6)
        and np.allclose(frequencies, reflection_freqs, rtol=0.0, atol=1e6)
    ):
        raise StraightValidationError("validation monitors use different frequency grids")

    distance_um = _propagation_distance(data, output_monitor)
    omega = 2.0 * np.pi * frequencies
    if forward.size >= 2:
        # Centre omega before the fit: comb frequencies are ~1e15 and the raw normal
        # equations lose precision on the slope.
        slope = float(
            np.polyfit(omega - omega.mean(), np.unwrap(np.angle(forward)), 1)[0]
        )
        # Tidy3D uses the e^{-i omega t} convention, so a forward wave carries
        # exp(+i k x) and its phase increases with frequency.  Hence n_g = +c dphi/domega / L.
        group_index = C_0 * slope / (distance_um * 1e-6)
    else:
        group_index = float("nan")
    return StraightReport(
        forward_power=np.abs(forward) ** 2,
        backward_power=np.abs(backward) ** 2,
        reflection_power=np.abs(reflection) ** 2,
        group_index=group_index,
        distance_um=distance_um,
    )


def format_report(report: StraightReport) -> str:
    """Render the validation numbers as one short block."""
    forward = ", ".join(f"{value:.4f}" for value in report.forward_power)
    backward = ", ".join(f"{value:.4g}" for value in report.backward_power)
    reflection = ", ".join(f"{value:.4g}" for value in report.reflection_power)
    return "\n".join(
        [
            "straight-waveguide validation",
            f"  forward |a|^2:     {forward}",
            f"  backward |a|^2:    {backward}",
            f"  reflection |a|^2:  {reflection}",
            f"  fitted group index ({report.distance_um:.1f} um): {report.group_index:.3f}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pta-check-straight",
        description="Check direction, normalization and phase of the straight-waveguide run.",
    )
    parser.add_argument("simulation_data", help="downloaded Tidy3D SimulationData file")
    args = parser.parse_args(argv)

    import tidy3d as td

    try:
        data = td.SimulationData.from_file(
            Path(args.simulation_data), monitor_names=["out_0", "refl_0"]
        )
    except FileNotFoundError:
        print(f"error: file not found: {args.simulation_data}", file=sys.stderr)
        return 1
    except Exception:
        print(
            f"error: cannot read {args.simulation_data}: not a valid Tidy3D result",
            file=sys.stderr,
        )
        return 1

    try:
        report = evaluate(data)
    except StraightValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
