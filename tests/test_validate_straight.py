from __future__ import annotations

import numpy as np
import pytest
import tidy3d as td
import xarray as xr

from pta.validate_straight import C_0, evaluate, main


def _mode_data(freqs, monitor, forward, backward):
    amps = xr.DataArray(
        np.stack([forward, backward], axis=1)[:, :, None],
        coords={"f": freqs, "direction": ["+", "-"], "mode_index": [0]},
        dims=("f", "direction", "mode_index"),
    )
    n_complex = xr.DataArray(
        np.ones((freqs.size, 1), dtype=complex),
        coords={"f": freqs, "mode_index": [0]},
        dims=("f", "mode_index"),
    )
    return td.ModeData(monitor=monitor, amps=amps, n_complex=n_complex)


def _synthetic_data(cfg, *, n_g=2.0):
    freqs = cfg.freqs_hz
    distance_um = cfg.design_lx + 2.0
    omega = 2.0 * np.pi * freqs
    # Tidy3D's e^{-i omega t} convention gives a forward wave exp(+i k x).
    forward = 0.99 * np.exp(1j * omega * n_g * distance_um * 1e-6 / C_0)
    backward = np.full(freqs.size, 1e-4 + 0.0j)
    reflection = np.full(freqs.size, 2e-4 + 0.0j)

    y0 = cfg.port_y(0)
    x_source = -distance_um / 2
    x_monitor = distance_um / 2
    output = td.ModeMonitor(
        center=(x_monitor, y0, 0.0),
        size=(0.0, 2.0, cfg.t_sin + 1.5),
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=1),
        name="out_0",
    )
    refl = td.ModeMonitor(
        center=(x_source - 0.4, y0, 0.0),
        size=(0.0, 2.0, cfg.t_sin + 1.5),
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=1),
        name="refl_0",
    )
    source = td.ModeSource(
        center=(x_source, y0, 0.0),
        size=(0.0, 2.5, cfg.t_sin + 1.5),
        source_time=td.GaussianPulse(
            freq0=float(np.mean(freqs)), fwidth=0.15 * float(np.mean(freqs))
        ),
        direction="+",
        mode_spec=td.ModeSpec(num_modes=1),
        name="input",
    )
    simulation = td.Simulation(
        center=(0.0, y0, 0.0),
        size=(distance_um + 4.0, 8.0, 2.8),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        boundary_spec=td.BoundarySpec.all_sides(td.PML()),
        monitors=[output, refl],
        sources=[source],
    )
    return td.SimulationData(
        simulation=simulation,
        data=(
            _mode_data(freqs, output, forward, backward),
            _mode_data(freqs, refl, np.zeros_like(forward), reflection),
        ),
    )


def test_evaluate_reports_powers_and_group_index(cfg):
    report = evaluate(_synthetic_data(cfg))

    np.testing.assert_allclose(report.forward_power, 0.99**2, rtol=1e-12)
    np.testing.assert_allclose(report.backward_power, 1e-8, rtol=1e-12)
    np.testing.assert_allclose(report.reflection_power, 4e-8, rtol=1e-12)
    assert report.distance_um == pytest.approx(cfg.design_lx + 2.0)
    assert report.group_index == pytest.approx(2.0, rel=1e-9)


def test_main_prints_report(tmp_path, cfg, capsys):
    path = tmp_path / "tidy3d_validate_straight.hdf5"
    _synthetic_data(cfg).to_file(path)

    exit_code = main([str(path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "straight-waveguide validation" in captured.out
    assert "forward |a|^2" in captured.out
    assert "0.9801" in captured.out
    assert "fitted group index (20.0 um): 2.000" in captured.out
