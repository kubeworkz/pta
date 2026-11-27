from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tidy3d as td
import tidy3d.plugins.invdes as tdi
import xarray as xr

from pta import em_import
from pta.cli import main
from pta.config import load_config
from pta.core import CoreModel, build_core, physics_report
from pta.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def em_cfg(tmp_path_factory):
    return load_config(
        ROOT / "configs" / "pta.yaml",
        N=2,
        wavelengths_um=[1.55, 1.56],
        artifacts_dir=tmp_path_factory.mktemp("em_config"),
    )


def _pads_density(cfg) -> np.ndarray:
    """Synthetic optimised mask: material pads at every port boundary."""
    density = np.zeros((cfg.nx, cfg.ny))
    dy = cfg.design_ly / cfg.ny
    dx = cfg.design_lx / cfg.nx
    half = int(round(1.0 / dy))
    depth = int(round(1.6 / dx))
    for index in range(cfg.N):
        row = int(round((cfg.port_y(index) + cfg.design_ly / 2) / dy))
        density[:depth, row - half : row + half] = 1.0
        density[-depth:, row - half : row + half] = 1.0
    return density


def _write_fixture(cfg, root: Path, *, scale: float = 1.0):
    """Synthetic Tidy3D export: a density with port pads and one monitor per input."""
    density = _pads_density(cfg)
    density_path = root / "optimized_density.npy"
    np.save(density_path, density)

    transmission = build_core(cfg).transmission() * scale
    monitors = root / "monitors"
    monitors.mkdir(parents=True, exist_ok=True)
    for index in range(cfg.N):
        np.savez(
            monitors / f"verify_in{index}.npz",
            amplitudes=transmission[:, :, index],
            freqs_hz=cfg.freqs_hz,
        )
    return density_path, monitors


def _mode_data(cfg, monitor, amplitudes):
    freqs = cfg.freqs_hz
    amps = xr.DataArray(
        np.stack([amplitudes, np.zeros_like(amplitudes)], axis=1)[:, :, None],
        coords={"f": freqs, "direction": ["+", "-"], "mode_index": [0]},
        dims=("f", "direction", "mode_index"),
    )
    n_complex = xr.DataArray(
        np.ones((cfg.K, 1), dtype=complex),
        coords={"f": freqs, "mode_index": [0]},
        dims=("f", "mode_index"),
    )
    return td.ModeData(monitor=monitor, amps=amps, n_complex=n_complex)


def _verify_data(cfg, amplitudes):
    """One input's ``SimulationData`` with ``out_0..N-1`` mode monitors."""
    monitors = []
    data = []
    for out_index in range(cfg.N):
        monitor = td.ModeMonitor(
            center=(cfg.design_lx / 2 + 1.0, cfg.port_y(out_index), 0.0),
            size=(0.0, 2.0, cfg.t_sin + 1.5),
            freqs=cfg.freqs_hz,
            mode_spec=td.ModeSpec(num_modes=1),
            name=f"out_{out_index}",
        )
        monitors.append(monitor)
        data.append(_mode_data(cfg, monitor, amplitudes[:, out_index]))
    simulation = td.Simulation(
        size=(cfg.design_lx + 4.0, cfg.design_ly + 4.0, 3.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        boundary_spec=td.BoundarySpec.all_sides(td.PML()),
        monitors=monitors,
        sources=[],
    )
    return td.SimulationData(simulation=simulation, data=tuple(data))


def _design_data(cfg, density, *, anisotropic=False, eps_bounds=None):
    """Inverse-design ``SimulationData`` whose custom medium carries the density."""
    xs = (np.arange(cfg.nx) + 0.5) * (cfg.design_lx / cfg.nx) - cfg.design_lx / 2
    ys = (np.arange(cfg.ny) + 0.5) * (cfg.design_ly / cfg.ny) - cfg.design_ly / 2
    zs = [-0.15, 0.0, 0.15]
    low, high = eps_bounds if eps_bounds is not None else (cfg.n_sio2**2, cfg.n_sin**2)
    eps_xy = low + density * (high - low)
    values = np.repeat(eps_xy[:, :, None], len(zs), axis=2)
    eps = td.SpatialDataArray(values, coords={"x": xs, "y": ys, "z": zs})
    if anisotropic:
        medium = td.CustomAnisotropicMedium(
            **{
                name: td.CustomMedium.from_eps_raw(eps, name=f"core_{name}")
                for name in ("xx", "yy", "zz")
            }
        )
    else:
        medium = td.CustomMedium.from_eps_raw(eps, name="core_medium")
    structure = td.Structure(
        geometry=td.Box(
            center=(0.0, 0.0, 0.0),
            size=(cfg.design_lx, cfg.design_ly, cfg.t_sin),
        ),
        medium=medium,
        name="core",
    )
    simulation = td.Simulation(
        size=(cfg.design_lx + 4.0, cfg.design_ly + 4.0, 3.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        boundary_spec=td.BoundarySpec.all_sides(td.PML()),
        structures=[structure],
        sources=[],
    )
    return td.SimulationData(simulation=simulation, data=())


def _inverse_design_result(cfg, density, bounds):
    """A real InverseDesignResult whose sim_last rebuilds the design structure."""
    region = tdi.TopologyDesignRegion(
        size=(cfg.design_lx, cfg.design_ly, cfg.t_sin),
        center=(0.0, 0.0, 0.0),
        eps_bounds=bounds,
        pixel_size=cfg.dl_opt,
    )
    base = td.Simulation(
        size=(cfg.design_lx + 4.0, cfg.design_ly + 4.0, 3.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.1),
        boundary_spec=td.BoundarySpec.all_sides(td.PML()),
        sources=[],
    )
    design = tdi.InverseDesign(
        design_region=region, task_name="unit-test", simulation=base
    )
    params = np.asarray(density, dtype=float)[:, :, None]
    return tdi.InverseDesignResult(design=design, params=(params,))


def test_extract_density_accepts_inverse_design_result(em_cfg):
    density = _pads_density(em_cfg)
    bounds = (2.0, 5.0)
    result = _inverse_design_result(em_cfg, density, bounds)

    assert isinstance(result, tdi.InverseDesignResult)
    assert isinstance(result.sim_last, td.Simulation)
    assert result.design.design_region.eps_bounds == bounds

    extracted = em_import.extract_density(result, em_cfg)
    np.testing.assert_allclose(extracted, density)


def test_extract_density_accepts_anisotropic_medium(em_cfg):
    density = _pads_density(em_cfg)
    design = _design_data(em_cfg, density, anisotropic=True)

    extracted = em_import.extract_density(design, em_cfg)
    np.testing.assert_allclose(extracted, density)


def test_extract_density_accepts_explicit_array(em_cfg):
    density = _pads_density(em_cfg)

    np.testing.assert_allclose(em_import.extract_density(density, em_cfg), density)
    with pytest.raises(em_import.EmImportError, match="shape"):
        em_import.extract_density(np.zeros((2, 2)), em_cfg)


def test_extract_density_uses_explicit_eps_bounds(em_cfg):
    density = _pads_density(em_cfg)
    bounds = (2.0, 5.0)
    design = _design_data(em_cfg, density, eps_bounds=bounds)

    extracted = em_import.extract_density(design, em_cfg, eps_bounds=bounds)
    np.testing.assert_allclose(extracted, density)


def test_extract_tidy3d_simulation_data(tmp_path, em_cfg):
    transmission = build_core(em_cfg).transmission()
    verify = [
        _verify_data(em_cfg, transmission[:, :, index]) for index in range(em_cfg.N)
    ]
    density = _pads_density(em_cfg)
    design = _design_data(em_cfg, density)

    extracted = em_import.extract_em_results(em_cfg, verify, design, tmp_path / "export")
    assert extracted.density_path.exists()
    imported = em_import.import_core(
        em_cfg, extracted.density_path, extracted.monitors_dir, tmp_path / "core.npz"
    )
    np.testing.assert_allclose(imported.core.transmission(), transmission, atol=1e-9)
    np.testing.assert_allclose(imported.core.density, density, atol=1e-9)

    result = em_import.import_from_simulation_data(
        em_cfg,
        verify,
        design,
        tmp_path / "core2.npz",
        work_dir=tmp_path / "work",
    )
    np.testing.assert_allclose(result.core.transmission(), transmission, atol=1e-9)


def test_import_writes_schema_valid_core(tmp_path, em_cfg):
    density_path, monitors = _write_fixture(em_cfg, tmp_path)

    result = em_import.import_core(em_cfg, density_path, monitors, tmp_path / "core.npz")
    loaded = CoreModel.from_npz(result.path)

    assert loaded.backend == "tidy3d"
    assert loaded.density is not None
    assert loaded.s_matrix.shape == (em_cfg.K, 2 * em_cfg.N, 2 * em_cfg.N)
    np.testing.assert_allclose(loaded.density, np.load(density_path))
    np.testing.assert_allclose(loaded.transmission(), build_core(em_cfg).transmission())

    report = physics_report(loaded)
    assert report["reciprocal"] is True
    assert report["passive"] is True


def test_import_rejects_non_passive_monitors(tmp_path, em_cfg):
    density_path, monitors = _write_fixture(em_cfg, tmp_path, scale=3.0)

    with pytest.raises(em_import.EmImportError, match="not passive"):
        em_import.import_core(em_cfg, density_path, monitors, tmp_path / "core.npz")


def test_import_rejects_wrong_density_shape(tmp_path, em_cfg):
    _, monitors = _write_fixture(em_cfg, tmp_path)
    bad = tmp_path / "bad_density.npy"
    np.save(bad, np.zeros((em_cfg.nx - 1, em_cfg.ny)))

    with pytest.raises(em_import.EmImportError, match="has shape"):
        em_import.import_core(em_cfg, bad, monitors, tmp_path / "core.npz")


def test_import_requires_every_monitor(tmp_path, em_cfg):
    density_path, monitors = _write_fixture(em_cfg, tmp_path)
    (monitors / f"verify_in{em_cfg.N - 1}.npz").unlink()

    with pytest.raises(em_import.EmImportError, match=f"input {em_cfg.N - 1}"):
        em_import.import_core(em_cfg, density_path, monitors, tmp_path / "core.npz")


def test_import_em_cli(tmp_path, em_cfg, capsys):
    density_path, monitors = _write_fixture(em_cfg, tmp_path)
    config_path = em_cfg.save(tmp_path / "config.yaml")

    exit_code = main(
        [
            "import-em",
            "--config",
            str(config_path),
            "--density",
            str(density_path),
            "--monitors",
            str(monitors),
            "--out",
            str(tmp_path / "artifacts"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (tmp_path / "artifacts" / "core_smatrix.npz").exists()
    assert "core:" in captured.out
    assert "backend tidy3d" in captured.out
    assert "reciprocal | passive" in captured.out
    assert captured.err == ""


def test_import_em_cli_from_simulation_data(tmp_path, em_cfg, capsys):
    transmission = build_core(em_cfg).transmission()
    verify = [
        _verify_data(em_cfg, transmission[:, :, index]) for index in range(em_cfg.N)
    ]
    design = _design_data(em_cfg, _pads_density(em_cfg))
    config_path = em_cfg.save(tmp_path / "config.yaml")

    verify_paths = []
    for index, sim_data in enumerate(verify):
        path = tmp_path / f"verify_in{index}.hdf5"
        sim_data.to_file(path)
        verify_paths.append(path)
    design_path = tmp_path / "design.hdf5"
    design.to_file(design_path)

    exit_code = main(
        [
            "import-em",
            "--config",
            str(config_path),
            "--design-data",
            str(design_path),
            "--verify-data",
            *(str(path) for path in verify_paths),
            "--out",
            str(tmp_path / "artifacts"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    core_path = tmp_path / "artifacts" / "core_smatrix.npz"
    assert core_path.exists()
    assert "backend tidy3d" in captured.out
    loaded = CoreModel.from_npz(core_path)
    np.testing.assert_allclose(loaded.transmission(), transmission, atol=1e-9)


def test_import_em_cli_rejects_mixed_modes(tmp_path, capsys):
    exit_code = main(
        [
            "import-em",
            "--density",
            str(tmp_path / "density.npy"),
            "--design-data",
            str(tmp_path / "design.hdf5"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "not both" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_import_em_cli_rejects_wrong_verify_count(tmp_path, capsys):
    exit_code = main(
        [
            "import-em",
            "--design-data",
            str(tmp_path / "design.hdf5"),
            "--verify-data",
            *(str(tmp_path / f"verify_in{index}.hdf5") for index in range(3)),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "expected 4 --verify-data files" in captured.err
    assert "got 3" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_import_em_cli_rejects_unreadable_simulation_data(tmp_path, capsys):
    bad = tmp_path / "bad_design.hdf5"
    bad.write_text("this is not a Tidy3D result", encoding="utf-8")

    exit_code = main(
        [
            "import-em",
            "--design-data",
            str(bad),
            "--verify-data",
            *(str(tmp_path / f"verify_in{index}.hdf5") for index in range(4)),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "cannot read SimulationData" in captured.err
    assert str(bad) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_all_ingests_synthetic_em_core(tmp_path, em_cfg, monkeypatch, capsys):
    density_path, monitors = _write_fixture(em_cfg, tmp_path)
    core = em_import.import_core(em_cfg, density_path, monitors, tmp_path / "core.npz")
    config_path = em_cfg.save(tmp_path / "config.yaml")
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "all",
            "--config",
            str(config_path),
            "--core",
            str(core.path),
            "--out",
            str(tmp_path / "run"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert (tmp_path / "run" / "core_smatrix.npz").exists()
    assert (tmp_path / "dist" / "pta_portal_bundle.zip").exists()
    manifest = json.loads((tmp_path / "run" / "layout_manifest.json").read_text(encoding="utf-8"))
    assert manifest["core_backend"] == "tidy3d"
    drc_core = json.loads((tmp_path / "run" / "drc_core.json").read_text(encoding="utf-8"))
    assert drc_core["stub_landing_limit_um"] == pytest.approx(em_cfg.stub_landing_em_max_um)


def test_pipeline_ingests_synthetic_em_core(tmp_path, em_cfg):
    density_path, monitors = _write_fixture(em_cfg, tmp_path)
    imported = em_import.import_core(em_cfg, density_path, monitors, tmp_path / "core.npz")
    cfg = em_cfg.with_overrides(artifacts_dir=tmp_path / "run")

    result = run_pipeline(cfg, core_path=imported.path, run_drc=True)

    assert result.core.backend == "tidy3d"
    assert result.drc_core["stub_landing_limit_um"] == pytest.approx(cfg.stub_landing_em_max_um)
    landings = result.drc_core["stub_landing_um"]
    assert set(landings) == {
        name for index in range(cfg.N) for name in (f"IN_{index}", f"OUT_{index}")
    }
    assert max(landings.values()) <= cfg.stub_landing_em_max_um
    assert result.drc_core["passes"] is True
    assert result.drc_chip["passes"] is True
    assert result.verification["passes"] is True
