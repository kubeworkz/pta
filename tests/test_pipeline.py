from __future__ import annotations

import json

import numpy as np
import pytest

from pta import io
from pta.cli import main
from pta.core import build_core
from pta.pipeline import run_pipeline


def test_chip_ports_and_blocks(pipeline_result, small_cfg):
    expected_ports = small_cfg.N * small_cfg.K + 3 * small_cfg.N
    assert len(pipeline_result.layout_result.ports) == expected_ports
    for name in pipeline_result.layout_result.ports:
        assert name.startswith(("COMB_SIG_IN_", "COMB_LO_IN_", "PD_POS_", "PD_NEG_"))


def test_gds_and_json_artifacts_exist(pipeline_result):
    for key in ("core_gds", "chip_gds", "layout", "drc_core", "drc_chip", "verify", "metrics", "core_npz"):
        path = pipeline_result.artifacts[key]
        assert path.exists() and path.stat().st_size > 0


def test_chip_smatrix_is_reciprocal(pipeline_result, small_cfg):
    path = pipeline_result.artifacts["chip_npz"]
    freqs, ports, matrix, meta = io.load_smatrix(path)
    assert matrix.shape == (small_cfg.K, len(ports), len(ports))
    assert meta["N"] == small_cfg.N
    for k in range(matrix.shape[0]):
        diagonal = np.abs(matrix[k] - matrix[k].T).max()
        assert diagonal < 1e-6


def test_signed_mvm_verification(pipeline_result):
    verification = pipeline_result.verification
    assert verification["passes"] is True
    assert verification["worst_relative_error"] < 0.02
    for line in verification["lines"]:
        assert len(line["measured"]) == len(line["ideal"])
        assert np.all(np.isfinite(line["measured"]))


def test_layout_manifest(pipeline_result):
    manifest = json.loads(pipeline_result.artifacts["layout"].read_text())
    assert manifest["chip"] == "PTA_GOLDEN_BUILD"
    assert manifest["bounds_um"]["width"] > 1000.0
    assert len(manifest["connections"]) == pipeline_result.cfg.N * pipeline_result.cfg.K + 2 * pipeline_result.cfg.N
    statuses = {block["status"] for block in manifest["blocks"]}
    assert "em_verified" in statuses


def test_ingested_core_without_density_fails_before_the_layout(tmp_path, cfg):
    """An S-matrix-only core must not be silently given the surrogate geometry."""
    core = build_core(cfg, backend="fdtd")
    core.density = None
    path = core.save(tmp_path / "ingested_without_density.npz")
    with pytest.raises(ValueError, match="no density map"):
        run_pipeline(cfg, core_path=path, run_drc=False)


def test_cli_reports_ingested_core_without_density(tmp_path, cfg, capsys):
    """The CLI must print the density error cleanly instead of a traceback."""
    core = build_core(cfg, backend="fdtd")
    core.density = None
    path = core.save(tmp_path / "ingested_without_density.npz")

    exit_code = main(["design", "--core", str(path), "--out", str(tmp_path / "artifacts")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "no density map" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_missing_core_file(tmp_path, capsys):
    """A --core path that does not exist must fail cleanly, not with a traceback."""
    missing = tmp_path / "does_not_exist.npz"

    exit_code = main(["design", "--core", str(missing), "--out", str(tmp_path / "artifacts")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "core file not found" in captured.err
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_malformed_core_file(tmp_path, capsys):
    """A file that is not an npz must fail cleanly, not with a traceback."""
    malformed = tmp_path / "malformed.npz"
    malformed.write_text("this is not an npz", encoding="utf-8")

    exit_code = main(["design", "--core", str(malformed), "--out", str(tmp_path / "artifacts")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "cannot read core" in captured.err
    assert "not a valid npz" in captured.err
    assert "allow_pickle" not in captured.err
    assert str(malformed) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_unreadable_config(tmp_path, capsys):
    """A directory passed to --config must fail cleanly, not with a traceback."""
    directory = tmp_path / "config_dir"
    directory.mkdir()

    exit_code = main(["design", "--config", str(directory)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "cannot read config" in captured.err
    assert "not a readable file" in captured.err
    assert str(directory) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_invalid_config(tmp_path, capsys):
    """Valid YAML that fails validation must use the invalid-config branch."""
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("not_a_real_key: 1\n", encoding="utf-8")

    exit_code = main(["design", "--config", str(invalid)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "invalid config" in captured.err
    assert "unknown configuration keys" in captured.err
    assert "not_a_real_key" in captured.err
    assert str(invalid) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_set_without_equals(capsys):
    """A --set value without '=' must fail cleanly and name the value."""
    exit_code = main(["design", "--set", "N"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "--set expects key=value" in captured.err
    assert "'N'" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(
    "order",
    [
        ("drc_min_width=1.0", "wg_width=2.0"),
        ("wg_width=2.0", "drc_min_width=1.0"),
    ],
)
def test_cli_accepts_interacting_set_overrides(order, tmp_path, capsys):
    """A pair that is valid together must pass in either argument order."""
    exit_code = main(
        [
            "metrics",
            "--set",
            order[0],
            "--set",
            order[1],
            "--out",
            str(tmp_path / "artifacts"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "written:" in captured.out


def test_cli_reports_invalid_set_value(capsys):
    """A known --set key with an invalid value must name the key and value."""
    exit_code = main(["design", "--set", "N=abc"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "--set N=abc" in captured.err
    assert "N must be an integer" in captured.err
    assert "invalid config" not in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_unknown_set_key(capsys):
    """An unknown --set key must name the key and value, not the config file."""
    exit_code = main(["design", "--set", "no_such_key=1"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "--set no_such_key=1" in captured.err
    assert "unknown configuration key" in captured.err
    assert "'no_such_key'" in captured.err
    assert "invalid config" not in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_malformed_set_override(capsys):
    """A bad --set value must name itself, not be reported as a config YAML error."""
    exit_code = main(["design", "--set", "N=[unclosed"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "--set" in captured.err
    assert "N=[unclosed" in captured.err
    assert "not valid YAML" in captured.err
    assert "cannot read config" not in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_missing_config_file(tmp_path, capsys):
    """A --config path that does not exist must fail cleanly, not with a traceback."""
    missing = tmp_path / "missing.yaml"

    exit_code = main(["design", "--config", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "config file not found" in captured.err
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_malformed_config_file(tmp_path, capsys):
    """Invalid YAML in --config must fail cleanly, not with a traceback."""
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("N: [unclosed", encoding="utf-8")

    exit_code = main(["design", "--config", str(malformed)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "cannot read config" in captured.err
    assert "not valid YAML" in captured.err
    assert str(malformed) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_missing_inputs_file(tmp_path, capsys):
    """A --inputs path that does not exist must fail cleanly, not with a traceback."""
    missing = tmp_path / "no_inputs.npz"

    exit_code = main(["design", "--inputs", str(missing), "--out", str(tmp_path / "artifacts")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "inputs file not found" in captured.err
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_malformed_inputs_file(tmp_path, capsys):
    """A file that is not an npz must fail cleanly, not with a traceback."""
    malformed = tmp_path / "malformed_inputs.npz"
    malformed.write_text("this is not an npz", encoding="utf-8")

    exit_code = main(["design", "--inputs", str(malformed), "--out", str(tmp_path / "artifacts")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "cannot read inputs" in captured.err
    assert "not a valid npz" in captured.err
    assert "allow_pickle" not in captured.err
    assert str(malformed) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_wrong_inputs_shape(tmp_path, capsys):
    """A readable npz with the wrong shape must fail cleanly, not with a traceback."""
    wrong = tmp_path / "wrong_shape.npz"
    np.savez(wrong, inputs=np.zeros((2, 2)))

    exit_code = main(["design", "--inputs", str(wrong), "--out", str(tmp_path / "artifacts")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "inputs must have shape" in captured.err
    assert "got (2, 2)" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_cli_reports_core_npz_missing_required_arrays(tmp_path, capsys):
    """A valid npz without the core arrays is a schema error, not a traceback."""
    incomplete = tmp_path / "incomplete_core.npz"
    np.savez(incomplete, backend=np.asarray("analytic"), note=np.asarray("no arrays"))

    exit_code = main(["design", "--core", str(incomplete), "--out", str(tmp_path / "artifacts")])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err.startswith("pta: error:")
    assert "cannot read core" in captured.err
    assert "not a valid npz" in captured.err
    assert str(incomplete) in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_core_stub_landing_is_reported(pipeline_result, small_cfg):
    landings = pipeline_result.drc_core["stub_landing_um"]
    expected = {f"{side}_{i}" for side in ("IN", "OUT") for i in range(small_cfg.N)}
    assert set(landings) == expected
    for name, depth in landings.items():
        assert 0.0 < depth <= small_cfg.stub_landing_max_um, name
    assert pipeline_result.drc_core["stub_landing_limit_um"] == small_cfg.stub_landing_max_um


def test_tidy3d_handoff_is_portal_ready(pipeline_result):
    import tidy3d as td

    text = pipeline_result.artifacts["t3d_verify_in0"].read_text(encoding="utf-8")
    assert text.isascii()
    sim = td.Simulation.model_validate_json(text)
    assert sim.plot_length_units == "um"
    assert len(sim.structures) == 2


def test_metrics_are_physical(pipeline_result):
    metrics = pipeline_result.metrics
    assert metrics["tops"] > 0
    assert metrics["tops_per_w"] > 0
    assert 1.0 < metrics["enob_bits"] < 16.0
    breakdown = metrics["power_w"]
    parts = breakdown["laser"] + breakdown["rf_drive"] + breakdown["dac_adc"] + breakdown["thermal_tuning"]
    np.testing.assert_allclose(parts, breakdown["total"], rtol=1e-12)
