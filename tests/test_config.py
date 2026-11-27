from __future__ import annotations

import numpy as np
import pytest
import yaml

from pta.config import AcceleratorConfig, load_config


def test_golden_build_dimensions(cfg):
    assert cfg.N == 4
    assert cfg.K == 3
    assert np.allclose(cfg.wavelengths_um, [1.545, 1.550, 1.555])


def test_derived_quantities(cfg):
    assert cfg.v_pi == pytest.approx(5.0)
    assert cfg.freqs_hz.shape == (3,)
    assert cfg.freqs_hz[0] == pytest.approx(299792458.0 / 1.545e-6, rel=1e-9)
    assert cfg.phi_ref.shape == (3,)
    assert cfg.nx == 450 and cfg.ny == 500
    assert cfg.lane_pitch == pytest.approx(3 * 127.0 + 250.0)


def test_port_layout_is_centred(cfg):
    ys = [cfg.port_y(i) for i in range(cfg.N)]
    assert ys == pytest.approx([-6.0, -2.0, 2.0, 6.0])
    assert sum(ys) == pytest.approx(0.0)


def test_layer_lookup(cfg):
    assert cfg.layer("sin_core") == (1, 0)
    assert cfg.layer("exclude") == (99, 0)
    with pytest.raises(KeyError):
        cfg.layer("gold")


def test_override_validation(cfg):
    scaled = cfg.with_overrides(N=8, wavelengths_um=[1.54, 1.55])
    assert scaled.N == 8 and scaled.K == 2
    assert scaled.wg_width == cfg.wg_width

    with pytest.raises((TypeError, ValueError)):
        load_config(None, nonsense=1)
    with pytest.raises(ValueError):
        load_config(None, drc_min_width=1.0)


def test_yaml_round_trip(cfg, tmp_path):
    path = cfg.save(tmp_path / "cfg.yaml")
    loaded = load_config(path)
    assert loaded.to_dict() == cfg.to_dict()
    assert yaml.safe_load(path.read_text())["layers"]["sin_core"] == [1, 0]


def test_defaults_are_independent():
    a, b = AcceleratorConfig(), AcceleratorConfig()
    a.layers["sin_core"] = (5, 0)
    assert b.layers["sin_core"] == (1, 0)
